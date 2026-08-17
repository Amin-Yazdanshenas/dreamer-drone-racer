# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Instrumented evaluation: WHY does lap success plateau at ~30-47%?

Runs the policy under DEPLOYMENT-PARITY conditions (sim.render_interval=100, the obs-camera
staleness the policy trained with — see evaluate_dreamer.py) and logs raw per-event records
that scripts/rl/analyze_lap_diag.py turns into tables/plots:

  crossings.csv  one row per gate pass:  gate idx, gate-frame center error (lateral/vertical),
                 image age + distance-flown-since-refresh (v*dt), visual-overlap flag,
                 action std / pre-tanh entropy, speed
  episodes.csv   one row per episode: length, gates, end type (collision/flyaway/timeout),
                 death diagnostics (target gate, distance/offsets at death, image age, v*dt),
                 spawn-survival flags
  stepstats.csv  action std / entropy aggregated near (<2 m) vs far from the target gate

Diagnostics only — no training-path changes. Both action modes run in one boot
(--modes both -> stochastic then deterministic).

Usage (remote box):
    python3 scripts/rl/diagnose_lap_failures.py \\
        --checkpoint logs/dreamer/r2dreamer/rgb/<RUN>/checkpoints/agent_best.pt \\
        --num_episodes 120 --num_envs 4 --modes both --headless \\
        --out_dir logs/dreamer/r2dreamer/rgb/<RUN>/eval/diag
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Instrumented lap-failure diagnostics.")
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-Dreamer-Play-v0")
parser.add_argument("--obs_mode", type=str, default="rgb", choices=["rgb", "mask", "rgb_mask"])
parser.add_argument("--agent", type=str, default="r2dreamer", choices=["r2dreamer", "ne_dreamer"])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=120, help="Episodes PER MODE.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--modes", type=str, default="both", choices=["stochastic", "deterministic", "both"])
parser.add_argument("--near_gate_m", type=float, default=2.5,
                    help="Death within this distance of the target gate counts as gate-related.")
parser.add_argument("--approach_steps", type=int, default=40,
                    help="Rolling per-step approach buffer length (40 x 0.03 s = 1.2 s) dumped at "
                         "every gate pass / gate-frame clip into approaches.csv.")
parser.add_argument("--out_dir", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tasks  # noqa: F401
from dreamer import DreamerConfig, DreamerIsaacEnvWrapper, DreamerV3Agent, NEDreamerV3Agent
from dreamer.env_wrapper import _extract_gate_mask_u8

_BASE = {"r2dreamer": "dreamer/configs/dreamer_base.yaml", "ne_dreamer": "dreamer/configs/ne_dreamer_base.yaml"}
_MODE = {
    "r2dreamer": {"rgb": "dreamer/configs/dreamer_rgb.yaml", "mask": "dreamer/configs/dreamer_mask.yaml",
                  "rgb_mask": "dreamer/configs/dreamer_rgb_mask.yaml"},
    "ne_dreamer": {"rgb": "dreamer/configs/ne_dreamer_rgb.yaml", "mask": "dreamer/configs/ne_dreamer_mask.yaml",
                   "rgb_mask": "dreamer/configs/ne_dreamer_rgb_mask.yaml"},
}
CTRL_DT = 0.03
_GAUSS_ENT = 1.4189385332046727  # 0.5*(1+log 2pi) per action dim


def _load_cfg():
    import yaml
    base = yaml.safe_load(open(_BASE[args_cli.agent]))
    over = yaml.safe_load(open(_MODE[args_cli.agent][args_cli.obs_mode]))
    cfg = DreamerConfig()
    for k, v in {**base, **over}.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.obs_mode = args_cli.obs_mode
    cfg.__post_init__()
    return cfg


def _quat_conj_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vector v into the frame of quaternion q (wxyz), batched."""
    w, x, y, z = q.unbind(-1)
    # inverse rotation = conjugate for unit quats
    qv = torch.stack([-x, -y, -z], dim=-1)
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(qv, t, dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v from local frame to world by quaternion q (wxyz), batched."""
    w = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between batched vectors."""
    an = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    bn = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.rad2deg(torch.acos((an * bn).sum(-1).clamp(-1.0, 1.0)))


def _term_fired(isaac, name: str) -> torch.Tensor:
    """Per-env bool: did termination term `name` fire this step. Defensive across API versions."""
    tm = isaac.termination_manager
    try:
        return tm.get_term(name).cpu()
    except Exception:
        pass
    try:
        return tm._term_dones[name].cpu()
    except Exception:
        return torch.zeros(isaac.num_envs, dtype=torch.bool)


def _overlap_frac(seg: torch.Tensor, isaac, command_name: str) -> np.ndarray:
    """Per-env fraction of OTHER-gate pixels inside the (dilated) bbox of the target-gate mask.
    Visual-confusion measure: >0 means another gate overlaps the target in view."""
    import cv2
    target = _extract_gate_mask_u8(seg, isaac, command_name)[..., 0].numpy()  # (N, H, W) u8
    seg_np = seg[..., 0].detach().cpu().numpy()  # (N, H, W) raw class ids
    out = np.zeros(target.shape[0], dtype=np.float32)
    for i in range(target.shape[0]):
        t = target[i] > 0
        n_t = int(t.sum())
        if n_t == 0:
            continue
        dil = cv2.dilate(t.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        # any-gate pixels = nonzero seg classes (gates are the only labeled semantics)
        any_gate = seg_np[i] > 0
        other = any_gate & ~t
        out[i] = float((other & dil).sum()) / n_t
    return out


def run_mode(env, agent, isaac, cam, cmd, mode: str, writers, n_episodes: int):
    from collections import deque
    N = env.num_envs
    deterministic = mode == "deterministic"
    agent.reset_carry(N)
    obs = env.reset()
    approach_buf = [deque(maxlen=args_cli.approach_steps) for _ in range(N)]
    event_ctr = [0] * N

    # per-env episode state
    ep_len = np.zeros(N, dtype=int)
    ep_gates = np.zeros(N, dtype=int)
    spawn_gate = cmd.next_gate_idx.cpu().numpy().copy()
    overlap_hist = [[] for _ in range(N)]          # rolling last-10 overlap flags
    # snapshots of the LAST ALIVE state (Isaac auto-resets in-step; post-step reads on the
    # death step describe the NEXT episode, so death diagnostics use the previous step)
    snap_pos = torch.zeros(N, 3)
    snap_gate_idx = np.zeros(N, dtype=int)
    snap_dist = np.zeros(N)
    snap_lat = np.zeros(N)
    snap_vert = np.zeros(N)
    snap_speed = np.zeros(N)

    # camera-staleness tracking (render tick is global)
    img_age = 0
    vdt = np.zeros(N)                               # meters flown since last camera refresh
    cam_sig = None

    # near/far action-stat accumulators
    acc = {b: [0, 0.0, 0.0] for b in ("near", "far")}  # count, std_sum, ent_sum

    episodes_done = 0
    crossings_w, episodes_w, _, approaches_w = writers

    def dump_approach(i: int, gate: int, outcome: str):
        """Write the rolling buffer rows belonging to THIS gate approach, t=0 at the event."""
        rows = [r for r in approach_buf[i] if r[0] == gate]
        if not rows:
            return
        event_ctr[i] += 1
        eid = f"{mode[:4]}_{i}_{event_ctr[i]}"
        n = len(rows)
        for k, r in enumerate(rows):
            approaches_w.writerow([mode, eid, i, gate, outcome, k - (n - 1),
                                   f"{(k - (n - 1)) * CTRL_DT:.3f}"]
                                  + [f"{v:.4f}" if isinstance(v, float) else v for v in r[1:]])

    while episodes_done < n_episodes and simulation_app.is_running():
        with torch.no_grad():
            action = agent.act(obs, is_first=obs["is_first"], deterministic=deterministic)
            # action-dist stats from the carry (post-act latent), no training-path changes
            stoch, deter, _ = agent._carry
            latent = agent.rssm.get_feat(stoch, deter)
            mean, log_std = agent.actor.dist_params(latent.to(agent.device))
            std_now = log_std.exp().mean(-1).float().cpu().numpy()
            ent_now = (log_std.sum(-1) + _GAUSS_ENT * mean.shape[-1]).float().cpu().numpy()
            act_mean_np = torch.tanh(mean).float().cpu().numpy()      # (N, 4) deterministic action
            act_exec_np = action.float().cpu().numpy()                # (N, 4) executed action
            std_dims_np = log_std.exp().float().cpu().numpy()         # (N, 4)

        # ---- pre-step snapshot of alive state ----
        robot = isaac.scene["robot"]
        pos_w = robot.data.root_pos_w.detach().cpu()
        vel_w = robot.data.root_lin_vel_w.detach().cpu()
        gate_pos = cmd.track.data.object_com_pos_w[cmd.env_ids, cmd.next_gate_idx].detach().cpu()
        gate_quat = cmd.next_gate_w[:, 3:7].detach().cpu()
        rel = _quat_conj_rotate(gate_quat, pos_w - gate_pos)   # gate frame: x=through, y=lat, z=vert
        snap_pos = pos_w
        snap_gate_idx = cmd.next_gate_idx.cpu().numpy().copy()
        snap_dist = torch.linalg.norm(pos_w - gate_pos, dim=-1).numpy()
        snap_lat = rel[:, 1].numpy()
        snap_vert = rel[:, 2].numpy()
        snap_speed = torch.linalg.norm(vel_w, dim=-1).numpy()

        # near/far action-stat buckets
        for i in range(N):
            b = "near" if snap_dist[i] < 2.0 else "far"
            acc[b][0] += 1
            acc[b][1] += float(std_now[i])
            acc[b][2] += float(ent_now[i])

        # visual overlap (privileged seg, diagnostics only)
        seg = cam.data.output.get("semantic_segmentation")
        ov = _overlap_frac(seg, isaac, env.command_name) if seg is not None else np.zeros(N)
        for i in range(N):
            overlap_hist[i].append(bool(ov[i] > 0.3))
            if len(overlap_hist[i]) > 10:
                overlap_hist[i].pop(0)

        # ---- approach geometry (for the rolling last-~1s buffer) ----
        quat_w = robot.data.root_quat_w.detach().cpu()
        fwd_local = torch.zeros(N, 3); fwd_local[:, 0] = 1.0
        drone_fwd = _quat_rotate(quat_w, fwd_local)                 # body +x in world
        gate_norm = _quat_rotate(gate_quat, fwd_local)              # gate through-axis in world
        to_gate = gate_pos - pos_w
        head_err = _angle_deg(drone_fwd, gate_norm).numpy()         # alignment w/ through-axis
        bear_err = _angle_deg(drone_fwd, to_gate).numpy()           # is nose pointing at gate
        vel_err = _angle_deg(vel_w, gate_norm).numpy()              # velocity vs through-axis
        rel_v = _quat_conj_rotate(gate_quat, vel_w)                 # velocity in gate frame
        for i in range(N):
            approach_buf[i].append((
                int(snap_gate_idx[i]), float(snap_dist[i]), float(snap_lat[i]), float(snap_vert[i]),
                float(rel[i, 0]), float(snap_speed[i]), float(rel_v[i, 0]), float(rel_v[i, 1]),
                float(rel_v[i, 2]), float(head_err[i]), float(bear_err[i]), float(vel_err[i]),
                float(act_mean_np[i, 0]), float(act_mean_np[i, 1]), float(act_mean_np[i, 2]),
                float(act_mean_np[i, 3]), float(act_exec_np[i, 0]), float(act_exec_np[i, 1]),
                float(act_exec_np[i, 2]), float(act_exec_np[i, 3]),
                float(np.abs(act_exec_np[i] - act_mean_np[i]).mean()),   # realized noise magnitude
                float(std_now[i]), float(std_dims_np[i].max()), float(ent_now[i]),
                int(img_age), float(vdt[i]), float(ov[i]),
            ))

        obs = env.step(action.cpu())
        ep_len += 1
        # camera refresh detection (cheap signature on one pixel row of env 0)
        rgb_raw = cam.data.output.get("rgb")
        sig = float(rgb_raw[0, 32].float().sum().item()) if rgb_raw is not None else 0.0
        if cam_sig is None or sig != cam_sig:
            cam_sig = sig
            img_age = 0
            vdt[:] = 0.0
        else:
            img_age += 1
        vdt += snap_speed * CTRL_DT

        passed = obs["gate_passed"].numpy().astype(bool)
        is_last = obs["is_last"].numpy().astype(bool)
        is_term = obs["is_terminal"].numpy().astype(bool)
        coll = _term_fired(isaac, "collision").numpy().astype(bool)
        flya = _term_fired(isaac, "flyaway").numpy().astype(bool)

        for i in range(N):
            if passed[i]:
                ep_gates[i] += 1
                dump_approach(i, int(snap_gate_idx[i]), "pass")
                crossings_w.writerow([mode, episodes_done, i, int(snap_gate_idx[i]),
                                      f"{snap_lat[i]:.4f}", f"{snap_vert[i]:.4f}",
                                      f"{np.hypot(snap_lat[i], snap_vert[i]):.4f}",
                                      img_age, f"{vdt[i]:.3f}",
                                      int(any(overlap_hist[i])),
                                      f"{std_now[i]:.4f}", f"{ent_now[i]:.4f}",
                                      f"{snap_speed[i]:.3f}"])
            if is_last[i]:
                if is_term[i] and coll[i]:
                    end = "collision"
                elif is_term[i] and flya[i]:
                    end = "flyaway"
                elif is_term[i]:
                    end = "termination_other"
                else:
                    end = "timeout"
                near = bool(snap_dist[i] < args_cli.near_gate_m)
                ground = bool(snap_pos[i, 2] < 0.15)
                if end == "collision" and near:
                    dump_approach(i, int(snap_gate_idx[i]), "clip")
                episodes_w.writerow([mode, episodes_done, i, int(ep_len[i]), int(ep_gates[i]),
                                     int(spawn_gate[i]), end, int(snap_gate_idx[i]),
                                     f"{snap_dist[i]:.3f}", f"{snap_lat[i]:.4f}", f"{snap_vert[i]:.4f}",
                                     int(near), int(ground), img_age, f"{vdt[i]:.3f}",
                                     int(any(overlap_hist[i])),
                                     int(ep_len[i] >= 15), int(ep_gates[i] >= 1)])
                episodes_done += 1
                ep_len[i] = 0
                ep_gates[i] = 0
                overlap_hist[i].clear()
                approach_buf[i].clear()
                spawn_gate[i] = int(cmd.next_gate_idx[i].item())
                if episodes_done % 25 == 0:
                    print(f"[diag] {mode}: {episodes_done}/{n_episodes} episodes", flush=True)
    return acc


def main():
    cfg = _load_cfg()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.sim.render_interval = 100      # deployment parity (see evaluate_dreamer.py)
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = DreamerIsaacEnvWrapper(gym_env, obs_mode=args_cli.obs_mode)
    device = args_cli.device or "cuda"

    agent = (NEDreamerV3Agent if args_cli.agent == "ne_dreamer" else DreamerV3Agent)(cfg, device=device)
    agent.load(args_cli.checkpoint)
    agent._step = max(agent._step, cfg.warmup_steps + 1)
    agent.eval_mode()

    isaac = env._isaac
    cam = isaac.scene["tiled_camera"]
    cmd = isaac.command_manager.get_term(env.command_name)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    fc = open(os.path.join(args_cli.out_dir, "crossings.csv"), "w", newline="")
    fe = open(os.path.join(args_cli.out_dir, "episodes.csv"), "w", newline="")
    fs = open(os.path.join(args_cli.out_dir, "stepstats.csv"), "w", newline="")
    fa = open(os.path.join(args_cli.out_dir, "approaches.csv"), "w", newline="")
    crossings_w = csv.writer(fc)
    episodes_w = csv.writer(fe)
    stepstats_w = csv.writer(fs)
    approaches_w = csv.writer(fa)
    approaches_w.writerow(["mode", "event_id", "env", "gate_idx", "outcome", "t_steps", "t_sec",
                           "dist_m", "lat_m", "vert_m", "along_m", "speed_mps",
                           "v_along", "v_lat", "v_vert", "head_err_deg", "bearing_err_deg",
                           "vel_align_deg", "am0", "am1", "am2", "am3",
                           "ax0", "ax1", "ax2", "ax3", "noise_mag",
                           "act_std", "act_std_max", "pretanh_ent", "img_age_steps", "vdt_m",
                           "overlap_frac"])
    crossings_w.writerow(["mode", "ep_ctr", "env", "gate_idx", "lat_m", "vert_m", "radial_m",
                          "img_age_steps", "vdt_m", "overlap", "act_std", "pretanh_ent", "speed_mps"])
    episodes_w.writerow(["mode", "ep_ctr", "env", "len_steps", "gates", "spawn_gate", "end_type",
                         "death_gate", "death_dist_m", "death_lat_m", "death_vert_m",
                         "death_near_gate", "death_on_ground", "death_img_age", "death_vdt_m",
                         "death_overlap", "survived_startup15", "survived_first_gate"])
    stepstats_w.writerow(["mode", "bucket", "steps", "std_sum", "ent_sum"])

    modes = ["stochastic", "deterministic"] if args_cli.modes == "both" else [args_cli.modes]
    for mode in modes:
        print(f"[diag] === mode: {mode} ({args_cli.num_episodes} episodes) ===", flush=True)
        acc = run_mode(env, agent, isaac, cam, cmd, mode,
                       (crossings_w, episodes_w, stepstats_w, approaches_w), args_cli.num_episodes)
        for b, (n, ssum, esum) in acc.items():
            stepstats_w.writerow([mode, b, n, f"{ssum:.4f}", f"{esum:.4f}"])
        fc.flush(); fe.flush(); fs.flush(); fa.flush()

    fc.close(); fe.close(); fs.close(); fa.close()
    print(f"[diag] DONE — wrote crossings/episodes/stepstats/approaches.csv to {args_cli.out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
