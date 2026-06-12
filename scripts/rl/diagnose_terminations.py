# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a trained policy and log per-episode termination causes.

Answers: WHAT kills the drone — collision (and where/when), flyaway (and right after
which gate pass?), or timeout. Prints a per-episode table and a summary breakdown.

Usage:
    python3 scripts/rl/diagnose_terminations.py \\
        --checkpoint <ckpt.pt> --episodes 60 --headless --enable_cameras
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-Dreamer-Play-v0")
parser.add_argument("--obs_mode", type=str, default="rgb", choices=["rgb", "mask", "rgb_mask"])
parser.add_argument("--agent", type=str, default="r2dreamer", choices=["r2dreamer", "ne_dreamer"])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=60)
parser.add_argument("--stochastic", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import collections

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tasks  # noqa: F401
from dreamer import DreamerConfig, DreamerIsaacEnvWrapper, DreamerV3Agent, NEDreamerV3Agent

_BASE = {"r2dreamer": "dreamer/configs/dreamer_base.yaml", "ne_dreamer": "dreamer/configs/ne_dreamer_base.yaml"}
_MODE = {
    "r2dreamer": {"rgb": "dreamer/configs/dreamer_rgb.yaml", "mask": "dreamer/configs/dreamer_mask.yaml",
                  "rgb_mask": "dreamer/configs/dreamer_rgb_mask.yaml"},
    "ne_dreamer": {"rgb": "dreamer/configs/ne_dreamer_rgb.yaml", "mask": "dreamer/configs/ne_dreamer_mask.yaml",
                   "rgb_mask": "dreamer/configs/ne_dreamer_rgb_mask.yaml"},
}


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


def main():
    cfg = _load_cfg()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = DreamerIsaacEnvWrapper(gym_env, obs_mode=args_cli.obs_mode)
    device = args_cli.device or "cuda"

    agent = NEDreamerV3Agent(cfg, device=device) if args_cli.agent == "ne_dreamer" else DreamerV3Agent(cfg, device=device)
    agent.load(args_cli.checkpoint)
    agent._step = max(agent._step, cfg.warmup_steps + 1)
    agent.eval_mode()
    agent.reset_carry(env.num_envs)

    isaac = env._isaac
    tm = isaac.termination_manager
    cmd = isaac.command_manager.get_term(env.command_name)
    robot = isaac.scene["robot"]

    rows = []
    ep_len, ep_gates = 0, 0
    last_pass_step = -999
    last_passed_gate = -1
    obs = env.reset()
    while len(rows) < args_cli.episodes and simulation_app.is_running():
        with torch.no_grad():
            action = agent.act(obs, is_first=obs["is_first"], deterministic=not args_cli.stochastic)
        obs = env.step(action.cpu())
        ep_len += 1
        if bool(obs["gate_passed"][0]):
            ep_gates += 1
            last_pass_step = ep_len
            # next_gate_idx already advanced past the passed gate by now is fine for attribution
            last_passed_gate = int(cmd.next_gate_idx[0].item()) - 1

        if bool(obs["is_last"][0]):
            # which termination term fired (read per-term dones from the manager)
            cause = "timeout"
            for name in tm.active_terms:
                try:
                    fired = bool(tm.get_term(name)[0].item())
                except Exception:
                    try:
                        fired = bool(tm._term_dones[name][0].item())
                    except Exception:
                        fired = False
                if fired and name != "time_out":
                    cause = name
            pos = robot.data.root_pos_w[0]
            tgt = cmd.command[0, :3]
            dist = float(torch.norm(pos - tgt).item())
            rows.append({
                "len": ep_len, "gates": ep_gates, "cause": cause,
                "dist_to_target": round(dist, 2), "z": round(float(pos[2].item()), 2),
                "steps_since_pass": (ep_len - last_pass_step) if ep_gates > 0 else -1,
                "last_passed_gate": last_passed_gate if ep_gates > 0 else -1,
            })
            print(f"ep {len(rows):3d} len={ep_len:4d} gates={ep_gates} cause={cause:10s} "
                  f"dist={dist:6.2f} z={float(pos[2].item()):5.2f} "
                  f"since_pass={(ep_len - last_pass_step) if ep_gates>0 else -1} "
                  f"last_gate={last_passed_gate if ep_gates>0 else -1}", flush=True)
            ep_len, ep_gates = 0, 0
            last_pass_step = -999
            last_passed_gate = -1
            obs = env.reset()
            agent.reset_carry(env.num_envs)

    print("\n=== SUMMARY ===", flush=True)
    causes = collections.Counter(r["cause"] for r in rows)
    print("causes:", dict(causes), flush=True)
    one_step = [r for r in rows if r["len"] <= 3]
    print(f"instant deaths (len<=3): {len(one_step)}  causes: {dict(collections.Counter(r['cause'] for r in one_step))}", flush=True)
    post_pass = [r for r in rows if 0 <= r["steps_since_pass"] <= 3]
    print(f"died within 3 steps of a pass: {len(post_pass)}  causes: {dict(collections.Counter(r['cause'] for r in post_pass))} "
          f"after gates: {dict(collections.Counter(r['last_passed_gate'] for r in post_pass))}", flush=True)
    fly = [r for r in rows if r["cause"] == "flyaway"]
    print(f"flyaway deaths: {len(fly)}  dist at death: {[r['dist_to_target'] for r in fly][:12]}", flush=True)
    ground = [r for r in rows if r["cause"] == "collision" and r["z"] < 0.3]
    print(f"collisions at z<0.3 (ground): {len(ground)} of {causes.get('collision',0)} collisions", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
