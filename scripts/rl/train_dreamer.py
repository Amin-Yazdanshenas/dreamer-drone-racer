# Copyright (c) 2025, Kousheek Chakraborty / Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Train a DreamerV3 agent on the Isaac Lab drone racing environment.

Usage examples:
    # RGB observations (closest to Dream to Fly paper)
    python3 scripts/rl/train_dreamer.py \\
        --task Isaac-Drone-Racer-Dreamer-RGB-v0 \\
        --obs_mode rgb --num_envs 32 --max_steps 2000000 \\
        --headless --enable_cameras

    # Binary segmentation mask only
    python3 scripts/rl/train_dreamer.py \\
        --task Isaac-Drone-Racer-Dreamer-Mask-v0 \\
        --obs_mode mask --num_envs 32 --max_steps 2000000 \\
        --headless --enable_cameras

    # RGB + mask (4-channel)
    python3 scripts/rl/train_dreamer.py \\
        --task Isaac-Drone-Racer-Dreamer-RGBMask-v0 \\
        --obs_mode rgb_mask --num_envs 32 --max_steps 2000000 \\
        --headless --enable_cameras

    # Resume from checkpoint
    python3 scripts/rl/train_dreamer.py \\
        --task Isaac-Drone-Racer-Dreamer-RGB-v0 \\
        --obs_mode rgb \\
        --checkpoint logs/dreamer/rgb/<run>/checkpoints/agent_latest.pt \\
        --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a DreamerV3 agent.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID.")
parser.add_argument(
    "--obs_mode", type=str, default="rgb", choices=["rgb", "mask", "rgb_mask"],
    help="Observation mode for DreamerV3 image encoder."
)
parser.add_argument(
    "--agent", type=str, default="r2dreamer", choices=["r2dreamer", "ne_dreamer"],
    help="Agent variant: r2dreamer (Barlow Twins) or ne_dreamer (causal transformer)."
)
parser.add_argument("--num_envs", type=int, default=None, help="Override number of envs.")
parser.add_argument("--max_steps", type=int, default=2_000_000, help="Total env steps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from .pt file.")
parser.add_argument("--record_fpv", action="store_true", default=False,
                    help="Record FPV camera for env 0; only keeps videos that pass ≥1 gate.")
parser.add_argument("--render_interval", type=int, default=None,
                    help="Physics steps between viewport renders. Default: 16 non-headless "
                         "(smooth ~25 Hz viz without blocking sim), 100 headless (unused). "
                         "Lower = smoother viewport but more sim stalls when training kicks in.")
parser.add_argument("--gatenet_ckpt", type=str, default=None,
                    help="Path to a trained GateNet checkpoint. If set, the mask channel is "
                         "predicted from RGB by GateNet instead of read from Isaac Sim's "
                         "semantic_segmentation. Tests sim-to-real perception. Use with "
                         "--obs_mode mask or rgb_mask.")
parser.add_argument("--config", type=str, default=None,
                    help="Path to dreamer YAML config (default: auto from obs_mode).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no_curriculum", action="store_true", default=False,
                    help="Disable the spawn_lerp_alpha annealing curriculum (keep the cfg value fixed).")
parser.add_argument("--no_imag_curriculum", action="store_true", default=False,
                    help="Disable the performance-gated imag_horizon auto-curriculum (keep H fixed at the cfg value).")
# Finetune / resume helpers (start a converged policy at the end of the curricula instead of
# restarting them from scratch — otherwise a naive resume drops imag_horizon back to stage 0).
parser.add_argument("--imag_start_stage", type=int, default=0,
                    help="Initial imag_horizon curriculum stage. Start at IMAG_STAGES[N] instead "
                         "of 0 on a resumed run (e.g. 3 = resume at H=48, then climb to 56/64).")
parser.add_argument("--spawn_done", action="store_true", default=False,
                    help="Start the spawn curriculum at its FINAL (fully-annealed) stage — for "
                         "finetuning a converged policy so it does not re-learn near-gate spawns.")
parser.add_argument("--entropy_min_final", type=float, default=None,
                    help="Anneal the entropy floor to this value (late sharpening). Use with "
                         "--entropy_anneal_start/--entropy_anneal_end.")
parser.add_argument("--entropy_anneal_start", type=int, default=None,
                    help="Env-step to begin annealing the entropy floor.")
parser.add_argument("--entropy_anneal_end", type=int, default=None,
                    help="Env-step to finish annealing the entropy floor.")
parser.add_argument("--lr_final", type=float, default=None,
                    help="Late LR-anneal target. Decays lr->lr_final to stabilise the critic at large "
                         "returns (fixes the spike-then-collapse). Use with --lr_anneal_start/_end.")
parser.add_argument("--lr_anneal_start", type=int, default=None,
                    help="Env-step to begin the LR anneal.")
parser.add_argument("--lr_anneal_end", type=int, default=None,
                    help="Env-step to finish the LR anneal.")
parser.add_argument("--lr_actor", type=float, default=None,
                    help="Split-LR: actor optimizer base LR (DreamerV3/SkyDreamer use 3e-5 while wm stays "
                         "1e-4). Damps the actor-sharpening collapse. Default falls back to --lr/cfg.lr.")
parser.add_argument("--lr_critic", type=float, default=None,
                    help="Split-LR: critic optimizer base LR (3e-5). Damps the value/return runaway (the "
                         "terminal collapse, see HANDOFF §2a). Default falls back to --lr/cfg.lr.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # always required for RGB/seg

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Belt-and-suspenders DLSS / raster-only
try:
    import carb
    _s = carb.settings.get_settings()
    _s.set("/rtx/post/dlss/execMode", 0)
    _s.set("/rtx/rendermode", "RasterOnly")
except Exception:
    pass

"""Rest follows after Isaac Sim init."""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import random
from collections import deque
from datetime import datetime

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tasks  # noqa: F401
from dreamer import DreamerConfig, DreamerIsaacEnvWrapper, DreamerV3Agent, NEDreamerV3Agent
from dreamer.replay_buffer import SequenceReplayBuffer

_OBS_MODE_TO_CONFIG = {
    "r2dreamer": {
        "rgb": "dreamer/configs/dreamer_rgb.yaml",
        "mask": "dreamer/configs/dreamer_mask.yaml",
        "rgb_mask": "dreamer/configs/dreamer_rgb_mask.yaml",
    },
    "ne_dreamer": {
        "rgb": "dreamer/configs/ne_dreamer_rgb.yaml",
        "mask": "dreamer/configs/ne_dreamer_mask.yaml",
        "rgb_mask": "dreamer/configs/ne_dreamer_rgb_mask.yaml",
    },
}

_AGENT_TO_BASE_CONFIG = {
    "r2dreamer": "dreamer/configs/dreamer_base.yaml",
    "ne_dreamer": "dreamer/configs/ne_dreamer_base.yaml",
}


def _load_config(args) -> DreamerConfig:
    """Load base config, overlay obs-mode config, then apply CLI overrides."""
    import yaml

    base_path = _AGENT_TO_BASE_CONFIG[args.agent]
    mode_path = args.config or _OBS_MODE_TO_CONFIG[args.agent][args.obs_mode]

    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(mode_path) as f:
        override = yaml.safe_load(f)

    merged = {**base, **override}
    cfg = DreamerConfig()
    for k, v in merged.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # CLI overrides (entropy-floor anneal for finetuning a converged policy).
    if args.entropy_min_final is not None:
        cfg.entropy_min_final = args.entropy_min_final
    if args.entropy_anneal_start is not None:
        cfg.entropy_anneal_start = args.entropy_anneal_start
    if args.entropy_anneal_end is not None:
        cfg.entropy_anneal_end = args.entropy_anneal_end
    if args.lr_final is not None:
        cfg.lr_final = args.lr_final
    if args.lr_anneal_start is not None:
        cfg.lr_anneal_start = args.lr_anneal_start
    if args.lr_anneal_end is not None:
        cfg.lr_anneal_end = args.lr_anneal_end
    if args.lr_actor is not None:
        cfg.lr_actor = args.lr_actor
    if args.lr_critic is not None:
        cfg.lr_critic = args.lr_critic

    cfg.obs_mode = args.obs_mode
    cfg.__post_init__()   # recompute image_channels from obs_mode
    return cfg


def main():
    torch.manual_seed(args_cli.seed)
    random.seed(args_cli.seed)

    cfg = _load_config(args_cli)

    # ----------------------------------------------------------------
    # Environment
    # ----------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    # Decouple viewport render rate from sim/decimation. Default render_interval = decimation (4)
    # means render every RL step → viewport blocks training when GPU is busy. Bump it to
    # decimation*4 = 16 physics ticks (~25 Hz viewport at sim_dt=1/400) for smooth non-headless
    # viz that doesn't stall sim; large value when headless since viewport is hidden anyway.
    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = args_cli.render_interval
    elif getattr(args_cli, "headless", False):
        env_cfg.sim.render_interval = 100
    else:
        env_cfg.sim.render_interval = env_cfg.decimation * 4
    print(f"[Sim] render_interval={env_cfg.sim.render_interval} (sim.dt={env_cfg.sim.dt})")

    if args_cli.record_fpv:
        env_cfg.commands.target.record_fpv = True
        print("[FPV] Recording enabled — only videos with ≥1 gate pass will be kept.")

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = DreamerIsaacEnvWrapper(
        gym_env,
        obs_mode=args_cli.obs_mode,
        gatenet_ckpt=args_cli.gatenet_ckpt,
    )

    device = args_cli.device or "cuda"

    # ----------------------------------------------------------------
    # Agent & replay buffer
    # ----------------------------------------------------------------
    if args_cli.agent == "ne_dreamer":
        agent = NEDreamerV3Agent(cfg, device=device)
    else:
        agent = DreamerV3Agent(cfg, device=device)
    replay = SequenceReplayBuffer(
        capacity=cfg.replay_capacity,
        seq_len=cfg.seq_len,
        num_envs=env.num_envs,
        device=device,
    )

    if args_cli.checkpoint:
        agent.load(args_cli.checkpoint)
        print(f"[DreamerV3] Resumed from {args_cli.checkpoint}")
        # The replay buffer is NOT checkpointed (review R2), so a resumed run starts buffer-empty.
        # Re-enter a random-action collection window of cfg.warmup_steps so the cold buffer
        # refills with exploration diversity before on-policy updates resume, instead of
        # immediately overfitting the resumed policy onto a thin fresh buffer.
        agent._warmup_until_step = agent._step + cfg.warmup_steps
        print(f"[DreamerV3] Resume: random-action buffer refill until env-step "
              f"{agent._warmup_until_step:,} (buffer starts empty — not checkpointed).",
              flush=True)

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(
        os.path.join("logs", "dreamer", args_cli.agent, args_cli.obs_mode, run_tag)
    )
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Dump the FULLY RESOLVED config (after yaml merge + CLI overrides) so every run records the
    # exact hyperparameters it used — no more guessing whether a run matched a baseline (e.g. the
    # lr=1e-4-vs-4e-5 confusion). Also dump the key CLI args.
    import yaml as _yaml
    _resolved = {k: v for k, v in vars(cfg).items() if not callable(v)}
    _resolved["_cli"] = {"task": args_cli.task, "obs_mode": args_cli.obs_mode, "agent": args_cli.agent,
                         "num_envs": env.num_envs, "max_steps": args_cli.max_steps,
                         "checkpoint": args_cli.checkpoint, "imag_start_stage": args_cli.imag_start_stage,
                         "spawn_done": args_cli.spawn_done}
    with open(os.path.join(log_dir, "resolved_config.yaml"), "w") as _f:
        _yaml.safe_dump(_resolved, _f, default_flow_style=False, sort_keys=True)
    # flush_secs=30 (default 120) prevents large blocking flushes after long uptime.
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"), flush_secs=30)
    print(f"[DreamerV3] Logging to {log_dir}  (resolved_config.yaml written)")

    # Write an initial checkpoint immediately so a future empty checkpoints/ dir can only
    # mean save_interval-triggered saves never fired (i.e. updates never happened), not that
    # the save path itself is broken. Cheap sanity probe — file is overwritten by the first
    # real save_interval-triggered save below.
    agent.save(os.path.join(ckpt_dir, "agent_init.pt"))
    print(f"[DreamerV3] Initial checkpoint written to {ckpt_dir}/agent_init.pt", flush=True)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    obs = env.reset()
    agent.reset_carry(env.num_envs)
    agent.train_mode()

    step = agent._step
    update_count = 0          # gradient update count (used for log/save triggers)
    ep_rewards = torch.zeros(env.num_envs)
    ep_gates = torch.zeros(env.num_envs, dtype=torch.float32)
    ep_lengths = torch.zeros(env.num_envs, dtype=torch.float32)
    ep_count = 0

    # Rolling per-step diagnostics (logged every gate_pass_log_every env-steps).
    # gate_pass_rate confirms gate-detection fix is working at the env level — should be > 0
    # within a few thousand env-steps once policy starts heading toward gates.
    gate_pass_log_every = max(1000, env.num_envs * 10)
    gate_pass_window_passes = 0
    gate_pass_window_steps = 0
    last_gate_pass_log = step

    # ----------------------------------------------------------------
    # Spawn-curriculum annealing.
    # A static spawn_lerp_alpha=0.9 spawns the drone AT the target gate: the first pass is
    # trivial, but on passing, the target switches to the next gate a full segment away and the
    # drone has never practised the approach — it can't chain, drifts, and the value collapses
    # (observed: episode_gates 0.40 -> 0.16, imag/value +0.05 -> -2.87 over 107k->330k steps).
    # The fix is to SHRINK alpha as the drone succeeds, so it progressively learns longer
    # approaches. Advance to the next (lower) alpha once a rolling window of recent episodes
    # averages >= ADVANCE_THRESHOLD gates/episode at the current alpha.
    cmd_term = env._isaac.command_manager.get_term(env.command_name)
    # Finer stages. The first engaged curriculum run advanced 0.9 -> 0.7 then collapsed: the
    # 1 m -> 3 m jump in approach distance was too large, the drone couldn't navigate it, gates
    # went to 0 and the value collapsed (-23). Smaller steps let it master each incremental
    # approach distance before the next.
    ALPHA_STAGES = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2]
    # ADVANCE_THRESHOLD lowered 0.6 -> 0.4. The first curriculum run peaked at ~0.50
    # gates/episode at spawn 0.9 and never reached 0.6, so the curriculum NEVER engaged
    # (spawn_lerp_alpha sat at 0.9 for 1.33M steps — effectively the static run). Set the
    # advance bar below the achievable plateau so the curriculum actually anneals the spawn
    # distance as the drone succeeds; ~0.4 gates/ep means roughly 40% of episodes clear a gate
    # at the current distance, enough signal to push the drone a stage further back.
    ADVANCE_THRESHOLD = 0.4          # mean gates/episode over the window to advance
    STAGE_WINDOW = 200               # rolling episodes considered
    MIN_EPISODES_PER_STAGE = 200     # don't advance before this many episodes at the stage
    use_curriculum = not args_cli.no_curriculum
    # --spawn_done: start at the final (fully-annealed) stage when finetuning a converged policy,
    # so it does not re-learn the easy near-gate spawns from scratch.
    stage_idx = (len(ALPHA_STAGES) - 1) if args_cli.spawn_done else 0
    recent_gates: deque = deque(maxlen=STAGE_WINDOW)
    episodes_at_stage = 0
    if use_curriculum:
        cmd_term.cfg.spawn_lerp_alpha = ALPHA_STAGES[stage_idx]
        print(f"[CURRICULUM] spawn_lerp_alpha = {ALPHA_STAGES[stage_idx]} (stage {stage_idx}/"
              f"{len(ALPHA_STAGES)-1}); anneals down as gates/episode >= {ADVANCE_THRESHOLD}",
              flush=True)

    # ----------------------------------------------------------------
    # Imagination-horizon auto-curriculum (performance-gated).
    # Single-gate passing saturates (~0.85+ gates/ep) long before multi-gate CHAINING develops,
    # because the actor trains on imagined rollouts of imag_horizon * dt seconds — at H=24,
    # dt=0.03 that is 0.72 s, far shorter than a 2-4 s gate-to-gate flight. The actor literally
    # cannot imagine reaching the next gate, so it masters the single-gate skill within reach but
    # the >=2-chain rate plateaus (observed: ~5% -> ~8% then flat over 10M steps at fixed H=24).
    # This grows the imagination reach ONLY once single-gate is mastered AND chaining has stalled,
    # extending the planning horizon to span more of the inter-gate transition. batch_size shrinks
    # in lockstep so the AC imagination-graph memory (~ H * batch * seq_len) stays ~constant —
    # the 4090 runs at ~96% VRAM, so a naive H bump would OOM. Both cfg fields are read fresh each
    # agent.update(), so live mutation takes effect on the next gradient step.
    # Imag-horizon curriculum stages = [24..48], the EXACT proven 30% baseline (run 2026-06-13).
    # batch shrinks in lockstep to hold H*batch ~= 384 (VRAM-neutral). The H=56..80 extension was
    # reverted: it only engages after single-gate mastery + chain-stall (dormant for ~20M steps),
    # and we are reproducing the clean baseline first before re-adding any single variable. The
    # extended stages live in git history (commit bb5eecd) for a later horizon test.
    IMAG_STAGES = [24, 32, 40, 48]
    BATCH_STAGES = [16, 12, 10, 8]   # H*batch ~= 384 (VRAM-neutral)
    H_WINDOW = 600                      # rolling episodes for the chaining metric
    H_MIN_EPISODES = 600                # min episodes at an H-stage before it may advance
    H_GATES_GATE = 0.85                 # single-gate must be mastered to grow H
    H_STALL_PATIENCE = 800              # episodes of no >=2-chain improvement => stalled
    H_IMPROVE_EPS = 0.005               # >=0.5 pt counts as a real chain-rate improvement
    use_imag_curriculum = use_curriculum and not args_cli.no_imag_curriculum
    # --imag_start_stage: resume the imag curriculum at a later stage (e.g. 3 = H=48) instead of
    # restarting at H=24, so a finetuned converged policy keeps its planning reach and climbs on.
    h_stage = max(0, min(args_cli.imag_start_stage, len(IMAG_STAGES) - 1))
    h_recent_gates: deque = deque(maxlen=H_WINDOW)
    h_episodes_at_stage = 0
    h_best_chain2 = 0.0
    h_stall_episodes = 0
    # Best-checkpoint tracking. agent_latest is overwritten every save_interval and agent_final
    # only captures the end-state, so when chaining peaks mid-run then erodes the peak policy is
    # lost. Save agent_best.pt whenever the smoothed gates/ep over a rolling window beats the best
    # seen so far. Smoothed (not _best_gates, which is a single-episode running max → noisy outlier).
    BEST_WINDOW = 600
    BEST_MIN_EPISODES = 600          # window must be warm before first best-save (skip early noise)
    BEST_IMPROVE_EPS = 0.02          # smoothed gates/ep must beat prior best by this to re-save
    best_recent_gates: deque = deque(maxlen=BEST_WINDOW)
    best_smoothed_gates = 0.0
    if use_imag_curriculum:
        agent.cfg.imag_horizon = IMAG_STAGES[h_stage]
        agent.cfg.batch_size = BATCH_STAGES[h_stage]
        print(f"[IMAG-CURRICULUM] imag_horizon={IMAG_STAGES[h_stage]} batch_size={BATCH_STAGES[h_stage]} "
              f"(stage {h_stage}/{len(IMAG_STAGES)-1}); grows once spawn fully annealed, gates/ep>={H_GATES_GATE}, "
              f"and >=2-chain rate stalls for {H_STALL_PATIENCE} eps", flush=True)

    while step < args_cli.max_steps:
        # --- Collect ---
        with torch.no_grad():
            actions = agent.act(obs, is_first=obs["is_first"])

        next_obs = env.step(actions.cpu())

        # DEBUG: when gate_passed observation is True, log the reward the env returned for that
        # transition AND the post-pass command state so we can verify next_gate_idx advances and
        # target_pos_b retargets to the next gate. The previous 5M-step run never passed more
        # than one gate per episode, so this print confirms whether the command pipeline is
        # actually re-pointing the drone after a successful pass.
        gp = next_obs["gate_passed"]
        if gp.any():
            idx = gp.nonzero(as_tuple=True)[0]
            rew_vals = next_obs["reward"][idx]
            cmd = env._isaac.command_manager.get_term(env.command_name)
            next_idx = cmd.next_gate_idx[idx].detach().cpu().tolist()
            # State layout (16-dim): ang_vel[0:3] quat[3:7] lin_vel[7:10] target_pos_b[10:13]
            # gate_normal_b[13:16]. (The old print used [-3:], which is now gate_normal_b, not
            # target_pos_b — fixed.)
            target_pb = next_obs["state"][idx, 10:13].detach().cpu().tolist()
            gate_nb = next_obs["state"][idx, 13:16].detach().cpu().tolist()
            print(
                f"[GATE-PASS] step={step}  envs={idx.tolist()}  "
                f"reward={[round(float(v), 3) for v in rew_vals]}  "
                f"next_gate_idx={next_idx}  "
                f"target_pos_b={[[round(float(c), 2) for c in row] for row in target_pb]}  "
                f"gate_normal_b={[[round(float(c), 2) for c in row] for row in gate_nb]}  "
                f"episode_gates={[float(ep_gates[i].item()) for i in idx]}  "
                f"is_last={[bool(next_obs['is_last'][i].item()) for i in idx]}",
                flush=True,
            )

        replay.add(
            obs,
            actions.cpu(),
            next_obs["reward"],
            obs["is_first"],
            next_obs["is_last"],
            next_obs["is_terminal"],
        )

        done_mask = next_obs["is_last"]

        # Episode tracking
        ep_rewards += next_obs["reward"]
        ep_lengths += 1.0
        gp_now = next_obs["gate_passed"].float()
        ep_gates += gp_now

        # Per-step gate pass rate (sum over envs / env-steps in window) — independent of episode end.
        gate_pass_window_passes += int(gp_now.sum().item())
        gate_pass_window_steps += env.num_envs
        if step - last_gate_pass_log >= gate_pass_log_every and gate_pass_window_steps > 0:
            rate = gate_pass_window_passes / gate_pass_window_steps
            writer.add_scalar("env/gate_pass_rate", rate, step)
            writer.add_scalar("replay/buffer_size", len(replay), step)
            writer.add_scalar("replay/num_episodes", replay.num_episodes, step)
            writer.add_scalar("curriculum/spawn_lerp_alpha", cmd_term.cfg.spawn_lerp_alpha, step)
            writer.add_scalar("curriculum/imag_horizon", agent.cfg.imag_horizon, step)
            writer.add_scalar("curriculum/batch_size", agent.cfg.batch_size, step)
            if len(h_recent_gates) > 0:
                writer.add_scalar("curriculum/chain2_rate",
                                  sum(1 for z in h_recent_gates if z >= 2) / len(h_recent_gates), step)
            gate_pass_window_passes = 0
            gate_pass_window_steps = 0
            last_gate_pass_log = step
        if done_mask.any():
            for i in done_mask.nonzero(as_tuple=True)[0]:
                gates_i = ep_gates[i].item()
                writer.add_scalar("env/episode_reward", ep_rewards[i].item(), step)
                writer.add_scalar("env/episode_length", ep_lengths[i].item(), step)
                writer.add_scalar("env/episode_gates", gates_i, step)
                if gates_i > agent._best_gates:
                    agent._best_gates = gates_i
                # Smoothed best-checkpoint: save agent_best.pt when rolling gates/ep peaks, so the
                # peak policy survives later erosion (agent_latest gets overwritten past the peak).
                best_recent_gates.append(gates_i)
                if (
                    len(best_recent_gates) >= BEST_MIN_EPISODES
                    and (sum(best_recent_gates) / len(best_recent_gates)) > best_smoothed_gates + BEST_IMPROVE_EPS
                ):
                    best_smoothed_gates = sum(best_recent_gates) / len(best_recent_gates)
                    agent.save(os.path.join(ckpt_dir, "agent_best.pt"))
                    writer.add_scalar("env/best_smoothed_gates", best_smoothed_gates, step)
                    print(f"[DreamerV3] step={step:,}  NEW BEST smoothed gates/ep="
                          f"{best_smoothed_gates:.3f}  -> agent_best.pt", flush=True)
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0.0
                ep_gates[i] = 0.0
                ep_count += 1
                # Curriculum: track per-episode gate count and advance (lower alpha) when the
                # drone is reliably passing at the current spawn distance.
                if use_curriculum:
                    recent_gates.append(gates_i)
                    episodes_at_stage += 1
                    if (
                        stage_idx < len(ALPHA_STAGES) - 1
                        and episodes_at_stage >= MIN_EPISODES_PER_STAGE
                        and len(recent_gates) >= STAGE_WINDOW
                        and (sum(recent_gates) / len(recent_gates)) >= ADVANCE_THRESHOLD
                    ):
                        prev_alpha = ALPHA_STAGES[stage_idx]
                        stage_idx += 1
                        new_alpha = ALPHA_STAGES[stage_idx]
                        cmd_term.cfg.spawn_lerp_alpha = new_alpha
                        print(f"[CURRICULUM] step={step:,}  advance stage {stage_idx-1}->{stage_idx}  "
                              f"spawn_lerp_alpha {prev_alpha} -> {new_alpha}  "
                              f"(rolling gates/ep={sum(recent_gates)/len(recent_gates):.2f})",
                              flush=True)
                        recent_gates.clear()
                        episodes_at_stage = 0

                # Imagination-horizon auto-curriculum. Engages only AFTER the spawn curriculum is
                # fully annealed (last alpha stage) — growing H while alpha is still shrinking would
                # confound the two signals. Grows H when single-gate is mastered (rolling gates/ep
                # >= H_GATES_GATE) AND the >=2-chain rate has not improved for H_STALL_PATIENCE
                # episodes (early-stopping-style patience on the chain rate).
                if use_imag_curriculum and stage_idx == len(ALPHA_STAGES) - 1:
                    h_recent_gates.append(gates_i)
                    h_episodes_at_stage += 1
                    if len(h_recent_gates) >= H_WINDOW:
                        chain2 = sum(1 for z in h_recent_gates if z >= 2) / len(h_recent_gates)
                        if chain2 > h_best_chain2 + H_IMPROVE_EPS:
                            h_best_chain2 = chain2
                            h_stall_episodes = 0
                        else:
                            h_stall_episodes += 1
                        gates_mean = sum(h_recent_gates) / len(h_recent_gates)
                        if (
                            h_stage < len(IMAG_STAGES) - 1
                            and h_episodes_at_stage >= H_MIN_EPISODES
                            and gates_mean >= H_GATES_GATE
                            and h_stall_episodes >= H_STALL_PATIENCE
                        ):
                            h_stage += 1
                            agent.cfg.imag_horizon = IMAG_STAGES[h_stage]
                            agent.cfg.batch_size = BATCH_STAGES[h_stage]
                            print(f"[IMAG-CURRICULUM] step={step:,}  advance stage {h_stage-1}->{h_stage}  "
                                  f"imag_horizon {IMAG_STAGES[h_stage-1]} -> {IMAG_STAGES[h_stage]}  "
                                  f"batch_size {BATCH_STAGES[h_stage-1]} -> {BATCH_STAGES[h_stage]}  "
                                  f"(gates/ep={gates_mean:.2f}, >=2-chain plateaued at {h_best_chain2:.3f})",
                                  flush=True)
                            h_best_chain2 = 0.0
                            h_stall_episodes = 0
                            h_episodes_at_stage = 0

        obs = next_obs
        step += env.num_envs
        agent._step = step

        # Replay-side diagnostic: confirms episodes are flushing and the buffer is sampleable.
        # If `can_sample` stays False after warmup, every agent.update(replay) returns None →
        # update_count never increments → save_interval never fires → checkpoints/ stays empty.
        if step % gate_pass_log_every == 0:
            print(
                f"[REPLAY] step={step:,}  buffer={len(replay):,}  "
                f"episodes={replay.num_episodes}  "
                f"can_sample={replay._can_sample(cfg.batch_size)}  "
                f"updates={update_count:,}",
                flush=True,
            )

        # --- Learn ---
        # Gate on agent._warmup_until_step (== cfg.warmup_steps for fresh runs; raised on resume
        # to re-collect a random-action window into the cold buffer — review R2).
        if step >= agent._warmup_until_step and step % (cfg.update_every * env.num_envs) == 0:
            for _ in range(cfg.n_grad_steps):
                metrics = agent.update(replay)
                if metrics:
                    update_count += 1
                    # Log and save by gradient-update count, not env-step count.
                    # This fires reliably regardless of num_envs or training speed.
                    if update_count % cfg.log_interval == 0:
                        for k, v in metrics.items():
                            writer.add_scalar(k, v, step)
                    if update_count % cfg.save_interval == 0:
                        agent.save(os.path.join(ckpt_dir, "agent_latest.pt"))
                        print(f"[DreamerV3] step={step:,}  updates={update_count:,}"
                              f"  episodes={ep_count}  buffer={len(replay):,}"
                              f"  best_gates={agent._best_gates:.0f}")

        if not simulation_app.is_running():
            break

    # Final checkpoint
    agent.save(os.path.join(ckpt_dir, "agent_final.pt"))
    writer.close()
    env.close()
    print(f"[DreamerV3] Training complete. Logs: {log_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
