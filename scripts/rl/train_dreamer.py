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
parser.add_argument("--config", type=str, default=None,
                    help="Path to dreamer YAML config (default: auto from obs_mode).")
parser.add_argument("--seed", type=int, default=42)

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
_REPO_ROOT = Path(__file__).parent.resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import random
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
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = DreamerIsaacEnvWrapper(gym_env, obs_mode=args_cli.obs_mode)

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

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(
        os.path.join("logs", "dreamer", args_cli.agent, args_cli.obs_mode, run_tag)
    )
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    print(f"[DreamerV3] Logging to {log_dir}")

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

    while step < args_cli.max_steps:
        # --- Collect ---
        with torch.no_grad():
            actions = agent.act(obs, is_first=obs["is_first"])

        next_obs = env.step(actions.cpu())

        replay.add(
            obs,
            actions.cpu(),
            next_obs["reward"],
            obs["is_first"],
            next_obs["is_last"],
        )

        done_mask = next_obs["is_last"]

        # Episode tracking
        ep_rewards += next_obs["reward"]
        ep_lengths += 1.0
        ep_gates += next_obs["gate_passed"].float()
        if done_mask.any():
            for i in done_mask.nonzero(as_tuple=True)[0]:
                gates_i = ep_gates[i].item()
                writer.add_scalar("env/episode_reward", ep_rewards[i].item(), step)
                writer.add_scalar("env/episode_length", ep_lengths[i].item(), step)
                writer.add_scalar("env/episode_gates", gates_i, step)
                if gates_i > agent._best_gates:
                    agent._best_gates = gates_i
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0.0
                ep_gates[i] = 0.0
                ep_count += 1

        obs = next_obs
        step += env.num_envs
        agent._step = step

        # --- Learn ---
        if step >= cfg.warmup_steps and step % (cfg.update_every * env.num_envs) == 0:
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
