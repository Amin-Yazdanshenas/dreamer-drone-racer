# Copyright (c) 2025, Kousheek Chakraborty / Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained DreamerV3 agent on the Isaac Lab drone racing environment.

Usage:
    python3 scripts/rl/evaluate_dreamer.py \\
        --task Isaac-Drone-Racer-Dreamer-Play-v0 \\
        --obs_mode rgb \\
        --checkpoint logs/dreamer/rgb/<run>/checkpoints/agent_best.pt \\
        --num_episodes 10 \\
        --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a DreamerV3 agent.")
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-Dreamer-Play-v0")
parser.add_argument(
    "--obs_mode", type=str, default="rgb", choices=["rgb", "mask", "rgb_mask"]
)
parser.add_argument(
    "--agent", type=str, default="r2dreamer", choices=["r2dreamer", "ne_dreamer"],
    help="Agent variant matching the checkpoint."
)
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Sample actions instead of using tanh(mean).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
parser.add_argument("--num_episodes", type=int, default=10, help="Episodes to evaluate.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", action="store_true", default=False,
                    help="Record an evaluation video (requires --enable_cameras).")
parser.add_argument("--video_length", type=int, default=500)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import carb
    carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)
except Exception:
    pass

"""Rest follows after Isaac Sim init."""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).parent.resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
import csv
import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tasks  # noqa: F401
from dreamer import DreamerConfig, DreamerIsaacEnvWrapper, DreamerV3Agent, NEDreamerV3Agent

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
    import yaml

    base_path = _AGENT_TO_BASE_CONFIG[args.agent]
    mode_path = _OBS_MODE_TO_CONFIG[args.agent][args.obs_mode]
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
    cfg.__post_init__()
    return cfg


def main():
    cfg = _load_config(args_cli)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    gym_env = gym.make(args_cli.task, cfg=env_cfg,
                       render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        log_dir = os.path.dirname(os.path.dirname(args_cli.checkpoint))
        video_dir = os.path.join(log_dir, "videos", "eval")
        gym_env = gym.wrappers.RecordVideo(
            gym_env,
            video_folder=video_dir,
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = DreamerIsaacEnvWrapper(gym_env, obs_mode=args_cli.obs_mode)
    device = args_cli.device or "cuda"

    if args_cli.agent == "ne_dreamer":
        agent = NEDreamerV3Agent(cfg, device=device)
    else:
        agent = DreamerV3Agent(cfg, device=device)
    agent.load(args_cli.checkpoint)
    # Skip warmup gating during eval — checkpoint may have step < warmup_steps if loaded early.
    agent._step = max(agent._step, cfg.warmup_steps + 1)
    agent.eval_mode()
    agent.reset_carry(env.num_envs)

    # ----------------------------------------------------------------
    # Evaluation loop
    # ----------------------------------------------------------------
    results = []
    obs = env.reset()
    ep_reward = 0.0
    ep_length = 0
    ep_gates = 0
    episodes_done = 0

    print(f"[DreamerV3] Evaluating {args_cli.num_episodes} episodes …")

    while episodes_done < args_cli.num_episodes and simulation_app.is_running():
        with torch.no_grad():
            action = agent.act(
                obs,
                is_first=obs["is_first"],
                deterministic=not args_cli.stochastic,
            )

        obs = env.step(action.cpu())

        ep_reward += obs["reward"].sum().item()
        ep_length += 1

        if obs["is_last"].any():
            # Extract gate count from env info if available
            try:
                gates = env._isaac.extras.get("metrics", {}).get("gates_passed_episode", 0)
                if hasattr(gates, "item"):
                    gates = float(gates.mean().item())
                ep_gates = float(gates)
            except Exception:
                ep_gates = 0.0

            results.append({
                "episode": episodes_done + 1,
                "reward": ep_reward,
                "length": ep_length,
                "gates": ep_gates,
                "collision": int(obs["is_terminal"].any().item()),
            })
            print(f"  ep {episodes_done + 1:3d}  reward={ep_reward:8.1f}  "
                  f"length={ep_length:4d}  gates={ep_gates:.1f}")

            ep_reward = 0.0
            ep_length = 0
            ep_gates = 0
            episodes_done += 1

            obs = env.reset()
            agent.reset_carry(env.num_envs)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    if results:
        rewards = [r["reward"] for r in results]
        lengths = [r["length"] for r in results]
        gates = [r["gates"] for r in results]
        collisions = [r["collision"] for r in results]

        print("\n--- Evaluation Summary ---")
        print(f"  Episodes        : {len(results)}")
        print(f"  Reward   mean±std: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
        print(f"  Length   mean   : {np.mean(lengths):.1f}")
        print(f"  Gates    mean   : {np.mean(gates):.2f}")
        print(f"  Collision rate  : {np.mean(collisions):.2f}")

        log_dir = os.path.dirname(os.path.dirname(args_cli.checkpoint))
        csv_path = os.path.join(log_dir, "eval_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length",
                                                    "gates", "collision"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
