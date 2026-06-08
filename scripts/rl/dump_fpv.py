# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Dump the drone's FPV camera feed while a trained policy flies, to an mp4.

Diagnostic for the camera-angle question: shows EXACTLY what the policy's camera sees,
with the current target gate's segmentation highlighted, plus a per-frame readout of
how many target-gate pixels are in view. If the gate slides off the top of the frame
(and the pixel count drops to 0) as the drone pitches forward toward the next gate, the
level (no-tilt) camera is losing the target during forward flight.

Frame layout: [ FPV RGB | FPV + target-gate overlay (red) ], upscaled, with text:
    gate_px=<count>   in_frame=<yes/no>   target_idx=<i>

Usage (on the box with the checkpoint, non-headless or headless both fine):
    python3 scripts/rl/dump_fpv.py \\
        --checkpoint logs/dreamer/r2dreamer/rgb/<RUN>/checkpoints/agent_latest.pt \\
        --steps 1500 --fps 20 --out logs/dreamer/r2dreamer/rgb/<RUN>/eval/fpv.mp4

Add --stochastic to sample actions (matches training), --headless to skip the viewport.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump drone FPV camera feed under a trained policy.")
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-Dreamer-Play-v0")
parser.add_argument("--obs_mode", type=str, default="rgb", choices=["rgb", "mask", "rgb_mask"])
parser.add_argument("--agent", type=str, default="r2dreamer", choices=["r2dreamer", "ne_dreamer"])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=1500, help="Total RL steps to record.")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--cell", type=int, default=320, help="Upscaled panel size.")
parser.add_argument("--stochastic", action="store_true", default=False)
parser.add_argument("--out", type=str, required=True, help="Output .mp4 path.")
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

import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tasks  # noqa: F401
from dreamer import DreamerConfig, DreamerIsaacEnvWrapper, DreamerV3Agent, NEDreamerV3Agent
from dreamer.env_wrapper import _extract_gate_mask_u8, _to_rgb_u8

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
    agent._step = max(agent._step, cfg.warmup_steps + 1)   # skip warmup random actions
    agent.eval_mode()
    agent.reset_carry(env.num_envs)

    isaac = env._isaac
    cam = isaac.scene["tiled_camera"]

    frames = []
    in_frame_count = 0
    gate_seen_steps = 0
    obs = env.reset()
    for t in range(args_cli.steps):
        with torch.no_grad():
            action = agent.act(obs, is_first=obs["is_first"], deterministic=not args_cli.stochastic)
        obs = env.step(action.cpu())

        # FPV RGB the policy's camera sees
        rgb = _to_rgb_u8(cam.data.output.get("rgb"))[0].numpy()        # (H, W, 3) uint8
        # Target-gate mask (255 where the CURRENT target gate is visible)
        seg = cam.data.output.get("semantic_segmentation")
        mask = _extract_gate_mask_u8(seg, isaac, env.command_name)[0, ..., 0].numpy()  # (H, W) uint8
        gate_px = int((mask > 0).sum())
        in_view = gate_px > 0
        if in_view:
            gate_seen_steps += 1
        tidx = int(isaac.command_manager.get_term(env.command_name).next_gate_idx[0].item())

        cell = args_cli.cell
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        p_rgb = cv2.resize(bgr, (cell, cell), interpolation=cv2.INTER_NEAREST)
        ov = p_rgb.copy()
        sel = cv2.resize(mask, (cell, cell), interpolation=cv2.INTER_NEAREST) > 0
        ov[sel] = (0, 0, 230)
        p_ov = cv2.addWeighted(p_rgb, 0.5, ov, 0.5, 0)
        frame = np.concatenate([p_rgb, p_ov], axis=1)
        txt = f"gate_px={gate_px}  in_frame={'YES' if in_view else 'NO'}  target={tidx}"
        cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(frame)
        if not simulation_app.is_running():
            break

    # Write mp4 via ffmpeg (libx264; plays everywhere).
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    h, w = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}",
           "-pix_fmt", "bgr24", "-r", str(args_cli.fps), "-i", "-", "-an",
           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args_cli.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()

    pct = 100.0 * gate_seen_steps / max(1, len(frames))
    print(f"[dump_fpv] wrote {args_cli.out}  ({len(frames)} frames @ {args_cli.fps} fps)")
    print(f"[dump_fpv] target gate visible in {gate_seen_steps}/{len(frames)} frames ({pct:.1f}%) "
          f"— if this is low, the camera is losing the gate during flight.")


if __name__ == "__main__":
    main()
    simulation_app.close()
