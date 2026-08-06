# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Record a video of a SUCCESSFUL full lap: third-person track view + the drone's FPV
camera feed on the side.

Runs the policy episode-by-episode and keeps the FIRST episode that passes >= --min_gates
gates (7 = one full lap); earlier episodes are discarded. Frame layout:

    [ third-person viewport (viewer.eye overview) | FPV 64x64 upscaled | HUD ]

Train/eval parity note: the policy trained with sim.render_interval=100 (camera refresh
every ~8 control steps). Recording needs a fresh render EVERY control step, which would
feed the policy an out-of-distribution fresh-frame stream (the collapse found in
3c4bb92). Fix: render every step for the video, but feed the policy a CACHED image
refreshed only every --obs_refresh control steps — the video is smooth, the policy sees
its training cadence.

Usage (on the box with the checkpoint):
    python3 scripts/rl/record_lap_video.py \\
        --checkpoint logs/dreamer/r2dreamer/rgb/<RUN>/checkpoints/agent_best.pt \\
        --stochastic --headless --out <RUN>/eval/lap_composite.mp4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a successful full lap with FPV side panel.")
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-Dreamer-Play-v0")
parser.add_argument("--obs_mode", type=str, default="rgb", choices=["rgb", "mask", "rgb_mask"])
parser.add_argument("--agent", type=str, default="r2dreamer", choices=["r2dreamer", "ne_dreamer"])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--min_gates", type=int, default=7, help="Keep the first episode with >= this many gates.")
parser.add_argument("--max_episodes", type=int, default=60, help="Give up after this many episodes.")
parser.add_argument("--obs_refresh", type=int, default=8,
                    help="Refresh the POLICY's cached camera image every N control steps (training parity).")
parser.add_argument("--fps", type=int, default=33, help="Output fps (control dt 0.03 s -> 33 ~= real time).")
parser.add_argument("--fpv_size", type=int, default=512, help="Upscaled FPV panel size (px).")
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
from dreamer.env_wrapper import _to_rgb_u8

_BASE = {"r2dreamer": "dreamer/configs/dreamer_base.yaml", "ne_dreamer": "dreamer/configs/ne_dreamer_base.yaml"}
_MODE = {
    "r2dreamer": {"rgb": "dreamer/configs/dreamer_rgb.yaml", "mask": "dreamer/configs/dreamer_mask.yaml",
                  "rgb_mask": "dreamer/configs/dreamer_rgb_mask.yaml"},
    "ne_dreamer": {"rgb": "dreamer/configs/ne_dreamer_rgb.yaml", "mask": "dreamer/configs/ne_dreamer_mask.yaml",
                   "rgb_mask": "dreamer/configs/ne_dreamer_rgb_mask.yaml"},
}

CTRL_DT = 0.03  # decimation 12 x sim dt 1/400


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


def _compose(view: np.ndarray, fpv_rgb: np.ndarray, gates: int, min_gates: int, t_s: float) -> np.ndarray:
    """[viewport | FPV panel] -> single BGR frame. All dims even for yuv420p."""
    s = args_cli.fpv_size
    vh = max(576, s + 64)
    vh += vh % 2
    vw = int(round(view.shape[1] * vh / view.shape[0]))
    vw += vw % 2
    main = cv2.resize(cv2.cvtColor(view, cv2.COLOR_RGB2BGR), (vw, vh), interpolation=cv2.INTER_AREA)

    panel = np.zeros((vh, s, 3), dtype=np.uint8)
    fpv = cv2.resize(cv2.cvtColor(fpv_rgb, cv2.COLOR_RGB2BGR), (s, s), interpolation=cv2.INTER_NEAREST)
    panel[:s] = fpv
    cv2.putText(panel, "FPV 64x64 (policy input)", (8, s + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    hud = f"Gates {gates}/{min_gates}   t={t_s:5.1f}s"
    cv2.putText(panel, hud, (8, s + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return np.concatenate([main, panel], axis=1)


def _write_mp4(frames: list, out_path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    h, w = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}",
           "-pix_fmt", "bgr24", "-r", str(fps), "-i", "-", "-an",
           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    cfg = _load_cfg()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    # Overview camera framing the whole 7-gate track (x in [-5,10], y in [-5,5]).
    env_cfg.viewer.eye = (16.0, -16.0, 11.0)
    env_cfg.viewer.lookat = (2.5, 0.0, 1.0)
    # render_interval stays at the PLAY default (= decimation): fresh render every control
    # step for the VIDEO. Policy-side staleness is reproduced via the cached-image shim below.
    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = DreamerIsaacEnvWrapper(gym_env, obs_mode=args_cli.obs_mode)
    device = args_cli.device or "cuda"

    agent = (NEDreamerV3Agent if args_cli.agent == "ne_dreamer" else DreamerV3Agent)(cfg, device=device)
    agent.load(args_cli.checkpoint)
    agent._step = max(agent._step, cfg.warmup_steps + 1)
    agent.eval_mode()
    agent.reset_carry(env.num_envs)

    isaac = env._isaac
    cam = isaac.scene["tiled_camera"]

    obs = env.reset()
    cached_img = obs["image"].clone()
    frames: list = []
    ep_gates, ep_steps, episodes_done = 0, 0, 0
    print(f"[record_lap] hunting for a >={args_cli.min_gates}-gate episode "
          f"(max {args_cli.max_episodes} episodes) ...", flush=True)

    while episodes_done < args_cli.max_episodes and simulation_app.is_running():
        # Parity shim: policy sees the cached image, refreshed every --obs_refresh steps.
        if ep_steps % args_cli.obs_refresh == 0 or bool(obs["is_first"].any()):
            cached_img = obs["image"].clone()
        obs_for_policy = dict(obs)
        obs_for_policy["image"] = cached_img

        with torch.no_grad():
            action = agent.act(obs_for_policy, is_first=obs["is_first"],
                               deterministic=not args_cli.stochastic)
        obs = env.step(action.cpu())
        ep_steps += 1
        ep_gates += int(obs["gate_passed"].sum().item())

        view = gym_env.render()                                   # (H, W, 3) uint8 RGB
        fpv = _to_rgb_u8(cam.data.output.get("rgb"))[0].numpy()   # (64, 64, 3) uint8
        frames.append(_compose(view, fpv, ep_gates, args_cli.min_gates, ep_steps * CTRL_DT))

        if bool(obs["is_last"].any()):
            episodes_done += 1
            if ep_gates >= args_cli.min_gates:
                print(f"[record_lap] SUCCESS: episode {episodes_done} passed {ep_gates} gates "
                      f"in {ep_steps * CTRL_DT:.1f}s — writing video", flush=True)
                _write_mp4(frames, args_cli.out, args_cli.fps)
                print(f"[record_lap] wrote {args_cli.out}  ({len(frames)} frames @ {args_cli.fps} fps)",
                      flush=True)
                return
            print(f"[record_lap] episode {episodes_done}: {ep_gates} gates — discarding", flush=True)
            frames.clear()
            ep_gates, ep_steps = 0, 0
            agent.reset_carry(env.num_envs)

    print(f"[record_lap] FAILED: no >={args_cli.min_gates}-gate episode in "
          f"{episodes_done} episodes.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
