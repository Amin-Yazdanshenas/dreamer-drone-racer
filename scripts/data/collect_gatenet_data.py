# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Collect (RGB image, gate segmentation mask) pairs for GateNet supervised training.

Runs the Dreamer racing env with random actions; for each physics step records the
TiledCamera RGB output plus the any-gate binary mask derived from semantic_segmentation.
Output is a single compressed .npz consumed by scripts/train_gatenet.py.

The collected mask is the SkyDreamer convention (Appendix A): a single binary map where
1 = ANY gate pixel, 0 = background. GateNet learns to "see gates" without knowing which
one is the next target; gate ordering is handled downstream by the flight-plan logic.

Usage:
    python3 scripts/data/collect_gatenet_data.py \\
        --num_steps 100000 \\
        --num_envs 64 \\
        --output data/gatenet/train.npz \\
        --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect GateNet supervised data.")
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-Dreamer-RGB-v0")
parser.add_argument("--num_steps", type=int, default=100_000,
                    help="Total transitions to collect (across all envs).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--output", type=str, default="data/gatenet/train.npz")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Belt-and-suspenders DLSS / raster-only — matches train_dreamer.py
try:
    import carb
    _s = carb.settings.get_settings()
    _s.set("/rtx/post/dlss/execMode", 0)
    _s.set("/rtx/rendermode", "RasterOnly")
except Exception:
    pass

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


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    # Headless data collection — skip viewport renders.
    env_cfg.sim.render_interval = 100

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    isaac_env = gym_env.unwrapped
    camera = isaac_env.scene["tiled_camera"]
    device = isaac_env.device

    print(f"[GateNet-collect] task={args_cli.task} num_envs={args_cli.num_envs} "
          f"target_steps={args_cli.num_steps}")

    images_chunks: list = []
    masks_chunks: list = []
    collected = 0

    # We need the gate class-id set, but Isaac Sim populates idToLabels lazily — a gate
    # only appears once it's been rendered in some camera. Pre-warm: step the env with
    # random actions until the discovered class set matches the known gate count, or we
    # run out of patience. Doing this BEFORE the collection loop guarantees every gate
    # is labelled correctly in every saved frame.
    gate_class_ids: set = set()
    expected_num_gates = int(isaac_env.scene["track"].num_objects)
    print(f"[GateNet-collect] track has {expected_num_gates} gates; pre-warming to discover labels …")
    gym_env.reset()
    for warm_it in range(500):
        action = torch.rand(args_cli.num_envs, 4, device=device) * 2.0 - 1.0
        gym_env.step(action)
        info = camera.data.info.get("semantic_segmentation", {})
        for k, v in info.get("idToLabels", {}).items():
            if isinstance(v, dict):
                name = v.get("class")
                if isinstance(name, str) and name.startswith("gate_"):
                    try:
                        gate_class_ids.add(int(k))
                    except (TypeError, ValueError):
                        pass
        if len(gate_class_ids) >= expected_num_gates:
            break
    print(f"[GateNet-collect] discovered after {warm_it + 1} warm steps: "
          f"gate class IDs = {sorted(gate_class_ids)} "
          f"({len(gate_class_ids)}/{expected_num_gates})")
    if len(gate_class_ids) < expected_num_gates:
        print("[GateNet-collect] WARN: some gates were never visible during warm-up; "
              "their pixels will be labelled as background in this run. Increase warm "
              "step budget or num_envs.")
    # Fresh reset for the actual collection.
    gym_env.reset()

    iters = (args_cli.num_steps + args_cli.num_envs - 1) // args_cli.num_envs

    for it in range(iters):
        # Uniform random actions in [-1, 1]^4 — broad coverage of the visual state space.
        action = torch.rand(args_cli.num_envs, 4, device=device) * 2.0 - 1.0
        gym_env.step(action)

        rgb = camera.data.output.get("rgb")                    # (N, H, W, C) uint8/float
        seg = camera.data.output.get("semantic_segmentation")  # (N, H, W, 4) uint8

        if rgb is None or seg is None:
            print(f"[GateNet-collect] WARN: missing camera output at iter {it}, skipping")
            continue

        # Normalise RGB to (N, H, W, 3) uint8.
        if rgb.dtype != torch.uint8:
            rgb_u8 = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        else:
            rgb_u8 = rgb
        rgb_u8 = rgb_u8[..., :3]   # drop alpha if present

        # Filter the segmentation map to ONLY pixels whose class label starts with "gate_".
        # Re-check every step because Isaac Sim populates idToLabels lazily — a gate's
        # class ID only appears once that gate has been rendered in at least one camera.
        # On the first iteration only the gates visible from initial spawn poses are
        # registered, so a single-shot snapshot misses the rest of the track.
        info = camera.data.info.get("semantic_segmentation", {})
        id_to_labels = info.get("idToLabels", {})
        prev_count = len(gate_class_ids)
        for k, v in id_to_labels.items():
            if isinstance(v, dict):
                name = v.get("class")
                if isinstance(name, str) and name.startswith("gate_"):
                    try:
                        gate_class_ids.add(int(k))
                    except (TypeError, ValueError):
                        pass
        if len(gate_class_ids) > prev_count:
            print(f"[GateNet-collect] gate class IDs (it={it}): {sorted(gate_class_ids)}")

        class_id = seg[..., 0]
        if gate_class_ids:
            # Bitwise-OR over the gate class set.
            gate_mask = torch.zeros_like(class_id, dtype=torch.bool)
            for cid in gate_class_ids:
                gate_mask |= (class_id == cid)
            gate_mask = gate_mask.to(torch.uint8) * 255
        else:
            # Fallback: no idToLabels available → keep old behaviour but warn loudly.
            gate_mask = (class_id > 0).to(torch.uint8) * 255

        images_chunks.append(rgb_u8.cpu().numpy())
        masks_chunks.append(gate_mask.cpu().numpy())
        collected += rgb_u8.shape[0]

        if (it + 1) % 50 == 0:
            print(f"  iter {it + 1}/{iters}  collected={collected}")

        if not simulation_app.is_running():
            break

    images = np.concatenate(images_chunks, axis=0)[: args_cli.num_steps]
    masks = np.concatenate(masks_chunks, axis=0)[: args_cli.num_steps]

    # Sanity stats
    gate_pixel_frac = float((masks > 0).mean())
    visible_frac = float((masks.reshape(masks.shape[0], -1) > 0).any(axis=1).mean())
    print(f"[GateNet-collect] images {images.shape} {images.dtype}  "
          f"masks {masks.shape} {masks.dtype}  "
          f"gate_pixel_frac={gate_pixel_frac:.4f}  visible_frac={visible_frac:.4f}")

    os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
    np.savez_compressed(args_cli.output, images=images, masks=masks)
    print(f"[GateNet-collect] saved to {args_cli.output}")

    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
