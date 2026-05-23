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
parser.add_argument("--frame_stack", type=int, default=1,
                    help="Number of consecutive RGB frames to concatenate along the "
                         "channel axis at each step (gives the network temporal parallax "
                         "for better depth/pos_b estimates). 1 = single frame (default), "
                         "3 = RGB×3 → 9 input channels.")
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
import isaaclab.utils.math as math_utils
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
    # Per-frame extras (single-target legacy — kept for backward compatibility):
    target_idx_chunks: list = []
    target_pos_b_chunks: list = []
    target_quat_b_chunks: list = []
    target_pos_w_chunks: list = []
    target_quat_w_chunks: list = []
    target_visible_chunks: list = []
    # NEW: per-frame per-gate arrays for multi-gate detection training.
    all_pos_b_chunks: list = []     # (N, G, 3)
    all_quat_b_chunks: list = []    # (N, G, 4)
    all_pos_w_chunks: list = []     # (N, G, 3)
    all_quat_w_chunks: list = []    # (N, G, 4)
    all_visible_chunks: list = []   # (N, G) uint8
    collected = 0

    # Rolling buffer of last K RGB frames per env. Stored as numpy uint8 to avoid
    # holding GPU memory between iterations. Padded by duplicating the first frame
    # so every saved sample has exactly K frames available (no zero-frames at t=0).
    K = max(1, int(args_cli.frame_stack))
    prev_frames: list = []  # list of K (N, H, W, 3) uint8 numpy arrays — oldest first
    print(f"[GateNet-collect] frame_stack K={K}  (effective input channels = {3 * K})")

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

        rgb_np = rgb_u8.cpu().numpy()           # (N, H, W, 3) uint8

        # Update rolling K-frame buffer. On first iteration, pre-fill with copies of
        # the current frame so the saved stack always has K frames (no zero frames).
        if not prev_frames:
            prev_frames = [rgb_np.copy() for _ in range(K)]
        else:
            prev_frames.pop(0)
            prev_frames.append(rgb_np)

        # Stack oldest → newest along the channel axis. Last 3 channels = current frame.
        if K == 1:
            stacked_np = rgb_np
        else:
            stacked_np = np.concatenate(prev_frames, axis=-1)   # (N, H, W, 3*K) uint8

        images_chunks.append(stacked_np)
        masks_chunks.append(gate_mask.cpu().numpy())

        # --- Per-frame pose labels for the comparative-study regression heads ---
        # target_idx is the index (0..num_gates-1) of the gate the drone is currently
        # supposed to fly through. Pull world-frame pose of that gate, then convert
        # to drone body frame for the body-frame pose targets.
        cmd = isaac_env.command_manager.get_term("target")
        target_idx = cmd.next_gate_idx.long()                          # (N,)
        track = isaac_env.scene["track"]
        env_ids = torch.arange(args_cli.num_envs, device=device)
        gate_pos_w = track.data.object_com_pos_w[env_ids, target_idx]   # (N, 3)
        gate_quat_w = track.data.object_quat_w[env_ids, target_idx]     # (N, 4) wxyz

        robot = isaac_env.scene["robot"]
        drone_pos_w = robot.data.root_pos_w                              # (N, 3)
        drone_quat_w = robot.data.root_quat_w                            # (N, 4)

        # Body-frame position via Isaac Lab helper (gives gate position expressed in
        # the drone body frame).
        target_pos_b, _ = math_utils.subtract_frame_transforms(
            drone_pos_w, drone_quat_w, gate_pos_w,
        )
        # Body-frame orientation: gate_quat_b = drone_quat_w^{-1} ⊗ gate_quat_w
        drone_quat_w_inv = math_utils.quat_inv(drone_quat_w)
        target_quat_b = math_utils.quat_mul(drone_quat_w_inv, gate_quat_w)

        # Visibility: target gate has its own class_id (target_idx + 2 since IDs are
        # contiguous starting at 2 on this track) — derive from idToLabels lookup
        # we already built. For each env, check if THAT specific class id is present
        # in the seg map. Fallback: any gate visible.
        # idToLabels keys are strings; build name→id lookup once.
        # We use the per-frame label set for correctness.
        info = camera.data.info.get("semantic_segmentation", {})
        name_to_id = {}
        for k, v in info.get("idToLabels", {}).items():
            if isinstance(v, dict):
                nm = v.get("class")
                if isinstance(nm, str) and nm.startswith("gate_"):
                    try:
                        name_to_id[nm] = int(k)
                    except (TypeError, ValueError):
                        pass
        # Per-env class id of the *target* gate
        target_cls = torch.zeros(args_cli.num_envs, dtype=class_id.dtype, device=class_id.device)
        for i in range(args_cli.num_envs):
            tname = f"gate_{int(target_idx[i].item()) + 1}"
            target_cls[i] = name_to_id.get(tname, 0)
        # Visible iff the target-class id appears in ANY pixel of that env's seg map
        visible = ((class_id == target_cls[:, None, None]).reshape(args_cli.num_envs, -1)
                   .any(dim=1).to(torch.uint8))

        target_idx_chunks.append(target_idx.to(torch.uint8).cpu().numpy())
        target_pos_b_chunks.append(target_pos_b.cpu().numpy().astype(np.float32))
        target_quat_b_chunks.append(target_quat_b.cpu().numpy().astype(np.float32))
        target_pos_w_chunks.append(gate_pos_w.cpu().numpy().astype(np.float32))
        target_quat_w_chunks.append(gate_quat_w.cpu().numpy().astype(np.float32))
        target_visible_chunks.append(visible.cpu().numpy())

        # ---------- Per-gate (multi-gate detection) ground truth ----------
        # For every gate on the track, compute body-frame pose and per-gate visibility.
        # Used by the multi-gate output head — network must predict pose for every gate
        # it sees, not just the target. Shapes: (N_envs, num_gates, ...).
        N_envs = args_cli.num_envs
        G = expected_num_gates

        all_pos_w_t = track.data.object_com_pos_w.to(device)         # (N, G, 3)
        all_quat_w_t = track.data.object_quat_w.to(device)           # (N, G, 4)

        # Body-frame pose: subtract_frame_transforms expects flat (N*G, *) tensors.
        drone_pos_w_exp = drone_pos_w.unsqueeze(1).expand(-1, G, -1).reshape(-1, 3)
        drone_quat_w_exp = drone_quat_w.unsqueeze(1).expand(-1, G, -1).reshape(-1, 4)
        all_pos_w_flat = all_pos_w_t.reshape(-1, 3)
        all_quat_w_flat = all_quat_w_t.reshape(-1, 4)
        all_pos_b_flat, _ = math_utils.subtract_frame_transforms(
            drone_pos_w_exp, drone_quat_w_exp, all_pos_w_flat,
        )
        all_pos_b_t = all_pos_b_flat.reshape(N_envs, G, 3)

        drone_quat_inv_exp = math_utils.quat_inv(drone_quat_w).unsqueeze(1)
        drone_quat_inv_exp = drone_quat_inv_exp.expand(-1, G, -1).reshape(-1, 4)
        all_quat_b_flat = math_utils.quat_mul(drone_quat_inv_exp, all_quat_w_flat)
        all_quat_b_t = all_quat_b_flat.reshape(N_envs, G, 4)

        # Per-gate visibility: each gate's class id appears in this env's seg map?
        # Build (G,) tensor of class IDs (0 when label unknown — gate counted invisible).
        gate_cls_per_idx = torch.zeros(G, dtype=class_id.dtype, device=class_id.device)
        for g in range(G):
            gate_cls_per_idx[g] = name_to_id.get(f"gate_{g + 1}", 0)
        # Compare seg pixels against each gate's class id, reduce H,W.
        flat_seg = class_id.reshape(N_envs, -1)                      # (N, H*W)
        # (N, H*W, 1) == (1, 1, G) → (N, H*W, G) bool → any over pixel axis
        per_gate_vis = (
            flat_seg.unsqueeze(-1) == gate_cls_per_idx.view(1, 1, G)
        ).any(dim=1).to(torch.uint8)                                 # (N, G)

        all_pos_b_chunks.append(all_pos_b_t.cpu().numpy().astype(np.float32))
        all_quat_b_chunks.append(all_quat_b_t.cpu().numpy().astype(np.float32))
        all_pos_w_chunks.append(all_pos_w_t.cpu().numpy().astype(np.float32))
        all_quat_w_chunks.append(all_quat_w_t.cpu().numpy().astype(np.float32))
        all_visible_chunks.append(per_gate_vis.cpu().numpy())

        collected += rgb_u8.shape[0]

        if (it + 1) % 50 == 0:
            print(f"  iter {it + 1}/{iters}  collected={collected}")

        if not simulation_app.is_running():
            break

    images = np.concatenate(images_chunks, axis=0)[: args_cli.num_steps]
    masks = np.concatenate(masks_chunks, axis=0)[: args_cli.num_steps]
    target_idx_arr = np.concatenate(target_idx_chunks, axis=0)[: args_cli.num_steps]
    pos_b_arr = np.concatenate(target_pos_b_chunks, axis=0)[: args_cli.num_steps]
    quat_b_arr = np.concatenate(target_quat_b_chunks, axis=0)[: args_cli.num_steps]
    pos_w_arr = np.concatenate(target_pos_w_chunks, axis=0)[: args_cli.num_steps]
    quat_w_arr = np.concatenate(target_quat_w_chunks, axis=0)[: args_cli.num_steps]
    target_visible_arr = np.concatenate(target_visible_chunks, axis=0)[: args_cli.num_steps]

    all_pos_b_arr = np.concatenate(all_pos_b_chunks, axis=0)[: args_cli.num_steps]
    all_quat_b_arr = np.concatenate(all_quat_b_chunks, axis=0)[: args_cli.num_steps]
    all_pos_w_arr = np.concatenate(all_pos_w_chunks, axis=0)[: args_cli.num_steps]
    all_quat_w_arr = np.concatenate(all_quat_w_chunks, axis=0)[: args_cli.num_steps]
    all_visible_arr = np.concatenate(all_visible_chunks, axis=0)[: args_cli.num_steps]

    # Sanity stats
    gate_pixel_frac = float((masks > 0).mean())
    visible_frac = float((masks.reshape(masks.shape[0], -1) > 0).any(axis=1).mean())
    target_visible_frac = float(target_visible_arr.mean())
    print(f"[GateNet-collect] images {images.shape} {images.dtype}  "
          f"masks {masks.shape} {masks.dtype}  "
          f"gate_pixel_frac={gate_pixel_frac:.4f}  any_visible_frac={visible_frac:.4f}  "
          f"target_visible_frac={target_visible_frac:.4f}")
    print(f"[GateNet-collect] pose ranges: "
          f"pos_b min/max={pos_b_arr.min():.2f}/{pos_b_arr.max():.2f}  "
          f"pos_w min/max={pos_w_arr.min():.2f}/{pos_w_arr.max():.2f}")
    # Per-gate stats
    per_gate_vis_frac = all_visible_arr.mean(axis=0)   # (G,)
    avg_n_visible = float(all_visible_arr.sum(axis=1).mean())
    print(f"[GateNet-collect] per-gate visibility frac: "
          f"{[float(round(v, 3)) for v in per_gate_vis_frac]}  "
          f"avg gates in view per frame={avg_n_visible:.2f}")

    os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        images=images,
        masks=masks,
        # Single-target legacy (kept for backward compat with old training runs).
        target_idx=target_idx_arr,
        target_pos_b=pos_b_arr,
        target_quat_b=quat_b_arr,
        target_pos_w=pos_w_arr,
        target_quat_w=quat_w_arr,
        target_visible=target_visible_arr,
        # Multi-gate per-frame arrays: shape (N, G, ...) (or (N, G) for visibility).
        all_pos_b=all_pos_b_arr,
        all_quat_b=all_quat_b_arr,
        all_pos_w=all_pos_w_arr,
        all_quat_w=all_quat_w_arr,
        all_visible=all_visible_arr,
        num_gates=np.array([expected_num_gates], dtype=np.int32),
        frame_stack=np.array([K], dtype=np.int32),
    )
    print(f"[GateNet-collect] saved to {args_cli.output}")

    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
