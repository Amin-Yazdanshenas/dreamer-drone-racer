# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Render a side-by-side mp4 of GateNet predictions.

Loads a trained GateNet checkpoint and a sample .npz, runs the network in batches,
then writes an mp4 with each frame laid out horizontally:

    [ RGB | GT mask overlay (green) | Pred mask overlay (red) | RGB | GT | Pred ]

Each panel is upsampled from 64×64 to a configurable cell size (default 256×256)
with nearest-neighbour interpolation so the mask boundaries stay crisp. The video
is intended as a short clip (a few hundred frames) for paper / report figures.

Usage:
    python3 scripts/render_gatenet_video.py \\
        --checkpoint logs/gatenet/<RUN>/checkpoints/gatenet_best.pt \\
        --data data/gatenet/train.npz \\
        --num_frames 300 \\
        --fps 30 \\
        --output logs/gatenet/<RUN>/eval/gatenet_demo.mp4

No Isaac Sim or GPU required for the video render itself (cv2 + numpy only) —
the GateNet forward pass uses whatever device is available.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dreamer.gatenet import GateNet  # noqa: E402


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------

def _upscale(img: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour upscale a (H, W) or (H, W, C) uint8 image to (size, size)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int],
             alpha: float = 0.5) -> np.ndarray:
    """Composite a binary mask onto an RGB image with alpha-blended color.

    rgb:   (H, W, 3) uint8 BGR
    mask:  (H, W) bool / uint8 (>0 = positive)
    color: (B, G, R) tuple
    """
    out = rgb.copy()
    sel = mask > 0
    overlay = np.array(color, dtype=np.uint8)
    out[sel] = (alpha * overlay + (1.0 - alpha) * rgb[sel]).astype(np.uint8)
    return out


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    """Draw a white text label with thin black outline in the top-left corner."""
    cv2.putText(panel, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
    cv2.putText(panel, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return panel


def _compose_frame(rgb_u8: np.ndarray, gt_u8: np.ndarray, pred_u8: np.ndarray,
                   cell: int) -> np.ndarray:
    """Build a (cell, 3*cell, 3) uint8 frame: [RGB | RGB+GT | RGB+Pred].

    The collector supports frame stacking (`--frame_stack K` → 3K input channels),
    in which case `rgb_u8` is (H, W, 3K). For display we always show the most
    recent frame, which sits in the last 3 channels (channel order: oldest..newest).
    """
    if rgb_u8.shape[-1] != 3:
        # Frame-stacked input — keep only the latest frame for visualization.
        rgb_u8 = rgb_u8[..., -3:]
    # cv2 expects BGR. Stored RGB → swap once here.
    rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    p_rgb = _upscale(rgb_bgr, cell)
    p_gt = _overlay(p_rgb, _upscale(gt_u8, cell), color=(0, 200, 0))     # green
    p_pr = _overlay(p_rgb, _upscale(pred_u8, cell), color=(0, 0, 230))   # red

    _label(p_rgb, "RGB")
    _label(p_gt, "GT mask")
    _label(p_pr, "GateNet pred")

    return np.concatenate([p_rgb, p_gt, p_pr], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Render a GateNet prediction mp4.")
    ap.add_argument("--checkpoint", required=True, help="Path to gatenet_best.pt")
    ap.add_argument("--data", required=True, help="Path to data .npz from collect_gatenet_data.py")
    ap.add_argument("--num_frames", type=int, default=300,
                    help="Number of frames to render. Sequential from `--start_idx`.")
    ap.add_argument("--start_idx", type=int, default=0,
                    help="First sample index in the .npz to render.")
    ap.add_argument("--cell", type=int, default=256,
                    help="Upscaled cell size for each panel (square).")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Sigmoid threshold for binary mask prediction.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", type=str, required=True,
                    help="Output .mp4 path. Parent dir created if missing.")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[render_gatenet_video] device={device}")

    # Load checkpoint -------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    f = int(ckpt.get("f", 1))
    in_ch = int(ckpt.get("in_channels", 3))
    multi_gate = bool(ckpt.get("multi_gate", False))
    num_gates = int(ckpt.get("num_gates", 0))
    print(f"[render_gatenet_video] ckpt: f={f} in_ch={in_ch} multi_gate={multi_gate} "
          f"num_gates={num_gates} val_iou={ckpt.get('val_iou', '?')}")

    model = GateNet(in_channels=in_ch, f=f, num_gates=num_gates, multi_gate=multi_gate).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load data -------------------------------------------------------------
    z = np.load(args.data)
    images = z["images"]
    masks = z["masks"]
    n_total = len(images)
    end_idx = min(args.start_idx + args.num_frames, n_total)
    idx_range = np.arange(args.start_idx, end_idx)
    print(f"[render_gatenet_video] data N={n_total}  rendering frames "
          f"[{args.start_idx}, {end_idx})")

    # Forward pass in batches ----------------------------------------------
    H, W = images.shape[1], images.shape[2]
    preds = np.empty((len(idx_range), H, W), dtype=np.uint8)
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(idx_range), bs):
            chunk = idx_range[i: i + bs]
            x = torch.from_numpy(images[chunk]).to(device).float() / 255.0
            x = x.permute(0, 3, 1, 2).contiguous()
            if multi_gate:
                logits = model(x)["mask_logits"][0]
            elif num_gates > 0:
                tidx = torch.from_numpy(z["target_idx"][chunk]).long().to(device)
                oh = torch.nn.functional.one_hot(tidx, num_gates).float()
                logits = model(x, oh)["mask_logits"][0]
            else:
                logits = model(x)[0]
            prob = torch.sigmoid(logits)
            p_u8 = (prob > args.threshold).to(torch.uint8) * 255
            preds[i: i + len(chunk)] = p_u8.squeeze(1).cpu().numpy()

    # Write mp4 -------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    frame_h = args.cell
    frame_w = args.cell * 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {args.output}")

    for k, src_idx in enumerate(idx_range):
        rgb_u8 = images[src_idx]
        gt_u8 = masks[src_idx]
        pr_u8 = preds[k]
        frame = _compose_frame(rgb_u8, gt_u8, pr_u8, cell=args.cell)
        writer.write(frame)
    writer.release()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[render_gatenet_video] wrote {args.output}  ({len(idx_range)} frames, "
          f"{frame_w}x{frame_h} @ {args.fps} fps, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
