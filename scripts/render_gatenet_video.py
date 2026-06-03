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
                   cell: int, canvas: tuple[int, int] | None = None) -> np.ndarray:
    """Build a 3-panel frame: [RGB | RGB+GT | RGB+Pred].

    cell: side length of each square panel before any canvas padding.
    canvas: optional (W, H) to letterbox/pillarbox the row onto. When None,
            the frame is exactly (cell, 3·cell, 3).

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

    row = np.concatenate([p_rgb, p_gt, p_pr], axis=1)   # (cell, 3*cell, 3)

    if canvas is None:
        return row

    cw, ch = canvas
    out = np.zeros((ch, cw, 3), dtype=np.uint8)
    # Centre the row on a black canvas. If the row is larger than the canvas
    # in either axis, scale down preserving aspect ratio.
    rh, rw = row.shape[:2]
    if rw > cw or rh > ch:
        scale = min(cw / rw, ch / rh)
        new_w = int(round(rw * scale))
        new_h = int(round(rh * scale))
        row = cv2.resize(row, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rh, rw = row.shape[:2]
    y0 = (ch - rh) // 2
    x0 = (cw - rw) // 2
    out[y0:y0 + rh, x0:x0 + rw] = row
    return out


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
    ap.add_argument("--canvas", type=str, default=None,
                    help="Output canvas size 'WxH' (e.g. 1920x1080). The 3-panel row "
                         "is letterboxed centred onto this. Skip to keep raw row size.")
    ap.add_argument("--youtube", action="store_true",
                    help="Preset for 1080p YouTube upload: cell=640, canvas=1920x1080. "
                         "Overrides --cell and --canvas if those weren't explicitly set.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Sigmoid threshold for binary mask prediction.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", type=str, required=True,
                    help="Output .mp4 path. Parent dir created if missing.")
    ap.add_argument("--codec", type=str, default="avc1",
                    choices=["avc1", "mp4v", "XVID", "MJPG", "ffmpeg"],
                    help="FourCC codec. 'avc1' = H.264 (default, plays in most apps). "
                         "'mp4v' is the OpenCV default but isn't recognised by some "
                         "media players. 'ffmpeg' shells out to /usr/bin/ffmpeg "
                         "(needs ffmpeg installed) which always produces a playable mp4.")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    # Apply --youtube preset before reading cell/canvas downstream.
    if args.youtube:
        if args.cell == 256:        # untouched default
            args.cell = 640
        if args.canvas is None:
            args.canvas = "1920x1080"

    canvas_wh: tuple[int, int] | None = None
    if args.canvas:
        try:
            cw, ch = (int(s) for s in args.canvas.lower().split("x"))
        except ValueError as exc:
            raise SystemExit(f"--canvas expects WxH (e.g. 1920x1080), got {args.canvas!r}") from exc
        canvas_wh = (cw, ch)

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
    if canvas_wh is not None:
        frame_w, frame_h = canvas_wh
    else:
        frame_h = args.cell
        frame_w = args.cell * 3

    if args.codec == "ffmpeg":
        _write_with_ffmpeg(args.output, args.fps, frame_w, frame_h, idx_range,
                           images, masks, preds, cell=args.cell, canvas=canvas_wh)
    else:
        _write_with_opencv(args.output, args.fps, frame_w, frame_h, args.codec,
                           idx_range, images, masks, preds, cell=args.cell,
                           canvas=canvas_wh)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[render_gatenet_video] wrote {args.output}  ({len(idx_range)} frames, "
          f"{frame_w}x{frame_h} @ {args.fps} fps, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Writer backends
# ---------------------------------------------------------------------------

def _write_with_opencv(path: str, fps: int, w: int, h: int, codec: str,
                       idx_range, images, masks, preds, cell: int,
                       canvas: tuple[int, int] | None) -> None:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {path} with codec '{codec}'. "
            f"Try --codec ffmpeg (requires /usr/bin/ffmpeg)."
        )
    for k, src_idx in enumerate(idx_range):
        frame = _compose_frame(images[src_idx], masks[src_idx], preds[k],
                               cell=cell, canvas=canvas)
        writer.write(frame)
    writer.release()


def _write_with_ffmpeg(path: str, fps: int, w: int, h: int,
                       idx_range, images, masks, preds, cell: int,
                       canvas: tuple[int, int] | None) -> None:
    """Pipe raw BGR frames into ffmpeg → libx264 mp4. Plays everywhere."""
    import subprocess
    import shutil

    ffmpeg = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for k, src_idx in enumerate(idx_range):
        frame = _compose_frame(images[src_idx], masks[src_idx], preds[k],
                               cell=cell, canvas=canvas)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited {rc} while writing {path}")


if __name__ == "__main__":
    main()
