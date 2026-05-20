# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained GateNet checkpoint on collected (image, mask) data.

Offline — no Isaac Sim required. Computes IoU / precision / recall / pixel accuracy
on a held-out split, then dumps a grid of side-by-side comparison PNGs so you can
eyeball where predictions fail.

Usage:
    python3 scripts/eval_gatenet.py \\
        --checkpoint logs/gatenet/<RUN>/checkpoints/gatenet_best.pt \\
        --data data/gatenet/train.npz \\
        --num_samples 32 \\
        --output_dir logs/gatenet/<RUN>/eval

Outputs:
    <output_dir>/metrics.txt           — overall scores
    <output_dir>/grid.png              — 8×4 grid of (RGB | GT | Pred | Overlay)
    <output_dir>/worst_iou.png         — same layout, but the 16 worst-IoU samples
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dreamer.gatenet import GateNet  # noqa: E402


def _per_image_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-image IoU. pred/gt: (N, H, W) uint8/bool. Returns (N,) float."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).reshape(pred.shape[0], -1).sum(axis=1).astype(np.float32)
    union = (pred | gt).reshape(pred.shape[0], -1).sum(axis=1).astype(np.float32)
    # Both empty → perfect score (vacuously true).
    iou = np.where(union > 0, inter / np.clip(union, 1, None), 1.0)
    return iou


def _make_grid_png(out_path: str, rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                   ious: np.ndarray) -> None:
    """Save an N×4 grid: RGB | GT | Pred | Overlay (GT=green, Pred=red, intersect=yellow).

    rgb : (N, H, W, 3) uint8
    gt  : (N, H, W)    uint8 0/255
    pred: (N, H, W)    uint8 0/255
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = rgb.shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        gt_b = gt[i] > 0
        pr_b = pred[i] > 0
        overlay = np.zeros((*gt[i].shape, 3), dtype=np.uint8)
        overlay[gt_b] = (0, 200, 0)         # GT only → green
        overlay[pr_b] = (200, 0, 0)         # Pred only → red
        overlay[gt_b & pr_b] = (220, 220, 0)  # intersect → yellow

        axes[i, 0].imshow(rgb[i]); axes[i, 0].set_title(f"RGB (IoU={ious[i]:.2f})", fontsize=8)
        axes[i, 1].imshow(gt[i], cmap="gray", vmin=0, vmax=255); axes[i, 1].set_title("GT", fontsize=8)
        axes[i, 2].imshow(pred[i], cmap="gray", vmin=0, vmax=255); axes[i, 2].set_title("Pred", fontsize=8)
        axes[i, 3].imshow(overlay); axes[i, 3].set_title("Overlay (G=GT R=Pred Y=both)", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a GateNet checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to gatenet_best.pt")
    p.add_argument("--data", required=True, help="Path to data .npz from collect_gatenet_data.py")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of samples to use for eval (deterministic seed-42 split, "
                        "matches scripts/train_gatenet.py).")
    p.add_argument("--num_samples", type=int, default=32,
                   help="How many random samples to render into the grid PNG.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sigmoid threshold applied to GateNet logits.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Default: sibling 'eval/' folder next to the checkpoint.")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[eval_gatenet] device={device}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    f = int(ckpt.get("f", 1))
    in_ch = int(ckpt.get("in_channels", 3))
    with_pose = bool(ckpt.get("with_pose", False))
    num_gates = int(ckpt.get("num_gates", 0))
    print(f"[eval_gatenet] ckpt: f={f} in_ch={in_ch} with_pose={with_pose} "
          f"num_gates={num_gates} "
          f"epoch={ckpt.get('epoch', '?')} val_loss={ckpt.get('val_loss', '?')} "
          f"val_iou={ckpt.get('val_iou', '?')}")

    model = GateNet(in_channels=in_ch, f=f, num_gates=num_gates).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ------------------------------------------------------------------
    # Load val split (same seed-42 split as the trainer)
    # ------------------------------------------------------------------
    z = np.load(args.data)
    images = z["images"]    # (N, H, W, 3) uint8
    masks = z["masks"]      # (N, H, W)    uint8 0/255
    n_total = len(images)
    n_val = int(n_total * args.val_split)
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_total)
    val_idx = perm[-n_val:]
    print(f"[eval_gatenet] data N={n_total}  val_split={args.val_split} → n_val={n_val}")

    # ------------------------------------------------------------------
    # Run model in batches over val set
    # ------------------------------------------------------------------
    all_pred = np.empty((n_val, images.shape[1], images.shape[2]), dtype=np.uint8)
    # Pose accumulators (only if checkpoint has pose head).
    pose_target_idx = z["target_idx"] if with_pose and "target_idx" in z else None
    if with_pose and pose_target_idx is None:
        print("[eval_gatenet] WARN: ckpt has pose head but data .npz lacks target_idx; "
              "skipping pose metrics.")
        with_pose = False
    pose_records = {
        "pos_b_pred": [], "pos_b_gt": [],
        "quat_b_pred": [], "quat_b_gt": [],
        "pos_w_pred": [], "pos_w_gt": [],
        "visible_pred": [], "visible_gt": [],
    } if with_pose else None

    bs = args.batch_size
    with torch.no_grad():
        for start in range(0, n_val, bs):
            end = min(start + bs, n_val)
            idx_chunk = val_idx[start:end]
            x_np = images[idx_chunk]
            x = torch.from_numpy(x_np).to(device).float() / 255.0  # (B, H, W, 3)
            x = x.permute(0, 3, 1, 2).contiguous()                  # (B, 3, H, W)
            if with_pose:
                tidx = torch.from_numpy(z["target_idx"][idx_chunk]).long().to(device)
                oh = torch.nn.functional.one_hot(tidx, num_gates).float()
                out = model(x, oh)
                logits = out["mask_logits"][0]
                pose_records["pos_b_pred"].append(out["pos_b"].cpu().numpy())
                pose_records["pos_b_gt"].append(z["target_pos_b"][idx_chunk])
                pose_records["quat_b_pred"].append(out["quat_b"].cpu().numpy())
                pose_records["quat_b_gt"].append(z["target_quat_b"][idx_chunk])
                pose_records["pos_w_pred"].append(out["pos_w"].cpu().numpy())
                pose_records["pos_w_gt"].append(z["target_pos_w"][idx_chunk])
                pose_records["visible_pred"].append(
                    (torch.sigmoid(out["visible"]) > 0.5).cpu().numpy().astype(np.uint8))
                pose_records["visible_gt"].append(z["target_visible"][idx_chunk])
            else:
                logits = model(x)[0]
            prob = torch.sigmoid(logits)
            pred = (prob > args.threshold).to(torch.uint8) * 255    # (B, 1, H, W)
            all_pred[start:end] = pred.squeeze(1).cpu().numpy()

    gt = masks[val_idx]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    iou = _per_image_iou(all_pred, gt)
    pred_b = all_pred > 0
    gt_b = gt > 0
    tp = (pred_b & gt_b).sum()
    fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b & gt_b).sum()
    tn = (~pred_b & ~gt_b).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    pix_acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    visible_gt = (gt.reshape(n_val, -1) > 0).any(axis=1)
    print()
    print(f"=== GateNet eval (n_val={n_val}) ===")
    print(f"  mean IoU                : {iou.mean():.4f}")
    print(f"  median IoU              : {np.median(iou):.4f}")
    print(f"  IoU on visible frames   : {iou[visible_gt].mean():.4f}  (n={visible_gt.sum()})")
    print(f"  IoU on empty frames     : {iou[~visible_gt].mean():.4f}  (n={(~visible_gt).sum()})")
    print(f"  precision (pixel-wise)  : {precision:.4f}")
    print(f"  recall    (pixel-wise)  : {recall:.4f}")
    print(f"  pixel accuracy          : {pix_acc:.4f}")
    print(f"  worst-IoU sample        : {iou.min():.4f}")
    print(f"  best-IoU sample         : {iou.max():.4f}")
    print()

    pose_metrics = {}
    if with_pose:
        pos_b_pred = np.concatenate(pose_records["pos_b_pred"], axis=0)
        pos_b_gt = np.concatenate(pose_records["pos_b_gt"], axis=0)
        quat_b_pred = np.concatenate(pose_records["quat_b_pred"], axis=0)
        quat_b_gt = np.concatenate(pose_records["quat_b_gt"], axis=0)
        pos_w_pred = np.concatenate(pose_records["pos_w_pred"], axis=0)
        pos_w_gt = np.concatenate(pose_records["pos_w_gt"], axis=0)
        vis_pred = np.concatenate(pose_records["visible_pred"], axis=0)
        vis_gt = np.concatenate(pose_records["visible_gt"], axis=0)

        vis_mask = vis_gt.astype(bool)
        pos_b_err_m = np.linalg.norm(pos_b_pred - pos_b_gt, axis=-1)
        pos_w_err_m = np.linalg.norm(pos_w_pred - pos_w_gt, axis=-1)
        quat_dot = np.abs(np.einsum("bi,bi->b", quat_b_pred, quat_b_gt))
        quat_angle_rad = 2.0 * np.arccos(np.clip(quat_dot, 0.0, 1.0))
        quat_angle_deg = np.rad2deg(quat_angle_rad)
        vis_acc = (vis_pred == vis_gt).mean()
        # Recall/precision on visibility classification
        tp = int(((vis_pred == 1) & (vis_gt == 1)).sum())
        fp = int(((vis_pred == 1) & (vis_gt == 0)).sum())
        fn = int(((vis_pred == 0) & (vis_gt == 1)).sum())
        vis_prec = tp / max(tp + fp, 1)
        vis_recall = tp / max(tp + fn, 1)

        print(f"=== Pose head ===")
        print(f"  pos_b L2 error (visible) : mean={pos_b_err_m[vis_mask].mean():.3f} m  "
              f"median={np.median(pos_b_err_m[vis_mask]):.3f} m  "
              f"(n_visible={int(vis_mask.sum())})")
        print(f"  pos_b L2 error (all)     : mean={pos_b_err_m.mean():.3f} m")
        print(f"  pos_w L2 error           : mean={pos_w_err_m.mean():.3f} m  "
              f"median={np.median(pos_w_err_m):.3f} m")
        print(f"  quat_b angle (visible)   : mean={quat_angle_deg[vis_mask].mean():.2f}°  "
              f"median={np.median(quat_angle_deg[vis_mask]):.2f}°")
        print(f"  visibility accuracy      : {vis_acc:.4f}  "
              f"(precision={vis_prec:.3f}  recall={vis_recall:.3f})")
        print()

        pose_metrics = {
            "pos_b_err_visible_mean_m": float(pos_b_err_m[vis_mask].mean()),
            "pos_b_err_visible_median_m": float(np.median(pos_b_err_m[vis_mask])),
            "pos_w_err_mean_m": float(pos_w_err_m.mean()),
            "quat_b_angle_visible_mean_deg": float(quat_angle_deg[vis_mask].mean()),
            "quat_b_angle_visible_median_deg": float(np.median(quat_angle_deg[vis_mask])),
            "visible_accuracy": float(vis_acc),
            "visible_precision": float(vis_prec),
            "visible_recall": float(vis_recall),
        }

    # ------------------------------------------------------------------
    # Side-by-side PNG grids
    # ------------------------------------------------------------------
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "..", "eval")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as fh:
        fh.write(f"checkpoint: {args.checkpoint}\n")
        fh.write(f"n_val: {n_val}\n")
        fh.write(f"mean_iou: {iou.mean():.4f}\n")
        fh.write(f"median_iou: {np.median(iou):.4f}\n")
        fh.write(f"iou_visible: {iou[visible_gt].mean():.4f}\n")
        fh.write(f"iou_empty: {iou[~visible_gt].mean():.4f}\n")
        fh.write(f"precision: {precision:.4f}\n")
        fh.write(f"recall: {recall:.4f}\n")
        fh.write(f"pixel_accuracy: {pix_acc:.4f}\n")
        for k, v in pose_metrics.items():
            fh.write(f"{k}: {v:.4f}\n")

    # Random sample grid
    n_show = min(args.num_samples, n_val)
    rng2 = np.random.default_rng(seed=0)
    sample_idx = rng2.choice(n_val, size=n_show, replace=False)
    _make_grid_png(
        os.path.join(out_dir, "grid.png"),
        images[val_idx[sample_idx]],
        gt[sample_idx],
        all_pred[sample_idx],
        iou[sample_idx],
    )

    # Worst-IoU grid
    worst_idx = np.argsort(iou)[:16]
    _make_grid_png(
        os.path.join(out_dir, "worst_iou.png"),
        images[val_idx[worst_idx]],
        gt[worst_idx],
        all_pred[worst_idx],
        iou[worst_idx],
    )

    print(f"[eval_gatenet] artifacts → {out_dir}")


if __name__ == "__main__":
    main()
