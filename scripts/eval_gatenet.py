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

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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


def _quat_rotate_vec(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q = (w, x, y, z). Isaac Lab convention.

    quat_wxyz: (..., 4)  v: (..., 3)   → returns (..., 3).
    """
    w = quat_wxyz[..., 0]; x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]; z = quat_wxyz[..., 3]
    # Quaternion → rotation matrix (row-vector convention applied to v).
    R = np.stack([
        1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w),
        2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y),
    ], axis=-1).reshape(*quat_wxyz.shape[:-1], 3, 3)
    return np.einsum("...ij,...j->...i", R, v)


def _draw_birdseye(ax, pos_b_gt: np.ndarray, quat_b_gt: np.ndarray,
                   pos_b_pred: np.ndarray, quat_b_pred: np.ndarray,
                   visible, gate_half_width: float = 0.75,
                   span_m: float = 8.0,
                   hfov_deg: float = 47.0) -> None:
    """Top-down (x_b vs y_b body frame) sketch.

    Accepts EITHER single-gate or multi-gate inputs:
      - Single-gate: pos_b_gt shape (3,), quat (4,), visible scalar.
      - Multi-gate : pos_b_gt shape (G, 3), quat (G, 4), visible (G,).
    Auto-detects from `pos_b_gt.ndim`. In multi-gate mode, every gate that is
    EITHER GT-visible or pred-visible is drawn, labelled with its index (1..G).
    """
    # Drone triangle (forward = +x)
    ax.add_patch(mpatches.Polygon(
        [[0.4, 0.0], [-0.2, 0.25], [-0.2, -0.25]],
        closed=True, color="blue", alpha=0.8, zorder=4))

    # Camera FOV wedge (light grey). Centred on +x_b, half-angle = hfov_deg/2.
    half = 0.5 * np.deg2rad(hfov_deg)
    ax.add_patch(mpatches.Wedge(
        center=(0.0, 0.0),
        r=span_m * 1.5,
        theta1=-np.rad2deg(half),
        theta2=+np.rad2deg(half),
        facecolor="lightgrey", edgecolor="grey", alpha=0.25,
        zorder=0,
    ))

    multi = pos_b_gt.ndim == 2
    if multi:
        G = pos_b_gt.shape[0]
        vis_arr = np.asarray(visible).astype(bool).reshape(-1)
        pos_errs = []
        ang_errs = []
        for g in range(G):
            v = bool(vis_arr[g])
            if not v:
                continue   # only draw gates that are at least GT-visible
            line_style = "-"
            line_width = 2.2
            for pos_b, quat_b, color in (
                (pos_b_gt[g], quat_b_gt[g], "green"),
                (pos_b_pred[g], quat_b_pred[g], "red"),
            ):
                lateral = _quat_rotate_vec(quat_b, np.array([0.0, 1.0, 0.0]))
                end_a = pos_b[:2] + gate_half_width * lateral[:2]
                end_b = pos_b[:2] - gate_half_width * lateral[:2]
                ax.plot([end_a[0], end_b[0]], [end_a[1], end_b[1]],
                        color=color, linewidth=line_width, linestyle=line_style, zorder=3)
                ax.scatter([pos_b[0]], [pos_b[1]], color=color, s=12, zorder=3)
            # Label gate index near the GT centre
            ax.annotate(f"g{g + 1}", (pos_b_gt[g, 0], pos_b_gt[g, 1]),
                        fontsize=6, color="black",
                        xytext=(2, 2), textcoords="offset points")
            pos_errs.append(float(np.linalg.norm(pos_b_gt[g] - pos_b_pred[g])))
            dot = abs(float((quat_b_gt[g] * quat_b_pred[g]).sum()))
            ang_errs.append(float(np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))))
        n_vis = int(vis_arr.sum())
        if pos_errs:
            title = (f"BEV (multi)  n_vis={n_vis}  "
                     f"meanΔp={np.mean(pos_errs):.2f}m  meanΔθ={np.mean(ang_errs):.1f}°")
        else:
            title = f"BEV (multi)  no visible gates"
        ax.set_title(title, fontsize=7)
    else:
        # Single-gate legacy path.
        line_style = "--" if not visible else "-"
        line_width = 2.0 if not visible else 2.6
        for pos_b, quat_b, color in (
            (pos_b_gt, quat_b_gt, "green"),
            (pos_b_pred, quat_b_pred, "red"),
        ):
            lateral = _quat_rotate_vec(quat_b, np.array([0.0, 1.0, 0.0]))
            end_a = pos_b[:2] + gate_half_width * lateral[:2]
            end_b = pos_b[:2] - gate_half_width * lateral[:2]
            ax.plot([end_a[0], end_b[0]], [end_a[1], end_b[1]],
                    color=color, linewidth=line_width, linestyle=line_style, zorder=3)
            ax.scatter([pos_b[0]], [pos_b[1]], color=color, s=15, zorder=3)
        pos_err = float(np.linalg.norm(pos_b_gt - pos_b_pred))
        dot = abs(float((quat_b_gt * quat_b_pred).sum()))
        ang_deg = float(np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))
        vis_str = "vis=Y" if visible else "vis=N (pred unconstrained)"
        ax.set_title(f"BEV  Δp={pos_err:.2f}m  Δθ={ang_deg:.1f}°  {vis_str}", fontsize=7)

    ax.set_xlim(-span_m, span_m); ax.set_ylim(-span_m, span_m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_xlabel("x_b [m]  (forward)", fontsize=6); ax.set_ylabel("y_b [m]", fontsize=6)
    ax.tick_params(labelsize=6)


def _make_grid_png(out_path: str, rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                   ious: np.ndarray,
                   pose_gt: dict | None = None,
                   pose_pred: dict | None = None,
                   visible: np.ndarray | None = None) -> None:
    """Save an N×K grid: RGB | GT | Pred | Overlay [| BEV].

    rgb : (N, H, W, 3) uint8
    gt  : (N, H, W)    uint8 0/255
    pred: (N, H, W)    uint8 0/255
    pose_gt / pose_pred : optional dicts with keys 'pos_b' (N, 3), 'quat_b' (N, 4).
                         When provided, an extra column shows a top-down BEV with
                         pose error overlay. visible: (N,) bool/uint8 — for caption.
    """
    n = rgb.shape[0]
    with_pose = pose_gt is not None and pose_pred is not None
    cols = 5 if with_pose else 4
    fig, axes = plt.subplots(n, cols, figsize=(2 * cols, 2 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        gt_b = gt[i] > 0
        pr_b = pred[i] > 0
        overlay = np.zeros((*gt[i].shape, 3), dtype=np.uint8)
        overlay[gt_b] = (0, 200, 0)
        overlay[pr_b] = (200, 0, 0)
        overlay[gt_b & pr_b] = (220, 220, 0)

        # When the dataset was collected with --frame_stack > 1 the images are 9-, 12-, …
        # channel stacks. Show only the LAST (most recent) RGB frame in the visualization.
        rgb_show = rgb[i, ..., -3:] if rgb.shape[-1] > 3 else rgb[i]
        axes[i, 0].imshow(rgb_show); axes[i, 0].set_title(f"RGB (IoU={ious[i]:.2f})", fontsize=8)
        axes[i, 1].imshow(gt[i], cmap="gray", vmin=0, vmax=255); axes[i, 1].set_title("GT mask", fontsize=8)
        axes[i, 2].imshow(pred[i], cmap="gray", vmin=0, vmax=255); axes[i, 2].set_title("Pred mask", fontsize=8)
        axes[i, 3].imshow(overlay); axes[i, 3].set_title("Overlay (G=GT R=Pred Y=both)", fontsize=8)
        for ax in axes[i, :4]:
            ax.axis("off")

        if with_pose:
            # visible[i] can be either a scalar uint8/bool (single-target ckpt) or a
            # (G,) numpy array (multi-gate ckpt). _draw_birdseye auto-detects mode
            # from pos_b_gt.ndim, so just pass the entry through unchanged.
            vis_i = visible[i] if visible is not None else True
            _draw_birdseye(
                axes[i, 4],
                pose_gt["pos_b"][i], pose_gt["quat_b"][i],
                pose_pred["pos_b"][i], pose_pred["quat_b"][i],
                visible=vis_i,
            )

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
    multi_gate = bool(ckpt.get("multi_gate", False))
    num_gates = int(ckpt.get("num_gates", 0))
    print(f"[eval_gatenet] ckpt: f={f} in_ch={in_ch} with_pose={with_pose} "
          f"multi_gate={multi_gate} num_gates={num_gates} "
          f"epoch={ckpt.get('epoch', '?')} val_loss={ckpt.get('val_loss', '?')} "
          f"val_iou={ckpt.get('val_iou', '?')}")

    model = GateNet(in_channels=in_ch, f=f, num_gates=num_gates, multi_gate=multi_gate).to(device)
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

    # Determine which pose-label keys to load (single-target vs multi-gate).
    if with_pose:
        if multi_gate:
            need_keys = ("all_pos_b", "all_quat_b", "all_pos_w", "all_visible")
        else:
            need_keys = ("target_idx", "target_pos_b", "target_quat_b",
                         "target_pos_w", "target_visible")
        if any(k not in z for k in need_keys):
            print(f"[eval_gatenet] WARN: ckpt has pose head but data .npz lacks {need_keys}; "
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
                if multi_gate:
                    out = model(x)
                    logits = out["mask_logits"][0]
                    pose_records["pos_b_pred"].append(out["pos_b"].cpu().numpy())     # (B, G, 3)
                    pose_records["pos_b_gt"].append(z["all_pos_b"][idx_chunk])
                    pose_records["quat_b_pred"].append(out["quat_b"].cpu().numpy())   # (B, G, 4)
                    pose_records["quat_b_gt"].append(z["all_quat_b"][idx_chunk])
                    pose_records["pos_w_pred"].append(out["pos_w"].cpu().numpy())     # (B, G, 3)
                    pose_records["pos_w_gt"].append(z["all_pos_w"][idx_chunk])
                    pose_records["visible_pred"].append(
                        (torch.sigmoid(out["visible"]) > 0.5).cpu().numpy().astype(np.uint8))
                    pose_records["visible_gt"].append(z["all_visible"][idx_chunk])
                else:
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

        # Reshape multi-gate arrays so per-(b, g) cells become flat rows. Then the
        # rest of the metric code is identical to the single-target path.
        if multi_gate:
            pos_b_pred = pos_b_pred.reshape(-1, 3)
            pos_b_gt = pos_b_gt.reshape(-1, 3)
            quat_b_pred = quat_b_pred.reshape(-1, 4)
            quat_b_gt = quat_b_gt.reshape(-1, 4)
            pos_w_pred = pos_w_pred.reshape(-1, 3)
            pos_w_gt = pos_w_gt.reshape(-1, 3)
            vis_pred = vis_pred.reshape(-1)
            vis_gt = vis_gt.reshape(-1)

        vis_mask = vis_gt.astype(bool)
        pos_b_err_m = np.linalg.norm(pos_b_pred - pos_b_gt, axis=-1)
        pos_w_err_m = np.linalg.norm(pos_w_pred - pos_w_gt, axis=-1)
        quat_dot = np.abs(np.einsum("bi,bi->b", quat_b_pred, quat_b_gt))
        quat_angle_rad = 2.0 * np.arccos(np.clip(quat_dot, 0.0, 1.0))
        quat_angle_deg = np.rad2deg(quat_angle_rad)
        vis_acc = (vis_pred == vis_gt).mean()
        tp = int(((vis_pred == 1) & (vis_gt == 1)).sum())
        fp = int(((vis_pred == 1) & (vis_gt == 0)).sum())
        fn = int(((vis_pred == 0) & (vis_gt == 1)).sum())
        vis_prec = tp / max(tp + fp, 1)
        vis_recall = tp / max(tp + fn, 1)

        unit = "per-(frame,gate)" if multi_gate else "per-frame"
        print(f"=== Pose head ({unit}) ===")
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
        if multi_gate:
            G = num_gates
            # Per-gate visibility accuracy (helps see if any one gate is hard to detect).
            per_gate_acc = (vis_pred.reshape(-1, G) == vis_gt.reshape(-1, G)).mean(0)
            print(f"  per-gate visibility acc  : {[round(float(v), 3) for v in per_gate_acc]}")
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

    # If pose head is enabled, pre-stack the per-sample pose arrays so grids can
    # include a body-frame bird's-eye view (column 5). For multi_gate, predictions
    # and GT are (N, G, ...) — the BEV helper handles both shapes.
    if with_pose:
        if multi_gate:
            # Re-pull stacked arrays in the (N, G, ...) layout (pre-flatten copies).
            pose_pred_all = {
                "pos_b": np.concatenate(
                    [arr.reshape(arr.shape[0], num_gates, 3) if arr.ndim == 3 else arr
                     for arr in pose_records["pos_b_pred"]], axis=0),
                "quat_b": np.concatenate(
                    [arr.reshape(arr.shape[0], num_gates, 4) if arr.ndim == 3 else arr
                     for arr in pose_records["quat_b_pred"]], axis=0),
            }
            pose_gt_all = {
                "pos_b": z["all_pos_b"][val_idx],
                "quat_b": z["all_quat_b"][val_idx],
            }
            visible_all = z["all_visible"][val_idx]    # (n_val, G)
        else:
            pose_pred_all = {
                "pos_b": np.concatenate(pose_records["pos_b_pred"], axis=0),
                "quat_b": np.concatenate(pose_records["quat_b_pred"], axis=0),
            }
            pose_gt_all = {
                "pos_b": z["target_pos_b"][val_idx],
                "quat_b": z["target_quat_b"][val_idx],
            }
            visible_all = z["target_visible"][val_idx]
    else:
        pose_pred_all = pose_gt_all = None
        visible_all = None

    def _slice_pose(idx_arr):
        if not with_pose:
            return None, None, None
        return (
            {"pos_b": pose_gt_all["pos_b"][idx_arr], "quat_b": pose_gt_all["quat_b"][idx_arr]},
            {"pos_b": pose_pred_all["pos_b"][idx_arr], "quat_b": pose_pred_all["quat_b"][idx_arr]},
            visible_all[idx_arr],
        )

    # Random sample grid
    n_show = min(args.num_samples, n_val)
    rng2 = np.random.default_rng(seed=0)
    sample_idx = rng2.choice(n_val, size=n_show, replace=False)
    pg, pp, vis = _slice_pose(sample_idx)
    _make_grid_png(
        os.path.join(out_dir, "grid.png"),
        images[val_idx[sample_idx]],
        gt[sample_idx],
        all_pred[sample_idx],
        iou[sample_idx],
        pose_gt=pg, pose_pred=pp, visible=vis,
    )

    # Worst-IoU grid
    worst_idx = np.argsort(iou)[:16]
    pg, pp, vis = _slice_pose(worst_idx)
    _make_grid_png(
        os.path.join(out_dir, "worst_iou.png"),
        images[val_idx[worst_idx]],
        gt[worst_idx],
        all_pred[worst_idx],
        iou[worst_idx],
        pose_gt=pg, pose_pred=pp, visible=vis,
    )

    # Worst-pose grid (only when pose head present) — highlight large pos_b errors
    # where the target gate IS visible. These are the failure modes that matter for
    # downstream control.
    if with_pose:
        pose_err = np.linalg.norm(
            pose_pred_all["pos_b"] - pose_gt_all["pos_b"], axis=-1
        )
        vis_mask = visible_all.astype(bool)

        if multi_gate:
            # Per-frame "badness" = MAX pos_b error over visible gates. Frames with no
            # visible gates are excluded from the worst-pose grid (no signal to score).
            err_masked = np.where(vis_mask, pose_err, -1.0)        # (N, G)
            per_frame_err = err_masked.max(axis=1)                  # (N,)
            cand = np.where(vis_mask.any(axis=1))[0]
        else:
            per_frame_err = pose_err                                # (N,)
            cand = np.where(vis_mask)[0]

        if len(cand) > 0:
            worst_pose = cand[np.argsort(per_frame_err[cand])[::-1][:16]]
            pg, pp, vis = _slice_pose(worst_pose)
            _make_grid_png(
                os.path.join(out_dir, "worst_pose.png"),
                images[val_idx[worst_pose]],
                gt[worst_pose],
                all_pred[worst_pose],
                iou[worst_pose],
                pose_gt=pg, pose_pred=pp, visible=vis,
            )

    print(f"[eval_gatenet] artifacts → {out_dir}")


if __name__ == "__main__":
    main()
