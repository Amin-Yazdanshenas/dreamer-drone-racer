# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Supervised training of GateNet (SkyDreamer U-Net) on collected (RGB, mask) pairs.

No Isaac Sim required — pure PyTorch. Run AFTER you have a .npz from
scripts/data/collect_gatenet_data.py.

Augmentations follow SkyDreamer Appendix A.c:
    - shot noise: Gaussian additive noise with std=40/255
    - mask erosion: 50%-of-batch chance to erode the mask by ~1 pixel
                    (avg-pool 3x3 then threshold ≥ 0.99)

Loss: multi-scale Dice + 2·BCE with paper weighting [4, 2, 1, 1, 1].

Usage:
    python3 scripts/train_gatenet.py \\
        --data data/gatenet/train.npz \\
        --epochs 50 \\
        --batch_size 256 \\
        --f 1
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dreamer.gatenet import GateNet, gatenet_loss, pose_loss, pose_loss_multi  # noqa: E402


class GateDataset(Dataset):
    """In-memory dataset of (RGB image, gate mask, optional pose) tuples."""

    def __init__(self, npz_path: str, augment: bool = True, with_pose: bool = False,
                 multi_gate: bool = False):
        z = np.load(npz_path)
        self.images: np.ndarray = z["images"]
        self.masks: np.ndarray = z["masks"]
        self.augment = augment
        self.with_pose = with_pose
        self.multi_gate = multi_gate

        assert self.images.shape[0] == self.masks.shape[0], "image/mask count mismatch"
        assert self.images.dtype == np.uint8, "expected uint8 images"
        assert self.images.shape[1:3] == self.masks.shape[1:3], "spatial mismatch"

        if with_pose:
            for key in ("target_idx", "target_pos_b", "target_quat_b",
                        "target_pos_w", "target_visible"):
                if key not in z:
                    raise KeyError(
                        f"--with_pose requires '{key}' in {npz_path}. "
                        "Re-collect data with the updated collect_gatenet_data.py."
                    )
            self.target_idx: np.ndarray = z["target_idx"]
            self.target_pos_b: np.ndarray = z["target_pos_b"]
            self.target_quat_b: np.ndarray = z["target_quat_b"]
            self.target_pos_w: np.ndarray = z["target_pos_w"]
            self.target_visible: np.ndarray = z["target_visible"]
            self.num_gates: int = int(z["num_gates"][0]) if "num_gates" in z \
                                   else int(self.target_idx.max() + 1)

            if multi_gate:
                for key in ("all_pos_b", "all_quat_b", "all_pos_w", "all_visible"):
                    if key not in z:
                        raise KeyError(
                            f"--multi_gate requires '{key}' in {npz_path}. "
                            "Re-collect with the latest collect_gatenet_data.py."
                        )
                self.all_pos_b: np.ndarray = z["all_pos_b"]      # (N, G, 3) f32
                self.all_quat_b: np.ndarray = z["all_quat_b"]    # (N, G, 4) f32
                self.all_pos_w: np.ndarray = z["all_pos_w"]      # (N, G, 3) f32
                self.all_visible: np.ndarray = z["all_visible"]  # (N, G) uint8
        else:
            self.num_gates = 0

        print(f"[GateDataset] {len(self.images)} samples  "
              f"image_shape={self.images.shape}  mask_shape={self.masks.shape}  "
              f"augment={augment}  with_pose={with_pose} "
              f"num_gates={self.num_gates}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).float() / 255.0  # (H, W, 3)
        img = img.permute(2, 0, 1).contiguous()                   # (3, H, W)

        mask = torch.from_numpy(self.masks[idx]).float() / 255.0  # (H, W)
        mask = mask.unsqueeze(0)                                   # (1, H, W)

        if self.augment:
            img = (img + torch.randn_like(img) * (40.0 / 255.0)).clamp(0.0, 1.0)
            if torch.rand(()) < 0.5:
                eroded = F.avg_pool2d(mask.unsqueeze(0), kernel_size=3, stride=1, padding=1)
                mask = (eroded.squeeze(0) >= 0.99).float()

        if not self.with_pose:
            return img, mask

        if self.multi_gate:
            return {
                "img": img,
                "mask": mask,
                "pos_b": torch.from_numpy(self.all_pos_b[idx]).float(),       # (G, 3)
                "quat_b": torch.from_numpy(self.all_quat_b[idx]).float(),     # (G, 4)
                "pos_w": torch.from_numpy(self.all_pos_w[idx]).float(),       # (G, 3)
                "visible": torch.from_numpy(self.all_visible[idx]).float(),   # (G,)
            }

        return {
            "img": img,
            "mask": mask,
            "target_idx": int(self.target_idx[idx]),
            "pos_b": torch.from_numpy(self.target_pos_b[idx]).float(),
            "quat_b": torch.from_numpy(self.target_quat_b[idx]).float(),
            "pos_w": torch.from_numpy(self.target_pos_w[idx]).float(),
            "visible": float(self.target_visible[idx]),
        }


def _onehot(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(idx.long(), num_classes=num_classes).float()


@torch.no_grad()
def _validate(model: GateNet, loader: DataLoader, device: torch.device,
              with_pose: bool, num_gates: int, multi_gate: bool = False) -> dict:
    model.eval()
    losses = []
    ious = []
    pose_b_errs, pos_w_errs, quat_b_errs, vis_accs = [], [], [], []

    for batch in loader:
        if with_pose:
            img = batch["img"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            if multi_gate:
                out = model(img)
                mask_logits = out["mask_logits"]
                mask_loss, _ = gatenet_loss(mask_logits, mask)
                targets = {
                    "pos_b": batch["pos_b"].to(device),
                    "quat_b": batch["quat_b"].to(device),
                    "pos_w": batch["pos_w"].to(device),
                    "visible": batch["visible"].to(device),
                }
                pose_l, _ = pose_loss_multi(out, targets)
            else:
                tidx = batch["target_idx"].to(device, non_blocking=True)
                oh = _onehot(tidx, num_gates)
                out = model(img, oh)
                mask_logits = out["mask_logits"]
                mask_loss, _ = gatenet_loss(mask_logits, mask)
                targets = {
                    "pos_b": batch["pos_b"].to(device),
                    "quat_b": batch["quat_b"].to(device),
                    "pos_w": batch["pos_w"].to(device),
                    "visible": batch["visible"].to(device),
                }
                pose_l, _ = pose_loss(out, targets)

            losses.append((mask_loss + pose_l).item())

            # Mask IoU on the full-res prediction
            pred = (torch.sigmoid(mask_logits[0]) > 0.5).float()
            inter = (pred * mask).sum().item()
            union = ((pred + mask) > 0).float().sum().clamp(min=1).item()
            ious.append(inter / union)

            # Pose metrics — averaged over visible-only entries (flat across gates
            # for multi_gate mode).
            vis_mask = targets["visible"].bool()
            if vis_mask.any():
                pose_b_err = (out["pos_b"][vis_mask] - targets["pos_b"][vis_mask]).pow(2).sum(-1).sqrt()
                pose_b_errs.append(pose_b_err.mean().item())
                quat_dot = (out["quat_b"][vis_mask] * targets["quat_b"][vis_mask]).sum(-1).abs()
                quat_b_errs.append((1.0 - quat_dot.mean()).item())
            pos_w_err = (out["pos_w"] - targets["pos_w"]).pow(2).sum(-1).sqrt().mean()
            pos_w_errs.append(pos_w_err.item())
            vis_pred = (torch.sigmoid(out["visible"]) > 0.5).float()
            vis_accs.append((vis_pred == targets["visible"]).float().mean().item())
        else:
            img, mask = batch
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            outputs = model(img)
            loss, _ = gatenet_loss(outputs, mask)
            losses.append(loss.item())
            pred = (torch.sigmoid(outputs[0]) > 0.5).float()
            inter = (pred * mask).sum().item()
            union = ((pred + mask) > 0).float().sum().clamp(min=1).item()
            ious.append(inter / union)

    out_dict = {"loss": float(np.mean(losses)), "iou": float(np.mean(ious))}
    if with_pose:
        out_dict["pos_b_err_m"] = float(np.mean(pose_b_errs)) if pose_b_errs else float("nan")
        out_dict["pos_w_err_m"] = float(np.mean(pos_w_errs))
        out_dict["quat_b_1mdot"] = float(np.mean(quat_b_errs)) if quat_b_errs else float("nan")
        out_dict["visible_acc"] = float(np.mean(vis_accs))
    return out_dict


def main() -> None:
    p = argparse.ArgumentParser(description="Train GateNet (SkyDreamer U-Net).")
    p.add_argument("--data", type=str, required=True,
                   help="Path to .npz from collect_gatenet_data.py.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--f", type=int, default=1,
                   help="Channel scaling factor. Paper: f=2 at 196x196, f=4 at 384x384. "
                        "For 64x64 input default f=1 (smaller model).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default=None,
                   help="Override log dir (default: logs/gatenet/<timestamp>).")
    p.add_argument("--with_pose", action="store_true", default=False,
                   help="Train pose regression heads (target gate body-frame pos/quat + "
                        "world-frame pos + visibility logit). Requires pose fields in the "
                        ".npz from the updated collect_gatenet_data.py.")
    p.add_argument("--multi_gate", action="store_true", default=False,
                   help="Predict pose for ALL gates simultaneously instead of one target "
                        "gate. Drops the target_idx conditioning input. Requires --with_pose "
                        "and the per-gate (all_*) arrays from the latest collector.")
    p.add_argument("--mask_weight", type=float, default=1.0,
                   help="Scale on the U-Net multi-scale mask loss when --with_pose is set.")
    p.add_argument("--pose_weight", type=float, default=1.0,
                   help="Scale on the pose-loss aggregate when --with_pose is set.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[GateNet] device={device}  f={args.f}  bs={args.batch_size}  lr={args.lr}")

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    if args.multi_gate and not args.with_pose:
        raise SystemExit("--multi_gate requires --with_pose")

    ds_full = GateDataset(args.data, augment=True,
                          with_pose=args.with_pose, multi_gate=args.multi_gate)
    val_size = int(len(ds_full) * args.val_split)
    train_size = len(ds_full) - val_size

    ds_train, ds_val = random_split(
        ds_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    ds_val_clean = GateDataset(args.data, augment=False,
                               with_pose=args.with_pose, multi_gate=args.multi_gate)
    ds_val.dataset = ds_val_clean   # type: ignore[attr-defined]
    num_gates = ds_full.num_gates

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # -----------------------------------------------------------------
    # Model + optimizer
    # -----------------------------------------------------------------
    in_channels = ds_full.images.shape[-1]
    model = GateNet(
        in_channels=in_channels,
        f=args.f,
        num_gates=num_gates if args.with_pose else 0,
        multi_gate=args.multi_gate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[GateNet] in_channels={in_channels}  n_params={n_params:,}  "
          f"with_pose={args.with_pose}  multi_gate={args.multi_gate}  num_gates={num_gates}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_dir or os.path.join("logs", "gatenet", run_tag)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"), flush_secs=30)
    print(f"[GateNet] logging to {log_dir}")

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    best_val_loss = float("inf")
    step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if args.with_pose:
                img = batch["img"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                targets = {
                    "pos_b": batch["pos_b"].to(device),
                    "quat_b": batch["quat_b"].to(device),
                    "pos_w": batch["pos_w"].to(device),
                    "visible": batch["visible"].to(device),
                }
                if args.multi_gate:
                    out = model(img)
                    pose_l, pose_m = pose_loss_multi(out, targets)
                else:
                    tidx = batch["target_idx"].to(device, non_blocking=True)
                    oh = _onehot(tidx, num_gates)
                    out = model(img, oh)
                    pose_l, pose_m = pose_loss(out, targets)
                mask_l, mask_m = gatenet_loss(out["mask_logits"], mask)
                loss = args.mask_weight * mask_l + args.pose_weight * pose_l
                sub_metrics = {f"mask/{k}": v for k, v in mask_m.items()}
                sub_metrics.update({f"pose/{k}": v for k, v in pose_m.items()})
            else:
                img, mask = batch
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                outputs = model(img)
                loss, sub_metrics = gatenet_loss(outputs, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

            if step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                for k, v in sub_metrics.items():
                    writer.add_scalar(f"train/{k}", v, step)
            step += 1

        train_loss = epoch_loss / max(1, n_batches)

        val_metrics = _validate(model, val_loader, device, args.with_pose, num_gates,
                                multi_gate=args.multi_gate)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, step)
        pose_str = ""
        if args.with_pose:
            pose_str = (f"  pos_b_err={val_metrics['pos_b_err_m']:.3f}m  "
                        f"quat_err={val_metrics['quat_b_1mdot']:.3f}  "
                        f"pos_w_err={val_metrics['pos_w_err_m']:.3f}m  "
                        f"vis_acc={val_metrics['visible_acc']:.3f}")
        print(f"epoch {epoch + 1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_iou={val_metrics['iou']:.4f}{pose_str}")

        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_metrics": val_metrics,
            "f": args.f,
            "in_channels": in_channels,
            "with_pose": args.with_pose,
            "multi_gate": args.multi_gate,
            "num_gates": num_gates if args.with_pose else 0,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "gatenet_latest.pt"))
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(ckpt, os.path.join(ckpt_dir, "gatenet_best.pt"))
            print(f"   → new best val_loss={best_val_loss:.4f}, saved gatenet_best.pt")

    writer.close()
    print(f"[GateNet] done. best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
