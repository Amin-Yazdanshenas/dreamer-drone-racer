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

_REPO_ROOT = Path(__file__).parent.resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dreamer.gatenet import GateNet, gatenet_loss  # noqa: E402


class GateDataset(Dataset):
    """In-memory dataset of (RGB image, gate mask) pairs."""

    def __init__(self, npz_path: str, augment: bool = True):
        z = np.load(npz_path)
        # (N, H, W, 3) uint8 → kept on CPU as uint8 to save RAM; converted in __getitem__.
        self.images: np.ndarray = z["images"]
        # (N, H, W) uint8 in {0, 255}
        self.masks: np.ndarray = z["masks"]
        self.augment = augment

        assert self.images.shape[0] == self.masks.shape[0], "image/mask count mismatch"
        assert self.images.dtype == np.uint8, "expected uint8 images"
        assert self.images.shape[1:3] == self.masks.shape[1:3], "spatial mismatch"

        print(f"[GateDataset] {len(self.images)} samples  "
              f"image_shape={self.images.shape}  mask_shape={self.masks.shape}  "
              f"augment={augment}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).float() / 255.0  # (H, W, 3)
        img = img.permute(2, 0, 1).contiguous()                   # (3, H, W)

        mask = torch.from_numpy(self.masks[idx]).float() / 255.0  # (H, W)
        mask = mask.unsqueeze(0)                                   # (1, H, W)

        if self.augment:
            # Shot noise — std=40/255 per SkyDreamer to mimic low-exposure camera noise.
            img = (img + torch.randn_like(img) * (40.0 / 255.0)).clamp(0.0, 1.0)

            # Mask erosion: 50% chance, ~1 pixel erosion via 3×3 avg-pool + 0.99 threshold.
            if torch.rand(()) < 0.5:
                # avg_pool2d needs (N, C, H, W)
                eroded = F.avg_pool2d(mask.unsqueeze(0), kernel_size=3, stride=1, padding=1)
                mask = (eroded.squeeze(0) >= 0.99).float()

        return img, mask


@torch.no_grad()
def _validate(model: GateNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses = []
    ious = []
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        outputs = model(img)
        loss, _ = gatenet_loss(outputs, mask)
        losses.append(loss.item())
        # IoU on full-resolution prediction
        pred = (torch.sigmoid(outputs[0]) > 0.5).float()
        inter = (pred * mask).sum().item()
        union = ((pred + mask) > 0).float().sum().clamp(min=1).item()
        ious.append(inter / union)
    return {"loss": float(np.mean(losses)), "iou": float(np.mean(ious))}


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
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[GateNet] device={device}  f={args.f}  bs={args.batch_size}  lr={args.lr}")

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    ds_full = GateDataset(args.data, augment=True)
    val_size = int(len(ds_full) * args.val_split)
    train_size = len(ds_full) - val_size

    ds_train, ds_val = random_split(
        ds_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Val should not be augmented. Subset wraps the original dataset reference, so
    # toggle the flag on the underlying object during validation passes by using a
    # second non-augmented view.
    ds_val_clean = GateDataset(args.data, augment=False)
    ds_val.dataset = ds_val_clean   # type: ignore[attr-defined]

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
    model = GateNet(in_channels=in_channels, f=args.f).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[GateNet] in_channels={in_channels}  n_params={n_params:,}")

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
        for img, mask in train_loader:
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

        # Validation
        val_metrics = _validate(model, val_loader, device)
        writer.add_scalar("val/loss", val_metrics["loss"], step)
        writer.add_scalar("val/iou", val_metrics["iou"], step)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_iou={val_metrics['iou']:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "f": args.f,
            "in_channels": in_channels,
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
