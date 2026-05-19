"""SkyDreamer-style GateNet (arXiv:2510.14783 Appendix A).

U-Net with multi-scale supervision (5 output heads at different resolutions).
Each conv block: 2× (3×3 Conv + BatchNorm + ReLU). Down via MaxPool2d,
up via ConvTranspose2d with skip concatenation. Output via 1×1 conv to 1
channel + sigmoid (here we return logits; apply sigmoid externally).

Channel widths follow the paper's `inc-64, down1-128, down2-256, down3-512,
down4-512` ladder, scaled by `1/f`. Paper uses f=2 at 196×196 and f=4 at
384×384; for the Isaac sim 64×64 input we default f=1.

Loss: 4·L0 + 2·L1 + L2 + L3 + L4 with L_i = Dice + 2·BCE.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """SkyDreamer block: two stacked 3×3 Conv-BN-ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _Down(nn.Module):
    """MaxPool then double_conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _double_conv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    """Transposed conv upsample, concat skip, then double_conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Transposed conv: in_ch → in_ch // 2 (paper-style halving on upsample).
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class GateNet(nn.Module):
    """SkyDreamer GateNet U-Net.

    Input:  (B, in_channels, H, W) float in [0, 1].
    Output: list of 5 logit maps [y0, y1, y2, y3, y4]:
            y0 — full resolution (H,W)
            y1 — H/2, W/2
            y2 — H/4, W/4
            y3 — H/8, W/8
            y4 — H/16, W/16 (bottleneck)

    Apply `sigmoid` externally to get gate-probability masks.
    """

    def __init__(self, in_channels: int = 3, f: int = 1):
        super().__init__()

        def c(n: int) -> int:
            return max(1, n // f)

        self.inc = _double_conv(in_channels, c(64))
        self.down1 = _Down(c(64), c(128))
        self.down2 = _Down(c(128), c(256))
        self.down3 = _Down(c(256), c(512))
        self.down4 = _Down(c(512), c(512))

        self.up1 = _Up(c(512), c(512), c(256))
        self.up2 = _Up(c(256), c(256), c(128))
        self.up3 = _Up(c(128), c(128), c(64))
        self.up4 = _Up(c(64), c(64), c(64))

        # Multi-scale 1×1 output heads (outcN-1 in paper notation).
        self.outc0 = nn.Conv2d(c(64), 1, kernel_size=1)
        self.outc1 = nn.Conv2d(c(64), 1, kernel_size=1)
        self.outc2 = nn.Conv2d(c(128), 1, kernel_size=1)
        self.outc3 = nn.Conv2d(c(256), 1, kernel_size=1)
        self.outc4 = nn.Conv2d(c(512), 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Encoder
        x1 = self.inc(x)               # (B, 64,  H,    W)
        x2 = self.down1(x1)            # (B, 128, H/2,  W/2)
        x3 = self.down2(x2)            # (B, 256, H/4,  W/4)
        x4 = self.down3(x3)            # (B, 512, H/8,  W/8)
        x5 = self.down4(x4)            # (B, 512, H/16, W/16)  bottleneck

        y4 = self.outc4(x5)            # H/16 prediction

        # Decoder with skip connections
        x = self.up1(x5, x4)           # (B, 256, H/8,  W/8)
        y3 = self.outc3(x)
        x = self.up2(x, x3)            # (B, 128, H/4,  W/4)
        y2 = self.outc2(x)
        x = self.up3(x, x2)            # (B, 64,  H/2,  W/2)
        y1 = self.outc1(x)
        x = self.up4(x, x1)            # (B, 64,  H,    W)
        y0 = self.outc0(x)             # full-res prediction

        return [y0, y1, y2, y3, y4]

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Inference helper — returns binary mask at full resolution.

        x: (B, in_channels, H, W) float in [0, 1].
        Returns: (B, 1, H, W) uint8 in {0, 1}.
        """
        outputs = self.forward(x)
        prob = torch.sigmoid(outputs[0])
        return (prob > threshold).to(torch.uint8)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-batch soft Dice loss. logits/target shape (B, 1, H, W)."""
    pred = torch.sigmoid(logits)
    dims = (-3, -2, -1)   # treat channel as part of the spatial reduction
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (union + eps)
    return (1 - dice).mean()


def gatenet_loss(
    outputs: List[torch.Tensor], target_full_res: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """Multi-scale Dice + 2·BCE loss with output-specific scaling.

    total = 4·L0 + 2·L1 + L2 + L3 + L4    where L_i = Dice + 2·BCE
    Higher-resolution outputs weighted more (paper convention).

    target_full_res: (B, 1, H, W) float in [0, 1] — binary gate mask at the
    input resolution. Lower-res targets are produced by avg-pool then >0.5
    threshold so we don't lose thin structures.
    """
    scales = [1, 2, 4, 8, 16]                  # downsample factor for y0..y4
    weights = [4.0, 2.0, 1.0, 1.0, 1.0]

    total = torch.tensor(0.0, device=target_full_res.device)
    metrics: dict = {}

    for i, (y, scale, w) in enumerate(zip(outputs, scales, weights)):
        if scale == 1:
            t = target_full_res
        else:
            t = F.avg_pool2d(target_full_res, kernel_size=scale)
            t = (t > 0.5).float()

        dice = _dice_loss(y, t)
        bce = F.binary_cross_entropy_with_logits(y, t)
        L_i = dice + 2.0 * bce
        total = total + w * L_i

        metrics[f"L{i}"] = float(L_i.item())
        metrics[f"dice{i}"] = float(dice.item())

    metrics["total"] = float(total.item())
    return total, metrics
