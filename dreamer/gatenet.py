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
    """SkyDreamer GateNet U-Net with optional pose regression head.

    Two modes:

    1. Mask-only (default, `num_gates=0`) — paper original. Returns a list of 5
       multi-scale logit maps for ANY-gate segmentation.

    2. Mask + pose (`num_gates > 0`) — adds a pose-regression branch that
       consumes the bottleneck feature plus a one-hot target-gate index and
       outputs the target gate's body-frame position/orientation and world-
       frame position, plus a visibility logit. Returns a dict instead of a
       list. Used for the standalone-perception comparative-study baseline.

    Forward signature changes accordingly:
        mask-only: forward(image) -> list[5 × (B, 1, h, w)]
        with pose: forward(image, target_idx_onehot) -> dict
            {
              "mask_logits": list[5 × (B, 1, h, w)],
              "pos_b":  (B, 3)  — target gate position in body frame (metres),
              "quat_b": (B, 4)  — target gate orientation in body frame (wxyz, unit),
              "pos_w":  (B, 3)  — target gate position in world frame (metres),
              "visible": (B,)   — logit; sigmoid gives P(target gate in view),
            }
    """

    def __init__(self, in_channels: int = 3, f: int = 1, num_gates: int = 0,
                 pose_hidden: int = 256):
        super().__init__()
        self.num_gates = num_gates

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

        # Optional pose head. Consumes bottleneck feature (pooled) + one-hot
        # target index → MLP → (pos_b, quat_b, pos_w, visible).
        if num_gates > 0:
            self._pool = nn.AdaptiveAvgPool2d(1)
            in_dim = c(512) + num_gates
            self.pose_trunk = nn.Sequential(
                nn.Linear(in_dim, pose_hidden), nn.SiLU(),
                nn.Linear(pose_hidden, pose_hidden), nn.SiLU(),
            )
            self.head_pos_b = nn.Linear(pose_hidden, 3)
            self.head_quat_b = nn.Linear(pose_hidden, 4)
            self.head_pos_w = nn.Linear(pose_hidden, 3)
            self.head_visible = nn.Linear(pose_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def _decode_masks(self, x1, x2, x3, x4, x5):
        y4 = self.outc4(x5)
        x = self.up1(x5, x4); y3 = self.outc3(x)
        x = self.up2(x, x3);  y2 = self.outc2(x)
        x = self.up3(x, x2);  y1 = self.outc1(x)
        x = self.up4(x, x1);  y0 = self.outc0(x)
        return [y0, y1, y2, y3, y4]

    def forward(self, x: torch.Tensor, target_idx_onehot: torch.Tensor | None = None):
        x1, x2, x3, x4, x5 = self._encode(x)
        mask_logits = self._decode_masks(x1, x2, x3, x4, x5)

        if self.num_gates == 0:
            return mask_logits

        if target_idx_onehot is None:
            raise ValueError(
                "GateNet was constructed with num_gates > 0; pass target_idx_onehot "
                f"of shape (B, {self.num_gates})."
            )

        feat = self._pool(x5).flatten(1)                       # (B, c(512))
        feat = torch.cat([feat, target_idx_onehot.float()], dim=-1)
        trunk = self.pose_trunk(feat)
        quat = self.head_quat_b(trunk)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return {
            "mask_logits": mask_logits,
            "pos_b": self.head_pos_b(trunk),
            "quat_b": quat,
            "pos_w": self.head_pos_w(trunk),
            "visible": self.head_visible(trunk).squeeze(-1),
        }

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Inference helper — returns binary mask at full resolution.

        x: (B, in_channels, H, W) float in [0, 1].
        Returns: (B, 1, H, W) uint8 in {0, 1}.
        """
        out = self.forward(x) if self.num_gates == 0 else None
        if out is None:
            # Pose-enabled model — feed a zero one-hot just for inference of the mask.
            zero_onehot = torch.zeros(x.shape[0], self.num_gates, device=x.device)
            out = self.forward(x, zero_onehot)["mask_logits"]
        prob = torch.sigmoid(out[0])
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


def quat_geodesic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Geodesic-style loss between unit quaternions: 1 - |q_pred · q_target|.

    Both inputs (B, 4) assumed unit-norm (or normalised inside this fn). The
    absolute value handles the q ≡ -q double-cover ambiguity.
    """
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp(min=eps)
    target = target / target.norm(dim=-1, keepdim=True).clamp(min=eps)
    dot = (pred * target).sum(dim=-1).abs()
    return (1.0 - dot).mean()


def pose_loss(
    pred: dict,
    targets: dict,
    weights: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Combined pose-head loss.

    pred  : dict with keys pos_b (B, 3), quat_b (B, 4), pos_w (B, 3), visible (B,)
    targets: dict with keys pos_b, quat_b, pos_w, visible. Same shapes; visible is
             float (B,) in {0, 1}.
    weights: optional override for term weights; defaults to
             {'pos_b': 1.0, 'quat_b': 1.0, 'pos_w': 0.5, 'visible': 0.5,
              'mask_visible_only': True}.
             If 'mask_visible_only' is True, pos_b/quat_b losses are computed
             only over frames where the target gate is visible.
    """
    w = {
        "pos_b": 1.0,
        "quat_b": 1.0,
        "pos_w": 0.5,
        "visible": 0.5,
        "mask_visible_only": True,
    }
    if weights:
        w.update(weights)

    vis = targets["visible"].float()                            # (B,)
    vis_mask = vis.bool() if w["mask_visible_only"] else torch.ones_like(vis, dtype=torch.bool)
    n_visible = int(vis_mask.sum().item())

    if n_visible > 0:
        pos_b_err = (pred["pos_b"] - targets["pos_b"]) ** 2
        L_pos_b = pos_b_err[vis_mask].mean()
        L_quat_b = quat_geodesic_loss(pred["quat_b"][vis_mask], targets["quat_b"][vis_mask])
    else:
        L_pos_b = torch.tensor(0.0, device=pred["pos_b"].device)
        L_quat_b = torch.tensor(0.0, device=pred["quat_b"].device)

    # World pose is always supervised — track is fixed so it's a label-lookup task
    # invariant to visibility.
    L_pos_w = F.mse_loss(pred["pos_w"], targets["pos_w"])
    L_visible = F.binary_cross_entropy_with_logits(pred["visible"], vis)

    total = (w["pos_b"] * L_pos_b + w["quat_b"] * L_quat_b
             + w["pos_w"] * L_pos_w + w["visible"] * L_visible)

    metrics = {
        "L_pos_b": float(L_pos_b.item()),
        "L_quat_b": float(L_quat_b.item()),
        "L_pos_w": float(L_pos_w.item()),
        "L_visible": float(L_visible.item()),
        "pose_total": float(total.item()),
        "n_visible": n_visible,
    }
    return total, metrics
