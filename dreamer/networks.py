"""Shared neural network building blocks for DreamerV3 (PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence


# ---------------------------------------------------------------------------
# MLP with LayerNorm + SiLU (DreamerV3 default activation)
# ---------------------------------------------------------------------------

class NormMLP(nn.Module):
    """Multi-layer perceptron with LayerNorm after each linear (except output)."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int,
                 act: str = "silu", norm: bool = True):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(_act(act))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _act(name: str) -> nn.Module:
    return {"silu": nn.SiLU(), "elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[name]


# ---------------------------------------------------------------------------
# CNN Image Encoder  (64×64 → 512-dim embedding)
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """Four-layer strided CNN followed by a linear projection.

    Input:  (B, C, 64, 64)  — C = 1 (mask), 3 (RGB), 4 (RGB+mask)
    Output: (B, embed_dim)
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 512, depth: int = 32):
        super().__init__()
        # depth = base channel count; each layer doubles
        d = depth
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d, 4, stride=2), nn.ReLU(),       # 31×31
            nn.Conv2d(d, d * 2, 4, stride=2), nn.ReLU(),             # 14×14
            nn.Conv2d(d * 2, d * 4, 4, stride=2), nn.ReLU(),         # 6×6
            nn.Conv2d(d * 4, d * 8, 4, stride=2), nn.ReLU(),         # 2×2
        )
        conv_out = d * 8 * 2 * 2  # 32*8*4 = 8192 for depth=32? let's compute
        # For 64×64 input with the strides above: 64→31→14→6→2  → 2×2×(d*8)
        self.proj = nn.Linear(conv_out, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  [float32, normalised to [0,1]]
        h = self.conv(x)
        return self.proj(h.flatten(1))


# ---------------------------------------------------------------------------
# CNN Image Decoder  (latent → 64×64 image)
# ---------------------------------------------------------------------------

class ImageDecoder(nn.Module):
    """Transpose-conv decoder from RSSM latent → reconstructed image logits.

    Output logits are in ℝ (apply sigmoid for probability, BCEWithLogitsLoss for training).
    """

    def __init__(self, latent_dim: int, out_channels: int = 3, depth: int = 32):
        super().__init__()
        d = depth
        self.proj = nn.Linear(latent_dim, d * 32)    # → 32d flat, reshape to (32d, 1, 1)
        # upsample: 1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(d * 32, d * 8, 5, stride=2), nn.ReLU(),  # 5×5
            nn.ConvTranspose2d(d * 8, d * 4, 5, stride=2), nn.ReLU(),   # 13×13
            nn.ConvTranspose2d(d * 4, d * 2, 6, stride=2), nn.ReLU(),   # 30×30
            nn.ConvTranspose2d(d * 2, d, 6, stride=2), nn.ReLU(),        # 64×64
            nn.ConvTranspose2d(d, out_channels, 1),                       # 64×64, C
        )
        self.d = d

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        h = self.proj(z).reshape(z.shape[0], self.d * 32, 1, 1)
        return self.deconv(h)  # (B, C, 64, 64)


# ---------------------------------------------------------------------------
# State Encoder  (10-dim kinematics → 64-dim embedding)
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """Small MLP for encoding IMU/kinematics state vector."""

    def __init__(self, in_dim: int = 10, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
