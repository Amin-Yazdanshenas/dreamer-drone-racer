"""R2-Dreamer neural network building blocks — ported from NM512/r2dreamer."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools import weight_init_


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

def _get_act(name: str) -> nn.Module:
    acts = {
        "SiLU": nn.SiLU(),
        "silu": nn.SiLU(),
        "ReLU": nn.ReLU(),
        "relu": nn.ReLU(),
        "ELU": nn.ELU(),
        "elu": nn.ELU(),
        "GELU": nn.GELU(),
        "gelu": nn.GELU(),
    }
    if name not in acts:
        raise ValueError(f"Unknown activation: {name}")
    return acts[name]


# ---------------------------------------------------------------------------
# LambdaLayer
# ---------------------------------------------------------------------------

class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


# ---------------------------------------------------------------------------
# BlockLinear — R2-Dreamer's efficient block-diagonal linear layer
# ---------------------------------------------------------------------------

class BlockLinear(nn.Module):
    """Block-diagonal linear: splits input into `blocks` chunks, applies separate linear to each.

    Equivalent to a linear with a block-diagonal weight matrix but more memory efficient.
    in_features and out_features must both be divisible by blocks.
    """

    def __init__(self, in_features: int, out_features: int, blocks: int, bias: bool = True):
        super().__init__()
        assert in_features % blocks == 0, f"in_features {in_features} not divisible by blocks {blocks}"
        assert out_features % blocks == 0, f"out_features {out_features} not divisible by blocks {blocks}"
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.in_block = in_features // blocks
        self.out_block = out_features // blocks

        # Weight shape: (blocks, out_block, in_block)
        self.weight = nn.Parameter(torch.empty(blocks, self.out_block, self.in_block))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, 0, self.in_block ** -0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        shape = x.shape[:-1]
        x = x.reshape(-1, self.blocks, self.in_block)       # (B, blocks, in_block)
        # einsum: b = block, i = in_block, o = out_block, n = batch
        out = torch.einsum("nbi,boi->nbo", x, self.weight)  # (B, blocks, out_block)
        out = out.reshape(*shape, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Conv2dSamePad — convolution with "same" padding
# ---------------------------------------------------------------------------

class Conv2dSamePad(nn.Module):
    """Conv2d with same-padding (keeps spatial dims with stride=1)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, **kwargs):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            pad = self.kernel_size - 1
            x = F.pad(x, [pad // 2, pad - pad // 2, pad // 2, pad - pad // 2])
        return self.conv(x)


# ---------------------------------------------------------------------------
# RMSNorm2D — RMS normalisation over channels for CNN feature maps
# ---------------------------------------------------------------------------

class RMSNorm2D(nn.Module):
    """RMS normalisation over channel dimension for (B, C, H, W) tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.pow(2).mean(dim=1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# ConvEncoder — R2-Dreamer image encoder
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    """Strided CNN image encoder.

    Input:  (B, C, H, W) float [0, 1], will shift by -0.5 internally
    Output: (B, out_dim)

    depth: base channel multiplier
    mults: per-layer channel multipliers (list)
    kernel_size: conv kernel size
    act: activation name
    norm: use RMSNorm2D after each conv
    """

    def __init__(self, in_channels: int, depth: int = 16,
                 mults: Sequence[int] = (1, 2, 4, 8),
                 kernel_size: int = 4, act: str = "SiLU",
                 norm: bool = True):
        super().__init__()
        self.depth = depth
        layers: List[nn.Module] = []
        ch = in_channels
        for m in mults:
            out_ch = depth * m
            layers.append(nn.Conv2d(ch, out_ch, kernel_size, stride=2))
            if norm:
                layers.append(RMSNorm2D(out_ch))
            layers.append(_get_act(act))
            ch = out_ch
        self.conv = nn.Sequential(*layers)
        self._out_channels = ch
        self.apply(weight_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, H, W) - shift to zero-mean
        x = x - 0.5
        shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        h = self.conv(x)
        return h.flatten(1).reshape(*shape, -1)

    @property
    def out_dim(self) -> int:
        return self._compute_out_dim()

    def _compute_out_dim(self) -> int:
        return self._out_channels


def _conv_out_spatial(h: int, w: int, mults, kernel_size: int) -> Tuple[int, int]:
    """Compute spatial output after strided convolutions."""
    for _ in mults:
        h = (h - kernel_size) // 2 + 1
        w = (w - kernel_size) // 2 + 1
    return h, w


def build_conv_encoder(in_channels: int, image_h: int, image_w: int,
                       depth: int = 16, mults: Sequence[int] = (1, 2, 4, 8),
                       kernel_size: int = 4, act: str = "SiLU",
                       norm: bool = True) -> Tuple["ConvEncoder", int]:
    """Build ConvEncoder and compute its output dimension."""
    enc = ConvEncoder(in_channels, depth, mults, kernel_size, act, norm)
    h_out, w_out = _conv_out_spatial(image_h, image_w, mults, kernel_size)
    out_dim = depth * mults[-1] * h_out * w_out
    return enc, out_dim


# ---------------------------------------------------------------------------
# MLP — with optional RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.weight


class MLP(nn.Module):
    """MLP with optional RMSNorm + activation between layers.

    Uses BlockLinear if blocks > 1, otherwise standard Linear.
    """

    def __init__(self, in_dim: int, out_dim: int, units: int = 256,
                 layers: int = 2, act: str = "SiLU", norm: bool = True,
                 blocks: int = 1):
        super().__init__()
        net: List[nn.Module] = []
        ch = in_dim
        for _ in range(layers):
            if blocks > 1:
                net.append(BlockLinear(ch, units, blocks))
            else:
                net.append(nn.Linear(ch, units))
            if norm:
                net.append(RMSNorm(units))
            net.append(_get_act(act))
            ch = units
        if blocks > 1 and out_dim % blocks == 0:
            net.append(BlockLinear(ch, out_dim, blocks))
        else:
            net.append(nn.Linear(ch, out_dim))
        self.net = nn.Sequential(*net)
        self.apply(weight_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MLPHead — thin wrapper that applies symlog to inputs optionally
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """MLP with optional symlog input transform."""

    def __init__(self, in_dim: int, out_dim: int, units: int = 256,
                 layers: int = 2, act: str = "SiLU", norm: bool = True,
                 symlog_inputs: bool = False, blocks: int = 1):
        super().__init__()
        self.symlog_inputs = symlog_inputs
        self.mlp = MLP(in_dim, out_dim, units, layers, act, norm, blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.symlog_inputs:
            from .distributions import symlog
            x = symlog(x)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Projector — for Barlow Twins / contrastive SSL
# ---------------------------------------------------------------------------

class Projector(nn.Module):
    """Two-layer MLP projector for Barlow Twins loss."""

    def __init__(self, in_dim: int, proj_dim: int = 256, hidden_dim: int = 256,
                 act: str = "SiLU"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            RMSNorm(hidden_dim),
            _get_act(act),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.apply(weight_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# ReturnEMA — exponential moving average for return normalisation (R2-Dreamer)
# ---------------------------------------------------------------------------

class ReturnEMA(nn.Module):
    """EMA percentile tracker for return normalisation.

    Tracks 5th / 95th percentile via EMA. Provides scale = max(1, p95 - p5).
    Compatible with nn.Module for state_dict save/load.
    """

    def __init__(self, alpha: float = 0.02):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def scale(self) -> torch.Tensor:
        return torch.clamp(self.hi - self.lo, min=1.0)

    def update(self, values: torch.Tensor) -> None:
        v = values.detach().float()
        lo = torch.quantile(v, 0.05)
        hi = torch.quantile(v, 0.95)
        if not self.initialized.item():
            self.lo.copy_(lo)
            self.hi.copy_(hi)
            self.initialized.fill_(True)
        else:
            self.lo.copy_(self.lo * (1 - self.alpha) + lo * self.alpha)
            self.hi.copy_(self.hi * (1 - self.alpha) + hi * self.alpha)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return values / self.scale.to(values.device)


# ---------------------------------------------------------------------------
# DroneEncoder — multi-modal encoder for image + state obs
# ---------------------------------------------------------------------------

class DroneEncoder(nn.Module):
    """Encoder for drone obs: image (H, W, C) + state vector.

    image_shape: (H, W, C)
    state_dim: dimension of state vector
    cnn_depth: base channel count for ConvEncoder
    mlp_units: hidden and output size for state MLP branch
    act: activation name
    """

    def __init__(self, image_shape: Tuple[int, int, int], state_dim: int,
                 cnn_depth: int = 16, mults: Sequence[int] = (1, 2, 4, 8),
                 kernel_size: int = 4, mlp_units: int = 256, act: str = "SiLU"):
        super().__init__()
        H, W, C = image_shape
        self.image_shape = image_shape
        self.state_dim = state_dim

        # CNN branch for image (expects CHW float [0,1])
        self.cnn, cnn_out_dim = build_conv_encoder(
            C, H, W, depth=cnn_depth, mults=mults,
            kernel_size=kernel_size, act=act, norm=True,
        )

        # MLP branch for state vector
        self.state_mlp = MLPHead(
            in_dim=state_dim,
            out_dim=mlp_units,
            units=mlp_units,
            layers=2,
            act=act,
            norm=True,
            symlog_inputs=True,
        )

        self._cnn_out_dim = cnn_out_dim
        self._out_dim = cnn_out_dim + mlp_units

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        """obs['image']: (..., H, W, C) float [0,1]; obs['state']: (..., state_dim).

        Returns (..., out_dim).
        """
        image = obs["image"]  # (..., H, W, C)
        state = obs["state"]  # (..., state_dim)

        # Rearrange image to (..., C, H, W) for CNN
        shape = image.shape[:-3]
        H, W, C = image.shape[-3], image.shape[-2], image.shape[-1]
        img = image.reshape(-1, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        img_emb = self.cnn(img)  # (B, cnn_out_dim)
        img_emb = img_emb.reshape(*shape, -1)

        st_emb = self.state_mlp(state)  # (..., mlp_units)

        return torch.cat([img_emb, st_emb], dim=-1)  # (..., out_dim)


# ---------------------------------------------------------------------------
# Legacy compatibility aliases (kept so old imports don't break tests)
# ---------------------------------------------------------------------------

class NormMLP(nn.Module):
    """Legacy alias — prefer MLP. Multi-layer perceptron with LayerNorm + SiLU."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int,
                 act: str = "silu", norm: bool = True):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(_get_act(act))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):
    """Legacy ImageEncoder for old code compatibility."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 512, depth: int = 32):
        super().__init__()
        d = depth
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d, 4, stride=2), nn.ReLU(),
            nn.Conv2d(d, d * 2, 4, stride=2), nn.ReLU(),
            nn.Conv2d(d * 2, d * 4, 4, stride=2), nn.ReLU(),
            nn.Conv2d(d * 4, d * 8, 4, stride=2), nn.ReLU(),
        )
        conv_out = d * 8 * 2 * 2
        self.proj = nn.Linear(conv_out, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.proj(h.flatten(1))


class ImageDecoder(nn.Module):
    """Legacy ImageDecoder for old code compatibility."""

    def __init__(self, latent_dim: int, out_channels: int = 3, depth: int = 32):
        super().__init__()
        d = depth
        self.proj = nn.Linear(latent_dim, d * 32)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(d * 32, d * 8, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(d * 8, d * 4, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(d * 4, d * 2, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(d * 2, d, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(d, out_channels, 1),
        )
        self.d = d

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).reshape(z.shape[0], self.d * 32, 1, 1)
        return self.deconv(h)


class StateEncoder(nn.Module):
    """Legacy StateEncoder for old code compatibility."""

    def __init__(self, in_dim: int = 10, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# NEDreamerTransformer — causal temporal transformer for NE-Dreamer
# ---------------------------------------------------------------------------

class NEDreamerTransformer(nn.Module):
    """Causal transformer that predicts encoder embeddings from RSSM features.

    Ported from https://github.com/corl-team/nedreamer (MIT License).

    With use_actions=True, tokens are interleaved as [f0, a0, f1, a1, ...].
    Causal masking ensures prediction at t only sees history up to t.
    heads_next[k] predicts embed[t + k + 1] from the action token at t.
    """

    def __init__(
        self,
        feat_dim: int,
        output_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.0,
        use_actions: bool = True,
        use_same: bool = False,
        use_next: bool = True,
        predict_horizon: int = 1,
    ):
        super().__init__()
        assert use_same or use_next, "at least one of use_same or use_next must be True"
        assert predict_horizon >= 1

        self.hidden_dim = hidden_dim
        self.use_actions = use_actions
        self.use_same = use_same
        self.use_next = use_next
        self.predict_horizon = predict_horizon

        self.f_embed = nn.Linear(feat_dim, hidden_dim)
        if use_actions:
            self.a_embed = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.pos_embed = nn.Parameter(torch.zeros(1, 2 * max_seq_len, hidden_dim))
        else:
            self.a_embed = None
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def _head():
            return nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.heads_next = nn.ModuleList([_head() for _ in range(predict_horizon)]) if use_next else None
        self.head_same = _head() if use_same else None

        self._init_weights()
        nn.init.normal_(self.pos_embed, std=0.02)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(
        self, feat: torch.Tensor, actions: torch.Tensor | None = None
    ):
        """
        Args:
            feat:    (B, T, feat_dim) — RSSM features (stoch + deter)
            actions: (B, T, action_dim) — required when use_actions=True

        Returns (depends on use_same / use_next):
            use_same only:  e_hat_same   (B, T, output_dim)
            use_next only:  e_hat_next   list of (B, T-1-k, output_dim) per horizon k
            both:           (e_hat_same, e_hat_next_list)
        """
        B, T, _ = feat.shape
        device = feat.device

        tok_f = self.f_embed(feat)  # (B, T, H)

        if self.use_actions:
            assert actions is not None
            tok_a = self.a_embed(actions.float())        # (B, T, H)
            tokens = torch.stack([tok_f, tok_a], dim=2).reshape(B, 2 * T, -1)
            tokens = tokens + self.pos_embed[:, : tokens.size(1)]
            h = self.transformer(tokens, mask=self._causal_mask(tokens.size(1), device))

            f_idx = torch.arange(0, 2 * T, 2, device=device)
            h_same = h[:, f_idx]                        # (B, T, H) — feat positions

            a_idx = torch.arange(1, 2 * T - 1, 2, device=device)
            h_next = h[:, a_idx]                        # (B, T-1, H) — action positions
        else:
            tokens = tok_f + self.pos_embed[:, :T]
            h = self.transformer(tokens, mask=self._causal_mask(T, device))
            h_same = h
            h_next = h[:, :-1]                          # (B, T-1, H)

        e_hat_same = self.head_same(h_same) if self.use_same else None

        e_hat_next_list: list | None = None
        if self.use_next:
            T_h = h_next.shape[1]                       # T-1
            e_hat_next_list = []
            for k in range(self.predict_horizon):
                if k < T_h:
                    e_hat_next_list.append(self.heads_next[k](h_next[:, : T_h - k]))
                else:
                    e_hat_next_list.append(None)

        if self.use_same and self.use_next:
            return e_hat_same, e_hat_next_list
        if self.use_same:
            return e_hat_same
        return e_hat_next_list
