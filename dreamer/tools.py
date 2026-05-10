"""Utility functions for R2-Dreamer port."""

from __future__ import annotations

import numpy as np
import torch


def to_f32(x):
    return x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


def to_i32(x):
    return x.int() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.int32)


def rpad(x: torch.Tensor, n: int) -> torch.Tensor:
    """Append n dims of size 1 to tensor shape."""
    return x.reshape(*x.shape, *([1] * n))


def weight_init_(module: torch.nn.Module) -> None:
    """Apply lecun normal init recursively."""
    if hasattr(module, "weight") and module.weight is not None:
        torch.nn.init.normal_(module.weight, 0, module.weight.shape[-1] ** -0.5)
    if hasattr(module, "bias") and module.bias is not None:
        torch.nn.init.zeros_(module.bias)
