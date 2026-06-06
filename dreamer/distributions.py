"""Distributions — faithful R2-Dreamer port (NM512/r2dreamer).

Key port from local code:
- OneHotDist accepts unimix_ratio that blends softmax probs with a uniform prior
  (probs = (1 - r) * softmax(logits) + r / K). This stops categories from sharpening
  too aggressively early in training, which is what `wm/kl stuck at floor` looks
  like when posterior collapse hits.
- rsample() uses straight-through Gumbel-Softmax (`F.gumbel_softmax(hard=True)`)
  instead of the previous `torch.multinomial` + STE trick.
- sample() raises NotImplementedError — the upstream code disables non-rsample
  paths so callers can't accidentally drop gradient.
- TwoHot is kept compatible with the existing reward / value heads: bins are
  linear over [-20, 20] and the caller is expected to symlog the target before
  calling `log_prob`. `mode()` returns the value in symlog-space too; callers
  symexp it back. This matches how `agent.py:_world_model_loss` already feeds
  the reward head.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd


# ---------------------------------------------------------------------------
# symlog / symexp
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


# ---------------------------------------------------------------------------
# OneHotDist — categorical with unimix and Gumbel-Softmax straight-through
# ---------------------------------------------------------------------------

class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    """One-hot categorical with optional uniform-mix prior and Gumbel-Softmax rsample.

    Args:
        logits: (..., K) raw logits.
        unimix_ratio: float in [0, 1]. The forward distribution's probs are
            `(1 - r) * softmax(logits) + r / K`. r=0 reproduces vanilla softmax.
    """

    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.0):
        probs = F.softmax(logits.float(), dim=-1)
        if unimix_ratio > 0.0:
            uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
            probs = (1.0 - unimix_ratio) * probs + unimix_ratio * uniform
        # Convert back to logits for the parent class — `log` is safe because
        # the uniform mix guarantees strictly positive probabilities.
        super().__init__(logits=torch.log(probs.clamp_min(1e-20)))

    @property
    def mode(self) -> torch.Tensor:
        idx = torch.argmax(self.logits, dim=-1)
        oh = F.one_hot(idx, self.logits.shape[-1]).to(self.logits.dtype)
        return oh.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape=torch.Size(), temperature: float = 1.0) -> torch.Tensor:
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Use rsample() for the straight-through Gumbel-Softmax path.")


# ---------------------------------------------------------------------------
# MultiOneHotDist — product of independent OneHotDists with shared `K`
# ---------------------------------------------------------------------------

class MultiOneHotDist:
    """Independent categoricals over the last two dims.

    Two construction modes for backwards compatibility:

    1. `MultiOneHotDist(logits)` where logits is (..., S, K). The categoricals
       all share the same K, so the call collapses to a single Independent(OneHotDist).
       This is how the local RSSM and reward / value heads use it.
    2. `MultiOneHotDist(logits, shape=[K1, K2, ...])` where logits is (..., sum(shape)).
       Faithful-port signature; splits the trailing axis according to `shape` and
       wraps a OneHotDist per split. Used if you ever need heterogeneous categoricals.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        shape: Sequence[int] | None = None,
        unimix_ratio: float = 0.0,
    ):
        self.unimix_ratio = float(unimix_ratio)
        if shape is None:
            # logits: (..., S, K). One OneHotDist over the last dim.
            self._dist = torchd.Independent(OneHotDist(logits, unimix_ratio=self.unimix_ratio), 1)
            self._split = False
            self._shape = None
        else:
            splits = torch.split(logits, list(shape), dim=-1)
            self._onehots = [OneHotDist(s, unimix_ratio=self.unimix_ratio) for s in splits]
            self._dist = None
            self._split = True
            self._shape = list(shape)

    @property
    def mode(self) -> torch.Tensor:
        if self._split:
            return torch.cat([d.mode for d in self._onehots], dim=-1)
        return self._dist.base_dist.mode

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        if self._split:
            return torch.cat([d.rsample() for d in self._onehots], dim=-1)
        return self._dist.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Use rsample() for the straight-through Gumbel-Softmax path.")

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._split:
            splits = torch.split(value, self._shape, dim=-1)
            return sum(d.log_prob(s) for d, s in zip(self._onehots, splits))
        return self._dist.log_prob(value)

    def entropy(self) -> torch.Tensor:
        if self._split:
            return sum(d.entropy() for d in self._onehots)
        return self._dist.entropy()

    def kl(self, other: "MultiOneHotDist") -> torch.Tensor:
        return kl(self.logits, other.logits).sum(-1)

    @property
    def logits(self) -> torch.Tensor:
        if self._split:
            return torch.cat([d.logits for d in self._onehots], dim=-1)
        return self._dist.base_dist.logits


# ---------------------------------------------------------------------------
# TwoHot — symexp twohot regression head (DreamerV3)
# ---------------------------------------------------------------------------

class TwoHot:
    """Symexp-twohot target distribution (R2-Dreamer / DreamerV3 reward & value heads).

    `bins` is a linear lattice in symlog-space (i.e. [-20, 20] by default). Targets
    passed to `log_prob` are expected to already be in symlog-space; `mode()` returns
    the symlog-space prediction and the caller symexps to get the raw value.
    """

    def __init__(self, logits: torch.Tensor, low: float = -20.0, high: float = 20.0):
        self.logits = logits
        self.bins = logits.shape[-1]
        self.low = low
        self.high = high
        # Bins ALWAYS float32, regardless of logits dtype. Under bf16/fp16 autocast the reward
        # and critic logits are low-precision; building the bin lattice in that dtype quantised
        # the twohot soft-label interpolation weights (~3 sig digits), biasing the reward/value
        # regression by up to ~0.8 nat/sample (review N1). log_prob upcasts the logits to f32, so
        # f32 bins keep the whole interpolation consistent.
        self._bin_values = torch.linspace(low, high, self.bins, device=logits.device, dtype=torch.float32)

    def mode(self) -> torch.Tensor:
        probs = torch.softmax(self.logits.float(), dim=-1)
        return (probs * self._bin_values).sum(-1)

    def mean(self) -> torch.Tensor:
        return self.mode()

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        target = target.float().clamp(self.low, self.high)
        bins = self._bin_values
        idx = torch.searchsorted(bins.contiguous(), target.contiguous())
        idx = idx.clamp(1, self.bins - 1)
        lower = idx - 1
        upper = idx
        lo_val = bins[lower]
        hi_val = bins[upper]
        w_hi = (target - lo_val) / (hi_val - lo_val + 1e-8)
        w_lo = 1.0 - w_hi
        target_probs = torch.zeros_like(self.logits, dtype=torch.float32)
        target_probs.scatter_(-1, lower.unsqueeze(-1), w_lo.float().unsqueeze(-1))
        target_probs.scatter_add_(-1, upper.unsqueeze(-1), w_hi.float().unsqueeze(-1))
        log_probs = F.log_softmax(self.logits.float(), dim=-1)
        return (target_probs * log_probs).sum(-1)


# ---------------------------------------------------------------------------
# MSEDist / SymlogDist / Bound
# ---------------------------------------------------------------------------

class MSEDist:
    def __init__(self, pred: torch.Tensor):
        self.pred = pred

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        return -0.5 * (self.pred - target).pow(2).sum(-1)

    def mode(self) -> torch.Tensor:
        return self.pred


class SymlogDist:
    def __init__(self, pred: torch.Tensor, reduce_dims: int = 1):
        self.pred = pred
        self.reduce_dims = reduce_dims

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        diff = symlog(self.pred) - symlog(target)
        return -0.5 * diff.pow(2).sum(list(range(-self.reduce_dims, 0)))

    def mode(self) -> torch.Tensor:
        return self.pred


class Bound:
    def __init__(self, dist, bound: float):
        self.dist = dist
        self.bound = bound

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(target)

    def mode(self) -> torch.Tensor:
        return self.dist.mode().clamp(-self.bound, self.bound)


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def bounded_normal(mean: torch.Tensor, std: float = 1.0, bound: float = 1.0) -> torch.Tensor:
    return (mean + std * torch.randn_like(mean)).clamp(-bound, bound)


def binary(logits: torch.Tensor):
    return torch.distributions.Bernoulli(logits=logits)


def symexp_twohot(logits: torch.Tensor, low: float = -20.0, high: float = 20.0) -> TwoHot:
    return TwoHot(logits, low=low, high=high)


def symlog_mse(pred: torch.Tensor) -> SymlogDist:
    return SymlogDist(pred)


def mse(pred: torch.Tensor) -> MSEDist:
    return MSEDist(pred)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def kl(logits_left: torch.Tensor, logits_right: torch.Tensor) -> torch.Tensor:
    """Per-category KL between two batches of categorical logits.

    Both tensors share the same shape; reduces only the trailing K dim.
    Returns the per-(...,) KL — caller can `.sum(-1)` over remaining category dims.
    """
    log_p = F.log_softmax(logits_left, dim=-1)
    log_q = F.log_softmax(logits_right, dim=-1)
    p = torch.softmax(logits_left, dim=-1)
    return (p * (log_p - log_q)).sum(-1)
