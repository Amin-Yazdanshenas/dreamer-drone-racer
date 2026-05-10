"""R2-Dreamer distributions — ported from NM512/r2dreamer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, OneHotCategorical, constraints


# ---------------------------------------------------------------------------
# symlog / symexp
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# OneHotDist — straight-through one-hot categorical
# ---------------------------------------------------------------------------

class OneHotDist(OneHotCategorical):
    """One-hot categorical with straight-through gradient."""

    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def mode(self) -> torch.Tensor:
        _mode = F.one_hot(self.base_dist._categorical.mode, self.event_shape[-1])
        return _mode.detach() + self.probs - self.probs.detach()

    def sample(self, sample_shape=torch.Size()):
        # straight-through: use one-hot but allow gradient through probs
        with torch.no_grad():
            s = super().sample(sample_shape)
        return s.detach() + self.probs - self.probs.detach()


# ---------------------------------------------------------------------------
# MultiOneHotDist — independent OneHotDist per category
# ---------------------------------------------------------------------------

class MultiOneHotDist:
    """Product of independent OneHotCategoricals.

    logits: (..., num_cats, num_classes)
    """

    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self._shape = logits.shape  # (..., num_cats, num_classes)

    @property
    def num_cats(self) -> int:
        return self._shape[-2]

    @property
    def num_classes(self) -> int:
        return self._shape[-1]

    def mode(self) -> torch.Tensor:
        """Returns (..., num_cats * num_classes) one-hot (straight-through)."""
        probs = torch.softmax(self.logits, dim=-1)
        idx = self.logits.argmax(dim=-1)  # (..., num_cats)
        one_hot = F.one_hot(idx, self.num_classes).float()  # (..., num_cats, num_classes)
        # straight-through
        one_hot = one_hot.detach() + probs - probs.detach()
        return one_hot.flatten(-2)  # (..., num_cats * num_classes)

    def sample(self) -> torch.Tensor:
        probs = torch.softmax(self.logits, dim=-1)
        idx = torch.multinomial(probs.flatten(0, -2), 1).squeeze(-1)
        idx = idx.reshape(self._shape[:-1])
        one_hot = F.one_hot(idx, self.num_classes).float()
        return (one_hot.detach() + probs - probs.detach()).flatten(-2)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., num_cats * num_classes)"""
        x = x.reshape(*x.shape[:-1], self.num_cats, self.num_classes)
        log_probs = F.log_softmax(self.logits, dim=-1)
        return (x * log_probs).sum(-1).sum(-1)

    def entropy(self) -> torch.Tensor:
        probs = torch.softmax(self.logits, dim=-1)
        log_probs = F.log_softmax(self.logits, dim=-1)
        ent = -(probs * log_probs).sum(-1)  # (..., num_cats)
        return ent.sum(-1)

    def kl(self, other: "MultiOneHotDist") -> torch.Tensor:
        """KL(self || other), shape: (...)"""
        p = torch.softmax(self.logits, dim=-1)
        log_p = F.log_softmax(self.logits, dim=-1)
        log_q = F.log_softmax(other.logits, dim=-1)
        kl_per_cat = (p * (log_p - log_q)).sum(-1)  # (..., num_cats)
        return kl_per_cat.sum(-1)


# ---------------------------------------------------------------------------
# TwoHot distribution
# ---------------------------------------------------------------------------

class TwoHot:
    """Symexp-twohot target distribution (R2-Dreamer / DreamerV3 reward head).

    bins: number of bins (default 255)
    low, high: range of the symexp-space bins
    """

    def __init__(self, logits: torch.Tensor, low: float = -20.0, high: float = 20.0):
        self.logits = logits
        self.bins = logits.shape[-1]
        self.low = low
        self.high = high
        self._bin_values = torch.linspace(low, high, self.bins, device=logits.device,
                                          dtype=logits.dtype)

    def mode(self) -> torch.Tensor:
        probs = torch.softmax(self.logits, dim=-1)
        return (probs * self._bin_values).sum(-1)

    def mean(self) -> torch.Tensor:
        return self.mode()

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        """target: (...) scalar, in symlog space already."""
        target = target.clamp(self.low, self.high)
        bins = self._bin_values
        # Find lower and upper bin indices
        idx = torch.searchsorted(bins.contiguous(), target.contiguous())
        idx = idx.clamp(1, self.bins - 1)
        lower = idx - 1
        upper = idx
        lo_val = bins[lower]
        hi_val = bins[upper]
        # Linear interpolation weights
        w_hi = (target - lo_val) / (hi_val - lo_val + 1e-8)
        w_lo = 1.0 - w_hi
        # Build soft target — use float32 to avoid dtype mismatch under AMP
        target_probs = torch.zeros_like(self.logits, dtype=torch.float32)
        target_probs.scatter_(-1, lower.unsqueeze(-1), w_lo.float().unsqueeze(-1))
        target_probs.scatter_add_(-1, upper.unsqueeze(-1), w_hi.float().unsqueeze(-1))
        log_probs = F.log_softmax(self.logits.float(), dim=-1)
        return (target_probs * log_probs).sum(-1)


# ---------------------------------------------------------------------------
# MSEDist / SymlogDist / Bound
# ---------------------------------------------------------------------------

class MSEDist:
    """Unit-variance Gaussian (MSE loss)."""

    def __init__(self, pred: torch.Tensor):
        self.pred = pred

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        return -0.5 * (self.pred - target).pow(2).sum(-1)

    def mode(self) -> torch.Tensor:
        return self.pred


class SymlogDist:
    """MSE in symlog space."""

    def __init__(self, pred: torch.Tensor, reduce_dims: int = 1):
        self.pred = pred
        self.reduce_dims = reduce_dims

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        diff = symlog(self.pred) - symlog(target)
        return -0.5 * diff.pow(2).sum(list(range(-self.reduce_dims, 0)))

    def mode(self) -> torch.Tensor:
        return self.pred


class Bound:
    """Clamps a base distribution's mode to [-bound, bound]."""

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

def bounded_normal(mean: torch.Tensor, std: float = 1.0,
                   bound: float = 1.0) -> torch.Tensor:
    """Sample from N(mean, std) and clamp to [-bound, bound]."""
    return (mean + std * torch.randn_like(mean)).clamp(-bound, bound)


def binary(logits: torch.Tensor) -> torch.distributions.Bernoulli:
    return torch.distributions.Bernoulli(logits=logits)


def symexp_twohot(logits: torch.Tensor, low: float = -20.0,
                  high: float = 20.0) -> TwoHot:
    return TwoHot(logits, low=low, high=high)


def symlog_mse(pred: torch.Tensor) -> SymlogDist:
    return SymlogDist(pred)


def mse(pred: torch.Tensor) -> MSEDist:
    return MSEDist(pred)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def kl(post: MultiOneHotDist, prior: MultiOneHotDist,
        free: float = 0.0, balance: float = 0.8) -> torch.Tensor:
    """Balanced KL: balance * KL(post_sg || prior) + (1-balance) * KL(post || prior_sg).

    free: free-nats threshold (clip KL below this value)
    """
    # Detached versions
    post_sg_logits = post.logits.detach()
    prior_sg_logits = prior.logits.detach()

    dyn_kl = MultiOneHotDist(post_sg_logits).kl(prior)   # KL(post_sg || prior)
    rep_kl = MultiOneHotDist(post.logits).kl(MultiOneHotDist(prior_sg_logits))  # KL(post || prior_sg)

    if free > 0.0:
        dyn_kl = dyn_kl.clamp(min=free)
        rep_kl = rep_kl.clamp(min=free)

    return balance * dyn_kl + (1 - balance) * rep_kl
