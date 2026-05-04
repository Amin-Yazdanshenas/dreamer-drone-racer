"""DreamerV3 math utilities — pure PyTorch, no Isaac Sim dependency."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Symlog / symexp transforms (DreamerV3 §3.1)
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform: sign(x) * log(|x| + 1)."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (x.abs().exp() - 1.0)


# ---------------------------------------------------------------------------
# Two-hot encoding for symlog targets (DreamerV3 §3.2)
# ---------------------------------------------------------------------------

TWOHOT_BINS: int = 255
TWOHOT_LOW: float = -20.0
TWOHOT_HIGH: float = 20.0


def twohot_encode(x: torch.Tensor, bins: int = TWOHOT_BINS,
                  low: float = TWOHOT_LOW, high: float = TWOHOT_HIGH) -> torch.Tensor:
    """Encode scalar x into a two-hot vector over `bins` uniformly spaced buckets.

    Two adjacent bins receive fractional weight summing to 1, proportional to
    how close x falls between their centres.  Shape: (...) → (..., bins).
    """
    edges = torch.linspace(low, high, bins, device=x.device, dtype=x.dtype)
    x_clamp = x.clamp(low, high)
    # lower bin index
    idx = torch.bucketize(x_clamp, edges, right=True) - 1
    idx = idx.clamp(0, bins - 2)
    # fractional weight for upper bin
    lower = edges[idx]
    upper = edges[idx + 1]
    weight_upper = (x_clamp - lower) / (upper - lower + 1e-8)
    weight_lower = 1.0 - weight_upper
    target = torch.zeros(*x.shape, bins, device=x.device, dtype=x.dtype)
    target.scatter_(-1, idx.unsqueeze(-1), weight_lower.unsqueeze(-1))
    target.scatter_(-1, (idx + 1).unsqueeze(-1), weight_upper.unsqueeze(-1))
    return target


def twohot_loss(logits: torch.Tensor, targets: torch.Tensor,
                bins: int = TWOHOT_BINS,
                low: float = TWOHOT_LOW, high: float = TWOHOT_HIGH) -> torch.Tensor:
    """Cross-entropy loss between predicted distribution and two-hot target.

    logits: (..., bins)   targets: (...)  scalar values
    Returns scalar mean loss.
    """
    target_dist = twohot_encode(targets, bins=bins, low=low, high=high)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()


def twohot_mean(logits: torch.Tensor,
                bins: int = TWOHOT_BINS,
                low: float = TWOHOT_LOW, high: float = TWOHOT_HIGH) -> torch.Tensor:
    """Expected value of a two-hot distribution over bin centres.

    logits: (..., bins) → (...,) scalar
    """
    edges = torch.linspace(low, high, bins, device=logits.device, dtype=logits.dtype)
    probs = F.softmax(logits, dim=-1)
    return (probs * edges).sum(dim=-1)


# ---------------------------------------------------------------------------
# Lambda-return (DreamerV3 §3.3)
# ---------------------------------------------------------------------------

def lambda_return(rewards: torch.Tensor, values: torch.Tensor,
                  continues: torch.Tensor, gamma: float = 0.997,
                  lam: float = 0.95) -> torch.Tensor:
    """Compute TD-lambda returns for imagined trajectories.

    Args:
        rewards:   (T, B) — reward at each imagined step
        values:    (T+1, B) — value estimates; last element is bootstrap
        continues: (T, B) — 1 - done (continuation probability)
        gamma:     discount factor
        lam:       lambda for TD-lambda
    Returns:
        targets: (T, B) — lambda-return targets
    """
    T = rewards.shape[0]
    targets = torch.zeros_like(rewards)
    last = values[T]  # bootstrap value at step T
    for t in reversed(range(T)):
        td = rewards[t] + gamma * continues[t] * values[t + 1]
        last = td + gamma * lam * continues[t] * (last - values[t + 1])
        targets[t] = last
    return targets


# ---------------------------------------------------------------------------
# Straight-through gradient for categorical samples
# ---------------------------------------------------------------------------

def straight_through_sample(probs: torch.Tensor) -> torch.Tensor:
    """Sample one-hot from categorical probs with straight-through gradients.

    probs: (..., num_classes) — must sum to 1
    Returns: (..., num_classes) one-hot, gradients flow through probs.
    """
    idx = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).squeeze(-1)
    idx = idx.reshape(*probs.shape[:-1])
    sample = F.one_hot(idx, probs.shape[-1]).to(probs.dtype)
    return (sample - probs).detach() + probs


def gumbel_straight_through(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Gumbel-softmax with straight-through estimator.

    logits: (..., num_classes)
    Returns: (..., num_classes) approximately one-hot with gradients.
    """
    gumbels = -torch.empty_like(logits).exponential_().log()
    y_soft = F.softmax((logits + gumbels) / temperature, dim=-1)
    idx = y_soft.argmax(dim=-1)
    y_hard = F.one_hot(idx, logits.shape[-1]).to(logits.dtype)
    return (y_hard - y_soft).detach() + y_soft
