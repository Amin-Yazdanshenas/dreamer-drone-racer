"""DreamerV3 Actor and Critic — train entirely on imagined latent trajectories."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import NormMLP
from .utils import lambda_return, symlog, symexp, twohot_loss, TWOHOT_BINS, gumbel_straight_through
from .world_model import RSSMState


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class DreamerActor(nn.Module):
    """Continuous squashed-Gaussian actor over imagined latent states.

    Input:  latent (h, z concatenated)
    Output: tanh-squashed actions in [-1, 1]^action_dim
    """

    LOG_STD_MIN = -2.0   # hard floor: prevents std < exp(-2)=0.135 and entropy collapse
    LOG_STD_MAX = 2.0

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256,
                 init_std: float = 1.0):
        super().__init__()
        self.action_dim = action_dim
        self.net = NormMLP(latent_dim, [hidden_dim] * 4, action_dim * 2, norm=True)
        self._init_std = init_std

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action_sample, log_prob, mean_action).

        action_sample and mean_action are tanh-squashed.
        """
        out = self.net(latent)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        # Reparameterised sample
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        action = torch.tanh(pre_tanh)

        # Log prob with tanh correction
        log_prob = _gaussian_log_prob(pre_tanh, mean, std) - _tanh_log_det(action)

        return action, log_prob, torch.tanh(mean)

    def act(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action + log_prob (used inside RSSM imagination)."""
        action, log_prob, _ = self.forward(latent)
        return action, log_prob

    def act_deterministic(self, latent: torch.Tensor) -> torch.Tensor:
        """Return tanh(mean) — for evaluation."""
        out = self.net(latent)
        mean, _ = out.chunk(2, dim=-1)
        return torch.tanh(mean)


def _gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor,
                       std: torch.Tensor) -> torch.Tensor:
    return (-0.5 * ((x - mean) / std) ** 2 - std.log()
            - 0.5 * torch.log(torch.tensor(2 * 3.14159265, device=x.device))).sum(-1)


def _tanh_log_det(action: torch.Tensor) -> torch.Tensor:
    return torch.log(1.0 - action ** 2 + 1e-6).sum(-1)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class DreamerCritic(nn.Module):
    """Distributional critic over imagined latent states (symlog twohot target).

    Outputs twohot logits; value estimate = symexp(twohot_mean(logits)).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256,
                 twohot_bins: int = TWOHOT_BINS):
        super().__init__()
        self.twohot_bins = twohot_bins
        self.net = NormMLP(latent_dim, [hidden_dim] * 4, twohot_bins, norm=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Returns twohot logits: (..., twohot_bins)."""
        return self.net(latent)

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        """Scalar value estimate via symexp(E[symlog(v)])."""
        from .utils import twohot_mean
        logits = self.forward(latent)
        return symexp(twohot_mean(logits, bins=self.twohot_bins))

    def loss(self, latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Symlog twohot regression loss.

        latent: (T*B, latent_dim)
        target: (T*B,) scalar lambda-return values
        """
        logits = self.forward(latent)
        return twohot_loss(logits, symlog(target), bins=self.twohot_bins)


# ---------------------------------------------------------------------------
# Return normalizer (DreamerV3 §B — prevents pessimism spiral)
# ---------------------------------------------------------------------------

class ReturnNormalizer:
    """EMA of 5th/95th percentile of lambda returns. Normalises actor targets by
    max(1, pct95 - pct5) so gradient scale stays O(1) regardless of reward magnitude.

    State is plain Python floats — not an nn.Module. Save/load via state_dict().
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self._lo: Optional[float] = None   # EMA of 5th percentile
        self._hi: Optional[float] = None   # EMA of 95th percentile

    @property
    def scale(self) -> float:
        if self._lo is None:
            return 1.0
        return max(1.0, self._hi - self._lo)

    def update(self, values: torch.Tensor) -> None:
        v = values.detach().float()
        lo = torch.quantile(v, 0.05).item()
        hi = torch.quantile(v, 0.95).item()
        if self._lo is None:
            self._lo, self._hi = lo, hi
        else:
            self._lo = self.decay * self._lo + (1.0 - self.decay) * lo
            self._hi = self.decay * self._hi + (1.0 - self.decay) * hi

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return values / self.scale

    def state_dict(self) -> dict:
        return {"lo": self._lo, "hi": self._hi, "decay": self.decay}

    def load_state_dict(self, d: dict) -> None:
        self._lo = d.get("lo")
        self._hi = d.get("hi")
        self.decay = d.get("decay", self.decay)


# ---------------------------------------------------------------------------
# Actor-critic update (operates on imagined trajectories)
# ---------------------------------------------------------------------------

def actor_critic_loss(
    actor: DreamerActor,
    critic: DreamerCritic,
    target_critic: DreamerCritic,
    world_model: "nn.Module",
    init_state: "RSSMState",  # noqa: F821
    gamma: float = 0.997,
    lam: float = 0.95,
    horizon: int = 15,
    entropy_scale: float = 3e-4,
    entropy_min: float = 1.0,
    return_normalizer: Optional[ReturnNormalizer] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute actor and critic losses over an imagined horizon.

    Returns: (actor_loss, critic_loss, metrics)
    """
    # Imagination rollout
    rssm = world_model.rssm
    imag_states, imag_actions, log_probs = rssm.imagine_sequence(init_state, actor, horizon)

    T, B = imag_actions.shape[:2]
    latents = imag_states.latent   # (T, B, latent_dim)
    latents_flat = latents.reshape(T * B, -1)

    # Predict rewards and continues along imagination
    rew_logits = world_model.reward_head(latents_flat)          # (T*B, bins)
    cont_logits = world_model.continue_head(latents_flat)       # (T*B, 1)

    from .utils import twohot_mean
    rewards = symexp(twohot_mean(rew_logits, bins=world_model.twohot_bins)).reshape(T, B)
    continues = torch.sigmoid(cont_logits).squeeze(-1).reshape(T, B)

    # Bootstrap V(s_T): one extra img_step from the last imagined state to avoid
    # the bias of using V(s_{T-1}) as a proxy for V(s_T).
    with torch.no_grad():
        vals = target_critic.value(latents_flat).reshape(T, B)   # V(s_0)..V(s_{T-1})

        last_state = RSSMState(imag_states.h[-1], imag_states.z[-1])
        boot_action, _ = actor.act(last_state.latent)
        _, h_boot = rssm.img_step(last_state, boot_action)
        z_boot_logits = rssm.prior_mlp(h_boot)
        z_boot = gumbel_straight_through(
            z_boot_logits.reshape(-1, rssm.z_cats, rssm.z_classes)
        ).reshape(-1, rssm.z_dim)
        boot_latent = torch.cat([h_boot, z_boot], dim=-1)
        bootstrap = target_critic.value(boot_latent)             # V(s_T), shape (B,)

        vals_with_boot = torch.cat([vals, bootstrap.unsqueeze(0)], dim=0)  # (T+1, B)

    # Lambda returns
    targets = lambda_return(rewards, vals_with_boot, continues, gamma=gamma, lam=lam)  # (T, B)
    targets_flat = targets.reshape(T * B)           # keep grad for actor (flows through imagination)
    targets_sg   = targets_flat.detach()            # stop-gradient copy for critic

    # Critic loss — uses symlog twohot (scale-invariant), no normalisation needed
    critic_loss = critic.loss(latents_flat.detach(), targets_sg)

    # Return normalisation (DreamerV3 §B): update stats on detached targets, then
    # divide actor targets by max(1, pct95 - pct5) to keep gradient O(1).
    if return_normalizer is not None:
        return_normalizer.update(targets_flat)
        actor_targets = return_normalizer.normalize(targets_flat)
    else:
        actor_targets = targets_flat

    # Actor loss: gradient flows actor → actions → RSSM → reward/cont heads → lambda returns
    actor_loss = -(actor_targets).mean()
    entropy = -log_probs.reshape(T * B).mean()
    # Entropy bonus + floor: subtract normal bonus, then ADD penalty when below floor.
    # When entropy < entropy_min: d(actor_loss)/d(entropy) = -2*entropy_scale → pushes up.
    # (Adding F.relu term, not subtracting — subtracting cancels gradient to zero.)
    actor_loss = actor_loss - entropy_scale * entropy
    actor_loss = actor_loss + entropy_scale * F.relu(entropy_min - entropy)

    metrics = {
        "actor/loss": actor_loss.item(),
        "actor/entropy": entropy.item(),
        "critic/loss": critic_loss.item(),
        "imag/reward_mean": rewards.mean().item(),
        "imag/value_mean": vals.mean().item(),
        "imag/return_mean": targets.mean().item(),
        "return/scale": return_normalizer.scale if return_normalizer is not None else 1.0,
    }
    return actor_loss, critic_loss, metrics
