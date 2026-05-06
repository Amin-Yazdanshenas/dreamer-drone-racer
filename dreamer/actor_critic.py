"""DreamerV3 Actor and Critic — train entirely on imagined latent trajectories."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import NormMLP
from .utils import lambda_return, symlog, symexp, twohot_loss, TWOHOT_BINS


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class DreamerActor(nn.Module):
    """Continuous squashed-Gaussian actor over imagined latent states.

    Input:  latent (h, z concatenated)
    Output: tanh-squashed actions in [-1, 1]^action_dim
    """

    LOG_STD_MIN = -5.0
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

    # Bootstrap value at T from target critic
    # We need value at each step t=0..T and bootstrap at T
    with torch.no_grad():
        all_latents = latents_flat  # (T*B,)
        vals = target_critic.value(all_latents).reshape(T, B)   # (T, B)
        # bootstrap: use last critic value
        bootstrap = vals[-1]   # (B,)
        vals_with_boot = torch.cat([vals, bootstrap.unsqueeze(0)], dim=0)  # (T+1, B)

    # Lambda returns
    targets = lambda_return(rewards, vals_with_boot, continues, gamma=gamma, lam=lam)  # (T, B)
    targets_flat = targets.reshape(T * B)           # keep grad for actor (flows through imagination)
    targets_sg   = targets_flat.detach()            # stop-gradient copy for critic

    # Critic loss — detach targets so critic update doesn't backprop through world model
    critic_loss = critic.loss(latents_flat.detach(), targets_sg)

    # Actor loss: gradient flows actor → actions → RSSM → reward/cont heads → lambda returns
    actor_loss = -(targets_flat).mean()
    entropy = -log_probs.reshape(T * B).mean()
    actor_loss = actor_loss - entropy_scale * entropy

    metrics = {
        "actor/loss": actor_loss.item(),
        "actor/entropy": (-entropy).item(),
        "critic/loss": critic_loss.item(),
        "imag/reward_mean": rewards.mean().item(),
        "imag/value_mean": vals.mean().item(),
        "imag/return_mean": targets.mean().item(),
    }
    return actor_loss, critic_loss, metrics
