"""R2-Dreamer RSSM — ported from NM512/r2dreamer.

Uses BlockLinear-based GRU (Deter) and MultiOneHotDist posterior/prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import MultiOneHotDist, kl as kl_loss
from .networks import BlockLinear, MLP, RMSNorm


# ---------------------------------------------------------------------------
# Deterministic recurrent state (GRU-like with BlockLinear)
# ---------------------------------------------------------------------------

class Deter(nn.Module):
    """Deterministic recurrent state using a BlockLinear GRU cell.

    This replaces the standard GRUCell with a block-diagonal variant for
    efficiency, matching R2-Dreamer's implementation.
    """

    def __init__(self, inp_dim: int, h_dim: int, blocks: int = 8, act: str = "SiLU"):
        super().__init__()
        self.h_dim = h_dim
        self.blocks = blocks

        # Input projection: maps cat(z, action_embed) → h_dim
        self.inp_proj = nn.Linear(inp_dim, h_dim, bias=False)
        self.inp_norm = RMSNorm(h_dim)

        # GRU gates using BlockLinear for the recurrent path
        # Reset gate, update gate, new gate — each has in_proj + h_proj
        self.r_in = nn.Linear(h_dim, h_dim, bias=False)
        self.r_h = BlockLinear(h_dim, h_dim, blocks, bias=True)

        self.z_in = nn.Linear(h_dim, h_dim, bias=False)
        self.z_h = BlockLinear(h_dim, h_dim, blocks, bias=True)

        self.n_in = nn.Linear(h_dim, h_dim, bias=False)
        self.n_h = BlockLinear(h_dim, h_dim, blocks, bias=True)

        self.out_norm = RMSNorm(h_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x: (..., inp_dim), h: (..., h_dim) → new_h: (..., h_dim)."""
        x_proj = self.inp_norm(self.inp_proj(x))

        r = torch.sigmoid(self.r_in(x_proj) + self.r_h(h))
        z = torch.sigmoid(self.z_in(x_proj) + self.z_h(h))
        n = torch.tanh(self.n_in(x_proj) + r * self.n_h(h))

        new_h = (1 - z) * n + z * h
        return self.out_norm(new_h)


# ---------------------------------------------------------------------------
# RSSM — R2-Dreamer style with MultiOneHotDist
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    """Recurrent State-Space Model matching R2-Dreamer architecture.

    Uses:
    - BlockLinear Deter cell for h_{t+1}
    - MultiOneHotDist for posterior z_t and prior z_t
    - stoch x discrete discrete latent variables (num_cats x num_classes)
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        h_dim: int = 2048,
        stoch: int = 32,
        discrete: int = 16,
        blocks: int = 8,
        hidden: int = 256,
        obs_layers: int = 1,
        img_layers: int = 2,
        dyn_layers: int = 1,
        act: str = "SiLU",
    ):
        super().__init__()
        self.h_dim = h_dim
        self.stoch = stoch          # number of categorical variables
        self.discrete = discrete    # number of classes per categorical
        self.z_dim = stoch * discrete
        self.blocks = blocks

        # Action embedding
        self.action_embed = MLP(action_dim, h_dim, units=h_dim, layers=dyn_layers, act=act,
                                norm=True, blocks=1)

        # Deterministic recurrent cell
        # Input: cat(z_flat, action_embed) → h_dim
        self.deter = Deter(inp_dim=self.z_dim + h_dim, h_dim=h_dim, blocks=blocks, act=act)

        # Posterior MLP: (h, embed) → stoch*discrete logits
        self.post_mlp = MLP(h_dim + embed_dim, self.z_dim, units=hidden,
                            layers=obs_layers, act=act, norm=True, blocks=1)

        # Prior MLP: h → stoch*discrete logits
        self.prior_mlp = MLP(h_dim, self.z_dim, units=hidden,
                             layers=img_layers, act=act, norm=True, blocks=1)

    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (stoch_zeros, deter_zeros) each shape (B, z_dim) / (B, h_dim)."""
        stoch = torch.zeros(batch_size, self.z_dim, device=device)
        deter = torch.zeros(batch_size, self.h_dim, device=device)
        return stoch, deter

    def img_step(self, prev_stoch: torch.Tensor, prev_deter: torch.Tensor,
                 action: torch.Tensor,
                 reset_f: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagination step (prior only).

        Returns: (prior_logits, prior_stoch, new_deter)
        prior_logits: (B, stoch, discrete) — reshaped to 2D

        reset_f: optional (B, 1) float mask. When provided, the action embedding is gated to
        zero on reset positions so the bias term of action_embed cannot leak a constant signal
        into a fresh episode's RSSM state.
        """
        a_emb = self.action_embed(action)                          # (B, h_dim)
        if reset_f is not None:
            a_emb = a_emb * (1 - reset_f)
        deter_inp = torch.cat([prev_stoch, a_emb], dim=-1)        # (B, z_dim + h_dim)
        new_deter = self.deter(deter_inp, prev_deter)              # (B, h_dim)

        prior_logits_flat = self.prior_mlp(new_deter)             # (B, z_dim)
        prior_logits = prior_logits_flat.reshape(-1, self.stoch, self.discrete)
        prior_dist = MultiOneHotDist(prior_logits)
        prior_stoch = prior_dist.mode()                            # (B, z_dim) st straight-through

        return prior_logits, prior_stoch, new_deter

    def obs_step(self, prev_stoch: torch.Tensor, prev_deter: torch.Tensor,
                 action: torch.Tensor, embed: torch.Tensor,
                 reset: Optional[torch.Tensor] = None) -> Tuple[
                     torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Observation step (posterior).

        reset: (B,) bool or (B, 1) float — if True, zero out state for that env.

        Returns: (post_logits, post_stoch, prior_logits, prior_stoch, new_deter)
        All stoch tensors are (B, z_dim), deter is (B, h_dim),
        logits are (B, stoch, discrete).
        """
        reset_f: Optional[torch.Tensor] = None
        if reset is not None:
            reset_f = reset.float()
            if reset_f.dim() == 1:
                reset_f = reset_f.unsqueeze(-1)
            prev_stoch = prev_stoch * (1 - reset_f)
            prev_deter = prev_deter * (1 - reset_f)
            # Don't zero the raw action here — action_embed has a bias, so zero action leaks a
            # constant bias vector. Instead pass reset_f to img_step so the EMBEDDED action is
            # gated to zero.

        prior_logits, prior_stoch, new_deter = self.img_step(prev_stoch, prev_deter, action, reset_f=reset_f)

        post_logits_flat = self.post_mlp(torch.cat([new_deter, embed], dim=-1))  # (B, z_dim)
        post_logits = post_logits_flat.reshape(-1, self.stoch, self.discrete)
        post_dist = MultiOneHotDist(post_logits)
        post_stoch = post_dist.mode()                              # (B, z_dim) straight-through

        return post_logits, post_stoch, prior_logits, prior_stoch, new_deter

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor,
                initial: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                is_first: Optional[torch.Tensor] = None):
        """Process a (B, T) sequence of embeddings and actions.

        Args:
            embeds:   (B, T, embed_dim)
            actions:  (B, T, action_dim)
            is_first: (B, T) bool
            initial:  optional (stoch, deter) to start from — defaults to zeros

        Returns:
            post_logits:  (B, T, stoch, discrete)
            post_stoch:   (B, T, z_dim)
            prior_logits: (B, T, stoch, discrete)
            prior_stoch:  (B, T, z_dim)
            deters:       (B, T, h_dim)
        """
        B, T, _ = embeds.shape
        device = embeds.device

        if initial is not None:
            prev_stoch, prev_deter = initial
        else:
            prev_stoch, prev_deter = self.initial_state(B, device)

        # Shift actions: action[t] was taken FROM obs[t], we need prev_action for RSSM step
        zero_act = torch.zeros(B, 1, actions.shape[-1], device=device)
        prev_actions = torch.cat([zero_act, actions[:, :-1]], dim=1)  # (B, T, A)

        post_logits_list, post_stoch_list = [], []
        prior_logits_list, prior_stoch_list = [], []
        deter_list = []

        for t in range(T):
            reset_t = (is_first[:, t].float().unsqueeze(-1)
                       if is_first is not None
                       else torch.zeros(B, 1, device=device))  # (B, 1)
            (post_l, post_s, prior_l, prior_s, new_deter) = self.obs_step(
                prev_stoch, prev_deter, prev_actions[:, t], embeds[:, t], reset=reset_t
            )
            post_logits_list.append(post_l)
            post_stoch_list.append(post_s)
            prior_logits_list.append(prior_l)
            prior_stoch_list.append(prior_s)
            deter_list.append(new_deter)
            prev_stoch = post_s
            prev_deter = new_deter

        post_logits = torch.stack(post_logits_list, dim=1)   # (B, T, stoch, discrete)
        post_stoch = torch.stack(post_stoch_list, dim=1)     # (B, T, z_dim)
        prior_logits = torch.stack(prior_logits_list, dim=1) # (B, T, stoch, discrete)
        prior_stoch = torch.stack(prior_stoch_list, dim=1)   # (B, T, z_dim)
        deters = torch.stack(deter_list, dim=1)               # (B, T, h_dim)

        return post_logits, post_stoch, prior_logits, prior_stoch, deters

    def imagine(self, actor_fn, init_stoch: torch.Tensor, init_deter: torch.Tensor,
                horizon: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagination rollout using actor_fn(latent) → (action, log_prob).

        Returns: (stoch_seq, deter_seq, action_seq, log_prob_seq)
        Each: (horizon, B, *) — note first dim is time.
        """
        stoch, deter = init_stoch, init_deter
        stochs, deters, acts, lps = [], [], [], []

        for _ in range(horizon):
            latent = torch.cat([deter, stoch], dim=-1)
            action, log_prob = actor_fn(latent)
            prior_logits, prior_stoch, new_deter = self.img_step(stoch, deter, action)
            stoch = prior_stoch
            deter = new_deter
            stochs.append(stoch)
            deters.append(deter)
            acts.append(action)
            lps.append(log_prob)

        return (
            torch.stack(stochs),   # (H, B, z_dim)
            torch.stack(deters),   # (H, B, h_dim)
            torch.stack(acts),     # (H, B, A)
            torch.stack(lps),      # (H, B)
        )
