"""RSSM — faithful port of NM512/r2dreamer (commit referenced May 2026).

Differences from previous local rewrite:
- Deter cell uses three separate Linear+RMSNorm+SiLU projections for (deter, stoch, action)
  rather than concatenating raw inputs before a single projection.
- Action L1-normalisation: action / clip(|action|, min=1.0).detach() before projection.
- Block-broadcast trick: projected inputs are expanded over `blocks` groups and concatenated
  with per-block deterministic state slices, then processed by BlockLinear hidden layers
  and a BlockLinear 3*deter gate projection.
- Gate order is (reset, cand, update) — update has the offset `sigmoid(update - 1)` from
  the upstream code (forget-biased: keeps more of prev_deter early in training).
- Posterior latent uses straight-through Gumbel-Softmax via the upstream OneHotDist.

obs_step returns (stoch, deter, post_logit). img_step returns (stoch, deter).
Compute prior logits separately via `prior(deter)` when you need KL.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd

from .distributions import OneHotDist, kl as kl_per_cat
from .networks import BlockLinear


def _rpad(t: torch.Tensor, n: int) -> torch.Tensor:
    """Right-pad a tensor with `n` trailing singleton dims."""
    return t.reshape(*t.shape, *([1] * n))


class Deter(nn.Module):
    """Upstream R2-Dreamer block-GRU deterministic transition cell.

    Inputs: stoch (B, S, K), deter (B, D), action (B, A).
    """

    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        action_dim: int,
        hidden: int,
        blocks: int,
        dyn_layers: int = 1,
        act: str = "SiLU",
    ):
        super().__init__()
        self.deter_dim = int(deter_dim)
        self.blocks = int(blocks)
        self.dyn_layers = int(dyn_layers)
        act_cls = getattr(nn, act)

        # Three independent input projections — keeps each modality on its own statistics
        # before the block-broadcast concat (vs. the previous code's single Linear on a
        # raw concat which mixed scales).
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            act_cls(),
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            act_cls(),
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(action_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            act_cls(),
        )

        # Hidden block-MLP stack — `in_ch` accounts for the block-broadcast concat of
        # 3*hidden input projections + deter/blocks per group.
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter_dim // self.blocks) * self.blocks
        for i in range(self.dyn_layers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter_dim, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter_dim, eps=1e-4, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act_cls())
            in_ch = deter_dim

        # Final block-GRU projection producing 3 gates' worth of activations.
        self._dyn_gru = BlockLinear(in_ch, 3 * deter_dim, self.blocks)

    def _flat2group(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.blocks, -1)

    def _group2flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch: torch.Tensor, deter: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = action.shape[0]

        # Flatten the categorical stochastic state (B, S, K) → (B, S*K)
        stoch_flat = stoch.reshape(B, -1)

        # Bound action magnitude in [-1, 1] without breaking gradient near the boundary.
        # The detach() means the clip itself isn't differentiated — only the raw scaling.
        action = action / torch.clip(torch.abs(action), min=1.0).detach()

        # (B, hidden)
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch_flat)
        x2 = self._dyn_in2(action)

        # (B, 3*hidden)
        x = torch.cat([x0, x1, x2], dim=-1)

        # Block-broadcast: (B, blocks, 3*hidden)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)

        # Per-block concat with deter slices: (B, blocks, deter/blocks + 3*hidden)
        # → flatten back to (B, deter + 3*hidden*blocks)
        x = self._group2flat(torch.cat([self._flat2group(deter), x], dim=-1))

        # (B, deter)
        x = self._dyn_hid(x)
        # (B, 3*deter)
        x = self._dyn_gru(x)

        # Block-wise split into (reset, cand, update). Each gate is (B, deter).
        gates = torch.chunk(self._flat2group(x), 3, dim=-1)
        reset, cand, update = (self._group2flat(g) for g in gates)

        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        # Upstream offset: biases the update gate toward forgetting the candidate at init,
        # so deter persists more of its prior value early in training.
        update = torch.sigmoid(update - 1.0)

        return update * cand + (1.0 - update) * deter


class RSSM(nn.Module):
    """Recurrent State-Space Model — faithful R2-Dreamer port.

    Public state shapes:
        stoch: (B, stoch, discrete) — categorical latent, one-hot along last dim.
        deter: (B, h_dim)            — recurrent hidden state.

    `get_feat(stoch, deter)` returns the flattened concat used by reward / cont / actor / critic.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        h_dim: int = 2048,
        stoch: int = 32,
        discrete: int = 32,
        blocks: int = 8,
        hidden: int = 768,
        obs_layers: int = 1,
        img_layers: int = 2,
        dyn_layers: int = 1,
        unimix_ratio: float = 0.01,
        act: str = "SiLU",
    ):
        super().__init__()
        self.h_dim = int(h_dim)
        self.stoch = int(stoch)
        self.discrete = int(discrete)
        self.blocks = int(blocks)
        self.flat_stoch = self.stoch * self.discrete
        self.feat_size = self.flat_stoch + self.h_dim
        self.unimix_ratio = float(unimix_ratio)
        act_cls = getattr(nn, act)

        # Deterministic transition
        self._deter_net = Deter(
            deter_dim=self.h_dim,
            stoch_dim=self.flat_stoch,
            action_dim=action_dim,
            hidden=hidden,
            blocks=self.blocks,
            dyn_layers=dyn_layers,
            act=act,
        )

        # Posterior MLP: (deter, embed) → (stoch * discrete) logits → reshape to (stoch, discrete)
        self._obs_net = nn.Sequential()
        inp = self.h_dim + embed_dim
        for i in range(obs_layers):
            self._obs_net.add_module(f"obs_lin_{i}", nn.Linear(inp, hidden, bias=True))
            self._obs_net.add_module(f"obs_norm_{i}", nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32))
            self._obs_net.add_module(f"obs_act_{i}", act_cls())
            inp = hidden
        self._obs_net.add_module("obs_logit", nn.Linear(inp, self.flat_stoch, bias=True))

        # Prior MLP: deter → (stoch * discrete) logits
        self._img_net = nn.Sequential()
        inp = self.h_dim
        for i in range(img_layers):
            self._img_net.add_module(f"img_lin_{i}", nn.Linear(inp, hidden, bias=True))
            self._img_net.add_module(f"img_norm_{i}", nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32))
            self._img_net.add_module(f"img_act_{i}", act_cls())
            inp = hidden
        self._img_net.add_module("img_logit", nn.Linear(inp, self.flat_stoch))

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        stoch = torch.zeros(batch_size, self.stoch, self.discrete, dtype=torch.float32, device=device)
        deter = torch.zeros(batch_size, self.h_dim, dtype=torch.float32, device=device)
        return stoch, deter

    # Back-compat alias for older call-sites that expect a (B, stoch*discrete) flat stoch.
    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        stoch, deter = self.initial(batch_size, device)
        return stoch.reshape(batch_size, -1), deter

    def get_feat(self, stoch: torch.Tensor, deter: torch.Tensor) -> torch.Tensor:
        if stoch.dim() == deter.dim() + 1:
            stoch = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        return torch.cat([deter, stoch], dim=-1)

    def get_dist(self, logit: torch.Tensor):
        """Independent OneHotDist over the last `stoch` categoricals."""
        return torchd.Independent(OneHotDist(logit, unimix_ratio=self.unimix_ratio), 1)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def obs_step(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        reset: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Posterior step: deter transition then posterior conditional on embed.

        Returns (stoch (B, S, K), deter (B, D), post_logit (B, S, K)).
        """
        # Promote a flat stoch (B, S*K) to (B, S, K) for back-compat with old call-sites.
        if stoch.dim() == 2:
            stoch = stoch.reshape(stoch.shape[0], self.stoch, self.discrete)

        if reset is not None:
            r_stoch = _rpad(reset, stoch.dim() - reset.dim()).to(stoch.dtype)
            r_deter = _rpad(reset, deter.dim() - reset.dim()).to(deter.dtype)
            r_act = _rpad(reset, prev_action.dim() - reset.dim()).to(prev_action.dtype)
            stoch = torch.where(r_stoch.bool(), torch.zeros_like(stoch), stoch)
            deter = torch.where(r_deter.bool(), torch.zeros_like(deter), deter)
            prev_action = torch.where(r_act.bool(), torch.zeros_like(prev_action), prev_action)

        deter = self._deter_net(stoch, deter, prev_action)
        x = torch.cat([deter, embed], dim=-1)
        logit_flat = self._obs_net(x)
        logit = logit_flat.reshape(*logit_flat.shape[:-1], self.stoch, self.discrete)
        dist = self.get_dist(logit)
        new_stoch = dist.rsample() if sample else self._mode_one_hot(logit)
        return new_stoch, deter, logit

    def img_step(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        prev_action: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prior / imagination step (no observation).

        Returns (stoch (B, S, K), deter (B, D)).
        """
        if stoch.dim() == 2:
            stoch = stoch.reshape(stoch.shape[0], self.stoch, self.discrete)
        deter = self._deter_net(stoch, deter, prev_action)
        new_stoch, _ = self.prior(deter, sample=sample)
        return new_stoch, deter

    def prior(self, deter: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prior distribution over stoch given deter. Returns (stoch, logit)."""
        logit_flat = self._img_net(deter)
        logit = logit_flat.reshape(*logit_flat.shape[:-1], self.stoch, self.discrete)
        dist = self.get_dist(logit)
        new_stoch = dist.rsample() if sample else self._mode_one_hot(logit)
        return new_stoch, logit

    @staticmethod
    def _mode_one_hot(logit: torch.Tensor) -> torch.Tensor:
        idx = logit.argmax(dim=-1)
        oh = F.one_hot(idx, logit.shape[-1]).to(logit.dtype)
        # Straight-through so the deterministic mode still backprops to the logits.
        return oh.detach() + logit - logit.detach()

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def observe(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        is_first: Optional[torch.Tensor] = None,
        initial: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the posterior rollout over a sequence.

        Args:
            embeds:    (B, T, embed_dim)
            actions:   (B, T, A)   — actions taken FROM each obs; prev_action used internally
            is_first:  (B, T) bool — True on the first step of an episode
            initial:   optional (stoch, deter) override

        Returns (post_stochs (B, T, S, K), deters (B, T, D), post_logits (B, T, S, K)).
        """
        B, T, _ = embeds.shape
        device = embeds.device

        if initial is None:
            stoch, deter = self.initial(B, device)
        else:
            stoch, deter = initial
            if stoch.dim() == 2:
                stoch = stoch.reshape(B, self.stoch, self.discrete)

        # Shift actions so that obs[t] is paired with prev_action a_{t-1}.
        zero_act = torch.zeros(B, 1, actions.shape[-1], device=device, dtype=actions.dtype)
        prev_actions = torch.cat([zero_act, actions[:, :-1]], dim=1)

        post_stochs, deters, post_logits = [], [], []
        for t in range(T):
            reset_t = is_first[:, t] if is_first is not None else None
            stoch, deter, logit = self.obs_step(stoch, deter, prev_actions[:, t], embeds[:, t], reset=reset_t)
            post_stochs.append(stoch)
            deters.append(deter)
            post_logits.append(logit)

        return (
            torch.stack(post_stochs, dim=1),  # (B, T, S, K)
            torch.stack(deters, dim=1),       # (B, T, D)
            torch.stack(post_logits, dim=1),  # (B, T, S, K)
        )

    def imagine_with_action(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Open-loop rollout under a known action sequence. Returns (stochs, deters)."""
        L = actions.shape[1]
        stochs, deters = [], []
        for t in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, t])
            stochs.append(stoch)
            deters.append(deter)
        return torch.stack(stochs, dim=1), torch.stack(deters, dim=1)

    def imagine(
        self,
        actor_fn,
        init_stoch: torch.Tensor,
        init_deter: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Actor-driven imagination rollout.

        actor_fn takes the flattened feature `get_feat(stoch, deter)` and returns
        (action, entropy). Returns (stochs, deters, actions, entropies) each (H, B, *).
        """
        if init_stoch.dim() == 2:
            init_stoch = init_stoch.reshape(init_stoch.shape[0], self.stoch, self.discrete)

        stoch, deter = init_stoch, init_deter
        stochs, deters, acts, ents = [], [], [], []
        for _ in range(horizon):
            feat = self.get_feat(stoch, deter)
            action, entropy = actor_fn(feat)
            stoch, deter = self.img_step(stoch, deter, action)
            stochs.append(stoch)
            deters.append(deter)
            acts.append(action)
            ents.append(entropy)
        return torch.stack(stochs), torch.stack(deters), torch.stack(acts), torch.stack(ents)

    # ------------------------------------------------------------------
    # KL
    # ------------------------------------------------------------------

    def _unimix_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply unimix smoothing to raw logits.

        Mirrors OneHotDist.__init__ so the KL objective and the sampling distribution agree
        on the same effective categorical. Without this the network samples from the unimixed
        distribution but the loss pushes raw logits — they target slightly different optima
        and the unimix floor on each category is silently ignored by the KL term.
        """
        probs = torch.softmax(logits.float(), dim=-1)
        if self.unimix_ratio > 0.0:
            uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
            probs = (1.0 - self.unimix_ratio) * probs + self.unimix_ratio * uniform
        return torch.log(probs.clamp_min(1e-20))

    def kl_loss(
        self,
        post_logit: torch.Tensor,
        prior_logit: torch.Tensor,
        free: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (dyn_loss, rep_loss) summed over categoricals, clipped at `free` per-category.

        dyn_loss: KL(stop-grad(post) || prior) — trains the prior to match the posterior.
        rep_loss: KL(post || stop-grad(prior)) — trains the posterior to stay close to the prior.

        Both KLs are computed on unimixed logits so they match the categorical distribution
        the network actually samples from. Upstream NM512/r2dreamer uses raw logits here —
        we diverge intentionally for sampling/loss consistency, which matches Hafner's
        original DreamerV3 reference convention.

        The clip(min=free) acts as a per-category free-bits floor: when a category's KL is
        already below `free`, the gradient on that category is zeroed (the floor isn't
        backpropagated through torch.clip), preventing the loss from forcing the posterior
        to collapse onto the prior.
        """
        post_u = self._unimix_logits(post_logit)
        prior_u = self._unimix_logits(prior_logit)

        # Per-category KL between independent categoricals.
        # logits shape: (..., stoch, discrete) — kl_per_cat sums over -1, returning (..., stoch).
        rep = kl_per_cat(post_u, prior_u.detach())
        dyn = kl_per_cat(post_u.detach(), prior_u)
        rep = torch.clip(rep, min=free)
        dyn = torch.clip(dyn, min=free)
        return dyn.sum(-1), rep.sum(-1)
