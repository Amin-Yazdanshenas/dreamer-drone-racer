"""DreamerV3 World Model: RSSM + encoder/decoder + heads (PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import ImageDecoder, ImageEncoder, NormMLP, StateEncoder
from .utils import gumbel_straight_through, symlog, twohot_loss


# ---------------------------------------------------------------------------
# RSSM state container
# ---------------------------------------------------------------------------

@dataclass
class RSSMState:
    h: torch.Tensor   # (B, h_dim)  deterministic recurrent state
    z: torch.Tensor   # (B, z_cats * z_classes)  stochastic latent (one-hot flat)

    def detach(self) -> "RSSMState":
        return RSSMState(self.h.detach(), self.z.detach())

    @property
    def latent(self) -> torch.Tensor:
        return torch.cat([self.h, self.z], dim=-1)


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    """Recurrent State-Space Model (DreamerV3 §2).

    Deterministic path:  h_{t+1} = GRU(h_t,  cat(z_t, embed_a_t))
    Posterior:           z_t     ~ Categorical( MLP(h_t, embed_obs_t) )
    Prior:               z_t     ~ Categorical( MLP(h_t) )

    z is represented as one-hot over (z_cats × z_classes) using
    Gumbel-softmax with straight-through gradients.
    """

    def __init__(self, embed_dim: int, action_dim: int,
                 h_dim: int = 512, z_cats: int = 32, z_classes: int = 32,
                 mlp_dim: int = 256):
        super().__init__()
        self.h_dim = h_dim
        self.z_cats = z_cats
        self.z_classes = z_classes
        self.z_dim = z_cats * z_classes

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, mlp_dim), nn.LayerNorm(mlp_dim), nn.SiLU(),
        )
        # GRU: input = cat(z_flat, action_embed)
        self.gru = nn.GRUCell(self.z_dim + mlp_dim, h_dim)

        # Posterior: MLP(h, embed_obs) → logits (z_cats, z_classes)
        self.posterior_mlp = NormMLP(h_dim + embed_dim, [mlp_dim], self.z_dim, norm=True)

        # Prior: MLP(h) → logits
        self.prior_mlp = NormMLP(h_dim, [mlp_dim], self.z_dim, norm=True)

    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return RSSMState(h, z)

    def img_step(self, prev: RSSMState, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One imagination step: returns (prior_logits, new_h).

        prior_logits: (B, z_cats*z_classes)
        """
        a_emb = self.action_embed(action)
        gru_input = torch.cat([prev.z, a_emb], dim=-1)
        h_new = self.gru(gru_input, prev.h)
        prior_logits = self.prior_mlp(h_new)
        return prior_logits, h_new

    def obs_step(self, prev: RSSMState, action: torch.Tensor,
                 embed: torch.Tensor) -> Tuple[RSSMState, torch.Tensor, torch.Tensor]:
        """One observation step: returns (new_state, post_logits, prior_logits).

        Uses Gumbel straight-through for z sample.
        """
        prior_logits, h_new = self.img_step(prev, action)

        post_logits = self.posterior_mlp(torch.cat([h_new, embed], dim=-1))
        post_logits_2d = post_logits.reshape(-1, self.z_cats, self.z_classes)
        z_new = gumbel_straight_through(post_logits_2d).reshape(-1, self.z_dim)

        return RSSMState(h_new, z_new), post_logits, prior_logits

    def observe_sequence(self, embeds: torch.Tensor, actions: torch.Tensor,
                         is_first: torch.Tensor) -> Tuple[RSSMState, torch.Tensor, torch.Tensor]:
        """Process a full (T, B) sequence of embeddings and actions.

        Args:
            embeds:   (T, B, embed_dim)
            actions:  (T, B, action_dim)  — action taken FROM step t (replay buffer convention)
            is_first: (T, B) bool — resets h,z to zero on episode start

        Returns:
            states:       RSSMState with h,z of shape (T, B, *)
            post_logits:  (T, B, z_dim)
            prior_logits: (T, B, z_dim)
        """
        T, B, _ = embeds.shape
        device = embeds.device
        state = self.initial_state(B, device)

        # RSSM recurrence: h_t = GRU(h_{t-1}, cat(z_{t-1}, embed(a_{t-1})))
        # Replay buffer stores actions[t] = action taken FROM obs[t].
        # We need a_{t-1}: shift by 1 and prepend zeros for t=0.
        zero_act = torch.zeros(1, B, actions.shape[-1], device=device)
        prev_actions = torch.cat([zero_act, actions[:-1]], dim=0)  # (T, B, A)

        hs, zs, post_list, prior_list = [], [], [], []
        for t in range(T):
            # Reset state on episode boundary
            reset = is_first[t].float().unsqueeze(-1)  # (B, 1)
            h = state.h * (1 - reset)
            z = state.z * (1 - reset)
            # Also zero the previous action at episode start (no preceding action exists)
            prev_act_t = prev_actions[t] * (1 - reset)
            state = RSSMState(h, z)

            state, post_l, prior_l = self.obs_step(state, prev_act_t, embeds[t])
            hs.append(state.h)
            zs.append(state.z)
            post_list.append(post_l)
            prior_list.append(prior_l)

        h_seq = torch.stack(hs, dim=0)      # (T, B, h_dim)
        z_seq = torch.stack(zs, dim=0)      # (T, B, z_dim)
        post_seq = torch.stack(post_list, dim=0)
        prior_seq = torch.stack(prior_list, dim=0)
        return RSSMState(h_seq, z_seq), post_seq, prior_seq

    def imagine_sequence(self, init_state: RSSMState, actor: "nn.Module",
                         horizon: int) -> Tuple["RSSMState", torch.Tensor, torch.Tensor]:
        """Imagination rollout: T steps using actor, returns from init_state.

        Returns:
            states:  RSSMState (horizon, B, *)
            actions: (horizon, B, action_dim)
            log_probs: (horizon, B) actor log-probability
        """
        B = init_state.h.shape[0] if init_state.h.dim() == 2 else init_state.h.shape[1]
        if init_state.h.dim() == 3:
            # flatten T*B
            h = init_state.h.reshape(-1, self.h_dim)
            z = init_state.z.reshape(-1, self.z_dim)
            state = RSSMState(h, z)
        else:
            state = init_state

        hs, zs, acts, lps = [], [], [], []
        for _ in range(horizon):
            action, log_prob = actor.act(state.latent)
            prior_logits, h_new = self.img_step(state, action)
            prior_logits_2d = prior_logits.reshape(-1, self.z_cats, self.z_classes)
            z_new = gumbel_straight_through(prior_logits_2d).reshape(-1, self.z_dim)
            state = RSSMState(h_new, z_new)
            hs.append(state.h)
            zs.append(state.z)
            acts.append(action)
            lps.append(log_prob)

        return (
            RSSMState(torch.stack(hs), torch.stack(zs)),
            torch.stack(acts),
            torch.stack(lps),
        )


# ---------------------------------------------------------------------------
# Dense prediction heads (reward, continue)
# ---------------------------------------------------------------------------

class DenseHead(nn.Module):
    """MLP prediction head with configurable output size."""

    def __init__(self, latent_dim: int, out_dim: int, hidden_dim: int = 256, layers: int = 2):
        super().__init__()
        self.net = NormMLP(latent_dim, [hidden_dim] * layers, out_dim, norm=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


# ---------------------------------------------------------------------------
# World Model (assembles all components)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """DreamerV3 World Model.

    Encodes observations, runs RSSM, decodes and predicts reward/continue.
    """

    def __init__(self,
                 in_channels: int,          # image channels: 1, 3, or 4
                 action_dim: int,
                 h_dim: int = 512,
                 z_cats: int = 32,
                 z_classes: int = 32,
                 mlp_dim: int = 256,
                 cnn_depth: int = 32,
                 state_dim: int = 10,
                 twohot_bins: int = 255,
                 mask_pos_weight: float = 9.0):
        super().__init__()
        self.twohot_bins = twohot_bins
        self.in_channels = in_channels
        # pos_weight for mask channels in BCE: gate pixels are ~10% of image → weight ~9
        self.mask_pos_weight = mask_pos_weight

        img_embed_dim = 512
        state_embed_dim = 64
        embed_dim = img_embed_dim + state_embed_dim

        self.image_encoder = ImageEncoder(in_channels, embed_dim=img_embed_dim, depth=cnn_depth)
        self.state_encoder = StateEncoder(state_dim, state_embed_dim)

        self.rssm = RSSM(embed_dim, action_dim, h_dim, z_cats, z_classes, mlp_dim)

        latent_dim = h_dim + z_cats * z_classes
        self.image_decoder = ImageDecoder(latent_dim, out_channels=in_channels, depth=cnn_depth)
        self.reward_head = DenseHead(latent_dim, twohot_bins, hidden_dim=mlp_dim)
        self.continue_head = DenseHead(latent_dim, 1, hidden_dim=mlp_dim)

    @property
    def latent_dim(self) -> int:
        return self.rssm.h_dim + self.rssm.z_dim

    def encode(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Encode one timestep of observations.

        image: (B, C, H, W) float32 in [0,1]
        state: (B, state_dim) float32
        Returns: (B, embed_dim)
        """
        img_emb = self.image_encoder(image)
        st_emb = self.state_encoder(state)
        return torch.cat([img_emb, st_emb], dim=-1)

    def loss(self, batch: dict, beta_pred: float = 1.0,
             beta_dyn: float = 1.0, beta_rep: float = 0.1,
             free_bits: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """Compute total world-model loss over a (T, B) batch.

        batch keys: image (T,B,C,H,W), state (T,B,D), action (T,B,A),
                    reward (T,B), is_first (T,B), is_last (T,B)

        Returns: (total_loss, metrics_dict)
        """
        T, B = batch["reward"].shape
        device = batch["reward"].device

        # Encode all timesteps
        img = batch["image"].reshape(T * B, *batch["image"].shape[2:])   # (T*B, C, H, W)
        st = batch["state"].reshape(T * B, -1)
        embed = self.encode(img, st).reshape(T, B, -1)                    # (T, B, embed_dim)

        # Prepend zero action (for first step in sequence)
        actions = batch["action"]    # (T, B, A)

        # Run RSSM
        rssm_state, post_logits, prior_logits = self.rssm.observe_sequence(
            embed, actions, batch["is_first"]
        )

        # Flatten for head computation
        latent = rssm_state.latent.reshape(T * B, -1)   # (T*B, latent_dim)

        # --- Image reconstruction loss ---
        recon_logits = self.image_decoder(latent)   # (T*B, C, H, W)
        target_img = batch["image"].reshape(T * B, *batch["image"].shape[2:])
        if self.in_channels in (1, 4):
            # Mask channel is sparse (~5-10% gate pixels). Weight positive class to prevent
            # the trivial all-black solution from dominating the BCE loss.
            pw = torch.ones_like(recon_logits)
            pw[:, -1:, :, :] = self.mask_pos_weight  # last channel = mask
            img_loss = F.binary_cross_entropy_with_logits(recon_logits, target_img,
                                                          pos_weight=pw, reduction="mean")
        else:
            img_loss = F.binary_cross_entropy_with_logits(recon_logits, target_img, reduction="mean")

        # --- Reward prediction loss (symlog twohot) ---
        rew_logits = self.reward_head(latent)                              # (T*B, bins)
        rew_target = symlog(batch["reward"].reshape(T * B))
        rew_loss = twohot_loss(rew_logits, rew_target, bins=self.twohot_bins)

        # --- Continue prediction loss ---
        cont_logits = self.continue_head(latent).squeeze(-1)              # (T*B,)
        cont_target = (1.0 - batch["is_last"].float()).reshape(T * B)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        # --- KL divergence: dyn + rep (free bits = 1.0 nats) ---
        post_2d = post_logits.reshape(T * B, self.rssm.z_cats, self.rssm.z_classes)
        prior_2d = prior_logits.reshape(T * B, self.rssm.z_cats, self.rssm.z_classes)

        kl_dyn = _kl_categorical(post_2d.detach(), prior_2d, free_bits=free_bits).mean()
        kl_rep = _kl_categorical(post_2d, prior_2d.detach(), free_bits=free_bits).mean()

        total = (beta_pred * (img_loss + rew_loss + cont_loss)
                 + beta_dyn * kl_dyn
                 + beta_rep * kl_rep)

        metrics = {
            "wm/image_loss": img_loss.item(),
            "wm/reward_loss": rew_loss.item(),
            "wm/cont_loss": cont_loss.item(),
            "wm/kl_dyn": kl_dyn.item(),
            "wm/kl_rep": kl_rep.item(),
            "wm/total": total.item(),
        }
        return total, metrics, rssm_state


def _kl_categorical(post_logits: torch.Tensor, prior_logits: torch.Tensor,
                    free_bits: float = 0.0) -> torch.Tensor:
    """KL(posterior || prior) — no free-bits floor.

    free_bits=0.0: full gradient always flows to posterior, forcing z to encode
    observations from the very first update. The KL penalty is kept small via
    beta_dyn=0.1 to prevent instability.

    logits: (B, z_cats, z_classes) → returns (B,)
    """
    log_post = F.log_softmax(post_logits, dim=-1)
    log_prior = F.log_softmax(prior_logits, dim=-1)
    post_probs = log_post.exp()
    kl_per_cat = (post_probs * (log_post - log_prior)).sum(dim=-1)  # (B, z_cats)
    if free_bits > 0.0:
        kl_per_cat = kl_per_cat.clamp(min=free_bits)
    return kl_per_cat.sum(dim=-1)                                    # (B,)
