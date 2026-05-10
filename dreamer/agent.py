"""R2-Dreamer Agent — DreamerV3Agent interface wrapping R2-Dreamer architecture.

Replaces the old buggy custom DreamerV3 with R2-Dreamer's:
- BlockLinear RSSM
- Barlow Twins auxiliary loss instead of image decoder
- LaProp optimizer + AGC gradient clipping
- float16 AMP + GradScaler
- repval loss
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import MultiOneHotDist, TwoHot, kl as kl_loss, symexp, symlog
from .networks import DroneEncoder, MLP, MLPHead, Projector, ReturnEMA
from .optim import LaProp, clip_grad_agc_
from .replay_buffer import SequenceReplayBuffer
from .rssm import RSSM


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DreamerConfig:
    # Observation
    obs_mode: str = "rgb"
    image_channels: int = 3
    state_dim: int = 13
    action_dim: int = 4

    # CNN encoder
    cnn_depth: int = 16
    cnn_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    cnn_kernel: int = 4
    mlp_units: int = 256

    # RSSM (R2-Dreamer style)
    h_dim: int = 2048             # deter size
    stoch: int = 32               # number of categorical variables
    discrete: int = 16            # number of classes per categorical
    hidden: int = 256             # MLP hidden size inside RSSM
    blocks: int = 8               # block-diagonal blocks for Deter
    obs_layers: int = 1
    img_layers: int = 2
    dyn_layers: int = 1

    # World model losses
    kl_free: float = 1.0
    beta_dyn: float = 1.0
    beta_rep: float = 0.1

    # Actor-critic
    imag_horizon: int = 15
    horizon: int = 333            # gamma = 1 - 1/333 ≈ 0.997
    lam: float = 0.95
    entropy_scale: float = 3e-4
    slow_target_fraction: float = 0.02

    # Barlow Twins
    barlow_lambd: float = 5e-4
    loss_scale_barlow: float = 0.05

    # Loss scales
    loss_scale_dyn: float = 1.0
    loss_scale_rep: float = 0.1
    loss_scale_rew: float = 1.0
    loss_scale_con: float = 1.0
    loss_scale_policy: float = 1.0
    loss_scale_value: float = 1.0
    loss_scale_repval: float = 0.3

    # Optimizer (LaProp + AGC)
    lr: float = 4e-5
    agc: float = 0.3
    pmin: float = 1e-3
    eps: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.999
    warmup: int = 1000            # LR warmup steps (grad update steps)

    # Training
    seq_len: int = 64
    batch_size: int = 16
    warmup_steps: int = 2000     # env steps before first update
    update_every: int = 1
    n_grad_steps: int = 4

    # Replay
    replay_capacity: int = 2_000_000

    # Speed
    compile: bool = True
    amp_dtype: str = "float16"

    # Logging
    log_interval: int = 50
    save_interval: int = 200

    @classmethod
    def from_yaml(cls, path: str) -> "DreamerConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        cfg = cls()
        for k, v in d.items():
            # Skip read-only properties (e.g. gamma, image_channels)
            if isinstance(getattr(type(cfg), k, None), property):
                continue
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def __post_init__(self) -> None:
        channels = {"rgb": 3, "mask": 1, "rgb_mask": 4}
        self.image_channels = channels.get(self.obs_mode, 3)

    @property
    def gamma(self) -> float:
        return 1.0 - 1.0 / self.horizon


# ---------------------------------------------------------------------------
# Actor and Critic networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Squashed-Gaussian actor over imagined latent states."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, latent_dim: int, action_dim: int, units: int = 256,
                 layers: int = 4):
        super().__init__()
        self.action_dim = action_dim
        self.net = MLP(latent_dim, action_dim * 2, units=units, layers=layers)

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob)."""
        out = self.net(latent)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        action = torch.tanh(pre_tanh)

        log_prob = _gaussian_log_prob(pre_tanh, mean, std) - _tanh_log_det(action)
        return action, log_prob

    def act_deterministic(self, latent: torch.Tensor) -> torch.Tensor:
        out = self.net(latent)
        mean, _ = out.chunk(2, dim=-1)
        return torch.tanh(mean)


def _gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor,
                       std: torch.Tensor) -> torch.Tensor:
    log2pi = 0.9189385332046727  # 0.5 * log(2π)
    return (-0.5 * ((x - mean) / std) ** 2 - std.log() - log2pi).sum(-1)


def _tanh_log_det(action: torch.Tensor) -> torch.Tensor:
    return torch.log(1.0 - action ** 2 + 1e-6).sum(-1)


class Critic(nn.Module):
    """Distributional critic — symexp-twohot regression."""

    TWOHOT_BINS = 255
    TWOHOT_LOW = -20.0
    TWOHOT_HIGH = 20.0

    def __init__(self, latent_dim: int, units: int = 256, layers: int = 4):
        super().__init__()
        self.net = MLP(latent_dim, self.TWOHOT_BINS, units=units, layers=layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)

    def value(self, latent: torch.Tensor) -> torch.Tensor:
        logits = self.forward(latent)
        dist = TwoHot(logits, low=self.TWOHOT_LOW, high=self.TWOHOT_HIGH)
        return symexp(dist.mode())

    def loss(self, latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Symlog twohot regression loss."""
        logits = self.forward(latent)
        dist = TwoHot(logits, low=self.TWOHOT_LOW, high=self.TWOHOT_HIGH)
        return -dist.log_prob(symlog(target)).mean()


# ---------------------------------------------------------------------------
# Barlow Twins loss
# ---------------------------------------------------------------------------

def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor,
                      lambd: float = 5e-4) -> torch.Tensor:
    """Barlow Twins self-supervised loss.

    z1, z2: (B, D) projected embeddings (not normalised — we normalise here).
    """
    B, D = z1.shape
    # Normalise each feature across the batch
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)

    # Cross-correlation matrix
    c = (z1.T @ z2) / B  # (D, D)

    on_diag = (1 - c.diagonal()).pow(2).sum()
    off_diag = _off_diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag


def _off_diagonal(mat: torch.Tensor) -> torch.Tensor:
    n = mat.shape[0]
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ---------------------------------------------------------------------------
# Lambda return
# ---------------------------------------------------------------------------

def lambda_return(rewards: torch.Tensor, values: torch.Tensor,
                  continues: torch.Tensor, gamma: float,
                  lam: float) -> torch.Tensor:
    """Compute TD-lambda returns.

    rewards: (T, B), values: (T+1, B), continues: (T, B)
    Returns: (T, B)
    """
    T = rewards.shape[0]
    last_val = values[-1]
    targets = []
    for t in reversed(range(T)):
        last_val = rewards[t] + gamma * continues[t] * (
            (1 - lam) * values[t + 1] + lam * last_val
        )
        targets.append(last_val)
    targets.reverse()
    return torch.stack(targets, dim=0)


# ---------------------------------------------------------------------------
# DreamerV3Agent — main interface
# ---------------------------------------------------------------------------

class DreamerV3Agent:
    """R2-Dreamer agent compatible with the Isaac Lab training script interface.

    Training script interface:
        agent = DreamerV3Agent(cfg, obs_space, device)
        agent.reset_carry(num_envs)
        agent.train_mode()
        agent._step          (int, set externally)
        agent.act(obs, is_first) → actions (N, 4)
        agent.update(replay_buffer) → metrics dict
        agent.save(path), agent.load(path)
        agent._best_gates    (float, set externally)
    """

    def __init__(self, cfg: DreamerConfig, device: str = "cuda",
                 obs_space=None):
        self.cfg = cfg
        self.device = torch.device(device)
        self._amp_dtype = getattr(torch, cfg.amp_dtype)
        self._amp_device = "cuda" if self.device.type == "cuda" else "cpu"

        # Encoder
        image_shape = (64, 64, cfg.image_channels)  # (H, W, C)
        self.encoder = DroneEncoder(
            image_shape=image_shape,
            state_dim=cfg.state_dim,
            cnn_depth=cfg.cnn_depth,
            mults=cfg.cnn_mults,
            kernel_size=cfg.cnn_kernel,
            mlp_units=cfg.mlp_units,
        ).to(self.device)
        embed_dim = self.encoder.out_dim

        # RSSM
        self.rssm = RSSM(
            embed_dim=embed_dim,
            action_dim=cfg.action_dim,
            h_dim=cfg.h_dim,
            stoch=cfg.stoch,
            discrete=cfg.discrete,
            blocks=cfg.blocks,
            hidden=cfg.hidden,
            obs_layers=cfg.obs_layers,
            img_layers=cfg.img_layers,
            dyn_layers=cfg.dyn_layers,
        ).to(self.device)

        latent_dim = cfg.h_dim + cfg.stoch * cfg.discrete

        # Heads
        self.reward_head = MLPHead(latent_dim, 255, units=cfg.hidden, layers=2).to(self.device)
        self.cont_head = MLPHead(latent_dim, 1, units=cfg.hidden, layers=2).to(self.device)

        # Barlow Twins projector
        self.projector = Projector(
            in_dim=embed_dim, proj_dim=cfg.mlp_units, hidden_dim=cfg.mlp_units
        ).to(self.device)

        # Actor and critics
        self.actor = Actor(latent_dim, cfg.action_dim, units=cfg.hidden, layers=4).to(self.device)
        self.critic = Critic(latent_dim, units=cfg.hidden, layers=4).to(self.device)
        self.target_critic = Critic(latent_dim, units=cfg.hidden, layers=4).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # Return EMA (for value normalisation)
        self.return_ema = ReturnEMA(alpha=cfg.slow_target_fraction).to(self.device)

        # Single LaProp optimizer for all world-model params
        wm_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_head.parameters())
            + list(self.cont_head.parameters())
            + list(self.projector.parameters())
        )
        self.opt_wm = LaProp(wm_params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        self.opt_actor = LaProp(self.actor.parameters(), lr=cfg.lr,
                                betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        self.opt_critic = LaProp(self.critic.parameters(), lr=cfg.lr,
                                 betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)

        # AMP GradScaler
        _scaler_enabled = (cfg.amp_dtype == "float16" and self.device.type == "cuda")
        self._scaler = torch.amp.GradScaler("cuda", enabled=_scaler_enabled)

        # Internal carry: (stoch, deter, prev_action)
        self._carry: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        self._step: int = 0
        self._best_gates: float = 0.0
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def reset_carry(self, num_envs: int) -> None:
        stoch, deter = self.rssm.initial_state(num_envs, self.device)
        prev_action = torch.zeros(num_envs, self.cfg.action_dim, device=self.device)
        self._carry = (stoch, deter, prev_action)

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor],
            is_first: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return actions (N, action_dim). Updates internal RSSM carry.

        During warmup, returns random actions.
        obs["image"]: (N, H, W, C) uint8
        obs["state"]: (N, state_dim) float32
        is_first: (N,) bool — resets carry for done envs
        """
        N = obs["state"].shape[0]

        if self._step < self.cfg.warmup_steps:
            return torch.rand(N, self.cfg.action_dim) * 2 - 1

        if self._carry is None or self._carry[0].shape[0] != N:
            self.reset_carry(N)

        stoch, deter, prev_action = self._carry

        # Zero carry for reset envs
        if is_first is not None:
            rst = is_first.to(self.device).float().unsqueeze(-1)  # (N, 1)
            stoch = stoch * (1 - rst)
            deter = deter * (1 - rst)
            prev_action = prev_action * (1 - rst)

        # Preprocess obs
        image = obs["image"].to(self.device).float() / 255.0    # (N, H, W, C)
        state = obs["state"].to(self.device)
        obs_in = {"image": image, "state": state}

        with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
            embed = self.encoder(obs_in)
            reset_mask = is_first.to(self.device) if is_first is not None else None
            (_, post_stoch, _, _, new_deter) = self.rssm.obs_step(
                stoch, deter, prev_action, embed, reset=reset_mask
            )

        post_stoch = post_stoch.float()
        new_deter = new_deter.float()
        latent = torch.cat([new_deter, post_stoch], dim=-1)

        with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
            action, _ = self.actor(latent)
        action = action.float()

        self._carry = (post_stoch, new_deter, action)
        return action.cpu()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, replay_buffer: SequenceReplayBuffer) -> Optional[Dict[str, float]]:
        """One gradient update step. Returns metrics or None if not ready."""
        batch = replay_buffer.sample(self.cfg.batch_size)
        if batch is None:
            return None

        batch = {k: v.to(self.device) for k, v in batch.items()}
        # image: (B, T, H, W, C) uint8 → float [0,1] and rearrange to (B, T, C, H, W) -- done in preprocess
        metrics = self._update_step(batch)
        self._update_count += 1

        # LR warmup: scale LR linearly for first `warmup` grad steps
        if self._update_count <= self.cfg.warmup:
            lr_scale = self._update_count / self.cfg.warmup
            for opt in (self.opt_wm, self.opt_actor, self.opt_critic):
                for pg in opt.param_groups:
                    pg["lr"] = self.cfg.lr * lr_scale

        # Soft-update target critic
        _soft_update(self.target_critic, self.critic, self.cfg.slow_target_fraction)

        return metrics

    def _preprocess(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert image uint8→float, keep (B, T, ...) format."""
        out = dict(batch)
        # image: (B, T, H, W, C) uint8 → float [0,1] → (B, T, H, W, C)
        out["image"] = batch["image"].float() / 255.0
        return out

    def _update_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        data = self._preprocess(batch)
        B, T = data["reward"].shape[:2]

        # -- World model update --
        self.opt_wm.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
            wm_loss, wm_metrics, post_stoch, deters = self._world_model_loss(data, B, T)

        self._scaler.scale(wm_loss).backward()
        self._scaler.unscale_(self.opt_wm)
        clip_grad_agc_(
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_head.parameters())
            + list(self.cont_head.parameters())
            + list(self.projector.parameters()),
            clip=self.cfg.agc, pmin=self.cfg.pmin,
        )
        self._scaler.step(self.opt_wm)
        self._scaler.update()

        # -- Actor-critic update (imagination) --
        init_stoch = post_stoch.detach().reshape(B * T, -1)
        init_deter = deters.detach().reshape(B * T, -1)

        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
            ac_loss, ac_metrics = self._actor_critic_loss(init_stoch, init_deter)

        self._scaler.scale(ac_loss).backward()
        self._scaler.unscale_(self.opt_actor)
        self._scaler.unscale_(self.opt_critic)
        clip_grad_agc_(self.actor.parameters(), clip=self.cfg.agc, pmin=self.cfg.pmin)
        clip_grad_agc_(self.critic.parameters(), clip=self.cfg.agc, pmin=self.cfg.pmin)
        self._scaler.step(self.opt_actor)
        self._scaler.step(self.opt_critic)
        self._scaler.update()

        return {**wm_metrics, **ac_metrics}

    def _world_model_loss(self, data: Dict[str, torch.Tensor], B: int, T: int):
        """Compute world model loss. Returns (loss, metrics, post_stoch, deters)."""
        # Encode all timesteps
        image = data["image"]           # (B, T, H, W, C) float
        state = data["state"]           # (B, T, D)

        # Build obs dict for encoder
        obs_enc = {
            "image": image.reshape(B * T, *image.shape[2:]),
            "state": state.reshape(B * T, -1),
        }
        embed = self.encoder(obs_enc).reshape(B, T, -1)   # (B, T, embed_dim)

        # Run RSSM observe
        post_logits, post_stoch, prior_logits, prior_stoch, deters = self.rssm.observe(
            embed, data["action"], is_first=data["is_first"].bool()
        )

        # Flatten to (B*T, *) for heads
        latent = torch.cat([deters, post_stoch], dim=-1).reshape(B * T, -1)

        # Reward prediction (symexp-twohot)
        rew_logits = self.reward_head(latent)             # (B*T, 255)
        rew_target = symlog(data["reward"].reshape(B * T))
        rew_dist = TwoHot(rew_logits)
        rew_loss = -rew_dist.log_prob(rew_target).mean()

        # Continue prediction
        cont_logits = self.cont_head(latent).squeeze(-1)  # (B*T,)
        cont_target = (1.0 - data["is_last"].float()).reshape(B * T)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        # KL loss (balanced, with free nats)
        post_dist = MultiOneHotDist(post_logits.reshape(B * T, self.cfg.stoch, self.cfg.discrete))
        prior_dist = MultiOneHotDist(prior_logits.reshape(B * T, self.cfg.stoch, self.cfg.discrete))
        kl = kl_loss(post_dist, prior_dist, free=self.cfg.kl_free, balance=0.8)
        kl_mean = kl.mean()

        # Barlow Twins auxiliary loss (two augmented views of same embedding)
        # Use consecutive pairs (t, t+1) in sequence as two "views"
        embed_flat = embed.reshape(B * T, -1)
        if T >= 2:
            z1 = self.projector(embed_flat[:B * (T - 1)])
            z2 = self.projector(embed_flat[B:])
            bt_loss = barlow_twins_loss(z1, z2, lambd=self.cfg.barlow_lambd)
        else:
            bt_loss = embed_flat.new_zeros(1).squeeze()

        total = (
            self.cfg.loss_scale_rew * rew_loss
            + self.cfg.loss_scale_con * cont_loss
            + self.cfg.loss_scale_dyn * kl_mean
            + self.cfg.loss_scale_barlow * bt_loss
        )

        metrics = {
            "wm/rew_loss": rew_loss.item(),
            "wm/cont_loss": cont_loss.item(),
            "wm/kl": kl_mean.item(),
            "wm/barlow": bt_loss.item(),
            "wm/total": total.item(),
        }
        return total, metrics, post_stoch, deters

    def _actor_critic_loss(self, init_stoch: torch.Tensor,
                           init_deter: torch.Tensor):
        """Compute actor + critic losses over imagined horizon."""
        H = self.cfg.imag_horizon
        gamma = self.cfg.gamma

        # Imagination rollout
        stoch_seq, deter_seq, act_seq, lp_seq = self.rssm.imagine(
            self.actor, init_stoch, init_deter, H
        )  # each: (H, B, *)

        latent_seq = torch.cat([deter_seq, stoch_seq], dim=-1)  # (H, B, latent_dim)
        HH, B2, _ = latent_seq.shape
        latent_flat = latent_seq.reshape(HH * B2, -1)

        # Predicted rewards and continues along imagination
        rew_logits = self.reward_head(latent_flat)
        cont_logits = self.cont_head(latent_flat).squeeze(-1)

        rew_dist = TwoHot(rew_logits)
        rewards = symexp(rew_dist.mode()).reshape(HH, B2)
        continues = torch.sigmoid(cont_logits).reshape(HH, B2)

        # Bootstrap V(s_H) using target critic — no grad
        with torch.no_grad():
            vals = self.target_critic.value(latent_flat).reshape(HH, B2)
            # One extra step beyond horizon
            last_stoch = stoch_seq[-1]
            last_deter = deter_seq[-1]
            last_action, _ = self.actor(torch.cat([last_deter, last_stoch], dim=-1))
            _, boot_stoch, boot_deter = self.rssm.img_step(last_stoch, last_deter, last_action)
            boot_latent = torch.cat([boot_deter, boot_stoch], dim=-1)
            bootstrap = self.target_critic.value(boot_latent)            # (B2,)

        vals_with_boot = torch.cat([vals, bootstrap.unsqueeze(0)], dim=0)  # (H+1, B2)
        targets = lambda_return(rewards, vals_with_boot, continues,
                                gamma=gamma, lam=self.cfg.lam)           # (H, B2)

        # Update return EMA
        self.return_ema.update(targets)
        targets_norm = self.return_ema.normalize(targets)

        # Actor loss
        log_probs = lp_seq.reshape(HH * B2)
        entropy = -log_probs.mean()
        actor_loss = -targets_norm.reshape(HH * B2).mean()
        actor_loss = actor_loss - self.cfg.entropy_scale * entropy

        # Critic loss (twohot regression on stopped targets)
        targets_sg = targets.detach()
        crit_loss = self.critic.loss(latent_flat.detach(), targets_sg.reshape(HH * B2))

        # Repval loss: critic on post-hoc reprojected latents (online critic self-distillation)
        repval_loss = self.critic.loss(latent_flat.detach(),
                                       vals.detach().reshape(HH * B2))

        total_ac = (
            self.cfg.loss_scale_policy * actor_loss
            + self.cfg.loss_scale_value * crit_loss
            + self.cfg.loss_scale_repval * repval_loss
        )

        metrics = {
            "actor/loss": actor_loss.item(),
            "actor/entropy": entropy.item(),
            "critic/loss": crit_loss.item(),
            "critic/repval_loss": repval_loss.item(),
            "imag/reward_mean": rewards.mean().item(),
            "imag/value_mean": vals.mean().item(),
            "imag/return_mean": targets.mean().item(),
            "return/scale": self.return_ema.scale.item(),
        }
        return total_ac, metrics

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        self.encoder.train()
        self.rssm.train()
        self.reward_head.train()
        self.cont_head.train()
        self.projector.train()
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.encoder.eval()
        self.rssm.eval()
        self.reward_head.eval()
        self.cont_head.eval()
        self.projector.eval()
        self.actor.eval()
        self.critic.eval()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "rssm": self.rssm.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "cont_head": self.cont_head.state_dict(),
            "projector": self.projector.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "return_ema": self.return_ema.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "step": self._step,
            "best_gates": self._best_gates,
            "update_count": self._update_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.rssm.load_state_dict(ckpt["rssm"])
        self.reward_head.load_state_dict(ckpt["reward_head"])
        self.cont_head.load_state_dict(ckpt["cont_head"])
        if "projector" in ckpt:
            self.projector.load_state_dict(ckpt["projector"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        if "return_ema" in ckpt:
            self.return_ema.load_state_dict(ckpt["return_ema"])
        if "opt_wm" in ckpt:
            self.opt_wm.load_state_dict(ckpt["opt_wm"])
            self.opt_actor.load_state_dict(ckpt["opt_actor"])
            self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self._step = ckpt.get("step", 0)
        self._best_gates = ckpt.get("best_gates", 0.0)
        self._update_count = ckpt.get("update_count", 0)
        print(f"[R2-Dreamer] Loaded checkpoint from {path} (step={self._step})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor,
                      lambd: float = 5e-4) -> torch.Tensor:
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
    c = (z1.T @ z2) / B
    on_diag = (1 - c.diagonal()).pow(2).sum()
    off_diag = _off_diagonal(c).pow(2).sum()
    return on_diag + lambd * off_diag


def _off_diagonal(mat: torch.Tensor) -> torch.Tensor:
    n = mat.shape[0]
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _soft_update(target: nn.Module, source: nn.Module, alpha: float) -> None:
    """EMA update: target = (1-alpha)*target + alpha*source."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1 - alpha).add_(sp.data, alpha=alpha)
