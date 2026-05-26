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

from .distributions import TwoHot, symexp, symlog
from .networks import DroneEncoder, MLP, MLPHead, ReturnEMA
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
    discrete: int = 32            # number of classes per categorical (upstream R2-Dreamer)
    hidden: int = 768             # MLP hidden size inside RSSM (upstream R2-Dreamer)
    blocks: int = 8               # block-diagonal blocks for Deter
    obs_layers: int = 1
    img_layers: int = 2
    dyn_layers: int = 1
    unimix_ratio: float = 0.01    # uniform-mix smoothing in OneHotDist (prevents posterior collapse)

    # World model losses — beta_dyn / beta_rep now actually wired (see _world_model_loss).
    kl_free: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1

    # Actor-critic
    imag_horizon: int = 15
    horizon: int = 333            # gamma = 1 - 1/333 ≈ 0.997
    lam: float = 0.95
    entropy_scale: float = 3e-4
    entropy_min: float = 1.0       # floor: add penalty when H[π] drops below this
    entropy_floor_weight: float = 1e-2  # weight on F.relu(entropy_min - entropy) floor penalty
    slow_target_fraction: float = 0.02

    # Barlow Twins
    barlow_lambd: float = 5e-4
    loss_scale_barlow: float = 0.05

    # Informed-Dreamer privileged-state decoder (SkyDreamer-style grounding).
    # When priv_state_dim > 0 the WM grows a decoder head MLP that reconstructs the
    # privileged ground-truth state from the latent. Forces the latent to encode
    # physical pose+velocity, not just image-consistent features. Setting dim=0
    # disables the head and reverts to pure Barlow Twins repr.
    priv_state_dim: int = 12
    loss_scale_decoder: float = 1.0

    # Loss scales
    # NOTE: loss_scale_dyn / loss_scale_rep are kept for back-compat with older YAML files
    # but are NO LONGER USED by the WM loss. The KL weighting is now done with beta_dyn /
    # beta_rep on the separately returned dyn / rep losses (faithful R2-Dreamer port).
    loss_scale_dyn: float = 1.0
    loss_scale_rep: float = 0.1
    loss_scale_rew: float = 1.0
    loss_scale_con: float = 1.0
    loss_scale_policy: float = 1.0
    loss_scale_value: float = 1.0
    loss_scale_repval: float = 0.3

    # NE-Dreamer (ignored by DreamerV3Agent / R2-Dreamer)
    ne_hidden_dim: int = 256
    ne_num_layers: int = 2
    ne_num_heads: int = 4
    ne_dropout: float = 0.0
    ne_use_actions: bool = True
    ne_use_same: bool = False
    ne_use_next: bool = True
    ne_predict_horizon: int = 1
    ne_horizon_discount: float = 0.9
    ne_loss_type: str = "barlow"
    ne_lambd: float = 5e-4
    ne_weight_same: float = 1.0
    ne_weight_next: float = 1.0

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
    # Was 2.0 → std max = exp(2) = 7.4, far wider than the tanh squashing range. Pre-tanh
    # samples landed in [-15, +15], tanh saturated to ±1, and `_tanh_log_det → log(1e-6) = -13`
    # per saturated dim inflated reported entropy to ~13 nats (true tanh-squashed entropy ≤ 2.77
    # for action_dim=4). entropy_scale * inflated_entropy then REWARDED saturation — self-
    # reinforcing flyaway. LOG_STD_MAX=0 caps std at 1.0 → pre-tanh stays in N(mean, 1), tanh
    # rarely saturates, entropy stays in physical range.
    LOG_STD_MAX = 0.0

    def __init__(self, latent_dim: int, action_dim: int, units: int = 256,
                 layers: int = 4):
        super().__init__()
        self.action_dim = action_dim
        self.net = MLP(latent_dim, action_dim * 2, units=units, layers=layers)

    # 0.5 * (1 + log(2π))  — per-dim Gaussian entropy constant
    _GAUSSIAN_ENT_CONST = 1.4189385332046727

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, entropy).

        entropy: analytic differential entropy of the *pre-tanh* Gaussian, summed
        across action dim, shape (B,). SAC-style regularizer — closed form, no
        MC variance, bounded since log_std is clamped. Replaces the old one-sample
        log-prob estimate `H ≈ -log_prob(sampled_action)` which had high variance
        AND was unbounded below when tanh saturated (log(1 - y^2) → -inf).
        """
        out = self.net(latent)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        action = torch.tanh(pre_tanh)

        # H[N(mean, std)] = sum_i (log std_i + 0.5 * (1 + log 2π))
        entropy = log_std.sum(-1) + self._GAUSSIAN_ENT_CONST * mean.shape[-1]
        return action, entropy

    def act_deterministic(self, latent: torch.Tensor) -> torch.Tensor:
        out = self.net(latent)
        mean, _ = out.chunk(2, dim=-1)
        return torch.tanh(mean)


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
            unimix_ratio=cfg.unimix_ratio,
        ).to(self.device)

        latent_dim = cfg.h_dim + cfg.stoch * cfg.discrete

        # Heads
        self.reward_head = MLPHead(latent_dim, 255, units=cfg.hidden, layers=2).to(self.device)
        self.cont_head = MLPHead(latent_dim, 1, units=cfg.hidden, layers=2).to(self.device)

        # Informed-Dreamer privileged-state decoder. Trained with symlog-MSE so it can
        # handle both small (gate-frame coords <1m) and large (world pos ~10m) targets
        # without the loss being dominated by world-pos magnitude.
        self.priv_decoder: Optional[MLPHead] = None
        if cfg.priv_state_dim > 0:
            self.priv_decoder = MLPHead(
                latent_dim, cfg.priv_state_dim, units=cfg.hidden, layers=2
            ).to(self.device)

        # Barlow Twins projectors: one for RSSM latent, one for encoder embed.
        # They project into the same shared dim so BT can align them.
        self.projector_rssm = nn.Linear(latent_dim, cfg.mlp_units, bias=False).to(self.device)
        self.projector_embed = nn.Linear(embed_dim, cfg.mlp_units, bias=False).to(self.device)

        # Actor and critics
        self.actor = Actor(latent_dim, cfg.action_dim, units=cfg.hidden, layers=4).to(self.device)
        self.critic = Critic(latent_dim, units=cfg.hidden, layers=4).to(self.device)
        self.target_critic = Critic(latent_dim, units=cfg.hidden, layers=4).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # Return EMA (for value normalisation)
        self.return_ema = ReturnEMA(alpha=cfg.slow_target_fraction).to(self.device)

        self.opt_wm = LaProp(self._get_wm_params(), lr=cfg.lr,
                             betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
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
    # World-model parameter and repr-loss hooks (override in subclasses)
    # ------------------------------------------------------------------

    def _get_wm_extra_params(self) -> list:
        """Extra WM params (projectors / transformer / priv decoder). Uses hasattr/None
        guards for subclass safety."""
        params: list = []
        if hasattr(self, "projector_rssm"):
            params += list(self.projector_rssm.parameters())
        if hasattr(self, "projector_embed"):
            params += list(self.projector_embed.parameters())
        if hasattr(self, "ne_transformer"):
            params += list(self.ne_transformer.parameters())
        if getattr(self, "priv_decoder", None) is not None:
            params += list(self.priv_decoder.parameters())
        return params

    def _get_wm_params(self) -> list:
        return (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_head.parameters())
            + list(self.cont_head.parameters())
            + self._get_wm_extra_params()
        )

    @property
    def _repr_loss_metric_key(self) -> str:
        return "wm/barlow"

    def _repr_loss(self, latent: torch.Tensor, embed_flat: torch.Tensor,
                   actions: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """R2-Dreamer: single-step RSSM↔encoder Barlow Twins. Override for NE-Dreamer."""
        z1 = self.projector_rssm(latent)
        z2 = self.projector_embed(embed_flat.detach())
        return barlow_twins_loss(z1, z2, lambd=self.cfg.barlow_lambd)

    # ------------------------------------------------------------------
    # Checkpoint helpers (override in subclasses to add extra keys)
    # ------------------------------------------------------------------

    def _build_checkpoint(self) -> dict:
        ckpt: dict = {
            "encoder": self.encoder.state_dict(),
            "rssm": self.rssm.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "cont_head": self.cont_head.state_dict(),
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
        }
        if hasattr(self, "projector_rssm"):
            ckpt["projector_rssm"] = self.projector_rssm.state_dict()
        if hasattr(self, "projector_embed"):
            ckpt["projector_embed"] = self.projector_embed.state_dict()
        if hasattr(self, "ne_transformer"):
            ckpt["ne_transformer"] = self.ne_transformer.state_dict()
        if getattr(self, "priv_decoder", None) is not None:
            ckpt["priv_decoder"] = self.priv_decoder.state_dict()
        return ckpt

    def _load_checkpoint(self, ckpt: dict) -> None:
        self.encoder.load_state_dict(ckpt["encoder"])
        self.rssm.load_state_dict(ckpt["rssm"])
        self.reward_head.load_state_dict(ckpt["reward_head"])
        self.cont_head.load_state_dict(ckpt["cont_head"])
        if "projector_rssm" in ckpt and hasattr(self, "projector_rssm"):
            self.projector_rssm.load_state_dict(ckpt["projector_rssm"])
        if "projector_embed" in ckpt and hasattr(self, "projector_embed"):
            self.projector_embed.load_state_dict(ckpt["projector_embed"])
        if "ne_transformer" in ckpt and hasattr(self, "ne_transformer"):
            self.ne_transformer.load_state_dict(ckpt["ne_transformer"])
        if "priv_decoder" in ckpt and getattr(self, "priv_decoder", None) is not None:
            self.priv_decoder.load_state_dict(ckpt["priv_decoder"])
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

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def reset_carry(self, num_envs: int) -> None:
        # stoch shape is (B, stoch, discrete) for the upstream-port RSSM.
        stoch, deter = self.rssm.initial(num_envs, self.device)
        prev_action = torch.zeros(num_envs, self.cfg.action_dim, device=self.device)
        self._carry = (stoch, deter, prev_action)

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor],
            is_first: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> torch.Tensor:
        """Return actions (N, action_dim). Updates internal RSSM carry.

        During warmup, returns random actions (deterministic flag ignored).
        deterministic=True: use tanh(mean), no Gaussian sampling — for eval/inference.
        obs["image"]: (N, H, W, C) uint8
        obs["state"]: (N, state_dim) float32
        is_first: (N,) bool — resets carry for done envs
        """
        N = obs["state"].shape[0]

        if self._carry is None or self._carry[0].shape[0] != N:
            self.reset_carry(N)

        stoch, deter, prev_action = self._carry

        # Note: we don't pre-zero carry here — obs_step's `reset` arg handles per-env
        # state and action zeroing inside the RSSM (upstream behaviour).

        # Preprocess obs
        image = obs["image"].to(self.device).float() / 255.0    # (N, H, W, C)
        state = obs["state"].to(self.device)
        obs_in = {"image": image, "state": state}

        # Always step the RSSM forward — even during warmup. Otherwise _carry stays at zeros
        # for the entire warmup phase, and the first real act() after warmup runs the RSSM
        # mid-episode with zero history (is_first=False), producing a garbage latent for ~seq_len
        # steps and polluting replay.
        with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
            embed = self.encoder(obs_in)
            reset_mask = is_first.to(self.device).bool() if is_first is not None else None
            post_stoch, new_deter, _ = self.rssm.obs_step(
                stoch, deter, prev_action, embed, reset=reset_mask,
                sample=not deterministic,
            )

        post_stoch = post_stoch.float()
        new_deter = new_deter.float()
        latent = self.rssm.get_feat(post_stoch, new_deter)

        if self._step < self.cfg.warmup_steps:
            # Random action during warmup. Carry the SAME action we return so prev_action in the
            # next step matches what was actually applied to the environment.
            action = (torch.rand(N, self.cfg.action_dim, device=self.device) * 2 - 1)
        elif deterministic:
            with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
                action = self.actor.act_deterministic(latent)
            action = action.float()
        else:
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
        if self._update_count < self.cfg.warmup:
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
        clip_grad_agc_(self._get_wm_params(), clip=self.cfg.agc, pmin=self.cfg.pmin)
        self._scaler.step(self.opt_wm)
        self._scaler.update()

        # -- Actor-critic update (imagination) --
        # post_stoch is (B, T, stoch, discrete); flatten the time/batch axes for parallel
        # imagination starts. Keep the (stoch, discrete) tail so the RSSM sees real categoricals.
        init_stoch = post_stoch.detach().reshape(B * T, self.cfg.stoch, self.cfg.discrete)
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

        # Run RSSM observe → (post_stoch (B,T,S,K), deters (B,T,D), post_logit (B,T,S,K))
        post_stoch, deters, post_logit = self.rssm.observe(
            embed, data["action"], is_first=data["is_first"].bool()
        )
        # Compute prior logits separately from deter (upstream pattern). The prior MLP
        # only depends on deter, so this matches the dyn KL definition exactly.
        _, prior_logit = self.rssm.prior(deters)

        # Flatten to (B*T, *) for heads. KL/repr/decoder use the full latent sequence;
        # reward + cont use a slice that pairs the *arrival* latent with the transition
        # that produced it (see next block).
        latent_seq = self.rssm.get_feat(post_stoch, deters)   # (B, T, latent_dim)
        latent = latent_seq.reshape(B * T, -1)

        # Reward + cont temporal alignment fix.
        #
        # Replay stores at index t: (obs_t, action_t, reward_AFTER_action_t, is_last_AFTER_action_t).
        # RSSM observe shifts actions back so latent_t = posterior(obs_t, prev_action_{t-1})
        # which represents state s_t. That means reward[t] in storage is the reward of the
        # transition LEAVING s_t — not the reward ARRIVING AT s_t.
        #
        # Imagination, on the other hand, evaluates reward_head on imagined states AFTER
        # img_step (stoch_seq[k] = s_{k+1}). For the lambda-return at imag step k to receive
        # the reward of imag step k's transition, reward_head must be trained to predict
        # "reward arriving at state X" — i.e. latent_{t+1} ↔ reward[t]. Same shift applies
        # to the continuation head (is_last[t] = "transition leaving s_t terminated" =
        # "s_{t+1} is the terminal arrival state").
        #
        # We drop the first timestep of (latent, reward/cont) pairs because there is no
        # incoming transition to attribute to s_0.
        T_eff = T - 1
        latent_rew = latent_seq[:, 1:].reshape(B * T_eff, -1)

        rew_logits = self.reward_head(latent_rew)
        rew_target = symlog(data["reward"][:, :-1].reshape(B * T_eff))
        rew_dist = TwoHot(rew_logits)
        rew_loss = -rew_dist.log_prob(rew_target).mean()

        cont_logits = self.cont_head(latent_rew).squeeze(-1)
        cont_target = (1.0 - data["is_last"].float())[:, :-1].reshape(B * T_eff)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        # KL loss — upstream split into separate dyn/rep with per-category free bits.
        # beta_dyn = KL(stop-grad post || prior) — pulls prior toward posterior.
        # beta_rep = KL(post || stop-grad prior) — pulls posterior toward prior.
        dyn_loss_vec, rep_loss_vec = self.rssm.kl_loss(post_logit, prior_logit, free=self.cfg.kl_free)
        dyn_loss = dyn_loss_vec.mean()
        rep_loss = rep_loss_vec.mean()
        kl_loss_val = self.cfg.beta_dyn * dyn_loss + self.cfg.beta_rep * rep_loss
        # Unclamped diagnostic — total KL without the free-bits floor. If << dyn_loss + rep_loss
        # over many steps, posterior has collapsed onto the prior despite the floor's gradient
        # mask. Computed on unimixed logits to match the loss objective.
        with torch.no_grad():
            from .distributions import kl as kl_per_cat
            post_u = self.rssm._unimix_logits(post_logit)
            prior_u = self.rssm._unimix_logits(prior_logit)
            kl_unclamped = kl_per_cat(post_u, prior_u).sum(-1).mean()

        embed_flat = embed.reshape(B * T, -1)
        repr_loss = self._repr_loss(latent, embed_flat, data["action"], B, T)

        # Informed-Dreamer privileged-state decoder. Symlog-MSE so world-frame magnitudes
        # (drone pos up to ~10 m) don't drown out gate-frame coords (<1 m).
        decoder_loss = torch.zeros((), device=latent.device, dtype=latent.dtype)
        if self.priv_decoder is not None and "priv_state" in data:
            priv_pred = self.priv_decoder(latent)                              # (B*T, priv_dim)
            priv_target = data["priv_state"].reshape(B * T, -1).to(latent.dtype)
            decoder_loss = F.mse_loss(symlog(priv_pred), symlog(priv_target))

        total = (
            self.cfg.loss_scale_rew * rew_loss
            + self.cfg.loss_scale_con * cont_loss
            + kl_loss_val
            + self.cfg.loss_scale_barlow * repr_loss
            + self.cfg.loss_scale_decoder * decoder_loss
        )

        # Replay buffer reward stats — confirm +10 gate rewards are entering the batch.
        rew_max = data["reward"].max().item()
        rew_min = data["reward"].min().item()
        rew_abs_mean = data["reward"].abs().mean().item()

        metrics = {
            "wm/rew_loss": rew_loss.item(),
            "wm/cont_loss": cont_loss.item(),
            "wm/dyn_loss": dyn_loss.item(),
            "wm/rep_loss": rep_loss.item(),
            "wm/kl": kl_loss_val.item(),
            "wm/kl_unclamped": kl_unclamped.item(),
            self._repr_loss_metric_key: repr_loss.item(),
            "wm/decoder_loss": decoder_loss.item() if isinstance(decoder_loss, torch.Tensor) else 0.0,
            "wm/total": total.item(),
            "replay/reward_max": rew_max,
            "replay/reward_min": rew_min,
            "replay/reward_abs_mean": rew_abs_mean,
        }
        return total, metrics, post_stoch, deters

    def _actor_critic_loss(self, init_stoch: torch.Tensor,
                           init_deter: torch.Tensor):
        """Compute actor + critic losses over imagined horizon."""
        H = self.cfg.imag_horizon
        gamma = self.cfg.gamma

        # Imagination rollout. stoch_seq shape: (H, B, S, K); deter_seq: (H, B, D).
        stoch_seq, deter_seq, act_seq, ent_seq = self.rssm.imagine(
            self.actor, init_stoch, init_deter, H
        )

        HH = stoch_seq.shape[0]
        B2 = stoch_seq.shape[1]
        latent_seq = self.rssm.get_feat(stoch_seq, deter_seq)  # (H, B, latent_dim)
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
            last_feat = self.rssm.get_feat(last_stoch, last_deter)
            last_action, _ = self.actor(last_feat)
            boot_stoch, boot_deter = self.rssm.img_step(last_stoch, last_deter, last_action)
            boot_latent = self.rssm.get_feat(boot_stoch, boot_deter)
            bootstrap = self.target_critic.value(boot_latent)            # (B2,)

        vals_with_boot = torch.cat([vals, bootstrap.unsqueeze(0)], dim=0)  # (H+1, B2)
        targets = lambda_return(rewards, vals_with_boot, continues,
                                gamma=gamma, lam=self.cfg.lam)           # (H, B2)

        # Update return EMA
        self.return_ema.update(targets)
        targets_norm = self.return_ema.normalize(targets)

        # Actor loss: maximise value + entropy bonus.
        # Use a SEPARATE floor-penalty term — the old `entropy + F.relu(min - entropy)` formulation
        # algebraically collapses to a constant `entropy_min` when entropy < min, killing the gradient
        # exactly when the policy needed it most. Now: baseline entropy bonus + extra penalty for
        # dropping below the floor, both differentiable.
        #
        # entropy is the ANALYTIC pre-tanh Gaussian entropy returned by Actor.forward —
        # closed form, bounded since log_std is clamped, no MC noise.
        entropy = ent_seq.reshape(HH * B2).mean()
        floor_pen = F.relu(torch.tensor(self.cfg.entropy_min, device=entropy.device) - entropy)
        actor_loss = -targets_norm.reshape(HH * B2).mean()
        actor_loss = actor_loss - self.cfg.entropy_scale * entropy + self.cfg.entropy_floor_weight * floor_pen

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

        # actor/entropy above is the analytic PRE-tanh Gaussian entropy — it can exceed the
        # tanh-squashed policy's theoretical max (~2.77 nats for action_dim=4). These two
        # diagnostics measure the real squashed action distribution: abs_mean shows whether
        # the actor commits to non-trivial actions, sat_frac measures tanh saturation.
        with torch.no_grad():
            action_abs_mean = act_seq.abs().mean()
            action_sat_frac = (act_seq.abs() > 0.95).float().mean()

        metrics = {
            "actor/loss": actor_loss.item(),
            "actor/entropy": entropy.item(),
            "actor/floor_pen": floor_pen.item(),
            "actor/action_abs_mean": action_abs_mean.item(),
            "actor/action_sat_frac": action_sat_frac.item(),
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
        if hasattr(self, "projector_rssm"):
            self.projector_rssm.train()
        if hasattr(self, "projector_embed"):
            self.projector_embed.train()
        if hasattr(self, "ne_transformer"):
            self.ne_transformer.train()
        if getattr(self, "priv_decoder", None) is not None:
            self.priv_decoder.train()
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.encoder.eval()
        self.rssm.eval()
        self.reward_head.eval()
        self.cont_head.eval()
        if hasattr(self, "projector_rssm"):
            self.projector_rssm.eval()
        if hasattr(self, "projector_embed"):
            self.projector_embed.eval()
        if hasattr(self, "ne_transformer"):
            self.ne_transformer.eval()
        if getattr(self, "priv_decoder", None) is not None:
            self.priv_decoder.eval()
        self.actor.eval()
        self.critic.eval()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._build_checkpoint(), path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._load_checkpoint(ckpt)
        print(f"[R2-Dreamer] Loaded checkpoint from {path} (step={self._step})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft_update(target: nn.Module, source: nn.Module, alpha: float) -> None:
    """EMA update: target = (1-alpha)*target + alpha*source."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1 - alpha).add_(sp.data, alpha=alpha)
