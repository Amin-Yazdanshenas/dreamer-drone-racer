"""DreamerV3Agent — top-level interface for training and evaluation."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .actor_critic import DreamerActor, DreamerCritic, ReturnNormalizer, actor_critic_loss
from .replay_buffer import SequenceReplayBuffer
from .world_model import RSSMState, WorldModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DreamerConfig:
    # Observation
    obs_mode: str = "rgb"                  # "rgb" | "mask" | "rgb_mask"
    image_channels: int = 3               # derived from obs_mode
    state_dim: int = 10                   # ang_vel(3)+quat(4)+target_pos_b(3)
    action_dim: int = 4

    # RSSM
    h_dim: int = 2048
    z_cats: int = 32
    z_classes: int = 32
    mlp_dim: int = 768
    cnn_depth: int = 48

    # World model losses
    beta_pred: float = 3.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.5

    # Actor-critic
    horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    entropy_scale: float = 3e-2
    target_critic_ema: float = 0.98       # EMA update rate for target critic

    # Training
    seq_len: int = 32
    batch_size: int = 32
    lr_world: float = 1e-4
    lr_actor: float = 3e-5
    lr_critic: float = 3e-5
    grad_clip: float = 100.0
    warmup_steps: int = 2000
    update_every: int = 1                 # env steps between gradient updates
    n_grad_steps: int = 4

    # Replay
    replay_capacity: int = 2_000_000

    # Speed
    compile: bool = True                  # torch.compile the networks
    amp_dtype: str = "bfloat16"          # "bfloat16" (recommended) or "float16"

    # Logging / checkpointing (counted in gradient updates, not env steps)
    log_interval: int = 50        # log every N gradient updates
    save_interval: int = 200      # checkpoint every N gradient updates

    @classmethod
    def from_yaml(cls, path: str) -> "DreamerConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def __post_init__(self) -> None:
        channels = {"rgb": 3, "mask": 1, "rgb_mask": 4}
        self.image_channels = channels.get(self.obs_mode, 3)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DreamerV3Agent:
    """DreamerV3 agent: world model + actor-critic, trained from a replay buffer."""

    def __init__(self, cfg: DreamerConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self._amp_dtype = getattr(torch, cfg.amp_dtype)  # torch.bfloat16 or torch.float16
        self._amp_device = "cuda" if self.device.type == "cuda" else "cpu"

        self.world_model = WorldModel(
            in_channels=cfg.image_channels,
            action_dim=cfg.action_dim,
            h_dim=cfg.h_dim,
            z_cats=cfg.z_cats,
            z_classes=cfg.z_classes,
            mlp_dim=cfg.mlp_dim,
            cnn_depth=cfg.cnn_depth,
            state_dim=cfg.state_dim,
        ).to(self.device)

        latent_dim = cfg.h_dim + cfg.z_cats * cfg.z_classes

        self.actor = DreamerActor(latent_dim, cfg.action_dim, cfg.mlp_dim).to(self.device)
        self.critic = DreamerCritic(latent_dim, cfg.mlp_dim).to(self.device)

        # target_critic created before compile so deepcopy works cleanly
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # Optimizers reference the underlying parameters — must be created before compile
        self.opt_wm = torch.optim.Adam(self.world_model.parameters(), lr=cfg.lr_world,
                                       eps=1e-8)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor,
                                          eps=1e-8)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic,
                                           eps=1e-8)

        # torch.compile — wraps modules but underlying params stay the same objects,
        # so optimizers above continue to update the right tensors
        if cfg.compile and self.device.type == "cuda":
            self.world_model  = torch.compile(self.world_model,  mode="reduce-overhead", fullgraph=False)
            self.actor        = torch.compile(self.actor,        mode="reduce-overhead", fullgraph=False)
            self.critic       = torch.compile(self.critic,       mode="reduce-overhead", fullgraph=False)
            self.target_critic= torch.compile(self.target_critic,mode="reduce-overhead", fullgraph=False)
            print(f"[DreamerV3] torch.compile enabled (mode=reduce-overhead, dtype={cfg.amp_dtype})")

        # Per-env RSSM carry (updated after each env step)
        self._rssm_state: Optional[RSSMState] = None

        self._return_normalizer = ReturnNormalizer(decay=0.99)
        self._step: int = 0
        self._best_gates: float = 0.0

    # ------------------------------------------------------------------
    # Acting in the environment
    # ------------------------------------------------------------------

    def reset_carry(self, num_envs: int) -> None:
        self._rssm_state = self.world_model.rssm.initial_state(num_envs, self.device)

    @torch.no_grad()
    def act(self, obs_dict: Dict[str, torch.Tensor],
            deterministic: bool = False,
            prev_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return actions given current observations, update internal RSSM carry.

        During warmup (before warmup_steps), returns random actions.
        prev_actions: (N, action_dim) actions from the previous step (zeros on episode start).
        """
        if self._step < self.cfg.warmup_steps:
            N = obs_dict["state"].shape[0]
            return torch.rand(N, self.cfg.action_dim, device=self.device) * 2 - 1

        image = obs_dict["image"].to(self.device).float() / 255.0      # (N,H,W,C) → float
        image = image.permute(0, 3, 1, 2)                               # (N,C,H,W)
        state = obs_dict["state"].to(self.device)

        N = state.shape[0]
        if prev_actions is not None:
            prev_action = prev_actions.to(self.device)
        else:
            prev_action = torch.zeros(N, self.cfg.action_dim, device=self.device)

        if self._rssm_state is None or self._rssm_state.h.shape[0] != N:
            self.reset_carry(N)

        # Reset carry for envs starting a new episode
        if "is_first" in obs_dict and self._rssm_state is not None:
            first_mask = obs_dict["is_first"].to(self.device).float().unsqueeze(-1)  # (N, 1)
            self._rssm_state = RSSMState(
                self._rssm_state.h * (1 - first_mask),
                self._rssm_state.z * (1 - first_mask),
            )

        with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
            embed = self.world_model.encode(image, state)
            new_state, _, _ = self.world_model.rssm.obs_step(
                self._rssm_state, prev_action, embed
            )

        # Store carry in fp32 — prevents bfloat16 error accumulation across episodes
        self._rssm_state = RSSMState(new_state.h.float(), new_state.z.float())

        latent = self._rssm_state.latent
        if deterministic:
            return self.actor.act_deterministic(latent)
        with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
            action, _, _ = self.actor(latent)
        return action.float()

    # ------------------------------------------------------------------
    # Learning updates
    # ------------------------------------------------------------------

    def update(self, replay: SequenceReplayBuffer) -> Optional[Dict[str, float]]:
        """One gradient update step. Returns metrics dict or None if not ready."""
        batch = replay.sample(self.cfg.batch_size)
        if batch is None:
            return None

        batch = {k: v.to(self.device) for k, v in batch.items()}
        # image: (T, B, H, W, C) float [0,1] → convert to (T, B, C, H, W)
        batch["image"] = batch["image"].permute(0, 1, 4, 2, 3).contiguous()

        metrics: Dict[str, float] = {}

        # --- World model ---
        self.opt_wm.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
            wm_loss, wm_metrics, rssm_state = self.world_model.loss(
                batch,
                beta_pred=self.cfg.beta_pred,
                beta_dyn=self.cfg.beta_dyn,
                beta_rep=self.cfg.beta_rep,
            )
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.grad_clip)
        self.opt_wm.step()
        metrics.update(wm_metrics)

        # --- Actor & Critic (imagination) ---
        init = RSSMState(
            rssm_state.h.detach().reshape(-1, rssm_state.h.shape[-1]),
            rssm_state.z.detach().reshape(-1, rssm_state.z.shape[-1]),
        )

        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self._amp_device, dtype=self._amp_dtype):
            a_loss, c_loss, ac_metrics = actor_critic_loss(
                self.actor, self.critic, self.target_critic,
                self.world_model, init,
                gamma=self.cfg.gamma, lam=self.cfg.lam,
                horizon=self.cfg.horizon, entropy_scale=self.cfg.entropy_scale,
                return_normalizer=self._return_normalizer,
            )

        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.opt_actor.step()

        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.opt_critic.step()

        metrics.update(ac_metrics)

        # EMA update target critic
        _ema_update(self.target_critic, self.critic, self.cfg.target_critic_ema)

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # state_dict() delegates through compiled wrapper to underlying module
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "return_normalizer": self._return_normalizer.state_dict(),
            "step": self._step,
            "best_gates": self._best_gates,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(ckpt["world_model"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        if "opt_wm" in ckpt:
            self.opt_wm.load_state_dict(ckpt["opt_wm"])
            self.opt_actor.load_state_dict(ckpt["opt_actor"])
            self.opt_critic.load_state_dict(ckpt["opt_critic"])
        if "return_normalizer" in ckpt:
            self._return_normalizer.load_state_dict(ckpt["return_normalizer"])
        self._step = ckpt.get("step", 0)
        self._best_gates = ckpt.get("best_gates", 0.0)
        print(f"[DreamerV3] Loaded checkpoint from {path} (step={self._step})")

    def eval_mode(self) -> None:
        self.world_model.eval()
        self.actor.eval()
        self.critic.eval()

    def train_mode(self) -> None:
        self.world_model.train()
        self.actor.train()
        self.critic.train()


def _ema_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(tau).add_(sp.data, alpha=1 - tau)
