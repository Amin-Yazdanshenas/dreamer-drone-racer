"""NEDreamerV3Agent — NE-Dreamer agent for comparison with R2-Dreamer.

Key difference: replaces single-step Barlow Twins (R2-Dreamer) with a causal
temporal transformer that predicts future encoder embeddings from the RSSM
feature sequence (NE-Dreamer, CORL team, https://github.com/corl-team/nedreamer).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .agent import DreamerConfig, DreamerV3Agent
from .networks import NEDreamerTransformer
from .optim import LaProp


class NEDreamerV3Agent(DreamerV3Agent):
    """NE-Dreamer agent.

    Subclasses DreamerV3Agent and overrides only the representation loss.
    The causal transformer predicts embed[t+k] from the RSSM latent sequence
    up to t, grounding world-model representations in temporal dynamics rather
    than single-step embedding alignment.
    """

    def __init__(self, cfg: DreamerConfig, device: str = "cuda", obs_space=None):
        super().__init__(cfg, device, obs_space)

        # Remove R2-Dreamer projectors — NE-Dreamer doesn't use them
        del self.projector_rssm
        del self.projector_embed

        latent_dim = cfg.h_dim + cfg.stoch * cfg.discrete   # 2560
        embed_dim = self.encoder.out_dim                     # 768

        self.ne_transformer = NEDreamerTransformer(
            feat_dim=latent_dim,
            output_dim=embed_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.ne_hidden_dim,
            num_layers=cfg.ne_num_layers,
            num_heads=cfg.ne_num_heads,
            max_seq_len=cfg.seq_len,
            dropout=cfg.ne_dropout,
            use_actions=cfg.ne_use_actions,
            use_same=cfg.ne_use_same,
            use_next=cfg.ne_use_next,
            predict_horizon=cfg.ne_predict_horizon,
        ).to(self.device)

        # Rebuild opt_wm: encoder + rssm + heads + ne_transformer (no projectors)
        self.opt_wm = LaProp(
            self._get_wm_params(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )

    # ------------------------------------------------------------------
    # Repr loss override
    # ------------------------------------------------------------------

    @property
    def _repr_loss_metric_key(self) -> str:
        return "wm/ne_loss"

    def _repr_loss(
        self,
        latent: torch.Tensor,
        embed_flat: torch.Tensor,
        actions: torch.Tensor,
        B: int,
        T: int,
    ) -> torch.Tensor:
        """NE-Dreamer: causal transformer → multi-horizon embedding prediction loss."""
        feat = latent.reshape(B, T, -1)               # (B, T, latent_dim)
        acts = actions.reshape(B, T, -1)              # (B, T, action_dim)
        embed_seq = embed_flat.reshape(B, T, -1).detach()  # (B, T, embed_dim)

        result = self.ne_transformer(feat, acts)

        if self.cfg.ne_use_same and self.cfg.ne_use_next:
            e_hat_same, e_hat_next_list = result
        elif self.cfg.ne_use_same:
            e_hat_same, e_hat_next_list = result, None
        else:
            e_hat_same, e_hat_next_list = None, result

        D = embed_seq.shape[-1]
        off_diag = ~torch.eye(D, dtype=torch.bool, device=latent.device)

        def _barlow(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            N = x1.shape[0]
            x1n = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2n = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
            c = torch.mm(x1n.T, x2n) / N
            return (torch.diagonal(c) - 1).pow(2).sum() + self.cfg.ne_lambd * c[off_diag].pow(2).sum()

        def _cosine(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            return -(F.normalize(x1, dim=-1) * F.normalize(x2, dim=-1)).sum(-1).mean()

        loss_fn = _barlow if self.cfg.ne_loss_type == "barlow" else _cosine

        total = torch.tensor(0.0, device=latent.device)
        weight_sum = torch.tensor(0.0, device=latent.device)

        if e_hat_same is not None:
            x1 = e_hat_same.reshape(B * T, D)
            x2 = embed_seq.reshape(B * T, D)
            total = total + self.cfg.ne_weight_same * loss_fn(x1, x2)
            weight_sum = weight_sum + self.cfg.ne_weight_same

        if e_hat_next_list is not None:
            disc = self.cfg.ne_horizon_discount
            for k, e_hat_k in enumerate(e_hat_next_list):
                if e_hat_k is None or k + 1 >= T:
                    continue
                e_target = embed_seq[:, k + 1:]     # (B, T-k-1, D)
                min_len = min(e_hat_k.shape[1], e_target.shape[1])
                if min_len <= 0:
                    continue
                x1 = e_hat_k[:, :min_len].reshape(B * min_len, D)
                x2 = e_target[:, :min_len].reshape(B * min_len, D)
                w = self.cfg.ne_weight_next * (disc ** k)
                total = total + w * loss_fn(x1, x2)
                weight_sum = weight_sum + w

        return total / weight_sum.clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Print override (label in checkpoint log)
    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._load_checkpoint(ckpt)
        print(f"[NE-Dreamer] Loaded checkpoint from {path} (step={self._step})")
