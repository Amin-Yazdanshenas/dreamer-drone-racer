"""Unit tests for NE-Dreamer agent and NEDreamerTransformer.

No Isaac Sim required — pure PyTorch.
"""

import torch
import pytest

from dreamer.agent import DreamerConfig, DreamerV3Agent
from dreamer.ne_agent import NEDreamerV3Agent
from dreamer.networks import NEDreamerTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ne_cfg():
    return DreamerConfig(
        amp_dtype="bfloat16",
        compile=False,
        h_dim=64,
        stoch=4,
        discrete=4,
        hidden=32,
        blocks=2,
        mlp_units=32,
        cnn_depth=4,
        seq_len=8,
        batch_size=2,
        ne_hidden_dim=32,
        ne_num_layers=1,
        ne_num_heads=2,
        ne_predict_horizon=2,
        ne_use_same=False,
        ne_use_next=True,
    )


@pytest.fixture
def r2_cfg():
    return DreamerConfig(
        amp_dtype="bfloat16",
        compile=False,
        h_dim=64,
        stoch=4,
        discrete=4,
        hidden=32,
        blocks=2,
        mlp_units=32,
        cnn_depth=4,
        seq_len=8,
        batch_size=2,
    )


def _dummy_batch(cfg, device="cpu"):
    B, T = cfg.batch_size, cfg.seq_len
    H, W, C = 64, 64, cfg.image_channels
    return {
        "image": torch.randint(0, 255, (B, T, H, W, C), dtype=torch.uint8),
        "state": torch.randn(B, T, cfg.state_dim),
        "action": torch.randn(B, T, cfg.action_dim),
        "reward": torch.randn(B, T),
        "is_first": torch.zeros(B, T, dtype=torch.bool),
        "is_last": torch.zeros(B, T, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# NEDreamerTransformer shape tests
# ---------------------------------------------------------------------------

class TestNEDreamerTransformer:
    def test_next_only_output_shapes(self):
        B, T, feat_dim, out_dim, act_dim = 2, 8, 16, 12, 4
        model = NEDreamerTransformer(
            feat_dim=feat_dim, output_dim=out_dim, action_dim=act_dim,
            hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=T,
            use_same=False, use_next=True, predict_horizon=2,
        )
        feat = torch.randn(B, T, feat_dim)
        acts = torch.randn(B, T, act_dim)
        result = model(feat, acts)
        assert isinstance(result, list)
        assert len(result) == 2
        # k=0: predicts T-1 steps (h_next has T-1 positions, k=0 uses all)
        assert result[0].shape == (B, T - 1, out_dim)
        # k=1: predicts T-2 steps
        assert result[1].shape == (B, T - 2, out_dim)

    def test_same_only_output_shapes(self):
        B, T, feat_dim, out_dim, act_dim = 2, 8, 16, 12, 4
        model = NEDreamerTransformer(
            feat_dim=feat_dim, output_dim=out_dim, action_dim=act_dim,
            hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=T,
            use_same=True, use_next=False, predict_horizon=1,
        )
        feat = torch.randn(B, T, feat_dim)
        acts = torch.randn(B, T, act_dim)
        result = model(feat, acts)
        assert result.shape == (B, T, out_dim)

    def test_both_heads_output_shapes(self):
        B, T, feat_dim, out_dim, act_dim = 2, 8, 16, 12, 4
        model = NEDreamerTransformer(
            feat_dim=feat_dim, output_dim=out_dim, action_dim=act_dim,
            hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=T,
            use_same=True, use_next=True, predict_horizon=1,
        )
        feat = torch.randn(B, T, feat_dim)
        acts = torch.randn(B, T, act_dim)
        e_same, e_next_list = model(feat, acts)
        assert e_same.shape == (B, T, out_dim)
        assert e_next_list[0].shape == (B, T - 1, out_dim)

    def test_no_actions_mode(self):
        B, T, feat_dim, out_dim, act_dim = 2, 6, 16, 12, 4
        model = NEDreamerTransformer(
            feat_dim=feat_dim, output_dim=out_dim, action_dim=act_dim,
            hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=T,
            use_actions=False, use_same=False, use_next=True, predict_horizon=1,
        )
        feat = torch.randn(B, T, feat_dim)
        result = model(feat, actions=None)
        assert result[0].shape == (B, T - 1, out_dim)

    def test_gradients_flow(self):
        feat_dim, out_dim, act_dim = 16, 12, 4
        model = NEDreamerTransformer(
            feat_dim=feat_dim, output_dim=out_dim, action_dim=act_dim,
            hidden_dim=16, num_layers=1, num_heads=2, max_seq_len=8,
            use_same=False, use_next=True, predict_horizon=1,
        )
        feat = torch.randn(2, 8, feat_dim, requires_grad=True)
        acts = torch.randn(2, 8, act_dim)
        result = model(feat, acts)
        loss = result[0].sum()
        loss.backward()
        assert feat.grad is not None


# ---------------------------------------------------------------------------
# Agent integration tests
# ---------------------------------------------------------------------------

class TestNEDreamerAgent:
    def test_no_projectors_after_init(self, ne_cfg):
        agent = NEDreamerV3Agent(ne_cfg, device="cpu")
        assert not hasattr(agent, "projector_rssm")
        assert not hasattr(agent, "projector_embed")
        assert hasattr(agent, "ne_transformer")

    def test_world_model_loss_no_nan(self, ne_cfg):
        agent = NEDreamerV3Agent(ne_cfg, device="cpu")
        batch = {k: v.to("cpu") for k, v in _dummy_batch(ne_cfg).items()}
        batch["image"] = batch["image"].float() / 255.0
        B, T = ne_cfg.batch_size, ne_cfg.seq_len
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, metrics, _, _ = agent._world_model_loss(batch, B, T)
        assert not torch.isnan(loss), f"WM loss is NaN: {loss}"
        assert "wm/ne_loss" in metrics, f"Expected wm/ne_loss key, got: {list(metrics.keys())}"

    def test_repr_loss_metric_key(self, ne_cfg, r2_cfg):
        ne_agent = NEDreamerV3Agent(ne_cfg, device="cpu")
        r2_agent = DreamerV3Agent(r2_cfg, device="cpu")
        assert ne_agent._repr_loss_metric_key == "wm/ne_loss"
        assert r2_agent._repr_loss_metric_key == "wm/barlow"

    def test_wm_params_include_transformer(self, ne_cfg):
        agent = NEDreamerV3Agent(ne_cfg, device="cpu")
        wm_params = agent._get_wm_params()
        transformer_params = set(id(p) for p in agent.ne_transformer.parameters())
        wm_param_ids = set(id(p) for p in wm_params)
        assert transformer_params.issubset(wm_param_ids), "transformer params not in wm_params"

    def test_checkpoint_roundtrip(self, ne_cfg, tmp_path):
        agent = NEDreamerV3Agent(ne_cfg, device="cpu")
        path = str(tmp_path / "ne_dreamer.pt")
        agent._step = 999
        agent.save(path)

        agent2 = NEDreamerV3Agent(ne_cfg, device="cpu")
        agent2.load(path)
        assert agent2._step == 999
        # Compare transformer weights
        for (n1, p1), (n2, p2) in zip(
            agent.ne_transformer.named_parameters(),
            agent2.ne_transformer.named_parameters(),
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_act_returns_actions(self, ne_cfg):
        agent = NEDreamerV3Agent(ne_cfg, device="cpu")
        agent.eval_mode()
        agent._step = ne_cfg.warmup_steps + 1
        obs = {
            "image": torch.zeros(2, 64, 64, ne_cfg.image_channels, dtype=torch.uint8),
            "state": torch.zeros(2, ne_cfg.state_dim),
        }
        actions = agent.act(obs)
        assert actions.shape == (2, ne_cfg.action_dim)
        assert actions.abs().max() <= 1.0 + 1e-5


class TestR2DreamerUnchanged:
    """Verify R2-Dreamer agent still works correctly after refactoring."""

    def test_world_model_loss_no_nan(self, r2_cfg):
        agent = DreamerV3Agent(r2_cfg, device="cpu")
        batch = {k: v.to("cpu") for k, v in _dummy_batch(r2_cfg).items()}
        batch["image"] = batch["image"].float() / 255.0
        B, T = r2_cfg.batch_size, r2_cfg.seq_len
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, metrics, _, _ = agent._world_model_loss(batch, B, T)
        assert not torch.isnan(loss)
        assert "wm/barlow" in metrics

    def test_has_projectors(self, r2_cfg):
        agent = DreamerV3Agent(r2_cfg, device="cpu")
        assert hasattr(agent, "projector_rssm")
        assert hasattr(agent, "projector_embed")
        assert not hasattr(agent, "ne_transformer")
