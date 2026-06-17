"""Unit tests for the actor-critic / imagination value targets.

Guards the C1 fix (lambda-return value-target alignment) and the A1 fix (discounted
imagination weighting). Pure PyTorch — no Isaac Sim.
"""

import math

import pytest
import torch

from dreamer.agent import DreamerConfig, DreamerV3Agent, lambda_return


# ---------------------------------------------------------------------------
# lambda_return golden tests — the core math the C1 alignment feeds.
#
# Convention: lambda_return(rewards (T), values (T+1), continues (T), gamma, lam).
#   values[t]   = V(departure state of transition t)  = V(s_t)
#   values[t+1] = V(arrival state)                     = V(s_{t+1})  (the bootstrap)
# ---------------------------------------------------------------------------

def test_lambda_return_monte_carlo():
    # gamma=1, lam=1, continues=1 -> targets[t] = sum_{j>=t} r[j] + values[T].
    rewards = torch.tensor([[1.0], [2.0], [3.0]])          # (T=3, B=1)
    values = torch.tensor([[0.0], [0.0], [0.0], [0.0]])    # (T+1=4, B=1), bootstrap=0
    continues = torch.ones(3, 1)
    out = lambda_return(rewards, values, continues, gamma=1.0, lam=1.0).squeeze(-1)
    assert torch.allclose(out, torch.tensor([6.0, 5.0, 3.0]), atol=1e-5), out


def test_lambda_return_monte_carlo_with_bootstrap():
    # Same but bootstrap V(s_H)=30 -> [6+30, 5+30, 3+30]. (This is the value the C1 fix uses:
    # the LAST reached state V(s_H), not the buggy V(s_{H+1}).)
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0], [30.0]])
    continues = torch.ones(3, 1)
    out = lambda_return(rewards, values, continues, gamma=1.0, lam=1.0).squeeze(-1)
    assert torch.allclose(out, torch.tensor([36.0, 35.0, 33.0]), atol=1e-5), out


def test_lambda_return_one_step_td():
    # lam=0 -> 1-step TD: targets[t] = r[t] + gamma*cont[t]*values[t+1].
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    continues = torch.ones(3, 1)
    out = lambda_return(rewards, values, continues, gamma=0.9, lam=0.0).squeeze(-1)
    expected = torch.tensor([1.0 + 0.9 * 20, 2.0 + 0.9 * 30, 3.0 + 0.9 * 40])
    assert torch.allclose(out, expected, atol=1e-5), out


def test_lambda_return_termination_kills_bootstrap():
    # continues[1]=0 -> the bootstrap/future past transition 1 is severed.
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    continues = torch.tensor([[1.0], [0.0], [1.0]])
    out = lambda_return(rewards, values, continues, gamma=1.0, lam=1.0).squeeze(-1)
    # t=2: 3 + 1*40 = 43 ; t=1: 2 + 0*... = 2 ; t=0: 1 + 1*((1-1)*v1 + 1*last=2) = 3
    assert torch.allclose(out, torch.tensor([3.0, 2.0, 43.0]), atol=1e-5), out


# ---------------------------------------------------------------------------
# Integration: _actor_critic_loss runs end-to-end with the C1 alignment and A1
# weighting, producing finite losses. Exercises imagine() -> departure-aligned
# value array -> lambda_return -> weighted actor/critic loss.
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg():
    return DreamerConfig(
        amp_dtype="bfloat16", compile=False,
        h_dim=64, stoch=4, discrete=4, hidden=32, blocks=2,
        mlp_units=32, cnn_depth=4, seq_len=8, batch_size=2,
        imag_horizon=5,
    )


def test_actor_critic_loss_finite(tiny_cfg):
    torch.manual_seed(0)
    agent = DreamerV3Agent(tiny_cfg, device="cpu")
    B, T = tiny_cfg.batch_size, tiny_cfg.seq_len
    batch = {
        "image": torch.randint(
            0, 255, (B, T, tiny_cfg.image_size, tiny_cfg.image_size, tiny_cfg.image_channels),
            dtype=torch.uint8,
        ).float() / 255.0,
        "state": torch.randn(B, T, tiny_cfg.state_dim),
        "action": torch.randn(B, T, tiny_cfg.action_dim),
        "reward": torch.randn(B, T),
        "is_first": torch.zeros(B, T, dtype=torch.bool),
        "is_last": torch.zeros(B, T, dtype=torch.bool),
    }
    _, _, post_stoch, deters = agent._world_model_loss(batch, B, T)
    init_stoch = post_stoch.detach().reshape(B * T, tiny_cfg.stoch, tiny_cfg.discrete)
    init_deter = deters.detach().reshape(B * T, -1)

    ac_loss, metrics = agent._actor_critic_loss(init_stoch, init_deter)
    assert torch.isfinite(ac_loss), ac_loss
    for k in ("actor/loss", "critic/loss", "imag/value_mean", "imag/return_mean"):
        assert k in metrics and math.isfinite(metrics[k]), (k, metrics.get(k))
    # repval disabled by default (A2)
    assert metrics["critic/repval_loss"] == 0.0


def test_repval_disabled_by_default(tiny_cfg):
    assert tiny_cfg.loss_scale_repval == 0.0


def test_amp_dtype_default_is_bf16():
    # gradscaler-fp16-only-default-mismatch: default must match the only configs used.
    assert DreamerConfig().amp_dtype == "bfloat16"
