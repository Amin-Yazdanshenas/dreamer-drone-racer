"""In-sim sanity test for the CTBR rate controller.

Steps Isaac Lab with a HAND-CRAFTED constant action — no agent, no policy.
Logs achieved body rates + drone altitude, prints tracking error vs commanded.

Usage:
    conda activate isaacsim
    python3 scripts/test_ctbr.py --phase hover                     # hover
    python3 scripts/test_ctbr.py --phase roll  --rate_norm 0.5     # +5 rad/s roll
    python3 scripts/test_ctbr.py --phase pitch --rate_norm 0.5     # +5 rad/s pitch
    python3 scripts/test_ctbr.py --phase yaw   --rate_norm 0.5     # +1 rad/s yaw
    python3 scripts/test_ctbr.py --phase climb                     # full collective
    python3 scripts/test_ctbr.py --phase descend                   # min collective
    python3 scripts/test_ctbr.py --phase all                       # run all phases sequentially

Add --headless to skip viewport; default opens the GUI so you can watch.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="In-sim CTBR rate-controller test")
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-Dreamer-RGB-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--phase",
    type=str,
    default="all",
    choices=["hover", "roll", "pitch", "yaw", "climb", "descend", "all"],
)
parser.add_argument(
    "--rate_norm",
    type=float,
    default=0.5,
    help="Body-rate action component in [-1, 1]. 0.5 = half of max_rate (5 rad/s for rp, 1 rad/s for yaw).",
)
parser.add_argument(
    "--steps",
    type=int,
    default=200,
    help="Number of RL steps per phase (at 100 Hz RL rate = 2 s per phase).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required by the Dreamer task scene

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Post-launch imports ---
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]   # scripts/test_ctbr.py → repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tasks  # noqa: F401


# ---------------------------------------------------------------------------
# CTBR action interpretation (matches CTBRAction.process_actions):
#   action = [c, ωx_norm, ωy_norm, ωz_norm] each in [-1, 1]
#   c=0  → hover thrust   (after the asymmetric-linear fix in actions.py)
#   c=+1 → max thrust     c=-1 → 0 thrust
#   ωx_norm = 1 → max_roll_rate (10 rad/s)
#   ωy_norm = 1 → max_pitch_rate (10 rad/s)
#   ωz_norm = 1 → max_yaw_rate (2 rad/s)
# ---------------------------------------------------------------------------

MAX_RATES = (10.0, 10.0, 2.0)  # roll, pitch, yaw — keep in sync with ctbr_gains.yaml


def make_action(num_envs: int, c: float, ωx_n: float, ωy_n: float, ωz_n: float) -> torch.Tensor:
    """Build a constant action tensor of shape (num_envs, 4)."""
    a = torch.tensor([[c, ωx_n, ωy_n, ωz_n]], dtype=torch.float32)
    return a.repeat(num_envs, 1)


PHASES = {
    "hover":   {"c": 0.0, "rates": (0.0, 0.0, 0.0), "desc": "c=0  (hover_thrust)"},
    "roll":    {"c": 0.0, "rates": (None, 0.0, 0.0), "desc": "ωx_des set"},
    "pitch":   {"c": 0.0, "rates": (0.0, None, 0.0), "desc": "ωy_des set"},
    "yaw":     {"c": 0.0, "rates": (0.0, 0.0, None), "desc": "ωz_des set"},
    "climb":   {"c": 1.0, "rates": (0.0, 0.0, 0.0), "desc": "c=+1 (max thrust)"},
    "descend": {"c": -1.0, "rates": (0.0, 0.0, 0.0), "desc": "c=-1 (0 thrust)"},
}


def resolve_rates(phase: str, rate_norm: float) -> tuple[float, float, float]:
    """Replace None entries in PHASES[phase] with rate_norm."""
    spec = PHASES[phase]["rates"]
    return tuple(rate_norm if r is None else r for r in spec)


def commanded_rates_rad_s(rate_norms: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(n * m for n, m in zip(rate_norms, MAX_RATES))


# ---------------------------------------------------------------------------
# Test loop
# ---------------------------------------------------------------------------

def run_phase(env, phase: str, rate_norm: float, n_steps: int) -> None:
    rates_n = resolve_rates(phase, rate_norm)
    cmd_rates = commanded_rates_rad_s(rates_n)
    spec = PHASES[phase]
    print()
    print(f"================ phase: {phase} ================")
    print(f"  action = [c={spec['c']:.2f}, ωx={rates_n[0]:.2f}, ωy={rates_n[1]:.2f}, ωz={rates_n[2]:.2f}]")
    print(f"  cmd rates (rad/s) = ({cmd_rates[0]:+.2f}, {cmd_rates[1]:+.2f}, {cmd_rates[2]:+.2f})")
    print(f"  meaning: {spec['desc']}")

    env.reset()
    action = make_action(env.unwrapped.num_envs, spec["c"], *rates_n).to(env.unwrapped.device)

    # Buffers for first env
    rates_log: list[tuple[float, float, float]] = []
    z_log: list[float] = []

    for _ in range(n_steps):
        env.step(action)
        robot = env.unwrapped.scene["robot"]
        ωb = robot.data.root_ang_vel_b[0].cpu().tolist()  # body-frame rates for env 0
        z = robot.data.root_pos_w[0, 2].item()
        rates_log.append(tuple(ωb))
        z_log.append(z)

    # Summary
    last_25 = rates_log[-max(1, n_steps // 4):]  # last quarter = settled region
    mean_ωx = sum(r[0] for r in last_25) / len(last_25)
    mean_ωy = sum(r[1] for r in last_25) / len(last_25)
    mean_ωz = sum(r[2] for r in last_25) / len(last_25)
    z_start = z_log[0]
    z_end = z_log[-1]

    print()
    print(f"  settled body rates (rad/s) = ({mean_ωx:+.3f}, {mean_ωy:+.3f}, {mean_ωz:+.3f})")
    print(f"  tracking error      (rad/s) = ({mean_ωx - cmd_rates[0]:+.3f}, "
          f"{mean_ωy - cmd_rates[1]:+.3f}, {mean_ωz - cmd_rates[2]:+.3f})")
    print(f"  altitude z: start={z_start:+.3f}m  end={z_end:+.3f}m  Δ={z_end - z_start:+.3f}m")

    # Pass/fail heuristics
    rate_tol = 0.5  # rad/s settled tracking error tolerance
    err_ωx = abs(mean_ωx - cmd_rates[0])
    err_ωy = abs(mean_ωy - cmd_rates[1])
    err_ωz = abs(mean_ωz - cmd_rates[2])
    rates_ok = err_ωx < rate_tol and err_ωy < rate_tol and err_ωz < rate_tol

    if phase == "hover":
        # Tighter altitude criterion when hovering
        alt_ok = abs(z_end - z_start) < 0.3  # ≤30 cm drift over 2 s
        print(f"  HOVER altitude drift {'OK' if alt_ok else 'FAIL'} (|Δz| < 0.3 m)")
        print(f"  RATES {'OK' if rates_ok else 'FAIL'}")
    elif phase == "climb":
        climbed = (z_end - z_start) > 0.5
        print(f"  CLIMB {'OK' if climbed else 'FAIL'} (Δz > +0.5 m)")
    elif phase == "descend":
        dropped = (z_end - z_start) < -0.5
        print(f"  DESCEND {'OK' if dropped else 'FAIL'} (Δz < -0.5 m)")
    else:
        print(f"  RATE-TRACKING {'OK' if rates_ok else 'FAIL'} (|err| < {rate_tol} rad/s)")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    # Disable curriculum/random spawn quirks for clean tests — spawn at gate 0 with no noise.
    env_cfg.commands.target.spawn_lerp_alpha = 0.0
    env_cfg.commands.target.spawn_forward_velocity = 0.0
    env_cfg.commands.target.randomise_start = False

    env = gym.make(args_cli.task, cfg=env_cfg)

    phases = list(PHASES.keys()) if args_cli.phase == "all" else [args_cli.phase]
    for phase in phases:
        run_phase(env, phase, args_cli.rate_norm, args_cli.steps)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
