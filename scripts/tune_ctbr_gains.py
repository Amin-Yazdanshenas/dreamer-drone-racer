"""Auto-tune PD rate controller gains for CTBR action space.

No Isaac Sim required. Simulates closed-loop step response per axis using
known drone inertia, optimises (kp, kd) to minimise ITAE with torque saturation.

Usage:
    python3 scripts/tune_ctbr_gains.py [--plot]

Outputs:
    dreamer/configs/ctbr_gains.yaml
"""

import argparse
import math
import os

import numpy as np
import yaml
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Drone physical parameters (match ControlActionCfg + paper)
# ---------------------------------------------------------------------------
MASS       = 0.6076          # kg
J          = np.array([0.002410, 0.001800, 0.003759])  # kg m² (roll, pitch, yaw)
GRAVITY    = 9.81            # m/s²

ARM_LENGTH  = 0.035          # m (ControlActionCfg)
THRUST_COEF = 2.25e-7        # N/(rad/s)²
DRAG_COEF   = 1.5e-9         # N·m/(rad/s)²
OMEGA_MAX   = 5145.0         # rad/s

# Derived max torques (X-frame, effective arm = L/√2)
_arm_eff = ARM_LENGTH / math.sqrt(2.0)
TAU_MAX_ROLL  = 2.0 * _arm_eff * THRUST_COEF * OMEGA_MAX**2   # ≈ 0.295 N·m
TAU_MAX_PITCH = TAU_MAX_ROLL
TAU_MAX_YAW   = 4.0 * DRAG_COEF  * OMEGA_MAX**2               # ≈ 0.000159 N·m
TAU_MAX       = np.array([TAU_MAX_ROLL, TAU_MAX_PITCH, TAU_MAX_YAW])

MAX_THRUST    = 4.0 * THRUST_COEF * OMEGA_MAX**2               # ≈ 23.84 N
HOVER_THRUST  = MASS * GRAVITY                                  # ≈ 5.96 N

# Rate limits: roll/pitch generous, yaw capped (weak authority)
MAX_ROLL_RATE  = 10.0   # rad/s
MAX_PITCH_RATE = 10.0   # rad/s
MAX_YAW_RATE   = 2.0    # rad/s  (weak yaw torque → keep small)

# Tune duration and target step size per axis
T_SIM   = 1.0     # seconds of simulated step response
DT_SIM  = 1/400   # physics timestep

STEP_RATE = np.array([5.0, 5.0, 1.0])  # rad/s step size per axis


# ---------------------------------------------------------------------------
# Closed-loop PD simulation
# ---------------------------------------------------------------------------

def simulate_step(kp: float, kd: float, J_axis: float, tau_max: float,
                  omega_des: float, t_end: float = T_SIM, dt: float = DT_SIM):
    """Simulate closed-loop PD rate control for one axis.

    Dynamics: J * dω/dt = τ_cmd  (linearised, ignoring cross-coupling)
    τ_cmd = clip(kp*(ω_des - ω) - kd*dω/dt, -τ_max, τ_max)

    Returns (t_array, omega_array).
    """
    times = [0.0]
    omegas = [0.0]

    omega = 0.0
    t = 0.0
    while t < t_end:
        err = omega_des - omega
        # derivative of error ≈ -dω/dt (ω_des constant)
        # Use previous step's dω/dt approximation via explicit Euler
        tau = kp * err
        tau = np.clip(tau, -tau_max, tau_max)
        domega_dt = tau / J_axis

        # Add derivative term using current angular accel estimate
        tau_pd = kp * err - kd * domega_dt
        tau_pd = np.clip(tau_pd, -tau_max, tau_max)
        domega_dt = tau_pd / J_axis

        omega += domega_dt * dt
        t += dt
        times.append(t)
        omegas.append(omega)

    return np.array(times), np.array(omegas)


def itae_cost(params, J_axis: float, tau_max: float, omega_des: float) -> float:
    """ITAE cost for a given (kp, kd) pair."""
    kp, kd = params
    if kp <= 0 or kd < 0:
        return 1e6
    t, omega = simulate_step(kp, kd, J_axis, tau_max, omega_des)
    error = np.abs(omega_des - omega)
    return float(np.trapz(t * error, t))


# ---------------------------------------------------------------------------
# Auto-tune per axis
# ---------------------------------------------------------------------------

def tune_axis(name: str, J_axis: float, tau_max: float, omega_des: float,
              plot: bool = False):
    print(f"\n--- Tuning {name} axis ---")
    print(f"  J={J_axis:.6f} kg·m²  τ_max={tau_max:.6f} N·m  ω_des={omega_des:.1f} rad/s")

    # Analytical starting point: pure P with 20 Hz bandwidth
    kp0 = J_axis / 0.05   # τ_ctrl = 0.05s
    kd0 = 0.0

    result = minimize(
        itae_cost,
        x0=[kp0, kd0],
        args=(J_axis, tau_max, omega_des),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000},
        bounds=[(1e-4, 10.0), (0.0, 2.0)],
    )

    kp_opt, kd_opt = result.x
    kp_opt = max(kp_opt, 1e-4)
    kd_opt = max(kd_opt, 0.0)

    t, omega = simulate_step(kp_opt, kd_opt, J_axis, tau_max, omega_des)
    # Metrics
    ss_error  = abs(omega[-1] - omega_des)
    overshoot = max(0.0, omega.max() - omega_des) / omega_des * 100 if omega_des > 0 else 0.0
    # Rise time: first crossing of 90% of target
    above_90  = np.where(omega >= 0.9 * omega_des)[0]
    rise_time = t[above_90[0]] if len(above_90) > 0 else float("inf")

    print(f"  kp={kp_opt:.6f}  kd={kd_opt:.6f}")
    print(f"  Rise time: {rise_time*1000:.1f} ms  Overshoot: {overshoot:.1f}%  SS error: {ss_error:.4f} rad/s")

    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 3))
            plt.plot(t, omega, label=f"ω ({name})")
            plt.axhline(omega_des, color="r", linestyle="--", label="target")
            plt.xlabel("time (s)")
            plt.ylabel("angular rate (rad/s)")
            plt.title(f"CTBR PD Auto-tune — {name} axis  (kp={kp_opt:.4f}, kd={kd_opt:.4f})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"/tmp/ctbr_tune_{name}.png", dpi=100)
            print(f"  Plot saved to /tmp/ctbr_tune_{name}.png")
        except ImportError:
            print("  matplotlib not available — skipping plot")

    return float(kp_opt), float(kd_opt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Save step-response plots")
    args = parser.parse_args()

    print("=" * 60)
    print("CTBR PD Auto-Tune")
    print(f"  TAU_MAX  roll={TAU_MAX_ROLL:.6f}  pitch={TAU_MAX_PITCH:.6f}  yaw={TAU_MAX_YAW:.8f}")
    print(f"  MAX_THRUST={MAX_THRUST:.3f} N  HOVER={HOVER_THRUST:.3f} N")
    print("=" * 60)

    axes = [
        ("roll",  J[0], TAU_MAX_ROLL,  STEP_RATE[0]),
        ("pitch", J[1], TAU_MAX_PITCH, STEP_RATE[1]),
        ("yaw",   J[2], TAU_MAX_YAW,   STEP_RATE[2]),
    ]

    gains = {}
    for name, J_axis, tau_max, omega_des in axes:
        kp, kd = tune_axis(name, J_axis, tau_max, omega_des, plot=args.plot)
        gains[name] = {"kp": round(kp, 6), "kd": round(kd, 6)}

    out = {
        "roll":  gains["roll"],
        "pitch": gains["pitch"],
        "yaw":   gains["yaw"],
        "max_roll_rate":  MAX_ROLL_RATE,
        "max_pitch_rate": MAX_PITCH_RATE,
        "max_yaw_rate":   MAX_YAW_RATE,
        "max_thrust":     round(MAX_THRUST, 4),
        "hover_thrust":   round(HOVER_THRUST, 4),
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "dreamer", "configs", "ctbr_gains.yaml"
    )
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Gains saved to {out_path}")
    print(yaml.dump(out, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
