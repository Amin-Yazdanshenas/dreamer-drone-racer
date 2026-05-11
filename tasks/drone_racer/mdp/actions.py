# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
import yaml
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics import Allocation, Motor
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ControlAction(ActionTerm):
    r"""Body torque control action term.

    This action term applies a wrench to the drone body frame based on action commands

    """

    cfg: ControlActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._allocation = Allocation(
            num_envs=self.num_envs,
            arm_length=self.cfg.arm_length,
            thrust_coeff=self.cfg.thrust_coef,
            drag_coeff=self.cfg.drag_coef,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._motor = Motor(
            num_envs=self.num_envs,
            taus=self.cfg.taus,
            init=self.cfg.init,
            max_rate=self.cfg.max_rate,
            min_rate=self.cfg.min_rate,
            dt=env.physics_dt,
            use=self.cfg.use_motor_model,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # TODO: make more explicit (thrust = 6, rates = 6, attitude = 6) all happen to be 6, but they represent different things
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):

        self._raw_actions[:] = actions
        clamped = self._raw_actions.clamp_(-1.0, 1.0)
        mapped = (clamped + 1.0) / 2.0
        omega_ref = self.cfg.omega_max * mapped
        omega_real = self._motor.compute(omega_ref)
        self._processed_actions = self._allocation.compute(omega_real)

        log(self._env, ["a1", "a2", "a3", "a4"], self._raw_actions)
        log(self._env, ["w1", "w2", "w3", "w4"], omega_real)

    def apply_actions(self):
        self._thrust[:, 0, 2] = self._processed_actions[:, 0]
        self._moment[:, 0, :] = self._processed_actions[:, 1:]
        self._robot.permanent_wrench_composer.set_forces_and_torques(self._thrust, self._moment, body_ids=self._body_id)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def reset(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0

        self._motor.reset(env_ids)
        self._robot.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._env.scene.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


# ---------------------------------------------------------------------------
# CTBR Action — Collective Thrust + Body Rates (acro mode equivalent)
# ---------------------------------------------------------------------------

class CTBRAction(ActionTerm):
    """CTBR action term: actor outputs [c, ω_x, ω_y, ω_z] in [-1, 1].

    A PD rate controller runs at physics frequency (400 Hz) converting body-rate
    setpoints to torques.  Gains are loaded from a YAML file produced by
    scripts/tune_ctbr_gains.py.

    Action mapping:
        c   ∈ [-1, 1]  →  F_z = ((c+1)/2) × max_thrust      (collective thrust)
        ω_x ∈ [-1, 1]  →  ω_des_x = ω_x × max_roll_rate     (roll rate setpoint)
        ω_y ∈ [-1, 1]  →  ω_des_y = ω_y × max_pitch_rate
        ω_z ∈ [-1, 1]  →  ω_des_z = ω_z × max_yaw_rate
    """

    cfg: "CTBRActionCfg"

    def __init__(self, cfg: "CTBRActionCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]
        self._dt = env.physics_dt

        # Load tuned PD gains
        gains_path = os.path.abspath(cfg.gains_path)
        with open(gains_path) as f:
            g = yaml.safe_load(f)

        self._kp = torch.tensor(
            [g["roll"]["kp"], g["pitch"]["kp"], g["yaw"]["kp"]],
            device=self.device, dtype=torch.float32,
        )
        self._kd = torch.tensor(
            [g["roll"]["kd"], g["pitch"]["kd"], g["yaw"]["kd"]],
            device=self.device, dtype=torch.float32,
        )
        self._max_rates = torch.tensor(
            [g["max_roll_rate"], g["max_pitch_rate"], g["max_yaw_rate"]],
            device=self.device, dtype=torch.float32,
        )
        self._max_thrust: float = g["max_thrust"]
        self._hover_thrust: float = g["hover_thrust"]

        # Max torques: roll/pitch from arm+thrust, yaw from drag
        arm_eff = cfg.arm_length / math.sqrt(2.0)
        t_single = cfg.thrust_coef * cfg.omega_max ** 2
        tau_rp = 2.0 * arm_eff * t_single
        tau_yaw = 4.0 * cfg.drag_coef * cfg.omega_max ** 2
        self._tau_max = torch.tensor(
            [tau_rp, tau_rp, tau_yaw], device=self.device, dtype=torch.float32
        )

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        self._collective = torch.zeros(self.num_envs, device=self.device)
        self._prev_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_err = torch.zeros(self.num_envs, 3, device=self.device)
        self._thrust_buf = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment_buf = torch.zeros(self.num_envs, 1, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        # Asymmetric linear map: c=0 → hover_thrust, c=+1 → max_thrust, c=-1 → 0.
        # The old map ((c+1)/2)*max_thrust put hover at c=-0.5, so an untrained Gaussian-mean-zero
        # policy commanded 2× hover thrust at every step → drone climbed to flyaway. Centering
        # neutral action on hover lets a near-zero policy at least hold altitude while it learns.
        c = self._raw_actions[:, 0]
        self._collective[:] = (
            self._hover_thrust
            + torch.where(
                c >= 0,
                c * (self._max_thrust - self._hover_thrust),
                c * self._hover_thrust,
            )
        ).clamp(min=0.0)
        self._omega_des[:] = self._raw_actions[:, 1:] * self._max_rates
        log(self._env, ["c", "wx_des", "wy_des", "wz_des"], self._raw_actions)

    def apply_actions(self) -> None:
        omega_cur = self._robot.data.root_ang_vel_b          # (N, 3) body frame

        # PD on rate error (correct PD form). Old code computed -kd * d(omega_cur)/dt which only
        # coincides with -kd * d(err)/dt when omega_des is held constant — fails at every RL
        # decimation boundary where omega_des steps. Track err derivative directly.
        err = self._omega_des - omega_cur
        derr_dt = (err - self._prev_err) / self._dt
        tau = self._kp * err + self._kd * derr_dt
        tau = tau.clamp(-self._tau_max, self._tau_max)

        self._thrust_buf[:, 0, 2] = self._collective
        self._moment_buf[:, 0, :] = tau
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            self._thrust_buf, self._moment_buf, body_ids=self._body_id
        )
        self._prev_ang_vel[:] = omega_cur
        self._prev_err[:] = err

    def reset(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._raw_actions[env_ids] = 0.0
        self._omega_des[env_ids] = 0.0
        self._collective[env_ids] = 0.0
        self._prev_ang_vel[env_ids] = 0.0
        self._prev_err[env_ids] = 0.0
        self._robot.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@configclass
class CTBRActionCfg(ActionTermCfg):
    """Configuration for CTBRAction."""

    class_type: type[ActionTerm] = CTBRAction
    asset_name: str = "robot"
    gains_path: str = "dreamer/configs/ctbr_gains.yaml"
    arm_length: float = 0.035
    thrust_coef: float = 2.25e-7
    drag_coef: float = 1.5e-9
    omega_max: float = 5145.0


@configclass
class ControlActionCfg(ActionTermCfg):
    """
    See :class:`ControlAction` for more details.
    """

    class_type: type[ActionTerm] = ControlAction
    """ Class of the action term."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
    arm_length: float = 0.035
    """Length of the arms of the drone in meters."""
    drag_coef: float = 1.5e-9
    """Drag torque coefficient."""
    thrust_coef: float = 2.25e-7
    """Thrust coefficient.
    Calculated with 5145 rad/s max angular velociy, thrust to weight: 4, mass: 0.6076 kg and gravity: 9.81 m/s^2.
    thrust_coef = (4 * 0.6076 * 9.81) / (4 * 5145**2) = 2.25e-7."""
    omega_max: float = 5145.0
    """Maximum angular velocity of the drone motors in rad/s.
    Calculated with 1950KV motor, with 6S LiPo battery with 4.2V per cell.
    1950 * 6 * 4.2 = 49,140 RPM ~= 5145 rad/s."""
    taus: list[float] = (0.0001, 0.0001, 0.0001, 0.0001)
    """Time constants for each motor."""
    init: list[float] = (2572.5, 2572.5, 2572.5, 2572.5)
    """Initial angular velocities for each motor in rad/s."""
    max_rate: list[float] = (50000.0, 50000.0, 50000.0, 50000.0)
    """Maximum rate of change of angular velocities for each motor in rad/s^2."""
    min_rate: list[float] = (-50000.0, -50000.0, -50000.0, -50000.0)
    """Minimum rate of change of angular velocities for each motor in rad/s^2."""
    use_motor_model: bool = False
    """Flag to determine if motor delay is bypassed."""
