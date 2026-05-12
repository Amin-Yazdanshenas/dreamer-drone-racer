# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Buffer for action_smoothness reward — stores previous action per env, keyed by id(env)
_PREV_ACTION_BUFFER: dict[int, torch.Tensor] = {}


def pos_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute sum of squared errors
    return torch.sum(torch.square(asset.data.root_pos_w - target_pos_tensor), dim=1)


def pos_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    distance = torch.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return 1 - torch.tanh(distance / std)


def progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asymmetric progress reward: only reward forward progress, no penalty for retreating.

    Symmetric `prev_dist - cur_dist` is double-edged — random initial direction means 50% of
    the time the drone is punished for moving, expected gradient ~0, policy converges to
    "hover and don't crash" local optimum. Clamping to >=0 makes the signal one-sided so even
    noisy exploration learns to move toward the gate.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    target_pos = env.command_manager.get_term(command_name).command[:, :3]
    previous_pos = env.command_manager.get_term(command_name).previous_pos
    current_pos = asset.data.root_pos_w

    prev_distance = torch.norm(previous_pos - target_pos, dim=1)
    current_distance = torch.norm(current_pos - target_pos, dim=1)

    progress = prev_distance - current_distance
    return progress.clamp(min=0.0)


def gate_passed(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward for passing a gate."""
    missed = (-1.0) * env.command_manager.get_term(command_name).gate_missed
    passed = (1.0) * env.command_manager.get_term(command_name).gate_passed
    return missed + passed


def lookat_next_gate(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for looking at the next gate."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    drone_pos = asset.data.root_pos_w
    drone_att = asset.data.root_quat_w
    next_gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    vec_to_gate = next_gate_pos - drone_pos
    vec_to_gate = math_utils.normalize(vec_to_gate)

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(env.num_envs, 3)
    drone_x_axis = math_utils.quat_apply(drone_att, x_axis)
    drone_x_axis = math_utils.normalize(drone_x_axis)

    dot = (drone_x_axis * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    return torch.exp(-angle / std)


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)


def velocity_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward drone for flying toward the target gate.

    Returns exp(-angle / std) where angle is between linear velocity and the gate direction.
    Suppressed when speed < 0.1 m/s to avoid spurious reward at hover.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    drone_pos = asset.data.root_pos_w
    lin_vel_w = asset.data.root_lin_vel_w
    gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    vec_to_gate = math_utils.normalize(gate_pos - drone_pos)
    speed = torch.norm(lin_vel_w, dim=1, keepdim=True).clamp(min=1e-6)
    vel_dir = lin_vel_w / speed

    dot = (vel_dir * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    moving = (speed.squeeze(1) > 0.1).float()
    reward = torch.exp(-angle / std) * moving

    log(env, ["vel_align_angle"], angle.unsqueeze(1))
    return reward


def action_smoothness(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize large changes in motor commands between consecutive steps.

    Returns ||a_t - a_{t-1}||^2 per environment. Uses a module-level buffer for a_{t-1}.
    """
    current_action = env.action_manager.action.detach()  # (N, 4)
    N = current_action.shape[0]
    env_id = id(env)

    if env_id not in _PREV_ACTION_BUFFER or _PREV_ACTION_BUFFER[env_id].shape[0] != N:
        _PREV_ACTION_BUFFER[env_id] = torch.zeros_like(current_action)

    prev_action = _PREV_ACTION_BUFFER[env_id]
    delta = current_action - prev_action
    penalty = torch.sum(delta ** 2, dim=1)

    _PREV_ACTION_BUFFER[env_id] = current_action.clone()

    log(env, ["action_smoothness"], penalty.unsqueeze(1))
    return penalty


def gate_offset_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    near_plane_dist: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral/vertical offset from gate center when drone is near the gate plane.

    Active only within near_plane_dist meters along the gate normal. Returns the L2 distance
    in the gate's lateral-vertical plane (perpendicular to the gate normal direction).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    N = env.num_envs

    gate_pose = env.command_manager.get_term(command_name).command  # (N, 7)
    gate_pos = gate_pose[:, :3]
    gate_quat = gate_pose[:, 3:7]
    drone_pos = asset.data.root_pos_w

    # Gate normal = x-axis of gate frame rotated to world frame
    x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=gate_pos.device).expand(N, 3)
    gate_normal = math_utils.quat_apply(gate_quat, x_axis)  # (N, 3)

    rel_pos = drone_pos - gate_pos
    dist_along_normal = (rel_pos * gate_normal).sum(dim=1).abs()

    # Project rel_pos onto gate plane and measure offset
    proj_on_normal = (rel_pos * gate_normal).sum(dim=1, keepdim=True) * gate_normal
    offset_in_plane = rel_pos - proj_on_normal
    offset_dist = torch.norm(offset_in_plane, dim=1)

    near_gate = (dist_along_normal < near_plane_dist).float()
    penalty = offset_dist * near_gate

    log(env, ["gate_offset"], offset_dist.unsqueeze(1))
    log(env, ["near_gate_plane"], near_gate.unsqueeze(1))
    return penalty
