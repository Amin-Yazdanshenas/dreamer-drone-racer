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
    asymmetric: bool = True,
) -> torch.Tensor:
    """Progress toward target gate (prev_distance - current_distance).

    asymmetric=True (default): clamp to >=0 — only reward forward progress. Used by Dreamer;
    helps under-trained policies escape "hover" local optimum.

    asymmetric=False: signed (negative when retreating). Required by legacy PPO tasks —
    PPO needs the negative gradient to push policy away from bad actions; without it
    std explodes and policy never commits to gate-passing.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    target_pos = env.command_manager.get_term(command_name).command[:, :3]
    previous_pos = env.command_manager.get_term(command_name).previous_pos
    current_pos = asset.data.root_pos_w

    prev_distance = torch.norm(previous_pos - target_pos, dim=1)
    current_distance = torch.norm(current_pos - target_pos, dim=1)

    progress = prev_distance - current_distance
    if asymmetric:
        progress = progress.clamp(min=0.0)
    return progress


def gate_passed(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    penalize_miss: bool = False,
) -> torch.Tensor:
    """Reward for passing the current target gate, computed INLINE.

    penalize_miss=False (default): +1 when drone passes through bbox center, 0 otherwise.
    Used by Dreamer; avoids "gate area = dangerous" learning trap.

    penalize_miss=True: +1 for pass, -1 for plane-crossing-while-off-center. Required by
    legacy PPO tasks (upstream behavior).

    Isaac Lab calls reward_manager BEFORE command_manager (manager_based_rl_env.py:208 vs 232),
    so reading cmd.gate_passed (filled by _update_command in command_manager.compute) at reward
    time always returns the PREVIOUS step's stale value. Worse, env_wrapper zeroes the
    accumulator before each step, so the +30 reward never reaches replay.

    Workaround: compute the plane-crossing + bbox check inline using the same prev/current
    positions as _update_command, but here it runs at the correct time (after physics, before
    reward bookkeeping). prev_robot_pos_w is the snapshot saved at the END of the LAST
    _update_command call → approximately the position at the START of this RL step.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    cur_pos = asset.data.root_pos_w
    prev_pos = cmd.prev_robot_pos_w
    gate_pose = cmd.next_gate_w
    gate_pos = gate_pose[:, :3]
    gate_quat = gate_pose[:, 3:7]
    half_size = cmd.gate_size / 2.0

    x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=cur_pos.device).expand_as(cur_pos)
    gate_normal = math_utils.quat_apply(gate_quat, x_axis)
    rel_old = prev_pos - gate_pos
    rel_new = cur_pos - gate_pos
    proj_old = (rel_old * gate_normal).sum(dim=-1)
    proj_new = (rel_new * gate_normal).sum(dim=-1)
    crossing = (proj_old < 0) & (proj_new > 0)

    # Gate-frame opening check. Transform the drone position into the gate frame and bound the
    # IN-PLANE (local y, z) offset against the opening half-size. The previous world-axis bbox
    # (|dx|,|dy|,|dz| < half) is only correct for axis-aligned gates; the track has a yaw-rotated
    # 225° gate (drone_racer_env_cfg.py gate "3"), for which the world cube does not match the
    # rotated opening — a drone passing well off to the side could be counted as a pass and vice
    # versa. local x is the through-normal direction (already handled by the plane `crossing`),
    # so only y, z bound the opening.
    rel_gate, _ = math_utils.subtract_frame_transforms(gate_pos, gate_quat, cur_pos)
    in_gate = (rel_gate[:, 1].abs() < half_size) & (rel_gate[:, 2].abs() < half_size)

    passed = crossing & in_gate
    if penalize_miss:
        missed = crossing & ~in_gate
        return passed.float() - missed.float()
    # Centering-scaled pass bonus: reward CLEAN (centered) crossings more than edge grazes, to
    # teach tighter threading lines (the policy currently learns "pass somehow", clipping frames —
    # ~96-100% of deaths are gate-frame collisions). Paid ONLY at the crossing event: a sparse,
    # per-pass scaling, NOT a dense plane-local field, so it avoids the parking-wall optimum that
    # forced gate_offset_penalty to weight 0. offset_norm in [0,1]: 0 = dead-centre, 1 = frame
    # edge. Multiplier in [0.5, 1.0] — an off-centre pass still pays (it IS a pass), a centred pass
    # pays double, giving a gradient toward the centre.
    offset_norm = (torch.maximum(rel_gate[:, 1].abs(), rel_gate[:, 2].abs()) / half_size).clamp(0.0, 1.0)
    center_mult = 1.0 - 0.5 * offset_norm
    return passed.float() * center_mult


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
