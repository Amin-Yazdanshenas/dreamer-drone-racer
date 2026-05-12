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
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_after_prev_gate(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    gate_pose: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg_name: str = "robot",
    forward_offset: float = 1.0,
    initial_lin_vel_world: torch.Tensor | None = None,
):
    """Reset the asset along the forward axis of the previous gate.

    forward_offset (m): distance along the gate's local +x axis where the drone is spawned.
    Used as a curriculum knob — small (1 m) for hard task starting near prev gate, larger
    (e.g. 5 m) to bias spawn closer to next gate so untrained random policies have a chance
    of accidentally passing the next gate.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg_name]

    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    gate_pos = gate_pose[env_ids, :3]
    gate_quat = gate_pose[env_ids, 3:7]
    offset = torch.tensor([forward_offset, 0.0, 0.0], device=asset.device).expand(len(env_ids), 3)
    offset_world = math_utils.quat_apply(gate_quat, offset)
    pos_after_prev_gate = gate_pos + offset_world

    # pos_after_prev_gate is already in world frame (built from world-frame gate_pos + world offset).
    # Adding root_states[:, 0:3] (asset local default, e.g. (0,0,1)) would offset the drone by the
    # default-hover height again — unintended. Use env origins + gate-relative spawn only.
    positions = env.scene.env_origins[env_ids] + pos_after_prev_gate + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples
    # Add per-env world-frame initial linear velocity bias (e.g. toward next gate) so the drone
    # spawns already moving in the right direction — gives an untrained policy useful gradient
    # data without relying on lucky random rate noise to break it out of rest.
    # initial_lin_vel_world is sized (num_envs, 3); index by env_ids to match `velocities` rows.
    if initial_lin_vel_world is not None:
        velocities[:, 0:3] = velocities[:, 0:3] + initial_lin_vel_world[env_ids]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
