# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp import illegal_contact as _illegal_contact
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_contact_grace(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    grace_steps: int = 3,
) -> torch.Tensor:
    """illegal_contact, but ignored during the first ``grace_steps`` control steps after reset.

    The Sim 5.1 ContactSensor reports a residual phantom force spike on teleport-spawn that
    occasionally exceeds the threshold, killing fresh episodes at step 1 mid-air
    (diagnose_terminations measured ~7% of episodes dying at step<=3, half of them step-1
    mid-air collisions). A real crash cannot occur off a mid-air spawn within a few control
    steps, so contacts inside the grace window are discarded.
    """
    fired = _illegal_contact(env, threshold=threshold, sensor_cfg=sensor_cfg)
    return fired & (env.episode_length_buf > grace_steps)


def flyaway(
    env: ManagerBasedRLEnv,
    distance: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's is too far away from the target position."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute distance
    distance_tensor = torch.linalg.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return distance_tensor > distance
