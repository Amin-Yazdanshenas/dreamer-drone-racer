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
from isaaclab.sensors import TiledCamera

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_GATE_LABEL_TO_CLASS_ID: dict[str, int] = {}

# Frame-stacking buffer for gate_perception_features — keyed by id(env)
_PERC_BUFFER: dict[int, torch.Tensor] = {}

# Pinhole constants for 64×64 camera with default PinholeCameraCfg
# fx = focal_length_mm / horizontal_aperture_mm * width_px = 24.0 / 20.955 * 64
_FX_PIXELS: float = 73.3
_GATE_REAL_WIDTH_M: float = 1.5
_DIST_MAX_M: float = 20.0


def gate_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    command_name: str = "target",
) -> torch.Tensor:
    """Binary mask of the current target gate only.

    Returns (N, H*W) float32: 1.0 = target gate pixel, 0.0 = everything else.
    Requires colorize_semantic_segmentation=False (RGBA bytes = raw uint32 class ID).
    Each gate has a unique semantic label (gate_1, gate_2, …); per-env filtering by
    next_gate_idx selects only the gate the drone should fly through next.
    """
    camera: TiledCamera = env.scene[sensor_cfg.name]
    seg = camera.data.output["semantic_segmentation"]  # (N, H, W, 4) uint8

    global _GATE_LABEL_TO_CLASS_ID
    if not _GATE_LABEL_TO_CLASS_ID:
        info = camera.data.info.get("semantic_segmentation", {})
        id_to_labels = info.get("idToLabels", {})
        if id_to_labels:
            _GATE_LABEL_TO_CLASS_ID = {
                v["class"]: int(k)
                for k, v in id_to_labels.items()
                if isinstance(v, dict) and "class" in v
            }

    target_idx = env.command_manager.get_term(command_name).next_gate_idx  # (N,) int32

    if _GATE_LABEL_TO_CLASS_ID:
        class_ids = torch.tensor(
            [_GATE_LABEL_TO_CLASS_ID.get(f"gate_{int(i.item()) + 1}", int(i.item()) + 1) for i in target_idx],
            dtype=seg.dtype,
            device=seg.device,
        )
    else:
        # Fallback: gate_N registered in insertion order → class ID N (1-indexed)
        class_ids = (target_idx + 1).to(dtype=seg.dtype)

    mask = (seg[..., 0] == class_ids[:, None, None]).float()  # (N, H, W)
    return mask.reshape(env.num_envs, -1)


def gate_perception_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    command_name: str = "target",
    num_frames: int = 1,
    seg_noise_prob: float = 0.0,
) -> torch.Tensor:
    """Compact 9-dim gate geometry features extracted from the semantic segmentation mask.

    Per frame (9 features):
      [0] visible         — 1.0 if any mask pixel nonzero, else 0.0
      [1] centroid_x      — pixel-weighted col mean, normalized to [-1, 1]
      [2] centroid_y      — pixel-weighted row mean, normalized to [-1, 1]
      [3] bbox_xmin       — leftmost active col, normalized to [-1, 1]
      [4] bbox_ymin       — topmost active row, normalized to [-1, 1]
      [5] bbox_xmax       — rightmost active col, normalized to [-1, 1]
      [6] bbox_ymax       — bottommost active row, normalized to [-1, 1]
      [7] area_norm       — active_pixels / (H*W), in [0, 1]
      [8] dist_approx_norm — pinhole distance estimate, normalized to [0, 1]

    Returns (N, 9*num_frames). When num_frames > 1, oldest frame is first.
    seg_noise_prob: per-env probability of dropping the entire mask each step.
    """
    camera: TiledCamera = env.scene[sensor_cfg.name]
    seg = camera.data.output["semantic_segmentation"]  # (N, H, W, 4) uint8
    N, H, W, _ = seg.shape
    device = seg.device

    global _GATE_LABEL_TO_CLASS_ID
    if not _GATE_LABEL_TO_CLASS_ID:
        info = camera.data.info.get("semantic_segmentation", {})
        id_to_labels = info.get("idToLabels", {})
        if id_to_labels:
            _GATE_LABEL_TO_CLASS_ID = {
                v["class"]: int(k)
                for k, v in id_to_labels.items()
                if isinstance(v, dict) and "class" in v
            }

    target_idx = env.command_manager.get_term(command_name).next_gate_idx  # (N,) int32

    if _GATE_LABEL_TO_CLASS_ID:
        class_ids = torch.tensor(
            [_GATE_LABEL_TO_CLASS_ID.get(f"gate_{int(i.item()) + 1}", int(i.item()) + 1) for i in target_idx],
            dtype=seg.dtype,
            device=device,
        )
    else:
        class_ids = (target_idx + 1).to(dtype=seg.dtype)

    mask = (seg[..., 0] == class_ids[:, None, None]).float()  # (N, H, W)

    if seg_noise_prob > 0.0:
        drop = (torch.rand(N, device=device) < seg_noise_prob).float()
        mask = mask * (1.0 - drop[:, None, None])

    # Coordinate grids: pixel 0 → -1.0, pixel W-1 → +1.0
    col_coords = torch.linspace(-1.0, 1.0, W, device=device)
    row_coords = torch.linspace(-1.0, 1.0, H, device=device)
    col_grid = col_coords[None, None, :].expand(N, H, W)
    row_grid = row_coords[None, :, None].expand(N, H, W)

    pixel_count = mask.sum(dim=(1, 2))              # (N,)
    safe_count = pixel_count.clamp(min=1.0)
    visible = (pixel_count > 0).float()             # (N,)
    zero = torch.zeros(N, device=device)

    # Weighted centroid (zeroed automatically when mask=0)
    centroid_x = (mask * col_grid).sum(dim=(1, 2)) / safe_count
    centroid_y = (mask * row_grid).sum(dim=(1, 2)) / safe_count

    # Bounding box: fill non-gate pixels with sentinel values
    col_min_fill = torch.where(mask.bool(), col_grid, torch.ones_like(col_grid))
    row_min_fill = torch.where(mask.bool(), row_grid, torch.ones_like(row_grid))
    col_max_fill = torch.where(mask.bool(), col_grid, -torch.ones_like(col_grid))
    row_max_fill = torch.where(mask.bool(), row_grid, -torch.ones_like(row_grid))

    bbox_xmin = col_min_fill.reshape(N, -1).min(dim=1).values
    bbox_ymin = row_min_fill.reshape(N, -1).min(dim=1).values
    bbox_xmax = col_max_fill.reshape(N, -1).max(dim=1).values
    bbox_ymax = row_max_fill.reshape(N, -1).max(dim=1).values

    has_gate = visible.bool()
    bbox_xmin = torch.where(has_gate, bbox_xmin, zero)
    bbox_ymin = torch.where(has_gate, bbox_ymin, zero)
    bbox_xmax = torch.where(has_gate, bbox_xmax, zero)
    bbox_ymax = torch.where(has_gate, bbox_ymax, zero)

    area_norm = pixel_count / float(H * W)

    # Distance estimate via pinhole model: dist = (real_width * fx) / apparent_width_px
    bbox_width_px = (bbox_xmax - bbox_xmin).clamp(min=0.0) * (W / 2.0)
    dist_approx = (_GATE_REAL_WIDTH_M * _FX_PIXELS) / bbox_width_px.clamp(min=1.0)
    dist_approx_norm = (dist_approx / _DIST_MAX_M).clamp(0.0, 1.0) * visible

    log(env, ["gate_visible"], visible.unsqueeze(1))
    log(env, ["gate_centroid_x"], centroid_x.unsqueeze(1))
    log(env, ["gate_centroid_y"], centroid_y.unsqueeze(1))
    log(env, ["gate_dist_approx"], (dist_approx_norm * _DIST_MAX_M).unsqueeze(1))

    feat = torch.stack(
        [visible, centroid_x, centroid_y, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, area_norm, dist_approx_norm],
        dim=1,
    )  # (N, 9)

    if num_frames <= 1:
        return feat

    # Frame stacking: maintain rolling buffer of last num_frames observations
    env_id = id(env)
    if env_id not in _PERC_BUFFER or _PERC_BUFFER[env_id].shape != (N, 9 * num_frames):
        _PERC_BUFFER[env_id] = feat.repeat(1, num_frames)
    buf = torch.cat([_PERC_BUFFER[env_id][:, 9:], feat], dim=1)
    _PERC_BUFFER[env_id] = buf
    return buf  # (N, 9*num_frames)


def flat_image(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera")) -> torch.Tensor:
    """FPV camera image converted to grayscale and flattened to a 1-D vector per env.

    Returns shape (num_envs, H*W). Values are normalised to [0, 1].
    """
    camera: TiledCamera = env.scene[sensor_cfg.name]
    rgb = camera.data.output["rgb"]  # (N, H, W, 3), float32 in [0, 1]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]  # (N, H, W)
    return gray.reshape(env.num_envs, -1)  # (N, H*W)


def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    log(env, ["vx", "vy", "vz"], lin_vel)
    return lin_vel


def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
    log(env, ["wx", "wy", "wz"], ang_vel)
    return ang_vel


def root_quat_w(
    env: ManagerBasedRLEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    log(env, ["qw", "qx", "qy", "qz"], quat)
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_rotmat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation (3x3 flattened rotation matrix) in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    return flat_rotmat


def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w
    log(env, ["px", "py", "pz"], position)
    return position


def root_pose_g(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asset root position in the gate frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
    drone_pose_w = asset.data.root_state_w[:, :7]  # (num_envs, 7)

    # Extract positions and quaternions
    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    drone_pos_w = drone_pose_w[:, :3]
    drone_quat_w = drone_pose_w[:, 3:7]

    # Compute drone pose in gate frame
    # Inverse gate quaternion
    gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

    # Position of drone in gate frame
    rel_pos = drone_pos_w - gate_pos_w
    drone_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos)

    # Orientation of drone in gate frame
    drone_quat_g = math_utils.quat_mul(gate_quat_w_inv, drone_quat_w)

    # Concatenate position and quaternion
    position = torch.cat([drone_pos_g, drone_quat_g], dim=-1)

    return position


def next_gate_pose_g(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Asset root position in the gate frame."""
    gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
    next_gate_pose_w = env.command_manager.get_term(command_name).next_gate  # (num_envs, 7)

    # Extract positions and quaternions
    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    next_gate_pos_w = next_gate_pose_w[:, :3]
    next_gate_quat_w = next_gate_pose_w[:, 3:7]

    # Compute drone pose in gate frame
    # Inverse gate quaternion
    gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

    # Position of drone in gate frame
    rel_pos = next_gate_pos_w - gate_pos_w
    next_gate_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos)

    # Orientation of drone in gate frame
    next_gate_quat_g = math_utils.quat_mul(gate_quat_w_inv, next_gate_quat_w)

    # Concatenate position and quaternion
    position = torch.cat([next_gate_pos_g, next_gate_quat_g], dim=-1)

    return position


def target_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of target in body frame."""

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    pos_b, _ = math_utils.subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor)

    return pos_b
