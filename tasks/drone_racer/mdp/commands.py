# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import cv2
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass

from .events import reset_after_prev_gate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GateTargetingCommand(CommandTerm):
    """Command generator that generates a pose command from a uniform distribution."""

    cfg: GateTargetingCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: GateTargetingCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.cfg = cfg

        # FPV video recording
        if self.cfg.record_fpv:
            self.video_id = 0
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera")
            self.sensor: TiledCamera = self._env.scene.sensors[self.sensor_cfg.name]

        # extract the robot and track for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.track: RigidObjectCollection = env.scene[cfg.track_name]
        self.gate_size = cfg.gate_size
        self.num_gates = self.track.num_objects

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in simulation world frame
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        # IMPORTANT: clone here. Isaac Lab's root_pos_w is a VIEW of the live state buffer; without
        # .clone() prev_robot_pos_w aliases the live tensor and gate-pass detection compares pos
        # to itself every sub-step → passed_gate_plane always False and progress reward = 0.
        self.prev_robot_pos_w = self.robot.data.root_pos_w.clone()
        self._gate_missed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.next_gate_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # Sticky accumulators — OR across all physics sub-steps in one RL step.
        # Reset by reset_step_accumulators() (called from env wrapper before env.step).
        self._gate_passed_accum = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_missed_accum = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.next_gate_w = torch.zeros(self.num_envs, 7, device=self.device)
        # per-episode counters (reset in _resample_command, incremented in _update_command)
        self._gates_passed_episode = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._laps_completed_episode = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def __str__(self) -> str:
        msg = "GateTargetingCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.next_gate_w

    @property
    def gate_missed(self) -> torch.Tensor:
        # Pass dominates miss. With decimation > 1, drone hovering near the gate plane can cross
        # multiple times within one RL step — some crossings inside bbox (just_passed) and others
        # at the bbox edge (just_missed). OR-accumulation makes both flags True simultaneously,
        # and the reward `+passed - missed` cancels to 0. Mask missed-only when passed didn't
        # also fire, so a successful gate pass never gets cancelled by a marginal edge wobble.
        return self._gate_missed_accum & ~self._gate_passed_accum

    @property
    def gate_passed(self) -> torch.Tensor:
        return self._gate_passed_accum

    def reset_step_accumulators(self) -> None:
        """Reset sticky gate flags. Call from env wrapper before each env.step()."""
        self._gate_passed_accum.zero_()
        self._gate_missed_accum.zero_()

    @property
    def previous_pos(self) -> torch.Tensor:
        return self.prev_robot_pos_w

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        episode = self._env.extras.setdefault("episode", {})
        episode["Episode/gates_per_episode"] = self._gates_passed_episode.float().mean()
        episode["Episode/laps_per_episode"] = self._laps_completed_episode.float().mean()
        episode["Episode/success_rate"] = (self._laps_completed_episode > 0).float().mean()

    def _resample_command(self, env_ids: Sequence[int]):
        # Release and reinitialize video writer only after the first iteration
        if hasattr(self, "out") and self.cfg.record_fpv:
            self.out.release()
            print(f"FPV video saved as fpv_{self.video_id}.mp4")
            self.video_id += 1

        if self.cfg.record_fpv:
            self.out = cv2.VideoWriter(f"fpv_{self.video_id}.mp4", self.fourcc, 100, (1000, 1000))

        self._gates_passed_episode[env_ids] = 0
        self._laps_completed_episode[env_ids] = 0

        # Clear accumulators for reset envs. We DELAY syncing prev_robot_pos_w until AFTER
        # reset_after_prev_gate has written the new spawn pose to physics — otherwise
        # prev_pos captures the crashed pose and root_pos_w becomes the spawn pose, so the
        # later _update_command sees prev != current and triggers a ghost gate crossing.
        self._gate_passed_accum[env_ids] = False
        self._gate_missed_accum[env_ids] = False
        self.prev_robot_pos_w = self.prev_robot_pos_w.clone()

        if self.cfg.randomise_start is None:
            self.next_gate_idx[env_ids] = 0

        else:
            if self.cfg.randomise_start:
                self.next_gate_idx[env_ids] = torch.randint(
                    low=0, high=self.num_gates, size=(len(env_ids),), device=self.device, dtype=torch.int32
                )
            else:
                self.next_gate_idx[env_ids] = 1

            # prev_indices may be -1 when next_gate_idx==0 — wrap to last gate.
            prev_indices = (self.next_gate_idx - 1) % self.num_gates
            next_indices = self.next_gate_idx  # not yet incremented = next gate to pass
            prev_pos = self.track.data.object_com_pos_w[self.env_ids, prev_indices]
            prev_quat = self.track.data.object_quat_w[self.env_ids, prev_indices]
            next_pos = self.track.data.object_com_pos_w[self.env_ids, next_indices]

            # Curriculum spawn: position is the LERP between prev_gate and next_gate.
            # spawn_lerp_alpha ∈ [0, 1]: 0 = at prev gate, 1 = at next gate, 0.5 = exact halfway.
            # The previous "forward_offset" approach failed on sharp track turns where prev gate's
            # forward axis doesn't point toward next gate. True lerp follows the segment regardless
            # of gate orientation. Drone retains prev_gate's orientation (so it's facing along the
            # outgoing forward direction of the gate it just passed).
            alpha = self.cfg.spawn_lerp_alpha
            lerp_pos = (1.0 - alpha) * prev_pos + alpha * next_pos
            # Pass lerp_pos as the "gate" pose with prev_gate's quat so reset_after_prev_gate
            # applies zero extra forward offset (already accounted for in the lerp).
            spawn_pose = torch.cat([lerp_pos, prev_quat], dim=1)

            # Direction prev→next (world frame) for the initial velocity bias.
            dir_vec = next_pos - prev_pos
            dir_norm = torch.norm(dir_vec, dim=1, keepdim=True).clamp(min=1e-6)
            init_vel = (dir_vec / dir_norm) * self.cfg.spawn_forward_velocity

            reset_after_prev_gate(
                env=self._env,
                env_ids=env_ids,
                gate_pose=spawn_pose,
                forward_offset=0.0,   # offset baked into lerp_pos already
                initial_lin_vel_world=init_vel,
                # Tightened from ±0.5 m / ±45° to keep the spawn inside the next gate's z-bbox
                # (half-size 0.75 m) and the drone near level so it doesn't diverge before the
                # rate controller can react. Re-widen once policy is competent (curriculum).
                pose_range={
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "z": (-0.2, 0.2),
                    "roll": (-0.1, 0.1),    # ±5.7°
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.3, 0.3),     # ±17°
                },
                velocity_range={
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                asset_cfg_name=self.cfg.asset_name,
            )

        # Snapshot prev_pos AFTER any pose write done by reset_after_prev_gate. For the
        # randomise_start=None branch (no respawn), root_pos_w is unchanged so this just
        # re-syncs to the current value, which is also correct.
        self.prev_robot_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids]

    def _update_command(self):
        if self.cfg.record_fpv:
            image = self.sensor.data.output["rgb"][0].cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.out.write(image)

        next_gate_positions = self.track.data.object_com_pos_w[self.env_ids, self.next_gate_idx]
        next_gate_orientations = self.track.data.object_quat_w[self.env_ids, self.next_gate_idx]
        self.next_gate_w = torch.cat([next_gate_positions, next_gate_orientations], dim=1)

        # Gate passing logic — full 3D plane crossing using quaternion-rotated x-axis as normal.
        # Old code used yaw-only 2D normal, which (a) failed on tilted gates and (b) dropped z entirely,
        # so a drone flying ABOVE the gate could trigger a plane crossing and accumulate a ghost
        # gate_missed via the bbox check. Now we project the full 3D rel-pos onto the gate normal.
        x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, 3)
        gate_normal = math_utils.quat_apply(self.next_gate_w[:, 3:7], x_axis)  # (N, 3)

        rel_old = self.prev_robot_pos_w - self.next_gate_w[:, :3]
        rel_new = self.robot.data.root_pos_w - self.next_gate_w[:, :3]
        pos_old_projected = (rel_old * gate_normal).sum(dim=-1)
        pos_new_projected = (rel_new * gate_normal).sum(dim=-1)
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)

        # Accumulate with OR so gate passes at any physics sub-step are captured.
        # (With decimation=4, overwriting would discard ~75% of real gate passes.)
        just_passed = passed_gate_plane & (
            torch.all(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) < (self.gate_size / 2), dim=1)
        )
        just_missed = passed_gate_plane & (
            torch.any(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) > (self.gate_size / 2), dim=1)
        )
        self._gate_passed_accum |= just_passed
        self._gate_missed_accum |= just_missed
        # Keep _gate_passed/_gate_missed as current-sub-step for internal counter update
        self._gate_passed = just_passed
        self._gate_missed = just_missed

        # Update per-episode counters (check last gate before index wraps)
        lap_completed = self._gate_passed & (self.next_gate_idx == self.num_gates - 1)
        self._gates_passed_episode += self._gate_passed.int()
        self._laps_completed_episode += lap_completed.int()

        # Update next gate target for the envs that passed the gate
        self.next_gate_idx[self._gate_passed] += 1
        self.next_gate_idx = self.next_gate_idx % self.num_gates

        # CRITICAL: clone to take a real snapshot. Without .clone() prev_robot_pos_w becomes an
        # alias to the live root_pos_w buffer (Isaac Lab returns a view), so on the next sub-step
        # rel_old == rel_new → passed_gate_plane always False → no gate detection, zero progress
        # reward, no learning signal. The single .clone() in _resample_command only fixed the
        # first sub-step after reset, leaving 99% of sub-steps broken.
        self.prev_robot_pos_w = self.robot.data.root_pos_w.clone()

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                # -- goal pose
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                # -- current body pose
                self.drone_visualizer = VisualizationMarkers(self.cfg.drone_visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
            self.drone_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
                self.drone_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        self.target_visualizer.visualize(self.next_gate_w[:, :3], self.next_gate_w[:, 3:])
        self.drone_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)


@configclass
class GateTargetingCommandCfg(CommandTermCfg):
    """Configuration for gate targeting command generator."""

    class_type: type = GateTargetingCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    track_name: str = MISSING
    """Name of the track in the environment for which the commands are generated."""

    randomise_start: bool | None = None
    """If True, the starting gate is randomised at every reset."""

    record_fpv: bool = False
    """If True, the first-person view (FPV) camera is recorded during the simulation."""

    gate_size: float = 1.5
    """Size of the gate bounding box in meters (half-size 0.75 m for default 1.5)."""

    spawn_lerp_alpha: float = 0.3
    """Curriculum knob in [0, 1]. Drone spawns at LERP(prev_gate_pos, next_gate_pos, alpha).
    0.0 = at prev gate, 0.5 = exact halfway, 1.0 = at next gate (very easy — just sit there).
    Use 0.7 to bootstrap an untrained policy through early exploration. Once gate-pass rate
    exceeds ~5%, drop toward 0.3 — drone must navigate further, forcing it to learn the actual
    flight skill instead of overfitting to "drift and pass the nearby gate". Drop to 0.0 once
    policy can reliably chain 2+ gates.
    """

    spawn_forward_velocity: float = 1.5
    """Initial velocity (m/s) along the prev_gate → next_gate direction at spawn. Helps an
    untrained Gaussian-mean=0 policy collect data where the drone is ALREADY moving toward
    the gate — random rate noise then doesn't have to overcome zero translation from rest.
    Set to 0.0 to disable.
    """
    """Size of the gate in meters. This is used to determine if the drone has passed through the gate."""

    target_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    drone_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    target_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
    drone_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
