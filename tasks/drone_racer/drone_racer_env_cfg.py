# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .track_generator import generate_track

from assets.five_in_drone import FIVE_IN_DRONE  # isort:skip


@configclass
class DroneRacerSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # track
    track: RigidObjectCollectionCfg = generate_track(
        track_config={
            "1": {"pos": (0.0, 0.0, 1.0), "yaw": 0.0},
            "2": {"pos": (10.0, 5.0, 0.0), "yaw": 0.0},
            "3": {"pos": (10.0, -5.0, 0.0), "yaw": (5 / 4) * torch.pi},
            "4": {"pos": (-5.0, -5.0, 2.5), "yaw": torch.pi},
            "5": {"pos": (-5.0, -5.0, 0.0), "yaw": 0.0},
            "6": {"pos": (5.0, 0.0, 0.0), "yaw": (1 / 2) * torch.pi},
            "7": {"pos": (0.0, 5.0, 0.0), "yaw": torch.pi},
        }
    )

    # robot
    robot: ArticulationCfg = FIVE_IN_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # Sim 5.1 ContactSensor returns a phantom force on the robot body at rest. Narrow to /body
    # (drop props per NVIDIA PhysX guidance — stale data on small high-frequency parts) and
    # add history/update_period/force_threshold so the buffered net force is comparable across
    # sub-steps. illegal_contact threshold is also bumped above the residual noise floor.
    collision_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        history_length=3,
        update_period=0.0,
        force_threshold=10.0,
        debug_vis=False,
    )
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/body", debug_vis=False)
    # Camera tilted ~30° UP (rot = quat 30° about body -Y). A level camera (rot identity)
    # pitched down with the drone when it accelerated forward, dropping the target gate out of
    # the top of the frame — dump_fpv measured the target gate visible in only 15% of frames and
    # the drone crashed after gate 1 because it flew blind toward gate 2. Real FPV racing drones
    # mount the camera tilted up 20-45° for exactly this reason. (w,x,y,z) = (cos15°,0,-sin15°,0).
    # VERIFY DIRECTION: re-run scripts/rl/dump_fpv.py — gate-visible % should rise well above 15%;
    # if it drops, flip the sign of the y component to +0.2588.
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.14, 0.0, 0.05), rot=(0.9659, 0.0, -0.2588, 0.0), convention="world"),
        data_types=["semantic_segmentation"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(),
        width=64,
        height=64,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model=False)


@configclass
class DreamerActionsCfg:
    """CTBR action space for DreamerV3 — matches the Dream to Fly paper."""

    control_action: mdp.CTBRActionCfg = mdp.CTBRActionCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations: FPV camera (flattened grayscale) + IMU.
        These are the only observations available at deployment — no ground truth."""

        image = ObsTerm(func=mdp.gate_mask)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        imu_att = ObsTerm(func=mdp.imu_orientation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True  # flat vector: 64*64 (target gate mask) + 3 + 4 = 4103

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations: privileged ground-truth state used only during training."""

        position = ObsTerm(func=mdp.root_pos_w)
        attitude = ObsTerm(func=mdp.root_quat_w)
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True  # flat vector: 20-dim

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    # TODO: Resetting base happens in the command reset also for the moment
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-3.5, -1.5),
                "y": (-0.5, 0.5),
                "z": (0.5, 1.5),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # intervals — push_robot disabled for Dreamer: random forces destabilize the untrained
    # policy, cascade crashes, and the resulting reset spikes stall the viewport. Legacy PPO
    # re-enables this via LegacyEventCfg.
    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.2),
    #     params={
    #         "force_range": (-0.1, 0.1),
    #         "torque_range": (-0.05, 0.05),
    #     },
    # )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target = mdp.GateTargetingCommandCfg(
        asset_name="robot",
        track_name="track",
        randomise_start=None,
        record_fpv=False,
        resampling_time_range=(1e9, 1e9),
        debug_vis=False,
        # Dreamer-tuned spawn: lerp 30% between prev/next gate, no extra forward offset,
        # initial 1.5 m/s velocity bias toward next gate so untrained mean=0 policy still
        # collects forward-motion data. Legacy PPO uses LegacyCommandsCfg below.
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP — Dreamer-tuned.

    Empirical 5M-step run (2026-05-25_02-59-24) with upstream PPO weights produced zero gate
    passes: terminating=-500 + gate_passed=-400-on-miss + signed progress collapsed the actor
    into a hover-safely policy. Legacy PPO weights live in LegacyRewardsCfg below.
    """

    # Reward shaping (post-2.8M env-step diagnostic): progress weight=100 BACKFIRED — drone
    # learned to exploit drift toward target as a steady reward source, treating gate-passing as
    # too risky for marginal extra reward. Gate pass rate dropped 8× between 2.0M and 2.8M.
    # Solution: shrink progress shaping and grow the gate spike so gate-passing dominates.
    # - terminating −2 (softened crash penalty)
    # - ang_vel_l2 -0.01 (re-enabled): the in-sim CTBR sanity test (scripts/test_ctbr.py) showed
    #   that any sustained body rate of 2.5 rad/s tilts the drone past 90° within 1 second, so a
    #   stochastic actor with no attitude-stabilisation signal cannot stay airborne long enough
    #   to reach a second gate. Per-step penalty ≈ -0.0075 when tumbling at 5 rad/s, ≈ 0 when
    #   stable — comparable in magnitude to the near_gate dense reward so the actor has a real
    #   gradient pushing it toward low body rates.
    # - progress weight 10, asymmetric clamp (no negative progress reward)
    # - gate_passed 10000: dt-scales to +100 per pass (10× bigger than max drift reward)
    # - lookat_next kept small
    # terminating -20 -> -1000 (death-penalty dt-scaling fix, post-Phase-2 review). Isaac Lab
    # multiplies every reward term by dt (now 0.03 s at decimation=12), so sparse penalties need
    # weight ~= desired_spike / dt — the same arithmetic applied to gate_passed but never to this
    # term. -20 delivered only -0.6/crash while imag/value sat at -16..-32: in imagination a
    # crashing state (target ~ -0.6, future zeroed by the continuation weight) beat a continuing
    # state (target ~ V ~ -16) by ~+15 — the lambda targets literally rewarded dying, which is the
    # crash-after-pass mechanism behind the 2-gate ceiling and 100% eval collisions. -1000 ->
    # -30/crash makes death strictly worse than the worst observed continuation (|V| <= ~32), so
    # surviving always dominates crashing.
    terminating = RewTerm(func=mdp.is_terminated, weight=-1000.0)
    # Dialled back to -0.02 after the -0.05 run showed catastrophic instability:
    # episode_length swung 1 -> 340 -> 66 RL steps, episode_reward swung -0.06 -> -2.44 -> +3.76,
    # return/scale escalated to +6.06 (vs +4.25 at -0.01). The actor briefly mastered a 3.4s
    # episode but then collapsed -- the penalty was strong enough to invert the value landscape
    # mid-training. -0.02 puts the per-episode cost at ~-1.0 (≈30% of reward at +3.3), strong
    # enough to push action_abs_mean down without destabilising the actor's target distribution.
    # ang_vel_l2 -0.02 -> -0.002. -0.02 was 200x the PPO baseline (-0.0001) and PENALISES the
    # rotation the drone needs to turn toward / line up with a gate — a dense negative that
    # fights navigation and feeds the recurring value collapse. The actor no longer saturates
    # (action_sat_frac ~0.03 with the mean-tanh/Jacobian-entropy fixes), so the strong
    # attitude prior is no longer needed to prevent tumbling. -0.002 keeps a mild damping
    # without taxing legitimate maneuvering.
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.002)
    # progress is now SIGNED (asymmetric=False) — this is the parking-proof, potential-based
    # dense reward the working skrl-PPO baseline uses. It rewards closing distance to the target
    # gate AND PENALISES drifting away. A 5M-step run with proximity shaping (near_gate) + the
    # old asymmetric progress produced a drone that wandered near gates and never passed
    # (eval: 0/10 gates, collision rate 1.0, "drifts away from the gate"); the absolute
    # proximity salary peaked just before the gate and collapsed on passing, so the value-correct
    # actor learned to loiter, not cross. Signed progress has no static spot that pays — the only
    # way to keep earning is to keep closing -> pass -> close on the next gate -> repeat — and it
    # directly punishes the observed drift. Pre-C1-fix this term was clamped to avoid drift
    # exploitation, but with the value targets now correct (review C1) the signed form is right
    # and matches the proven PPO reward.
    progress = RewTerm(
        func=mdp.progress,
        weight=50.0,
        params={"command_name": "target", "asymmetric": False},
    )
    # gate_passed 1000 -> 500 (Phase-2 loop iter1). With decimation=12 (dt=0.03) the weight
    # dt-scales to the per-pass spike: 1000 -> +30/pass, now 500 -> +15/pass. The completed 5M run
    # (2026-06-09_00-11-51) showed return/scale inflating to ~50 and imag/value collapsing to -28
    # at scale; the audit traced return/scale directly to the raw +30 spike in return space (the
    # ReturnEMA p95-p5 spread). 200k snapshots are pre-symptom (return/scale ~9 there), so the
    # halving is justified by the full-run trajectory, not the early snapshot. +15/pass is still
    # the dominant positive (~15x a strong dense step) without stretching the return distribution
    # as hard. Predicted effect: return/scale settles toward ~25 at scale, less actor destabilisation.
    gate_passed = RewTerm(
        func=mdp.gate_passed,
        weight=500.0,
        params={"command_name": "target", "penalize_miss": False},
    )
    lookat_next = RewTerm(func=mdp.lookat_next_gate, weight=0.5, params={"command_name": "target", "std": 0.5})
    # gate_offset_penalty DISABLED after a controlled A/B at 5M. Enabled at -2 (run
    # 2026-06-12_01-37-34, fixed env): stochastic pass rate fell 73% -> 50%, max chain 4 -> 3,
    # imag/value went negative again (-8..-16 vs +2..+5.8) while the env bug fixes in the same
    # run verifiably removed only spurious deaths — isolating the regression to this term. Same
    # parking-wall lesson as near_gate: ANY dense negative concentrated around the gate plane
    # disrupts the approach more than it improves threading. Keep weight 0; crossing precision
    # has to come from the +pass spike and (if needed) a TIGHTER pass bbox (gate_size), not a
    # plane-local penalty.
    gate_offset_penalty = RewTerm(
        func=mdp.gate_offset_penalty,
        weight=0.0,
        params={"command_name": "target", "near_plane_dist": 2.0},
    )
    # near_gate DISABLED (weight 0). This absolute-proximity term was the parking wall: it
    # rewards being CLOSE, but passing a gate switches the target to the next (far) gate, so the
    # reward collapses on passing — the dense landscape peaks just before the gate plane and
    # drops on the far side. Across the whole session (steep std=1.5, moderate std=3.0, weights
    # 5/20/25) the value-correct actor always learned to loiter at the approach and never cross
    # (final eval: 0/10 gates, drifts away, collision rate 1.0). Potential-based signed progress
    # above replaces it — no static spot pays, so loitering is not an optimum. Kept here at
    # weight 0 (not deleted) so a small proximity bonus can be re-introduced later if the pure
    # progress + gate_passed signal proves too sparse near the gate centre.
    near_gate = RewTerm(
        func=mdp.pos_error_tanh,
        weight=0.0,
        params={"command_name": "target", "std": 3.0},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # flyaway 15 -> 18 m (BUG FIX). flyaway measures distance to the TARGET gate, and the
    # gate3->gate4 segment is 15.21 m — longer than the old 15 m threshold. The moment the drone
    # passed gate 3 the command retargeted to gate 4, already beyond the limit, and the episode
    # terminated with the -30 penalty AS THE REWARD FOR PASSING. diagnose_terminations confirmed:
    # every death-within-3-steps-of-a-pass was a flyaway right after gate idx 2. Chains through
    # gate 3 were geometrically impossible. 18 m = longest segment (15.21) + pass overshoot +
    # spawn noise margin, while still bounding the negative-drift tail that the 15 m value was
    # introduced for (50 m originally).
    flyaway = DoneTerm(func=mdp.flyaway, params={"command_name": "target", "distance": 18.0})
    # Collision via grace-period wrapper (BUG FIX): the Sim 5.1 ContactSensor teleport-spawn
    # phantom is usually <10 N post-mass-fix but diagnose_terminations measured ~7% of episodes
    # dying at step<=3 with a mid-air step-1 collision — residual spikes above the 10 N
    # threshold. illegal_contact_grace ignores contacts during the first 3 control steps after
    # reset (a real crash cannot physically occur off a mid-air spawn that fast); after the
    # grace window it is identical to mdp.illegal_contact at 10 N.
    collision = DoneTerm(
        func=mdp.illegal_contact_grace,
        params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 10.0, "grace_steps": 3},
    )


@configclass
class NoCamObservationsCfg:
    """Observation specs for the ground-truth-only (no camera) variant."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations: full privileged ground-truth state (20-dim)."""

        position = ObsTerm(func=mdp.root_pos_w)
        attitude = ObsTerm(func=mdp.root_quat_w)
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True  # flat vector: 3+4+3+3+3+4 = 20-dim

    policy: PolicyCfg = PolicyCfg()
    critic: None = None  # shared network reads the same OBSERVATIONS


@configclass
class DroneRacerEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=4096, env_spacing=0.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # MDP settings
        self.events.reset_base = None
        self.commands.target.randomise_start = True

        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneRacerEnvCfg_PLAY(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=1, env_spacing=0.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # Disable push robot events
        self.events.push_robot = None

        # Enable RGB alongside segmentation so the FPV visualization window can show both
        self.scene.tiled_camera.data_types = ["rgb", "semantic_segmentation"]

        # Enable recording fpv footage
        # self.commands.target.record_fpv = True

        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneRacerEnvCfg_NoCam(ManagerBasedRLEnvCfg):
    """Training variant: ground-truth state for both actor and critic — no camera."""

    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=4096, env_spacing=0.0)
    observations: NoCamObservationsCfg = NoCamObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.events.reset_base = None
        self.commands.target.randomise_start = True

        # camera not needed — disable to save GPU memory and simulation time
        self.scene.tiled_camera = None

        self.decimation = 4
        self.episode_length_s = 20
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneRacerEnvCfg_NoCam_PLAY(ManagerBasedRLEnvCfg):
    """Play/inference variant: ground-truth state, no camera."""

    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=1, env_spacing=0.0)
    observations: NoCamObservationsCfg = NoCamObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.events.push_robot = None

        # Enable RGB + segmentation for FPV debug window (visualization only — not used as policy observations)
        self.scene.tiled_camera.data_types = ["rgb", "semantic_segmentation"]

        self.decimation = 4
        self.episode_length_s = 20
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


# ============================================================
# MonoRace observation configs
# ============================================================


@configclass
class MonoRaceObservationsCfg:
    """Observation specs for MonoRace: compact gate perception features + drone state.

    Policy (actor): 9-dim gate perception + 17-dim state = 26-dim total.
    Critic: privileged ground-truth state (same as ObservationsCfg.CriticCfg, 20-dim).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """26-dim actor observations: gate geometry (9) + lin_vel(3) + ang_vel(3) + quat(4) + action(4) + target(3)."""

        gate_perc = ObsTerm(
            func=mdp.gate_perception_features,
            params={"command_name": "target", "num_frames": 1, "seg_noise_prob": 0.0},
        )
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        attitude = ObsTerm(func=mdp.root_quat_w)
        actions = ObsTerm(func=mdp.last_action)
        target_pos = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True  # 9 + 3 + 3 + 4 + 4 + 3 = 26-dim

    @configclass
    class CriticCfg(ObsGroup):
        """20-dim critic observations: privileged ground-truth state (same as ObservationsCfg.CriticCfg)."""

        position = ObsTerm(func=mdp.root_pos_w)
        attitude = ObsTerm(func=mdp.root_quat_w)
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True  # 3 + 4 + 3 + 3 + 3 + 4 = 20-dim

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class MonoRaceObsFrameStackCfg:
    """Observation specs for MonoRace with K=3 frame stacking: 27 + 17 = 44-dim policy obs."""

    @configclass
    class PolicyCfg(ObsGroup):
        """44-dim actor observations: 3×gate perception (27) + state (17)."""

        gate_perc = ObsTerm(
            func=mdp.gate_perception_features,
            params={"command_name": "target", "num_frames": 3, "seg_noise_prob": 0.0},
        )
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        attitude = ObsTerm(func=mdp.root_quat_w)
        actions = ObsTerm(func=mdp.last_action)
        target_pos = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True  # 27 + 3 + 3 + 4 + 4 + 3 = 44-dim

    @configclass
    class CriticCfg(ObsGroup):
        """20-dim critic observations: same as MonoRaceObservationsCfg.CriticCfg."""

        position = ObsTerm(func=mdp.root_pos_w)
        attitude = ObsTerm(func=mdp.root_quat_w)
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True  # 20-dim

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================
# MonoRace reward config
# ============================================================


@configclass
class MonoRaceRewardsCfg(RewardsCfg):
    """Enhanced rewards for MonoRace: inherits base rewards, adds velocity alignment, smoothness, gate offset."""

    velocity_alignment = RewTerm(
        func=mdp.velocity_alignment,
        weight=0.2,
        params={"command_name": "target", "std": 0.5},
    )
    action_smoothness = RewTerm(
        func=mdp.action_smoothness,
        weight=-0.001,
        params={},
    )
    gate_offset_penalty = RewTerm(
        func=mdp.gate_offset_penalty,
        weight=-0.5,
        params={"command_name": "target", "near_plane_dist": 1.5},
    )


# ============================================================
# MonoRace env configs (training + play)
# ============================================================


@configclass
class DroneRacerEnvCfg_MonoRace(ManagerBasedRLEnvCfg):
    """Training variant: MonoRace perception pipeline, asymmetric AC, 512 envs (camera constraint)."""

    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=512, env_spacing=0.0)
    observations: MonoRaceObservationsCfg = MonoRaceObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: MonoRaceRewardsCfg = MonoRaceRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.events.reset_base = None
        self.commands.target.randomise_start = True
        self.decimation = 4
        self.episode_length_s = 20
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneRacerEnvCfg_MonoRace_PLAY(ManagerBasedRLEnvCfg):
    """Inference variant: 1 env, RGB + segmentation for FPV visualization."""

    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=1, env_spacing=0.0)
    observations: MonoRaceObservationsCfg = MonoRaceObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: MonoRaceRewardsCfg = MonoRaceRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.events.push_robot = None
        # Enable RGB alongside segmentation for FPV debug window
        self.scene.tiled_camera.data_types = ["rgb", "semantic_segmentation"]
        self.decimation = 4
        self.episode_length_s = 20
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


# ============================================================
# DreamerV3 env configs
# ============================================================


@configclass
class DreamerObservationsCfg:
    """Observations for DreamerV3 training.

    PolicyCfg contains only the 10-dim kinematics state vector.
    Image data is extracted directly from the scene by DreamerIsaacEnvWrapper —
    NOT through the Isaac Lab observation manager — so no image ObsTerm is defined here.
    No CriticCfg: DreamerV3 does not use asymmetric actor-critic.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        attitude = ObsTerm(func=mdp.root_quat_w)
        target_pos = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DroneRacerEnvCfg_Dreamer(ManagerBasedRLEnvCfg):
    """DreamerV3 training variant: 32 envs, RGB + semantic segmentation enabled.

    32 envs is a conservative default for 6 GB VRAM when both RGB rendering
    and segmentation are active simultaneously.
    """

    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=32, env_spacing=0.0)
    observations: DreamerObservationsCfg = DreamerObservationsCfg()
    actions: DreamerActionsCfg = DreamerActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.events.reset_base = None
        # randomise_start=True now that the gate-chain pipeline has been verified end-to-end:
        # the 446k-step run produced four valid [GATE-PASS] events with next_gate_idx advancing
        # correctly and reward ≈ +100 each. Switching back to random starting gates gives the
        # actor coverage across all 7 gate-to-gate transitions on the track instead of only
        # the gate0→gate1 segment, which it needs to learn full lap navigation.
        self.commands.target.randomise_start = True
        # EXTREME curriculum: spawn 90% of the way from the previous gate toward the NEXT gate
        # (spawn_lerp_alpha 0.6 -> 0.9). A 3.8M-step run with 0.6 + signed progress never passed
        # a gate — the drone flew ~3 s, drifted away from the target (net-negative progress, value
        # collapsed to ~-21), and the +100 gate spike never entered its experience. At 0.9 the
        # drone spawns essentially AT the gate, facing through it with a forward velocity bias, so
        # it should pass within a step or two and the gate_passed reward floods the data — the
        # actor finally learns passing is hugely positive. This is also DIAGNOSTIC: if the drone
        # still can't pass when spawned right at the gate, the problem is control/algorithm, not
        # reward shaping. Anneal back down once gate_pass_rate is consistently > ~5%.
        self.commands.target.spawn_lerp_alpha = 0.9
        # Both RGB and segmentation must be available for all three obs modes
        self.scene.tiled_camera.data_types = ["rgb", "semantic_segmentation"]
        # decimation 4 -> 12: the actor's DECISION rate drops from 100 Hz to ~33 Hz (control dt
        # 0.01 s -> 0.03 s). The inner CTBR PD rate loop still runs at the 400 Hz physics rate
        # (it tracks the held setpoint every substep), so attitude control is unaffected — only
        # how often the actor re-decides. This makes the imagination horizon cover real flight
        # time: imag_horizon=15 * 0.03 s = 0.45 s (was 10 * 0.01 = 0.1 s), and a seq_len=64
        # training sequence now spans ~1.9 s — enough to contain an inter-gate approach. The
        # 0.1 s planning horizon was far shorter than the 1-2 s gate-to-gate flight, so the actor
        # could never imagine reaching a gate and the bootstrapped value collapsed on the
        # net-negative dense reward. This is the structural fix after reward shaping plateaued.
        self.decimation = 12
        self.episode_length_s = 20
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneRacerEnvCfg_Dreamer_PLAY(ManagerBasedRLEnvCfg):
    """DreamerV3 evaluation variant: 1 env."""

    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=1, env_spacing=0.0)
    observations: DreamerObservationsCfg = DreamerObservationsCfg()
    actions: DreamerActionsCfg = DreamerActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.events.push_robot = None
        self.scene.tiled_camera.data_types = ["rgb", "semantic_segmentation"]
        # debug_vis stays OFF. The gate/drone coordinate-frame VisualizationMarkers are real USD
        # prims, so the TiledCamera renders them INTO the observation — they appeared in the FPV
        # and made eval see an image that training never did (markers off in the training cfg).
        # Keeping it off means eval sees the same clean image as training. (To watch the target
        # gate live, read the [GATE-PASS] prints or use a separate viewport marker, not one the
        # drone camera can see.)
        self.commands.target.debug_vis = False
        # CRITICAL: eval MUST match the training spawn + control timescale, or the policy is
        # tested out-of-distribution and looks broken even when it isn't.
        #   - reset_base=None + randomise_start=True + curriculum spawn (was: fixed far-corner
        #     reset_base + randomise_start=None, so eval dropped the drone at a corner targeting
        #     gate 0 — an init the policy never trained on -> 0 gates in eval despite ~30% in
        #     training).
        #   - spawn_lerp_alpha=0.2 matches where the training curriculum ended (full-segment
        #     approach). Raise toward 0.9 to eval the easier near-gate starts the policy also saw.
        #   - decimation=12 matches training (was 4 -> the policy re-decided every 0.01 s in eval
        #     vs 0.03 s in training; the action timescale mismatch alone breaks behaviour).
        self.events.reset_base = None
        self.commands.target.randomise_start = True
        self.commands.target.spawn_lerp_alpha = 0.2
        self.decimation = 12
        self.episode_length_s = 20
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


# ============================================================
# CTBR-action variants for skrl-PPO sanity test
# ============================================================
# Inherit the existing skrl/PPO obs configs but swap ActionsCfg → DreamerActionsCfg so the
# action term is CTBRActionCfg (collective + body rates). action_dim stays 4. Reuses the
# existing skrl_cfg.yaml / skrl_cfg_nocam.yaml — same PPO network shapes.


@configclass
class DroneRacerEnvCfg_CTBR(DroneRacerEnvCfg):
    """Camera + IMU asymmetric AC, CTBR action."""

    actions: DreamerActionsCfg = DreamerActionsCfg()


@configclass
class DroneRacerEnvCfg_CTBR_PLAY(DroneRacerEnvCfg_PLAY):
    """Camera + IMU eval, CTBR action."""

    actions: DreamerActionsCfg = DreamerActionsCfg()


@configclass
class DroneRacerEnvCfg_NoCam_CTBR(DroneRacerEnvCfg_NoCam):
    """Ground-truth-only training, CTBR action — fastest sanity test."""

    actions: DreamerActionsCfg = DreamerActionsCfg()


@configclass
class DroneRacerEnvCfg_NoCam_CTBR_PLAY(DroneRacerEnvCfg_NoCam_PLAY):
    """Ground-truth-only eval, CTBR action."""

    actions: DreamerActionsCfg = DreamerActionsCfg()


# ============================================================
# UPSTREAM-COMPATIBLE LEGACY CFGS for skrl-PPO tasks
# ============================================================
# We aggressively re-tuned rewards/curriculum for Dreamer (see DroneRacerEnvCfg_Dreamer
# above). Those changes break the upstream skrl-PPO baseline, which was tuned with
# different reward magnitudes (terminating=-500, gate_passed=400, push events on, spawn
# at prev_gate +1m). Restore upstream values here so legacy PPO tasks behave as they did
# before our Dreamer work.


@configclass
class LegacyEventCfg(EventCfg):
    """Upstream EventCfg: push_robot interval events re-enabled for domain randomization."""

    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 0.2),
        params={
            "force_range": (-0.1, 0.1),
            "torque_range": (-0.05, 0.05),
        },
    )


@configclass
class LegacyCommandsCfg:
    """Upstream-style spawn: at prev_gate +1m forward, no lerp, no velocity bias."""

    target = mdp.GateTargetingCommandCfg(
        asset_name="robot",
        track_name="track",
        randomise_start=None,
        record_fpv=False,
        resampling_time_range=(1e9, 1e9),
        debug_vis=False,
        spawn_lerp_alpha=0.0,           # at prev gate
        spawn_forward_offset=1.0,       # +1 m along prev_gate forward axis
        spawn_forward_velocity=0.0,     # no initial velocity
    )


@configclass
class LegacyTerminationsCfg:
    """Upstream-style terminations: flyaway distance 20 m (was widened to 50 for Dreamer)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flyaway = DoneTerm(func=mdp.flyaway, params={"command_name": "target", "distance": 20.0})
    collision = DoneTerm(
        func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 10.0}
    )


@configclass
class LegacyRewardsCfg:
    """Upstream reward weights AND function behavior (symmetric progress + miss penalty)."""

    terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.0001)
    # asymmetric=False → signed progress (negative for retreat). PPO needs this gradient
    # to push policy away from bad actions; without it std explodes.
    progress = RewTerm(
        func=mdp.progress,
        weight=20.0,
        params={"command_name": "target", "asymmetric": False},
    )
    # penalize_miss=True → upstream behavior: +1 pass / -1 miss.
    gate_passed = RewTerm(
        func=mdp.gate_passed,
        weight=400.0,
        params={"command_name": "target", "penalize_miss": True},
    )
    lookat_next = RewTerm(func=mdp.lookat_next_gate, weight=0.1, params={"command_name": "target", "std": 0.5})


@configclass
class DroneRacerEnvCfg_NoCam_Legacy(DroneRacerEnvCfg_NoCam):
    """Legacy upstream-tuned NoCam motor-omega training — restores PPO convergence behavior."""

    commands: LegacyCommandsCfg = LegacyCommandsCfg()
    events: LegacyEventCfg = LegacyEventCfg()
    rewards: LegacyRewardsCfg = LegacyRewardsCfg()
    terminations: LegacyTerminationsCfg = LegacyTerminationsCfg()


@configclass
class DroneRacerEnvCfg_NoCam_Legacy_PLAY(DroneRacerEnvCfg_NoCam_PLAY):
    commands: LegacyCommandsCfg = LegacyCommandsCfg()
    events: LegacyEventCfg = LegacyEventCfg()
    rewards: LegacyRewardsCfg = LegacyRewardsCfg()
    terminations: LegacyTerminationsCfg = LegacyTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Upstream PLAY behavior: no camera, no IMU. Lets the user run without --enable_cameras.
        self.scene.tiled_camera = None
        self.scene.imu = None
        # CRITICAL: enable randomise_start so _resample_command calls reset_after_prev_gate
        # on every episode reset. Without this, after a crash the drone stays in the crashed
        # pose forever (next_gate_idx gets set to 0 but no pose write happens).
        self.commands.target.randomise_start = True
        # Disable the default fixed-corner reset_base — randomise_start handles repositioning
        # to a random gate, which matches training and gives the policy a real chance to fly.
        self.events.reset_base = None


@configclass
class DroneRacerEnvCfg_NoCam_CTBR_Legacy(DroneRacerEnvCfg_NoCam_CTBR):
    """Legacy upstream-tuned + CTBR action — true CTBR vs PPO validation."""

    commands: LegacyCommandsCfg = LegacyCommandsCfg()
    events: LegacyEventCfg = LegacyEventCfg()
    rewards: LegacyRewardsCfg = LegacyRewardsCfg()
    terminations: LegacyTerminationsCfg = LegacyTerminationsCfg()


@configclass
class DroneRacerEnvCfg_NoCam_CTBR_Legacy_PLAY(DroneRacerEnvCfg_NoCam_CTBR_PLAY):
    commands: LegacyCommandsCfg = LegacyCommandsCfg()
    events: LegacyEventCfg = LegacyEventCfg()
    rewards: LegacyRewardsCfg = LegacyRewardsCfg()
    terminations: LegacyTerminationsCfg = LegacyTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.tiled_camera = None
        self.scene.imu = None
        # Same respawn fix as the motor-omega Legacy_PLAY.
        self.commands.target.randomise_start = True
        self.events.reset_base = None
