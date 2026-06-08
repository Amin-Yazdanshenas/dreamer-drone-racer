"""DreamerIsaacEnvWrapper — bridges Isaac Lab gymnasium env to DreamerV3 dict-obs protocol."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import isaaclab.utils.math as math_utils
import torch


# Module-level gate label → class ID cache (populated once from tiled_camera info)
_GATE_LABEL_TO_CLASS_ID: Dict[str, int] = {}


class DreamerIsaacEnvWrapper:
    """Wraps a raw Isaac Lab gymnasium env to produce dict observations for DreamerV3.

    Does NOT use SkrlVecEnvWrapper. Accesses camera and robot data directly from the
    Isaac Lab scene to build separate image and state tensors.

    obs_mode options:
        "rgb"      : (N, H, W, 3) uint8 RGB image
        "mask"     : (N, H, W, 1) uint8 binary gate mask (0 or 255)
        "rgb_mask" : (N, H, W, 4) uint8 RGB + gate mask concatenated

    State vector (16-dim float32):
        ang_vel_b (3) + quat_w (4) + lin_vel_b (3) + target_pos_b (3) + gate_normal_b (3)

    Returned obs dict keys:
        image    : (N, H, W, C) uint8
        state    : (N, 13) float32
        reward   : (N,) float32  — only after step(), not after reset()
        is_first : (N,) bool
        is_last  : (N,) bool     — True on terminated or truncated
        is_terminal : (N,) bool  — True on terminated (crash / flyaway)
    """

    def __init__(self, gym_env, obs_mode: str = "rgb", command_name: str = "target",
                 gatenet_ckpt: Optional[str] = None):
        self.env = gym_env
        self.obs_mode = obs_mode
        self.command_name = command_name
        self._isaac = gym_env.unwrapped
        self.num_envs: int = self._isaac.num_envs
        self._is_first: torch.Tensor = torch.ones(self.num_envs, dtype=torch.bool,
                                                   device="cpu")

        # Optional GateNet for vision-based mask prediction (SkyDreamer Appendix A).
        # When provided, mask is predicted from RGB instead of read from Isaac Sim's
        # privileged semantic_segmentation channel — tests the sim-to-real path. The
        # GateNet is loaded once, kept on the same device as the camera output, and
        # run with eval()+no_grad() inside _extract_obs.
        self._gatenet = None
        if gatenet_ckpt is not None:
            from .gatenet import GateNet  # local import — avoid forcing torch on bare imports
            print(f"[DreamerIsaacEnvWrapper] loading GateNet from {gatenet_ckpt}")
            ckpt = torch.load(gatenet_ckpt, map_location="cpu")
            f = int(ckpt.get("f", 1))
            in_ch = int(ckpt.get("in_channels", 3))
            self._gatenet = GateNet(in_channels=in_ch, f=f)
            self._gatenet.load_state_dict(ckpt["model"])
            self._gatenet.eval()
            # Move to GPU once we know where the camera output lives.
            cam = self._isaac.scene["tiled_camera"]
            seg_or_rgb = cam.data.output.get("rgb") or cam.data.output.get("semantic_segmentation")
            if seg_or_rgb is not None:
                self._gatenet = self._gatenet.to(seg_or_rgb.device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, torch.Tensor]:
        self.env.reset()
        # Clear stale gate accumulator from the previous episode
        self._isaac.command_manager.get_term(self.command_name).reset_step_accumulators()
        # self._is_first governs what the NEXT step() call returns; reset obs is already marked
        # is_first=True below. Without zeroing here, the first step() would also return is_first=True
        # (double reset), causing the RSSM to discard the first step's temporal context every episode.
        self._is_first = torch.zeros(self.num_envs, dtype=torch.bool, device="cpu")
        obs = self._extract_obs()
        obs["reward"] = torch.zeros(self.num_envs, device="cpu")
        obs["is_first"] = torch.ones(self.num_envs, dtype=torch.bool, device="cpu")
        obs["is_last"] = torch.zeros(self.num_envs, dtype=torch.bool, device="cpu")
        obs["is_terminal"] = torch.zeros(self.num_envs, dtype=torch.bool, device="cpu")
        return obs

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Step the env with (N, 4) actions. Returns full obs dict."""
        # Reset sticky gate accumulators before new physics sub-steps so that
        # gate passes at ANY sub-step are captured (not just the last one).
        cmd = self._isaac.command_manager.get_term(self.command_name)
        cmd.reset_step_accumulators()
        _, rew, terminated, truncated, _ = self.env.step(actions)

        is_last = (terminated | truncated).cpu()
        is_terminal = terminated.cpu()
        rew_cpu = rew.cpu().float() if isinstance(rew, torch.Tensor) else torch.tensor(rew, dtype=torch.float32)

        # Gate pass signal: read directly from command manager (updated inside env.step)
        gate_passed = self._isaac.command_manager.get_term(
            self.command_name
        ).gate_passed.cpu()  # (N,) bool

        obs = self._extract_obs()
        obs["reward"] = rew_cpu
        obs["gate_passed"] = gate_passed
        # is_first for THIS frame = is_last of THIS step. Isaac Lab auto-resets in-step, so the
        # obs returned here is already the next episode's first frame whenever is_last is True;
        # it must therefore carry is_first=True NOW. The previous code carried the PREVIOUS
        # step's flag forward (self._is_first), landing is_first=True one frame too late — on the
        # new episode's SECOND frame — so the RSSM reset never fired on the true first frame and
        # leaked the dying episode's deter/action into the new episode (review I1).
        obs["is_first"] = is_last.clone()
        obs["is_last"] = is_last
        obs["is_terminal"] = is_terminal
        return obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_obs(self) -> Dict[str, torch.Tensor]:
        isaac = self._isaac
        robot = isaac.scene["robot"]
        cam = isaac.scene["tiled_camera"]

        # --- Image ---
        need_rgb = self.obs_mode in ("rgb", "rgb_mask") or self._gatenet is not None
        if need_rgb:
            rgb_raw = cam.data.output.get("rgb")  # (N, H, W, 4) RGBA uint8 or float
            rgb_u8 = _to_rgb_u8(rgb_raw)          # (N, H, W, 3) uint8

        if self.obs_mode in ("mask", "rgb_mask"):
            if self._gatenet is not None:
                # Vision-predicted mask — no privileged segmentation. Drone sees a noisy
                # all-gates mask exactly like the SkyDreamer deployment pipeline.
                mask_u8 = _gatenet_predict_mask_u8(self._gatenet, rgb_u8)  # (N, H, W, 1)
            else:
                seg = cam.data.output.get("semantic_segmentation")  # (N, H, W, 4) uint8
                mask_u8 = _extract_gate_mask_u8(seg, isaac, self.command_name)  # (N, H, W, 1)

        if self.obs_mode == "rgb":
            image = rgb_u8
        elif self.obs_mode == "mask":
            image = mask_u8
        else:
            image = torch.cat([rgb_u8, mask_u8], dim=-1)   # (N, H, W, 4)

        # --- State: ang_vel(3) + quat(4) + lin_vel(3) + target_pos_b(3) + gate_normal_b(3) = 16 ---
        ang_vel = robot.data.root_ang_vel_b.cpu()           # (N, 3)
        quat = robot.data.root_quat_w.cpu()                 # (N, 4)
        lin_vel = robot.data.root_lin_vel_b.cpu()           # (N, 3)
        target_pb = _compute_target_pos_b(robot, isaac, self.command_name).cpu()  # (N, 3)
        # gate_normal_b: the gate's through-direction (local +x) expressed in the drone body
        # frame. target_pos_b alone tells the actor WHERE the gate centre is, but not which way to
        # CROSS it — for racing the correct action is to go through along the gate normal from the
        # right side, not just reach the centre. Giving the actor the gate facing closes that gap.
        gate_nb = _compute_gate_normal_b(robot, isaac, self.command_name).cpu()   # (N, 3)
        state = torch.cat([ang_vel, quat, lin_vel, target_pb, gate_nb], dim=-1).float()

        # --- Privileged state for Informed-Dreamer decoder (12-dim) ---
        # Ground-truth signals the actor does NOT see but the WM decoder reconstructs.
        # Forces the latent to encode drone pose + velocity in both world and next-gate frame.
        priv = _compute_priv_state(robot, isaac, self.command_name).cpu()  # (N, 12)

        return {"image": image.cpu(), "state": state, "priv_state": priv}

    # ------------------------------------------------------------------
    # Passthroughs
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.env.close()

    @property
    def step_dt(self) -> float:
        try:
            return self.env.step_dt
        except AttributeError:
            return self._isaac.step_dt


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_rgb_u8(rgb_raw: torch.Tensor) -> torch.Tensor:
    """Convert TiledCamera rgb output to (N, H, W, 3) uint8.

    The camera may output RGBA uint8 or float32 in [0, 1].
    """
    if rgb_raw is None:
        raise RuntimeError("TiledCamera has no 'rgb' output. Check data_types config.")
    x = rgb_raw.cpu()
    # Drop alpha if present
    x = x[..., :3]
    if x.dtype == torch.uint8:
        return x
    # Float → uint8
    return (x.clamp(0, 1) * 255).to(torch.uint8)


def _extract_gate_mask_u8(seg: Optional[torch.Tensor],
                          isaac_env, command_name: str) -> torch.Tensor:
    """Return (N, H, W, 1) uint8 gate mask where 255 = target gate pixel, 0 = background."""
    if seg is None:
        N = isaac_env.num_envs
        cam = isaac_env.scene["tiled_camera"]
        H = cam.image_shape[0]
        W = cam.image_shape[1]
        return torch.zeros(N, H, W, 1, dtype=torch.uint8)

    global _GATE_LABEL_TO_CLASS_ID
    if not _GATE_LABEL_TO_CLASS_ID:
        cam = isaac_env.scene["tiled_camera"]
        info = cam.data.info.get("semantic_segmentation", {})
        id_to_labels = info.get("idToLabels", {})
        if id_to_labels:
            _GATE_LABEL_TO_CLASS_ID = {
                v["class"]: int(k)
                for k, v in id_to_labels.items()
                if isinstance(v, dict) and "class" in v
            }

    target_idx = isaac_env.command_manager.get_term(command_name).next_gate_idx  # (N,)
    N = target_idx.shape[0]
    device = seg.device

    if _GATE_LABEL_TO_CLASS_ID:
        class_ids = torch.tensor(
            [_GATE_LABEL_TO_CLASS_ID.get(f"gate_{int(i.item()) + 1}", int(i.item()) + 1)
             for i in target_idx],
            dtype=seg.dtype, device=device,
        )
    else:
        class_ids = (target_idx + 1).to(dtype=seg.dtype, device=device)

    # seg[..., 0] is the low byte of the class ID
    binary = (seg[..., 0] == class_ids[:, None, None])    # (N, H, W) bool
    mask_u8 = (binary.float() * 255).to(torch.uint8)      # (N, H, W) uint8
    return mask_u8.unsqueeze(-1).cpu()                     # (N, H, W, 1)


@torch.no_grad()
def _gatenet_predict_mask_u8(gatenet, rgb_u8: torch.Tensor) -> torch.Tensor:
    """Run GateNet on RGB to produce (N, H, W, 1) uint8 binary gate mask.

    rgb_u8: (N, H, W, 3) uint8 on any device.
    Returns: (N, H, W, 1) uint8 on CPU (0 or 255).

    SkyDreamer convention: any-gate mask (all gates merged). Differs from the
    Isaac Sim path which filters to the target gate via class_id lookup.
    """
    device = next(gatenet.parameters()).device
    x = rgb_u8.to(device).float() / 255.0          # (N, H, W, 3)
    x = x.permute(0, 3, 1, 2).contiguous()          # (N, 3, H, W)
    outputs = gatenet(x)                            # list of multi-scale logits
    prob = torch.sigmoid(outputs[0])                # full-res, (N, 1, H, W)
    mask = (prob > 0.5).to(torch.uint8) * 255       # binary 0/255
    return mask.permute(0, 2, 3, 1).contiguous().cpu()  # (N, H, W, 1) uint8


def _compute_target_pos_b(robot, isaac_env, command_name: str) -> torch.Tensor:
    """Compute target gate centre in drone body frame. Returns (N, 3)."""
    target_pos_w = isaac_env.command_manager.get_term(command_name).command[:, :3]
    pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w
    )
    return pos_b


def _compute_gate_normal_b(robot, isaac_env, command_name: str) -> torch.Tensor:
    """Compute the target gate's through-direction (gate local +x) in the drone body frame.

    Returns (N, 3) unit-ish vector. Gives the actor the gate's facing/orientation, which
    target_pos_b (centre direction) does not encode.
    """
    cmd = isaac_env.command_manager.get_term(command_name)
    gate_quat_w = cmd.command[:, 3:7]                                  # (N, 4)
    x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=gate_quat_w.device).expand(gate_quat_w.shape[0], 3)
    gate_x_w = math_utils.quat_apply(gate_quat_w, x_axis)             # gate normal in world
    gate_x_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, gate_x_w)  # in body frame
    return gate_x_b


def _compute_priv_state(robot, isaac_env, command_name: str) -> torch.Tensor:
    """Privileged ground-truth state for Informed-Dreamer decoder.

    Layout (12-dim, all float32):
        [0:3]   p_w  drone position in world frame
        [3:6]   v_w  drone linear velocity in world frame
        [6:9]   p_g  drone position in next-gate frame (origin = gate centre, axes = gate)
        [9:12]  v_g  drone linear velocity in next-gate frame

    Symlog-normalised inside the agent's decoder loss (positions can reach ~10 m, velocities
    ~20 m/s — symlog squashes both to similar magnitude).
    """
    cmd = isaac_env.command_manager.get_term(command_name)
    gate_pose = cmd.command  # (N, 7) — [pos_w, quat_w] of next gate
    gate_pos_w = gate_pose[:, :3]
    gate_quat_w = gate_pose[:, 3:7]

    pos_w = robot.data.root_pos_w        # (N, 3)
    vel_w = robot.data.root_lin_vel_w    # (N, 3)

    # Drone in gate frame: translate then rotate by gate_quat^{-1}
    p_g, _ = math_utils.subtract_frame_transforms(gate_pos_w, gate_quat_w, pos_w)
    # Velocity is a free vector — only rotate
    v_g = math_utils.quat_apply_inverse(gate_quat_w, vel_w)

    return torch.cat([pos_w, vel_w, p_g, v_g], dim=-1).float()  # (N, 12)
