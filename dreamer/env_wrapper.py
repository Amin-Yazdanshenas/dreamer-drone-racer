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

    State vector (13-dim float32):
        ang_vel_b (3) + quat_w (4) + lin_vel_b (3) + target_pos_b (3)

    Returned obs dict keys:
        image    : (N, H, W, C) uint8
        state    : (N, 13) float32
        reward   : (N,) float32  — only after step(), not after reset()
        is_first : (N,) bool
        is_last  : (N,) bool     — True on terminated or truncated
        is_terminal : (N,) bool  — True on terminated (crash / flyaway)
    """

    def __init__(self, gym_env, obs_mode: str = "rgb", command_name: str = "target"):
        self.env = gym_env
        self.obs_mode = obs_mode
        self.command_name = command_name
        self._isaac = gym_env.unwrapped
        self.num_envs: int = self._isaac.num_envs
        self._is_first: torch.Tensor = torch.ones(self.num_envs, dtype=torch.bool,
                                                   device="cpu")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, torch.Tensor]:
        self.env.reset()
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
        _, rew, terminated, truncated, _ = self.env.step(actions)

        is_last = (terminated | truncated).cpu()
        is_terminal = terminated.cpu()
        rew_cpu = rew.cpu().float() if isinstance(rew, torch.Tensor) else torch.tensor(rew, dtype=torch.float32)

        obs = self._extract_obs()
        obs["reward"] = rew_cpu
        obs["is_first"] = self._is_first.clone()
        obs["is_last"] = is_last
        obs["is_terminal"] = is_terminal

        # Next step is_first = True only for envs that just ended
        self._is_first = is_last.clone()
        return obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_obs(self) -> Dict[str, torch.Tensor]:
        isaac = self._isaac
        robot = isaac.scene["robot"]
        cam = isaac.scene["tiled_camera"]

        # --- Image ---
        if self.obs_mode in ("rgb", "rgb_mask"):
            rgb_raw = cam.data.output.get("rgb")  # (N, H, W, 4) RGBA uint8 or float
            rgb_u8 = _to_rgb_u8(rgb_raw)          # (N, H, W, 3) uint8

        if self.obs_mode in ("mask", "rgb_mask"):
            seg = cam.data.output.get("semantic_segmentation")  # (N, H, W, 4) uint8
            mask_u8 = _extract_gate_mask_u8(seg, isaac, self.command_name)  # (N, H, W, 1)

        if self.obs_mode == "rgb":
            image = rgb_u8
        elif self.obs_mode == "mask":
            image = mask_u8
        else:
            image = torch.cat([rgb_u8, mask_u8], dim=-1)   # (N, H, W, 4)

        # --- State: ang_vel(3) + quat(4) + lin_vel(3) + target_pos_b(3) = 13 ---
        ang_vel = robot.data.root_ang_vel_b.cpu()           # (N, 3)
        quat = robot.data.root_quat_w.cpu()                 # (N, 4)
        lin_vel = robot.data.root_lin_vel_b.cpu()           # (N, 3)
        target_pb = _compute_target_pos_b(robot, isaac, self.command_name).cpu()  # (N, 3)
        state = torch.cat([ang_vel, quat, lin_vel, target_pb], dim=-1).float()

        return {"image": image.cpu(), "state": state}

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


def _compute_target_pos_b(robot, isaac_env, command_name: str) -> torch.Tensor:
    """Compute target gate centre in drone body frame. Returns (N, 3)."""
    target_pos_w = isaac_env.command_manager.get_term(command_name).command[:, :3]
    pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w
    )
    return pos_b
