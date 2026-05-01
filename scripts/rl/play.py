# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--renderer",
    type=str,
    default="RasterOnly",
    choices=["RasterOnly", "RayTracedLighting", "PathTracing"],
    help="Renderer to use. RasterOnly avoids RTX AccelStruct VRAM exhaustion on 6 GB GPUs.",
)
parser.add_argument("--log", type=int, default=None, help="Log the observations and metrics.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Belt-and-suspenders DLSS disable. Renderer is already set via --renderer (AppLauncher → SimulationApp).
try:
    import carb

    carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)
except Exception:
    pass

"""Rest everything follows."""

import os
import time

import cv2
import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)

import tasks  # noqa: F401
from utils.logger import CSVLogger

# config shortcuts
algorithm = args_cli.algorithm.lower()

_DISP = 256  # display size per panel (pixels)


def _tensor_to_rgb_u8(rgb_t: torch.Tensor) -> np.ndarray:
    """TiledCamera rgb is uint8 0–255 or float in [0, 1] depending on pipeline; return H×W×3 uint8 RGB."""
    x = rgb_t.detach().cpu().float().numpy()
    if x.ndim == 3 and x.shape[2] >= 3:
        x = x[..., :3]
    mx = float(x.max()) if x.size else 0.0
    if mx <= 1.0 + 1e-3:
        u8 = (x * 255.0).clip(0, 255).astype(np.uint8)
    else:
        u8 = x.clip(0, 255).astype(np.uint8)
    return u8


def _show_fpv(camera, target_class_id: int | None = None) -> None:
    """Debug FPV window: RGB (left) | segmentation mask (right).

    Mask colors: green = target gate, red = other visible gates, black = background.
    Matches the training observation — only the target gate is in the policy's mask.
    """
    out = camera.data.output
    rgb_raw = out.get("rgb")
    if rgb_raw is None and "rgba" in out:
        rgb_raw = out["rgba"][..., :3]
    seg_raw = out.get("semantic_segmentation")
    if rgb_raw is None or seg_raw is None:
        return

    rgb_u8 = _tensor_to_rgb_u8(rgb_raw[0])
    rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    rgb_bgr = cv2.resize(rgb_bgr, (_DISP, _DISP), interpolation=cv2.INTER_NEAREST)

    class_ch = seg_raw[0].detach().cpu().numpy()[..., 0]  # (H, W) low byte of class ID
    mask_bgr = np.zeros((*class_ch.shape, 3), dtype=np.uint8)
    if target_class_id is not None:
        mask_bgr[class_ch == target_class_id] = (0, 255, 0)                             # target → green
        mask_bgr[(class_ch > 0) & (class_ch != target_class_id)] = (0, 0, 255)          # other gates → red
    else:
        mask_bgr[class_ch > 0] = (0, 255, 0)
    mask_bgr = cv2.resize(mask_bgr, (_DISP, _DISP), interpolation=cv2.INTER_NEAREST)

    combined = np.hstack([rgb_bgr, mask_bgr])
    try:
        cv2.imshow("FPV  |  RGB (left)   target=green  other=red (right)", combined)
        cv2.waitKey(1)
    except cv2.error:
        pass  # no display available (headless) — skip


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.log and args_cli.num_envs > 1:
        raise ValueError("Logging is only supported for a single agent. Set --num_envs to 1.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        print("[INFO] Pre-trained checkpoint download is not supported in this version of isaaclab.")
        return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        try:
            resume_path = get_checkpoint_path(
                log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
            )
        except Exception:
            resume_path = None
            print("[INFO] No checkpoint found — running with randomly initialized weights.")
    log_dir = os.path.dirname(os.path.dirname(resume_path)) if resume_path else log_root_path

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Second viewport showing the drone's onboard camera (GUI mode only)
    if not args_cli.headless:
        try:
            from omni.kit.viewport.utility import create_viewport_window
            from pxr import Sdf

            _fpv_vp = create_viewport_window(
                name="Drone FPV Camera", width=512, height=512, position_x=0, position_y=0
            )
            _fpv_vp.viewport_api.camera_path = Sdf.Path("/World/envs/env_0/Robot/body/camera")
            print("[INFO] Second viewport 'Drone FPV Camera' created.")
        except Exception as e:
            print(f"[INFO] Could not create second viewport: {e}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.log:
        if not resume_path:
            raise ValueError("--log requires a checkpoint. Use --checkpoint to specify one.")
        logger = CSVLogger(log_dir)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # Grab tiled camera for FPV (skrl wrapper forwards .scene to the Isaac Lab env)
    fpv_camera = None
    try:
        fpv_camera = env.scene["tiled_camera"]
    except (AttributeError, KeyError) as exc:
        print(
            f"[WARN] Could not access scene['tiled_camera'] for FPV overlay ({exc!r}). "
            "Use a camera task (e.g. Isaac-Drone-Racer-Play-v0) and --enable_cameras."
        )

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    from tasks.drone_racer.agents.cam_runner import CAM_TASKS, CamRunner

    if args_cli.task in CAM_TASKS:
        runner = CamRunner(env, experiment_cfg)
    else:
        runner = Runner(env, experiment_cfg)

    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
    # set agent to evaluation mode
    for model in runner.agent.models.values():
        if hasattr(model, "set_running_mode"):
            model.set_running_mode("eval")
        else:
            model.eval()

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    num_episode = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, rew, terminated, truncated, info = env.step(actions)

        if fpv_camera is not None:
            try:
                _target_cmd = env.unwrapped.command_manager.get_term("target")
                _t_class_id = int(_target_cmd.next_gate_idx[0].item()) + 1  # 0-indexed → 1-indexed class ID
            except Exception:
                _t_class_id = None
            _show_fpv(fpv_camera, target_class_id=_t_class_id)

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        if args_cli.log:
            if truncated or terminated:
                num_episode += 1
                logger.save()
                if num_episode >= args_cli.log:
                    break
            logger.log(info["metrics"])

    cv2.destroyAllWindows()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
