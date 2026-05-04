![Isaac Drone Racer](media/motion_trace1.jpg)

---

# Isaac Drone Racer

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-silver.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/kousheekc/isaac_drone_racer/blob/master/.github/workflows/pre-commit.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Isaac Drone Racer** is an open-source simulation framework for autonomous drone racing, developed on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab). It is designed for training reinforcement learning policies in realistic racing environments, with a focus on accurate physics and modular design.

Autonomous drone racing is an active area of research. This project builds on insights from that body of work, combining them with massively parallel simulation to train racing policies within minutes — offering a fast and flexible platform for experimentation and benchmarking.

You can watch the related video here: [https://youtu.be/wLTYtpEUEEk](https://youtu.be/wLTYtpEUEEk)

## Features

Key highlights of the Isaac Drone Racer project:

1. **Accurate Physics Modeling** — Simulates rotor dynamics, aerodynamic drag, and power consumption to closely match real-world quadrotor behavior.
2. **Low-Level Flight Controller** — Built-in attitude and rate controllers.
3. **Manager-Based Design** — Modular architecture using IsaacLab's [manager based architecture](https://isaac-sim.github.io/IsaacLab/main/source/refs/reference_architecture/index.html#manager-based).
4. **Onboard Sensor Suite** — Includes simulated fisheye camera, IMU and collision detection.
5. **Track Generator** — Dynamically generate custom race tracks.
6. **Logger and Plotter** — Integrated tools for monitoring and visualizing flight behavior.
7. **Multiple RL training modes** — PPO (asymmetric AC, MonoRace compact state) and DreamerV3 world-model RL, all coexisting without conflict.

### Prerequisites
- Workstation capable of running Isaac Sim (see [link](https://github.com/isaac-sim/IsaacSim?tab=readme-ov-file#prerequisites-and-environment-setup))
- [Git](https://git-scm.com/downloads) & [Git LFS](https://git-lfs.com)
- [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install) for local installation or [Docker](https://docs.docker.com/engine/install/ubuntu/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Requirements
This project has been developed and tested with:

- **Isaac Sim 5.1**
- **Isaac Lab 2.3.2**
- **Python 3.10**
- **Ubuntu 22.04 (x64)**

## Setup

### Option A — Restore the exact conda environment (recommended)

An exported conda environment is provided in [environment.yml](environment.yml). This installs Isaac Sim 5.1, Isaac Lab 2.3.2, and all other pinned dependencies in one step, without needing to follow the upstream Isaac Lab installation guide manually.

```bash
# 1. Clone this repository
git clone https://github.com/Amin-Yazdanshenas/isaac_drone_racer.git
cd isaac_drone_racer

# 2. Create the conda environment from the lockfile (~10–20 min, downloads ~10 GB)
conda env create -f environment.yml

# 3. Activate it
conda activate isaacsim

# 4. Install the project modules in editable mode
pip3 install -e .
```

### Option B — Manual installation

1. Follow the [Isaac Lab pip installation instructions](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html), with the following modifications:
- After cloning the Isaac Lab repository:
```bash
git clone git@github.com:isaac-sim/IsaacLab.git
```

- Checkout the `v2.3.2` release tag:
```bash
cd IsaacLab
git checkout v2.3.2
```

2. Clone Isaac Drone Racer:
```bash
git clone https://github.com/Amin-Yazdanshenas/isaac_drone_racer.git
```

3. Install the modules in editable mode
```bash
cd isaac_drone_racer
pip3 install -e .
```

## Usage

Tasks are registered as standard Gym environments. Three distinct RL training approaches are available:

| Mode | Algorithm | Task IDs | Actor input | Notes |
|------|-----------|----------|-------------|-------|
| **Asymmetric AC** (camera) | PPO via skrl | `Isaac-Drone-Racer-v0` / `-Play-v0` | FPV 64×64 grayscale + IMU | Privileged GT critic |
| **Ground-truth only** | PPO via skrl | `Isaac-Drone-Racer-NoCam-v0` / `-NoCam-Play-v0` | Full GT state | Fastest; no camera |
| **MonoRace** (compact perception) | PPO via skrl | `Isaac-Drone-Racer-MonoRace-v0` / `-MonoRace-Play-v0` | 9-dim gate geometry + IMU | Inspired by MAVLab 2025 |
| **DreamerV3** (world model) | DreamerV3 | `Isaac-Drone-Racer-Dreamer-{RGB,Mask,RGBMask}-v0` / `-Dreamer-Play-v0` | Raw 64×64 image (RGB / mask / both) | Separate train/eval scripts |

---

### PPO — Asymmetric Actor-Critic (Camera + IMU)

Trains a deployable policy using only onboard sensors (FPV camera + IMU). The critic sees privileged ground-truth state during training only.

```bash
# Train
python3 scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --enable_cameras --num_envs 512

# Play (headless — works on ≤ 6 GB VRAM; OpenCV window shows RGB + gate mask)
python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --enable_cameras --headless --num_envs 1
```

Checkpoints: `logs/skrl/drone_racer/`

---

### PPO — Ground-Truth Only (No Camera)

Full simulator state as observations; camera disabled. Best for rapid iteration on rewards/dynamics.

```bash
# Train
python3 scripts/rl/train.py --task Isaac-Drone-Racer-NoCam-v0 --headless --num_envs 4096

# Play
python3 scripts/rl/play.py --task Isaac-Drone-Racer-NoCam-Play-v0 --num_envs 1
```

Checkpoints: `logs/skrl/drone_racer_nocam/`

---

### PPO — MonoRace Compact Perception

Inspired by the MAVLab 2025 MonoRace approach. Extracts 9 compact geometric features from the semantic segmentation mask (centroid, bounding box, area, pinhole distance estimate) and combines them with drone dynamics into a 26-dim state fed to a lightweight 37K-parameter MLP policy. More sample-efficient than raw pixel input.

```bash
# Train (512 envs — camera-constrained on 6 GB VRAM)
python3 scripts/rl/train.py --task Isaac-Drone-Racer-MonoRace-v0 --headless --enable_cameras --num_envs 512

# Play
python3 scripts/rl/play.py --task Isaac-Drone-Racer-MonoRace-Play-v0 --enable_cameras --headless --num_envs 1
```

Checkpoints: `logs/skrl/drone_racer_monorace/`

---

### DreamerV3 — World-Model RL

Inspired by [Dream to Fly (Romero et al., 2025)](https://arxiv.org/abs/2501.14377). A self-contained PyTorch DreamerV3 implementation learns a world model (RSSM) from 64×64 images, then trains an actor-critic entirely on *imagined* rollouts inside that world model. No skrl dependency — uses separate `train_dreamer.py` / `evaluate_dreamer.py` scripts.

**Three observation modes:**

| `--obs_mode` | Task ID | Image input | Channels |
|-------------|---------|-------------|---------|
| `rgb` | `Isaac-Drone-Racer-Dreamer-RGB-v0` | Raw RGB | 3 |
| `mask` | `Isaac-Drone-Racer-Dreamer-Mask-v0` | Binary gate mask | 1 |
| `rgb_mask` | `Isaac-Drone-Racer-Dreamer-RGBMask-v0` | RGB + gate mask | 4 |

All modes additionally receive a 10-dim kinematics state vector (ang_vel, quaternion, target gate position in body frame) through a separate MLP encoder fused into the RSSM.

```bash
# Train — RGB (closest to the Dream to Fly paper)
python3 scripts/rl/train_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-RGB-v0 \
    --obs_mode rgb --num_envs 32 --max_steps 2000000 \
    --headless --enable_cameras

# Train — binary gate mask only
python3 scripts/rl/train_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-Mask-v0 \
    --obs_mode mask --num_envs 32 --max_steps 2000000 \
    --headless --enable_cameras

# Train — RGB + mask (4-channel)
python3 scripts/rl/train_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-RGBMask-v0 \
    --obs_mode rgb_mask --num_envs 32 --max_steps 2000000 \
    --headless --enable_cameras

# Resume from checkpoint
python3 scripts/rl/train_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-RGB-v0 \
    --obs_mode rgb \
    --checkpoint logs/dreamer/rgb/<run>/checkpoints/agent_latest.pt \
    --headless --enable_cameras

# Evaluate / play (deterministic rollouts, saves eval_results.csv)
python3 scripts/rl/evaluate_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-Play-v0 \
    --obs_mode rgb \
    --checkpoint logs/dreamer/rgb/<run>/checkpoints/agent_latest.pt \
    --num_episodes 10 --headless --enable_cameras

# Monitor training
tensorboard --logdir logs/dreamer
```

Checkpoints: `logs/dreamer/<obs_mode>/<timestamp>/checkpoints/`

**Architecture (scaled for 6 GB VRAM):**
- RSSM: 512-dim recurrent state, 32×32 categorical stochastic latent
- All MLPs: 256 hidden units, 4 layers, LayerNorm + SiLU
- Imagination horizon: 15 steps
- Replay buffer: 500 K steps
- Warmup: 1 000 random steps before learning begins

> [!NOTE]
> You can pass additional CLI arguments supported by the [AppLauncher](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/launch_app.html). Additionally since IsaacLab supports the [Hydra Configuration System](https://isaac-sim.github.io/IsaacLab/main/source/features/hydra.html), task-specific parameters can be adjusted from CLI.
> For example, to disable the motor model during training:
> ```bash
> python3 scripts/rl/train.py --task Isaac-Drone-Racer-NoCam-v0 --headless --num_envs 4096 env.actions.control_action.use_motor_model=False
> ```

## Next Steps

- [ ] **Data-driven aerodynamic model pipeline** — integrate tools for data-driven system identification, calibration and include learned aerodynamic forces into the simulation environment.
- [ ] **Power consumption model** — incorporate a detailed power model that accounts for battery discharge based on current draw.
- [x] **Policy learning using onboard sensors** — asymmetric actor-critic training with FPV camera + IMU actor and privileged ground-truth critic.
- [x] **MonoRace compact perception** — compact gate geometry features (centroid, bbox, distance) replacing raw pixel input; ~37K-parameter MLP policy.
- [x] **DreamerV3 world-model RL** — self-contained PyTorch DreamerV3 with RGB / mask / RGB+mask observation modes; learns from imagined rollouts inside a learned world model.
- [ ] **Curriculum learning** — staged difficulty ramp (gate spacing, speed targets) for faster convergence.
- [ ] **Sim-to-real transfer** — domain randomisation of motor dynamics, aerodynamics, and camera noise for physical hardware deployment.


## Troubleshooting
- When running a workflow script, ensure that the IsaacLab conda environment is active:
```bash
conda activate env_isaaclab
```
- When launching Isaac Sim for the first time, it may take a significant amount of time to load (potentially 10 minutes). This is normal, please be patient.

### GPU VRAM requirements (GUI mode)

Isaac Sim 5.1 loads the full RTX renderer stack at startup when running in GUI mode (without `--headless`). This pre-allocates roughly 4–5 GB of VRAM for ray-tracing buffers, BVH acceleration structures, and the viewport framebuffer — before any simulation or PyTorch code runs. On GPUs with 6 GB or less (e.g. RTX 3060 Laptop), this exhausts all available VRAM and causes a `CUDA error: CUBLAS_STATUS_ALLOC_FAILED` crash during environment creation.

**Workaround: always use `--headless` on GPUs with ≤ 6 GB VRAM.** The tiled camera still renders correctly in headless mode. A debug visualization window (RGB + target gate mask overlay) is shown via OpenCV when `--enable_cameras` is set:

```bash
# Camera task — headless required on ≤ 6 GB GPUs
python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --enable_cameras --headless --num_envs 1
```

GUI mode (no `--headless`) requires a GPU with **≥ 8 GB VRAM**.

## Acknowledgement

- **Kaufmann, E., Bauersfeld, L., Loquercio, A., Müller, M., Koltun, V., & Scaramuzza, D.** (2023).
  *Champion-level drone racing using deep reinforcement learning*.
  [https://doi.org/10.1038/s41586-023-06419-4](https://doi.org/10.1038/s41586-023-06419-4)

- **Rudin, N., Hoeller, D., Reist, P., & Hutter, M.** (2022).
  *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*.
  [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)

- **Ferede, R., De Wagter, C., Izzo, D., & de Croon, G. C. H. E.** (2024).
  *End-to-end Reinforcement Learning for Time-Optimal Quadcopter Flight*.
  [https://doi.org/10.1109/ICRA57147.2024.10611665](https://doi.org/10.1109/ICRA57147.2024.10611665)

- **Romero, A., Shenai, A., Geles, I., Aljalbout, E., & Scaramuzza, D.** (2025).
  *Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight*.
  [arXiv:2501.14377](https://arxiv.org/abs/2501.14377)

- **Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J.** (2023).
  *Mastering Diverse Domains through World Models (DreamerV3)*.
  [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

## License
This project is licensed under the BSD 3-Clause License - see the [LICENSE](https://github.com/kousheekc/isaac_drone_racer/blob/master/LICENSE) file for details.

## Contact
Kousheek Chakraborty - kousheekc@gmail.com

Project Link: [https://github.com/Amin-Yazdanshenas/isaac_drone_racer](https://github.com/Amin-Yazdanshenas/isaac_drone_racer)

If you encounter any difficulties, feel free to reach out through the Issues section. If you find any bugs or have improvements to suggest, don't hesitate to make a pull request.
