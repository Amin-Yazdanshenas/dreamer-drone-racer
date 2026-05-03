# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Isaac Sim 5.1** + **Isaac Lab 2.3.2** + **Python 3.11** (conda env: `isaacsim`)
- Always activate the conda environment before running anything:
  ```bash
  conda activate isaacsim
  ```
- All scripts must be run from the repo root (`isaac_drone_racer/`).

## Commands

### Training
```bash
# Camera-based asymmetric actor-critic (requires --enable_cameras)
python3 scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --enable_cameras --num_envs 64

# Ground-truth only (faster, no camera)
python3 scripts/rl/train.py --task Isaac-Drone-Racer-NoCam-v0 --headless --num_envs 4096
```

### Play / Inference
```bash
python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --enable_cameras --num_envs 1
python3 scripts/rl/play.py --task Isaac-Drone-Racer-NoCam-Play-v0 --num_envs 1

# Play with CSV logging (single env only)
python3 scripts/rl/play.py --task Isaac-Drone-Racer-NoCam-Play-v0 --num_envs 1 --log 5
```

### Tests (no Isaac Sim required)
```bash
pytest tests/
```

### Linting
```bash
pre-commit run --all-files
```
Line length is **120**. Formatter is `black`, import sorter is `isort --profile black`.

## Architecture

### Task registration flow
`tasks/__init__.py` → imports `tasks/drone_racer/__init__.py` → registers 4 Gym IDs:
- `Isaac-Drone-Racer-v0` / `-Play-v0` — camera + IMU actor, ground-truth critic
- `Isaac-Drone-Racer-NoCam-v0` / `-NoCam-Play-v0` — ground-truth state only

Each Gym ID is bound to an env cfg class and a skrl YAML config in `tasks/drone_racer/agents/`.

### Env config (`drone_racer_env_cfg.py`)
Isaac Lab `@configclass` hierarchy. Key classes:
- `DroneRacerSceneCfg` — scene: ground, track, robot, sensors (IMU, collision, tiled camera)
- `ObservationsCfg` — **asymmetric AC**: `PolicyCfg` (camera grayscale 4096 + IMU 7 = 4103-dim) feeds `OBSERVATIONS`; `CriticCfg` (pos 3 + quat 4 + lin_vel 3 + ang_vel 3 + target_pos 3 + actions 4 = 20-dim) feeds `STATES`
- `NoCamObservationsCfg` — single `PolicyCfg` with 20-dim ground-truth; no critic group
- `DroneRacerEnvCfg` — training (4096 envs, both obs groups active, camera enabled)
- `DroneRacerEnvCfg_PLAY` — inference (critic obs kept active so skrl value network can init; push events disabled)
- `DroneRacerEnvCfg_NoCam` / `_NoCam_PLAY` — camera disabled in scene config

### MDP modules (`tasks/drone_racer/mdp/`)
All symbols are re-exported flat via `mdp/__init__.py` (which also re-exports `isaaclab.envs.mdp.*`).
- `actions.py` — `ControlAction`: maps 4 normalised actions → motor ω refs → `Allocation.compute()` → thrust + 3-axis moment applied as permanent wrench on body frame
- `commands.py` — `GateTargetingCommand`: tracks next gate index per env, detects gate passing via plane-crossing + bounding-box check, resets drone near the previous gate on episode reset, optionally records FPV video
- `observations.py` — `flat_image()` converts RGB→grayscale, flattens to (N, H×W); IMU and pose helpers; `target_pos_b()` returns target in body frame
- `rewards.py` / `terminations.py` / `events.py` — standard Isaac Lab term functions

### Dynamics (`dynamics/`)
Pure-PyTorch, no Isaac Sim dependency — usable in unit tests.
- `Allocation`: batched allocation matrix `(num_envs, 4, 4)` mapping `ω²` → `[Fz, Mx, My, Mz]`
- `Motor`: first-order lag model per motor with rate clamping; bypass with `use=False`

### CNN policy (`tasks/drone_racer/agents/`)
Camera tasks use `CamRunner(Runner)` which overrides `_generate_models` to inject:
- `CNNPolicy` (`GaussianMixin`): 4-layer conv (1→16→32→64→64) over 64×64 image → 1024 → Linear(64), cat with IMU(7) → MLP(256→256) → actions
- `MLPCritic` (`DeterministicMixin`): Linear(20→256→256→256→1) on STATES
No-cam tasks use standard skrl Runner with `skrl_cfg_nocam.yaml` (`separate: False`, shared MLP).

### Logging
- Training metrics are stored in `logs/skrl/drone_racer/` (camera) or `logs/skrl/drone_racer_nocam/` (no-cam)
- `utils/logger.py`: `log(env, keys, tensor)` stores per-step metrics in `env.extras["metrics"]`; `CSVLogger` writes episodes to CSV and calls `utils/plotter.py` on save

### Key constraints
- `--enable_cameras` is **required** for any task that has a `TiledCameraCfg` in the scene. Forgetting it raises `RuntimeError` at sim reset.
- Camera resolution is 64×64 pinhole. Changing it requires updating `IMAGE_H`/`IMAGE_W` constants in `agents/models.py`.
- `env.state_space` (STATES) is only non-None when the `critic` obs group exists in the env cfg. The play configs keep it active because skrl always instantiates the value network even at inference.
- `env_spacing=0.0` — all environments share the same world-space track; gate positions are absolute, not per-env-origin-relative.
