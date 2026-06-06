# CLAUDE.md

Guidance for Claude Code working in this repo. See [HANDOFF.md](HANDOFF.md) for live training state, recent bug fixes, and next experimental moves.

## Environment

- **Isaac Sim 5.1** + **Isaac Lab 2.3.2** + **Python 3.11** (conda env: `isaacsim`)
- **GPU**: RTX 4090 (24 GB VRAM). Non-headless Isaac Sim works fine.
- Activate before anything:
  ```bash
  conda activate isaacsim
  ```
- Run all scripts from repo root (`isaac_drone_racer/`).
- Repo: `git@github.com:Amin-Yazdanshenas/dreamer-drone-racer.git`, branch `master`.
- Binary assets (`*.usd`, `*.dae`, `*.glb`, `*.pt`, `*.mp4`, etc.) are Git LFS tracked.

## Two training paths

### 1. DreamerV3 family (ACTIVE — primary)

Model-based RL. Two agent variants share the same env and training loop:
- **R2-Dreamer**: Barlow Twins repr loss. `dreamer/agent.py` → `DreamerV3Agent`.
- **NE-Dreamer**: causal transformer repr loss (CORL-team NE-Dreamer port). `dreamer/ne_agent.py` → `NEDreamerV3Agent`.

Train:
```bash
python3 scripts/rl/train_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-RGB-v0 \
    --obs_mode rgb --agent r2dreamer \
    --num_envs 256 --headless --enable_cameras \
    --max_steps 5000000

# Optional flags:
#   --checkpoint <path>       resume from .pt checkpoint
#   --record_fpv              save mp4 of every gate-passing episode (env 0 only)
#   --render_interval N       physics steps between viewport renders (auto: 16 GUI, 100 headless)
```

Evaluate:
```bash
python3 scripts/rl/evaluate_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-Play-v0 \
    --obs_mode rgb --agent r2dreamer \
    --checkpoint logs/dreamer/r2dreamer/rgb/<RUN>/checkpoints/agent_latest.pt \
    --num_episodes 10 --enable_cameras

# --stochastic           sample actions instead of tanh(mean)
# --video                record mp4 of eval
```

### 2. skrl PPO (LEGACY — kept but unused)

Earlier camera-PPO + ground-truth-PPO attempt. Kept for reference but **not the active learning path**.

```bash
python3 scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --enable_cameras --num_envs 64
python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --enable_cameras --num_envs 1
```

Uses skrl `Runner` with `CNNPolicy` + `MLPCritic` (asymmetric AC). Don't add new features here — pour them into the Dreamer path.

## Task registry (`tasks/drone_racer/__init__.py`)

| Gym ID | Env cfg | Purpose |
|---|---|---|
| `Isaac-Drone-Racer-Dreamer-RGB-v0` | `DroneRacerEnvCfg_Dreamer` | Dreamer, RGB obs (active) |
| `Isaac-Drone-Racer-Dreamer-Mask-v0` | `DroneRacerEnvCfg_Dreamer` (mask) | Dreamer, segmentation mask |
| `Isaac-Drone-Racer-Dreamer-RGBMask-v0` | `DroneRacerEnvCfg_Dreamer` (rgb+mask) | Dreamer, 4-channel |
| `Isaac-Drone-Racer-Dreamer-Play-v0` | `DroneRacerEnvCfg_Dreamer_PLAY` | Eval |
| `Isaac-Drone-Racer-v0` / `-Play-v0` | `DroneRacerEnvCfg` | LEGACY skrl PPO (camera+IMU actor, GT critic) |
| `Isaac-Drone-Racer-NoCam-v0` / `-NoCam-Play-v0` | `DroneRacerEnvCfg_NoCam` | LEGACY skrl PPO (GT-only) |

## Architecture (DreamerV3 path)

### Env config (`tasks/drone_racer/drone_racer_env_cfg.py`)

Isaac Lab `@configclass`:
- `DroneRacerSceneCfg` — ground + track (7 gates) + drone + IMU + collision sensor + TiledCamera (64×64 RGB).
- `DroneRacerEnvCfg_Dreamer` — uses `DreamerActionsCfg` (CTBR), random-start curriculum, 256 envs, episode_length_s=20.
- `RewardsCfg` — see **Reward shaping** below.

### MDP modules (`tasks/drone_racer/mdp/`)
Flat re-exports via `mdp/__init__.py`. Also re-exports `isaaclab.envs.mdp.*`.

| File | Key class/fn | Notes |
|---|---|---|
| `commands.py` | `GateTargetingCommand` | Tracks `next_gate_idx` per env. Inline plane-crossing + bbox detection (using `prev_robot_pos_w.clone()` snapshot — DO NOT alias). Lerp-spawn curriculum (`spawn_lerp_alpha`, `spawn_forward_velocity`). |
| `rewards.py` | `progress`, `gate_passed` | `progress` is asymmetric (`.clamp(min=0)`). `gate_passed` computes plane-crossing **inline** because Isaac Lab calls `reward_manager.compute()` BEFORE `command_manager.compute()`. |
| `actions.py` | `CTBRAction`, `ControlAction` | CTBR: `c=0 → hover_thrust` (asymmetric linear), PD controller with `derr/dt` damping. ControlAction (motor-omega allocation) is legacy. |
| `events.py` | `reset_after_prev_gate` | Writes drone spawn pose; takes `forward_offset` and `initial_lin_vel_world` params. |
| `observations.py` | various | Note: Dreamer reads obs directly via `env_wrapper.py`, not through ObservationManager. |
| `terminations.py` | standard | `flyaway` (50 m), `collision`, `time_out`. |

### Reward shaping (current weights)

| Term | Weight | Notes |
|---|---|---|
| `terminating` | -2 | Crash penalty (softened from -4) |
| `ang_vel_l2` | 0 | Disabled |
| `progress` | 10 | Asymmetric, low weight to avoid drift exploitation |
| `gate_passed` | 10000 | Isaac Lab dt-scales → +100 per pass. Big enough to dominate progress. |
| `lookat_next` | 0.5 | Heading prior |

Isaac Lab multiplies all reward terms by `dt = sim_dt × decimation = 0.01s/step`. Sparse rewards need weight ≈ `desired_spike / dt`.

### Dreamer agent (`dreamer/`)

| File | Role |
|---|---|
| `agent.py` | `DreamerConfig` (dataclass), `DreamerV3Agent` (encoder + RSSM + reward/cont heads + actor + critic + LaProp + ReturnEMA). Inline `Actor` + `Critic` classes. Has `_repr_loss` hook for subclassing. |
| `ne_agent.py` | `NEDreamerV3Agent` — subclass adding `NEDreamerTransformer`. Overrides `_repr_loss`. |
| `rssm.py` | BlockLinear GRU `Deter` cell + `MultiOneHotDist` categorical latent (`stoch=32`, `discrete=16`). Posterior/prior with balanced KL + free bits per category. |
| `networks.py` | `DroneEncoder` (CNN 64×64×C → embed_dim + state MLP), `MLP`, `MLPHead`, `BlockLinear`, `RMSNorm`, `ReturnEMA`, `NEDreamerTransformer`. |
| `distributions.py` | `TwoHot` (symlog twohot, bins=255, range [-20,20]), `MultiOneHotDist`, `kl()` with per-category free bits, `symlog`/`symexp`. |
| `replay_buffer.py` | `SequenceReplayBuffer` — episode-level storage with cached sample-weights, pads short episodes to `seq_len`. |
| `env_wrapper.py` | `DreamerIsaacEnvWrapper` — bridges gym → DreamerV3 dict obs (image + state). State is **13-dim**: ang_vel_b(3) + quat_w(4) + lin_vel_b(3) + target_pos_b(3). The 10-dim `DreamerObservationsCfg.policy` is dead code — Isaac Lab's ObservationManager computes it but env_wrapper bypasses and reads `robot.data` directly. |
| `optim/laprop.py` | LaProp optimizer + AGC gradient clipping. |
| `configs/dreamer_base.yaml` | h_dim=2048, stoch=32, discrete=16, hidden=256, batch_size=16, seq_len=64, lr=4e-5. |
| `configs/ctbr_gains.yaml` | CTBR PD gains, `max_thrust=23.82 N`, `hover_thrust=5.96 N`. |
| `configs/ne_dreamer_*.yaml` | NE-Dreamer overrides. |
| (`actor_critic.py`, `world_model.py`, `utils.py` deleted) | Were dead — inline `Actor`/`Critic`/world-model-loss live in `agent.py`; symlog/twohot live in `distributions.py`. Removed in the post-review cleanup. |

### Dynamics (`dynamics/`)
Pure-PyTorch, no Isaac Sim deps. `Allocation` (omega² → wrench matrix) + `Motor` (first-order lag). Used by the LEGACY `ControlAction` only. Dreamer uses `CTBRAction`. `tests/test_dynamics.py` has a pre-existing import error (unrelated to current work).

### Logging (Dreamer)
- TB logs → `logs/dreamer/{r2dreamer|ne_dreamer}/{rgb|mask|rgb_mask}/<run-tag>/tensorboard/`
- Checkpoints → `<run-tag>/checkpoints/agent_latest.pt`
- Extra TB tags beyond Dreamer defaults: `env/gate_pass_rate`, `replay/reward_max|min|abs_mean`, `wm/kl_unclamped`, `replay/buffer_size`, `actor/floor_pen`. See HANDOFF.md §6 for meaning.

## Key constraints

- `--enable_cameras` is **required** for any task with a `TiledCameraCfg`. Forgetting → `RuntimeError` at sim reset.
- Camera res 64×64 pinhole. Changing requires updating image shapes in `dreamer/agent.py` (DroneEncoder ctor).
- `env_spacing=0.0` — all envs share the same world track. Gate positions absolute.
- Dreamer warmup: at `_step < cfg.warmup_steps`, `act()` returns random actions BUT still steps the RSSM (avoid zero-carry pollution).
- Isaac Lab order: `reward_manager.compute()` runs BEFORE `command_manager.compute()`. Any reward that depends on command state must read upstream state directly (see `gate_passed` in `rewards.py`).
- Don't rely on `cmd.gate_passed`/`cmd.gate_missed` properties for reward — they fill AFTER reward computation. Compute inline.

## Workflow conventions

- **Auto-commit**: every code change gets committed + pushed with conventional commit format (`fix:`, `feat:`, `refactor:`, `chore:`). User explicitly requested this.
- **Pre-commit hooks**: `pre-commit run --all-files`. Line length **120**. Formatter `black`. Imports `isort --profile black`.
- **Tests**: `pytest tests/ --ignore=tests/test_dynamics.py -q`. Should pass 13/13. (`test_dynamics.py` has stale import.)
- **Caveman mode**: user prefers terse responses. Drop articles/filler; full technical accuracy. Code/commits/security: write normal English.

## Pointers

- **Live training state, bug history, next moves**: [HANDOFF.md](HANDOFF.md)
- **In-flight planning notes**: `~/.claude/plans/floofy-popping-raccoon.md`
- **DreamerV3 paper**: Hafner et al. 2023
- **NE-Dreamer**: https://github.com/corl-team/nedreamer
