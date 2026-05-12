# Handoff — DreamerV3 Drone Racing

**Last updated:** 2026-05-12
**Repo:** `git@github.com:Amin-Yazdanshenas/dreamer-drone-racer.git`
**Hardware:** RTX 4090 (24 GB VRAM), Intel i9-14900K
**Stack:** Isaac Sim 5.1 + Isaac Lab 2.3.2 + Python 3.11 (conda env `isaacsim`) + PyTorch

---

## 1. Project goal

Train a DreamerV3-family world-model RL agent to fly a quadrotor through a 7-gate race track in Isaac Sim, using RGB camera + IMU only (no privileged ground truth). Two agent variants implemented:

- **R2-Dreamer** (active) — Barlow Twins repr loss (`dreamer/agent.py`)
- **NE-Dreamer** (alternate) — Causal transformer repr loss (`dreamer/ne_agent.py`)

Toggle: `--agent r2dreamer | ne_dreamer`

---

## 2. Current state

- **Training step**: ~2.79M env-steps on R2-Dreamer / RGB / 256 envs
- **`best_gates`**: **1** (stuck — drone never chains 2 gates in single episode)
- **Most recent checkpoint**: `logs/dreamer/r2dreamer/rgb/2026-05-12_11-07-14/checkpoints/agent_latest.pt`
- **Latest reward weights**:
  - `progress = 10.0` (asymmetric, only positive)
  - `gate_passed = 10000.0` (+100 per pass after dt-scaling)
  - `terminating = -2.0`
  - `lookat_next = 0.5`
  - `ang_vel_l2 = 0.0` (disabled)

---

## 3. Bug history (chronological)

All fixes already applied. Listed for context.

| # | File | Bug | Severity |
|---|------|-----|----------|
| 1 | `commands.py` | `prev_robot_pos_w = root_pos_w` aliased live tensor → gate detection compared pos to itself every sub-step | 🔴 critical |
| 2 | `commands.py` | Gate plane normal was yaw-only 2D, ignored z and tilted gates | 🔴 critical |
| 3 | `agent.py` | Entropy floor `entropy + F.relu(min - entropy)` algebraically cancels below floor (zero gradient) | 🔴 critical |
| 4 | `actions.py` | CTBR `c=0 → 2× hover thrust`, untrained Gaussian-mean=0 policy → flyaway | 🔴 critical |
| 5 | `rewards.py` | gate_passed reward used `cmd.gate_passed` accumulator BEFORE Isaac Lab's `command_manager.compute()` ran → reward always 0 | 🔴 critical |
| 6 | `commands.py` _resample | Ghost gate detection from teleport — synced `prev_pos` BEFORE `reset_after_prev_gate` wrote new pose | 🟡 high |
| 7 | `rewards.py` | Gate +30 / miss -30 cancelled to 0 when both fired in same RL step. Removed miss penalty entirely | 🟡 high |
| 8 | `env_cfg.py` | Isaac Lab dt-scales rewards. weight=30 → only +0.3 per gate. Bumped to 3000 → 10000 (+30 → +100) | 🟡 high |
| 9 | `agent.py` | RSSM not stepped during warmup → garbage latents after warmup ends | 🟡 medium |
| 10 | `replay_buffer.py` | Short episodes (<seq_len) silently discarded → no crash data in buffer | 🟡 medium |
| 11 | `replay_buffer.py` | Sample weights rebuilt every batch — O(N_eps) per sample → scaling cost | 🟢 low |
| 12 | `env_wrapper.py` | `reset_step_accumulators` zeroed gate accum at wrong time | 🟢 low |

**Inline doc on each fix** is in the source — search for "BUG" / "CRITICAL" / "fix" comments.

---

## 4. Curriculum knobs (in `commands.py:GateTargetingCommandCfg`)

| Knob | Default | Purpose |
|---|---|---|
| `spawn_lerp_alpha` | **0.3** | LERP(prev_gate, next_gate, alpha) at reset. 0 = at prev gate (hard), 1 = at next (trivial). Currently mid-difficulty. |
| `spawn_forward_velocity` | **1.5** | m/s initial velocity along prev→next direction. Helps untrained policy collect useful data. |
| `gate_size` | **1.5** | bbox half-size for gate pass detection. Could bump to 2.5 for easier chaining. |

---

## 5. Active open problem

**Drone never chains 2 gates.** After 2.79M env-steps + 4700 episodes, `best_gates` is still 1.

### Diagnosis chain so far

- Reward signal is verified flowing correctly (`[GATE-PASS] reward=30.x` events confirm).
- Critic learning: `imag/value_mean max=0.268`, `imag/return_mean max=0.328` (positive).
- World model encoding: `wm/kl_unclamped` reached **10.02** (RSSM is encoding observations).
- Policy committing: `actor/entropy min=0.60` at moments.
- BUT: progress weight=100 backfired — drone exploited drift as steady reward, gate-pass rate dropped 8×. Just reverted to weight=10 + gate weight=10000.

### Hypotheses (untested)

1. Gate bbox too tight (1.5 m) for post-gate-1 trajectory to thread.
2. Critic underestimates post-gate-1 value (chicken-and-egg: needs chained episodes to learn).
3. Imagination horizon too short (`imag_horizon=15` = 0.15 s) for chaining inference.
4. R2-Dreamer's single-step Barlow Twins is too weak for temporal credit assignment. NE-Dreamer's transformer might help here.

### Next experimental moves

| If `best_gates=1` after another 500K env-steps | Action |
|---|---|
| Reward weights still wrong | Try `gate_passed=20000` (more aggressive) |
| Bbox too tight | Bump `gate_size 1.5 → 2.5` (curriculum on success condition) |
| Critic stuck | Increase `imag_horizon 15 → 30` (more lookahead during policy learning) |
| Architecture issue | Switch to `--agent ne_dreamer` from scratch for temporal modeling |

---

## 6. Diagnostic metrics in TensorBoard

Beyond Dreamer defaults, the training script logs:

| Tag | Meaning |
|---|---|
| `env/gate_pass_rate` | per-step probability of gate pass across all envs in window |
| `replay/reward_max` | max raw reward in last sampled batch — should ≥ 30 with healthy buffer |
| `replay/reward_min` | min raw reward — should be small (no miss penalty) |
| `replay/reward_abs_mean` | reward signal magnitude in buffer |
| `wm/kl_unclamped` | raw KL before free-bits floor — distinguishes "posterior collapse" (~0) from "free zone" (~32) |
| `replay/buffer_size` / `num_episodes` | buffer health |
| `actor/floor_pen` | entropy-floor penalty firing (max>0 means entropy dipped below 1.0) |

`[GATE-PASS]` console lines also fire per env on real gate passes.

---

## 7. How to run

### Train (continue from latest)

```bash
conda activate isaacsim
python3 scripts/rl/train_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-RGB-v0 \
    --obs_mode rgb --agent r2dreamer \
    --num_envs 256 --headless --enable_cameras \
    --checkpoint logs/dreamer/r2dreamer/rgb/<RUN>/checkpoints/agent_latest.pt \
    --max_steps 5000000
```

### Train (fresh, NE-Dreamer)

```bash
python3 scripts/rl/train_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-RGB-v0 \
    --obs_mode rgb --agent ne_dreamer \
    --num_envs 128 --headless --enable_cameras
```

### Evaluate (visual, in viewer)

```bash
python3 scripts/rl/evaluate_dreamer.py \
    --task Isaac-Drone-Racer-Dreamer-Play-v0 \
    --obs_mode rgb --agent r2dreamer \
    --checkpoint logs/dreamer/r2dreamer/rgb/<RUN>/checkpoints/agent_latest.pt \
    --num_episodes 10 \
    --enable_cameras
```

### Record gate-pass videos

Add `--record_fpv` to train. Videos saved as `fpv_gates{N}_v{ID}.mp4` in repo root (only episodes with ≥1 gate pass kept).

### TensorBoard

```bash
tensorboard --logdir logs/dreamer --port 6006
```

### Tests (no Isaac Sim)

```bash
conda activate isaacsim && pytest tests/ --ignore=tests/test_dynamics.py -q
```

Currently **13/13 pass**. (`test_dynamics.py` has pre-existing import error, unrelated.)

---

## 8. Repo / git

- **Origin**: `git@github.com:Amin-Yazdanshenas/dreamer-drone-racer.git` (standalone, not forked)
- **Branch**: `master`
- **LFS-tracked**: `*.usd`, `*.dae`, `*.glb`, `*.pt`, `*.mp4`, etc. (see `.gitattributes`)
- **Workflow**: every code change is auto-committed + pushed with conventional commit messages (`fix:`, `feat:`, `refactor:`, `chore:`).

### Clone on new machine

```bash
git lfs install
git clone git@github.com:Amin-Yazdanshenas/dreamer-drone-racer.git
cd dreamer-drone-racer
git lfs pull   # download binary assets
```

---

## 9. Key files

| File | Role |
|---|---|
| `dreamer/agent.py` | R2-Dreamer agent (DreamerConfig + DreamerV3Agent) |
| `dreamer/ne_agent.py` | NE-Dreamer subclass with causal transformer |
| `dreamer/rssm.py` | RSSM (BlockLinear GRU + MultiOneHotDist categorical latent) |
| `dreamer/networks.py` | DroneEncoder, MLP, NEDreamerTransformer |
| `dreamer/replay_buffer.py` | Sequence replay buffer (variable-length episodes, padded) |
| `dreamer/env_wrapper.py` | Bridges Isaac Lab env → DreamerV3 dict obs |
| `scripts/rl/train_dreamer.py` | Training script |
| `scripts/rl/evaluate_dreamer.py` | Inference / video recording |
| `tasks/drone_racer/drone_racer_env_cfg.py` | Isaac Lab env config (scene, rewards, terminations) |
| `tasks/drone_racer/mdp/commands.py` | `GateTargetingCommand` — gate detection, curriculum spawn |
| `tasks/drone_racer/mdp/rewards.py` | progress / gate_passed / lookat / ang_vel_l2 |
| `tasks/drone_racer/mdp/actions.py` | CTBR rate controller (collective thrust + body rates) |
| `dreamer/configs/dreamer_base.yaml` | hyperparams (h_dim, batch_size, lr, etc.) |
| `dreamer/configs/ctbr_gains.yaml` | rate controller PD gains, max_thrust, hover_thrust |

---

## 10. Quick links

- **Last training run**: `logs/dreamer/r2dreamer/rgb/2026-05-12_11-07-14/`
- **Latest checkpoint**: `logs/dreamer/r2dreamer/rgb/2026-05-12_11-07-14/checkpoints/agent_latest.pt`
- **Plan file** (in-flight planning notes): `/home/aminys/.claude/plans/floofy-popping-raccoon.md`
- **DreamerV3 paper**: Hafner et al., "Mastering Diverse Domains through World Models" (2023)
- **NE-Dreamer ref**: CORL team, https://github.com/corl-team/nedreamer

---

## 11. Known limitations / TODOs

- **`tests/test_dynamics.py`** has pre-existing import error (`build_allocation_matrix` missing). Unrelated to current work.
- **`actor_critic.py`** is dead code — inline classes in `agent.py` are used instead. Cleanup candidate.
- **Imagination horizon** = 15 steps. May be too short for cross-gate planning. Try 30.
- **`return/scale`** EMA stuck at 1.0 even after 2.79M steps — return spread is below the 1.0 floor of `max(1, hi-lo)`. Symptom of single-gate cap.
- **No checkpoint pruning** — `agent_latest.pt` overwrites, but older runs accumulate. Manually clean.
- **Camera obs is RGB only**. Could try mask or rgb_mask for cleaner gate signal.
