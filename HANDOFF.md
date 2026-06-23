# Handoff — DreamerV3 Drone Racing

**Last updated:** 2026-06-23
**Repo:** `git@github.com:Amin-Yazdanshenas/dreamer-drone-racer.git`, branch `master`
**Active agent:** R2-Dreamer — a **PyTorch** DreamerV3 variant (Barlow-Twins repr loss + informed/privileged 12-dim decoder, NO image-reconstruction, NO JAX). 256 envs.
**Compute:** training runs on a REMOTE box `avl-super@100.117.154.65` (RTX 4090 24 GB VRAM, **62 GB host RAM**). The local machine is a 6 GB RTX 3060 laptop — too small for full training; only used for short headless evals/diagnostics.

> NOTE: CLAUDE.md says RSSM `discrete=16` and "state is 13-dim" — both STALE. Actual: `discrete=32`, policy state is now **22-dim** (see below).

---

## 1. Current state (what's running NOW)

- **Run `2026-06-22_17-37-15`** (commit `7cdf203`) — the proven **64px 30% baseline + ONLY the extended imag_horizon curriculum [24..80]**. Fresh from scratch, `--max_steps 45M`, log `~/train_64imag.log`.
- Tests one variable: does **more planning horizon** (old run maxed at H=48, plateaued ~30%) push chaining past 30%?
- Status ~06-23: healthy (proc alive, GPU ~60-90%, VRAM ~18 GB, RAM ~39 GB capped). Step ~2.5M, mean ~0.80 gates — still in the single-gate-learning phase (imag stays at 24 until mean ≥0.85). Slow (~0.3M steps/hr at n_grad 8).

### Best checkpoints / fallbacks (all on remote)
| Policy | Path | Note |
|---|---|---|
| **Best overall — 30% lap** | `2026-06-13_21-29-13/checkpoints/agent_snapshot_36M.pt` | the proven peak; KEEP |
| v2b ~18% peak | `2026-06-17_21-37-22/checkpoints/agent_best.pt` | 96px full-package (capped low) |

## 1a. Shared model (R2-Dreamer — constant across ALL runs below)

| Param | Value | | Param | Value |
|---|---|---|---|---|
| RSSM deter `h_dim` | 2048 | | seq_len / batch_size | 64 / 16 (stage 0) |
| stoch × discrete | 32 × 32 (→1024); latent 3072 | | num_envs | 256 |
| hidden / cnn_depth | 256 / 48 (mults [1,2,4,8]) | | decimation / ctrl dt | 12 / 0.03 s (sim 1/400) |
| lr (LaProp) | **1e-4** (yaml; overrides the 4e-5 dataclass default — comment says "4e-5" but value is 1e-4) | | entropy_scale / _min | 3e-3 / 1.0 floor |
| priv_decode / barlow | 12-dim, w=1.0 / w=0.05 | | kl_free / β_dyn / β_rep | 0.1 / 0.5 / 0.1 |
| action | CTBR (4-dim) | | episode / track | 20 s / 7 gates |
| reward | progress 50 (signed) + gate 500 + terminating −1000 + lookat 0.5 + ang_vel −0.002 | | | |

## 1b. RUN LOG — every training run this session (per-run knobs + outcome)

Only the **env/training knobs** differ run-to-run; the model above is constant. Camera-size and state-dim changes force a fresh-from-scratch run (encoder reshape).

| Run dir | img | state | n_grad | gate rwd | collis | imag stages | replay | Outcome |
|---|---|---|---|---|---|---|---|---|
| **2026-06-13_21-29-13** (baseline) | 64 | 16 | 8 | flat | 10 N | [24-48] | 2M | **★ peak 30.7% lap, mean ~5** then eroded — BEST policy |
| 2026-06-16_20-53-15 (finetune) | 64 | 16 | 8 | flat | 10 N | [24-64] | 2M | resume 36M + entropy-anneal + H→64 → **regressed** 30→19% |
| 2026-06-17_08-21-04 (v2) | 96 | 22 | 4 | center[.5,1] | 25 N | [24-64] | 2M | **OOM-killed** (CPU RAM, 96px×2M ≈ 55 GB) |
| 2026-06-17_21-37-22 (v2b) | 96 | 22 | 4 | center[.5,1] | 25 N | [24-64] | 1.2M | **18% lap peak, collapsed @30M** |
| 2026-06-20_12-40-36 (v3) | 96 | 22 | 6 | center[.8,1] | 25 N | [24-64] | 1.2M | trailed (lap ~1% @12M), slower than v2b |
| 2026-06-22_00-36-12 (abl96) | 96 | 16 | 8 | flat | 10 N | [24-48] | 1.2M | stuck single-gate **~0.73**, never advanced |
| 2026-06-22_10-51-05 (abl96b) | 96 | 16 | 8 | flat | 10 N | [24-80] | 1.2M | stuck single-gate **~0.72** |
| **2026-06-22_17-37-15** (current) | 64 | 16 | 8 | flat | 10 N | [24-80] | 2M | running; mean ~0.80 climbing (single-gate, 2.5M) |

**What the run log shows:** only **64px** runs ever mastered single-gate / reached 30%; every **96px** run stalled at single-gate ~0.72 (same encoder capacity can't handle 2.25× pixels) → **96px HURTS**. Every change off the baseline (centering reward, state 22, n_grad 4/6, collision 25, finetune levers) **regressed** it. Lesson: don't bundle changes — single-variable tests vs the 30% baseline only.

---

## 2. The story this session (2026-06-15 → 06-20)

1. **The 2-gate "invariant" ceiling was broken** by the performance-gated `imag_horizon` auto-curriculum (run `2026-06-13_21-29-13`). It chained full laps — **peaked ~30.7% full-lap, mean ~5 gates** at ~36M steps, then plateaued/eroded. Overturns loop_report.md's "ceiling is structural, not tunable". See memory `two-gate-ceiling-broken`.
2. **Deterministic eval collapse found** (still UNFIXED): `evaluate_dreamer.py` deterministic = **0.34 gates** vs `--stochastic` = **4.2** on the SAME checkpoint. Cause: `DreamerV3Agent.act()` (`dreamer/agent.py` ~L561) couples latent sampling to the action flag (`sample=not deterministic`), forcing the RSSM latent to its argmax mode (OOD). Eval/deploy with `--stochastic` until fixed. See memory `deterministic-eval-collapse`.
3. **Checkpoint best-gap fixed** (`agent_best.pt`, commit `2a1173d`): the loop only saved rolling `agent_latest` + `agent_final`, losing peak weights on erosion. Now saves `agent_best.pt` on a smoothed-gates improvement. See memory `checkpoint-best-gap`.
4. **Expert bottleneck review** traced the 30% wall to a **perception↔reward co-design gap** (drone clips gate FRAMES — ~96-100% of deaths are gate-frame collisions): 64px CNN → 2×2 grid can't resolve frame edges; no pixel-recon loss; reward only rewards binary pass; myopic 1-gate state; plus an update-bound loop (~0.3M steps/hr).
5. **v2 "full package"** (commit `51e3505`): camera 64→96, centering pass-bonus, 2-gate look-ahead state (16→22), n_grad_steps 8→4. **OOM-killed** on CPU RAM (96px replay buffer ×2M ≈ 55 GB > 62 GB) → fixed with `replay_capacity` 2M→**1.2M** (commit `a5d1dfc`).
6. **v2b run** (the fixed full package): learned **~2× faster early** but **capped lap ~18%** (below 30%) and then **collapsed ~30M** (mean 3.3→0.38, silent actor-critic divergence), recovering to ~10%. Diagnosis: centering reward gutted the gate signal ~2.5× (imag value +56 vs +140 at equal lap); n_grad_steps 4 too light (noisy value → collapse).
7. **v3 corrective** (commit `e8958e8`, RUNNING): centering softened **[0.5,1.0]→[0.8,1.0]** (restore gate signal, keep gentle threading nudge); n_grad_steps **4→6** (medium — a DreamerV3-on-TSC tuning study the user found reports no evidence higher train ratio helps convergence, and medium ratios work well). Kept 96px + look-ahead state + replay 1.2M.

---

## 3. Config state (`dreamer/configs/dreamer_base.yaml` unless noted)

| Knob | Value | Changed this session |
|---|---|---|
| `image_size` | **96** (was 64) | new field, wired to encoder + `TiledCameraCfg` width/height |
| `state_dim` | **22** | +next-gate pos_b(3) + next-gate normal_b(3) on top of the old 16 |
| `n_grad_steps` | **6** (8→4→6) | medium train ratio |
| `replay_capacity` | **1,200,000** (was 2M) | RAM fit at 96px |
| centering pass-bonus | mult **[0.8,1.0]** | `tasks/.../mdp/rewards.py` `gate_passed` |
| collision threshold | **25 N** (was 10) | `drone_racer_env_cfg.py` |
| `IMAG_STAGES` / `BATCH_STAGES` | `[24,32,40,48,56,64]` / `[16,12,10,8,7,6]` | extended past 48 (commit `b7f55f7`) |
| entropy anneal | no-op default | `entropy_min_final/anneal_start/anneal_end` exist (commit `b7f55f7`); CLI `--entropy_min_final` etc. Used by the failed finetune attempt; OFF for fresh runs. |
| h_dim 2048, stoch 32, discrete 32, seq_len 64, batch_size 16 | unchanged | |

Other CLI added (`scripts/rl/train_dreamer.py`): `--imag_start_stage N`, `--spawn_done` (resume a converged policy at the end of the curricula); `--entropy_min_final/--entropy_anneal_start/--entropy_anneal_end`.

---

## 4. Key commits this session

```
e8958e8 fix(train): corrective run — soften centering reward + medium train ratio   (v3, RUNNING)
a5d1dfc fix(replay): cap replay_capacity 2M->1.2M — 96px buffer OOM-killed the run
51e3505 feat: attack the lap-30% wall — perception, threading reward, look-ahead state, throughput
b7f55f7 feat(train): finetune levers (entropy anneal, imag 48->64, --imag_start_stage/--spawn_done)
2a1173d feat(train): save agent_best.pt on smoothed gates/ep peak
```

---

## 5. Pace reference (full-lap % vs steps)

- **Old run** (`2026-06-13_21-29-13`): lap 4.5% @24M, 14% @27M, 22% @33M, **30.7% peak @~36M**, then eroded.
- **v2b** (96px full package): lap ~1% @8M, 10% @19M, **~18% peak @27M**, collapsed @30M.
- Use these to judge v3: if v3 clears ~18% and keeps climbing past 30%, the perception+state fixes worked.

---

## 6. Next moves / open questions

**SkyDreamer paper lessons** (Verraest et al., Delft — DreamerV3 vision drone-racer, champion-level real-world; in `~/Downloads/clean_paper.md`). Things it does that we DON'T — all base-agnostic (apply to our PyTorch R2-Dreamer, no JAX needed), ranked:
1. **Segmentation MASKS, not RGB.** Their visual obs is a binary gate mask (64×64 via GateNet). Far easier to learn than RGB. **Repo already supports** `--obs_mode mask` + has GateNet. ⭐ likeliest fix for our slow single-gate learning.
2. **SHORT imagination horizon (16 steps, 0.18 s).** We've been *extending* to H=80 — backwards. Stop extending; go short.
3. **3-gate flight-plan vector** (rel pos+yaw of next 3 gates) — richer than our reverted 1-gate lookahead; resolves visual ambiguity.
4. **Smoothness reg on the policy MEAN** (λ=0.002) — kills bang-bang AND makes the deterministic policy smooth → would fix our deterministic-eval collapse.
5. **Center-scaled gate reward + pre/post virtual gates** — they USE centering (we reverted it); works because paired with pre/post collision-volume gates. No collision penalty, no perception reward (both match our findings).
6. **Staged schedule:** at 8M bump seq_len 64→256; at 13M drop entropy 3e-4→1e-5 + lr 4e-5→2e-6 (late fine-tune). train_ratio 128, replay 10M, default DreamerV3 12M model.

**Recommended order (single-variable, vs the 30% baseline):**
1. **`--obs_mode mask`** run (biggest, repo-ready). Kill RGB.
2. **Short imag horizon** (~H=6-16; stop the [24..80] extension).
3. **Fix deterministic-eval collapse** — 1-line decouple in `act()` (`dreamer/agent.py` ~L561): always `sample=True` for the latent, only the *action* is the mode. Until then eval/deploy `--stochastic`. (Or add the smoothness reg, which also fixes it.)
4. Then 3-gate flight-plan; center-scaled reward + pre/post gates.

**Open caveat:** R2-Dreamer is a hand-rolled PyTorch DreamerV3 port + Barlow Twins. If the masks/short-imag fixes still underperform, suspect **port faithfulness** vs official DreamerV3 (KL balance, free-bits, return-norm, symlog, WM-loss weights are easy to get subtly wrong).

---

## 7. Ops / how to monitor

```bash
# SSH (remote box, password is "1")
ssh avl-super@100.117.154.65

# pulse
pgrep -fc train_dreamer.py            # proc alive?
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
free -g                               # RAM — buffer caps ~48 GB at 1.2M; >58 GB = OOM risk
tail -1 ~/train_v3.log; grep -ac 'NEW BEST' ~/train_v3.log

# full gate stats: EventAccumulator on <run dir>/tensorboard, read env/episode_gates,
# window by step, compute mean / ge2 (>=2) / ge5 (>=5) / lap (>=7) / max.
```

- **Replay buffer is NOT saved to disk** — resumed runs start buffer-empty (warmup re-fills). So old run dirs hold only checkpoints (.pt, ~450-510 MB each) + TB logs; safe to delete old runs' `checkpoints/*.pt` to free disk (keep the two fallback runs above).
- Launch detached on remote: `setsid bash -c '... python3 -u scripts/rl/train_dreamer.py ...' > ~/train.log 2>&1 < /dev/null &` after `git fetch origin -q && git reset --hard origin/master -q`.
- 96px: VRAM ~19 GB (fits 24), RAM caps ~48 GB at replay 1.2M. Do NOT raise replay_capacity at 96px without RAM math (27.6 KB/image).
- Monitoring shorthand used in chat: `geN` = % of episodes passing ≥N gates; `lap` = ge7 (full 7-gate lap); `mean` = avg gates/episode.
