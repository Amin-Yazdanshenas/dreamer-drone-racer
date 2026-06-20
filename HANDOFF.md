# Handoff — DreamerV3 Drone Racing

**Last updated:** 2026-06-20
**Repo:** `git@github.com:Amin-Yazdanshenas/dreamer-drone-racer.git`, branch `master`
**Active agent:** R2-Dreamer (Barlow Twins), RGB obs, 256 envs.
**Compute:** training runs on a REMOTE box `avl-super@100.117.154.65` (RTX 4090 24 GB VRAM, **62 GB host RAM**). The local machine is a 6 GB RTX 3060 laptop — too small for full training; only used for short headless evals/diagnostics.

> NOTE: CLAUDE.md says RSSM `discrete=16` and "state is 13-dim" — both STALE. Actual: `discrete=32`, policy state is now **22-dim** (see below).

---

## 1. Current state (what's running NOW)

- **v3 corrective run** — launched 2026-06-20 on commit `e8958e8`, fresh from scratch, `--max_steps 50000000`. Log: `~/train_v3.log` on the remote. Newest dir under `logs/dreamer/r2dreamer/rgb/`.
- Healthy at last check (proc alive, GPU ~60%, VRAM ~19 GB, RAM filling toward ~48 GB cap). Very early (~0.5M steps when last seen).
- **What v3 tests:** does *96px perception + 2-gate look-ahead state* — with the gate reward kept strong and a medium train ratio — beat the old 30% lap ceiling? It strips the two changes that capped/crashed the previous attempt (see §2).

### Best checkpoints / fallbacks (all on remote)
| Policy | Path | Note |
|---|---|---|
| **Best overall — 30% lap** | `logs/dreamer/r2dreamer/rgb/2026-06-13_21-29-13/checkpoints/agent_snapshot_36M.pt` | the proven peak; KEEP |
| v2b ~18% peak | `logs/dreamer/r2dreamer/rgb/2026-06-17_21-37-22/checkpoints/agent_best.pt` | full-package run (capped low) |
| v3 live best | `<v3 run dir>/checkpoints/agent_best.pt` | auto-saved on smoothed-gates peak |

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

1. **Watch v3** to first verdict (~20-30M steps, ~1 day): does it beat 18% → 30%? If it caps ~18% again, perception+state alone aren't enough → revisit (the centering reward, or model size — the TSC paper suggests *smaller* models; or a pixel-recon/perceptual loss to force frame encoding).
2. **Fix the deterministic-eval collapse** (1-line decouple in `act()`: always `sample=True` for the latent, only the *action* mode). Until then deploy/eval `--stochastic`.
3. If v3 also collapses mid-run, consider entropy floor anneal late (`--entropy_min_final 0.2 ...`) — but note the earlier finetune-from-36M attempt with anneal REGRESSED, so be careful.

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
