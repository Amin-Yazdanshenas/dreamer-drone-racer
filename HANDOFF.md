# Handoff — DreamerV3 Drone Racing

**Last updated:** 2026-07-06
**Repo:** `git@github.com:Amin-Yazdanshenas/dreamer-drone-racer.git`, branch `master`
**Active agent:** R2-Dreamer — a **PyTorch** DreamerV3 variant (Barlow-Twins repr loss + informed/privileged 12-dim decoder, NO image-reconstruction, NO JAX). 256 envs.
**Compute:** training runs on a REMOTE box `avl-super@100.117.154.65` (RTX 4090 24 GB VRAM, **62 GB host RAM**). The local machine is a 6 GB RTX 3060 laptop — too small for full training; only used for short headless evals/diagnostics.

> NOTE: CLAUDE.md says RSSM `discrete=16` and "state is 13-dim" — both STALE. Actual: `discrete=32`, policy state is now **22-dim** (see below).

---

## 0. TRAINING-UPDATE PROTOCOL — what to report when the user asks "how is training / check the training"

Pull the active run's TB scalars (remote `event_accumulator`, `size_guidance={SCALARS:0}`) and report these, in this order. Keep it a scannable table, not prose.

1. **Header:** run id, step (M), wall-clock uptime, alive?/GPU util, rate (M/hr) + ETA to next milestone.
2. **SUCCESS-RATE BUCKETS (the headline).** `env/episode_gates` is **per-episode integer gate counts** (7 gates = 1 lap). Over the recent window (~last 5M steps) report the fraction of episodes with gates **≥2, ≥5, ≥7 (1 lap), ≥14 (2 laps)**, plus mean + max. This is the real "success rate," not gate_pass_rate (that's passes/step).
3. **Peak** smoothed gates/ep so far + at what step.
4. **COLLAPSE-FLAG PANEL** (current + max-so-far, with the kill thresholds):
   - `return/scale` — terminal value/return runaway. Kill value was **1567** (`2026-06-28`). Healthy < ~100.
   - `imag/value_mean` — value inflation. Kill value **802**. Healthy < ~100.
   - `actor/entropy` — actor-sharpening. Floor is 1.0; **< -1 and falling = crater risk** (crater run hit -3).
   - `actor/action_sat_frac` — bang-bang. **> 0.30 = danger** (normal ~0.10-0.15).
   - `critic/loss` — follower; note if spiking.
5. **Comparison:** vs gold at the matched step, and vs the **gold-peak target = 19.4% full-lap @36M** (mean 4.11). State whether on/above or below gold's trajectory.
6. **Verdict + next gate:** on-track / watch / trigger. Real danger windows: **~7M** (terminal runaway — split-LR CLEARED) and **~29M** (actor crater). **36M is NOT a wall** — gold was hand-stopped @38M ([[gold-baseline-hand-stopped-not-walled]]); past ~38M is uncharted.

Baseline success-rate references (fraction of episodes ≥N gates): gold PEAK[34-38M] mean 4.11 / ≥2:73% / ≥5:39% / ≥7:19.4% / ≥14:2.0%. crater[24-29M] 17.9% lap. Matched[15-20M]: split-LR mean 1.71 (≥5:5.4%, lap:0.8%) vs gold 1.49 (≥5:2.2%, lap:0.1%).

---

## 1. Current state (07-06: SPLIT-LR run FINISHED 50M — project-best on every axis; stability SOLVED)

**`2026-06-30_04-07-04` (SPLIT-LR) COMPLETED the full 50M cleanly** — zero collapses in 6d7h. Gold-baseline config + ONLY change = actor/critic lr 1e-4→**3e-5** (wm 1e-4), seed 42. Results:
- **NEW BEST smoothed gates/ep = 9.325 @41.9M** (gold's all-time: 6.87) → `agent_best.pt` = **best policy of the project**.
- **Peak window [41-42M]: 47.4% full-lap, 16.4% 2-lap** (gold peak: 19.4% / 2.0%). Max 19 gates ≈ 2.7 laps (episode-time capped at 20 s).
- **Both collapse modes CLEARED**: ~7M terminal-runaway window (return/scale max 105 vs kill 1567) and ~29M actor-crater window (entropy stayed positive). >38M uncharted region: no cliff — oscillated 30-47% lap.
- Post-peak (42→50M): slow **EROSION** to ~30% lap (drift/forgetting, flags all green — NOT a collapse; same pattern as the 06-13 run). Peak is banked in `agent_best.pt`.
- **Conclusion: split-LR validated. Stability problem SOLVED. The remaining fight is the ~45% CEILING** (perception/reward co-design — gate-frame clips at 64px) **+ post-peak erosion** (candidate: late entropy-floor anneal, or freeze-and-stop at peak).

**07-06 eval** of `agent_best.pt` (92 eps, stochastic): **46.7% full-lap, mean 6.24 gates** — deployment matches the training peak. (Also found a 10% instant-death cluster from an eval-only stale-camera-frame bug, since fixed.)

**07-08 EROSION-FIX RESUME (`2026-07-08_11-17-25`) — FALSIFIED, killed @46M.** Resumed `agent_best` (47%) with `entropy_floor_weight 1e-2→5e-2` + `--finetune_refill` + split-LR, testing whether a stronger entropy floor stops the post-peak erosion. It did NOT: policy settled to a ~34% lap band (~13pt below the 47% resume point) even though the floor held entropy at ~1.1. **Erosion is NOT entropy-driven — it's buffer/off-policy drift.** Mitigation = freeze-at-peak (`agent_best`), not entropy tweaks. See [[erosion-not-entropy-driven]]. Positive side-results: the resume-lr fix, `--finetune_refill` (policy refill — FIRST clean finetune-resume, held the policy vs 06-16's 30→19% regression), and the eval stale-frame fix all worked.

**GPU idle. Next (per plan): oracle-mask ablation** — the ceiling attack. Now the confirmed bottleneck is the ~45-47% ceiling (perception/reward), not stability (solved) or erosion (banked via agent_best).

### Best checkpoints / fallbacks (all on remote)
| Policy | Path | Note |
|---|---|---|
| **★★ BEST — 9.33 smoothed, 47% lap peak** | `2026-06-30_04-07-04/checkpoints/agent_best.pt` | split-LR peak @41.9M |
| split-LR final (eroded ~30%) | `2026-06-30_04-07-04/checkpoints/agent_final.pt` | end-state @50M |
| prev best ~40% lap | `2026-06-23_00-54-13/checkpoints/agent_best.pt` | gold-baseline peak (6.87 smoothed) |
| old 30% lap | `2026-06-13_21-29-13/checkpoints/agent_snapshot_36M.pt` | original peak; KEEP |

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
| 2026-06-22_17-37-15 (64+H80) | 64 | 16 | 8 | flat | 10 N | [24-80] | 2M | reverted to clean 64 baseline mid-run; see 06-23 run |
| **2026-06-23_00-54-13** (EXACT baseline) | 64 | 16 | 8 | flat | 10 N | [24-48] | 2M | **★ reproduced AND spiked to ~40% lap @36M (mean 6.3)**; transient dip @36.8M (→~4% one window) that RECOVERED, then **MANUALLY STOPPED @38M** (not a collapse) — `agent_best.pt` = the 40% peak |
| 2026-06-28 (lr-anneal) | 64 | 16 | 8 | flat | 10 N | [24-48] | 2M | baseline + lr-anneal 1e-4→2e-5 @26-34M. **CRITIC DIVERGED @8-11M (value→502)**, killed. Anneal never engaged (too late). |
| **2026-06-30_04-07-04 (SPLIT-LR)** | 64 | 16 | 8 | flat | 10 N | [24-48] | 2M | baseline + actor/critic lr **3e-5** (wm 1e-4), seed 42. **★★ COMPLETED 50M, zero collapses. NEW BEST 9.325 smoothed @41.9M; peak 47.4% lap / 16.4% 2-lap [41-42M]**; eroded to ~30% by 50M (drift, not collapse). |

**What the run log shows:** only **64px** runs ever chained well; **96px** stalled at single-gate ~0.72 (premature call — see note) → treat 96px as hurting for now. The 64px baseline is **reproducible to a ~40% peak** (higher than the 30% we were anchored to) but **collapses** after peaking. Don't bundle changes — single-variable tests vs the baseline only, and expect **±10pt run-to-run variance** (the exact peak isn't deterministic).

> Correction logged: the "96px stuck ~0.72" calls were made at ~3M, but the 64px baseline shows single-gate breakthrough happens ~4-5M — so 96px was likely judged too early, not necessarily broken. And early "lap %" reads used 2M windows that SMOOTHED AWAY a sharp 40% spike — always check `NEW BEST smoothed gates/ep` (the 600-ep peak detector) + 0.5M bins.

---

## 2a. THE KEY FINDING (06-27→30): ceiling ~40%; collapse = TWO families, WM not implicated

1. **The ceiling is ~40% lap, not 30%.** The exact-baseline run (`2026-06-23`) spiked to **~40% lap / mean 6.3 / value ~153 at 36M** — beating the original's 30.7%. My 2M-windowed text reads smoothed this spike to "21%"; the 0.5M-bin graph + the `NEW BEST smoothed gates/ep=6.87` log line revealed it. **`agent_best.pt` from that run is the new best policy.** NOTE: gold's 36.8M "collapse to ~4%" was a **transient one-window dip that RECOVERED** (gates 6.8→1.9→7.2), and the run was **manually stopped at 38M** — NOT an intrinsic wall/collapse. "36M" is not a danger point; no run has been left to run past ~38M.
2. **The genuine collapses are only TWO** (gold never truly crashed): the terminal value/return runaway (`2026-06-28`, @7M) and the recoverable actor-sharpening crater (`2026-06-17`, @29M, which itself recovered). Not "recurring crashes" — one lethal mode + one survivable dip.
3. **LOG-DIAGNOSIS (06-30, zero-GPU, from banked TB scalars).** Pulled the curves at three collapse onsets and asked the council's question: *does the world model lead?* **No.** See `collapse_diagnosis.png` (3-panel). Findings:
   - **World-model losses are DEAD FLAT in all three** (`wm/total`≈3.9, `wm/dyn_loss`≈3.8, `wm/rep_loss`, `wm/barlow` — sub-1% moves through every collapse). The "WM drift → bad imagined returns → critic chases" feedback-loop theory is **refuted**. WM is a bystander → skip all world-model fixes for the collapse.
   - **Stabilizers ARE present and live** (not missing): `slow_target_fraction 0.02` = target-critic EMA; `return/scale` tag = ReturnEMA return-normalization (active, 26–1567). So this is not a dropped-stabilizer bug.
   - **Three signatures, two families:**
     - **ACTOR-led crater** (`2026-06-17_21-37-22` @29.4M, recoverable): `actor/entropy` collapses **1.0 → -3.0** (punches *through* its 1.0 floor) + `action_sat_frac` 0.17→0.30 (bang-bang) **lead** → value craters → `critic_loss` rises (follower) → barlow ticks up **last**. Root: actor over-sharpens; `entropy_floor_weight=1e-2` too weak to hold.
     - **VALUE/RETURN runaway** (critic-led): mild = `2026-06-23` gold (value inflates 100→160, `critic_loss` *drops*, one transient dip→recovers); **severe/terminal** = `2026-06-28_00-17-01` (killed @11M): at ~7.1M `return/scale` **explodes 108→1567 (14×)**, `imag/value` 50→293 (raw max **802**), `critic_loss` 1.3→2.2, gates→0, policy entropy rises to 1.78 (actor loses all signal). ReturnEMA fails to contain the runaway.
4. **Corrected root cause.** NOT "critic too hot diverges, actor follows" (the old §2a claim — true only for the terminal run). It's **two independent levers**: actor over-sharpening (recoverable crater) AND value/return over-estimation runaway (the terminal killer). The critic is a *follower* in the recoverable craters but genuinely *leads* the terminal divergence.
5. **The late lr-anneal (26M) was the WRONG fix** — the value runaway hits at 7M (before any anneal). Robust fix is from-the-start.

**RECOMMENDED NEXT RUN (not started): split the lr + strengthen the entropy floor.**
   - **Lower CRITIC lr 1e-4→3e-5** — directly damps the value/return runaway (the terminal killer; vindicated by `2026-06-28`). Each WM-side stays 1e-4.
   - **Lower ACTOR lr 1e-4→3e-5 AND raise `entropy_floor_weight` 1e-2→~1e-1** — targets the actor-sharpening crater (entropy punched through the floor; the floor penalty is too weak).
   - **Optional 2nd guard on the runaway:** clamp / faster-decay `return/scale`, or shorten `horizon=333` (long horizon = high-variance returns = easier to run away).
   - Code has `opt_wm`/`opt_actor`/`opt_critic` already — add `lr_actor`/`lr_critic` cfg fields (default = `lr`, no-op).

(A late lr anneal CLI exists from this session — `--lr_final/--lr_anneal_start/--lr_anneal_end`, commit `0bfc257`, disabled by default — but the data says split-from-the-start + entropy floor are the levers. Council artifacts: `council-report-2026-06-30_01-50-05.html`.)

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
