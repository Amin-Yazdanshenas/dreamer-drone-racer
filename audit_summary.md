# Phase 1 — Deep Repo Audit Summary

Date: 2026-06-09. Commit audited: `c3e0716`. Auditor: Claude (xhigh).

## Verdict

**The pipeline is sound. No new code bugs found.** The three symptoms
(`return/scale` 48–61, `imag/value_mean` −28, `episode_gates` stalling at 2) are
**reward/horizon design consequences, not bugs.** Prior review cycles (C1, A1, A2, A3,
R1, N1, I1) already fixed the real defects; this audit re-verified every one is applied
end-to-end and traced the symptoms to their (non-bug) arithmetic origin.

## Category-by-category

| # | Category | Result | Evidence |
|---|---|---|---|
| 1 | Value target (C1/A1) | ✅ correct | `agent.py:830-842` — `vals_dep=[V(s_0..s_{H-1}), V(s_H)]`, `lambda_return` bootstraps `values[t+1]`, terminal `V(s_H)`. `agent.py:294-311` lambda_return standard. A1 weight `cumprod` w_0=1 `agent.py:851-855`. repval guarded `agent.py:882`. |
| 2 | Reward scaling | ✅ correct | `_world_model_loss agent.py:716-720` reward target `symlog(reward[:,:-1])`, arrival-aligned `latent_seq[:,1:]`, mask_pair, no double-count. ReturnEMA `networks.py:294-331` p95−p5 EMA. `return/scale≈50` is **raw +30 gate spike** in return space, not a bug. |
| 3 | Gate detection | ✅ correct, no off-by-one | Track keys 1-indexed strings `"1".."7"` (`drone_racer_env_cfg.py:41-47`) → labels `gate_1..gate_7`; `next_gate_idx` 0-indexed 0..6 (`commands.py:164,269`); mask lookup `gate_{idx+1}` (`env_wrapper.py:232`) maps correctly. Gate-frame pass detect via `subtract_frame_transforms`. Confirmed empirically (dump_fpv mask landed on gate). |
| 4 | Imagination rollout | ✅ consistent | `imagine` returns H arrival states s_1..s_H; departure-aligned value series length H+1; terminal bootstrap V(s_H). Eval path (`evaluate_dreamer.py`) uses `agent.act()` — no imagination, no divergence. `imag_horizon=15`, `lam=0.95`, `horizon=333` consistent train-only. |
| 5 | Curriculum annealing | ✅ live | `train_dreamer.py:285,365-386` mutates `cmd_term.cfg.spawn_lerp_alpha`; command reads it live at spawn (logs show 0.9→0.2 over the run). Advance gate: rolling-window mean ≥ ADVANCE_THRESHOLD, MIN_EPISODES_PER_STAGE, window full. |
| 6 | Observation pipeline | ✅ correct | image uint8→float `/255.0` in `agent.act() agent.py:548` and `_preprocess agent.py:617`; replay stores uint8 (`replay_buffer.py`). Camera tilt is a sim-side `OffsetCfg.rot` → rendered RGB already tilted; model sees it. State 16-dim documented. |
| 7 | Eval vs train env | ✅ matched (just fixed) | PLAY now `decimation=12`, `reset_base=None`, `randomise_start=True`, `spawn_lerp_alpha=0.2` matching training; `debug_vis=False` so markers no longer pollute the camera obs. (Commits `c42113d`, `eb1f5ae`.) |
| 8 | Hardcoded constants | ✅ none contradict CLI | `max_steps` default 2M but CLI-overridden; `num_envs` default None→CLI; reward weights in cfg, no shadow hardcode. |

## Non-bug findings (design / tuning levers for Phase 2)

- **F1 (medium):** `gamma`/`horizon=333` was not re-tuned when `decimation` went 4→12.
  Effective discount horizon tripled from ~3.3 s to **~10 s** (333 steps × 0.03 s).
  Over a net-negative dense reward this deepens `imag/value` toward −28
  (≈ −0.085/step × 333). Lowering `horizon` (e.g. 333→150, γ 0.997→0.993) would
  shrink the integrated-negative collapse — but trades off how far ahead the value can
  "see" the next gate. Ambiguous; a loop candidate, not a clear fix. Left as the Dreamer
  default for now.
- **F2 (root cause, design):** the **dense reward is net-negative for non-passing
  states** (signed `progress` < 0 while not closing, `ang_vel` penalty), so the value
  collapses negative; and the **imagination horizon (0.45 s) << gate distance at low
  alpha (~2 s)**, so the actor cannot imagine reaching a gate and can't learn multi-gate
  sequencing. `return/scale≈50` is the raw +30 gate spike vs the negative drift tail.
  **These are the three symptoms, and all are reward/horizon design** — addressed by the
  Phase 2 tuning loop (halve gate spike → return/scale; flyaway 50→15 already committed
  `c3e0716` → bound negative tail; horizon → value collapse depth).

## Codebase health (3 sentences)

The DreamerV3/R2-Dreamer implementation is correct and faithful end-to-end — value
targets, reward/continue alignment, KL, imagination convention, obs normalization, gate
detection, and curriculum all verified against the documented review fixes with no
regressions. The remaining problems are not software defects but reward-shaping and
control-timescale design choices whose arithmetic consequences are the observed metrics.
Progress from here is parameter tuning (gate-spike magnitude, termination distance,
discount horizon), which is exactly what the Phase 2 self-directed loop is built to do.

## Confidence that Phase 1 fixes alone resolve each symptom

| Symptom | Phase-1-alone confidence | Why |
|---|---|---|
| `return/scale` > 20 | **LOW** | No bug to fix; needs the gate-spike halving (Phase 2 intervention 1). |
| `imag/value_mean` < −10 | **LOW** | Reflects net-negative dense reward × long γ-horizon; needs reward/horizon tuning (Phase 2). |
| `episode_gates` stalls at 2 | **LOW** | Imagination-horizon vs gate-distance limit + deterministic undershoot; needs flyaway (done) + tuning + possibly longer imagination. |

**Conclusion:** Phase 1 ruled out code bugs and localized all three symptoms to tunable
design parameters. Hand off to the Phase 2 loop.
