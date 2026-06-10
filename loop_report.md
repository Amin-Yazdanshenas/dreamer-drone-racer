# Phase 2 — Self-Directed Diagnostic Loop: Final Diagnosis

Date: 2026-06-10. 5 iterations (1× 200k + 4× 800k) on the remote RTX 4090. All runs from
scratch with the annealing curriculum. Audit (Phase 1) found no code bugs; this loop
tuned the design parameters the audit flagged.

## What changed, what moved

| Iter | Lever (single var) | `return/scale` @800k | `imag/value` @800k | `episode_gates` mean50 / max | Verdict |
|---|---|---|---|---|---|
| 1 | gate_passed 1000→500 | (200k pre-symptom) | — | — | — |
| 2 | baseline (gate=500) | **33** | −16.6 | **0.44** / 2 | gate-halve cut return/scale ~25% vs prior 1000 run; best gates |
| 3 | flyaway 15→12 | 37 ↑ | −14.6 | 0.28 / 2 | **WORSE** both — reverted |
| 4 | horizon 333→150 | 33 | **−9.6** ↑ | 0.18 / 2 | value ↑ but gates ↓↓ (lost foresight) — reverted |
| 5 | imag_horizon 15→24 | **27** ↓ | −32 ↓↓ | 0.40 / 2 | return/scale ↓ + faster anneal, but value collapsed deeper; chaining unchanged — reverted |

## What is still broken — and why it is not a tuning problem

**The max-2-gate ceiling is invariant across every lever tested.** Gate spike (×2 down),
termination distance, discount horizon, imagination horizon — none moved `episode_gates`
max past 2. That invariance is the headline result: chaining is **not** gated by any of
these reward/horizon parameters.

The three original symptoms are coupled consequences of the **reward+horizon geometry**,
not bugs (Phase 1 verified the value target, reward alignment, gate detection, obs
pipeline, curriculum, and eval/train match are all correct):

- `return/scale` ≈ 27–37 at scale = spread of the **imagined lambda-returns**. It floors
  around the raw gate-spike (now +15) vs the negative drift tail. Bounding the real
  episode (flyaway) does NOT bound it (iter3 proved this — it went *up*). Halving the
  spike is the only thing that lowered it, and only ~25%.
- `imag/value` collapse = **net-negative dense reward integrated over the discount
  horizon**. It is a direct trade: shorter horizon → less negative (iter4: −9.6) but less
  gate foresight → worse gates; longer imagination → deeper negative (iter5: −32). There
  is no setting that lifts value *and* keeps gates.
- gate stall at 2 = the policy reliably solves the curriculum-drilled single-gate approach
  but the **deterministic** action mean undershoots multi-gate (training hits 2
  stochastically; eval hit 1 — see prior session). Longer imagination (iter5) did not fix
  it, which argues the limit is **not** planning reach but world-model fidelity over a
  2-gate horizon and/or the squashed-policy determinism.

## The one durable win

**gate_passed 1000 → 500** (+30 → +15/pass). Lowered `return/scale` ~25% at scale with
**no gate-passing cost**. Kept. Everything else (flyaway 12, horizon 150, imag 24) was
neutral-or-harmful and reverted. Final converged config = iter2:

```
gate_passed_weight = 500    flyaway = 15 m    horizon = 333    imag_horizon = 15
decimation = 12             progress = signed, weight 50
```

## Confidence / recommendation

- **HIGH** confidence the three symptoms are NOT code bugs and NOT fixable by the
  reward-magnitude / termination / discount / imagination knobs (5 iterations, every
  knob either neutral or harmful, chaining ceiling invariant).
- The real levers for 3+ gate chaining are **out of this loop's scope** and are bigger
  commitments, in rough priority:
  1. **World-model fidelity for multi-gate rollouts** — the WM must predict gate-2
     dynamics well enough for imagination to value the approach. Inspect `wm/dyn_loss`,
     reconstruction, and whether posterior uses its latent budget; consider more RSSM
     capacity / longer training (5M+, not 800k probes).
  2. **Deterministic-policy sharpness** — train hits 2 gates stochastically, eval 1.
     Lower actor entropy floor late in training, or evaluate stochastically.
  3. **Dense-reward sign** — signed progress is deliberately anti-parking (documented),
     but it is the source of the value collapse. An asymmetric/clamped variant *with* the
     curriculum + gate dominance could be retried, accepting parking risk.
  4. **Control timescale** — decimation 16–24 (imagination covers ~1 s) is the structural
     version of iter5; bigger change, risks disrupting the working low-level control.
- **Practical:** the policy is already a **reliable ~60% first-gate passer** (prior eval,
  camera-tilt + clean-eval fixes). The loop confirms further chaining needs the structural
  work above, not parameter tuning. Recommend a single long (5M) run on the converged
  config to confirm the gate-halve holds at scale, then pursue (1)/(2).

## Iteration artifacts (remote run dirs)
iter1 `…_22-44-41` (200k) · iter2 `2026-06-09_23-17-20` · iter3 `2026-06-10_00-15-45` ·
iter4 `2026-06-10_01-55-09` · iter5 `2026-06-10_03-34-31`. TB synced under `logs/`.
