# Gate 2-4 approach analysis (pass vs gate-frame clip)

Source: 85405 approach-steps from 2252 events.

## Per-gate results: deterministic vs stochastic

| gate | stoch attempts | stoch pass% | det attempts | det pass% | delta |
|---|---|---|---|---|---|
| 0 | 131 | 94.7% | 205 | 99.0% | +4.4 pt |
| 1 | 133 | 91.7% | 218 | 100.0% | +8.3 pt |
| 2 | 125 | 88.8% | 211 | 90.0% | +1.2 pt |
| 3 | 125 | 92.0% | 190 | 95.3% | +3.3 pt |
| 4 | 123 | 90.2% | 184 | 93.5% | +3.2 pt |
| 5 | 122 | 93.4% | 175 | 97.7% | +4.3 pt |
| 6 | 127 | 92.9% | 183 | 100.0% | +7.1 pt |

## Conditional pass probability P(G_i | reached G_i) and lap chain

**stochastic**: per-gate G0=94.7% G1=91.7% G2=88.8% G3=92.0% G4=90.2% G5=93.4% G6=92.9%
  - chained 7-gate product = **55.6%**  (observed lap rate 50.8%)
  - geometric-mean per-gate survival = 91.95%
**deterministic**: per-gate G0=99.0% G1=100.0% G2=90.0% G3=95.3% G4=93.5% G5=97.7% G6=100.0%
  - chained 7-gate product = **77.6%**  (observed lap rate 81.7%)
  - geometric-mean per-gate survival = 96.44%

## Visual overlap, gates 2-4 (other-gate pixels intruding on the target mask)

| gate | mode | attempts | overlap-attempt share | pass% overlap | pass% clean | penalty |
|---|---|---|---|---|---|---|
| 2 | stoch | 125 | 9.6% | 83.3% | 89.4% | -6.0 pt |
| 2 | deter | 211 | 4.7% | 70.0% | 91.0% | -21.0 pt |
| 3 | stoch | 125 | 21.6% | 92.6% | 91.8% | +0.8 pt |
| 3 | deter | 190 | 28.4% | 98.1% | 94.1% | +4.0 pt |
| 4 | stoch | 123 | 13.8% | 100.0% | 88.7% | +11.3 pt |
| 4 | deter | 184 | 9.8% | 94.4% | 93.4% | +1.1 pt |

## When do successful and failed approaches diverge?

Earliest sustained divergence (>=3 consecutive 30 ms bins at p<0.05, Welch). t is seconds before the event; effect = (clip - pass) / pooled sd.

### Gate 2  (301 passes, 35 clips)

| metric | first divergence | effect (clip-pass) | clip mean @ t=-1.0s | clip mean @ t=0 |
|---|---|---|---|---|
| distance to gate (m) | **-1.02 s** | +0.87 | 8.88 | 1.16 |
| lateral error (m) | **-0.87 s** | -1.07 | -3.93 | -0.93 |
| vertical error (m) | **-0.45 s** | +0.42 | 0.05 | -0.13 |
| radial error (m) | **-1.02 s** | +0.96 | 5.99 | 1.05 |
| speed (m/s) | **-0.96 s** | -0.75 | 7.23 | 7.22 |
| heading vs gate axis (deg) | **-1.17 s** | -0.69 | 71.18 | 69.57 |
| bearing to gate (deg) | **-1.17 s** | -0.92 | 66.73 | 103.95 |
| velocity vs gate axis (deg) | **-0.96 s** | +0.61 | 66.94 | 48.36 |
| realized action noise |exec-mean| () | — | — | 0.19 | 0.14 |
| actor action std () | **-0.99 s** | -0.80 | 0.79 | 0.78 |
| pre-tanh entropy (nats) | **-1.02 s** | -1.04 | 4.65 | 4.61 |
| image age (steps) | — | — | 7.01 | 7.49 |
| v*dt on stale frame (m) | — | — | 2.04 | 2.49 |
| gate overlap fraction () | **-0.87 s** | +0.44 | 0.35 | 0.17 |

### Gate 3  (296 passes, 19 clips)

| metric | first divergence | effect (clip-pass) | clip mean @ t=-1.0s | clip mean @ t=0 |
|---|---|---|---|---|
| distance to gate (m) | **-0.66 s** | +0.61 | 12.54 | 1.12 |
| lateral error (m) | — | — | -0.32 | -0.01 |
| vertical error (m) | — | — | -1.65 | 0.04 |
| radial error (m) | **-0.99 s** | +0.76 | 3.20 | 0.86 |
| speed (m/s) | **-1.05 s** | -0.78 | 7.88 | 12.63 |
| heading vs gate axis (deg) | **-1.17 s** | -1.15 | 63.84 | 95.38 |
| bearing to gate (deg) | **-0.87 s** | -0.51 | 81.91 | 101.67 |
| velocity vs gate axis (deg) | **-0.75 s** | +0.65 | 36.99 | 47.53 |
| realized action noise |exec-mean| () | — | — | 0.20 | 0.17 |
| actor action std () | **-1.17 s** | -0.99 | 0.74 | 0.67 |
| pre-tanh entropy (nats) | **-1.17 s** | -1.02 | 4.42 | 3.98 |
| image age (steps) | **-0.33 s** | -0.39 | 5.58 | 4.86 |
| v*dt on stale frame (m) | **-0.99 s** | -0.39 | 1.39 | 2.75 |
| gate overlap fraction () | **-0.51 s** | -0.22 | 0.00 | 0.47 |

### Gate 4  (283 passes, 24 clips)

| metric | first divergence | effect (clip-pass) | clip mean @ t=-1.0s | clip mean @ t=0 |
|---|---|---|---|---|
| distance to gate (m) | **-0.39 s** | +0.96 | 3.11 | 1.83 |
| lateral error (m) | — | — | 1.29 | -0.01 |
| vertical error (m) | **-0.57 s** | +1.33 | 0.98 | 1.30 |
| radial error (m) | **-0.60 s** | +1.55 | 2.86 | 1.72 |
| speed (m/s) | **-0.60 s** | +1.47 | 5.01 | 2.78 |
| heading vs gate axis (deg) | **-0.18 s** | -1.21 | 97.18 | 28.61 |
| bearing to gate (deg) | **-0.09 s** | +0.66 | 37.42 | 89.55 |
| velocity vs gate axis (deg) | **-0.54 s** | +1.94 | 78.21 | 88.76 |
| realized action noise |exec-mean| () | — | — | 0.12 | 0.17 |
| actor action std () | **-0.15 s** | +1.72 | 0.76 | 0.86 |
| pre-tanh entropy (nats) | **-0.15 s** | +1.69 | 4.52 | 5.03 |
| image age (steps) | — | — | 13.33 | 4.28 |
| v*dt on stale frame (m) | — | — | 3.10 | 0.61 |
| gate overlap fraction () | **-0.60 s** | -0.39 | 0.00 | 0.07 |

## Verdict: late noise vs earlier alignment vs visual ambiguity

- geometry/alignment diverging EARLY (<= -0.5 s): **12** metrics — earliest -1.17 s (heading vs gate axis)
- action-noise metrics diverging LATE (> -0.25 s): **2**
- visual-ambiguity metrics diverging at all: **5**

**=> EARLIER ALIGNMENT / PLANNING ERROR dominates.** Failed approaches are already geometrically distinct well before the gate; late action noise is not the trigger.
  Visual-ambiguity signal present in: ['gate overlap fraction', 'image age', 'v*dt on stale frame']

