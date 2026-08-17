# Lap-failure diagnostic report

## Mode: stochastic  (120 episodes, 811 crossings)

- lap rate (>= 7 gates): **50.8%**   mean gates 6.62  median 7  max 15
- lap | survived 15 steps: **56.5%**  (startup deaths: 10.0% of episodes)
- lap | passed >= 1 gate:  **59.2%**

### Failure taxonomy

| end type | count | share | of which near-gate | on-ground |
|---|---|---|---|---|
| collision | 103 | 85.8% | 68.9% | 14.6% |
| flyaway | 2 | 1.7% | 0.0% | 0.0% |
| timeout | 15 | 12.5% | 46.7% | 0.0% |

Gate-frame clips (collision within 2.5 m of target): **59.2%** of episodes, 68.9% of collisions.

### Per-gate pass rate (attempt = crossing or near-gate death while targeting)

| gate | attempts | passes | deaths@gate | pass rate |
|---|---|---|---|---|
| 0 | 127 | 120 | 7 | 94.5% |
| 1 | 137 | 127 | 10 | 92.7% |
| 2 | 122 | 109 | 13 | 89.3% |
| 3 | 121 | 108 | 13 | 89.3% |
| 4 | 127 | 110 | 17 | 86.6% |
| 5 | 122 | 113 | 9 | 92.6% |
| 6 | 126 | 124 | 2 | 98.4% |

### Overlapping-gate vs clean-view attempts

| view | attempts | passes | pass rate |
|---|---|---|---|
| overlapping | 100 | 87 | 87.0% |
| clean | 782 | 724 | 92.6% |

### Camera staleness (image age / distance flown on stale frame)

| event | n | img age (steps) mean/p90 | v*dt (m) mean/p90 |
|---|---|---|---|
| gate pass | 811 | 6.4 / 14 | 1.49 / 3.66 |
| gate-clip death | 71 | 5.5 / 11 | 1.66 / 4.07 |

### Gate-center error at crossing (gate frame: lateral / vertical, m)

- |lateral| mean 0.19  p90 0.42   |vertical| mean 0.16  p90 0.34
- radial p50 0.24  p90 0.51  p99 0.72  (p99 ~= empirical aperture half-width)
- at gate-clip deaths: radial mean 1.19 m

### Action distribution near (<2 m) vs far from target gate

| bucket | steps | mean std | mean pre-tanh entropy |
|---|---|---|---|
| near | 12492 | 0.747 | 4.42 |
| far | 29020 | 0.816 | 4.81 |

### Episode shape: len mean 339 steps (10.2s), gates-before-failure histogram in plots.

## Mode: deterministic  (120 episodes, 1232 crossings)

- lap rate (>= 7 gates): **75.8%**   mean gates 10.15  median 11  max 16
- lap | survived 15 steps: **81.2%**  (startup deaths: 6.7% of episodes)
- lap | passed >= 1 gate:  **82.7%**

### Failure taxonomy

| end type | count | share | of which near-gate | on-ground |
|---|---|---|---|---|
| collision | 73 | 60.8% | 63.0% | 12.3% |
| flyaway | 1 | 0.8% | 0.0% | 0.0% |
| timeout | 46 | 38.3% | 39.1% | 2.2% |

Gate-frame clips (collision within 2.5 m of target): **38.3%** of episodes, 63.0% of collisions.

### Per-gate pass rate (attempt = crossing or near-gate death while targeting)

| gate | attempts | passes | deaths@gate | pass rate |
|---|---|---|---|---|
| 0 | 182 | 182 | 0 | 100.0% |
| 1 | 190 | 190 | 0 | 100.0% |
| 2 | 200 | 180 | 20 | 90.0% |
| 3 | 181 | 170 | 11 | 93.9% |
| 4 | 172 | 159 | 13 | 92.4% |
| 5 | 173 | 171 | 2 | 98.8% |
| 6 | 180 | 180 | 0 | 100.0% |

### Overlapping-gate vs clean-view attempts

| view | attempts | passes | pass rate |
|---|---|---|---|
| overlapping | 152 | 142 | 93.4% |
| clean | 1126 | 1090 | 96.8% |

### Camera staleness (image age / distance flown on stale frame)

| event | n | img age (steps) mean/p90 | v*dt (m) mean/p90 |
|---|---|---|---|
| gate pass | 1232 | 6.6 / 15 | 1.40 / 3.60 |
| gate-clip death | 46 | 5.8 / 10 | 1.65 / 3.35 |

### Gate-center error at crossing (gate frame: lateral / vertical, m)

- |lateral| mean 0.13  p90 0.29   |vertical| mean 0.12  p90 0.25
- radial p50 0.17  p90 0.40  p99 0.67  (p99 ~= empirical aperture half-width)
- at gate-clip deaths: radial mean 1.14 m

### Action distribution near (<2 m) vs far from target gate

| bucket | steps | mean std | mean pre-tanh entropy |
|---|---|---|---|
| near | 18675 | 0.754 | 4.47 |
| far | 38749 | 0.839 | 4.93 |

### Episode shape: len mean 473 steps (14.2s), gates-before-failure histogram in plots.

## Sampled vs deterministic

| metric | stochastic | deterministic |
|---|---|---|
| lap rate | 50.8% | 75.8% |
| mean gates | 6.62 | 10.15 |
| gate-clip share | 59.2% | 38.3% |
| crossing radial p90 | 0.51 m | 0.40 m |

## Dominant-failure evidence ranking

1. GATE-FRAME PRECISION: 59% of episodes end as a collision within 2.5 m of the target gate — the compounding per-gate clip is the main lap killer.
1. SPAWN DEATHS: 10% of episodes die inside 15 steps — free lap-rate points if fixed.

