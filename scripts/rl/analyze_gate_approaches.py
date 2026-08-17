# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Focused approach analysis for the hard gates (default 2,3,4).

Consumes approaches.csv (+ crossings/episodes) from diagnose_lap_failures.py and answers:
does the residual failure come from LATE ACTION NOISE, EARLIER ALIGNMENT/PLANNING ERROR,
or VISUAL AMBIGUITY?

Method: every pass / gate-frame clip contributes its last ~1.2 s of per-step state, aligned
at t=0 (the event). For each metric we compare pass vs clip at each time offset with Welch's
t-test and report the EARLIEST sustained divergence (>=3 consecutive bins at p<0.05) — an
early divergence means the approach was already wrong (planning/alignment); a divergence
only in the last ~0.15 s means the trajectory was fine until late noise.

    python3 scripts/rl/analyze_gate_approaches.py --dir <diag_dir> [--gates 2 3 4]

Writes <dir>/approach_report.md and <dir>/approach_gate<N>.png (one per gate) + a combined
divergence summary.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--gates", type=int, nargs="+", default=[2, 3, 4])
p.add_argument("--focus", type=int, default=2, help="Gate to analyse in most depth.")
p.add_argument("--num_gates", type=int, default=7)
args = p.parse_args()

ap = pd.read_csv(os.path.join(args.dir, "approaches.csv"))
cr = pd.read_csv(os.path.join(args.dir, "crossings.csv"))
ep = pd.read_csv(os.path.join(args.dir, "episodes.csv"))
NG = args.num_gates
L = []


def w(s=""):
    L.append(s)
    print(s)


# Metrics analysed for divergence. (column, label, unit)
METRICS = [
    ("dist_m", "distance to gate", "m"),
    ("lat_m", "lateral error", "m"),
    ("vert_m", "vertical error", "m"),
    ("radial", "radial error", "m"),
    ("speed_mps", "speed", "m/s"),
    ("head_err_deg", "heading vs gate axis", "deg"),
    ("bearing_err_deg", "bearing to gate", "deg"),
    ("vel_align_deg", "velocity vs gate axis", "deg"),
    ("noise_mag", "realized action noise |exec-mean|", ""),
    ("act_std", "actor action std", ""),
    ("pretanh_ent", "pre-tanh entropy", "nats"),
    ("img_age_steps", "image age", "steps"),
    ("vdt_m", "v*dt on stale frame", "m"),
    ("overlap_frac", "gate overlap fraction", ""),
]

ap["radial"] = np.hypot(ap["lat_m"], ap["vert_m"])

w("# Gate 2-4 approach analysis (pass vs gate-frame clip)")
w()
w(f"Source: {len(ap)} approach-steps from "
  f"{ap.groupby(['mode', 'event_id']).ngroups} events.")
w()

# ---------------------------------------------------------------- per-gate mode comparison
w("## Per-gate results: deterministic vs stochastic")
w()
w("| gate | stoch attempts | stoch pass% | det attempts | det pass% | delta |")
w("|---|---|---|---|---|---|")
for g in range(NG):
    row = [f"| {g} "]
    rates = {}
    for m in ("stochastic", "deterministic"):
        c = cr[(cr["mode"] == m) & (cr["gate_idx"] == g)]
        d = ep[(ep["mode"] == m) & (ep["death_gate"] == g)
               & (ep["death_near_gate"] == 1) & (ep["end_type"] == "collision")]
        att = len(c) + len(d)
        rates[m] = 100.0 * len(c) / max(att, 1)
        row.append(f"| {att} | {rates[m]:.1f}% ")
    row.append(f"| {rates['deterministic'] - rates['stochastic']:+.1f} pt |")
    w("".join(row))
w()

# ------------------------------------------------- conditional pass probability / chain
w("## Conditional pass probability P(G_i | reached G_i) and lap chain")
w()
for m in ("stochastic", "deterministic"):
    ps, chain = [], 1.0
    for g in range(NG):
        c = len(cr[(cr["mode"] == m) & (cr["gate_idx"] == g)])
        d = len(ep[(ep["mode"] == m) & (ep["death_gate"] == g)
                   & (ep["death_near_gate"] == 1) & (ep["end_type"] == "collision")])
        pr = c / max(c + d, 1)
        ps.append(pr)
        chain *= pr
    e = ep[ep["mode"] == m]
    obs_lap = (e["gates"] >= NG).mean()
    w(f"**{m}**: per-gate " + " ".join(f"G{g}={100*x:.1f}%" for g, x in enumerate(ps)))
    w(f"  - chained 7-gate product = **{100*chain:.1f}%**  (observed lap rate {100*obs_lap:.1f}%)")
    w(f"  - geometric-mean per-gate survival = {100*np.prod(ps)**(1/NG):.2f}%")
w()

# ------------------------------------------------------------------ overlap for gates 2-4
w("## Visual overlap, gates 2-4 (other-gate pixels intruding on the target mask)")
w()
w("| gate | mode | attempts | overlap-attempt share | pass% overlap | pass% clean | penalty |")
w("|---|---|---|---|---|---|---|")
for g in args.gates:
    for m in ("stochastic", "deterministic"):
        c = cr[(cr["mode"] == m) & (cr["gate_idx"] == g)]
        d = ep[(ep["mode"] == m) & (ep["death_gate"] == g)
               & (ep["death_near_gate"] == 1) & (ep["end_type"] == "collision")]
        ov_p, cl_p = (c["overlap"] == 1).sum(), (c["overlap"] == 0).sum()
        ov_d, cl_d = (d["death_overlap"] == 1).sum(), (d["death_overlap"] == 0).sum()
        ov_rate = 100.0 * (ov_p + ov_d) / max(len(c) + len(d), 1)
        pov = 100.0 * ov_p / max(ov_p + ov_d, 1)
        pcl = 100.0 * cl_p / max(cl_p + cl_d, 1)
        w(f"| {g} | {m[:5]} | {len(c)+len(d)} | {ov_rate:.1f}% | {pov:.1f}% | {pcl:.1f}% "
          f"| {pov-pcl:+.1f} pt |")
w()

# ------------------------------------------------------------- divergence analysis per gate
def divergence(sub_p: pd.DataFrame, sub_c: pd.DataFrame, col: str):
    """Earliest sustained (>=3 bins, p<0.05) divergence time in seconds, plus effect size."""
    ts = sorted(sub_p["t_steps"].unique())
    run, first = 0, None
    for t in ts:
        a = sub_p[sub_p["t_steps"] == t][col].dropna()
        b = sub_c[sub_c["t_steps"] == t][col].dropna()
        if len(a) < 5 or len(b) < 5:
            run, first = 0, None
            continue
        _, pv = stats.ttest_ind(a, b, equal_var=False)
        if pv < 0.05:
            run += 1
            if run == 1:
                first = t
            if run >= 3:
                pooled = np.sqrt((a.var() + b.var()) / 2) or 1e-9
                return first * 0.03, (b.mean() - a.mean()) / pooled
        else:
            run, first = 0, None
    return None, None


w("## When do successful and failed approaches diverge?")
w()
w("Earliest sustained divergence (>=3 consecutive 30 ms bins at p<0.05, Welch). "
  "t is seconds before the event; effect = (clip - pass) / pooled sd.")
w()
summary_rows = []
for g in args.gates:
    sub = ap[ap["gate_idx"] == g]
    sp = sub[sub["outcome"] == "pass"]
    sc = sub[sub["outcome"] == "clip"]
    w(f"### Gate {g}  ({sp['event_id'].nunique()} passes, {sc['event_id'].nunique()} clips)")
    w()
    if sc["event_id"].nunique() < 5:
        w("_too few clips for a reliable test_")
        w()
        continue
    w("| metric | first divergence | effect (clip-pass) | clip mean @ t=-1.0s | clip mean @ t=0 |")
    w("|---|---|---|---|---|")
    for col, lab, unit in METRICS:
        if col not in sub.columns:
            continue
        t0, eff = divergence(sp, sc, col)
        m_far = sc[sc["t_steps"].between(-35, -30)][col].mean()
        m_end = sc[sc["t_steps"] >= -2][col].mean()
        tt = f"**{t0:+.2f} s**" if t0 is not None else "—"
        ee = f"{eff:+.2f}" if eff is not None else "—"
        w(f"| {lab} ({unit}) | {tt} | {ee} | {m_far:.2f} | {m_end:.2f} |")
        if t0 is not None:
            summary_rows.append((g, lab, t0, eff))
    w()

# ------------------------------------------------------------------------------- verdict
w("## Verdict: late noise vs earlier alignment vs visual ambiguity")
w()
if summary_rows:
    early = [r for r in summary_rows if r[2] <= -0.5]
    late = [r for r in summary_rows if r[2] > -0.25]
    geo_cols = {"lateral error", "vertical error", "radial error", "heading vs gate axis",
                "bearing to gate", "velocity vs gate axis"}
    noise_cols = {"realized action noise |exec-mean|", "actor action std", "pre-tanh entropy"}
    vis_cols = {"gate overlap fraction", "image age", "v*dt on stale frame"}
    early_geo = [r for r in early if r[1].split(" (")[0] in geo_cols]
    late_noise = [r for r in late if r[1].split(" (")[0] in noise_cols]
    vis_any = [r for r in summary_rows if r[1].split(" (")[0] in vis_cols]
    w(f"- geometry/alignment diverging EARLY (<= -0.5 s): **{len(early_geo)}** metrics"
      + (f" — earliest {min(r[2] for r in early_geo):+.2f} s ({min(early_geo, key=lambda r: r[2])[1]})"
         if early_geo else ""))
    w(f"- action-noise metrics diverging LATE (> -0.25 s): **{len(late_noise)}**")
    w(f"- visual-ambiguity metrics diverging at all: **{len(vis_any)}**")
    w()
    if early_geo and len(early_geo) >= max(1, len(late_noise)):
        w("**=> EARLIER ALIGNMENT / PLANNING ERROR dominates.** Failed approaches are already "
          "geometrically distinct well before the gate; late action noise is not the trigger.")
    elif late_noise:
        w("**=> LATE ACTION NOISE dominates.** Trajectories are statistically identical until "
          "the final fraction of a second.")
    if vis_any:
        w(f"  Visual-ambiguity signal present in: {sorted({r[1] for r in vis_any})}")
else:
    w("No sustained divergence found in any metric.")
w()

with open(os.path.join(args.dir, "approach_report.md"), "w") as f:
    f.write("\n".join(L) + "\n")

# --------------------------------------------------------------------------------- plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_METRICS = [("dist_m", "distance to gate (m)"), ("radial", "radial center error (m)"),
                ("lat_m", "lateral error (m)"), ("vert_m", "vertical error (m)"),
                ("head_err_deg", "heading vs gate axis (deg)"),
                ("bearing_err_deg", "bearing to gate (deg)"),
                ("speed_mps", "speed (m/s)"), ("noise_mag", "realized action noise"),
                ("act_std", "actor action std"), ("overlap_frac", "gate overlap fraction"),
                ("vdt_m", "v*dt stale frame (m)"), ("pretanh_ent", "pre-tanh entropy")]

for g in args.gates:
    sub = ap[ap["gate_idx"] == g]
    sp, sc = sub[sub["outcome"] == "pass"], sub[sub["outcome"] == "clip"]
    if sc["event_id"].nunique() < 3:
        continue
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    for axx, (col, lab) in zip(axes.ravel(), PLOT_METRICS):
        if col not in sub.columns:
            axx.axis("off")
            continue
        for df, c, name in ((sp, "#2c7fb8", "pass"), (sc, "#e8554c", "clip")):
            grp = df.groupby("t_steps")[col]
            t = np.array(sorted(grp.groups.keys())) * 0.03
            mu, se = grp.mean().values, grp.sem().values
            axx.plot(t, mu, color=c, lw=2, label=name)
            axx.fill_between(t, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.22)
        t0, _ = divergence(sp, sc, col)
        if t0 is not None:
            axx.axvline(t0, color="k", ls=":", lw=1.5)
            axx.text(t0, axx.get_ylim()[1], f" {t0:+.2f}s", va="top", fontsize=8)
        axx.axvline(0, color="gray", lw=0.8)
        axx.set_title(lab, fontsize=10)
        axx.set_xlabel("time before event (s)")
        axx.legend(fontsize=8)
    fig.suptitle(f"Gate {g}: successful vs gate-frame-clip approaches "
                 f"({sp['event_id'].nunique()} pass / {sc['event_id'].nunique()} clip) — "
                 f"dotted line = first sustained divergence", fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.dir, f"approach_gate{g}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")

print(f"wrote {os.path.join(args.dir, 'approach_report.md')}")
