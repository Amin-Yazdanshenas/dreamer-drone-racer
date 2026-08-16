# Copyright (c) 2025, Amin Yazdanshenas
# SPDX-License-Identifier: BSD-3-Clause

"""Turn diagnose_lap_failures.py CSVs into tables + plots + a dominant-failure verdict.

Pure post-processing — no sim, runs anywhere:
    python3 scripts/rl/analyze_lap_diag.py --dir <diag_out_dir>

Writes  <dir>/diag_report.md  and  <dir>/diag_plots.png.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--num_gates", type=int, default=7)
args = p.parse_args()

ep = pd.read_csv(os.path.join(args.dir, "episodes.csv"))
cr = pd.read_csv(os.path.join(args.dir, "crossings.csv"))
ss = pd.read_csv(os.path.join(args.dir, "stepstats.csv"))
NG = args.num_gates
L = []  # report lines


def w(s=""):
    L.append(s)
    print(s)


def pct(x, n):
    return f"{100.0 * x / max(n, 1):.1f}%"


w("# Lap-failure diagnostic report")
w()
modes = list(ep["mode"].unique())
for m in modes:
    e = ep[ep["mode"] == m]
    c = cr[cr["mode"] == m]
    n = len(e)
    lap = (e["gates"] >= NG).sum()
    w(f"## Mode: {m}  ({n} episodes, {len(c)} crossings)")
    w()
    w(f"- lap rate (>= {NG} gates): **{pct(lap, n)}**   mean gates {e['gates'].mean():.2f}  "
      f"median {e['gates'].median():.0f}  max {e['gates'].max()}")
    # conditional lap rates
    s15 = e[e["survived_startup15"] == 1]
    sg1 = e[e["survived_first_gate"] == 1]
    w(f"- lap | survived 15 steps: **{pct((s15['gates'] >= NG).sum(), len(s15))}**  "
      f"(startup deaths: {pct(n - len(s15), n)} of episodes)")
    w(f"- lap | passed >= 1 gate:  **{pct((sg1['gates'] >= NG).sum(), len(sg1))}**")
    w()

    # ---- failure taxonomy ----
    w("### Failure taxonomy")
    w()
    w("| end type | count | share | of which near-gate | on-ground |")
    w("|---|---|---|---|---|")
    for et, grp in e.groupby("end_type"):
        w(f"| {et} | {len(grp)} | {pct(len(grp), n)} | "
          f"{pct((grp['death_near_gate'] == 1).sum(), len(grp))} | "
          f"{pct((grp['death_on_ground'] == 1).sum(), len(grp))} |")
    coll = e[e["end_type"] == "collision"]
    gate_clip = coll[coll["death_near_gate"] == 1]
    w()
    w(f"Gate-frame clips (collision within 2.5 m of target): **{pct(len(gate_clip), n)}** of episodes, "
      f"{pct(len(gate_clip), max(len(coll), 1))} of collisions.")
    w()

    # ---- per-gate table ----
    w("### Per-gate pass rate (attempt = crossing or near-gate death while targeting)")
    w()
    w("| gate | attempts | passes | deaths@gate | pass rate |")
    w("|---|---|---|---|---|")
    for g in range(NG):
        passes = (c["gate_idx"] == g).sum()
        deaths = ((e["death_gate"] == g) & (e["death_near_gate"] == 1)
                  & (e["end_type"] == "collision")).sum()
        att = passes + deaths
        w(f"| {g} | {att} | {passes} | {deaths} | {pct(passes, att)} |")
    w()

    # ---- overlap split ----
    w("### Overlapping-gate vs clean-view attempts")
    w()
    ov_pass = (c["overlap"] == 1).sum()
    cl_pass = (c["overlap"] == 0).sum()
    ov_death = ((e["death_overlap"] == 1) & (e["death_near_gate"] == 1)
                & (e["end_type"] == "collision")).sum()
    cl_death = ((e["death_overlap"] == 0) & (e["death_near_gate"] == 1)
                & (e["end_type"] == "collision")).sum()
    w("| view | attempts | passes | pass rate |")
    w("|---|---|---|---|")
    w(f"| overlapping | {ov_pass + ov_death} | {ov_pass} | {pct(ov_pass, ov_pass + ov_death)} |")
    w(f"| clean | {cl_pass + cl_death} | {cl_pass} | {pct(cl_pass, cl_pass + cl_death)} |")
    w()

    # ---- staleness ----
    w("### Camera staleness (image age / distance flown on stale frame)")
    w()
    gd = gate_clip
    w("| event | n | img age (steps) mean/p90 | v*dt (m) mean/p90 |")
    w("|---|---|---|---|")
    w(f"| gate pass | {len(c)} | {c['img_age_steps'].mean():.1f} / {c['img_age_steps'].quantile(.9):.0f} "
      f"| {c['vdt_m'].mean():.2f} / {c['vdt_m'].quantile(.9):.2f} |")
    if len(gd):
        w(f"| gate-clip death | {len(gd)} | {gd['death_img_age'].mean():.1f} / "
          f"{gd['death_img_age'].quantile(.9):.0f} | {gd['death_vdt_m'].mean():.2f} / "
          f"{gd['death_vdt_m'].quantile(.9):.2f} |")
    w()

    # ---- center error ----
    w("### Gate-center error at crossing (gate frame: lateral / vertical, m)")
    w()
    ap = c["radial_m"].quantile(0.99)
    w(f"- |lateral| mean {c['lat_m'].abs().mean():.2f}  p90 {c['lat_m'].abs().quantile(.9):.2f}   "
      f"|vertical| mean {c['vert_m'].abs().mean():.2f}  p90 {c['vert_m'].abs().quantile(.9):.2f}")
    w(f"- radial p50 {c['radial_m'].quantile(.5):.2f}  p90 {c['radial_m'].quantile(.9):.2f}  "
      f"p99 {ap:.2f}  (p99 ~= empirical aperture half-width)")
    if len(gate_clip):
        w(f"- at gate-clip deaths: radial mean "
          f"{np.hypot(gate_clip['death_lat_m'], gate_clip['death_vert_m']).mean():.2f} m")
    w()

    # ---- action stats near gates ----
    s = ss[ss["mode"] == m].set_index("bucket")
    if {"near", "far"}.issubset(s.index):
        w("### Action distribution near (<2 m) vs far from target gate")
        w()
        w("| bucket | steps | mean std | mean pre-tanh entropy |")
        w("|---|---|---|---|")
        for b in ("near", "far"):
            r = s.loc[b]
            w(f"| {b} | {int(r['steps'])} | {r['std_sum'] / max(r['steps'], 1):.3f} "
              f"| {r['ent_sum'] / max(r['steps'], 1):.2f} |")
        w()

    # ---- episode length / gates before failure ----
    w(f"### Episode shape: len mean {e['len_steps'].mean():.0f} steps "
      f"({e['len_steps'].mean() * 0.03:.1f}s), gates-before-failure histogram in plots.")
    w()

# ---- mode comparison ----
if len(modes) == 2:
    w("## Sampled vs deterministic")
    w()
    w("| metric | " + " | ".join(modes) + " |")
    w("|---|---|---|")
    rows = {
        "lap rate": lambda e, c: pct((e["gates"] >= NG).sum(), len(e)),
        "mean gates": lambda e, c: f"{e['gates'].mean():.2f}",
        "gate-clip share": lambda e, c: pct(((e["end_type"] == "collision")
                                             & (e["death_near_gate"] == 1)).sum(), len(e)),
        "crossing radial p90": lambda e, c: f"{c['radial_m'].quantile(.9):.2f} m",
    }
    for name, fn in rows.items():
        vals = [fn(ep[ep["mode"] == m], cr[cr["mode"] == m]) for m in modes]
        w(f"| {name} | " + " | ".join(vals) + " |")
    w()

# ---- verdict ----
w("## Dominant-failure evidence ranking")
w()
e0 = ep[ep["mode"] == modes[0]]
c0 = cr[cr["mode"] == modes[0]]
n0 = len(e0)
clip_share = ((e0["end_type"] == "collision") & (e0["death_near_gate"] == 1)).sum() / max(n0, 1)
startup_share = (e0["survived_startup15"] == 0).sum() / max(n0, 1)
p99 = c0["radial_m"].quantile(0.99)
p90 = c0["radial_m"].quantile(0.90)
findings = []
if clip_share > 0.3:
    findings.append((clip_share, f"GATE-FRAME PRECISION: {clip_share*100:.0f}% of episodes end as a "
                                 f"collision within 2.5 m of the target gate — the compounding "
                                 f"per-gate clip is the main lap killer."))
if p90 / max(p99, 1e-6) > 0.75:
    findings.append((0.5, f"MARGIN EXHAUSTION: crossing radial p90 ({p90:.2f} m) is already "
                          f"{100*p90/p99:.0f}% of the empirical aperture ({p99:.2f} m) — typical "
                          f"crossings leave almost no clearance."))
if startup_share > 0.08:
    findings.append((startup_share, f"SPAWN DEATHS: {startup_share*100:.0f}% of episodes die inside "
                                    f"15 steps — free lap-rate points if fixed."))
gd0 = e0[(e0["end_type"] == "collision") & (e0["death_near_gate"] == 1)]
if len(gd0) and len(c0) and gd0["death_vdt_m"].mean() > 1.25 * c0["vdt_m"].mean():
    findings.append((0.4, f"STALENESS: gate-clip deaths happen on older frames "
                          f"(v*dt {gd0['death_vdt_m'].mean():.2f} m vs {c0['vdt_m'].mean():.2f} m "
                          f"at successful crossings) — visual latency contributes."))
for _, f in sorted(findings, reverse=True):
    w(f"1. {f}")
w()

with open(os.path.join(args.dir, "diag_report.md"), "w") as f:
    f.write("\n".join(L) + "\n")

# ---------------- plots ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
m0 = modes[0]
e0 = ep[ep["mode"] == m0]
c0 = cr[cr["mode"] == m0]

axx = axes[0, 0]
rates, deaths_g = [], []
for g in range(NG):
    ps = (c0["gate_idx"] == g).sum()
    de = ((e0["death_gate"] == g) & (e0["death_near_gate"] == 1)
          & (e0["end_type"] == "collision")).sum()
    rates.append(100 * ps / max(ps + de, 1))
    deaths_g.append(de)
axx.bar(range(NG), rates, color="#4c9be8")
axx.set_title(f"per-gate pass rate ({m0})"); axx.set_xlabel("gate"); axx.set_ylabel("%")
axx.set_ylim(min(80, min(rates) - 3), 100)
for g, d in enumerate(deaths_g):
    axx.text(g, rates[g] + 0.3, f"{d}✕", ha="center", fontsize=8)

axx = axes[0, 1]
tax = e0["end_type"].value_counts()
clip = ((e0["end_type"] == "collision") & (e0["death_near_gate"] == 1)).sum()
labels = ["gate-clip", "other collision", "flyaway", "timeout(alive)"]
vals = [clip, tax.get("collision", 0) - clip, tax.get("flyaway", 0), tax.get("timeout", 0)]
axx.bar(labels, vals, color=["#e8554c", "#e89b4c", "#b04ce8", "#4ce87a"])
axx.set_title("episode end taxonomy"); axx.tick_params(axis="x", rotation=15)

axx = axes[0, 2]
axx.scatter(c0["lat_m"], c0["vert_m"], s=8, alpha=0.4, label="pass", color="#4c9be8")
gd = e0[(e0["end_type"] == "collision") & (e0["death_near_gate"] == 1)]
axx.scatter(gd["death_lat_m"], gd["death_vert_m"], s=14, alpha=0.8, label="clip death",
            color="#e8554c")
axx.set_title("gate-frame offsets (m)"); axx.set_xlabel("lateral"); axx.set_ylabel("vertical")
axx.axhline(0, lw=0.5, c="gray"); axx.axvline(0, lw=0.5, c="gray")
axx.legend(); axx.set_aspect("equal")

axx = axes[1, 0]
axx.hist(c0["vdt_m"], bins=25, alpha=0.6, label="pass", color="#4c9be8", density=True)
if len(gd):
    axx.hist(gd["death_vdt_m"], bins=25, alpha=0.6, label="clip death", color="#e8554c",
             density=True)
axx.set_title("distance flown on stale frame (m)"); axx.legend()

axx = axes[1, 1]
axx.hist(e0["gates"], bins=range(0, int(e0["gates"].max()) + 2), color="#4c9be8",
         rwidth=0.9)
axx.axvline(NG - 0.5, color="#e8554c", ls="--", label="lap")
axx.set_title("gates reached per episode"); axx.legend()

axx = axes[1, 2]
if len(modes) == 2:
    laps = [100 * (ep[ep["mode"] == m]["gates"] >= NG).mean() for m in modes]
    axx.bar(modes, laps, color=["#4c9be8", "#8a6ce8"])
    axx.set_title("lap rate by action mode"); axx.set_ylabel("%")
else:
    axx.axis("off")

fig.suptitle("Lap-failure diagnostics", fontsize=14)
fig.tight_layout()
out_png = os.path.join(args.dir, "diag_plots.png")
fig.savefig(out_png, dpi=130)
print(f"\nwrote {os.path.join(args.dir, 'diag_report.md')} and {out_png}")
