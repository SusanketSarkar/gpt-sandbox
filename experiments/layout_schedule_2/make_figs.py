"""Render the ratio-sweep figures from all_runs.json. Run after the sweep completes."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parent
FIGS = EXP / "figs"
FIGS.mkdir(exist_ok=True)
rows = json.loads((EXP / "all_runs.json").read_text())

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# group helpers -------------------------------------------------------------
def group(key_sweep, key="val_loss"):
    g = defaultdict(list)
    for r in rows:
        if r["sweep"] == key_sweep and r.get(key) is not None:
            g[r["nA"]].append(r)
    return g

def stats(vals):
    import statistics as st
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0)

ATTN_FRAC = lambda nA, nM: nA / (nA + nM)

# ---- fig 1: val_loss vs attention fraction (jumbo raw + pm + fine) -----------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for sweep, marker, color in (("ratio_jumbo", "o", "#1f77b4"),
                             ("ratio_pm_jumbo", "s", "#d62728"),
                             ("ratio_fine_jumbo", "x", "#2ca02c")):
    g = group(sweep)
    if not g:
        continue
    xs = sorted(g)
    means, errs = [], []
    for nA in xs:
        m, e = stats([r["val_loss"] for r in g[nA]])
        means.append(m)
        errs.append(e)
    xs = [ATTN_FRAC(nA, g[nA][0]["nM"]) for nA in xs]
    ax.errorbar(xs, means, yerr=errs, marker=marker, color=color, capsize=3,
                label=sweep.replace("ratio_", ""), linewidth=1.5)
ax.axvline(0.5, color="gray", ls=":", lw=1)
ax.text(0.5, ax.get_ylim()[1], " 1:1 anchor", va="top", ha="left", fontsize=8, color="gray")
ax.set_xlabel("attention fraction of sublayers (nA / 48)")
ax.set_ylabel("FineWeb-edu val loss (best ckpt)")
ax.set_title("Language-modeling loss vs attention:MLP ratio (jumbo, 48 sublayers)")
ax.invert_xaxis() if False else None
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig1_val_by_ratio.png")
print("fig1")

# ---- fig 2: convergence curves (jumbo alt) ------------------------------------------------------
g = defaultdict(list)
for r in rows:
    if r["sweep"] == "ratio_jumbo" and r["arr"] == "alt" and r.get("val_curve"):
        g[r["nA"]].append(r)
if g:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cmap = plt.cm.viridis
    for i, nA in enumerate(sorted(g)):
        curve = sorted({(s, v) for r in g[nA] for s, v in r["val_curve"]})
        if not curve:
            continue
        xs = [c[0] for c in curve]
        ys = [c[1] for c in curve]
        ax.plot(xs, ys, color=cmap(i / max(1, len(g) - 1)), lw=1.6,
                label=f"{nA}A/{g[nA][0]['nM']}M")
    ax.set_xlabel("training step")
    ax.set_ylabel("val loss")
    ax.set_title("Convergence by ratio (jumbo, alt arrangement)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_convergence.png")
    print("fig2")

# ---- fig 3: long-range recall by distance, by ratio --------------------------------------------
g = group("ratio_jumbo", "recall_acc/64")
if g:
    dists = [8, 16, 32, 64, 128]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, d in zip(axes, (16, 64)):
        xs, ys, errs = [], [], []
        for nA in sorted(g):
            m, e = stats([r.get(f"recall_acc/{d}") for r in g[nA]])
            if m is None:
                continue
            xs.append(ATTN_FRAC(nA, g[nA][0]["nM"]))
            ys.append(m)
            errs.append(e or 0)
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, color="#1f77b4", linewidth=1.5)
        ax.set_xlabel("attention fraction")
        ax.set_title(f"recall acc at distance {d}")
    axes[0].set_ylabel("P(correct rel. to next token)")
    fig.suptitle("Long-range token retrieval vs attention proportion (jumbo)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_retrieval.png")
    print("fig3")

# ---- fig 4: synthetic task EM vs ratio ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for sweep, label, color in (("ratio_synth_recall", "one-hop recall (EM)", "#1f77b4"),
                            ("ratio_synth_hop2", "two-hop chain (EM)", "#ff7f0e"),
                            ("ratio_synth_local", "local rule (EM)", "#9467bd")):
    g = group(sweep, "task_em")
    if not g:
        continue
    xs, ys, errs = [], [], []
    for nA in sorted(g):
        m, e = stats([r["task_em"] for r in g[nA]])
        xs.append(ATTN_FRAC(nA, g[nA][0]["nM"]))
        ys.append(m)
        errs.append(e or 0)
    ax.errorbar(xs, ys, yerr=errs, marker="o", color=color, capsize=3, label=label, linewidth=1.5)
ax.set_xlabel("attention fraction (24 sublayers)")
ax.set_ylabel("task exact-match")
ax.set_title("Synthetic task accuracy vs attention proportion")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig4_tasks.png")
print("fig4")

# ---- fig 5: depth robustness --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for sweep, color, label in (("ratio_jumbo", "#1f77b4", "48 sublayers (d=512)"),
                            ("ratio_depth24", "#d62728", "24 sublayers (d=256)")):
    g = group(sweep)
    if not g:
        continue
    xs, ys, errs = [], [], []
    for nA in sorted(g):
        m, e = stats([r["val_loss"] for r in g[nA]])
        xs.append(ATTN_FRAC(nA, g[nA][0]["nM"]))
        ys.append(m)
        errs.append(e or 0)
    ax.errorbar(xs, ys, yerr=errs, marker="o", color=color, capsize=3, label=label, linewidth=1.5)
ax.set_xlabel("attention fraction")
ax.set_ylabel("val loss (edu_mix)")
ax.set_title("Does the ratio effect hold across depth?")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig5_depth.png")
print("fig5")

# ---- fig 6: cost side ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for sweep, color, label in (("ratio_jumbo", "#1f77b4", "raw (params vary)"),
                            ("ratio_pm_jumbo", "#d62728", "param-matched")):
    g = group(sweep)
    if not g:
        continue
    xs, ys, errs = [], [], []
    for nA in sorted(g):
        m, e = stats([r["val_loss"] for r in g[nA]])
        xs.append(ATTN_FRAC(nA, g[nA][0]["nM"]))
        ys.append(m)
        errs.append(e or 0)
    ax.errorbar(xs, ys, yerr=errs, marker="o", color=color, capsize=3, label=label, linewidth=1.5)
ax.set_xlabel("attention fraction")
ax.set_ylabel("val loss")
ax.set_title("Effect is not a parameter-count artifact (param-matched control)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig6_params_control.png")
print("fig6")

# ---- fig 7: stability ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
g = defaultdict(list)
for r in rows:
    if r["sweep"] == "ratio_jumbo" and r.get("gnorm_p95") is not None:
        g[r["nA"]].append(r)
if g:
    xs, ys, errs = [], [], []
    for nA in sorted(g):
        m, e = stats([r["gnorm_p95"] for r in g[nA]])
        xs.append(ATTN_FRAC(nA, g[nA][0]["nM"]))
        ys.append(m)
        errs.append(e or 0)
    ax.errorbar(xs, ys, yerr=errs, marker="o", color="#1f77b4", capsize=3, linewidth=1.5)
    ax.set_xlabel("attention fraction")
    ax.set_ylabel("p95 grad-norm")
    ax.set_title("Training stability (jumbo)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_stability.png")
    print("fig7")
print("done. figures in", FIGS)
