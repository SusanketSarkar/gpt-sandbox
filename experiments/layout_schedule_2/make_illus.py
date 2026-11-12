"""Blog illustration figures (layout strips, size-vs-ratio decomposition, recall bimodality).

Robust text placement: all labels sit OUTSIDE the plotted cells/bars so nothing clips.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

EXP = Path(__file__).resolve().parent
FIGS = EXP / "figs"
FIGS.mkdir(exist_ok=True)
rows = json.loads((EXP / "all_runs.json").read_text())

AT = "#3b6fb5"    # attention blue  (matches the HTML blog --attn)
ML = "#e8833a"    # MLP orange      (matches the HTML blog --mlp)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": False})


def alt(nA: int, nM: int) -> str:
    t = nA + nM
    if nA <= nM:
        pos = sorted({int(i * t / nA) for i in range(nA)})
        return "".join("A" if i in pos else "M" for i in range(t))
    return "".join("M" if c == "A" else "A" for c in alt(nM, nA))


# ================= fig0: what a ratio actually looks like (jumbo, 48 sublayers) ================
ratios = [(10, "4:1"), (12, "3:1"), (16, "2:1"), (24, "1:1"), (32, "1:2"), (36, "1:3"), (38, "1:4")]
nrows = len(ratios)
cell_w = 1.0 / 48
fig, ax = plt.subplots(figsize=(9.2, 0.60 * nrows + 1.1))
ax.set_xlim(-0.30, 1.03)
ax.set_ylim(-0.75, nrows - 0.42)
ax.axis("off")
for row, (nA, label) in enumerate(ratios):
    nM = 48 - nA
    s = alt(nA, nM)
    y = nrows - 1 - row                      # row0 (4:1) on top ... row6 (1:4) on bottom
    for i, ch in enumerate(s):
        ax.add_patch(Rectangle((i * cell_w, y - 0.29), cell_w * 0.94, 0.58,
                               facecolor=AT if ch == "A" else ML, edgecolor="none"))
    ax.text(-0.03, y, f"{label}   ({nA}A + {nM}M)", ha="right", va="center", fontsize=9.5)
# depth axis
ax.text(0.0, nrows - 0.40, "sublayer 0", ha="left", va="bottom", fontsize=8, color="#777")
ax.text(1.0, nrows - 0.40, "sublayer 47  →  logits", ha="right", va="bottom", fontsize=8, color="#777")
# legend (below the bottom row, well clear of the cells)
ly = -0.45
ax.add_patch(Rectangle((0.00, ly - 0.24), cell_w * 0.94, 0.48, facecolor=AT, edgecolor="none"))
ax.text(0.00 + cell_w + 0.015, ly, "attention sublayer", va="center", fontsize=8.5)
ax.add_patch(Rectangle((0.26, ly - 0.24), cell_w * 0.94, 0.48, facecolor=ML, edgecolor="none"))
ax.text(0.26 + cell_w + 0.015, ly, "MLP sublayer", va="center", fontsize=8.5)
ax.set_title("What a ratio actually is: the sublayer sequence of each jumbo model (alt arrangement)",
             fontsize=10.5, pad=26)
fig.tight_layout(pad=0.5)
fig.savefig(FIGS / "fig0_layouts.png")
print("fig0")

# ================= fig8: the crux — size vs ratio, as two clean bars ===========================
fig, ax = plt.subplots(figsize=(8.4, 3.1))
SIZE = "#9aa5b1"    # size effect (parameter count)
RATIO = "#d62728"   # genuine ratio effect (at fixed size)
h = 0.34
y1, y0 = 1.0, 0.0
a1, a2, a3 = 0.0588, 0.0038, 0.0118        # raw gap = 0.0588 + 0.0038 + 0.0118 = 0.0744
# bar 1: the raw sweep gap, segmented size | ratio | size
ax.add_patch(Rectangle((0, y1 - h / 2), a1, h, facecolor=SIZE, edgecolor="none"))
ax.add_patch(Rectangle((a1, y1 - h / 2), a2, h, facecolor=RATIO, edgecolor="none"))
ax.add_patch(Rectangle((a1 + a2, y1 - h / 2), a3, h, facecolor=SIZE, edgecolor="none"))
ax.text(a1 + a2 + a3 + 0.006, y1, "0.0744", ha="left", va="center", fontsize=11, fontweight="bold")
ax.text(a1 + a2 + a3 + 0.052, y1, "raw-sweep gap\n(16A vs 38A)", ha="left", va="center",
        fontsize=8.5, color="#555")
# bar 2: the param-matched spread — the entire genuine ratio effect
ax.add_patch(Rectangle((0, y0 - h / 2), 0.0132, h, facecolor=RATIO, edgecolor="none"))
ax.text(0.0132 + 0.006, y0, "0.0132", ha="left", va="center", fontsize=11, fontweight="bold", color=RATIO)
ax.text(0.0132 + 0.052, y0, "param-matched spread (all ~80M)\n= the entire genuine ratio effect",
        ha="left", va="center", fontsize=8.5, color="#555")
# left-side row labels
ax.text(-0.006, y1, "RAW\nsweep", ha="right", va="center", fontsize=9, color="#333")
ax.text(-0.006, y0, "PARAM-\nMATCHED", ha="right", va="center", fontsize=9, color="#333")
ax.set_xlim(-0.115, 0.34)
ax.set_ylim(-0.55, 1.55)
ax.axis("off")
ax.legend(handles=[Patch(facecolor=SIZE, label="size effect  (total parameter count)"),
                   Patch(facecolor=RATIO, label="genuine ratio effect  (at fixed size)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8.5, frameon=False)
ax.set_title("Decomposing the apparent 'ratio effect' — ≈95% is really the parameter count",
             fontsize=10.5, pad=10)
fig.tight_layout(pad=0.5)
fig.savefig(FIGS / "fig8_decomp.png")
print("fig8")

# ================= fig9: recall per-run EM strip (bimodality) =================================
g = defaultdict(list)
for r in rows:
    if r["sweep"] == "ratio_synth_recall" and r.get("task_em") is not None:
        g[r["nA"]].append(r["task_em"])
labels = {5: "4:1 (5A)", 6: "3:1 (6A)", 8: "2:1 (8A)", 12: "1:1 (12A)",
          16: "1:2 (16A)", 18: "1:3 (18A)", 19: "1:4 (19A)"}
xs = sorted(g)
import numpy as np
rng = np.random.default_rng(7)
fig, ax = plt.subplots(figsize=(7.8, 4.2))
for x, nA in enumerate(xs):
    vals = g[nA]
    jit = (rng.random(len(vals)) - 0.5) * 0.10
    for v, j in zip(vals, jit):
        c = "#2d7d2d" if v >= 0.5 else "#b23b3b"
        ax.scatter(x + j, v, s=70, color=c, edgecolor="white", linewidth=0.9, zorder=3)
    mean = sum(vals) / len(vals)
    ax.scatter(x, mean, marker="_", s=240, color="#222", zorder=4)
    ax.text(x, -0.13, f"{mean:.3f}", ha="center", va="top", fontsize=8, color="#555")
ax.axhline(0.5, color="#888", ls="--", lw=1)
ax.set_xlim(-0.6, 6.6)
ax.text(6.55, 0.515, "pass bar (EM 0.5)", fontsize=7.5, color="#666", ha="right")
ax.set_xticks(range(len(xs)))
ax.set_xticklabels([labels[n] for n in xs], fontsize=8.5)
ax.set_ylim(-0.25, 1.10)
ax.set_ylabel("exact-match (per run)")
ax.set_title("One-hop recall: every single run, by ratio  (green = pass, red = fail)", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "fig9_recall_dots.png")
print("fig9")
print("done.")
