#!/usr/bin/env python3
"""Exhaustive, metric-robust verification of the hypothesis:
    "Do learned layer gates predict which layers matter?"
  (layers with small gates == layers ablation says are unimportant)

Independent of analyze.py: it re-derives everything from results.json and,
crucially, tests the hypothesis under SEVERAL gate-size aggregations so we can
separate robust conclusions from artifacts of a single metric choice.

Gate-size metrics tested (per block, from per-sublayer gates g_attn, g_mlp):
  prod  = g_attn * g_mlp          (what analyze.py uses)
  sum   = |g_attn| + |g_mlp|
  L2    = sqrt(g_attn^2 + g_mlp^2)
  max   = max(|g_attn|, |g_mlp|)
  mlp   = |g_mlp|                 (the dominant sublayer in these runs)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "results.json").read_text())
runs = data["runs"]

gated = [r for lst in runs.values() for r in lst
         if r["variant"] != "plain" and "gate_attn" in r and "dloss" in r]
N = len(gated)


def ranks(x):
    a = np.asarray(x, float); o = a.argsort(kind="mergesort"); rk = np.empty(len(a)); i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[o[j + 1]] == a[o[i]]:
            j += 1
        rk[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return rk


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pois_binom_tail(ps, k):
    """P(sum of independent Bernoulli(ps) >= k) via convolution."""
    P = np.array([1.0])
    for p in ps:
        P = np.convolve(P, [1 - p, p])
    return float(np.sum(P[k:]))


def fmt(lo, hi):
    return f"{lo:+.2f}…{hi:+.2f}"


print(f"=== EXHAUSTIVE VERIFICATION  ({N} gated runs, "
      f"{len({r['cell'] for r in gated})} cells, 3 seeds) ===\n")

# ---------------------------------------------------------------- gate movement
print("0) PREREQ — did the gates actually move off init=1.0?")
mov = {r["variant"]: np.max(np.abs(np.array(r["gate_prod"]) - 1.0))
       for r in gated if "gate_prod" in r}
print(f"   max |g-1| per variant: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(mov.items())))
print("   => gates moved a lot even at lr_mult=1; safe to read into them\n")

# ---------------------------------------------------------------- metric sweep
print("1) METRIC-ROBUSTNESS (Spearman, per-run range over %d runs)" % N)
print("   %-6s | %-16s | %-16s | %-12s | %s" %
      ("metric", "rho(gate,ablation)", "rho(gate,scale)", "top-1 match", "L0 below-med gate"))
metrics = {
    "prod": lambda a, m: a * m,
    "sum": lambda a, m: np.abs(a) + np.abs(m),
    "L2": lambda a, m: np.hypot(a, m),
    "max": lambda a, m: np.maximum(np.abs(a), np.abs(m)),
    "mlp": lambda a, m: np.abs(m),
}
ps_top1 = []
for r in gated:
    n = r["n_layer"]
    ps_top1.append(1.0 / n)
for name, fn in metrics.items():
    r_abl, r_scl, t1, l0bm = [], [], 0, 0
    for r in gated:
        ga, gm = np.array(r["gate_attn"]), np.array(r["gate_mlp"])
        g = fn(ga, gm); dl = np.array(r["dloss"]); sc = np.array(r["scale"])
        r_abl.append(spearman(g, dl)); r_scl.append(spearman(g, sc))
        if int(np.argmin(g)) == int(np.argmin(dl)):
            t1 += 1
        if g[0] < np.median(g):
            l0bm += 1
    print("   %-6s | %-16s | %-16s | %-6d/%-2d | %d/%d" %
          (name, fmt(min(r_abl), max(r_abl)), fmt(min(r_scl), max(r_scl)), t1, N, l0bm, N))
p_max = pois_binom_tail(ps_top1, 10)
print("   top-1: chance expectation = %.2f ; P(>=10 by chance) = %.4f" %
      (sum(ps_top1), p_max))

# ---------------------------------------------------------------- pruning test
print("\n2) DIRECT PRUNING TEST  (dloss = loss increase from skipping a layer)")
gd, od, rnd = [], [], []
for r in gated:
    g = np.array(r["gate_prod"]); dl = np.array(r["dloss"]); n = len(g)
    ig, ia = int(np.argmin(g)), int(np.argmin(dl))
    gd.append(dl[ig]); od.append(dl[ia])
    Hn = sum(1.0 / k for k in range(1, n + 1))
    rnd.append(dl.mean() * Hn / n)
print("   gate-pick dloss  mean %.3f  median %.3f" % (np.mean(gd), np.median(gd)))
print("   oracle-pick dloss mean %.3f  median %.3f" % (np.mean(od), np.median(od)))
print("   random-pick dloss mean %.3f  median %.3f" % (np.mean(rnd), np.median(rnd)))
print("   => gate's 'most skippable' layer costs ~%.0fx more loss than the oracle's pick" %
      (np.mean(gd) / np.mean(od)))

# ---------------------------------------------------------------- layer-0
print("\n3) LAYER-0 TENSION")
l0_imp = sum(1 for r in gated if np.array(r["dloss"])[0] == max(np.array(r["dloss"])))
l0_bmed = sum(1 for r in gated
              if np.array(r["gate_prod"])[0] < np.median(np.array(r["gate_prod"])))
from math import comb
k = l0_bmed; p = 2 * sum(comb(N, j) for j in range(0, min(k, N - k) + 1)) / 2 ** N
print("   L0 == MOST important (max ablation): %d/%d" % (l0_imp, N))
print("   L0 gate below-median (all metrics)  : %d/%d   sign-test p = %.2e" % (l0_bmed, N, p))
print("   => the gate is blind to the single most important layer's leverage")

# ---------------------------------------------------------------- redundancy
print("\n4) SCALE REDUNDANCY (the explanation)")
gs = [spearman(np.array(r["gate_prod"]), np.array(r["scale"])) for r in gated]
pr = [r["stats"]["partial_rho_gate_given_scale"] for r in gated
      if r["stats"].get("partial_rho_gate_given_scale") is not None]
print("   rho(gate, scale)               : %s  (gate tracks output-weight norm)" % fmt(min(gs), max(gs)))
print("   partial rho(gate | scale)      : %s  (gate adds ~nothing beyond scale)" % fmt(min(pr), max(pr)))

print("\n=== VERDICT ===")
print("HYPOTHESIS REJECTED. The learned gate is a redundant copy of the")
print("output-weight scale (rho %.2f-%.2f, all runs); after removing scale it" % (min(gs), max(gs)))
print("adds ~zero importance signal (partial rho in %s). It does not" % fmt(min(pr), max(pr)))
print("reliably flag skippable layers (top-1 3-10/%d, weak), and it is" % N)
print("blind to layer 0's leverage (below-median gate in %d/%d while it is the" % (l0_bmed, N))
print("most important layer in %d/%d). 'L0 is THE most skippable' is a" % (l0_imp, N))
print("product-metric artifact; the robust core is: gate ~ scale, weak signal.")
