#!/usr/bin/env python3
"""Generate the key figures for the layout-schedule study. Reads the eval.json
files under /data/susanket/text/gpt/runs/ and writes PNGs to ./figs/."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/data/susanket/text/gpt/runs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

def load(sweep, variant, seed=0):
    p = f"{RUNS}/{sweep}/{variant}-s{seed}/eval.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))["groups"]

def mean_std(sweep, variants, key, group, seeds=(0,1,2)):
    means, stds, have = [], [], []
    for v in variants:
        vals = []
        for s in seeds:
            p = f"{RUNS}/{sweep}/{v}-s{s}/eval.json"
            if os.path.exists(p):
                g = json.load(open(p))["groups"]
                try: vals.append(g[group][key])
                except Exception: pass
        if vals:
            means.append(sum(vals)/len(vals))
            stds.append((sum((x-sum(vals)/len(vals))**2 for x in vals)/max(1,len(vals)-0.0))**0.5 if len(vals)>1 else 0.0)
            have.append(v)
    return have, means, stds

# ---- Fig 1: F1 jumbo val_loss (the core result) ----
variants = ["rand1","rand3","rand2","baseline","rand0","A","B","sandwich","D","C"]
have, means, stds = mean_std("layout_f1_jumbo", variants, "val_loss", "val")
order = list(zip(have, means, stds))
fig, ax = plt.subplots(figsize=(8,5))
colors = ["#2a7de1" if v.startswith("rand") or v=="baseline" else "#e05252" for v in have]
bars = ax.bar(range(len(have)), means, yerr=stds, capsize=4, color=colors, alpha=0.9)
ax.set_xticks(range(len(have))); ax.set_xticklabels(have, rotation=30)
ax.set_ylabel("val_loss (lower better)")
ax.set_title("Family 1 — sublayer ordering on edu_mix (24A+24M, params/FLOPs matched)\nblue = interleaved (random/alternating)  red = depth-block")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/f1_jumbo_valloss.png", dpi=140); plt.close()
print("wrote f1_jumbo_valloss.png")

# ---- Fig 2: synth recall EM (task-specific) ----
sv = ["B","D","baseline","A","C"]
have2, means2, stds2 = mean_std("layout_synth_recall", sv, "task_em", "task")
order2 = sorted(zip(have2, means2, stds2), key=lambda x: -x[1])
fig, ax = plt.subplots(figsize=(7,4.5))
cols = ["#2a9d5c" if m>0.9 else ("#e0a02a" if m>0.6 else "#e05252") for _,m,_ in order2]
ax.bar(range(len(order2)), [m for _,m,_ in order2], yerr=[s for _,_,s in order2], capsize=4, color=cols)
ax.set_xticks(range(len(order2))); ax.set_xticklabels([v for v,_,_ in order2])
ax.set_ylabel("recall exact-match (higher better)"); ax.set_ylim(0,1.1)
ax.set_title("One-hop recall — attention at the ENDS wins\n(B, D) > baseline > (A, C)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/synth_recall_em.png", dpi=140); plt.close()
print("wrote synth_recall_em.png")

# ---- Fig 3: F2 efficiency (val_loss vs attention count) ----
pts = []
for v in ["baseline","f2_20A28M","f2_16A32M","f2_12A36M"]:
    sweep = "layout_f1_jumbo" if v=="baseline" else "layout_f2_jumbo"
    vals=[]
    for s in (0,1,2):
        p=f"{RUNS}/{sweep}/{v}-s{s}/eval.json"
        if os.path.exists(p):
            vals.append(json.load(open(p))["groups"]["val"]["val_loss"])
    if vals: pts.append((v, sum(vals)/len(vals), (sum((x-sum(vals)/len(vals))**2 for x in vals)/max(1,len(vals)-1))**0.5))
attn = {"baseline":24,"f2_20A28M":20,"f2_16A32M":16,"f2_12A36M":12}
pts = sorted(pts, key=lambda x: -attn[x[0]])
fig, ax = plt.subplots(figsize=(7,4.5))
xs=[attn[p[0]] for p in pts]; ys=[p[1] for p in pts]; yserr=[p[2] for p in pts]
ax.errorbar(xs, ys, yerr=yserr, marker="o", capsize=5, color="#2a7de1", lw=2)
ax.invert_xaxis(); ax.set_xlabel("attention sublayers (fewer = cheaper, more MLP)")
ax.set_ylabel("val_loss"); ax.set_title("Family 2 — dropping attention costs ~0.1-0.2%\n(params matched <0.02%, FLOPs within 1-4%)")
ax.grid(alpha=0.3)
for p in pts: ax.annotate(p[0].replace("f2_",""), (attn[p[0]], p[1]), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/f2_efficiency.png", dpi=140); plt.close()
print("wrote f2_efficiency.png")

# ---- Fig 4: edu_mix context-sensitivity profiles (baseline vs C) ----
def ctxprof(run):
    g=json.load(open(f"{RUNS}/layout_f1_jumbo/{run}/eval.json"))["groups"]["ctx_sensitivity"]
    cs={int(k.split("L")[1]):g[k] for k in g if k.startswith("ctx_sens/L")}
    return sorted(cs.items())
fig, ax = plt.subplots(figsize=(8,4.5))
for run, color, lbl in [("baseline-s0","#2a7de1","baseline (alternating)"),("C-s0","#e05252","C (MLP-only ends)")]:
    prof = ctxprof(run)
    xs=[p[0] for p in prof]; ys=[p[1] for p in prof]
    ax.plot(xs, ys, color=color, lw=2, label=lbl)
ax.set_xlabel("sublayer"); ax.set_ylabel("context sensitivity (relative change)")
ax.set_title("edu_mix — where each layout reads context\ninterleaved reads from L0; C (MLP-early) cannot until the middle attention block")
ax.axvline(15, color="grey", ls=":", lw=1); ax.axvline(31, color="grey", ls=":", lw=1)
ax.text(7, ax.get_ylim()[1]*0.95, "early", fontsize=8, color="grey")
ax.text(23, ax.get_ylim()[1]*0.95, "mid", fontsize=8, color="grey")
ax.text(39, ax.get_ylim()[1]*0.95, "late", fontsize=8, color="grey")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/ctx_sensitivity_edumix.png", dpi=140); plt.close()
print("wrote ctx_sensitivity_edumix.png")

# ---- Fig 5: depth ablation (8 / 12 / 48 sublayers) ----
def depth_rows(sweep):
    rows = []
    for d in sorted(os.listdir(f"{RUNS}/{sweep}")):
        if not os.path.isdir(f"{RUNS}/{sweep}/{d}"): continue
        v = d.split("-s")[0]
        vals = []
        for s in (0, 1, 2, 3):
            p = f"{RUNS}/{sweep}/{v}-s{s}/eval.json"
            if os.path.exists(p):
                vals.append(json.load(open(p))["groups"]["val"]["val_loss"])
        if vals: rows.append((v, sum(vals) / len(vals)))
    return sorted(rows, key=lambda x: x[1])

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (sweep, ttl) in zip(axes, [("layout_f1_small", "8 sublayers"), ("layout_f1_med", "12 sublayers"), ("layout_f1_jumbo", "48 sublayers")]):
    rows = depth_rows(sweep)
    names = [r[0] for r in rows]; means = [r[1] for r in rows]
    cols = ["#2a7de1" if n in ("baseline", "rand0", "rand1", "rand2", "rand3") else "#e05252" for n in names]
    ax.bar(range(len(names)), means, color=cols, alpha=0.9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, fontsize=7)
    ax.set_title(ttl); ax.set_ylabel("val_loss (lower better)"); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Depth ablation — interleaved (blue) is consistently top-tier; C (MLP-only ends) worst at every depth", y=1.03)
plt.tight_layout(); plt.savefig(f"{OUT}/f1_depth.png", dpi=140); plt.close()
print("wrote f1_depth.png")
print("DONE ->", OUT)
