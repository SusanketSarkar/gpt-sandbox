"""Extended figure set for the HTML blog.

Reads the finished runs under /data/susanket/text/gpt/runs and writes PNGs to
experiments/layout_schedule/figs/. All figures use the same color language:
  A (attention) = blue, M (MLP) = orange.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = "/data/susanket/text/gpt/runs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

CA = "#3b6fb5"  # attention
CM = "#e8833a"  # mlp
GRAY = "#888888"
RANK = ["rand1", "rand3", "rand2", "baseline", "rand0",
        "A", "B", "sandwich", "D", "C"]
LABEL = {
    "baseline": "baseline (alternating)",
    "A": "A (MLP ends, attn mid)",
    "B": "B (attn ends, MLP mid)",
    "C": "C (MLP-only ends)",
    "D": "D (attn-only ends)",
    "sandwich": "sandwich",
    "rand0": "rand0", "rand1": "rand1", "rand2": "rand2", "rand3": "rand3",
}


def style(ax, title=None):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=12, weight="bold", pad=10)


def layout_of(variant, sweep="layout_f1_jumbo"):
    cfg = open(os.path.join(RUNS, sweep, f"{variant}-s0", "config.yaml")).read()
    for line in cfg.splitlines():
        if "layout:" in line:
            return line.split("layout:")[1].strip().strip('"')
    raise ValueError(f"no layout in {variant}")


def metrics_of(variant, seed, sweep="layout_f1_jumbo"):
    p = os.path.join(RUNS, sweep, f"{variant}-s{seed}", "eval.json")
    return json.load(open(p))["metrics"]


def mean_over_seeds(key, variants, seeds=3, sweep="layout_f1_jumbo"):
    return [np.mean([metrics_of(v, s, sweep)[key] for s in range(seeds)]) for v in variants]


# ---------------------------------------------------------------- 1. layout map
def fig_layout_map():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ylabels = []
    for i, v in enumerate(RANK):
        lay = layout_of(v)
        y = len(RANK) - 1 - i
        for x, ch in enumerate(lay):
            ax.add_patch(plt.Rectangle((x, y - 0.35), 1, 0.7,
                                       facecolor=CA if ch == "A" else CM,
                                       edgecolor="white", linewidth=0.5))
        ylabels.append(f"{LABEL[v]}")
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(-0.7, len(RANK) - 0.3)
    ax.set_yticks([len(RANK) - 1 - i for i in range(len(RANK))])
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("sublayer index (0 = input, 47 = output)", fontsize=10)
    ax.set_xticks(range(0, 48, 4))
    ax.grid(False)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    # depth bands
    for x0, x1, name in [(0, 16, "early"), (16, 32, "middle"), (32, 48, "late")]:
        ax.add_patch(plt.Rectangle((x0, len(RANK) - 0.15), x1 - x0, 0.28,
                                   facecolor="#dddddd", edgecolor="none"))
        ax.text((x0 + x1) / 2, len(RANK) - 0.01, name, ha="center", va="bottom",
                fontsize=9, color="#444444")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=CA, label="attention sublayer"),
                       Patch(facecolor=CM, label="MLP sublayer")],
              loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    style(ax, "The 10 layouts: 48 sublayers, all 24 attention + 24 MLP, only the order differs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "layout_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- 2. learning curves
def fig_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"baseline": "#111111", "A": CA, "C": CM, "rand1": "#4daf4d"}
    for v in ["baseline", "rand1", "A", "C"]:
        rows = [json.loads(l) for l in
                open(os.path.join(RUNS, "layout_f1_jumbo", f"{v}-s0", "metrics.jsonl"))]
        tr = [(r["step"], r["loss"]) for r in rows if r.get("split") == "train"]
        va = [(r["step"], r["val_loss"]) for r in rows if r.get("split") == "val"]
        axes[0].plot(*zip(*tr), color=colors[v], lw=2, alpha=0.9,
                     label=LABEL[v] if v != "rand1" else "rand1 (random)")
        axes[1].plot(*zip(*va), color=colors[v], lw=2, marker="o", ms=3,
                     label=LABEL[v] if v != "rand1" else "rand1 (random)")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("train loss")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    style(axes[0], "train loss (log scale)")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("val loss (edu_mix)")
    axes[1].legend(fontsize=8)
    style(axes[1], "val loss")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ 3. noise floor
def fig_noise_floor():
    means, stds = [], []
    for v in RANK:
        vals = [metrics_of(v, s)["val_loss"] for s in range(3)]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(RANK))
    bars = ax.bar(x, [means[i] for i in order],
                  yerr=[stds[i] for i in order], capsize=4,
                  color=[("#3b6fb5" if v.startswith("rand") or v == "baseline" else "#e8833a")
                         for v in [RANK[i] for i in order]],
                  edgecolor="#333333", linewidth=0.6, alpha=0.9, width=0.7)
    for i, o in enumerate(order):
        ax.text(i, means[o] + 0.004, f"{means[o]:.3f}", ha="center", fontsize=8)
    nf = 0.0027
    ax.axhline(min(means) + nf, color="#4daf4d", ls="--", lw=1.2)
    ax.text(len(RANK) - 0.4, min(means) + nf + 0.002, f"noise floor (+{nf:.3f})",
            ha="right", fontsize=8, color="#2d7d2d")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[RANK[i]] for i in order], rotation=18, fontsize=8)
    ax.set_ylabel("val loss (edu_mix, 3-seed mean ± std)")
    best, worst = min(means), max(means)
    ax.set_ylim(2.66, worst + 0.02)
    style(ax, f"Family 1 (jumbo): best–worst gap {worst-best:.3f} ≈ {round((worst-best)/nf)}× the 3-seed noise floor")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "noise_floor.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- 4-6. deep probes jumbo
def _layer_series(m, prefix):
    out = {}
    for k, v in m.items():
        if k.startswith(prefix + "/L"):
            out[int(k.split("L")[1])] = v
    return out


def fig_probe(prefix, ylabel, fname, title, sweep="layout_f1_jumbo",
              variants=("baseline", "C")):
    fig, ax = plt.subplots(figsize=(9, 4.3))
    colors = {"baseline": "#111111", "A": CA, "B": "#4daf4d", "C": CM}
    for v in variants:
        m = metrics_of(v, 0, sweep)
        s = _layer_series(m, prefix)
        xs = sorted(s)
        ax.plot(xs, [s[i] for i in xs], color=colors.get(v, GRAY), lw=2, label=LABEL[v])
    ax.set_xlabel("sublayer index (0 = first, 47 = last)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, 48)
    ax.legend(fontsize=9)
    style(ax, title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_delta_rms():
    fig, ax = plt.subplots(figsize=(9, 4.3))
    colors = {"baseline": "#111111", "A": CA, "C": CM}
    for v in ["A", "baseline", "C"]:
        rows = [json.loads(l) for l in
                open(os.path.join(RUNS, "layout_f1_jumbo", f"{v}-s0", "metrics.jsonl"))]
        last = [r for r in rows if r.get("split") == "train"][-1]
        s = {int(k.split("L")[1]): last[k] for k in last if k.startswith("delta_rms/L")}
        xs = sorted(s)
        ax.plot(xs, [s[i] for i in xs], color=colors[v], lw=2, label=LABEL[v])
    ax.set_xlabel("sublayer index")
    ax.set_ylabel("delta RMS (how much this sublayer changed the stream)")
    ax.set_yscale("log")
    ax.set_xlim(-1, 48)
    ax.legend(fontsize=9)
    style(ax, "Per-sublayer delta RMS at the end of training (final step)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "delta_rms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ 9. induction
def fig_induction():
    ks = ["induction_acc", "induction_gap"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k, name in zip(axes, ks, ["induction accuracy", "induction gap (loss: random − repeat)"]):
        vals = mean_over_seeds(k, RANK)
        order = np.argsort(vals)
        x = np.arange(len(RANK))
        ax.bar(x, [vals[i] for i in order],
               color=[("#3b6fb5" if RANK[i].startswith("rand") or RANK[i] == "baseline" else "#e8833a")
                      for i in order],
               edgecolor="#333333", linewidth=0.5, alpha=0.9)
        for i, o in enumerate(order):
            ax.text(i, vals[o] + 0.02, f"{vals[o]:.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[RANK[i]] for i in order], rotation=18, fontsize=8)
        ax.set_ylabel(name)
        style(ax)
    fig.suptitle("Induction-head probe (edu_mix jumbo): how well each layout learns to "
                 "copy the pattern  x … y … x → y", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "induction.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ 10. F2 curve
def fig_f2_curve():
    base = mean_over_seeds("val_loss", ["baseline"])
    rows = []
    for v, na in [("f2_20A28M", 20), ("f2_16A32M", 16), ("f2_12A36M", 12)]:
        ms = [metrics_of(v, s, sweep="layout_f2_jumbo") for s in range(3)]
        rows.append((na, np.mean([d["val_loss"] for d in ms]),
                     np.mean([d["flops_per_token"] for d in ms]),
                     np.mean([d["params_total"] for d in ms])))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = [24, 20, 16, 12]
    y = [base[0]] + [r[1] for r in rows]
    ax.plot(x, y, "o-", color="#111111", lw=2, ms=7)
    ax.annotate("baseline\n24 attn", (24, base[0]), textcoords="offset points",
                xytext=(-30, -22), fontsize=9)
    for (na, m, fl, p), yy in zip(rows, y[1:]):
        save = (1 - fl / 518258688.0) * 100
        d = yy - base[0]
        ax.annotate(f"{na} attn\nΔloss {d:+.3f}\nFLOPs {save:.1f}% lower",
                    (na, yy), textcoords="offset points", xytext=(18, 6), fontsize=8)
    ax.set_xlabel("number of attention sublayers (total sublayers fixed at 48)")
    ax.set_ylabel("val loss (edu_mix, 3-seed mean)")
    ax.set_ylim(2.69, 2.715)
    ax.set_xlim(11, 25)
    style(ax, "Family 2: swap attention for MLP (params matched) — quality is flat")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f2_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_layout_map()
    fig_learning_curves()
    fig_noise_floor()
    fig_probe("ctx_sens", "context sensitivity (rel. change when earlier tokens scrambled)",
              "ctx_sens_jumbo.png", "Context sensitivity per sublayer (edu_mix): baseline vs C")
    fig_probe("lens_loss", "logit-lens loss (loss if we stopped here)",
              "logit_lens_jumbo.png", "Logit lens: where does the prediction form? (baseline vs C)")
    fig_probe("ablate_dloss", "Δ val loss if this sublayer is zeroed",
              "ablation_jumbo.png", "Layer ablation: which sublayers matter? (baseline vs C)")
    fig_delta_rms()
    fig_probe("ctx_sens", "context sensitivity (rel. change when earlier tokens scrambled)",
              "ctx_sens_recall.png", "Context sensitivity per sublayer (synthetic one-hop recall): "
              "A vs B vs baseline", sweep="layout_synth_recall",
              variants=("A", "B", "baseline"))
    fig_induction()
    fig_f2_curve()
    print("wrote", sorted(os.listdir(OUT)))
