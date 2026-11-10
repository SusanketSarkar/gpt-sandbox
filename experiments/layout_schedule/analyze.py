"""Aggregate the layout-schedule study: tables, noise floor, and per-depth curves.

  python analyze.py --sweep <sweep_name> [--runs-dir DIR] [--key val_loss] [--plot DIR]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load_runs(runs_dir: Path):
    """yield (variant, seed, data, eval_dict) for every eval.json under runs_dir/<sweep>/*"""
    out = []
    for d in sorted(runs_dir.glob("*/eval.json")):
        m = re.match(r"(.+?)(?:-s(\d+))?$", d.parent.name)
        variant, seed = m.group(1), m.group(2)
        ev = json.loads(d.read_text())
        out.append((variant, seed, ev.get("data"), ev["groups"], ev.get("notes", {})))
    return out


def aggregate(runs, key):
    """mean +- std of runs[i][3][key] over seeds, per variant. key like 'val_loss' or 'val/val_loss'."""
    by_var = defaultdict(list)
    for variant, seed, data, groups, notes in runs:
        # key may be 'group/metric'
        if "/" in key:
            g, k = key.split("/", 1)
            v = groups.get(g, {}).get(k)
        else:
            v = next((groups[g].get(key) for g in groups if key in groups.get(g, {})), None)
        if v is not None and isinstance(v, (int, float)):
            by_var[variant].append(float(v))
    rows = []
    for variant, vals in by_var.items():
        if len(vals) >= 1:
            mean = sum(vals) / len(vals)
            std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
            rows.append((variant, len(vals), mean, std))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--runs-dir", default="/data/susanket/text/gpt/runs")
    ap.add_argument("--key", default="val/val_loss")
    ap.add_argument("--plot", default=None, help="output dir for curves")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    sweep_dir = runs_dir / args.sweep
    if not sweep_dir.exists():
        # maybe the runs are directly under runs_dir with the sweep name in the path
        candidates = list(runs_dir.glob(args.sweep)) + list(runs_dir.glob(f"*/{args.sweep}"))
        if not candidates:
            raise SystemExit(f"no sweep dir {sweep_dir}")
        sweep_dir = candidates[0]

    runs = load_runs(sweep_dir)
    print(f"{len(runs)} runs under {sweep_dir}\n")

    rows = aggregate(runs, args.key)
    rows.sort(key=lambda r: r[2])
    print(f"{'variant':14s} {'n':>2s}  {args.key}")
    for variant, n, mean, std in rows:
        print(f"{variant:14s} {n:>2d}  {mean:.4f}  +- {std:.4f}")
    if rows:
        best, worst = rows[0], rows[-1]
        print(f"\nbest {best[0]} {best[2]:.4f}   worst {worst[0]} {worst[2]:.4f}   gap {worst[2]-best[2]:+.4f}")

    # noise floor = std across seeds of the SAME variant
    noise = [r[3] for r in rows if r[1] >= 2]
    if noise:
        print(f"noise floor (mean std across seeds): {sum(noise)/len(noise):.4f}")

    if args.plot:
        _plot_curves(runs, sweep_dir, Path(args.plot))


def _plot_curves(runs, sweep_dir, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return
    out.mkdir(parents=True, exist_ok=True)

    def curve(groups, prefix, nmax=64):
        items = []
        for gname, gval in groups.items():
            for k, v in gval.items():
                if k.startswith(prefix) and isinstance(v, (int, float)):
                    m = re.match(r"L(\d+)$", k[len(prefix):])
                    if m:
                        i = int(m.group(1))
                        if i < nmax:
                            items.append((i, float(v)))
        items.sort()
        return items

    # ctx_sensitivity per variant (mean over seeds)
    cs = defaultdict(list)
    for variant, seed, data, groups, notes in runs:
        c = curve(groups, "ctx_sens/L")
        if c:
            cs[variant].append(dict(c))
    if cs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, curves in cs.items():
            n = max(len(c) for c in curves)
            xs = list(range(n))
            ys = [sum(c.get(i, float("nan")) for c in curves) / len(curves) for i in xs]
            ax.plot(xs, ys, marker="o", ms=3, label=variant)
        ax.set_xlabel("depth (sublayer index)")
        ax.set_ylabel("relative change vs random prefix (last pos)")
        ax.set_title("Context sensitivity by depth")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "ctx_sensitivity.png", dpi=120)
        print("wrote", out / "ctx_sensitivity.png")

    # layer_ablation: loss delta when removing each layer
    ab = defaultdict(list)
    for variant, seed, data, groups, notes in runs:
        c = curve(groups, "ablate_dloss/L")
        if c:
            ab[variant].append(dict(c))
    if ab:
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, curves in ab.items():
            n = max(len(c) for c in curves)
            xs = list(range(n))
            ys = [sum(c.get(i, float("nan")) for c in curves) / len(curves) for i in xs]
            ax.plot(xs, ys, marker="o", ms=3, label=variant)
        ax.set_xlabel("depth (sublayer index)")
        ax.set_ylabel("val loss delta when removing sublayer")
        ax.set_title("Layer ablation (loss increase on removal)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "layer_ablation.png", dpi=120)
        print("wrote", out / "layer_ablation.png")


if __name__ == "__main__":
    main()
