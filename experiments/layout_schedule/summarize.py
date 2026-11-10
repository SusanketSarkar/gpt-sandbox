"""Build the final figures + summary.md for the layout-schedule study.

    python summarize.py --runs-dir /data/susanket/text/gpt/runs \
        [--sweeps layout_synth_local_rule layout_synth_hop2 layout_f1_jumbo layout_f2_jumbo ...] \
        [--figs figs]
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

VARIANT_ORDER = ["baseline", "A", "B", "C", "D", "sandwich",
                 "rand0", "rand1", "rand2", "rand3", "rand4", "rand5",
                 "f2_20A28M", "f2_16A32M", "f2_12A36M"]


def load_runs(sweep_dir: Path):
    out = []
    for d in sorted(sweep_dir.glob("*/eval.json")):
        m = re.match(r"(.+?)(?:-s(\d+))?$", d.parent.name)
        ev = json.loads(d.read_text())
        out.append({"variant": m.group(1), "seed": m.group(2), "ev": ev})
    return out


def get(ev, key):
    groups = ev["groups"]
    if "/" in key:
        g, k = key.split("/", 1)
        return groups.get(g, {}).get(k)
    for g, gv in groups.items():
        if key in gv:
            return gv[key]
    return None


def aggregate(runs, key):
    by = defaultdict(list)
    for r in runs:
        v = get(r["ev"], key)
        if isinstance(v, (int, float)):
            by[r["variant"]].append(float(v))
    rows = []
    for var, vals in by.items():
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((x - mean) ** 2 for x in vals) / n) ** 0.5 if n > 1 else 0.0
        rows.append((var, n, mean, std))
    return rows


def order(rows):
    idx = {v: i for i, v in enumerate(VARIANT_ORDER)}
    return sorted(rows, key=lambda r: (idx.get(r[0], 999), r[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/data/susanket/text/gpt/runs")
    ap.add_argument("--sweeps", nargs="*", default=[
        "layout_synth_local_rule", "layout_synth_recall", "layout_synth_hop2",
        "layout_f1_jumbo", "layout_f2_jumbo", "layout_f1_small", "layout_f1_med"])
    ap.add_argument("--figs", default="figs")
    ap.add_argument("--summary", default="summary.md")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    figs = Path(args.figs); figs.mkdir(parents=True, exist_ok=True)
    md = ["# Layout-schedule study — results", ""]
    for sweep in args.sweeps:
        sd = runs_dir / sweep
        if not sd.exists():
            md.append(f"## {sweep}\n\n(not run yet)\n"); continue
        runs = load_runs(sd)
        if not runs:
            md.append(f"## {sweep}\n\n(no runs)\n"); continue
        is_synth = "synth" in sweep
        key = "val/val_loss"
        rows = order(aggregate(runs, key))
        md.append(f"## {sweep}  ({len(runs)} runs)\n")
        md.append(f"| variant | n | {key} |")
        md.append("|---|---|---|")
        for var, n, mean, std in rows:
            md.append(f"| {var} | {n} | {mean:.4f} ± {std:.4f} |")
        if is_synth:
            em = order(aggregate(runs, "task/task_em"))
            md.append(f"\n| variant | n | task_em |")
            md.append("|---|---|---|")
            for var, n, mean, std in em:
                md.append(f"| {var} | {n} | {mean:.3f} ± {std:.3f} |")
        best = min(rows, key=lambda r: r[2]); worst = max(rows, key=lambda r: r[2])
        md.append(f"\nbest **{best[0]}** {best[2]:.4f}, worst **{worst[0]}** {worst[2]:.4f}, "
                  f"gap {worst[2]-best[2]:+.4f}\n")
        # bar chart
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            names = [r[0] for r in rows]; vals = [r[2] for r in rows]; errs = [r[3] for r in rows]
            fig, ax = plt.subplots(figsize=(max(4, len(names)*0.6), 4))
            ax.bar(names, vals, yerr=errs, capsize=4, color="#4c72b0", alpha=0.85)
            ax.set_ylabel(key); ax.set_title(sweep)
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout(); fig.savefig(figs / f"{sweep}.png", dpi=120); plt.close(fig)
            md.append(f"![{sweep}]({figs.name}/{sweep}.png)\n")
        except ImportError:
            pass
    Path(args.summary).write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {args.summary} and {len(list(figs.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
