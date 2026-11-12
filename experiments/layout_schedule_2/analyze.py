"""Aggregate the ratio-sweep night into tables + summary JSON.

Usage: python analyze.py [--runs-dir /data/susanket/text/gpt/runs] [--out tables.md]
Robust to partial data: only completed runs (eval.json present) are included.
"""
from __future__ import annotations
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RUNS = Path("/data/susanket/text/gpt/runs")
EXP = Path(__file__).resolve().parent

VAR_RE = re.compile(r"^(?P<tag>[a-z0-9]+)_(?P<nA>\d+)A(?P<nM>\d+)M_(?P<arr>alt|r\d+|legacy)$")


def parse_run_dir(rd: Path):
    if rd.parent.name == "layout_f1_jumbo" and rd.name.startswith("baseline"):
        return {"sweep": "ratio_jumbo", "tag": "r11", "nA": 24, "nM": 24,
                "arr": "legacy", "seed": int(rd.name.split("-s")[1])}
    m = VAR_RE.match(rd.name.split("-s")[0])
    if not m:
        return None
    seed = int(rd.name.split("-s")[1])
    sweep = rd.parent.name
    tag, arr = m["tag"], m["arr"]
    if rd.parent.name == "layout_f1_jumbo":
        sweep, tag, arr = "ratio_jumbo", "r11", "legacy"
    return {"sweep": sweep, "tag": tag, "nA": int(m["nA"]), "nM": int(m["nM"]),
            "arr": arr, "seed": seed}


def load_run(rd: Path):
    try:
        ev = json.loads((rd / "eval.json").read_text())["metrics"]
        run = json.loads((rd / "run.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    rec = parse_run_dir(rd)
    if not rec:
        return None
    rec.update({
        "val_loss": ev.get("val_loss"),
        "task_em": ev.get("task_em"),
        "recall_acc/8": ev.get("recall_acc/8"), "recall_acc/16": ev.get("recall_acc/16"),
        "recall_acc/32": ev.get("recall_acc/32"), "recall_acc/64": ev.get("recall_acc/64"),
        "recall_acc/128": ev.get("recall_acc/128"),
        "params": run.get("params"), "params_nonemb": run.get("params_nonemb"),
        "flops_per_token": ev.get("flops_per_token"),
        "fwd_tok_per_s": ev.get("fwd_tok_per_s"),
        "best_val": run.get("best_val"),
        "data": run.get("data"),
        "wall_s": None,
    })
    try:
        t0 = datetime.fromisoformat(run["started"])
        t1 = datetime.fromisoformat(run["finished"])
        rec["wall_s"] = (t1 - t0).total_seconds()
    except (KeyError, ValueError):
        pass
    # convergence / stability from metrics.jsonl
    tr, val, gn = [], [], []
    try:
        for line in (rd / "metrics.jsonl").read_text().splitlines():
            d = json.loads(line)
            if d.get("split") == "train" and d.get("loss") is not None:
                tr.append((d["step"], d["loss"]))
                if d.get("gnorm") is not None:
                    gn.append(d["gnorm"])
            elif d.get("split") == "val" and d.get("val_loss") is not None:
                val.append((d["step"], d["val_loss"]))
    except FileNotFoundError:
        pass
    rec["val_curve"] = val
    if tr:
        for s in (1000, 2000, 3000, 4000, 5000, 6000, 12000, 16000):
            pts = [l for st_, l in tr if st_ >= s]
            rec[f"train_loss_{s}"] = pts[0] if pts else None
    if gn:
        rec["gnorm_mean"] = st.mean(gn)
        rec["gnorm_max"] = max(gn)
        rec["gnorm_p95"] = sorted(gn)[max(0, int(len(gn) * 0.95) - 1)]
        rec["gnorm_spikes"] = sum(1 for g in gn if g > 10)
    return rec


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0, 0
    m = st.mean(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    return m, sd, len(vals)


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    args = ap.parse_args()
    runs_root = Path(args.runs_dir)

    rows = []
    for sweep in sorted(p.name for p in runs_root.iterdir() if p.is_dir()):
        if not (sweep.startswith("ratio") or sweep.startswith("layout_f1")):
            continue
        for rd in sorted((runs_root / sweep).iterdir()):
            if rd.is_dir():
                rec = load_run(rd)
                if rec:
                    rec["run_dir"] = str(rd.relative_to(runs_root))
                    rows.append(rec)
    rows.sort(key=lambda r: (r["sweep"], r["nA"], r["arr"], r["seed"]))
    (EXP / "all_runs.json").write_text(json.dumps(rows, indent=1))
    print(f"loaded {len(rows)} completed runs")

    out = []
    W = out.append
    W("# ratio-sweep results (auto-generated)\n")
    W(f"generated {datetime.now().isoformat(timespec='minutes')} from {runs_root}\n")

    # ---- per-sweep ratio tables -------------------------------------------------------------
    sweeps = sorted(set(r["sweep"] for r in rows))
    for sw in sweeps:
        rr = [r for r in rows if r["sweep"] == sw]
        groups = defaultdict(list)
        for r in rr:
            groups[r["nA"]].append(r)
        W(f"\n## {sw}  ({len(rr)} runs)\n")
        cols = ["nA", "nM", "runs", "val_loss", "best_val", "task_em"]
        if any("recall_acc/64" in r for r in rr):
            cols += ["recall/16", "recall/64", "recall/128"]
        cols += ["params_M", "flops_M", "wall_min", "gnorm_p95", "spikes>10"]
        W("| " + " | ".join(cols) + " |")
        W("|" + "---|" * len(cols))
        for nA in sorted(groups):
            g = groups[nA]
            nM = g[0]["nM"]
            def A(k, nd=4):
                m, sd, _ = agg([r.get(k) for r in g])
                return fmt(m, nd)
            n_runs = len(g)
            row = [f"**{nA}**", nM, n_runs, A("val_loss"), A("best_val"), A("task_em")]
            if any("recall_acc/64" in r for r in rr):
                row += [A("recall_acc/16"), A("recall_acc/64"), A("recall_acc/128")]
            pm, _, _ = agg([r.get("params") for r in g])
            row += [fmt(pm / 1e6, 1) if pm else "—",
                    fmt((g[0].get("flops_per_token") or 0) / 1e6, 1),
                    fmt(st.mean([r["wall_s"] for r in g if r.get("wall_s")]) / 60, 1) if any(r.get("wall_s") for r in g) else "—",
                    A("gnorm_p95", 2), A("gnorm_spikes", 0)]
            W("| " + " | ".join(str(x) for x in row) + " |")

    # ---- headline comparisons (jumbo raw + pm) ---------------------------------------------
    def ratio_curve(sweep, key):
        rr = [r for r in rows if r["sweep"] == sweep]
        groups = defaultdict(list)
        for r in rr:
            groups[r["nA"]].append(r)
        curve = []
        for nA in sorted(groups):
            m, sd, n = agg([r.get(key) for r in groups[nA]])
            if m is not None:
                curve.append({"nA": nA, "nM": groups[nA][0]["nM"], "val": m, "sd": sd, "n": n})
        return curve

    summary = {}
    for sw, key in (("ratio_jumbo", "val_loss"), ("ratio_pm_jumbo", "val_loss"),
                    ("ratio_depth24", "val_loss"), ("ratio_fine_jumbo", "val_loss")):
        c = ratio_curve(sw, key)
        if len(c) >= 2:
            best = min(c, key=lambda x: x["val"])
            worst = max(c, key=lambda x: x["val"])
            summary[sw] = {"curve": c, "best": best, "worst": worst,
                           "spread": worst["val"] - best["val"]}
            W(f"\n### {sw}: best ratio {best['nA']}A/{best['nM']}M (val {best['val']:.4f}) "
              f"vs worst {worst['nA']}A/{worst['nM']}M (val {worst['val']:.4f}); spread {summary[sw]['spread']:.4f}\n")
        # monotonicity check across attention fraction
        if len(c) >= 3:
            diffs = [c[i + 1]["val"] - c[i]["val"] for i in range(len(c) - 1)]
            W(f"  step diffs (increasing attn): " + " ".join(f"{d:+.4f}" for d in diffs) + "\n")
            inv_u = all(d > 0 for d in diffs[:len(diffs) // 2]) and all(d < 0 for d in diffs[len(diffs) // 2:])
            W(f"  inverted-U shape: {'yes' if inv_u else 'no'}\n")

    # ---- synth task tables -------------------------------------------------------------------
    for sw in ("ratio_synth_recall", "ratio_synth_hop2", "ratio_synth_local"):
        rr = [r for r in rows if r["sweep"] == sw]
        if not rr:
            continue
        groups = defaultdict(list)
        for r in rr:
            groups[r["nA"]].append(r)
        W(f"\n## {sw}: task_em by ratio\n")
        W("| nA | nM | runs | task_em | val_loss |")
        W("|---|---|---|---|---|")
        for nA in sorted(groups):
            g = groups[nA]
            m_em, _, _ = agg([r.get("task_em") for r in g])
            m_val, _, _ = agg([r.get("val_loss") for r in g])
            W(f"| {nA} | {g[0]['nM']} | {len(g)} | {fmt(m_em,3)} | {fmt(m_val,4)} |")
        ems = [(nA, agg([r.get('task_em') for r in groups[nA]])[0]) for nA in sorted(groups)]
        ems = [(a, m) for a, m in ems if m is not None]
        if len(ems) >= 3:
            diffs = [ems[i + 1][1] - ems[i][1] for i in range(len(ems) - 1)]
            W(f"\n  task_em step diffs (increasing attn): " + " ".join(f"{d:+.3f}" for d in diffs) + "\n")

    # ---- arrangement interaction (jumbo) ------------------------------------------------------
    rr = [r for r in rows if r["sweep"] == "ratio_jumbo"]
    if rr:
        W("\n## arrangement interaction (ratio_jumbo, mean over seeds)\n")
        W("| ratio | alt | r0 | r1 | r2 | spread |")
        W("|---|---|---|---|---|---|")
        by_ratio = defaultdict(dict)
        for r in rr:
            if r.get("val_loss") is not None:
                by_ratio[r["nA"]].setdefault(r["arr"], []).append(r["val_loss"])
        for nA in sorted(by_ratio):
            a = {k: st.mean(v) for k, v in by_ratio[nA].items()}
            spread = (max(a.values()) - min(a.values())) if a else None
            W(f"| {nA}A | " + " | ".join(fmt(a.get(k), 4) for k in ("alt", "r0", "r1", "r2")) + f" | {fmt(spread,4)} |")

    # ---- convergence (jumbo) ------------------------------------------------------------------
    rr = [r for r in rows if r["sweep"] == "ratio_jumbo" and r.get("val_curve")]
    if rr:
        W("\n## convergence (jumbo, mean best-achieved val per ratio at each eval step)\n")
        steps = sorted(set(s for r in rr for s, _ in r["val_curve"]))
        W("| nA | " + " | ".join(str(s) for s in steps) + " | best | final |")
        W("|" + "---|" * (len(steps) + 3))
        groups = defaultdict(list)
        for r in rr:
            groups[r["nA"]].append(r)
        for nA in sorted(groups):
            row = [f"**{nA}**"]
            for s in steps:
                vals = [v for r in groups[nA] for ss, v in r["val_curve"] if ss == s]
                row.append(fmt(st.mean(vals), 4) if vals else "—")
            row.append(fmt(min(r["best_val"] for r in groups[nA] if r.get("best_val") is not None), 4))
            finals = [v for r in groups[nA] for ss, v in r["val_curve"] if ss == steps[-1]]
            row.append(fmt(st.mean(finals), 4) if finals else "—")
            W("| " + " | ".join(row) + " |")

    (EXP / "tables.md").write_text("\n".join(out) + "\n")
    summary["n_runs"] = len(rows)
    (EXP / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {EXP/'tables.md'} and {EXP/'summary.json'}")


if __name__ == "__main__":
    main()
