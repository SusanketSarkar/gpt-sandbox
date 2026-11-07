"""compare / plot / info / list: look at runs and models."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import registry
from .model.spec import select_spec
from .runs import find_runs, read_metrics
from .utils import fmt, human, read_json, table

DEFAULT_COLS = ["val_loss", "val_bpb", "val_acc1", "gain_vs_bigram", "ctx_gain", "induction_gap",
                "task_em", "gen_gap", "params_total", "fwdbwd_tok_per_s"]


def _load(run: Path) -> dict:
    info = read_json(run / "run.json")
    ev = read_json(run / "eval.json")["metrics"] if (run / "eval.json").exists() else {}
    return {"run": run, "info": info, "m": ev}


def compare(paths, cols=None, aggregate=True, baseline=None) -> str:
    runs = [_load(r) for r in find_runs(paths)]
    if not runs:
        return "no runs found (looked in: %s)" % (list(paths) or ["runs"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        key = r["info"].get("group") or r["run"].name
        key = f"{r['info'].get('sweep')}/{key}" if r["info"].get("sweep") and aggregate else r["run"].name if not aggregate else key
        groups[key].append(r)
    all_keys = set().union(*(r["m"].keys() for r in runs))
    cols = cols or [c for c in DEFAULT_COLS if c in all_keys] or sorted(all_keys)[:8]
    rows = []
    base_vals = None
    for name, rs in groups.items():
        row = [name, f"{len(rs)}" if aggregate else rs[0]["info"].get("status", "?")]
        vals = {}
        for c in cols:
            xs = [r["m"][c] for r in rs if c in r["m"] and r["m"][c] == r["m"][c]]
            if not xs:
                row.append("-")
                continue
            mean, sd = float(np.mean(xs)), float(np.std(xs)) if len(xs) > 1 else None
            vals[c] = mean
            if c in ("params_total", "flops_per_token"):
                row.append(human(mean))
            elif "tok_per_s" in c:
                row.append(human(mean))
            else:
                row.append(fmt(mean) + (f" ±{sd:.4f}" if sd is not None else ""))
        if base_vals is None:
            base_vals = vals
        dk = "val_bpb" if "val_bpb" in vals else "val_loss"
        d = (vals[dk] - base_vals[dk]) if dk in vals and base_vals and dk in base_vals else None
        row.append("-" if d is None else f"{d:+.4f}")
        rows.append(row)
    heads = ["run", "n" if aggregate else "status"] + cols + ["Δval_bpb vs first"]
    return table(rows, heads)


def plot(paths, keys, out="plot.png", x="step", smooth=0.9):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = find_runs(paths)
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for r in runs:
            rows = [m for m in read_metrics(r) if key in m]
            if not rows:
                continue
            xs = [m[x] for m in rows]
            ys = [m[key] for m in rows]
            if smooth and key in ("loss", "ce") and len(ys) > 5:
                s, sm = ys[0], []
                for y in ys:
                    s = smooth * s + (1 - smooth) * y
                    sm.append(s)
                ys = sm
            ax.plot(xs, ys, label=r.name)
        ax.set_xlabel(x)
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


def info(model, cfg) -> str:
    n = cfg.model.n_layer
    out = []
    rows = []
    for i in range(n):
        row = [str(i)]
        for kind in ("block", "attention", "mlp", "norm", "residual"):
            spec = select_spec(cfg.model[kind], i, n, kind)
            name, kw = registry.parse_spec(spec, kind)
            row.append(name + (f"{kw}" if kw else ""))
        ins = [f"{s}:{type(m).__name__}" for s, ml in model.blocks[i].inserts.items() for m in ml]
        row.append(", ".join(ins) or "-")
        rows.append(row)
    out.append(table(rows, ["layer", "block", "attention", "mlp", "norm", "residual", "inserts"]))
    mi = [f"{s}:{type(m).__name__}" for s, ml in model.model_inserts.items() for m in ml]
    out.append(f"\nposenc: {cfg.model.posenc}   model-level inserts: {mi or '-'}   tied embeddings: {cfg.model.tie_embeddings}")
    out.append(f"\nparameters: {model.num_params():,} total | {model.num_params(True):,} non-embedding | "
               f"~{human(model.flops_per_token())} train FLOPs/token")
    out.append("\n" + table([[k, c, f"{p:,}"] for k, c, p in model.param_breakdown()], ["parameter", "copies", "params (all copies)"]))
    return "\n".join(out)


def list_components() -> str:
    lines = []
    for kind, contract in registry.KINDS.items():
        lines.append(f"\n{kind}   --  {contract}")
        for n in registry.names(kind):
            o = registry.origin(kind, n)
            tag = "plugin" if "gpt_sandbox/" not in o.replace("\\", "/") else "builtin"
            lines.append(f"    {n:<22} [{tag}]")
    return "\n".join(lines)
