"""Analysis for the gates study: do learned layer gates predict which layers matter?

For every run under <runs_base>/gates_* (cells: s, m, m_edu, l_edu, d_edu) it extracts:

  gates      ckpt state dict: blocks.<i>.res_attn.res_g / blocks.<i>.res_mlp.res_g  (gated runs)
  ablation   eval.json: ablate_dloss/L<i>            (loss increase when block i is skipped)
  scale      ckpt: output-projection norms  attn.proj.weight, mlp.down.weight
             (the "gate is redundant with output scale" null)
  lens       eval.json: lens_loss/*                 (secondary importance measure)
  movement   metrics.jsonl: first vs last logged gate/L<i>/<site>
  val_loss   eval.json                              (gated vs plain parity, seed noise floor)

and computes, per run:
  Spearman/Pearson rho(gate, ablation)  [primary: product gate G_i = g_attn_i * g_mlp_i;
                                         plus |G_i|, and per-sublayer versions]
  permutation p-value (2000 shuffles of the ablation vector within the run)
  top-1 / top-2 layer agreement (lowest-gate layer == lowest-ablation layer?)
  the scale null: rho(scale, ablation), rho(G, scale)
  rho(G, lens marginal drop)

Outputs (next to this file): results.json, results.md, plots/*.png
Usage: python experiments/gates/analyze.py [runs_base]   (default /data/susanket/text/gpt/runs)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

RUNS_BASE = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/susanket/text/gpt/runs")
HERE = Path(__file__).resolve().parent
CELL_ORDER = ["gates_s", "gates_m", "gates_m_edu", "gates_l_edu", "gates_d_edu"]
N_PERM = 2000
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- helpers
def spearman(x, y):
    """Spearman rank correlation (average ties), no scipy dependency needed."""
    def rank(a):
        a = np.asarray(a, dtype=float)
        order = a.argsort(kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
            i = j + 1
        return ranks
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def perm_p(x, y, n_perm=N_PERM, method=spearman):
    obs = abs(method(x, y))
    if math.isnan(obs):
        return float("nan")
    y = np.asarray(y, float)
    cnt = 0
    for _ in range(n_perm):
        if abs(method(x, RNG.permutation(y))) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def load_state(run_dir: Path) -> dict:
    import torch
    f = run_dir / "ckpt_best.pt"
    if not f.exists():
        f = run_dir / "ckpt_last.pt"
    st = torch.load(f, map_location="cpu", weights_only=False)
    return st["model"]


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def variant_of(cfg: dict) -> str:
    res = (cfg.get("model") or {}).get("residual") or {}
    name = res.get("name", "plain") if isinstance(res, dict) else str(res)
    if name == "plain":
        return "plain"
    groups = (cfg.get("optim") or {}).get("groups") or []
    mult = 1
    for g in groups:
        if "res_" in str(g.get("match", "")):
            mult = g.get("lr_mult", 1)
    return f"gated{int(mult) if float(mult).is_integer() else mult}"


# ----------------------------------------------------------------------------- per run
def extract(run_dir: Path) -> dict | None:
    if not (run_dir / "run.json").exists():
        return None
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    info = read_json(run_dir / "run.json")
    if info.get("status") not in ("done", "interrupted", "failed"):
        if not (run_dir / "eval.json").exists():
            return None
    n = cfg["model"]["n_layer"]
    try:
        ev = read_json(run_dir / "eval.json")["metrics"]
    except FileNotFoundError:
        return None
    dloss = np.array([ev.get(f"ablate_dloss/L{i}", float("nan")) for i in range(n)], float)
    out = {
        "run": str(run_dir), "name": run_dir.name, "seed": info.get("seed"),
        "variant": variant_of(cfg), "cell": run_dir.parent.name, "status": info.get("status"),
        "n_layer": n, "d_model": cfg["model"]["d_model"], "data": cfg["data"]["name"],
        "val_loss": ev.get("val_loss"), "val_bpb": ev.get("val_bpb"),
        "dloss": dloss,
    }
    if ev.get("lens_loss/embed") is not None:
        prev = ev["lens_loss/embed"]
        out["lens_marginal"] = np.array([prev - ev.get(f"lens_loss/L{i}", float("nan")) for i in range(n)], float)
    # gates + output-weight scale straight from the checkpoint the eval used
    try:
        st = load_state(run_dir)
        ga, gm = [], []
        wa, wm = [], []
        for i in range(n):
            ka, km = f"blocks.{i}.res_attn.res_g", f"blocks.{i}.res_mlp.res_g"
            if ka in st:
                ga.append(float(st[ka])); gm.append(float(st[km]))
            wa.append(float(np.linalg.norm(np.asarray(st[f"blocks.{i}.attn.proj.weight"], dtype=np.float64))))
            wm.append(float(np.linalg.norm(np.asarray(st[f"blocks.{i}.mlp.down.weight"], dtype=np.float64))))
        if ga:
            ga, gm = np.array(ga), np.array(gm)
            out["gate_attn"], out["gate_mlp"] = ga, gm
            out["gate_prod"] = ga * gm
            out["gate_prod_abs"] = np.abs(ga * gm)
        out["scale"] = np.hypot(np.array(wa), np.array(wm))
        out["scale_attn"], out["scale_mlp"] = np.array(wa), np.array(wm)
    except Exception as e:
        out["state_err"] = repr(e)
    # gate movement from metrics.jsonl (first vs last logged value; init = 1.0)
    try:
        rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in rows if "gate/L0/attn" in r]
        if rows:
            first, last = rows[0], rows[-1]
            out["gate_move_attn"] = [abs(last[f"gate/L{i}/attn"] - 1.0) for i in range(n)]
            out["gate_move_mlp"] = [abs(last[f"gate/L{i}/mlp"] - 1.0) for i in range(n)]
            out["gate_first_attn"] = [first[f"gate/L{i}/attn"] for i in range(n)]
            out["gate_last_attn"] = [last[f"gate/L{i}/attn"] for i in range(n)]
            out["gate_log_first_step"] = first.get("step")
            out["gate_log_last_step"] = last.get("step")
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------- correlations
def run_stats(r: dict) -> dict:
    s = {}
    if "gate_prod" in r and not math.isnan(r["dloss"].min()):
        d = r["dloss"]
        g = r["gate_prod"]
        s["rho_gate_ablate"] = spearman(g, d)
        s["rho_gate_abs_ablate"] = spearman(r["gate_prod_abs"], d)
        s["rho_gate_attn_ablate"] = spearman(r["gate_attn"], d)
        s["rho_gate_mlp_ablate"] = spearman(r["gate_mlp"], d)
        s["pearson_gate_ablate"] = pearson(g, d)
        s["p_perm_gate_ablate"] = perm_p(g, d)
        if "lens_marginal" in r:
            s["rho_gate_lens"] = spearman(g, r["lens_marginal"])
        if "scale" in r:
            s["rho_scale_ablate"] = spearman(r["scale"], d)
            s["rho_gate_scale"] = spearman(g, r["scale"])
            # does the gate add anything beyond scale? (Pearson partial corr, standardized)
            sc = np.asarray(r["scale"], float)
            if sc.std() > 0 and d.std() > 0 and g.std() > 0:
                gs, ds, ss = (g - g.mean()) / g.std(), (d - d.mean()) / d.std(), (sc - sc.mean()) / sc.std()
                beta = float(np.dot(gs, ds) / np.dot(gs, gs))
                resid = ds - beta * ss
                s["partial_rho_gate_given_scale"] = float(np.dot(g - g.mean(), resid) / np.sqrt(np.dot(g - g.mean(), g - g.mean()) * np.dot(resid, resid)))
        # top-k agreement: lowest-gate layer(s) == lowest-ablation layer(s)
        order_g = np.argsort(g)
        order_d = np.argsort(d)
        s["top1_match"] = int(order_g[0] == order_d[0])
        s["top2_match"] = int(set(order_g[:2]) == set(order_d[:2]))
        s["argmin_gate"] = int(order_g[0])
        s["argmin_ablate"] = int(order_d[0])
    elif "scale" in r and not math.isnan(r["dloss"].min()):
        s["rho_scale_ablate"] = spearman(r["scale"], r["dloss"])
    return s


def summarize(runs: list[dict]) -> dict:
    agg = {}
    keyed = {}
    for r in runs:
        if not r:
            continue
        s = r.get("stats", {})
        key = (r["cell"], r["variant"])
        keyed.setdefault(key, []).append(s)
    for key, ss in sorted(keyed.items()):
        def stat(k, fn=np.mean):
            vals = [x[k] for x in ss if k in x and not (isinstance(x[k], float) and math.isnan(x[k]))]
            if not vals:
                return None
            out = float(fn(vals))
            out = {"mean": out, "std": float(np.std(vals)) if len(vals) > 1 else None, "n": len(vals),
                   "vals": [round(float(v), 4) for v in vals]}
            return out
        row = {}
        for k in ("rho_gate_ablate", "rho_gate_abs_ablate", "rho_gate_attn_ablate", "rho_gate_mlp_ablate",
                  "rho_scale_ablate", "rho_gate_scale", "partial_rho_gate_given_scale", "rho_gate_lens",
                  "p_perm_gate_ablate", "top1_match", "top2_match"):
            v = stat(k)
            if v:
                row[k] = v
        agg[key] = row
    return agg


# ----------------------------------------------------------------------------- main
def main():
    cells = {}
    for cell in CELL_ORDER:
        cdir = RUNS_BASE / cell
        if not cdir.exists():
            continue
        runs = []
        for d in sorted(cdir.iterdir()):
            r = extract(d)
            if r:
                r["stats"] = run_stats(r)
                runs.append(r)
        cells[cell] = runs

    agg = summarize([r for cell in cells.values() for r in cell])

    # pooled permutation test: within-run layer shuffles, mean |rho| across gated runs
    gated = [r for cell in cells.values() for r in cell if "gate_prod" in r and not math.isnan(r["dloss"].min())]
    obs = np.mean([abs(r["stats"]["rho_gate_ablate"]) for r in gated])
    sim = []
    for _ in range(N_PERM):
        tot = 0.0
        for r in gated:
            tot += abs(spearman(r["gate_prod"], RNG.permutation(r["dloss"])))
        sim.append(tot / len(gated))
    p_pooled = float(np.mean(np.asarray(sim) >= obs - 1e-12))

    # gate movement summary + val parity
    movement, parity = {}, {}
    for cell, runs in cells.items():
        mv = [r.get("gate_move_attn") for r in runs if r.get("gate_move_attn")]
        if mv:
            arr = np.stack(mv)
            movement[cell] = {"max_abs_move": float(arr.max()), "mean_abs_move": float(arr.mean()),
                              "min_abs_move": float(arr.min()), "n_runs": len(mv)}
        pl = [r["val_loss"] for r in runs if r["variant"] == "plain" and r.get("val_loss")]
        if len(pl) > 1:
            parity[cell] = {"plain_val_mean": float(np.mean(pl)), "plain_val_std": float(np.std(pl)), "n": len(pl)}
            for v in ("gated1", "gated30", "gated100"):
                gv = [r["val_loss"] for r in runs if r["variant"] == v and r.get("val_loss")]
                if gv:
                    parity[cell][f"{v}_val_mean"] = float(np.mean(gv))
                    parity[cell][f"{v}_delta_vs_plain"] = float(np.mean(gv) - np.mean(pl))

    out = {"runs": cells, "aggregate": {f"{c}|{v}": a for (c, v), a in agg.items()},
           "pooled": {"n_gated_runs": len(gated), "mean_abs_rho_observed": float(obs), "p_pooled": p_pooled},
           "gate_movement": movement, "val_parity": parity}
    def _ser(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        return float(o)

    (HERE / "results.json").write_text(json.dumps(out, indent=2, default=_ser))

    # ---- markdown
    md = []
    md.append("# Gates study - results\n")
    md.append(f"runs base: {RUNS_BASE}\n")
    md.append(f"gated runs analyzed: {len(gated)}; pooled mean |spearman(gate, ablation)| = {obs:.3f}, "
              f"pooled permutation p = {p_pooled:.4f} ({N_PERM} shuffles)\n")
    md.append("\n## Per-run correlations (gated runs)\n")
    md.append("| run | cell | variant | rho(gate,abl) | rho(|gate|,abl) | rho(ga,abl) | rho(gm,abl) | p_perm | rho(scale,abl) | rho(gate,scale) | part_rho(gate|scale) | top1 | top2 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in CELL_ORDER:
        for r in cells.get(cell, []):
            s = r.get("stats", {})
            if "rho_gate_ablate" not in s:
                continue
            md.append(f"| {r['name']} | {r['cell']} | {r['variant']} | {s['rho_gate_ablate']:+.3f} | "
                      f"{s.get('rho_gate_abs_ablate', float('nan')):+.3f} | {s.get('rho_gate_attn_ablate', float('nan')):+.3f} | "
                      f"{s.get('rho_gate_mlp_ablate', float('nan')):+.3f} | {s.get('p_perm_gate_ablate', float('nan')):.3f} | "
                      f"{s.get('rho_scale_ablate', float('nan')):+.3f} | {s.get('rho_gate_scale', float('nan')):+.3f} | "
                      f"{s.get('partial_rho_gate_given_scale', float('nan')):+.3f} | "
                      f"{s.get('top1_match', '-')} | {s.get('top2_match', '-')} |")
    md.append("\n## Gate movement (|g_final - 1.0| per layer, from checkpoint)\n")
    for cell, m in movement.items():
        md.append(f"- {cell}: max {m['max_abs_move']:.3f}, mean {m['mean_abs_move']:.3f}, min {m['min_abs_move']:.4f} over {m['n_runs']} runs")
    md.append("\n## val loss parity (plain seed-std = noise floor)\n")
    for cell, p in parity.items():
        md.append(f"- {cell}: plain {p['plain_val_mean']:.4f} +/- {p['plain_val_std']:.4f} (n={p['n']}" +
                  "".join(f"; {k} {v:+.4f}" for k, v in p.items() if k.startswith("gated")) + ")")
    md.append("\n## Per-layer vectors (gated30 runs)\n")
    for cell in CELL_ORDER:
        for r in cells.get(cell, []):
            if r.get("variant") != "gated30" or "gate_prod" not in r:
                continue
            md.append(f"\n### {r['name']} (seed {r['seed']})\n")
            md.append("| layer | gate_attn | gate_mlp | gate_prod | ablate_dloss | rank_g | rank_a |")
            md.append("|---|---|---|---|---|---|---|")
            rg = np.argsort(np.argsort(r["gate_prod"]))
            ra = np.argsort(np.argsort(r["dloss"]))
            for i in range(r["n_layer"]):
                md.append(f"| {i} | {r['gate_attn'][i]:+.3f} | {r['gate_mlp'][i]:+.3f} | "
                          f"{r['gate_prod'][i]:+.3f} | {r['dloss'][i]:.4f} | {rg[i]} | {ra[i]} |")
    (HERE / "results.md").write_text("\n".join(md) + "\n")

    # ---- plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pdir = HERE / "plots"
        pdir.mkdir(exist_ok=True)
        for cell in CELL_ORDER:
            for r in cells.get(cell, []):
                if "gate_prod" not in r:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
                g = r["gate_prod"]
                layers = np.arange(r["n_layer"])
                axes[0].scatter(g, r["dloss"], s=28)
                for i, (x, y) in enumerate(zip(g, r["dloss"])):
                    axes[0].annotate(str(i), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
                rho = r["stats"].get("rho_gate_ablate")
                axes[0].set_title(f"{r['name']}  spearman={rho:+.2f}" if rho is not None else r["name"])
                axes[0].set_xlabel("block gate g_attn*g_mlp (final)")
                axes[0].set_ylabel("ablate dloss (nats)")
                axes[0].grid(alpha=0.3)
                ax2 = axes[1]
                ax2.bar(layers - 0.18, g, width=0.36, label="gate product")
                ax2b = ax2.twinx()
                ax2b.bar(layers + 0.18, r["dloss"], width=0.36, alpha=0.7, label="ablation dloss")
                ax2.set_ylabel("gate product")
                ax2b.set_ylabel("ablate dloss")
                ax2.set_title(f"per-layer: {r['name']}")
                ax2.set_xticks(layers)
                fig.tight_layout()
                fig.savefig(pdir / f"{cell}_{r['name']}.png", dpi=130)
                plt.close(fig)
        # gate trajectories for one run per cell
        for cell in CELL_ORDER:
            runs = [r for r in cells.get(cell, []) if r.get("variant") == "gated30" and r.get("gate_last_attn")]
            if not runs:
                continue
            r = runs[0]
            rows = [json.loads(l) for l in (Path(r["run"]) / "metrics.jsonl").read_text().splitlines() if l.strip()]
            rows = [x for x in rows if "gate/L0/attn" in x]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for i in range(r["n_layer"]):
                steps = [x["step"] for x in rows]
                for site in ("attn", "mlp"):
                    vals = [np.sign(v) * np.log1p(abs(v)) for v in (x[f"gate/L{i}/{site}"] for x in rows)]
                    ax.plot(steps, vals, label=f"L{i}" if site == "attn" else None,
                            linestyle="-" if site == "attn" else "--")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("step")
            ax.set_ylabel("sign(g) * log(1+|g|)")
            ax.set_title(f"gate trajectories: {r['name']} (solid=attn, dashed=mlp)")
            ax.grid(alpha=0.3, which="both")
            fig.tight_layout()
            fig.savefig(pdir / f"{cell}_trajectories.png", dpi=130)
            plt.close(fig)
        print(f"plots -> {pdir}")
    except Exception as e:
        print(f"(plots skipped: {e!r})")

    print(json.dumps({"pooled": out["pooled"], "movement": movement, "parity": parity}, indent=2))
    print(f"wrote {HERE / 'results.json'} and {HERE / 'results.md'}")


if __name__ == "__main__":
    main()
