#!/usr/bin/env python3
"""Build experiments/gates/BLOG.html (self-contained, data inlined) from blog_template.html + results.json + run metrics."""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = "/data/susanket/text/gpt/runs"
HERE = os.path.dirname(os.path.abspath(__file__))


def f(x):
    return float(x)


def load_runs():
    results = json.load(open(os.path.join(HERE, "results.json")))
    runs = []
    for cell, cellruns in results["runs"].items():
        for r in cellruns:
            if r.get("variant") == "plain" or "gate_prod" not in r:
                continue
            s = r.get("stats", {})
            runs.append({
                "cell": cell, "name": r["name"], "seed": r["seed"], "variant": r["variant"],
                "n": r["n_layer"],
                "g": [round(f(v), 5) for v in r["gate_prod"]],
                "ga": [round(f(v), 5) for v in r.get("gate_attn", [])],
                "gm": [round(f(v), 5) for v in r.get("gate_mlp", [])],
                "d": [round(f(v), 5) for v in r["dloss"]],
                "s": [round(f(v), 2) for v in r["scale"]],
                "rho": s.get("rho_gate_ablate"), "p": s.get("p_perm_gate_ablate"),
                "rho_gs": s.get("rho_gate_scale"), "part": s.get("partial_rho_gate_given_scale"),
                "rho_sa": s.get("rho_scale_ablate"),
                "top1": s.get("top1_match"), "top2": s.get("top2_match"),
                "argmin_g": s.get("argmin_gate"), "argmin_a": s.get("argmin_ablate"),
            })
    return runs, results


def load_parity(results):
    out = {}
    for cell, v in results.get("val_parity", {}).items():
        gated = {}
        for k, val in v.items():
            if k.startswith("gated") and "delta" in k:
                variant = k.split("_delta_vs_plain")[0]
                gated[variant] = {"mean": v[k.replace("delta_vs_plain", "val_mean")], "delta": val}
        out[cell] = {"plain_mean": v["plain_val_mean"], "plain_std": v["plain_val_std"], "gated": gated}
    return out


def traj_series(cell, run, n):
    p = f"{RUNS}/{cell}/{run}/metrics.jsonl"
    steps, attn, mlp = [], [], []
    for line in open(p):
        d = json.loads(line)
        if d.get("split") != "train":
            continue
        steps.append(d["step"])
        a = [round(d[f"gate/L{i}/attn"], 4) for i in range(n)]
        m = [round(d[f"gate/L{i}/mlp"], 4) for i in range(n)]
        attn.append(a)
        mlp.append(m)
    vals = [a + m for a, m in zip(attn, mlp)]  # [attn_0..n, mlp_0..n]
    return steps, vals


def load_traj():
    s1_steps, s1_vals = traj_series("gates_s", "gated1-s0", 8)
    s30_steps, s30_vals = traj_series("gates_s", "gated30-s0", 8)
    k = min(len(s1_vals), len(s30_vals))
    s1_vals, s30_vals = s1_vals[:k], s30_vals[:k]
    s1_steps, s30_steps = s1_steps[:k], s30_steps[:k]
    d_steps, d_vals = traj_series("gates_d_edu", "gated30-s0", 16)
    return {
        "s": {"series": [
            {"label": "g1", "color": "#b48cff", "dash": True, "steps": s1_steps, "vals": s1_vals},
            {"label": "g30", "color": "#5cc8ff", "dash": False, "steps": s30_steps, "vals": s30_vals},
        ]},
        "d": {"series": [
            {"label": "g30", "color": "#5cc8ff", "dash": False, "steps": d_steps, "vals": d_vals},
        ]},
    }


def main():
    runs, results = load_runs()
    assert len(runs) == 27, f"expected 27 gated runs, got {len(runs)}"
    data = {
        "runs": runs,
        "parity": load_parity(results),
        "movement": results.get("gate_movement"),
        "pooled": results.get("pooled"),
        "traj": load_traj(),
        "meta": {
            "total_runs": 42, "gated_runs": 27, "tokens_B": 6.5,
            "wall": "~2h07m", "gpus": 3, "date": "2026-11-08",
        },
    }
    template = open(os.path.join(HERE, "blog_template.html")).read()
    marker = "const DATA = /*__DATA__*/ null;"
    assert marker in template, "DATA marker not found in template"
    payload = "const DATA = " + json.dumps(data, separators=(",", ":")) + ";"
    html = template.replace(marker, payload)

    # validate the page JS with esprima
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    assert m, "no script block found"
    try:
        import esprima
        esprima.parseScript(m.group(1), options={"tolerant": False})
        print("JS syntax: OK (esprima)")
    except ImportError:
        print("esprima not installed; skipping JS validation")
    except Exception as e:
        print(f"JS SYNTAX ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    out = os.path.join(HERE, "BLOG.html")
    open(out, "w").write(html)
    print(f"wrote {out} ({len(html)/1024:.0f} KB, data: {len(json.dumps(data))/1024:.0f} KB)")


if __name__ == "__main__":
    main()
