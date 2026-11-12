"""Generate layout_schedule_2 ratio-sweep configs + prioritized run list.

Ratio grid (total sublayers fixed; MLP:attention as the user specified):
  48-sublayer jumbo:  nA in {10, 12, 16, 24, 32, 36, 38}   (38M10A ... 10M38A)
  24-sublayer depth:  nA in { 5,  6,  8, 12, 16, 18, 19}   (19M5A ...  5M19A)
  refinement jumbo:   nA in {14, 18, 28, 34}               (fills the coarse gaps)

Families:
  raw   - d_ff fixed at the baseline default (the user's spec: width/attn/MLP-expansion fixed)
  pm    - d_ff tuned per ratio so total params match the 24A+24M baseline (<0.2%)
Arrangements per ratio: alt (evenly spread) + fixed-seed random permutations.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
CFG = ROOT / "configs" / "sweeps" / "ratio2"
OUT = Path(__file__).resolve().parent
CFG.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]


def alt(nA: int, nM: int) -> str:
    """Distribute nA A's and nM M's as evenly as possible."""
    total = nA + nM
    if nA <= nM:
        positions = sorted({int(i * total / nA) for i in range(nA)})
        posset = set(positions)
        return "".join("A" if i in posset else "M" for i in range(total))
    return "".join("M" if c == "A" else "A" for c in alt(nM, nA))


def rand_layout(nA: int, nM: int, seed: int) -> str:
    s = ["A"] * nA + ["M"] * nM
    random.Random(seed).shuffle(s)
    return "".join(s)


def arrangements(nA: int, nM: int, n_rand: int = 4):
    out = {"alt": alt(nA, nM)}
    for i in range(n_rand):
        out[f"r{i}"] = rand_layout(nA, nM, 1000 + nA * 7 + i)
    return out


def build_params(n_layer, d, n_head, nA, nM, d_ff, vocab=8192):
    """Exact param counts for a layout stack (mha: 4*d*d per A; swiglu: 3*d*d_ff per M)."""
    import torch
    from gpt_sandbox.plugins import default_plugin_paths, load_plugins
    load_plugins(default_plugin_paths())
    from gpt_sandbox.config import load_config
    from gpt_sandbox.model.gpt import GPT

    c = load_config(ROOT / "configs" / "layouts" / "ratio2_jumbo.yaml",
                    {"model.n_layer": n_layer, "model.d_model": d, "model.n_head": n_head,
                     "model.extra.layout": "A" * nA + "M" * nM,
                     "model.mlp": {"name": "swiglu", "d_ff": d_ff}}).model
    c.vocab_size = vocab
    torch.manual_seed(0)
    m = GPT(c)
    return m.num_params(), m.num_params(non_embedding=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-params", action="store_true", help="build every model on CPU to confirm param counts")
    args = ap.parse_args()

    # ---- exact baseline (1:1) non-embedding params at d=512 -------------------------------------
    d = 512
    _, P_ne = build_params(48, d, 8, 24, 24, 1376)
    norms = 49 * d  # 48 block norms + final norm
    perA = 4 * d * d
    perM = 3 * d  # * d_ff
    print(f"baseline 24A+24M d_ff=1376: params_nonemb={P_ne:,}  (norms={norms:,}, perA={perA:,})")
    assert P_ne == norms + 24 * perA + 24 * perM * 1376, "param formula mismatch"

    def tune_dff(nA, nM):
        exact = (P_ne - norms - nA * perA) / (nM * perM)
        dff = int(round(exact / 8)) * 8
        achieved = norms + nA * perA + nM * perM * dff
        return dff, (achieved - P_ne) / P_ne

    runlist = []  # {sweep, variant, seed, cfg, steps, prio, nA, nM, dff}

    def emit(sweep, variant, layout, cfg_rel, steps, prio, nA, nM, dff=None, data=None):
        assert layout.count("A") == nA and len(layout) == nA + nM, f"{variant}: bad layout {layout!r}"
        cfg = CFG / f"{sweep}__{variant}.yaml"
        lines = [f"base: ../../layouts/{cfg_rel}", f"name: {sweep}_{variant}",
                 "model:", "  extra:", f'    layout: "{layout}"']
        if dff is not None:
            lines.append(f"  mlp: {{name: swiglu, d_ff: {dff}}}")
        if data:
            lines.append(f"data: {{name: {data}}}")
        lines.append(f"train: {{max_steps: {steps}}}")
        cfg.write_text("\n".join(lines) + "\n")
        for seed in SEEDS:
            runlist.append({"sweep": sweep, "variant": variant, "seed": seed,
                            "cfg": str(cfg), "steps": steps, "prio": prio,
                            "nA": nA, "nM": nM, "dff": dff, "data": data})

    # ---- family 1: raw ratio sweep, jumbo (d_ff fixed = baseline default) -------------------------
    print("\n=== ratio_jumbo (raw, fixed d_ff) ===")
    for nA, tag in ((10, "r41"), (12, "r31"), (16, "r21"), (24, "r11"), (32, "r12"), (36, "r13"), (38, "r14")):
        nM = 48 - nA
        for arr, layout in arrangements(nA, nM, n_rand=3).items():
            emit("ratio_jumbo", f"{tag}_{nA}A{nM}M_{arr}", layout, "ratio2_jumbo.yaml", 6000, 0, nA, nM)
        print(f"  {tag}  {nA:2d}A + {nM:2d}M   params ~ {norms + nA*perA + nM*perM*1376:,}  "
              f"{(norms + nA*perA + nM*perM*1376 - P_ne)/P_ne*100:+.1f}% vs baseline")

    # ---- family 2: param-matched, jumbo (d_ff tuned per ratio) ------------------------------------
    print("\n=== ratio_pm_jumbo (params matched) ===")
    for nA, tag in ((10, "pmr41"), (12, "pmr31"), (16, "pmr21"), (32, "pmr12"), (36, "pmr13"), (38, "pmr14")):
        nM = 48 - nA
        dff, delta = tune_dff(nA, nM)
        for arr, layout in arrangements(nA, nM, n_rand=1).items():
            emit("ratio_pm_jumbo", f"{tag}_{nA}A{nM}M_{arr}", layout, "ratio2_jumbo.yaml", 6000, 2, nA, nM, dff=dff)
        print(f"  {tag}  {nA:2d}A + {nM:2d}M   d_ff={dff:5d}   params delta {delta*100:+.2f}%")

    # ---- family 3: depth ablation (24 sublayers, d=256) -------------------------------------------
    print("\n=== ratio_depth24 (24 sublayers) ===")
    for nA, tag in ((5, "r41"), (6, "r31"), (8, "r21"), (12, "r11"), (16, "r12"), (18, "r13"), (19, "r14")):
        nM = 24 - nA
        for arr, layout in arrangements(nA, nM, n_rand=1).items():
            emit("ratio_depth24", f"{tag}_{nA}A{nM}M_{arr}", layout, "ratio2_depth24.yaml", 12000, 3, nA, nM)
        print(f"  {tag}  {nA:2d}A + {nM:2d}M")

    # ---- family 4: synthetic task probes (24 sublayers, d=128) -------------------------------------
    for task, ds in (("recall", "synth_recall"), ("hop2", "synth_hop2"), ("local", "synth_local_rule")):
        print(f"\n=== ratio_synth_{task} ===")
        for nA, tag in ((5, "r41"), (6, "r31"), (8, "r21"), (12, "r11"), (16, "r12"), (18, "r13"), (19, "r14")):
            nM = 24 - nA
            for arr, layout in arrangements(nA, nM, n_rand=1).items():
                emit(f"ratio_synth_{task}", f"{tag}_{nA}A{nM}M_{arr}", layout, "ratio2_synth.yaml", 16000, 1, nA, nM, data=ds)

    # ---- family 5: refinement of the coarse jumbo curve --------------------------------------------
    print("\n=== ratio_fine_jumbo (intermediate ratios) ===")
    for nA in (14, 18, 28, 34):
        nM = 48 - nA
        for arr, layout in arrangements(nA, nM, n_rand=1).items():
            emit("ratio_fine_jumbo", f"rf_{nA}A{nM}M_{arr}", layout, "ratio2_jumbo.yaml", 6000, 4, nA, nM)
        print(f"  {nA:2d}A + {nM:2d}M   ({nM/nA:.2f}:1)")

    # ---- family 6 (stretch): longer training on anchor + both extremes ------------------------------
    print("\n=== ratio_jumbo_long (12000 steps, stretch) ===")
    for nA, tag in ((24, "r11"), (10, "r41"), (38, "r14"), (16, "r21"), (32, "r12")):
        nM = 48 - nA
        emit("ratio_jumbo_long", f"{tag}_{nA}A{nM}M_alt", alt(nA, nM), "ratio2_jumbo.yaml", 12000, 5, nA, nM)

    runlist.sort(key=lambda r: (r["prio"], r["sweep"], r["variant"], r["seed"]))
    (OUT / "runlist.json").write_text(json.dumps(runlist, indent=1))
    from collections import Counter
    c = Counter(r["sweep"] for r in runlist)
    print(f"\nwrote {len(runlist)} runs to {OUT/'runlist.json'}")
    for s, n in sorted(c.items()):
        print(f"  {s:22s} {n:4d} runs")

    if args.verify_params:
        print("\n=== verifying pm param matches by building models (CPU) ===")
        for nA, tag in ((10, "pmr41"), (12, "pmr31"), (16, "pmr21"), (32, "pmr12"), (36, "pmr13"), (38, "pmr14")):
            nM = 48 - nA
            dff, _ = tune_dff(nA, nM)
            _, p = build_params(48, d, 8, nA, nM, dff)
            print(f"  {tag} {nA}A{nM}M d_ff={dff}: nonemb={p:,}  delta={(p-P_ne)/P_ne*100:+.3f}%")


if __name__ == "__main__":
    main()
