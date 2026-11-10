"""Analysis for the depth-fragility study: "where is the residual stream fragile?"

Hypothesis: noise injected into the residual stream during training (after_attn, Gaussian,
RMS-scaled, training-only) hurts MORE in the middle layers than at the ends, because middle
layers carry the least redundant information.  Question: does information density vary by depth?

For every run under <runs_base>/depth_* it extracts:

  core      val_bpb / val_loss / gen_gap / train_loss_eval       (eval.json)
  profile   ablate_dloss/L<i>, lens_loss/*, resid_rms/L<i>       (eval.json --deep)
  variant   (layer L, sigma) parsed from the run name n<L>_sig<SS> / all_sig<SS> / plain
  scale     output-projection norms attn.proj / mlp.down         (checkpoint, "scale" null)
  traj      training loss + val loss over steps                  (metrics.jsonl)

and computes, per size (M=8L, D=16L):

  noise floor      mean +/- std of plain val_bpb over seeds (effects smaller = seed noise)
  fragility        Delta(L, sigma) = bpb(noise at L, sigma) - bpb(plain)
  shape test       Spearman/Pearson of the Delta(L) curve vs a centeredness prior
                   (linear bump + Gaussian bump), permutation p-value (2000 shuffles)
  anchor test      paired-over-seeds Delta(mid)-Delta(first), Delta(last)-Delta(mid),
                   sign consistency and magnitude in units of the plain seed std
  ablation corr    Spearman(Delta(L), ablate_dloss(L)) on the plain model + permutation p
  lens corr        Spearman(Delta(L), lens marginal drop)
  scale null       Spearman(Delta(L), output-proj scale)
  regulariser      gen_gap per variant vs plain; any variant beating plain; global vs mid dose
  dynamics         final train/val loss plain vs noise variants

Outputs (next to this file): results.json, results.md, plots/*.png
Usage: python experiments/depth_fragility/analyze.py [runs_base]   (default /data/susanket/text/gpt/runs)
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
N_PERM = 2000
RNG = np.random.default_rng(0)
SIZES = {"depth_m": "M (8L)", "depth_d": "D (16L)"}
CELLS = ["depth_m_core", "depth_d_core", "depth_m_profile", "depth_d_profile_a",
         "depth_d_profile_b", "depth_m_global", "depth_d_global"]


# ----------------------------------------------------------------------------- stats helpers
def spearman(x, y):
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
    cnt = sum(1 for _ in range(n_perm) if abs(method(x, RNG.permutation(y))) >= obs - 1e-12)
    return (cnt + 1) / (n_perm + 1)


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def parse_variant(name: str) -> tuple[str, int, float | None]:
    """'plain-s0' -> ('plain', -1, None); 'n4_sig03-s1' -> ('noise', 4, 0.3); 'all_sig10-s0' -> ('all', -1, 1.0).

    Run dir names carry a '-s<seed>' suffix; strip it first.
    """
    base = name.split("-s")[0] if "-s" in name else name
    if base.startswith("n") and "_sig" in base:
        L, s = base.split("_sig")
        return "noise", int(L[1:]), int(s) / 100.0
    if base.startswith("all_sig"):
        return "all", -1, int(base.split("_sig")[1]) / 100.0
    return "plain", -1, None


def anchor_for(n_layer: int) -> tuple[int, int, int]:
    return 0, n_layer // 2, n_layer - 1


def std1(a) -> float:
    a = np.asarray(a, float)
    return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")


# ----------------------------------------------------------------------------- per run
def extract(run_dir: Path) -> dict | None:
    if not (run_dir / "run.json").exists():
        return None
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    info = read_json(run_dir / "run.json")
    if info.get("status") != "done" or not (run_dir / "eval.json").exists():
        return None
    n = cfg["model"]["n_layer"]
    # kind/layer/sigma from the run's own config (the dir name is just a label)
    ins = (cfg.get("model") or {}).get("inserts") or []
    if not ins:
        kind, layer, sigma = "plain", -1, None
    else:
        spec = ins[0]
        lyr = spec.get("layers")
        kind = "all" if lyr == "all" else "noise"
        layer = -1 if lyr == "all" else int(lyr)
        sigma = float(spec.get("sigma", 0.0))
    try:
        ev = read_json(run_dir / "eval.json")["metrics"]
    except FileNotFoundError:
        return None
    out = {
        "run": str(run_dir), "name": run_dir.name, "cell": run_dir.parent.name,
        "seed": info.get("seed"), "kind": kind, "layer": layer, "sigma": sigma,
        "n_layer": n, "d_model": cfg["model"]["d_model"],
        "val_loss": ev.get("val_loss"), "val_bpb": ev.get("val_bpb"),
        "val_acc1": ev.get("val_acc1"), "gen_gap": ev.get("gen_gap"),
        "train_loss_eval": ev.get("train_loss_eval"), "ctx_gain": ev.get("ctx_gain"),
        "induction_gap": ev.get("induction_gap"),
    }
    dloss = [ev.get(f"ablate_dloss/L{i}", float("nan")) for i in range(n)]
    if not any(math.isnan(v) for v in dloss):
        out["dloss"] = np.array(dloss, float)
    lens = [ev.get(f"lens_loss/L{i}", float("nan")) for i in range(n)]
    if ev.get("lens_loss/embed") is not None and not any(math.isnan(v) for v in lens):
        emb = ev["lens_loss/embed"]
        out["lens"] = np.array(lens, float)
        out["lens_marginal"] = np.array([emb - lens[0]] + [lens[i - 1] - lens[i] for i in range(1, n)], float)
    rms = [ev.get(f"resid_rms/L{i}", float("nan")) for i in range(n)]
    if not any(math.isnan(v) for v in rms):
        out["resid_rms"] = np.array(rms, float)
    try:
        rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
        tr = [(r["step"], r["loss"]) for r in rows if r.get("split") == "train" and "loss" in r]
        va = [(r["step"], r["val_loss"]) for r in rows if r.get("split") == "val"]
        if tr:
            out["traj"] = tr
            out["val_traj"] = va
            out["final_train_loss"] = tr[-1][1]
            out["final_val_loss"] = va[-1][1] if va else None
    except Exception:
        pass
    try:
        import torch
        st = torch.load(run_dir / "ckpt_best.pt", map_location="cpu", weights_only=False)["model"]
        wa = [float(np.linalg.norm(np.asarray(st[f"blocks.{i}.attn.proj.weight"], np.float64))) for i in range(n)]
        wm = [float(np.linalg.norm(np.asarray(st[f"blocks.{i}.mlp.down.weight"], np.float64))) for i in range(n)]
        out["scale"] = np.hypot(np.array(wa), np.array(wm))
    except Exception as e:
        out["state_err"] = repr(e)
    return out


# ----------------------------------------------------------------------------- per size
def collect(size: str) -> dict:
    runs = []
    for cell in CELLS:
        if not cell.startswith(size + "_"):
            continue
        cdir = RUNS_BASE / cell
        if not cdir.exists():
            continue
        for d in sorted(cdir.iterdir()):
            r = extract(d)
            if r:
                runs.append(r)
    plain = [r for r in runs if r["kind"] == "plain"]
    noise = [r for r in runs if r["kind"] == "noise"]
    glob = [r for r in runs if r["kind"] == "all"]
    n_layer = runs[0]["n_layer"] if runs else 0
    first, mid, last = anchor_for(n_layer)
    return {"size": size, "label": SIZES[size], "runs": runs, "plain": plain, "noise": noise,
            "global": glob, "n_layer": n_layer, "first": first, "mid": mid, "last": last}


def fragility(sd: dict) -> dict:
    plain_bpb = [r["val_bpb"] for r in sd["plain"] if r.get("val_bpb") is not None]
    plain_mean = float(np.mean(plain_bpb))
    plain_std = std1(plain_bpb)
    by_key: dict[tuple, list] = {}
    for r in sd["noise"] + sd["global"]:
        L = r["layer"] if r["kind"] == "noise" else "all"
        by_key.setdefault((L, r["sigma"]), []).append(r)
    curve = {}
    for (L, s), rs in sorted(by_key.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
        b = np.array([x["val_bpb"] for x in rs if x.get("val_bpb") is not None], float)
        key = f"{'L' if L != 'all' else ''}{L}_s{int(s*100)}" if L != "all" else f"all_s{int(s*100)}"
        curve[key] = {
            "layer": L, "sigma": s, "n_seeds": len(b),
            "bpb_mean": float(b.mean()), "bpb_std": std1(b),
            "delta": float(b.mean() - plain_mean),
            "delta_over_floor": float((b.mean() - plain_mean) / plain_std) if plain_std > 0 else None,
            "gen_gap_mean": float(np.mean([x["gen_gap"] for x in rs if x.get("gen_gap") is not None]))
            if any(x.get("gen_gap") is not None for x in rs) else None,
            "seeds": [round(x["val_bpb"], 6) for x in rs if x.get("val_bpb") is not None],
        }
    return {"plain_mean": plain_mean, "plain_std": plain_std, "curve": curve}


def anchor_test(sd: dict, frag: dict) -> dict:
    first, mid, last = sd["first"], sd["mid"], sd["last"]
    floor = frag["plain_std"]
    out = {}
    sigmas = sorted({c["sigma"] for c in frag["curve"].values() if c["layer"] in (first, mid, last)})
    for s in sigmas:
        key = lambda L: f"L{L}_s{int(s*100)}"
        if not all(key(L) in frag["curve"] for L in (first, mid, last)):
            continue
        c = {L: frag["curve"][key(L)] for L in (first, mid, last)}
        mids, lastvs = [], []
        for i in range(min(len(c[L]["seeds"]) for L in (first, mid, last))):
            mids.append(c[first]["seeds"][i] - c[mid]["seeds"][i])   # >0 => middle more fragile
            lastvs.append(c[last]["seeds"][i] - c[mid]["seeds"][i])  # >0 => last more fragile than mid
        out[f"sig{int(s*100)}"] = {
            "sigma": s,
            "d_first_minus_mid_mean": float(np.mean(mids)),
            "d_last_minus_mid_mean": float(np.mean(lastvs)),
            "signs_mid_worse_than_first": [int(v > 0) for v in mids],
            "signs_mid_worse_than_last": [int(v < 0) for v in lastvs],
            "n_seeds_paired": len(mids),
            "in_floor_units_first_minus_mid": float(np.mean(mids) / floor) if floor > 0 else None,
            "in_floor_units_last_minus_mid": float(np.mean(lastvs) / floor) if floor > 0 else None,
            "delta_first": c[first]["delta"], "delta_mid": c[mid]["delta"], "delta_last": c[last]["delta"],
        }
    return out


def curve_delta(frag: dict, n: int, s: float) -> np.ndarray | None:
    d = [frag["curve"].get(f"L{L}_s{int(s*100)}", {}).get("delta", float("nan")) for L in range(n)]
    return None if any(math.isnan(v) for v in d) else np.array(d, float)


def shape_test(sd: dict, frag: dict) -> dict:
    n = sd["n_layer"]
    c = (n - 1) / 2.0
    lin = np.array([1 - abs(L - c) / c for L in range(n)])
    w = max(c / 2.0, 1e-6)
    gauss = np.array([math.exp(-((L - c) / w) ** 2) for L in range(n)])
    out = {}
    for s_label, s_val in (("s30", 0.3), ("s100", 1.0)):
        deltas = curve_delta(frag, n, s_val)
        if deltas is None:
            continue
        out[s_label] = {
            "curve": [round(float(v), 6) for v in deltas],
            "spearman_linear_bump": spearman(deltas, lin),
            "p_perm_linear": perm_p(deltas, lin),
            "pearson_linear_bump": pearson(deltas, lin),
            "spearman_gauss_bump": spearman(deltas, gauss),
            "p_perm_gauss": perm_p(deltas, gauss),
            "spearman_depth": spearman(deltas, np.arange(n)),
            "p_perm_depth": perm_p(deltas, np.arange(n)),
        }
    return out


def ablation_corr(sd: dict, frag: dict) -> dict:
    n = sd["n_layer"]
    plain_abl = [r["dloss"] for r in sd["plain"] if "dloss" in r]
    out = {}
    if plain_abl:
        dloss = np.mean(plain_abl, axis=0)
        scales = [r["scale"] for r in sd["runs"] if "scale" in r]
        scale = np.mean(scales, axis=0) if scales else None
        lens = [r["lens_marginal"] for r in sd["plain"] if "lens_marginal" in r]
    else:
        dloss = scale = lens = None
    for s_label, s_val in (("s30", 0.3), ("s100", 1.0)):
        deltas = curve_delta(frag, n, s_val)
        if deltas is None:
            continue
        row = {"rho_fragility_ablation": spearman(deltas, dloss) if dloss is not None else None,
               "p_perm_fragility_ablation": perm_p(deltas, dloss) if dloss is not None else None,
               "pearson_fragility_ablation": pearson(deltas, dloss) if dloss is not None else None}
        if scale is not None:
            row["rho_fragility_scale"] = spearman(deltas, scale)
            row["rho_ablation_scale"] = spearman(dloss, scale)
        if lens:
            row["rho_fragility_lens"] = spearman(deltas, np.mean(lens, axis=0))
        out[s_label] = row
    return out


def regulariser_checks(sd: dict, frag: dict) -> dict:
    plain_bpb = [r["val_bpb"] for r in sd["plain"] if r.get("val_bpb") is not None]
    plain_gg = [r["gen_gap"] for r in sd["plain"] if r.get("gen_gap") is not None]
    plain_mean = float(np.mean(plain_bpb))
    floor = frag["plain_std"]
    out = {"plain_gen_gap": float(np.mean(plain_gg)) if plain_gg else None,
           "variants_beating_plain": [], "gen_gap_by_variant": {}}
    for r in sd["noise"] + sd["global"]:
        if r.get("val_bpb") is None:
            continue
        out["gen_gap_by_variant"][r["name"]] = round(r["gen_gap"], 5) if r.get("gen_gap") is not None else None
        if r["val_bpb"] < plain_mean - floor:
            out["variants_beating_plain"].append(r["name"])
    for s in (0.1, 0.3, 1.0):
        gr = [x["val_bpb"] for x in sd["global"] if x["sigma"] == s and x.get("val_bpb") is not None]
        mid = [x["val_bpb"] for x in sd["noise"] if x["sigma"] == s and x["layer"] == sd["mid"] and x.get("val_bpb") is not None]
        if not gr or not mid:
            continue
        all_bpb, mid_bpb = float(np.mean(gr)), float(np.mean(mid))
        out[f"global_vs_mid_sig{int(s*100)}"] = {
            "delta_global": all_bpb - plain_mean, "delta_mid": mid_bpb - plain_mean,
            "ratio_global_mid": (all_bpb - plain_mean) / (mid_bpb - plain_mean) if abs(mid_bpb - plain_mean) > 1e-9 else None,
            "expected_ratio_if_pure_dose": float(sd["n_layer"]),
        }
    return out


def dynamics(sd: dict) -> dict:
    out = {}
    pt = [r["final_train_loss"] for r in sd["plain"] if r.get("final_train_loss")]
    pv = [r["final_val_loss"] for r in sd["plain"] if r.get("final_val_loss")]
    out["plain_final_train"] = float(np.mean(pt)) if pt else None
    out["plain_final_val"] = float(np.mean(pv)) if pv else None
    out["rows"] = [{"name": r["name"], "layer": r["layer"], "sigma": r["sigma"],
                    "final_train": r["final_train_loss"], "final_val": r.get("final_val_loss")}
                   for r in sd["noise"] + sd["global"] if r.get("final_train_loss") is not None]
    return out


def resid_profile(sd: dict) -> dict:
    rs = [r["resid_rms"] for r in sd["plain"] if "resid_rms" in r]
    if not rs:
        return {}
    m = np.mean(rs, axis=0)
    return {"resid_rms": [round(float(v), 5) for v in m],
            "ratio_last_first": float(m[-1] / m[0]) if m[0] > 0 else None}


def sensitivity_analysis(sd: dict, frag: dict) -> dict:
    """Test-time sensitivity profile (from sensitivity.json) + its shape and correlations.

    The sensitivity curve is per-layer delta val_bpb of the PLAIN model when one slice of
    the stream is noised at eval time. Correlate it against: ablation dloss (importance),
    lens marginal loss (readout progress), output-proj scale (bookkeeping confound), and the
    train-time fragility curve at the same sigma (adaptation vs static importance).
    """
    sens = sd.get("sensitivity")
    if not sens:
        return {}
    n = sens["n_layer"]
    out = {"n_layer": n, "plain_bpb": sens["plain"]["val_bpb"], "curves": {}}
    for L, sigs in sens["curve"].items():
        for k, v in sigs.items():
            out["curves"].setdefault(k, []).append({"layer": int(L[1:]), "sigma": float(k[3:]) / 100, **v})
    for k in out["curves"]:
        out["curves"][k].sort(key=lambda v: v["layer"])

    plain_abl = [r["dloss"] for r in sd["plain"] if "dloss" in r]
    dloss = np.mean(plain_abl, axis=0) if plain_abl else None
    lens = [r["lens_marginal"] for r in sd["plain"] if "lens_marginal" in r]
    lm = np.mean(lens, axis=0) if lens else None
    scales = [r["scale"] for r in sd["runs"] if "scale" in r]
    scale = np.mean(scales, axis=0) if scales else None

    st = {}
    for k, vs in sorted(out["curves"].items()):
        y = np.array([v["delta_bpb"] for v in vs], float)
        layers = np.array([v["layer"] for v in vs], float)
        c = (n - 1) / 2.0
        lin = 1 - np.abs(layers - c) / c
        g = np.exp(-((layers - c) / max(c / 2, 1e-6)) ** 2)
        row = {"sigma": vs[0]["sigma"], "curve": [round(float(v), 6) for v in y],
               "peak_layer": int(vs[int(np.argmax(y))]["layer"]),
               "min_layer": int(vs[int(np.argmin(y))]["layer"]),
               "spearman_linear_bump": spearman(y, lin), "p_perm_linear": perm_p(y, lin),
               "spearman_gauss_bump": spearman(y, g), "p_perm_gauss": perm_p(y, g),
               "spearman_depth": spearman(y, layers), "p_perm_depth": perm_p(y, layers)}
        if dloss is not None and len(y) == len(dloss):
            row["rho_vs_ablation"] = spearman(y, dloss)
            row["p_vs_ablation"] = perm_p(y, dloss)
        if lm is not None and len(y) == len(lm):
            row["rho_vs_lens_marginal"] = spearman(y, lm)
            row["p_vs_lens_marginal"] = perm_p(y, lm)
        if scale is not None and len(y) == len(scale):
            row["rho_vs_scale"] = spearman(y, scale)
        tf = frag.get("curve", {})
        x = [tf.get(f"L{v['layer']}_s{int(v['sigma']*100)}", {}).get("delta") for v in vs]
        if all(v is not None for v in x):
            row["rho_vs_train_fragility"] = spearman(y, np.array(x, float))
            row["p_vs_train_fragility"] = perm_p(y, np.array(x, float))
        st[k] = row
    out["per_sigma"] = st
    return out


# ----------------------------------------------------------------------------- main
def main():
    sens_file = HERE / "sensitivity.json"
    sens_all = json.loads(sens_file.read_text()) if sens_file.exists() else {}
    sizes, raw = {}, {}
    for size in ("depth_m", "depth_d"):
        sd = collect(size)
        if not sd["runs"]:
            continue
        frag = fragility(sd)
        # attach test-time sensitivity for the plain runs of this size (averaged over plain seeds)
        matching = [v for k, v in sens_all.items()
                    if v.get("n_layer") == sd["n_layer"] and (f"depth_{size[-1]}_core" in k or f"/{size}" in k)]
        if matching:
            sens = matching[0]
            if len(matching) > 1:
                import statistics
                avg = {"n_layer": sens["n_layer"], "n_seeds": len(matching),
                       "plain": {"val_bpb": float(np.mean([m["plain"]["val_bpb"] for m in matching]))},
                       "curve": {}}
                for L, sigs in sens["curve"].items():
                    for k, v in sigs.items():
                        vals = [m["curve"][L][k]["delta_bpb"] for m in matching if L in m["curve"] and k in m["curve"][L]]
                        avg["curve"].setdefault(L, {})[k] = {
                            "delta_bpb": float(np.mean(vals)),
                            "delta_bpb_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "delta_loss": float(np.mean([m["curve"][L][k].get("delta_loss", 0.0) for m in matching])),
                        }
                sens = avg
            sd["sensitivity"] = sens
        abla = ablation_corr(sd, frag)
        sizes[size] = {
            "label": sd["label"], "n_layer": sd["n_layer"], "anchors": [sd["first"], sd["mid"], sd["last"]],
            "n_runs": len(sd["runs"]), "n_plain": len(sd["plain"]), "n_noise": len(sd["noise"]),
            "n_global": len(sd["global"]),
            "noise_floor": {"n": len(sd["plain"]), "val_bpb_mean": frag["plain_mean"],
                             "val_bpb_std": frag["plain_std"]},
            "fragility": frag,
            "anchor_test": anchor_test(sd, frag),
            "shape_test": shape_test(sd, frag),
            "ablation_corr": abla,
            "sensitivity": sensitivity_analysis(sd, frag) or None,
            "regulariser": regulariser_checks(sd, frag),
            "dynamics": dynamics(sd),
            "resid_profile": resid_profile(sd),
        }
        raw[size] = sd
        print(f"{size}: {len(sd['runs'])} runs, plain bpb={frag['plain_mean']:.4f} +/- {frag['plain_std']:.4f}")

    def _ser(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        return o

    (HERE / "results.json").write_text(json.dumps({"sizes": sizes}, indent=2, default=_ser))
    write_markdown(sizes)
    write_plots(sizes, raw)
    print(f"wrote {HERE/'results.json'} and {HERE/'results.md'}")


def write_markdown(sizes: dict):
    md = ["# Depth-fragility study - results", "",
          "Hypothesis: RMS-scaled Gaussian noise injected after attention during training hurts MORE in the",
          "middle layers than at the ends (middle layers carry the least redundant information).",
          "Question: does the model's information density vary by depth?", "",
          f"runs base: {RUNS_BASE}", ""]
    for size, s in sizes.items():
        nf = s["noise_floor"]
        md.append(f"## {s['label']}  ({s['n_layer']} layers, {s['n_runs']} runs)")
        md.append("")
        md.append(f"plain val_bpb = {nf['val_bpb_mean']:.4f} +/- {nf['val_bpb_std']:.4f} (n={nf['n']}) <- seed noise floor")
        md.append("")
        md.append("| variant | layer | sigma | bpb mean | bpb std | Delta vs plain | Delta/floor | gen_gap |")
        md.append("|---|---|---|---|---|---|---|---|")
        for key in sorted(s["fragility"]["curve"], key=lambda k: (s["fragility"]["curve"][k]["sigma"], str(s["fragility"]["curve"][k]["layer"]))):
            c = s["fragility"]["curve"][key]
            md.append(f"| {key} | {c['layer']} | {c['sigma']} | {c['bpb_mean']:.4f} | {c['bpb_std']:.4f} | "
                      f"{c['delta']:+.4f} | {c['delta_over_floor']:+.2f} | {c.get('gen_gap_mean')} |")
        md.append("")
        at = s["anchor_test"]
        if at:
            md.append("### Anchor test (paired over seeds; d_first-mid > 0 means middle more fragile)")
            md.append("| sigma | d(first)-d(mid) | d(last)-d(mid) | signs mid<first | signs mid<last | in-floor f-m | in-floor l-m |")
            md.append("|---|---|---|---|---|---|---|")
            for k, r in at.items():
                md.append(f"| {k} | {r['d_first_minus_mid_mean']:+.4f} | {r['d_last_minus_mid_mean']:+.4f} | "
                          f"{r['signs_mid_worse_than_first']} | {r['signs_mid_worse_than_last']} | "
                          f"{r['in_floor_units_first_minus_mid']:+.2f} | {r['in_floor_units_last_minus_mid']:+.2f} |")
            md.append("")
        st = s["shape_test"]
        if st:
            md.append("### Curve shape (full-depth fragility curve vs centeredness priors)")
            md.append("| curve | rho(linear bump) | p_perm | rho(gauss bump) | p_perm | rho(depth) | p_perm |")
            md.append("|---|---|---|---|---|---|---|")
            for k, r in st.items():
                md.append(f"| {k} | {r['spearman_linear_bump']:+.3f} | {r['p_perm_linear']:.3f} | "
                          f"{r['spearman_gauss_bump']:+.3f} | {r['p_perm_gauss']:.3f} | "
                          f"{r['spearman_depth']:+.3f} | {r['p_perm_depth']:.3f} |")
            md.append("")
        ac = s["ablation_corr"]
        if ac:
            md.append("### Does plain-model layer importance predict fragility?")
            md.append("| curve | rho(frag, ablate) | p_perm | rho(frag, lens) | rho(frag, scale) | rho(ablate, scale) |")
            md.append("|---|---|---|---|---|---|")
            for k, r in ac.items():
                md.append(f"| {k} | {r['rho_fragility_ablation']:+.3f} | {r['p_perm_fragility_ablation']:.3f} | "
                          f"{r.get('rho_fragility_lens', float('nan')):+.3f} | {r.get('rho_fragility_scale', float('nan')):+.3f} | "
                          f"{r.get('rho_ablation_scale', float('nan')):+.3f} |")
            md.append("")
        reg = s["regulariser"]
        md.append("### Regulariser checks")
        md.append(f"- plain gen_gap = {reg['plain_gen_gap']:+.4f}")
        md.append(f"- variants beating plain by >1 floor: {reg['variants_beating_plain'] or 'none'}")
        for k, v in reg.items():
            if k.startswith("global_vs_mid"):
                ratio = f"{v['ratio_global_mid']:.2f}" if v["ratio_global_mid"] is not None else "n/a"
                md.append(f"- {k}: delta_global={v['delta_global']:+.4f}, delta_mid={v['delta_mid']:+.4f}, "
                          f"ratio={ratio} (pure-dose expectation ~{v['expected_ratio_if_pure_dose']})")
        dyn = s["dynamics"]
        if dyn.get("plain_final_train") is not None:
            md.append(f"- plain final train loss = {dyn['plain_final_train']:.4f}, final val = {dyn['plain_final_val']:.4f}")
        rp = s["resid_profile"]
        if rp:
            md.append(f"- stream RMS across depth: {rp['resid_rms']} (last/first = {rp['ratio_last_first']:.1f}x)")
        md.append("")
        sens = s.get("sensitivity")
        if sens:
            md.append("### Test-time sensitivity profile (plain model, eval-time noise, no retraining)")
            md.append("| sigma | rho(linear bump) | p | rho(gauss bump) | p | peak layer | min layer | rho(vs ablation) | p | rho(vs lens) | rho(vs train fragility) |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for k, r in sorted(sens["per_sigma"].items()):
                def g(rr, kk):
                    v = rr.get(kk)
                    return "—" if v is None else f"{v:+.3f}"
                md.append(f"| {k} | {g(r, 'spearman_linear_bump')} | {r['p_perm_linear']:.3f} | {g(r, 'spearman_gauss_bump')} | "
                          f"{r['p_perm_gauss']:.3f} | {r['peak_layer']} | {r['min_layer']} | "
                          f"{g(r, 'rho_vs_ablation')} | {r.get('p_vs_ablation', float('nan')):.3f} | "
                          f"{g(r, 'rho_vs_lens_marginal')} | {g(r, 'rho_vs_train_fragility')} |")
            md.append("")
            md.append("curve (delta val_bpb by layer) at sigma=1.0: "
                      + " ".join(f"L{i}={v:+.4f}" for i, v in enumerate(sens["per_sigma"]["sig100"]["curve"])))
            md.append("")
    (HERE / "results.md").write_text("\n".join(md) + "\n")


def write_plots(sizes: dict, raw: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plots skipped: {e!r})")
        return
    pdir = HERE / "plots"
    pdir.mkdir(exist_ok=True)
    s_colors = {1.0: "#d9534f", 0.3: "#f0ad4e", 0.1: "#5bc0de"}

    for size, s in sizes.items():
        sd = raw[size]
        n = s["n_layer"]
        frag = s["fragility"]
        floor = s["noise_floor"]["val_bpb_std"]
        # ---- 1. fragility curve
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        for s_val in (1.0, 0.3, 0.1):
            xs, ys, errs, nseeds = [], [], [], []
            for L in range(n):
                c = frag["curve"].get(f"L{L}_s{int(s_val*100)}")
                if c is None:
                    continue
                xs.append(L); ys.append(c["delta"]); errs.append(c["bpb_std"]); nseeds.append(c["n_seeds"])
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=errs, marker="o", ms=8 if 3 in nseeds else 5, lw=1.4,
                        color=s_colors[s_val], capsize=3,
                        label=f"sigma={s_val} ({'3 seeds' if 3 in nseeds else '1 seed'})")
        ax.axhline(0, color="k", lw=0.7)
        if floor > 0:
            ax.axhspan(-floor, floor, color="green", alpha=0.12)
            ax.text(n - 0.4, 1.25 * floor, "seed noise floor", ha="right", fontsize=8, color="green")
        ax.set_xlabel("layer (after_attn site)")
        ax.set_ylabel("Delta val_bpb vs plain")
        ax.set_title(f"{s['label']}: where is the residual stream fragile?")
        ax.set_xticks(range(n))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(pdir / f"fragility_{size}.png", dpi=130)
        plt.close(fig)
        # ---- 2. dose response at anchors
        first, mid, last = s["anchors"]
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        sigs = sorted({c["sigma"] for c in frag["curve"].values() if isinstance(c["layer"], int)})
        for L, lbl, mk in ((first, "first", "s"), (mid, "middle", "^"), (last, "last", "D")):
            xs = [sg for sg in sigs if frag["curve"].get(f"L{L}_s{int(sg*100)}")]
            ys = [frag["curve"][f"L{L}_s{int(sg*100)}"]["delta"] for sg in xs]
            if xs:
                ax.plot(xs, ys, marker=mk, ms=9, label=lbl, lw=1.5)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1.0])
        ax.set_xticklabels(["0.1", "0.3", "1.0"])
        ax.set_xlabel("sigma (noise std / stream RMS)")
        ax.set_ylabel("Delta val_bpb vs plain")
        ax.set_title(f"{s['label']}: dose response at anchor depths")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdir / f"dose_{size}.png", dpi=130)
        plt.close(fig)
        # ---- 3. fragility vs ablation (both from the plain model)
        plain_abl = [r["dloss"] for r in sd["plain"] if "dloss" in r]
        d100 = [frag["curve"].get(f"L{L}_s100", {}).get("delta") for L in range(n)]
        if plain_abl and all(v is not None for v in d100):
            dloss = np.mean(plain_abl, axis=0)
            d100 = np.array(d100, float)
            fig, ax = plt.subplots(figsize=(8.4, 4.6))
            Ls = np.arange(n)
            ax.bar(Ls - 0.2, dloss, width=0.4, alpha=0.65, label="ablate dloss (plain model)")
            ax2 = ax.twinx()
            ax2.plot(Ls, d100, "o-", color="#d9534f", lw=1.6, label="fragility at sigma=1.0")
            ax2.axhline(0, color="#d9534f", lw=0.7, alpha=0.5)
            ax.set_xlabel("layer")
            ax.set_ylabel("ablation dloss (nats)")
            ax2.set_ylabel("Delta val_bpb (sigma=1.0)")
            ax.set_title(f"{s['label']}: fragility vs plain-model layer importance")
            ax.set_xticks(range(n))
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=9)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(pdir / f"ablation_{size}.png", dpi=130)
            plt.close(fig)
        # ---- 4. trajectories: plain vs anchor noise at sigma=1.0
        trajs = {r["name"]: (r["traj"], r["val_traj"]) for r in sd["plain"] + sd["noise"]
                 if r.get("traj") and (r["kind"] == "plain" or (r["sigma"] == 1.0 and r["layer"] in s["anchors"]))}
        if trajs:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for name, (tr, va) in sorted(trajs.items()):
                steps = [p[0] for p in tr]
                axes[0].plot(steps, [p[1] for p in tr], lw=1.3, label=name)
                if va:
                    axes[1].plot([p[0] for p in va], [p[1] for p in va], lw=1.3, label=name)
            for ax_, t in ((axes[0], "train loss"), (axes[1], "val loss (un-noised eval)")):
                ax_.set_xlabel("step"); ax_.set_ylabel(t)
                ax_.set_xscale("log"); ax_.grid(alpha=0.3, which="both")
                ax_.legend(fontsize=8)
            axes[0].set_title(f"{s['label']}: train loss (sigma=1.0 noise at each anchor)")
            fig.tight_layout()
            fig.savefig(pdir / f"trajectories_{size}.png", dpi=130)
            plt.close(fig)
        # ---- 5. gen_gap bars
        gg = s["regulariser"]["gen_gap_by_variant"]
        if gg:
            names = list(gg.keys()); vals = [gg[k] for k in names]
            fig, ax = plt.subplots(figsize=(9.5, 4.0))
            ax.bar(range(len(names)), vals, color=["#5bc0de" if "plain" in nm else "#f0ad4e" for nm in names])
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
            ax.axhline(s["regulariser"]["plain_gen_gap"], color="k", lw=0.8, ls="--")
            ax.set_ylabel("gen_gap = val_loss - train_loss (nats)")
            ax.set_title(f"{s['label']}: gen_gap by variant (is noise a regulariser?)")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(pdir / f"gen_gap_{size}.png", dpi=130)
            plt.close(fig)
        # ---- 6. lens profile + resid rms (plain model)
        lens_runs = [r for r in sd["plain"] if "lens" in r]
        if lens_runs:
            lens = np.mean([r["lens"] for r in lens_runs], axis=0)
            emb = lens_runs[0].get("lens_embed")
            rms_runs = [r for r in sd["plain"] if "resid_rms" in r]
            fig, ax = plt.subplots(figsize=(8.4, 4.4))
            Ls = np.arange(n)
            ax.plot(Ls, lens, "o-", color="#5b6ee1", lw=1.6, label="lens loss (logit lens)")
            if rms_runs:
                rms = np.mean([r["resid_rms"] for r in rms_runs], axis=0)
                ax2 = ax.twinx()
                ax2.plot(Ls, rms, "s--", color="#888", lw=1.3, label="stream RMS (log)")
                ax2.set_yscale("log")
                ax2.set_ylabel("stream RMS (log)")
            ax.set_xlabel("layer (output of)")
            ax.set_ylabel("lens loss (nats)")
            ax.set_title(f"{s['label']}: what each layer adds to the stream (plain model)")
            ax.set_xticks(range(n))
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(pdir / f"lens_{size}.png", dpi=130)
            plt.close(fig)
        # ---- 7. train-time fragility vs test-time sensitivity (sigma=1.0)
        sens = s.get("sensitivity")
        d100 = [frag["curve"].get(f"L{L}_s100", {}).get("delta") for L in range(n)]
        if sens and all(v is not None for v in d100):
            stc = sens["per_sigma"].get("sig100", {})
            ysens = stc.get("curve")
            if ysens and len(ysens) == n:
                fig, ax = plt.subplots(figsize=(8.6, 4.6))
                Ls = np.arange(n)
                ax.plot(Ls, d100, "o-", color="#d9534f", lw=1.8,
                        label="train-time: hurt from noise at L (val_bpb, model trained with noise, evaluated clean)")
                ax2 = ax.twinx()
                ax2.plot(Ls, ysens, "s-", color="#5b6ee1", lw=1.8,
                         label="test-time: plain model, val_bpb with eval-noise at L")
                ax2.axhline(0, color="#5b6ee1", lw=0.7, alpha=0.4)
                ax.axhline(0, color="#d9534f", lw=0.7, alpha=0.4)
                if floor > 0:
                    ax.axhspan(-floor, floor, color="green", alpha=0.12, label="seed noise floor")
                ax.set_xlabel("layer")
                ax.set_ylabel("train-time Delta val_bpb (red)")
                ax2.set_ylabel("test-time Delta val_bpb (blue)")
                ax.set_title(f"{s['label']}: does the same depth matter in both senses?")
                ax.set_xticks(range(n))
                ax.grid(alpha=0.3)
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
                fig.tight_layout()
                fig.savefig(pdir / f"sensitivity_{size}.png", dpi=130, bbox_inches="tight")
                plt.close(fig)
    print(f"plots -> {pdir}")


if __name__ == "__main__":
    main()
