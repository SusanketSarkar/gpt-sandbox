"""Generate layout-schedule sweep files for every depth.

For a model of N attention + N MLP sublayers (2N sublayers total, n_layer=2N):
  Family 1 (ordering only, same multiset N A + N M, params & FLOPs identical by construction):
    baseline  AM AM AM ...
    A         MLP-heavy ends, attention-heavy middle
    B         attention-heavy ends, MLP-heavy middle
    C         MLP-only ends, attention-only middle
    D         attention-only ends, MLP-only middle
    sandwich  attention at one end, alternating middle, MLP at other end
    rand0-3   random permutations of the same multiset (null distribution)
  Family 2 (fewer attention sublayers, params matched by adjusting d_ff, total sublayers constant):
    20A+28M, 16A+32M, 12A+36M   (jumbo: 24A+24M -> reduce A, add M, tune d_ff)
"""
from __future__ import annotations
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
SWEEP = ROOT / "configs" / "sweeps"
SWEEP.mkdir(parents=True, exist_ok=True)


def layout_baseline(N):
    return "".join("AM" for _ in range(N))


def _region_counts(total, nA_target):
    """Split total slots into early/mid/late (25/50/25) and assign A/M to hit nA_target total."""
    early = total // 4
    late = total // 4
    mid = total - early - late
    return early, mid, late


def layout_A(N):
    """MLP-heavy ends, attention-heavy middle. Ends 25% mostly M, middle 50% mostly A."""
    L = 2 * N
    early, mid, late = L // 4, L // 2, L // 4
    # middle: 75% A; ends: 25% A
    a_mid = int(round(mid * 0.75)); a_early = int(round(early * 0.25)); a_late = N - a_mid - a_early
    a_late = max(0, min(late, a_late))
    a_early = max(0, min(early, N - a_mid - a_late))
    s = "M" * (early - a_early) + "A" * a_early
    s += "A" * a_mid + "M" * (mid - a_mid)
    s += "A" * a_late + "M" * (late - a_late)
    return _fix(s, N)


def layout_B(N):
    """attention-heavy ends, MLP-heavy middle."""
    L = 2 * N
    early, mid, late = L // 4, L // 2, L // 4
    a_early = int(round(early * 0.75)); a_mid = int(round(mid * 0.25)); a_late = N - a_early - a_mid
    a_late = max(0, min(late, a_late)); a_early = max(0, min(early, N - a_mid - a_late))
    s = "A" * a_early + "M" * (early - a_early)
    s += "A" * a_mid + "M" * (mid - a_mid)
    s += "A" * a_late + "M" * (late - a_late)
    return _fix(s, N)


def layout_C(N):
    """MLP-only ends, attention-only middle."""
    L = 2 * N
    early, mid, late = L // 4, L // 2, L // 4
    # middle all A (mid A's), ends all M. Need mid >= N for this to work.
    s = "M" * early + "A" * mid + "M" * late
    # adjust: we need exactly N A. middle has `mid` A. If mid>N, trim A from middle, fill with M.
    return _fix(s, N)


def layout_D(N):
    """attention-only ends, MLP-only middle."""
    L = 2 * N
    early, mid, late = L // 4, L // 2, L // 4
    s = "A" * early + "M" * mid + "A" * late
    return _fix(s, N)


def layout_sandwich(N):
    """attention block at start, alternating middle, MLP block at end."""
    L = 2 * N
    early, mid, late = L // 4, L // 2, L // 4
    s = "A" * early + "".join("AM" for _ in range(mid // 2)) + "M" * late
    return _fix(s, N)


def _fix(s, N):
    """Force exactly N A's (and N M's) by swapping from the tail, keeping prefix structure as much as possible."""
    s = list(s)
    curA = s.count("A")
    if curA > N:
        # remove excess A from the end
        excess = curA - N
        for i in range(len(s) - 1, -1, -1):
            if s[i] == "A" and excess:
                s[i] = "M"; excess -= 1
    elif curA < N:
        need = N - curA
        for i in range(len(s)):
            if s[i] == "M" and need:
                s[i] = "A"; need -= 1
    assert s.count("A") == N, f"could not fix layout to {N} A: {s}"
    assert len(s) == 2 * N
    return "".join(s)


def random_layout(N, seed):
    s = ["A"] * N + ["M"] * N
    random.Random(seed).shuffle(s)
    return "".join(s)


def family1(N):
    return {
        "baseline": layout_baseline(N),
        "A": layout_A(N),
        "B": layout_B(N),
        "C": layout_C(N),
        "D": layout_D(N),
        "sandwich": layout_sandwich(N),
        "rand0": random_layout(N, 101),
        "rand1": random_layout(N, 202),
        "rand2": random_layout(N, 303),
        "rand3": random_layout(N, 404),
    }


def family2(N):
    """Fewer attention sublayers, more MLP, same total sublayers (2N). Returns {name: (nA, layout)}.
    d_ff tuned later by the caller to match params."""
    out = {"baseline_24A24M": (N, layout_baseline(N))}
    for nA, tag in ((int(N * 5 / 6), "f2a"), (int(N * 2 / 3), "f2b"), (int(N * 1 / 2), "f2c")):
        nA = (nA // 2) * 2  # even
        nM = 2 * N - nA
        # attention-heavy ends, MLP-heavy middle (the schedule we'd keep if trimming attention)
        L = 2 * N
        early, mid, late = L // 4, L // 2, L // 4
        a_early = int(round(early * 0.75)); a_mid = int(round(mid * 0.25)); a_late = nA - a_early - a_mid
        a_late = max(0, min(late, a_late)); a_early = max(0, min(early, nA - a_mid - a_late))
        s = "A" * a_early + "M" * (early - a_early) + "A" * a_mid + "M" * (mid - a_mid) + "A" * a_late + "M" * (late - a_late)
        s = _fix_nA(s, nA, L)
        out[f"{tag}_{nA}A{nM}M"] = (nA, s)
    return out


def _fix_nA(s, nA, L):
    s = list(s)
    cur = s.count("A")
    if cur > nA:
        ex = cur - nA
        for i in range(len(s) - 1, -1, -1):
            if s[i] == "A" and ex:
                s[i] = "M"; ex -= 1
    else:
        need = nA - cur
        for i in range(len(s)):
            if s[i] == "M" and need:
                s[i] = "A"; need -= 1
    assert s.count("A") == nA and len(s) == L
    return "".join(s)


def _alternating(nA, nM):
    """Distribute nA A's and nM M's as evenly/alternating as possible."""
    total = nA + nM
    # place the minority as evenly spaced
    if nA <= nM:
        a, m = nA, nM
        out = []
        gap = total / a if a else 0
        positions = sorted({int(i * gap) for i in range(a)})
        posset = set(positions)
        for i in range(total):
            out.append("A" if i in posset else "M")
        return "".join(out)
    else:
        return "".join("M" if c == "A" else "A" for c in _alternating(nM, nA))


def _baseline_params(base_cfg_path, N, d):
    import copy, torch
    from gpt_sandbox.plugins import default_plugin_paths, load_plugins
    load_plugins(default_plugin_paths())
    from gpt_sandbox.config import load_config
    from gpt_sandbox.model.gpt import GPT
    c = copy.deepcopy(load_config(base_cfg_path).model)
    c.extra.layout = "AM" * N
    c.n_layer = 2 * N
    c.vocab_size = 8192
    torch.manual_seed(0)
    return GPT(c).num_params()


def tune_dff(base_cfg_path, nA, nM, d, target_params):
    """d_ff for (nA A + nM M) sublayers matching target_params. Params are linear in d_ff:
    P(d_ff) = C + nM*3*d*d_ff, where C = nA*attn + emb + norms. One build gives C."""
    import copy, torch
    from gpt_sandbox.plugins import default_plugin_paths, load_plugins
    load_plugins(default_plugin_paths())
    from gpt_sandbox.config import load_config
    from gpt_sandbox.model.gpt import GPT

    c = copy.deepcopy(load_config(base_cfg_path).model)
    c.extra.layout = "A" * nA + "M" * nM
    c.n_layer = nA + nM
    c.mlp.d_ff = 1024
    c.vocab_size = 8192
    torch.manual_seed(0)
    p1024 = GPT(c).num_params()
    C = p1024 - nM * 3 * d * 1024
    dff = (target_params - C) / (nM * 3 * d)
    return int(round(dff))


def emit_sweep(name, base, variants, seeds, common=None):
    import json
    lines = [f"name: {name}", f"base: {base}"]
    if common:
        lines.append("common: " + json.dumps(common).replace('"', "'"))
    lines.append(f"seeds: {seeds}")
    lines.append("variants:")
    for k, v in variants.items():
        if isinstance(v, str):
            lines.append(f'  {k}: {{model.extra.layout: "{v}"}}')
        else:
            lines.append(f"  {k}: {json.dumps(v).replace(chr(34), chr(39))}")
    (SWEEP / f"{name}.yaml").write_text("\n".join(lines) + "\n")
    print(f"wrote {SWEEP / (name + '.yaml')}  ({len(variants)} variants)")


def main():
    # jumbo: N=24 attention sublayers (n_layer=48, d=512)
    N = 24
    f1 = family1(N)
    print(f"\n=== Family 1 @ jumbo (N={N}, {2*N} sublayers) ===")
    for k, v in f1.items():
        print(f"  {k:10s} {v}  A={v.count('A')} M={v.count('M')}")
        assert v.count("A") == N and v.count("M") == N, f"{k} not matched: {v.count('A')}/{v.count('M')}"
    emit_sweep("layout_f1_jumbo", "../layouts/jumbo_base.yaml",
               {k: v for k, v in f1.items()}, [0, 1, 2])

    # Family 2: parameter-matched by tuning d_ff per variant (same total sublayers 2N).
    d = 512
    base_params = _baseline_params("configs/layouts/jumbo_base.yaml", N, d)
    print(f"\n=== Family 2 @ jumbo (N={N}, baseline params {base_params:,}) ===")
    f2_variants = {}
    for nA in (int(N * 5 / 6) // 2 * 2, int(N * 2 / 3) // 2 * 2, N // 2):
        nM = 2 * N - nA
        v = _alternating(nA, nM)
        dff = tune_dff("configs/layouts/jumbo_base.yaml", nA, nM, d, base_params)
        p_chk = base_params  # target
        f2_variants[f"f2_{nA}A{nM}M"] = {"model.extra.layout": v, "model.mlp": {"name": "swiglu", "d_ff": dff}}
        print(f"  f2_{nA}A{nM}M  d_ff={dff:5d}  layout={v}  A={v.count('A')} M={v.count('M')}")
    emit_sweep("layout_f2_jumbo", "../layouts/jumbo_base.yaml", f2_variants, [0, 1, 2])

    # depth_small: N=8 (n_layer=16, d=128)
    N = 8
    f1 = family1(N)
    print(f"\n=== Family 1 @ depth_small (N={N}, {2*N} sublayers) ===")
    for k, v in f1.items():
        assert v.count("A") == N and v.count("M") == N
    emit_sweep("layout_f1_small", "../layouts/depth_small_base.yaml", f1, [0, 1, 2])

    # depth_med: N=16 (n_layer=32, d=256)
    N = 16
    f1 = family1(N)
    print(f"\n=== Family 1 @ depth_med (N={N}, {2*N} sublayers) ===")
    for k, v in f1.items():
        assert v.count("A") == N and v.count("M") == N
    emit_sweep("layout_f1_med", "../layouts/depth_med_base.yaml", f1, [0, 1])


if __name__ == "__main__":
    main()
