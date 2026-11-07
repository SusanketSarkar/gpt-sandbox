"""`gpt data inspect`: what is actually in this dataset?"""
from __future__ import annotations

import math

import numpy as np

from ..utils import bar, human, table
from .stats import baselines, overlap


def _doc_lengths(arr, eos, cap=50_000_000):
    a = np.asarray(arr[:cap])
    idx = np.flatnonzero(a == eos)
    if len(idx) == 0:
        return np.array([len(a)])
    return np.diff(np.r_[-1, idx]) - 1


def _hist(vals, bins=10, width=34):
    vals = vals[vals > 0]
    if len(vals) == 0:
        return ["  (no data)"]
    edges = np.unique(np.geomspace(max(1, vals.min()), vals.max() + 1, bins + 1).astype(int))
    h, edges = np.histogram(vals, bins=edges)
    top = max(1, h.max())
    return [f"  {edges[i]:>8,} - {edges[i + 1] - 1:<8,} {bar(h[i] / top, width):<{width}} {h[i]:,}" for i in range(len(h))]


def _show(s: str, n: int) -> str:
    s = s[:n].replace("\n", "↵\n")
    return "\n".join("    | " + ln for ln in s.split("\n"))


def inspect_dataset(ds, samples: int = 2, window: int = 300, show_tokens: bool = False, seed: int = 0) -> None:
    m, tok, V = ds.meta, ds.tokenizer, ds.vocab_size
    print(f"\n== dataset '{ds.name}'  ({ds.path}) ==")
    rows = [[s, f"{m['splits'][s]['docs']:,}", f"{m['splits'][s]['tokens']:,}", f"{m['splits'][s]['chars']:,}",
             f"{m['splits'][s]['bytes'] / max(1, m['splits'][s]['tokens']):.2f}"] for s in ("train", "val")]
    print(table(rows, ["split", "docs", "tokens", "chars", "bytes/token"], "  "))
    print(f"  tokenizer: {m['tokenizer']['spec']}  vocab={V}  eos_id={ds.eos_id}  dtype={m['dtype']}")
    if m.get("task_sep"):
        print(f"  synthetic task: answer follows {m['task_sep']!r}  (enables the `task` exact-match metric)")
    rep = m.get("report")
    if rep:
        print(f"  curation: sources={rep['sources']} dropped(filters)={rep['filters_dropped']} dedup={rep['dedup_dropped']}")

    tr = np.asarray(ds.train[:50_000_000])
    counts = np.bincount(tr, minlength=V)
    p = counts / counts.sum()
    used = int((counts > 0).sum())
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    print(f"\n-- vocabulary (first {len(tr):,} train tokens) --")
    print(f"  tokens used: {used}/{V} ({100 * used / V:.1f}%)   unigram entropy: {ent:.3f} nats ({ent / math.log(2):.2f} bits)")
    top = np.argsort(-counts)[:12]
    print("  most frequent: " + "  ".join(f"{tok.token_str(int(i))}:{100 * p[i]:.1f}%" for i in top))
    if V <= 20000:
        rare = int((counts == 1).sum())
        print(f"  tokens seen exactly once: {rare}")

    print("\n-- document length (tokens, train) --")
    L = _doc_lengths(ds.train, ds.eos_id)
    if len(L):
        print(f"  n={len(L):,}  min={L.min():,}  p10={int(np.percentile(L, 10)):,}  median={int(np.median(L)):,}  "
              f"p90={int(np.percentile(L, 90)):,}  max={L.max():,}  mean={L.mean():,.0f}")
        print("\n".join(_hist(L)))

    print("\n-- reference losses on val (nats/token) --")
    b = baselines(ds)
    print(f"  uniform={b['uniform']:.3f}   unigram={b['unigram']:.3f}   bigram={b['bigram'] if b['bigram'] is None else round(b['bigram'], 3)}")
    print("  (a trained model should be well below the bigram number)")

    ov = overlap(ds)
    if ov["frac"] is not None:
        flag = "  <-- high: duplicated/leaky data (normal for tiny synthetic tasks)" if ov["frac"] > 0.2 else ""
        print(f"\n-- train/val overlap: {100 * ov['frac']:.1f}% of {ov['probes']} random val {ov['window']}-token windows appear in train{flag}")

    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        arr = ds.split_array(split)
        for _ in range(samples):
            i = int(rng.integers(0, max(1, len(arr) - window)))
            ids = np.asarray(arr[i : i + window]).tolist()
            print(f"\n-- sample ({split}, tokens {i:,}..{i + window:,}) --")
            print(_show(tok.decode(ids), 4000))
            if show_tokens:
                print("    tokens: " + " ".join(f"{t}:{tok.token_str(t)}" for t in ids[:60]) + " ...")
    print()
