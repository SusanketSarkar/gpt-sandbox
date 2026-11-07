"""Reference numbers and sanity checks on a dataset: n-gram baselines, train/val overlap."""
from __future__ import annotations

import math

import numpy as np

from ..utils import read_json, write_json

CAP = 50_000_000


def baselines(ds) -> dict:
    """val loss (nats/token) of: uniform, smoothed unigram, smoothed bigram fit on train. Cached on disk.
    Your model should beat all three; the gap to the bigram is a quick 'did it learn context' signal."""
    cache = ds.path / "baselines.json"
    if cache.exists():
        c = read_json(cache)
        if c.get("train_tokens") == len(ds.train) and c.get("val_tokens") == len(ds.val) and not ds.meta.get("retokenized"):
            return c
    V = ds.vocab_size
    tr = np.asarray(ds.train[:CAP]).astype(np.int64)
    va = np.asarray(ds.val).astype(np.int64)
    counts = np.bincount(tr, minlength=V).astype(np.float64)
    uni = (counts + 1) / (counts.sum() + V)
    out = {"train_tokens": len(ds.train), "val_tokens": len(ds.val), "uniform": math.log(V),
           "unigram": float(-np.log(uni[va]).mean()),
           "unigram_entropy_train": float(-(uni * np.log(uni)).sum()), "bigram": None}
    if V <= 4096 and len(va) > 1:
        pair = np.bincount(tr[:-1] * V + tr[1:], minlength=V * V).reshape(V, V).astype(np.float64)
        row = pair.sum(1, keepdims=True)
        p = (pair + uni[None, :]) / (row + 1.0)  # Dirichlet(1 * unigram) prior
        out["bigram"] = float(-np.log(p[va[:-1], va[1:]]).mean())
    if not ds.meta.get("retokenized"):
        write_json(cache, out)
    return out


def overlap(ds, window: int = 32, n_probe: int = 500, cap: int = 10_000_000, seed: int = 0) -> dict:
    """Fraction of random val windows (exact `window`-token matches) that also occur in train.
    Near 0 for natural text; high values mean duplicated / leaked data (expected for tiny synthetic tasks)."""
    tr = np.asarray(ds.train[:cap]).astype(np.uint64)
    va = np.asarray(ds.val).astype(np.uint64)
    if len(tr) <= window or len(va) <= window:
        return {"window": window, "frac": None, "probes": 0}
    P = np.uint64(1000003)

    def rolling(a):
        m = len(a) - window + 1
        h = np.zeros(m, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for j in range(window):
                h = h * P + a[j : j + m]
        return h

    ht = rolling(tr)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(va) - window, size=min(n_probe, len(va) - window))
    probes = np.zeros(len(starts), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(window):
            probes = probes * P + va[starts + j]
    return {"window": window, "frac": float(np.isin(probes, ht).mean()), "probes": int(len(starts))}
