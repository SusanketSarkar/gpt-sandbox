"""EvalContext + the loop that runs registered metrics and prints/saves their numbers."""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..registry import get, names
from ..utils import fmt, now_iso, table, write_json
from . import metrics as _builtin_metrics  # noqa: F401  (registers builtin metrics)


@dataclass
class ValStats:
    nll: np.ndarray  # (N, T) per-token negative log-likelihood (nats)
    top1: np.ndarray  # (N, T) bool
    top5: np.ndarray
    entropy: np.ndarray  # predictive entropy (nats)
    conf: np.ndarray  # max prob


class EvalContext:
    """What a metric function receives. Handy attributes/methods:
    ev.model, ev.ds, ev.cfg, ev.device, ev.amp(), ev.seq_len, ev.tokenizer
    ev.val_batches(max_batches) / ev.train_batches(n)   -> (x, y) tensors on device
    ev.val_stats  (cached per-token nll/top1/top5/entropy over val)
    """

    def __init__(self, model, ds, cfg, device, amp, batch_size=16, max_batches=None, seed=0, quiet=False):
        self.model, self.ds, self.cfg, self.device, self.amp = model, ds, cfg, device, amp
        self.batch_size, self.max_batches, self.seed, self.quiet = batch_size, max_batches, seed, quiet
        self.seq_len = cfg.model.seq_len
        self.tokenizer = ds.tokenizer
        self.notes: dict = {}  # free-form extras (e.g. text samples) saved into eval.json

    def val_batches(self, max_batches=None):
        mb = max_batches if max_batches is not None else self.max_batches
        for x, y in self.ds.chunks("val", self.seq_len, self.batch_size, mb):
            yield x.to(self.device), y.to(self.device)

    def train_batches(self, n_batches: int):
        rng = np.random.default_rng(self.seed + 12345)
        for _ in range(n_batches):
            x, y = self.ds.get_batch("train", self.batch_size, self.seq_len, rng)
            yield x.to(self.device), y.to(self.device)

    @functools.cached_property
    def val_stats(self) -> ValStats:
        return self.stats_over(self.val_batches())

    @torch.no_grad()
    def stats_over(self, batches) -> ValStats:
        self.model.eval()
        parts = {k: [] for k in ("nll", "top1", "top5", "entropy", "conf")}
        for x, y in batches:
            with self.amp():
                logits = self.model(x).logits
            lp = torch.log_softmax(logits.float(), -1)
            nll = -lp.gather(-1, y[..., None])[..., 0]
            p = lp.exp()
            top5 = lp.topk(min(5, lp.size(-1)), -1).indices
            parts["nll"].append(nll.cpu())
            parts["top1"].append((top5[..., 0] == y).cpu())
            parts["top5"].append((top5 == y[..., None]).any(-1).cpu())
            parts["entropy"].append(-(p * lp).sum(-1).cpu())
            parts["conf"].append(p.max(-1).values.cpu())
        return ValStats(**{k: torch.cat(v).numpy() for k, v in parts.items()})


def select_metrics(cfg, requested=None, deep: bool = False) -> list[str]:
    ms = list(requested) if requested else list(cfg.eval.metrics)
    if deep:
        ms += [m for m in cfg.eval.deep_metrics if m not in ms]
    return ms


def run_metrics(ev: EvalContext, metric_names: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m in metric_names:
        fn = get("metric", m)
        t0 = time.time()
        try:
            res = fn(ev)
        except Exception as e:
            if not ev.quiet:
                print(f"  [metric {m}] FAILED: {type(e).__name__}: {e}")
            res = {f"{m}_error": float("nan")}
            ev.model.train(False)
        if res:
            out[m] = {k: float(v) for k, v in res.items()}
        if not ev.quiet:
            print(f"  [metric {m}] {time.time() - t0:.1f}s")
    return out


HEADLINE = ["val_loss", "val_ppl", "val_bpb", "val_acc1", "val_acc5"]


def flatten(groups: dict[str, dict[str, float]]) -> dict[str, float]:
    flat = {}
    for g in groups.values():
        flat.update(g)
    return flat


def print_report(groups, title: str = "eval") -> None:
    print(f"\n=== {title} ===")
    for g, vals in groups.items():
        print(f"\n[{g}]")
        print(table([[k, fmt(v)] for k, v in vals.items()], indent="  "))
    print()


def evaluate(model, ds, cfg, device, amp, metrics=None, deep=False, batch_size=None, max_batches=None,
             seed=0, quiet=False):
    ev = EvalContext(model, ds, cfg, device, amp, batch_size or cfg.eval.batch_size,
                     max_batches if max_batches is not None else cfg.eval.max_batches, seed, quiet)
    model.eval()
    groups = run_metrics(ev, select_metrics(cfg, metrics, deep))
    return groups, ev.notes


def save_eval(run_dir, groups, step, data_name, ckpt, notes=None):
    write_json(run_dir / "eval.json", {"time": now_iso(), "step": step, "data": data_name, "ckpt": ckpt,
                                       "metrics": flatten(groups), "groups": groups, "notes": notes or {}})
