"""Builtin eval metrics. Each is fn(ev) -> {name: float}; `ev` is an EvalContext (see runner.py).

Add yours with `gpt new metric my_metric`, or:

    @register("metric", "my_metric")
    def my_metric(ev):
        ...
        return {"my_number": 1.23}

then add it to `eval.metrics` in the config (or pass `gpt eval RUN --metrics my_metric`).
Set `my_metric.deep = True` to run it only with `gpt eval --deep`.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from ..data.stats import baselines
from ..registry import register
from ..utils import sync
from .generate import generate

LN2 = math.log(2)


def deep(fn):
    fn.deep = True
    return fn


# ---- model facts -----------------------------------------------------------------------------------
@register("metric", "params")
def params(ev):
    m = ev.model
    return {"params_total": m.num_params(), "params_nonemb": m.num_params(non_embedding=True),
            "flops_per_token": m.flops_per_token()}


# ---- core quality numbers ----------------------------------------------------------------------------
@register("metric", "val")
def val(ev):
    s = ev.val_stats
    loss = float(s.nll.mean())
    return {
        "val_loss": loss,  # nats / token (tokenizer-dependent!)
        "val_ppl": math.exp(min(loss, 50)),
        "val_bpb": loss * ev.ds.tokens_per_byte / LN2,  # bits / byte (compare across tokenizers)
        "val_acc1": float(s.top1.mean()),
        "val_acc5": float(s.top5.mean()),
        "val_pred_entropy": float(s.entropy.mean()),  # ~ val_loss when calibrated
        "val_tokens": float(s.nll.size),
    }


@register("metric", "position")
def position(ev):
    """Loss by position in the context window. ctx_gain = early loss - late loss: how much the model
    profits from longer context (bigger = better use of context)."""
    nll = ev.val_stats.nll  # (N, T)
    T = nll.shape[1]
    edges = [0] + [2**k for k in range(0, 20) if 2**k < T] + [T]
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[f"pos_loss/{lo}" if hi - lo == 1 else f"pos_loss/{lo}-{hi - 1}"] = float(nll[:, lo:hi].mean())
    early, late = max(1, T // 16), max(1, T // 4)
    out["ctx_gain"] = float(nll[:, :early].mean() - nll[:, -late:].mean())
    out["late_loss"] = float(nll[:, -late:].mean())
    return out


@register("metric", "calibration")
def calibration(ev, n_bins: int = 15):
    s = ev.val_stats
    conf, ok = s.conf.ravel(), s.top1.ravel().astype(np.float64)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return {"ece": float(ece), "mean_confidence": float(conf.mean()), "overconfidence": float(conf.mean() - ok.mean())}


@register("metric", "generalization")
def generalization(ev):
    n = max(1, min(10, math.ceil(len(ev.val_stats.nll) / ev.batch_size)))
    tr = ev.stats_over(ev.train_batches(n))
    tl, vl = float(tr.nll.mean()), float(ev.val_stats.nll.mean())
    return {"train_loss_eval": tl, "gen_gap": vl - tl}


@register("metric", "baselines")
def baselines_metric(ev):
    b = baselines(ev.ds)
    vl = float(ev.val_stats.nll.mean())
    out = {"base_uniform": b["uniform"], "base_unigram": b["unigram"], "gain_vs_unigram": b["unigram"] - vl}
    if b.get("bigram") is not None:
        out.update(base_bigram=b["bigram"], gain_vs_bigram=b["bigram"] - vl)
    return out


# ---- behaviour probes ---------------------------------------------------------------------------------
@register("metric", "induction")
def induction(ev, n_seq: int = 64, half: int = 64):
    """Repeated random tokens [r r]: the 1st copy is unpredictable (loss >= ln V, usually higher since random
    tokens are off-distribution), a model with induction/copy ability predicts the 2nd copy far better.
    induction_gap = loss(1st) - loss(2nd): ~0 means no copying; large means strong induction."""
    T = min(half, ev.seq_len // 2)
    if T < 8:
        return {}
    V = ev.ds.vocab_size
    g = torch.Generator().manual_seed(1234)
    r = torch.randint(0, V - 1, (n_seq, T), generator=g)
    r = r + (r >= ev.ds.eos_id).long()  # skip the eos id
    x = torch.cat([r, r], 1).to(ev.device)
    ev.model.eval()
    with torch.no_grad(), ev.amp():
        logits = ev.model(x).logits.float()
    nll = -torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None])[..., 0]
    ok = (logits[:, :-1].argmax(-1) == x[:, 1:]).float()
    first, second = nll[:, : T - 1].mean(), nll[:, T : 2 * T - 1].mean()
    return {"induction_loss_random": float(first), "induction_loss_repeat": float(second),
            "induction_gap": float(first - second), "induction_acc": float(ok[:, T : 2 * T - 1].mean())}


@register("metric", "generation")
def generation(ev, prompt_len: int = 16, new_tokens: int = 96):
    """Sample free-running text and measure diversity/degeneration (not 'quality' by itself)."""
    n = ev.cfg.eval.gen_samples
    if n <= 0:
        return {}
    plen = min(prompt_len, ev.seq_len // 4)
    new = min(new_tokens, 4 * ev.seq_len)
    rng = np.random.default_rng(0)
    x, _ = ev.ds.get_batch("val", n, plen, rng)
    g = torch.Generator(device="cpu").manual_seed(0)
    out = generate(ev.model, x.to(ev.device), new, temperature=1.0, amp=ev.amp, generator=None if ev.device.type != "cpu" else g)
    gen = out[:, plen:].cpu().numpy()
    d = {k: [] for k in (1, 2, 3)}
    for row in gen:
        for k in d:
            grams = [tuple(row[i : i + k]) for i in range(len(row) - k + 1)]
            d[k].append(len(set(grams)) / max(1, len(grams)))
    V = ev.ds.vocab_size
    pg = np.bincount(gen.ravel(), minlength=V) + 1e-9
    pv = np.bincount(np.asarray(ev.ds.val[: 2_000_000]).astype(np.int64), minlength=V) + 1.0
    pg, pv = pg / pg.sum(), pv / pv.sum()
    ev.notes["samples"] = [ev.tokenizer.decode(r.tolist()) for r in out[:4].cpu().numpy()]
    return {**{f"gen_distinct{k}": float(np.mean(v)) for k, v in d.items()},
            "gen_unigram_kl": float((pg * np.log(pg / pv)).sum())}


@register("metric", "task")
def task(ev):
    """Exact-match accuracy on a synthetic task (only for datasets curated from `source: synthetic`)."""
    sep = ev.ds.task_sep
    if not sep:
        return {}
    n = ev.cfg.eval.task_samples
    rng = np.random.default_rng(0)
    lines_by_doc = [[l for l in d.split("\n") if sep in l] for _, d in zip(range(200), ev.ds.docs("val"))]
    pool = [(di, li) for di, ls in enumerate(lines_by_doc) for li in range(len(ls))]
    if not pool:
        return {}
    ok = tot = 0
    for k in rng.permutation(len(pool))[:n]:
        di, li = pool[k]
        lines = lines_by_doc[di]
        prompt_line, ans = lines[li].split(sep, 1)[0] + sep, lines[li].split(sep, 1)[1]
        context = "".join(l + "\n" for l in lines[max(0, li - 2) : li])
        ids = ev.tokenizer.encode(context + prompt_line)
        x = torch.tensor([ids], device=ev.device)
        out = generate(ev.model, x, len(ev.tokenizer.encode(ans)) + 1, temperature=0, amp=ev.amp)
        pred = ev.tokenizer.decode(out[0, len(ids) :].tolist()).split("\n")[0]
        ok += pred == ans
        tot += 1
    return {"task_em": ok / max(1, tot), "task_n": float(tot)}


@register("metric", "throughput")
def throughput(ev, steps: int = 4):
    m = ev.model
    x, y = next(ev.val_batches(1))
    was = m.training
    m.train()

    def step_fn(bwd):
        with ev.amp():
            out = m(x, y)
        if bwd:
            out.loss.backward()
            m.zero_grad(set_to_none=True)

    res = {}
    if ev.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for name, bwd in (("fwd", False), ("fwdbwd", True)):
        step_fn(bwd)
        sync(ev.device)
        t0 = time.time()
        for _ in range(steps):
            step_fn(bwd)
        sync(ev.device)
        res[f"{name}_tok_per_s"] = x.numel() * steps / (time.time() - t0)
    if ev.device.type == "cuda":
        res["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    m.train(was)
    return res


# ---- deep (slower) diagnostics: `gpt eval RUN --deep` ----------------------------------------------------
@register("metric", "logit_lens")
@deep
def logit_lens(ev, max_batches: int = 4):
    """Loss if you read out the residual stream after layer i (and its rms). Shows where predictions form."""
    n = ev.model.cfg.n_layer
    sums, rms, cnt = np.zeros(n + 1), np.zeros(n), 0
    ev.model.eval()
    with torch.no_grad():
        for x, y in ev.val_batches(max_batches):
            with ev.amp():
                out = ev.model(x, capture=True)
                streams = [out.ctx.x0] + out.ctx.resid
                for i, h in enumerate(streams):
                    lg = ev.model.readout(h).float()
                    sums[i] += torch.nn.functional.cross_entropy(lg.view(-1, lg.size(-1)), y.reshape(-1)).item()
            for i, h in enumerate(out.ctx.resid):
                rms[i] += h.float().pow(2).mean().sqrt().item()
            cnt += 1
    res = {"lens_loss/embed": sums[0] / cnt}
    for i in range(n):
        res[f"lens_loss/L{i}"] = sums[i + 1] / cnt
        res[f"resid_rms/L{i}"] = rms[i] / cnt
    return res


@register("metric", "layer_ablation")
@deep
def layer_ablation(ev, max_batches: int = 4):
    """Skip block i entirely (x -> x) and report the loss increase. Which layers matter?"""
    model = ev.model
    batches = list(ev.val_batches(max_batches))

    def mean_loss():
        tot = 0.0
        with torch.no_grad():
            for x, y in batches:
                with ev.amp():
                    tot += model(x, y).ce.item()
        return tot / len(batches)

    model.eval()
    base = mean_loss()
    res = {}
    for i, blk in enumerate(model.blocks):
        blk.forward = lambda x, ctx: x
        try:
            res[f"ablate_dloss/L{i}"] = mean_loss() - base
        finally:
            del blk.forward  # restore the class method
    return res
