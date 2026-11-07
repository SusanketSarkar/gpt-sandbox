"""`gpt check`: contract tests for a model config. Run it every time you write a new component."""
from __future__ import annotations

import math
import time

import torch

from .data.dataset import Dataset
from .model.gpt import GPT
from .utils import get_device, human, table


def _vocab(cfg) -> int:
    if cfg.model.vocab_size:
        return cfg.model.vocab_size
    try:
        return Dataset.load(cfg.data.name, cfg.data.dir).vocab_size
    except Exception:
        return 256


def run_checks(cfg, device: str = "cpu") -> bool:
    dev = get_device(device)
    cfg.model.vocab_size = _vocab(cfg)
    V, T = cfg.model.vocab_size, min(cfg.model.seq_len, 64)
    results: list[tuple[str, str, str]] = []

    def rec(status, name, msg=""):
        results.append((status, name, msg))

    torch.manual_seed(0)
    try:
        model = GPT(cfg.model).to(dev).eval()
    except Exception as e:
        print(f"FAIL  build: {type(e).__name__}: {e}")
        raise
    rec("PASS", "build", f"{human(model.num_params())} params")

    idx = torch.randint(0, V, (3, T), device=dev)
    tgt = torch.randint(0, V, (3, T), device=dev)  # independent of inputs (tied embeddings make tgt==input too easy)
    try:
        with torch.no_grad():
            out = model(idx, tgt)
        ok_shape = tuple(out.logits.shape) == (3, T, V)
        rec("PASS" if ok_shape else "FAIL", "forward shape", f"logits {tuple(out.logits.shape)} expected {(3, T, V)}")
        finite = bool(torch.isfinite(out.logits).all() and torch.isfinite(out.loss))
        rec("PASS" if finite else "FAIL", "finite outputs")
        l0, lnv = out.ce.item(), math.log(V)
        rec("PASS" if abs(l0 - lnv) < 0.5 else "WARN", "init loss ~ ln(V)",
            f"{l0:.3f} vs ln V = {lnv:.3f}" + ("" if abs(l0 - lnv) < 0.5 else "  (large init logits? check init_std / your init)"))
    except Exception as e:
        rec("FAIL", "forward", f"{type(e).__name__}: {e}")
        return _report(results)

    # causality: changing tokens at positions >= t0 must not change logits before t0
    with torch.no_grad():
        t0 = T // 2
        alt = idx.clone()
        alt[:, t0:] = torch.randint(0, V, alt[:, t0:].shape, device=dev)
        a, b = model(idx).logits, model(alt).logits
        diff = (a[:, :t0] - b[:, :t0]).abs().max().item()
        rec("PASS" if diff < 1e-4 else "FAIL", "causality", f"max |dlogit| on past positions = {diff:.2e}"
            + ("" if diff < 1e-4 else "  <-- the model looks at future tokens!"))
        # batch independence
        solo = model(idx[:1]).logits
        d2 = (solo - a[:1]).abs().max().item()
        rec("PASS" if d2 < 1e-4 else "FAIL", "batch independence", f"max diff {d2:.2e}")
        # variable length
        try:
            short = model(idx[:, : max(2, T // 3)]).logits
            same = (short - a[:, : short.size(1)]).abs().max().item()
            rec("PASS" if same < 1e-4 else "FAIL", "prefix consistency",
                f"logits of a short input match the long input's prefix (diff {same:.2e})")
        except Exception as e:
            rec("FAIL", "variable length", f"{type(e).__name__}: {e}")

    # gradients reach every parameter
    model.train()
    for p in model.parameters():
        p.grad = None
    out = model(idx, tgt)
    out.loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    zero = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().max() == 0]
    rec("PASS" if not dead else "FAIL", "gradients", f"no/NaN grad: {dead[:5]}" if dead else "all params get finite grads")
    if zero:
        rec("WARN", "zero gradients", f"{len(zero)} params have exactly-zero grad at init (ok for zero-init gates): {zero[:4]}")

    # can it overfit one batch?
    try:
        torch.manual_seed(1)
        m2 = GPT(cfg.model).to(dev).train()
        opt = torch.optim.AdamW(m2.parameters(), lr=3e-3)
        first = last = None
        for i in range(40):
            o = m2(idx, tgt)
            opt.zero_grad()
            o.loss.backward()
            opt.step()
            first = o.ce.item() if first is None else first
            last = o.ce.item()
        rec("PASS" if last < 0.6 * first else "WARN", "overfit one batch", f"{first:.2f} -> {last:.2f} in 40 steps")
    except Exception as e:
        rec("FAIL", "overfit one batch", f"{type(e).__name__}: {e}")

    # speed
    model.train()
    t = time.time()
    for _ in range(3):
        model(idx, tgt).loss.backward()
    rec("PASS", "speed", f"{(time.time() - t) / 3 * 1000:.0f} ms / fwd+bwd of {tuple(idx.shape)} on {dev}")
    return _report(results)


def _report(results) -> bool:
    print(table([[s, n, m] for s, n, m in results], ["", "check", "detail"]))
    bad = [r for r in results if r[0] == "FAIL"]
    print("\n" + ("all checks passed" if not bad else f"{len(bad)} check(s) FAILED") +
          (f" ({sum(r[0] == 'WARN' for r in results)} warning(s))" if any(r[0] == 'WARN' for r in results) else ""))
    return not bad
