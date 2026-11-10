"""`ctx_sensitivity` deep metric: where does the model actually integrate context?

For each layer i we compare the residual stream (at the LAST position) of a real sequence
against the same sequence with every earlier token replaced by a random one (the final token is
kept, so the only difference is the prefix). The per-layer relative change

    ||h_i(x) - h_i(rand_prefix + x[-1])|| / ||h_i(x)||

is low where the layer's output is (approximately) token-local and jumps where the layer starts
using the earlier context. Reading the curve across depth directly tests the
'early = local, middle = context integration, late = refinement' hypothesis.

Caveat (by design): the change at layer i also includes the effect of earlier layers reacting
to the perturbed prefix, so it is an upper-bound-style measure of where context enters the
computation, not a clean causal attribution.
"""
from __future__ import annotations

import torch

from gpt_sandbox import register


@register("metric", "ctx_sensitivity")
def ctx_sensitivity(ev, n_seq: int = 8, seed: int = 20261108):
    model = ev.model
    model.eval()
    T = ev.seq_len
    x, _ = next(ev.val_batches(1))
    n = min(n_seq, x.size(0))
    x = x[:n]
    last = x[:, -1:]
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = torch.randint(0, ev.ds.vocab_size, (n, T - 1), generator=g)
    # keep in-vocab and drop the eos token (map it to a harmless content index)
    eos = ev.ds.eos_id
    if eos is not None and 0 <= eos < ev.ds.vocab_size:
        rand = torch.where(rand == eos, torch.zeros_like(rand), rand)
    x2 = torch.cat([rand.to(ev.device), last], dim=1)

    with torch.no_grad(), ev.amp():
        o1 = model(x, capture=True)
        o2 = model(x2, capture=True)
    s1 = [o1.ctx.x0] + list(o1.ctx.resid)
    s2 = [o2.ctx.x0] + list(o2.ctx.resid)

    out = {}
    for i, (a, b) in enumerate(zip(s1, s2)):
        a_last = a[:, -1, :].detach().float()
        b_last = b[:, -1, :].detach().float()
        num = (a_last - b_last).norm(dim=1).mean()
        den = a_last.norm(dim=1).mean() + 1e-9
        cos = torch.nn.functional.cosine_similarity(a_last, b_last, dim=1).mean()
        tag = "embed" if i == 0 else f"L{i - 1}"
        out[f"ctx_sens/{tag}"] = float(num / den)
        out[f"ctx_sens_cos/{tag}"] = float(cos)
    return out
