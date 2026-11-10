"""`long_recall` metric: a recall probe as a function of distance.

For each distance d the input is [r r] where r is a random token sequence of length d.
Predicting the 2nd copy requires retrieving the exact token seen d positions earlier:

    recall_gap/d  = loss(1st copy) - loss(2nd copy)   (~0 => can't use context at that distance)
    recall_acc/d  = top-1 accuracy on the 2nd copy

A full-attention model keeps recall high as d grows (up to the context window); a
sliding-window model degrades to chance once d exceeds its window. This is the cleanest
single number for "how much long-context recall does this architecture retain?".

Registered as a builtin-style metric; add `long_recall` to `eval.metrics` to run it.
"""
from __future__ import annotations

import torch

from gpt_sandbox import register


@register("metric", "long_recall")
def long_recall(ev, n_seq: int = 32, distances=None):
    V = ev.ds.vocab_size
    eos = ev.ds.eos_id
    maxd = ev.seq_len // 2  # [r r] must fit in the context window
    if maxd < 8:
        return {}
    if distances is None:
        distances = [d for d in (8, 16, 32, 64, 128, 256, 512) if d <= maxd]
    if not distances:
        return {}

    g = torch.Generator().manual_seed(7)
    ev.model.eval()
    out = {}
    with torch.no_grad(), ev.amp():
        for d in distances:
            r = torch.randint(0, V - 1, (n_seq, d), generator=g)
            r = r + (r >= eos).long()  # shift past the eos id so the probe stays in-vocab
            x = torch.cat([r, r], dim=1).to(ev.device)  # (n, 2d)
            logits = ev.model(x).logits.float()
            nll = -torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None])[..., 0]
            ok = (logits[:, :-1].argmax(-1) == x[:, 1:]).float()
            first = nll[:, : d - 1].mean()
            second = nll[:, d : 2 * d - 1].mean()
            out[f"recall_gap/{d}"] = float(first - second)
            out[f"recall_acc/{d}"] = float(ok[:, d : 2 * d - 1].mean())
    out["recall_max_d"] = float(max(distances))
    return out
