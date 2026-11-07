"""Sampling. No KV cache on purpose: every step recomputes the (cropped) context, so ANY attention /
recurrent / weird component works for generation without implementing a cache. Fine for small models."""
from __future__ import annotations

import contextlib

import torch


@torch.no_grad()
def generate(model, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, eos_id: int | None = None, stop_on_eos: bool = False,
             amp=None, generator: torch.Generator | None = None):
    """idx: (B, T) prompt tokens -> (B, T + n) tokens. temperature 0 = greedy."""
    was_training = model.training
    model.eval()
    amp = amp or contextlib.nullcontext
    done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    for _ in range(max_new_tokens):
        with amp():
            logits = model(idx[:, -model.cfg.seq_len :]).logits[:, -1].float()
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p and top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.softmax(sl, -1).cumsum(-1)
                drop = cp - torch.softmax(sl, -1) > top_p
                sl = sl.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)
        if stop_on_eos and eos_id is not None:
            nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt)
            done = done | (nxt[:, 0] == eos_id)
        idx = torch.cat([idx, nxt], dim=1)
        if stop_on_eos and bool(done.all()):
            break
    model.train(was_training)
    return idx
