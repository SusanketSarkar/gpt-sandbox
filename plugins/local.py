"""Sliding-window (local) causal self-attention: each token attends only to the previous `window`
tokens. A drop-in, parameter-matched replacement for `mha` (same qkv/proj shapes) that is cheaper -
O(T*window) instead of O(T^2) - and cannot retrieve beyond `window` positions, so long-range recall
must come from the global (mha) layers. Mix it with full attention per layer, e.g.:

    attention:
      - {layers: "::4", name: mha}
      - {layers: all,  name: local, window: 32}

Set `window` >= seq_len and it becomes exactly causal full attention (used as an equivalence check).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_sandbox import register, repeat_kv  # noqa: F401


def window_mask(T: int, window: int, device) -> torch.BoolTensor:
    """Bool (T, T); True where row i may attend to column j: causal and within `window` tokens back."""
    i = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
    j = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    return (j <= i) & (j >= i - window + 1)


@register("attention", "local")
class Local(nn.Module):
    """Sliding-window causal self-attention. `window` = how far back each token can look (in tokens)."""

    def __init__(self, cfg, layer_idx, n_head: int | None = None, window: int = 32):
        super().__init__()
        d = cfg.d_model
        self.n_head = n_head or cfg.n_head
        self.head_dim = cfg.head_dim or d // self.n_head
        inner = self.n_head * self.head_dim
        self.window = int(window)
        self.qkv = nn.Linear(d, 3 * inner, bias=cfg.bias)
        self.proj = nn.Linear(inner, d, bias=cfg.bias)
        self.proj._resid_out = True
        self.attn_drop, self.resid_drop = cfg.dropout, nn.Dropout(cfg.dropout)

    def forward(self, x, ctx):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = ctx.posenc.rotate(q, k, ctx.positions)
        if k.size(1) != q.size(1):
            n_rep = q.size(1) // k.size(1)
            k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        mask = window_mask(T, self.window, x.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.attn_drop if ctx.training else 0.0, is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.resid_drop(self.proj(y))
