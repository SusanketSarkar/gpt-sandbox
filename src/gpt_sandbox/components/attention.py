from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from ..registry import register


def repeat_kv(x, n_rep: int):
    """(B, H_kv, T, hd) -> (B, H_kv * n_rep, T, hd): share each kv head across n_rep query heads."""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


def sdpa_causal(q, k, v, dropout_p: float = 0.0, training: bool = False):
    """Causal scaled-dot-product attention. q: (B,Hq,T,hd), k/v: (B,Hkv,T,hd) with Hq a multiple of Hkv."""
    if k.size(1) != q.size(1):
        n_rep = q.size(1) // k.size(1)
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p if training else 0.0, is_causal=True)


@register("attention", "mha")
class MHA(nn.Module):
    """Standard multi-head causal self-attention with fused qkv and optional rotary embeddings."""

    def __init__(self, cfg, layer_idx, n_head: int | None = None):
        super().__init__()
        d = cfg.d_model
        self.n_head = n_head or cfg.n_head
        self.head_dim = cfg.head_dim or d // self.n_head
        inner = self.n_head * self.head_dim
        self.qkv = nn.Linear(d, 3 * inner, bias=cfg.bias)
        self.proj = nn.Linear(inner, d, bias=cfg.bias)
        self.proj._resid_out = True
        self.attn_drop, self.resid_drop = cfg.dropout, nn.Dropout(cfg.dropout)

    def forward(self, x, ctx):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = ctx.posenc.rotate(q, k, ctx.positions)
        y = sdpa_causal(q, k, v, self.attn_drop, ctx.training)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.resid_drop(self.proj(y))
