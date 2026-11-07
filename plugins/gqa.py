"""attention plugin 'gqa' (scaffolded by `gpt new attention gqa`)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_sandbox import register, Ctx, sdpa_causal, repeat_kv, layer_component, BlockBase, PosEncBase  # noqa: F401


@register("attention", "gqa")
class Gqa(nn.Module):
    """Accepts a new n_kv_heads args and allows gqa"""

    def __init__(self, cfg, layer_idx, n_head: int | None = None, n_kv_heads: int = 1):
        super().__init__()
        d = cfg.d_model
        self.n_head = n_head or cfg.n_head
        self.head_dim = cfg.head_dim or d // self.n_head
        self.n_kv_heads = n_kv_heads
        
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        
        # Combined projection for Q, K, V with differing key/value head counts
        self.qkv = nn.Linear(d, q_dim + 2 * kv_dim, bias=cfg.bias)
        self.proj = nn.Linear(q_dim, d, bias=cfg.bias)
        self.proj._resid_out = True  # marks the residual-output projection for depth-scaled init
        self.attn_drop, self.resid_drop = cfg.dropout, nn.Dropout(cfg.dropout)

    def forward(self, x, ctx: Ctx):  # x: (B, T, d) -> (B, T, d)
        B, T, _ = x.shape
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        
        q, k, v = self.qkv(x).split([q_dim, kv_dim, kv_dim], dim=-1)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        q, k = ctx.posenc.rotate(q, k, ctx.positions)  # rope is applied here (no-op for other posenc)
        
        # sdpa_causal natively handles broadcasting when n_kv_heads < n_head
        y = sdpa_causal(q, k, v, self.attn_drop, ctx.training)
        y = y.transpose(1, 2).reshape(B, T, -1)
        ctx.log("gqa/y_rms", y.float().pow(2).mean().sqrt())  # shows up in metrics.jsonl on log steps
        return self.resid_drop(self.proj(y))