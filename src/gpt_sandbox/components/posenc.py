from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register


class PosEncBase(nn.Module):
    """Two hooks: `embed` (added to token embeddings) and `rotate` (applied to q/k inside attention)."""

    def embed(self, x, positions):
        return x

    def rotate(self, q, k, positions):  # q, k: (B, H, T, head_dim)
        return q, k


@register("posenc", "none")
class NoPosEnc(PosEncBase):
    def __init__(self, cfg):
        super().__init__()


@register("posenc", "learned")
class LearnedPosEnc(PosEncBase):
    def __init__(self, cfg):
        super().__init__()
        self.wpe = nn.Embedding(cfg.seq_len, cfg.d_model)

    def embed(self, x, positions):
        return x + self.wpe(positions)[None]


@register("posenc", "rope")
class RoPE(PosEncBase):
    """Rotary embeddings (GPT-NeoX half-split convention)."""

    def __init__(self, cfg, base: float = 10000.0, rotary_dim: int | None = None):
        super().__init__()
        hd = cfg.head_dim or cfg.d_model // cfg.n_head
        self.rot = rotary_dim or hd
        assert self.rot % 2 == 0 and self.rot <= hd, "rotary_dim must be even and <= head_dim"
        inv = 1.0 / (base ** (torch.arange(0, self.rot, 2).float() / self.rot))
        t = torch.arange(cfg.seq_len).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _rot(self, x, cos, sin):
        r = self.rot
        x1, x2, rest = x[..., : r // 2], x[..., r // 2 : r], x[..., r:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([out, rest], dim=-1) if rest.numel() else out

    def rotate(self, q, k, positions):
        cos = self.cos[positions].to(q.dtype)[None, None]  # (1,1,T,rot/2)
        sin = self.sin[positions].to(q.dtype)[None, None]
        return self._rot(q, cos, sin), self._rot(k, cos, sin)
