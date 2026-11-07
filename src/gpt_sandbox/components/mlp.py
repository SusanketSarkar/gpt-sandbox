from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from ..registry import register


@register("mlp", "gelu")
class GeluMLP(nn.Module):
    def __init__(self, cfg, layer_idx, d_ff: int | None = None):
        super().__init__()
        d, h = cfg.d_model, d_ff or cfg.d_ff or 4 * cfg.d_model
        self.up = nn.Linear(d, h, bias=cfg.bias)
        self.down = nn.Linear(h, d, bias=cfg.bias)
        self.down._resid_out = True  # gets the 1/sqrt(2*n_layer) init scaling
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, ctx):
        return self.drop(self.down(F.gelu(self.up(x))))


@register("mlp", "swiglu")
class SwiGLU(nn.Module):
    def __init__(self, cfg, layer_idx, d_ff: int | None = None):
        super().__init__()
        d = cfg.d_model
        h = d_ff or cfg.d_ff or int(round(8 * d / 3 / 32)) * 32  # ~param-matched with a 4x GELU MLP
        self.gate = nn.Linear(d, h, bias=cfg.bias)
        self.up = nn.Linear(d, h, bias=cfg.bias)
        self.down = nn.Linear(h, d, bias=cfg.bias)
        self.down._resid_out = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, ctx):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
