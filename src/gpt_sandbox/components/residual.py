from __future__ import annotations

import torch.nn as nn

from ..registry import register


@register("residual", "plain")
class PlainResidual(nn.Module):
    """x + delta. Swap this to change HOW every sublayer output joins the residual stream
    (learned scalar gates, per-channel scales, LayerScale, ReZero, interpolation, ...)."""

    def __init__(self, cfg, layer_idx, site):
        super().__init__()
        self.layer_idx, self.site = layer_idx, site  # site is "attn" or "mlp"

    def forward(self, x, delta, ctx):
        return x + delta
