"""residual plugin 'gated': x + g * delta, one learned scalar gate g per sublayer join (init 1.0).

For the study "do learned layer gates predict which layers matter?":

    model:
      residual: {name: gated}
    optim:
      groups:
        - {match: "res_", lr_mult: 30, weight_decay: 0}

The gate parameter is named `res_g` so the regex "res_" (and only "res_") selects it:
it lives at blocks.<i>.res_attn.res_g and blocks.<i>.res_mlp.res_g.
Gate values are logged to metrics.jsonl on log steps as gate/L<layer>/<site> (site = attn|mlp),
and are readable from the checkpoint state dict for post-hoc analysis.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gpt_sandbox import register


@register("residual", "gated")
class GatedResidual(nn.Module):
    def __init__(self, cfg, layer_idx, site, init: float = 1.0):
        super().__init__()
        self.layer_idx, self.site = layer_idx, site
        self.res_g = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x, delta, ctx):
        if ctx is not None and ctx.training:
            ctx.log(f"gate/L{self.layer_idx}/{self.site}", self.res_g)
        return x + self.res_g * delta
