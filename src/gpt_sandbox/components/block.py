from __future__ import annotations

import torch.nn as nn

from ..model.build import layer_component, layer_norm
from ..registry import register


class BlockBase(nn.Module):
    """Base class for blocks. Call `self.run_inserts(site, x, ctx)` at each site you support so that
    config-level `model.inserts` can attach modules there without touching block code."""

    SITES = ("block_start", "after_attn", "block_end")

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        self.inserts = nn.ModuleDict()  # site -> ModuleList, filled by GPT from cfg.inserts

    def run_inserts(self, site, x, ctx):
        if site in self.inserts:
            for m in self.inserts[site]:
                x = m(x, ctx)
        return x


@register("block", "prenorm")
class PreNormBlock(BlockBase):
    """x -> x + attn(norm(x)) -> x + mlp(norm(x)); every part comes from the registries."""

    def __init__(self, cfg, layer_idx):
        super().__init__(cfg, layer_idx)
        self.norm1 = layer_norm(cfg, layer_idx)
        self.attn = layer_component("attention", cfg, layer_idx)
        self.res_attn = layer_component("residual", cfg, layer_idx, "attn")
        self.norm2 = layer_norm(cfg, layer_idx)
        self.mlp = layer_component("mlp", cfg, layer_idx)
        self.res_mlp = layer_component("residual", cfg, layer_idx, "mlp")

    def forward(self, x, ctx):
        x = self.run_inserts("block_start", x, ctx)
        x = self.res_attn(x, self.attn(self.norm1(x), ctx), ctx)
        x = self.run_inserts("after_attn", x, ctx)
        x = self.res_mlp(x, self.mlp(self.norm2(x), ctx), ctx)
        x = self.run_inserts("block_end", x, ctx)
        return x
