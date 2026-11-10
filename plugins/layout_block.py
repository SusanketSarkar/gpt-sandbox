"""`layout` block: one sublayer per block, the operation chosen by a per-position string.

    model:
      n_layer: 16                     # must equal len(layout)
      block: {name: layout}
      extra: {layout: "AMAMAMAMAMAMAMAM"}   # 'A' = attention sublayer, 'M' = MLP sublayer

Each block is one norm + one component (attention or mlp, from the registries) + one residual
join, so the alternating baseline (`layout: "AMAM..."`) is the standard pre-norm stack built
from the same parts. Run the baseline *through this same plugin* so that init scales
(1/sqrt(2*n_layer) on residual-output projections), norms and residual joins are identical
across every variant of a sweep - only the order (or the A/M multiset) differs.

Logs (on logging steps, via ctx.log):
  delta_rms/L{i}           RMS of the sublayer's output
  out_stream_ratio/L{i}    ||sublayer output|| / ||residual stream before the join||
"""
from __future__ import annotations

from gpt_sandbox import BlockBase, layer_component, layer_norm, register


@register("block", "layout")
class LayoutBlock(BlockBase):
    """x -> x + op_i(norm(x)) for op_i in {attention, mlp} given by cfg.extra.layout[i]."""

    def __init__(self, cfg, layer_idx):
        super().__init__(cfg, layer_idx)
        layout = (cfg.extra or {}).get("layout")
        if not isinstance(layout, str) or layer_idx >= len(layout) or layout[layer_idx].upper() not in "AM":
            raise ValueError(
                f"model.extra.layout must be a string of 'A'/'M' with one char per layer "
                f"(got {layout!r}, layer {layer_idx})")
        if layer_idx == 0 and len(layout) != cfg.n_layer:
            raise ValueError(
                f"model.n_layer ({cfg.n_layer}) must equal len(model.extra.layout) ({len(layout)})")
        self.op = layout[layer_idx].upper()
        self.norm = layer_norm(cfg, layer_idx)
        self.sub = layer_component("attention" if self.op == "A" else "mlp", cfg, layer_idx)
        self.res = layer_component("residual", cfg, layer_idx, "attn" if self.op == "A" else "mlp")

    def forward(self, x, ctx):
        x = self.run_inserts("block_start", x, ctx)
        delta = self.sub(self.norm(x), ctx)
        if ctx.collect:
            d = delta.detach().float()
            s = x.detach().float()
            ctx.log(f"delta_rms/L{self.layer_idx}", d.pow(2).mean().sqrt())
            ctx.log(f"out_stream_ratio/L{self.layer_idx}", (d.pow(2).mean() / s.pow(2).mean()).sqrt())
        x = self.res(x, delta, ctx)
        x = self.run_inserts("block_end", x, ctx)
        return x
