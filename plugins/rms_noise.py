"""insert plugin 'rms_noise': training-only Gaussian noise on the residual stream, scaled to the stream's RMS.

    x <- x + sigma * rms(x) * N(0, I)

Used by the depth-fragility study ("where is the residual stream fragile?"):

    model:
      inserts:
        - {site: after_attn, name: rms_noise, layers: 4, sigma: 0.3}   # one layer
        - {site: after_attn, name: rms_noise, layers: all, sigma: 0.1} # every layer

The insert is a no-op at eval time (gated on ctx.training), so val_bpb / gen_gap / ablation
are all measured on the un-noised model - exactly the "hurt during training, read out clean"
protocol. sigma is a fraction of the stream's own RMS, so it is scale-invariant across
depths and model sizes: sigma=1.0 means the noise has the same std as the stream itself.
The injected std is logged as noise_std/L<layer> on collection steps (free otherwise).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gpt_sandbox import register


@register("insert", "rms_noise")
class RMSNoise(nn.Module):
    def __init__(self, cfg, layer_idx, site, sigma: float = 0.1, training_only: bool = True):
        super().__init__()
        self.sigma = float(sigma)
        self.training_only = bool(training_only)

    def forward(self, x, ctx):
        if self.sigma == 0.0 or ctx is None:
            return x
        if self.training_only and not ctx.training:
            return x
        rms = x.float().pow(2).mean().sqrt()
        scale = self.sigma * rms
        if ctx is not None and ctx.collect:
            ctx.log(f"noise_std/L{ctx.layer_idx}", scale)
        noise = torch.randn_like(x, dtype=torch.float32) * scale
        return x + noise.to(x.dtype)
