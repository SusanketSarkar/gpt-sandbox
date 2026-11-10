"""insert plugin 'stream_noise_eval': eval-time-only Gaussian noise on the residual stream, scaled to the stream's RMS.

    x <- x + sigma * rms(x) * N(0, I)

The probe-side twin of plugins/rms_noise.py: a no-op during training (ctx.training),
active at eval time. Used by the sensitivity profile (experiments/depth_fragility/sensitivity.py)
to measure how sensitive a trained model's final predictions are to a random perturbation
of one slice of the stream - a test-time, dependency-side view of "what is this layer carrying".
Registered under a distinct name so it never collides with run plugin snapshots.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gpt_sandbox import register


@register("insert", "stream_noise_eval")
class StreamNoiseEval(nn.Module):
    def __init__(self, cfg, layer_idx, site, sigma: float = 0.1):
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, x, ctx):
        if self.sigma == 0.0 or ctx is None or ctx.training:
            return x
        rms = x.float().pow(2).mean().sqrt()
        scale = self.sigma * rms
        if ctx is not None and ctx.collect:
            ctx.log(f"evalnoise_std/L{ctx.layer_idx}", scale)
        noise = torch.randn_like(x, dtype=torch.float32) * scale
        return x + noise.to(x.dtype)
