"""LR schedules return a multiplier on the base lr. Contract: fn(step, max_steps, warmup_steps, **kw) -> float."""
from __future__ import annotations

import math

from ..registry import register


def _warm(step, warmup):
    return (step + 1) / warmup if warmup > 0 and step < warmup else None


@register("schedule", "cosine")
def cosine(step, max_steps, warmup_steps, min_lr_ratio: float = 0.1):
    w = _warm(step, warmup_steps)
    if w is not None:
        return w
    t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@register("schedule", "linear")
def linear(step, max_steps, warmup_steps, min_lr_ratio: float = 0.0):
    w = _warm(step, warmup_steps)
    if w is not None:
        return w
    t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 1 - (1 - min_lr_ratio) * min(1.0, t)


@register("schedule", "constant")
def constant(step, max_steps, warmup_steps):
    w = _warm(step, warmup_steps)
    return 1.0 if w is None else w


@register("schedule", "wsd")
def wsd(step, max_steps, warmup_steps, decay_frac: float = 0.2, min_lr_ratio: float = 0.0):
    """warmup - stable - decay (linear to min_lr_ratio over the last decay_frac of training)."""
    w = _warm(step, warmup_steps)
    if w is not None:
        return w
    start = int(max_steps * (1 - decay_frac))
    if step < start:
        return 1.0
    t = (step - start) / max(1, max_steps - start)
    return 1 - (1 - min_lr_ratio) * min(1.0, t)
