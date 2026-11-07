"""Ctx: the object threaded through every component's forward.

It is how components share state and report diagnostics without changing signatures:

  ctx.positions    LongTensor (T,)  absolute positions of the current sequence
  ctx.posenc       the positional-encoding module (attention calls ctx.posenc.rotate(q, k, ctx.positions))
  ctx.x0           the embedding output (after dropout / embed inserts) - handy for skip-from-input tricks
  ctx.layer_idx    index of the block currently running
  ctx.store        dict for passing tensors between layers (e.g. ctx.store['v0'] = v in layer 0)
  ctx.training     bool
  ctx.log(k, v)    record a scalar diagnostic (only materialised on logging/eval steps; free otherwise)
  ctx.add_loss(k, v)  add an auxiliary loss term (already weighted) to the training loss
"""
from __future__ import annotations

import torch


class Ctx:
    def __init__(self, cfg, positions, training: bool = False, collect: bool = False, capture: bool = False):
        self.cfg = cfg
        self.positions = positions
        self.training = training
        self.collect = collect  # True on logging steps: ctx.log() stores values
        self.capture = capture  # True when the caller wants every block's output (logit lens etc.)
        self.posenc = None
        self.x0 = None
        self.layer_idx = 0
        self.store: dict = {}
        self.aux: dict = {}
        self.stats: dict = {}
        self.resid: list = []

    def log(self, name: str, value) -> None:
        if not self.collect:
            return
        if torch.is_tensor(value):
            value = value.detach().float().mean()
        self.stats[name] = value

    def add_loss(self, name: str, value) -> None:
        self.aux[name] = self.aux[name] + value if name in self.aux else value
