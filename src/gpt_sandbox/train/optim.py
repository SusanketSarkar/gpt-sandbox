"""Optimizers + param groups.

`optim.groups` lets you give any parameters their own lr / weight decay by regex on the parameter name::

    optim:
      groups:
        - {match: "gate", lr_mult: 10, weight_decay: 0}
        - {match: "tok_emb", lr_mult: 0.5}

First matching group wins; everything else falls into the default groups (decay for >=2-D weights,
no decay for norms/biases/1-D params). Custom optimizer contract: fn(groups, lr, **kw) -> Optimizer,
where `groups` is a list of dicts with `params`, `lr_mult`, `weight_decay`, `name`.
"""
from __future__ import annotations

import re

import torch

from ..registry import build, register


def make_param_groups(model, optim_cfg) -> list[dict]:
    custom = [(re.compile(g["match"]), g) for g in (optim_cfg.groups or [])]
    buckets: dict[str, dict] = {}
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        for k, (rx, g) in enumerate(custom):
            if rx.search(name):
                key = f"custom{k}:{g['match']}"
                b = buckets.setdefault(key, {"params": [], "names": [], "name": key,
                                             "lr_mult": g.get("lr_mult", 1.0),
                                             "weight_decay": g.get("weight_decay", optim_cfg.weight_decay)})
                break
        else:
            decay = p.ndim >= 2
            key = "decay" if decay else "no_decay"
            b = buckets.setdefault(key, {"params": [], "names": [], "name": key, "lr_mult": 1.0,
                                         "weight_decay": optim_cfg.weight_decay if decay else 0.0})
        b["params"].append(p)
        b["names"].append(name)
    unmatched = [g["match"] for _, g in custom if not any(b["name"].endswith(g["match"]) for b in buckets.values())]
    if unmatched:
        raise ValueError(f"optim.groups pattern(s) {unmatched} matched no parameter. "
                         f"Names look like e.g. {next(iter(model.named_parameters()))[0]!r}")
    return list(buckets.values())


@register("optimizer", "adamw")
def adamw(groups, lr, betas=(0.9, 0.95), eps=1e-8, **_):
    pg = [{"params": g["params"], "lr": lr * g["lr_mult"], "weight_decay": g["weight_decay"],
           "lr_mult": g["lr_mult"], "name": g["name"]} for g in groups]
    fused = torch.cuda.is_available() and all(p.is_cuda for g in groups for p in g["params"])
    return torch.optim.AdamW(pg, lr=lr, betas=tuple(betas), eps=eps, fused=fused or None)


@register("optimizer", "sgd")
def sgd(groups, lr, momentum: float = 0.9, **_):
    pg = [{"params": g["params"], "lr": lr * g["lr_mult"], "weight_decay": g["weight_decay"],
           "lr_mult": g["lr_mult"], "name": g["name"]} for g in groups]
    return torch.optim.SGD(pg, lr=lr, momentum=momentum)


def build_optimizer(model, optim_cfg):
    groups = make_param_groups(model, optim_cfg)
    spec = {"name": optim_cfg.name}
    kw = {"betas": optim_cfg.betas, "eps": optim_cfg.eps, **(optim_cfg.kwargs or {})}
    opt = build("optimizer", spec, groups, optim_cfg.lr) if not kw else _call(spec, groups, optim_cfg.lr, kw)
    for g in opt.param_groups:
        g.setdefault("lr_mult", 1.0)
    return opt


def _call(spec, groups, lr, kw):
    from ..registry import get

    fn = get("optimizer", spec["name"])
    return fn(groups, lr, **kw)
