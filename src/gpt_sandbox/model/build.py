from __future__ import annotations

from ..registry import build
from .spec import select_spec


def layer_component(kind: str, cfg, layer_idx: int, *extra_args, **extra_kw):
    """Build the `kind` component ('attention', 'mlp', ...) configured for this layer.

    Respects per-layer specs (lists / selectors). Constructor gets (cfg, layer_idx, *extra_args, **spec_kwargs).
    """
    spec = select_spec(cfg[kind], layer_idx, cfg.n_layer, what=f"model.{kind}")
    return build(kind, spec, cfg, layer_idx, *extra_args, **extra_kw)


def layer_norm(cfg, layer_idx: int, dim: int | None = None, field: str = "norm"):
    spec = select_spec(cfg[field] if cfg.get(field) is not None else cfg.norm, layer_idx, cfg.n_layer, what=f"model.{field}")
    return build("norm", spec, dim or cfg.d_model)
