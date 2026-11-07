"""Per-layer spec selection, so one config line can describe hybrid / heterogeneous stacks.

A spec field (attention, mlp, norm, block, residual) may be:
  "name" | {name: ..., **kw}                    same everywhere
  [spec, spec, spec]                            cycled over layers:  layer i uses spec[i % len]
  [{layers: <sel>, name: ...}, ...]             first entry whose selector matches the layer wins

Layer selectors: "all" | int (negatives ok) | "even" | "odd" | "first" | "last"
                 | python-slice string "a:b:c" (e.g. "::4", "3::4", "-2:") | a list of selectors.
"""
from __future__ import annotations

from collections.abc import Mapping


def layer_selected(selector, i: int, n: int) -> bool:
    if selector is None or selector == "all":
        return True
    if isinstance(selector, bool):
        raise ValueError("layers selector cannot be a bool")
    if isinstance(selector, int):
        return i == (selector % n)
    if isinstance(selector, (list, tuple)):
        return any(layer_selected(s, i, n) for s in selector)
    if isinstance(selector, str):
        s = selector.strip()
        if s == "even":
            return i % 2 == 0
        if s == "odd":
            return i % 2 == 1
        if s == "first":
            return i == 0
        if s == "last":
            return i == n - 1
        if ":" in s:
            parts = [int(p) if p.strip() else None for p in s.split(":")]
            return i in range(n)[slice(*parts)]
        if s.lstrip("-").isdigit():
            return i == (int(s) % n)
    raise ValueError(f"bad layers selector {selector!r}")


def _clean(spec):
    if isinstance(spec, Mapping):
        return {k: v for k, v in spec.items() if k != "layers"}
    return spec


def select_spec(spec, i: int, n: int, what: str = "spec"):
    """Return the single spec (str or dict, `layers` key removed) that applies to layer i of n."""
    if spec is None or isinstance(spec, (str, Mapping)):
        return _clean(spec)
    if not isinstance(spec, (list, tuple)) or not spec:
        raise ValueError(f"{what}: expected a name, a mapping, or a non-empty list, got {spec!r}")
    has_sel = [isinstance(s, Mapping) and "layers" in s for s in spec]
    if all(has_sel):
        for s in spec:
            if layer_selected(s["layers"], i, n):
                return _clean(s)
        raise ValueError(f"{what}: no entry's `layers` selector matches layer {i} (of {n}); add a catch-all `layers: all`")
    if not any(has_sel):
        return _clean(spec[i % len(spec)])
    raise ValueError(f"{what}: either every list entry has `layers:` or none does")
