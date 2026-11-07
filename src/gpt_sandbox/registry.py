"""Component registry: the one mechanism behind "adding code is easy".

Everything swappable (attention, mlp, norm, positional encoding, residual
connection, whole blocks, residual-stream inserts, optimizers, LR schedules,
eval metrics, tokenizers, data sources, filters, synthetic tasks) is a named entry
in a registry. Config files refer to entries by name::

    model:
      attention: {name: my_attention, some_kwarg: 3}

and a plugin file registers them::

    from gpt_sandbox import register

    @register("attention", "my_attention")
    class MyAttention(nn.Module): ...

Plugin files in ``./plugins/*.py`` are auto-imported. ``gpt new <kind> <name>``
scaffolds a working one.
"""
from __future__ import annotations

import difflib
import inspect
from collections.abc import Mapping
from typing import Any, Callable

# kind -> one-line contract (shown by `gpt list` and used by `gpt new`)
KINDS: dict[str, str] = {
    "attention": "nn.Module  __init__(cfg, layer_idx, **kw)  forward(x, ctx) -> x     [x: (B,T,d)]",
    "mlp": "nn.Module  __init__(cfg, layer_idx, **kw)  forward(x, ctx) -> x",
    "norm": "nn.Module  __init__(dim, **kw)  forward(x) -> x",
    "posenc": "nn.Module  __init__(cfg, **kw)  embed(x, positions) -> x ; rotate(q, k, positions) -> (q, k)",
    "residual": "nn.Module  __init__(cfg, layer_idx, site, **kw)  forward(x, delta, ctx) -> x   [how a sublayer output joins the stream]",
    "block": "nn.Module (subclass BlockBase)  __init__(cfg, layer_idx, **kw)  forward(x, ctx) -> x",
    "insert": "nn.Module  __init__(cfg, layer_idx, site, **kw)  forward(x, ctx) -> x   [transform the residual stream at a named site]",
    "optimizer": "fn(groups, lr, **kw) -> torch.optim.Optimizer   [groups: list of param-group dicts]",
    "schedule": "fn(step, max_steps, warmup_steps, **kw) -> float multiplier on the base LR",
    "metric": "fn(ev, **kw) -> dict[str, float]   [ev: EvalContext; set fn.deep=True to run only with --deep]",
    "tokenizer": "class with fit/encode/decode/to_dict/from_dict (see data/tokenizers.py)",
    "source": "fn(**kw) -> Iterator[str]   [yields raw documents]",
    "filter": "fn(doc, **kw) -> bool | str   [True keep, False drop, str = replace the doc]",
    "task": "fn(rng, **kw) -> str   [one synthetic line; set fn.sep = answer separator for exact-match eval]",
}

_REG: dict[str, dict[str, Any]] = {k: {} for k in KINDS}
_ORIGIN: dict[tuple[str, str], str] = {}


def _origin(obj: Any) -> str:
    try:
        return inspect.getsourcefile(obj) or "?"
    except (TypeError, OSError):
        return "?"


def register(kind: str, name: str | None = None, *, override: bool = False) -> Callable:
    """Decorator: ``@register("attention", "gqa")``. Use ``override=True`` to replace a builtin."""
    if kind not in KINDS:
        raise ValueError(f"unknown component kind {kind!r}; choose from {list(KINDS)}")

    def deco(obj):
        n = name or obj.__name__.lower()
        old = _REG[kind].get(n)
        if old is not None and not override and _origin(old) != _origin(obj):
            raise ValueError(
                f"{kind} {n!r} is already registered (from {_origin(old)}). "
                f"Pick another name, or pass override=True to replace it."
            )
        _REG[kind][n] = obj
        _ORIGIN[(kind, n)] = _origin(obj)
        return obj

    return deco


def names(kind: str) -> list[str]:
    return sorted(_REG[kind])


def origin(kind: str, name: str) -> str:
    return _ORIGIN.get((kind, name), "?")


def get(kind: str, name: str) -> Any:
    table = _REG[kind]
    if name not in table:
        close = difflib.get_close_matches(str(name), table, n=3)
        hint = f" Did you mean {close}?" if close else ""
        raise KeyError(
            f"No {kind} named {name!r}.{hint} Registered: {sorted(table)}. "
            f"(Plugins in ./plugins/*.py are auto-loaded; scaffold one with `gpt new {kind} {name}`.)"
        )
    return table[name]


def parse_spec(spec: Any, kind: str = "") -> tuple[str, dict]:
    """A spec is ``"name"`` or ``{name: ..., **kwargs}``. Returns (name, kwargs)."""
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, Mapping):
        d = dict(spec)
        if "name" not in d:
            raise ValueError(f"{kind or 'component'} spec needs a 'name' key, got {dict(spec)}")
        return d.pop("name"), d
    raise TypeError(f"{kind or 'component'} spec must be a string or a mapping, got {type(spec).__name__}")


def _check_kwargs(kind: str, name: str, obj: Any, kw: dict, skip: int) -> None:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return
    params = list(sig.parameters.values())
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return
    accepted = [p.name for p in params[skip:] if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    unknown = [k for k in kw if k not in accepted]
    if unknown:
        raise TypeError(
            f"{kind} {name!r} got unexpected config key(s) {unknown}. It accepts: {accepted or '(none)'}"
        )


def build(kind: str, spec: Any, *args: Any, **ctor_kw: Any) -> Any:
    """Instantiate the registered object named by ``spec`` as ``obj(*args, **ctor_kw, **spec_kwargs)``."""
    name, kw = parse_spec(spec, kind)
    obj = get(kind, name)
    _check_kwargs(kind, name, obj, kw, skip=len(args))
    return obj(*args, **ctor_kw, **kw)
