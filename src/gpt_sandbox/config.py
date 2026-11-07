"""Config: one YAML file, deep-merged over defaults, with `base:` inheritance and `--set a.b.c=v` overrides."""
from __future__ import annotations

import copy
import difflib
from collections.abc import Mapping
from pathlib import Path

import yaml

from .settings import base_dir

DEFAULTS: dict = {
    "name": None,
    "base": None,
    "seed": 1337,
    "plugins": [],  # extra plugin dirs/files (./plugins is always loaded)
    "runs_dir": "runs",
    "data": {"name": "shakespeare", "dir": "data"},
    "model": {
        "n_layer": 4,
        "d_model": 128,
        "n_head": 4,
        "head_dim": None,  # default d_model // n_head
        "d_ff": None,  # default depends on the mlp (4*d for gelu, ~8/3*d for swiglu)
        "seq_len": 128,
        "vocab_size": None,  # filled from the dataset
        "dropout": 0.0,
        "bias": False,
        "tie_embeddings": True,
        "init_std": 0.02,
        # --- swappable parts: "name" or {name: ..., **kwargs}; a list = cycle over layers,
        #     or (every entry having `layers:`) select by layer index/slice -> see README
        "norm": {"name": "rmsnorm"},
        "final_norm": None,  # defaults to `norm`
        "posenc": {"name": "rope"},
        "attention": {"name": "mha"},
        "mlp": {"name": "swiglu"},
        "residual": {"name": "plain"},
        "block": {"name": "prenorm"},
        "inserts": [],  # [{site: block_end, name: my_insert, layers: all, **kwargs}]
        "extra": {},  # free-form knobs for your components: cfg.extra.my_knob
    },
    "optim": {
        "name": "adamw",
        "lr": 1e-3,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "groups": [],  # [{match: "regex on param name", lr_mult: 10, weight_decay: 0}]
        "kwargs": {},  # extra kwargs for custom optimizers
    },
    "train": {
        "max_steps": 1000,
        "batch_size": 32,
        "grad_accum": 1,
        "grad_clip": 1.0,
        "warmup_steps": 50,
        "schedule": {"name": "cosine", "min_lr_ratio": 0.1},
        "eval_interval": 100,
        "eval_batches": 20,
        "log_interval": 20,
        "save_interval": 500,
        "precision": "auto",  # auto | bf16 | fp32
        "compile": False,
        "device": "auto",
        "final_eval": True,  # run the full `gpt eval` at the end of training
    },
    "eval": {
        "metrics": [
            "params", "val", "position", "calibration", "generalization",
            "baselines", "induction", "generation", "task", "throughput",
        ],
        "deep_metrics": ["logit_lens", "layer_ablation"],
        "batch_size": 16,
        "max_batches": None,
        "task_samples": 100,
        "gen_samples": 16,
    },
}


class Cfg(dict):
    """dict with attribute access (cfg.model.n_layer). Missing keys raise AttributeError."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(f"config has no key {k!r}") from None

    def __setattr__(self, k, v):
        self[k] = v


def wrap(x):
    if isinstance(x, Mapping):
        return Cfg({k: wrap(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return [wrap(v) for v in x]
    return x


def unwrap(x):
    if isinstance(x, Mapping):
        return {k: unwrap(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [unwrap(v) for v in x]
    return x


# fields holding a component spec {name: ..., **kwargs}: naming a *different* component replaces the old spec
# wholesale (otherwise the old component's kwargs would leak into the new one)
SPEC_KEYS = {"attention", "mlp", "norm", "final_norm", "posenc", "residual", "block", "schedule"}


def deep_merge(a: dict, b: Mapping) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            if k in SPEC_KEYS and "name" in v and v.get("name") != out[k].get("name"):
                out[k] = copy.deepcopy(v)
            else:
                out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path, _seen=()) -> dict:
    path = Path(path)
    if path in _seen:
        raise ValueError(f"circular `base:` in {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.get("base")
    if base:
        base_path = (path.parent / base) if not Path(base).is_absolute() else Path(base)
        if not base_path.exists():  # also allow paths relative to cwd
            base_path = Path(base)
        raw = deep_merge(_load_yaml(base_path, _seen + (path,)), {k: v for k, v in raw.items() if k != "base"})
    return raw


def _set_path(d, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = d
    for i, k in enumerate(keys[:-1]):
        nxt = keys[i + 1]
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            if k not in cur or cur[k] is None:
                cur[k] = [] if nxt.isdigit() else {}
            cur = cur[k]
    last = keys[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def parse_override(s: str):
    if "=" not in s:
        raise ValueError(f"override {s!r} must look like key.sub=value")
    k, v = s.split("=", 1)
    return k.strip(), yaml.safe_load(v)  # so 3, 1e-3, true, [1,2], {name: x} all work


def apply_overrides(raw: dict, overrides) -> dict:
    """overrides: list of 'a.b=v' strings, or a dict {'a.b': v}. A value REPLACES what was at that path."""
    items = overrides.items() if isinstance(overrides, Mapping) else [parse_override(s) for s in (overrides or [])]
    raw = copy.deepcopy(raw)
    for k, v in items:
        _set_path(raw, k, v)
    return raw


def validate(cfg: dict) -> None:
    for k in cfg:
        if k not in DEFAULTS:
            close = difflib.get_close_matches(k, DEFAULTS, n=2)
            raise ValueError(f"unknown top-level config key {k!r}." + (f" Did you mean {close}?" if close else ""))
    for sec in ("data", "model", "optim", "train", "eval"):
        for k in cfg.get(sec, {}):
            if k not in DEFAULTS[sec]:
                close = difflib.get_close_matches(k, DEFAULTS[sec], n=2)
                hint = f" Did you mean {close}?" if close else ""
                extra = " Custom knobs go under model.extra." if sec == "model" else ""
                raise ValueError(f"unknown key {sec}.{k}.{hint}{extra} Valid: {sorted(DEFAULTS[sec])}")
    m = cfg["model"]
    hd = m.get("head_dim") or m["d_model"] // m["n_head"]
    if not m.get("head_dim") and m["d_model"] % m["n_head"]:
        raise ValueError(f"model.d_model ({m['d_model']}) must be divisible by model.n_head ({m['n_head']}) "
                         f"(or set model.head_dim explicitly)")
    if hd % 2:
        raise ValueError(f"head_dim must be even (got {hd})")


def load_config(path=None, overrides=None) -> Cfg:
    raw = _load_yaml(Path(path)) if path else {}
    raw = apply_overrides(raw, overrides)
    merged = deep_merge(DEFAULTS, raw)
    base = base_dir()
    if base:
        if "runs_dir" not in raw:
            merged["runs_dir"] = str(base / "runs")
        if "dir" not in (raw.get("data") or {}):
            merged["data"]["dir"] = str(base / "data")
    validate(merged)
    return wrap(merged)


def dump_config(cfg, path) -> None:
    Path(path).write_text(yaml.safe_dump(unwrap(cfg), sort_keys=False))
