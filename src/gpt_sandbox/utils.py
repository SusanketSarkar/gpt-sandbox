from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(spec: str | None = "auto") -> torch.device:
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_factory(device: torch.device, precision: str = "auto"):
    """Returns (callable giving an autocast context, resolved precision name)."""
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "bf16" and device.type in ("cuda", "cpu"):
        return (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)), "bf16"
    return contextlib.nullcontext, "fp32"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def human(n: float) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def fmt(v, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if v != v:
            return "nan"
        if float(v).is_integer() and abs(v) < 1e15:
            return f"{int(v):,}"
        if abs(v) >= 1e5 or (v != 0 and abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:.{nd}f}"
    return str(v)


def table(rows: list[list], headers: list[str] | None = None, indent: str = "") -> str:
    rows = [[str(c) for c in r] for r in rows]
    if headers:
        rows = [list(map(str, headers))] + rows
    if not rows:
        return ""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for k, r in enumerate(rows):
        lines.append(indent + "  ".join(c.ljust(w) if j == 0 else c.rjust(w) for j, (c, w) in enumerate(zip(r, widths))))
        if headers and k == 0:
            lines.append(indent + "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def hash_obj(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def bar(frac: float, width: int = 30) -> str:
    frac = max(0.0, min(1.0, frac))
    return "█" * int(round(frac * width))
