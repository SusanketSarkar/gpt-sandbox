"""Local CLI settings. `gpt adddir PATH` stores a base directory for `data/` and `runs/`.

The settings file lives in the current working directory, so different projects can point at
different result directories. Without it, relative `data/` and `runs/` paths are used as before.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SETTINGS_NAME = ".gpt-sandbox.json"


def settings_path() -> Path:
    env = os.environ.get("GPT_SANDBOX_SETTINGS")
    return Path(env).expanduser() if env else Path.cwd() / SETTINGS_NAME


def _load() -> dict:
    p = settings_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict) -> None:
    p = settings_path()
    if not data:
        p.unlink(missing_ok=True)
        return
    p.write_text(json.dumps(data, indent=2) + "\n")


def base_dir() -> Path | None:
    env = os.environ.get("GPT_SANDBOX_HOME")
    if env:
        return Path(env).expanduser().resolve()
    value = _load().get("base_dir")
    return Path(value).expanduser().resolve() if value else None


def set_base_dir(path) -> Path:
    base = Path(path).expanduser().resolve()
    if base.exists() and not base.is_dir():
        raise ValueError(f"{base} exists and is not a directory")
    base.mkdir(parents=True, exist_ok=True)
    data = _load()
    data["base_dir"] = str(base)
    _save(data)
    return base


def reset_base_dir() -> None:
    data = _load()
    data.pop("base_dir", None)
    _save(data)


def resolve_dir(name: str, explicit=None) -> Path:
    """Explicit path wins; otherwise `<base>/<name>` if configured; otherwise `<name>` in cwd."""
    if explicit:
        return Path(explicit).expanduser()
    base = base_dir()
    return (base / name) if base else Path(name)


def runs_dir() -> Path:
    return resolve_dir("runs")


def data_dir() -> Path:
    return resolve_dir("data")
