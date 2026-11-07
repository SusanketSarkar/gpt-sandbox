"""Plugin loading: any ``*.py`` in a plugin dir is imported so its ``@register`` calls run."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

LOADED_DIRS: list[Path] = []
_LOADED_FILES: set[Path] = set()


def _import_file(path: Path) -> None:
    path = path.resolve()
    if path in _LOADED_FILES:
        return
    mod_name = f"gpt_sandbox_plugins.{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # make plugin errors point at the file
        raise RuntimeError(f"error while importing plugin {path}: {type(e).__name__}: {e}") from e
    _LOADED_FILES.add(path)


def load_plugins(paths) -> list[Path]:
    """Import plugin files/dirs. Files starting with ``_`` are helpers (importable, not auto-run)."""
    loaded = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_file():
            files, d = [p], p.parent
        elif p.is_dir():
            files, d = sorted(f for f in p.glob("*.py") if not f.name.startswith("_")), p
        else:
            continue
        d = d.resolve()
        if d not in LOADED_DIRS:
            LOADED_DIRS.append(d)
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))  # so plugins can `import _helpers`
        for f in files:
            _import_file(f)
            loaded.append(f)
    return loaded


def default_plugin_paths(extra=()) -> list[Path]:
    paths = [Path("plugins")]
    env = os.environ.get("GPT_SANDBOX_PLUGINS")
    if env:
        paths += [Path(x) for x in env.split(os.pathsep) if x]
    paths += [Path(x) for x in extra]
    return paths


def snapshot_plugins(dest: Path) -> int:
    """Copy every loaded plugin dir's .py files into ``dest`` so a run stays reproducible."""
    n = 0
    dest.mkdir(parents=True, exist_ok=True)
    for d in LOADED_DIRS:
        if d == dest.resolve():
            continue
        for f in d.glob("*.py"):
            shutil.copy2(f, dest / f.name)
            n += 1
    return n
