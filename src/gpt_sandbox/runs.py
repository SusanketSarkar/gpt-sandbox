"""A run = a directory: config.yaml, run.json, metrics.jsonl, ckpt_{last,best}.pt, eval.json, plugins/, tokenizer.json."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Cfg, dump_config, load_config, wrap
from .data.dataset import Dataset
from .data.tokenizers import tokenizer_from_dict
from .model.gpt import GPT
from .plugins import load_plugins
from .utils import amp_factory, get_device, read_json, write_json


def is_run(p: Path) -> bool:
    return (Path(p) / "run.json").exists()


def find_runs(paths) -> list[Path]:
    """Expand dirs (recursively) into run dirs; no paths -> everything under ./runs."""
    out = []
    for p in paths or ["runs"]:
        p = Path(p)
        if is_run(p):
            out.append(p)
        elif p.is_dir():
            out += sorted(q.parent for q in p.rglob("run.json"))
    return out


@dataclass
class LoadedRun:
    run_dir: Path
    cfg: Cfg
    model: GPT
    tokenizer: object
    step: int
    device: torch.device
    amp: object
    info: dict

    def dataset(self, name: str | None = None, data_dir: str | None = None) -> Dataset:
        name = name or self.cfg.data.name
        dd = data_dir or self.info.get("data_path_dir") or self.cfg.data.dir
        ds = Dataset.load(name, dd)
        return ds.retokenized(self.tokenizer)


def load_run(run_dir, ckpt: str = "best", device: str = "auto", load_plugin_snapshot: bool = True) -> LoadedRun:
    run_dir = Path(run_dir)
    if not is_run(run_dir):
        raise FileNotFoundError(f"{run_dir} is not a run directory (no run.json). Train one with `gpt train`.")
    if load_plugin_snapshot and (run_dir / "plugins").exists():
        load_plugins([run_dir / "plugins"])  # the exact component code this run was trained with
    cfg = load_config(run_dir / "config.yaml")
    info = read_json(run_dir / "run.json")
    f = run_dir / f"ckpt_{ckpt}.pt"
    if not f.exists():
        alt = run_dir / "ckpt_last.pt"
        if not alt.exists():
            raise FileNotFoundError(f"no checkpoint in {run_dir}")
        print(f"  (ckpt_{ckpt}.pt not found, using ckpt_last.pt)")
        f = alt
    state = torch.load(f, map_location="cpu", weights_only=False)
    dev = get_device(device)
    model = GPT(cfg.model)
    model.load_state_dict(state["model"])
    model.to(dev).eval()
    tok = tokenizer_from_dict(read_json(run_dir / "tokenizer.json"))
    amp, _ = amp_factory(dev, cfg.train.precision)
    return LoadedRun(run_dir, cfg, model, tok, state.get("step", -1), dev, amp, info)


def read_metrics(run_dir) -> list[dict]:
    f = Path(run_dir) / "metrics.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
