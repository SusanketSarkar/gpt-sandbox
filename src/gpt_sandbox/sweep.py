"""`gpt sweep file.yaml`: run several variants (x seeds), eval each, print a comparison.

    name: gqa_test               # optional (default: file stem)
    base: ../tiny.yaml
    seeds: [0, 1, 2]             # optional: repeat every variant per seed -> mean ± std in the table
    common: {train.max_steps: 400}
    variants:
      baseline: {}
      my_idea:  {model.attention: {name: my_attention}}
"""
from __future__ import annotations

import traceback
from pathlib import Path

import yaml

from .config import load_config
from .report import compare
from .train.trainer import train


def run_sweep(path, force=False, only=None, device=None):
    path = Path(path)
    spec = yaml.safe_load(path.read_text())
    name = spec.get("name") or path.stem
    base = (path.parent / spec["base"]) if not Path(spec["base"]).exists() else Path(spec["base"])
    seeds = spec.get("seeds") or [None]
    variants = spec["variants"]
    if only:
        variants = {k: v for k, v in variants.items() if k in only}
    done, failed = [], []
    for vname, ov in variants.items():
        for seed in seeds:
            ovr = {**(spec.get("common") or {}), **(ov or {})}
            if seed is not None:
                ovr["seed"] = seed
            if device:
                ovr["train.device"] = device
            cfg = load_config(base, ovr)
            run_name = f"{vname}" + (f"-s{seed}" if seed is not None else "")
            run_dir = Path(cfg.runs_dir) / name / run_name
            if (run_dir / "eval.json").exists() and not force:
                print(f"[sweep] skip {run_dir} (done; use --force to rerun)")
                done.append(run_dir)
                continue
            print(f"\n[sweep] ===== {name}/{run_name} =====")
            try:
                train(cfg, run_dir, overwrite=True, group=vname, sweep=name)
                done.append(run_dir)
            except Exception:
                traceback.print_exc()
                failed.append(run_name)
    print(f"\n[sweep] {name}: {len(done)} finished, {len(failed)} failed {failed or ''}")
    out = compare([Path(load_config(base).runs_dir) / name])
    print(out)
    (Path(load_config(base).runs_dir) / name / "summary.txt").write_text(out + "\n")
