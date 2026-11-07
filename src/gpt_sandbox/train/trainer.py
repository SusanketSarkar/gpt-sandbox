"""The training loop. Deliberately plain: read it top to bottom, change it freely."""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..config import dump_config, unwrap
from ..data.dataset import Dataset
from ..evals.runner import evaluate, flatten, print_report, save_eval
from ..model.gpt import GPT
from ..plugins import snapshot_plugins
from ..registry import get
from ..utils import amp_factory, get_device, human, now_iso, seed_everything, sync, write_json
from . import schedules  # noqa: F401
from .optim import build_optimizer


def _save_ckpt(path: Path, model, optimizer, step, best, cfg, with_opt=True):
    state = {"model": model.state_dict(), "step": step, "best_val": best, "cfg": unwrap(cfg)}
    if with_opt:
        state["optim"] = optimizer.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def quick_eval(model, ds, cfg, device, amp) -> dict:
    """Cheap fixed-window val loss (same windows every time and for every run) + a train-loss estimate."""
    model.eval()
    T, B, nb = cfg.model.seq_len, cfg.eval.batch_size, cfg.train.eval_batches
    tot = n = 0
    for x, y in ds.chunks("val", T, B, nb):
        with amp():
            tot += model(x.to(device), y.to(device)).ce.item() * x.size(0)
        n += x.size(0)
    model.train()
    return {"val_loss": tot / max(1, n)}


def train(cfg, run_dir: Path, resume: bool = False, overwrite: bool = False, group: str | None = None,
          sweep: str | None = None) -> Path:
    run_dir = Path(run_dir)
    if run_dir.exists() and not resume:
        if not overwrite:
            raise FileExistsError(f"{run_dir} exists. Use --overwrite, --resume, or another --name.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tc, mc = cfg.train, cfg.model
    device = get_device(tc.device)
    amp, prec = amp_factory(device, tc.precision)
    seed_everything(cfg.seed)

    ds = Dataset.load(cfg.data.name, cfg.data.dir)
    mc.vocab_size = ds.vocab_size
    model = GPT(mc).to(device)
    optimizer = build_optimizer(model, cfg.optim)
    sched = get("schedule", cfg.train.schedule["name"])
    sched_kw = {k: v for k, v in cfg.train.schedule.items() if k != "name"}

    step, best = 0, float("inf")
    if resume:
        st = torch.load(run_dir / "ckpt_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optim"])
        step, best = st["step"], st["best_val"]
        print(f"resumed from step {step}")
    else:
        dump_config(cfg, run_dir / "config.yaml")
        shutil.copy(ds.path / "tokenizer.json", run_dir / "tokenizer.json")
        n_plug = snapshot_plugins(run_dir / "plugins")
        if not n_plug:
            shutil.rmtree(run_dir / "plugins", ignore_errors=True)
    run_info = {
        "name": run_dir.name, "group": group or run_dir.name, "sweep": sweep, "seed": cfg.seed,
        "started": now_iso(), "status": "running", "data": cfg.data.name,
        "data_path_dir": str(Path(cfg.data.dir).resolve()), "device": str(device), "precision": prec,
        "params": model.num_params(), "params_nonemb": model.num_params(non_embedding=True),
    }
    write_json(run_dir / "run.json", run_info)

    fwd = torch.compile(model) if tc.compile else model
    tokens_per_step = tc.batch_size * tc.grad_accum * mc.seq_len
    print(f"run {run_dir} | device={device} {prec} | params {human(run_info['params'])} "
          f"({human(run_info['params_nonemb'])} non-emb) | {human(tokens_per_step)} tok/step | "
          f"{human(tokens_per_step * tc.max_steps)} tokens total")

    mfile = open(run_dir / "metrics.jsonl", "a")

    def log(rec):
        mfile.write(json.dumps(rec) + "\n")
        mfile.flush()

    model.train()
    t_last, tok_last = time.time(), 0
    try:
        while step < tc.max_steps:
            mult = sched(step, tc.max_steps, tc.warmup_steps, **sched_kw)
            for g in optimizer.param_groups:
                g["lr"] = cfg.optim.lr * g.get("lr_mult", 1.0) * mult
            is_log = (step + 1) % tc.log_interval == 0 or step == 0
            loss_acc = ce_acc = 0.0
            aux_acc: dict = {}
            stats: dict = {}
            for micro in range(tc.grad_accum):
                rng = np.random.default_rng([cfg.seed, step, micro])
                x, y = ds.get_batch("train", tc.batch_size, mc.seq_len, rng)
                x, y = x.to(device), y.to(device)
                with amp():
                    out = fwd(x, y, collect=is_log and micro == tc.grad_accum - 1)
                (out.loss / tc.grad_accum).backward()
                loss_acc += out.loss.item() / tc.grad_accum
                ce_acc += out.ce.item() / tc.grad_accum
                for k, v in out.aux.items():
                    aux_acc[k] = aux_acc.get(k, 0.0) + float(v) / tc.grad_accum
                if micro == tc.grad_accum - 1:
                    stats = {k: float(v) for k, v in out.ctx.stats.items()}
            if not math.isfinite(loss_acc):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss_acc}. "
                                         f"Try a lower optim.lr, more warmup, or run `gpt check`.")
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)) if tc.grad_clip else float("nan")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_log:
                sync(device)
                now = time.time()
                tps = (step - tok_last) * tokens_per_step / max(1e-9, now - t_last) if step > tok_last else 0
                t_last, tok_last = now, step
                rec = {"step": step, "tokens": step * tokens_per_step, "split": "train", "loss": loss_acc, "ce": ce_acc,
                       "lr": cfg.optim.lr * mult, "gnorm": gnorm, "tok_per_s": tps,
                       **{f"aux/{k}": v for k, v in aux_acc.items()}, **stats}
                log(rec)
                extra = "".join(f" | {k} {v:.4g}" for k, v in {**{f'aux/{k}': v for k, v in aux_acc.items()}}.items())
                print(f"step {step:>6}/{tc.max_steps} | loss {ce_acc:.4f} | lr {cfg.optim.lr * mult:.2e} | "
                      f"gnorm {gnorm:.2f} | {human(tps)} tok/s{extra}")
            if step % tc.eval_interval == 0 or step == tc.max_steps:
                ev = quick_eval(model, ds, cfg, device, amp)
                log({"step": step, "tokens": step * tokens_per_step, "split": "val", **ev})
                mark = ""
                if ev["val_loss"] < best:
                    best = ev["val_loss"]
                    _save_ckpt(run_dir / "ckpt_best.pt", model, optimizer, step, best, cfg, with_opt=False)
                    mark = "  *best*"
                print(f"step {step:>6} | VAL loss {ev['val_loss']:.4f}{mark}")
            if step % tc.save_interval == 0 or step == tc.max_steps:
                _save_ckpt(run_dir / "ckpt_last.pt", model, optimizer, step, best, cfg)
    except BaseException as e:
        run_info.update(status="failed" if not isinstance(e, KeyboardInterrupt) else "interrupted", error=repr(e))
        write_json(run_dir / "run.json", run_info)
        _save_ckpt(run_dir / "ckpt_last.pt", model, optimizer, step, best, cfg)
        raise
    finally:
        mfile.close()

    run_info.update(status="done", finished=now_iso(), steps=step, best_val=best)
    write_json(run_dir / "run.json", run_info)
    if tc.final_eval:
        print("\nrunning final eval (best checkpoint) ...")
        best_f = run_dir / "ckpt_best.pt"
        if best_f.exists():
            model.load_state_dict(torch.load(best_f, map_location=device, weights_only=False)["model"])
        groups, notes = evaluate(model, ds, cfg, device, amp, quiet=True)
        save_eval(run_dir, groups, step, ds.name, "best", notes)
        print_report(groups, f"eval: {run_dir.name}")
    return run_dir
