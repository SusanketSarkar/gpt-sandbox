"""Test-time sensitivity profile: how sensitive is a trained model's val_bpb to a random
Gaussian perturbation of each slice of the residual stream, injected at EVAL time only
(plugin stream_noise_eval, a no-op during training).

No retraining: takes a trained plain run, loads its checkpoint, and re-evaluates it with
one insert at a time. For every layer L and sigma in {0.1, 0.3, 1.0}:

    d(L, sigma) = val_bpb(plain model + eval-noise at L) - val_bpb(plain model)

This is the dependency-side view of "what is this slice of the stream carrying": if the
final predictions depend strongly on the middle of the stream, the middle should be
sensitive. Contrast with the training-time fragility (main study): a flat test-time
profile with a non-flat training profile means the training effect is adaptation /
regularisation, not static importance; matching shapes mean the same layers matter in
both senses.

Usage:
    python experiments/depth_fragility/sensitivity.py RUN_DIR [RUN_DIR ...] \
        [--sigmas 0.1,0.3,1.0] [--device cuda:0] [--out experiments/depth_fragility/sensitivity.json]

Output: JSON {run_dir: {plain: {...}, curve: {L<i>: {sig<SS>: {val_loss, val_bpb, delta}}}}}
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from gpt_sandbox import plugins as P
from gpt_sandbox.evals.runner import evaluate
from gpt_sandbox.model.gpt import GPT
from gpt_sandbox.plugins import load_plugins
from gpt_sandbox.runs import load_run

REPO = Path(__file__).resolve().parents[2]


def flat(groups) -> dict:
    out = {}
    for g in groups.values():
        out.update(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sigmas", default="0.1,0.3,1.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "sensitivity.json"))
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",") if s]

    # load the eval-noise plugin (distinct registry name; never collides with run snapshots)
    load_plugins([REPO / "plugins" / "stream_noise_eval.py"])

    result = {}
    for rd in args.runs:
        run_dir = Path(rd)
        lr = load_run(run_dir, device=args.device)
        n = lr.cfg.model.n_layer
        ds = lr.dataset()
        print(f"== {run_dir.name} (n_layer={n}) ==")
        base_groups, _ = evaluate(lr.model, ds, lr.cfg, lr.device, lr.amp,
                                 metrics=["val"], quiet=True)
        base = {k: flat(base_groups).get(k) for k in ("val_loss", "val_bpb", "val_ppl")}
        print(f"plain: val_loss={base['val_loss']:.4f} val_bpb={base['val_bpb']:.4f}")

        base_state = lr.model.state_dict()
        curve = {}
        cfg0 = lr.cfg
        for L in range(n):
            for s in sigmas:
                from gpt_sandbox.config import Cfg
                cfg2 = Cfg(cfg0.model)
                cfg2["inserts"] = [{"site": "after_attn", "name": "stream_noise_eval",
                                    "layers": L, "sigma": s}]
                m2 = GPT(cfg2)
                m2.load_state_dict(base_state)
                m2.to(lr.device).eval()
                g, _ = evaluate(m2, ds, cfg0, lr.device, lr.amp,
                               metrics=["val"], quiet=True)
                v = {k: flat(g).get(k) for k in ("val_loss", "val_bpb")}
                key = f"sig{int(s*100)}"
                curve.setdefault(f"L{L}", {})[key] = {
                    "val_loss": v["val_loss"], "val_bpb": v["val_bpb"],
                    "delta_bpb": v["val_bpb"] - base["val_bpb"],
                    "delta_loss": v["val_loss"] - base["val_loss"],
                }
                del m2
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"  L{L}: " + "  ".join(f"{k}={c['delta_bpb']:+.4f}" for k, c in sorted(curve[f'L{L}'].items())))
        result[str(run_dir)] = {
            "n_layer": n,
            "plain": base,
            "sigmas": sigmas,
            "curve": curve,
        }
    save_merged(args.out, result)
    print(f"wrote {args.out}")


def save_merged(path, new_entries):
    """Merge new_entries into the existing JSON with an exclusive lock + atomic replace,
    so concurrent sensitivity runs never clobber each other."""
    import fcntl
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            existing = {}
            if path.exists():
                try:
                    existing = json.loads(path.read_text())
                except Exception as e:
                    print(f"warn: existing {path} unparseable ({e}); starting fresh")
            existing.update(new_entries)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(existing, f, indent=2)
            os.replace(tmp, str(path))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
