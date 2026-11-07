"""Command line: `gpt <command>`. See README for the 60-second tour.

  gpt data curate|inspect|list|retokenize     build and look at datasets (once)
  gpt info / check -c CONFIG                  inspect a model config / run contract tests on it
  gpt train -c CONFIG -n NAME [--set k=v ..]  train
  gpt eval RUN [--deep] [--data NAME]         a bunch of numbers for a trained model
  gpt query RUN --prompt "..."                sample / next-token probs / score text
  gpt compare [RUN ...]  |  gpt plot RUN ...  tables and curves across runs
  gpt sweep FILE.yaml                         many variants x seeds, one table
  gpt adddir [PATH]                           store where data/ and runs/ live (--reset)
  gpt new KIND NAME                           scaffold a plugin component
  gpt list                                    every registered component
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _add_set(p):
    p.add_argument("--set", "-s", nargs="+", default=[], metavar="KEY=VAL",
                   help="override config values, e.g. --set model.n_layer=8 'model.attention={name: gqa}'")


def _bootstrap_config(args):
    """Load config, then plugins (cwd ./plugins + cfg.plugins), then return cfg."""
    from . import components  # noqa: F401
    from .config import load_config
    from .plugins import default_plugin_paths, load_plugins

    cfg = load_config(args.config, args.set)
    load_plugins(default_plugin_paths(cfg.plugins))
    return cfg


def _load_run(args, **kw):
    from .runs import load_run

    return load_run(args.run, ckpt=args.ckpt, device=args.device, **kw)


# ----------------------------------------------------------------------------------------- data
def cmd_data(args):
    from .config import apply_overrides
    from .data import curate as cur
    from .data.dataset import Dataset, list_datasets
    from .plugins import default_plugin_paths, load_plugins
    from .settings import resolve_dir
    import yaml

    data_dir = resolve_dir("data", args.data_dir)
    load_plugins(default_plugin_paths())
    if args.dcmd == "curate":
        spec = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
        spec = apply_overrides(spec, args.set)
        if args.name:
            spec["name"] = args.name
        if args.files:
            spec["sources"] = [{"name": "files", "paths": args.files, "doc_split": args.doc_split}]
        if args.tokenizer:
            spec["tokenizer"] = args.tokenizer
        if args.val_fraction is not None:
            spec["val_fraction"] = args.val_fraction
        load_plugins(spec.get("plugins", []))
        spec.setdefault("name", Path(args.config).stem if args.config else None)
        if not spec.get("dir"):
            spec["dir"] = str(data_dir)
        cur.curate(spec, overwrite=args.overwrite)
    elif args.dcmd == "inspect":
        from .data.inspect_data import inspect_dataset

        inspect_dataset(Dataset.load(args.name, data_dir), args.samples, args.window, args.tokens)
    elif args.dcmd == "list":
        from .utils import table

        rows = []
        for n in list_datasets(data_dir):
            d = Dataset.load(n, data_dir)
            m = d.meta
            rows.append([n, m["tokenizer"]["spec"].get("name"), m["tokenizer"]["vocab_size"],
                         f"{m['splits']['train']['tokens']:,}", f"{m['splits']['val']['tokens']:,}"])
        print(table(rows, ["dataset", "tokenizer", "vocab", "train tokens", "val tokens"]) if rows else "no datasets yet")
    elif args.dcmd == "retokenize":
        cur.retokenize(args.src, args.new_name, args.tokenizer, data_dir, args.overwrite)


# ----------------------------------------------------------------------------------------- model
def cmd_info(args):
    from .check import _vocab
    from .model.gpt import GPT
    from .report import info

    cfg = _bootstrap_config(args)
    cfg.model.vocab_size = _vocab(cfg)
    print(info(GPT(cfg.model), cfg))


def cmd_check(args):
    from .check import run_checks

    cfg = _bootstrap_config(args)
    if not run_checks(cfg, args.device):
        sys.exit(1)


def cmd_train(args):
    from .train.trainer import train

    def set_device(cfg):
        if args.device:
            cfg.train.device = args.device
        if args.gpu is not None:
            cfg.train.device = f"cuda:{args.gpu}"

    if args.resume:
        from .config import load_config
        from .plugins import load_plugins

        run_dir = Path(args.resume)
        load_plugins([run_dir / "plugins"] if (run_dir / "plugins").exists() else [])
        cfg = load_config(run_dir / "config.yaml", args.set)
        set_device(cfg)
        train(cfg, run_dir, resume=True)
        return
    cfg = _bootstrap_config(args)
    name = args.name or cfg.name or f"{Path(args.config).stem if args.config else 'run'}-{time.strftime('%m%d-%H%M%S')}"
    set_device(cfg)
    train(cfg, Path(cfg.runs_dir) / name, overwrite=args.overwrite, group=name)


def cmd_eval(args):
    from .evals.runner import evaluate, print_report, save_eval

    lr = _load_run(args)
    ds = lr.dataset(args.data, args.data_dir)
    print(f"evaluating {lr.run_dir} (step {lr.step}) on '{ds.name}'"
          + ("  [val re-tokenized with this run's tokenizer]" if ds.meta.get("retokenized") else ""))
    groups, notes = evaluate(lr.model, ds, lr.cfg, lr.device, lr.amp,
                             metrics=args.metrics.split(",") if args.metrics else None, deep=args.deep,
                             batch_size=args.batch_size, max_batches=args.max_batches)
    print_report(groups, f"eval: {lr.run_dir.name}")
    if notes.get("samples"):
        print("sample generations:")
        for s in notes["samples"][:2]:
            print("  | " + s[:300].replace("\n", "\n  | ") + "\n")
    if not args.no_save and (args.data is None) and not args.metrics:
        save_eval(lr.run_dir, groups, lr.step, ds.name, args.ckpt, notes)
        print(f"saved {lr.run_dir / 'eval.json'}")


def cmd_query(args):
    import torch

    from .evals.generate import generate

    lr = _load_run(args)
    tok, model, dev = lr.tokenizer, lr.model, lr.device

    def encode(prompt):
        ids = tok.encode(prompt)
        if args.bos or not ids:
            ids = [tok.eos_id] + ids
        return torch.tensor([ids], device=dev)

    def sample(prompt, n, max_new, temp, top_k, top_p):
        g = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed is not None else None
        x = encode(prompt).repeat(n, 1)
        out = generate(model, x, max_new, temp, top_k, top_p, amp=lr.amp, generator=g if dev.type == "cpu" else None)
        for i, row in enumerate(out):
            plen = x.size(1) - (1 if (args.bos or not prompt) else 0)
            text = tok.decode(row[1 if (args.bos or not prompt) else 0 :].tolist())
            head, tail = text[: len(prompt)], text[len(prompt) :]
            print(f"--- sample {i + 1}/{n} ---\n\033[1m{head}\033[0m{tail}" if sys.stdout.isatty() else f"--- sample {i + 1}/{n} ---\n{head}{tail}")

    def nextk(prompt, k):
        with torch.no_grad(), lr.amp():
            lg = model(encode(prompt)[:, -model.cfg.seq_len :]).logits[0, -1].float()
        p = torch.softmax(lg / max(args.temperature, 1e-6), -1)
        top = p.topk(k)
        print(f"next-token distribution after {prompt!r}:")
        for pr, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {pr:7.2%}  {tok.token_str(i)}")

    def score(text):
        ids = [tok.eos_id] + tok.encode(text)
        ids = ids[-model.cfg.seq_len :]
        x = torch.tensor([ids], device=dev)
        with torch.no_grad(), lr.amp():
            lp = torch.log_softmax(model(x[:, :-1]).logits[0].float(), -1)
        nll = -lp.gather(-1, x[0, 1:, None])[:, 0]
        for t, l in zip(ids[1:], nll.tolist()):
            print(f"  {l:6.2f}  {'█' * min(40, int(l * 4)):<40} {tok.token_str(t)}")
        print(f"mean nll {nll.mean():.3f} nats/token   ppl {nll.mean().exp():.2f}   ({len(nll)} tokens)")

    def once(prompt):
        if args.next:
            nextk(prompt, args.next)
        elif args.score:
            score(prompt)
        else:
            sample(prompt, args.n, args.max_new_tokens, args.temperature, args.top_k, args.top_p)

    if args.interactive:
        print("interactive: type a prompt. commands: :temp 0.8  :k 40  :p 0.9  :n 3  :len 200  :next 10  :quit")
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.startswith(":"):
                cmd, *rest = line[1:].split()
                if cmd in ("q", "quit"):
                    break
                v = rest[0] if rest else None
                mapping = {"temp": ("temperature", float), "k": ("top_k", int), "p": ("top_p", float),
                           "n": ("n", int), "len": ("max_new_tokens", int), "next": ("next", int)}
                if cmd in mapping and v:
                    setattr(args, mapping[cmd][0], mapping[cmd][1](v))
                continue
            once(line.replace("\\n", "\n"))
    elif args.file:
        for ln in Path(args.file).read_text().split("\n\n\n" if "\n\n\n" in Path(args.file).read_text() else "\n"):
            if ln.strip():
                print(f"\n##### {ln[:60]!r}")
                once(ln)
    else:
        once((args.prompt or "").replace("\\n", "\n"))


# ----------------------------------------------------------------------------------------- reports
def cmd_adddir(args):
    from .settings import base_dir, reset_base_dir, set_base_dir

    if args.reset:
        reset_base_dir()
        print(f"results base dir reset to the current directory ({Path.cwd()})")
    elif args.path:
        print(f"results base dir set to {set_base_dir(args.path)}")
    else:
        base = base_dir()
        print(f"results base dir: {base}" if base else f"results base dir: {Path.cwd()} (default)")


def cmd_compare(args):
    from .report import compare
    from .settings import runs_dir

    print(compare(args.runs or [str(runs_dir())], args.cols.split(",") if args.cols else None, not args.no_agg))


def cmd_plot(args):
    from .report import plot

    print("wrote", plot(args.runs, args.keys.split(","), args.out, args.x))


def cmd_sweep(args):
    from .config import load_config
    from .plugins import default_plugin_paths, load_plugins
    from .sweep import run_sweep

    load_plugins(default_plugin_paths())
    run_sweep(args.file, args.force, args.only.split(",") if args.only else None, args.device)


def cmd_new(args):
    from .scaffold import new_plugin

    new_plugin(args.kind, args.name, args.dir)


def cmd_list(args):
    from . import components  # noqa: F401
    from .data import filters, sources, synthetic, tokenizers  # noqa: F401
    from .evals import metrics  # noqa: F401
    from .plugins import default_plugin_paths, load_plugins
    from .report import list_components
    from .train import optim, schedules  # noqa: F401

    load_plugins(default_plugin_paths())
    print(list_components())


# ----------------------------------------------------------------------------------------- parser
def build_parser():
    P = argparse.ArgumentParser(prog="gpt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = P.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="curate / inspect / list / retokenize datasets")
    ds = d.add_subparsers(dest="dcmd", required=True)
    c = ds.add_parser("curate", help="sources -> filters -> dedup -> split -> tokenize -> data/<name>")
    c.add_argument("--config", "-c")
    c.add_argument("--name", "-n")
    c.add_argument("--files", nargs="+", help="quick path: glob(s) of text files")
    c.add_argument("--doc-split", default="whole", help="whole | blank | lines | regex:<pat> (with --files)")
    c.add_argument("--tokenizer", "-t", help="char | byte | gpt2 | bpe:4096")
    c.add_argument("--val-fraction", type=float)
    c.add_argument("--overwrite", action="store_true")
    _add_set(c)
    i = ds.add_parser("inspect", help="stats, vocab, doc lengths, baselines, leakage, samples")
    i.add_argument("name")
    i.add_argument("--samples", type=int, default=2)
    i.add_argument("--window", type=int, default=300)
    i.add_argument("--tokens", action="store_true", help="also show token ids")
    ds.add_parser("list")
    r = ds.add_parser("retokenize", help="re-encode a dataset's cleaned docs with another tokenizer")
    r.add_argument("src")
    r.add_argument("new_name")
    r.add_argument("--tokenizer", "-t", required=True)
    r.add_argument("--overwrite", action="store_true")
    for sp in (c, i, ds.choices["list"], r):
        sp.add_argument("--data-dir", default=None)
    d.set_defaults(fn=cmd_data)

    for name, fn, h in (("info", cmd_info, "show the resolved model: per-layer components, params"),
                        ("check", cmd_check, "contract tests: shapes, causality, grads, overfit-one-batch")):
        p = sub.add_parser(name, help=h)
        p.add_argument("--config", "-c")
        p.add_argument("--device", default="cpu")
        _add_set(p)
        p.set_defaults(fn=fn)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("--config", "-c")
    t.add_argument("--name", "-n")
    t.add_argument("--device")
    t.add_argument("--gpu", type=int, metavar="N", help="use cuda:N, e.g. --gpu 0")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--resume", metavar="RUN_DIR")
    _add_set(t)
    t.set_defaults(fn=cmd_train)

    def run_args(p):
        p.add_argument("run")
        p.add_argument("--ckpt", default="best", choices=["best", "last"])
        p.add_argument("--device", default="auto")

    e = sub.add_parser("eval", help="evaluate a run: a bunch of quality numbers")
    run_args(e)
    e.add_argument("--data", help="evaluate on another curated dataset's val split (re-tokenized)")
    e.add_argument("--data-dir")
    e.add_argument("--metrics", help="comma list, e.g. val,induction (default: eval.metrics from the config)")
    e.add_argument("--deep", action="store_true", help="add logit_lens + layer_ablation")
    e.add_argument("--batch-size", type=int)
    e.add_argument("--max-batches", type=int)
    e.add_argument("--no-save", action="store_true")
    e.set_defaults(fn=cmd_eval)

    q = sub.add_parser("query", help="generate / inspect next-token probs / score text")
    run_args(q)
    q.add_argument("--prompt", "-p", default="")
    q.add_argument("--file", help="prompts file (one per line)")
    q.add_argument("--interactive", "-i", action="store_true")
    q.add_argument("--max-new-tokens", "-m", type=int, default=200)
    q.add_argument("--temperature", "-T", type=float, default=0.8)
    q.add_argument("--top-k", type=int, default=None)
    q.add_argument("--top-p", type=float, default=None)
    q.add_argument("--n", type=int, default=1)
    q.add_argument("--seed", type=int, default=None)
    q.add_argument("--bos", action="store_true", help="prepend <eos> (document start)")
    q.add_argument("--next", type=int, metavar="K", help="show the top-K next-token probabilities instead of sampling")
    q.add_argument("--score", action="store_true", help="per-token loss of the prompt text")
    q.set_defaults(fn=cmd_query)

    cp = sub.add_parser("compare", help="table of eval numbers across runs (seeds aggregated)")
    cp.add_argument("runs", nargs="*")
    cp.add_argument("--cols")
    cp.add_argument("--no-agg", action="store_true")
    cp.set_defaults(fn=cmd_compare)

    pl = sub.add_parser("plot", help="training curves (needs matplotlib)")
    pl.add_argument("runs", nargs="+")
    pl.add_argument("--keys", default="loss,val_loss")
    pl.add_argument("--x", default="step", choices=["step", "tokens"])
    pl.add_argument("--out", default="plot.png")
    pl.set_defaults(fn=cmd_plot)

    sw = sub.add_parser("sweep", help="run variants x seeds from a yaml file")
    sw.add_argument("file")
    sw.add_argument("--force", action="store_true")
    sw.add_argument("--only")
    sw.add_argument("--device")
    sw.set_defaults(fn=cmd_sweep)

    ad = sub.add_parser("adddir", help="set the base directory for data/ and runs/")
    ad.add_argument("path", nargs="?", help="directory to store data/ and runs/ (show current if omitted)")
    ad.add_argument("--reset", action="store_true", help="go back to using the current directory")
    ad.set_defaults(fn=cmd_adddir)

    n = sub.add_parser("new", help="scaffold plugins/<name>.py for a component kind")
    n.add_argument("kind")
    n.add_argument("name")
    n.add_argument("--dir", default="plugins")
    n.set_defaults(fn=cmd_new)

    sub.add_parser("list", help="list all registered components").set_defaults(fn=cmd_list)
    return P


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, TypeError) as e:
        if "--debug" in sys.argv or __import__("os").environ.get("GPT_DEBUG"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {msg}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
