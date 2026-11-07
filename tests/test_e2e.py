"""End-to-end smoke tests: curate -> check -> train -> eval -> query, plus every plugin scaffold."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gpt_sandbox import cli

TINY = {
    "data": {"name": "t"}, "model": {"n_layer": 2, "d_model": 32, "n_head": 2, "seq_len": 32},
    "optim": {"lr": 3e-3}, "train": {"max_steps": 20, "batch_size": 8, "eval_interval": 10, "log_interval": 10,
                                       "warmup_steps": 2, "eval_batches": 2, "save_interval": 10, "device": "cpu"},
    "eval": {"batch_size": 8, "task_samples": 10, "gen_samples": 2},
}


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "plugins").mkdir()
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(TINY))
    (tmp_path / "d.yaml").write_text(yaml.safe_dump({
        "name": "t", "sources": [{"name": "synthetic", "task": "recall", "n_docs": 60, "lines_per_doc": 16}],
        "tokenizer": "char", "dedup": False}))
    cli.main(["data", "curate", "-c", "d.yaml"])
    return tmp_path


def test_full_pipeline(ws, capsys):
    cli.main(["data", "inspect", "t"])
    cli.main(["check", "-c", "tiny.yaml"])
    cli.main(["train", "-c", "tiny.yaml", "-n", "r1"])
    assert (ws / "runs/r1/eval.json").exists()
    import json
    m = json.loads((ws / "runs/r1/eval.json").read_text())["metrics"]
    for k in ("val_loss", "val_bpb", "induction_gap", "task_em", "base_bigram", "params_total", "fwdbwd_tok_per_s"):
        assert k in m, k
    cli.main(["eval", "runs/r1", "--deep", "--max-batches", "2"])
    cli.main(["query", "runs/r1", "-p", "ab1?a>", "-m", "5", "--seed", "0"])
    cli.main(["query", "runs/r1", "-p", "ab1?a>", "--next", "5"])
    cli.main(["query", "runs/r1", "-p", "ab1?a>1", "--score"])
    cli.main(["compare", "runs"])
    cli.main(["train", "--resume", "runs/r1", "--set", "train.max_steps=30"])


def test_config_variants_via_set(ws):
    cfgs = ["model.attention={name: mha, n_head: 1}", "model.mlp={name: gelu}", "model.posenc={name: learned}",
            "model.norm={name: layernorm}",
            "model.attention=[{layers: '::2', name: mha, n_head: 1}, {layers: all, name: mha}]"]
    for s in cfgs:
        cli.main(["check", "-c", "tiny.yaml", "--set", s])


@pytest.mark.parametrize("kind", ["attention", "mlp", "norm", "posenc", "residual", "insert", "block"])
def test_scaffolds_pass_check(ws, kind):
    cli.main(["new", kind, "myx"])
    key = {"attention": "model.attention", "mlp": "model.mlp", "norm": "model.norm", "posenc": "model.posenc",
           "residual": "model.residual", "block": "model.block"}.get(kind)
    s = "model.inserts=[{site: block_end, name: myx}]" if kind == "insert" else f"{key}={{name: myx}}"
    cli.main(["check", "-c", "tiny.yaml", "--set", s])


def test_causality_check_catches_leaks(ws):
    (ws / "plugins/leaky.py").write_text('''
import torch, torch.nn as nn
from gpt_sandbox import register
@register("attention", "leaky")
class Leaky(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__(); self.p = nn.Linear(cfg.d_model, cfg.d_model)
    def forward(self, x, ctx):
        return self.p(x.mean(1, keepdim=True).expand_as(x))   # mixes in future tokens
''')
    with pytest.raises(SystemExit) as e:
        cli.main(["check", "-c", "tiny.yaml", "--set", "model.attention={name: leaky}"])
    assert e.value.code == 1


def test_insert_gate_experiment_and_snapshot(ws):
    (ws / "plugins/gate.py").write_text('''
import torch, torch.nn as nn
from gpt_sandbox import register
@register("insert", "gate")
class Gate(nn.Module):
    def __init__(self, cfg, layer_idx, site, init=0.0):
        super().__init__(); self.g = nn.Parameter(torch.tensor(float(init)))
    def forward(self, x, ctx):
        ctx.log(f"gate/L{self.layer_idx}", self.g)
        return x * (1 + self.g)
''')
    cli.main(["train", "-c", "tiny.yaml", "-n", "g", "--set", "model.inserts=[{site: block_end, name: gate, init: 0.1}]",
              "optim.groups=[{match: 'gate', lr_mult: 5, weight_decay: 0}]"])
    assert (ws / "runs/g/plugins/gate.py").exists()
    (ws / "plugins/gate.py").unlink()  # eval must still work from the snapshot
    import gpt_sandbox.registry as r
    r._REG["insert"].pop("gate")
    cli.main(["eval", "runs/g", "--metrics", "val"])


def test_sweep_and_bad_override(ws):
    (ws / "sw.yaml").write_text(yaml.safe_dump({"base": "tiny.yaml", "seeds": [0, 1], "common": {"train.max_steps": 10},
                                                 "variants": {"a": {}, "b": {"model.mlp": {"name": "gelu"}}}}))
    cli.main(["sweep", "sw.yaml"])
    assert (ws / "runs/sw/summary.txt").exists()
    with pytest.raises(SystemExit):
        cli.main(["check", "-c", "tiny.yaml", "--set", "model.n_layers=3"])


def test_spec_override_replaces_not_merges(ws):
    from gpt_sandbox.config import load_config

    cfg = load_config("tiny.yaml", ["train.schedule={name: constant}", "model.attention={name: mha, n_head: 1}"])
    assert dict(cfg.train.schedule) == {"name": "constant"}
    assert cfg.model.attention.n_head == 1
    cfg = load_config("tiny.yaml", ["train.schedule.min_lr_ratio=0.5"])  # same name -> merge
    assert cfg.train.schedule.min_lr_ratio == 0.5 and cfg.train.schedule.name == "cosine"
