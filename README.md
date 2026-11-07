# gpt-sandbox

A small, hackable lab for **autoregressive GPT experiments**: curate a dataset once, swap any part of the
model with a few lines of code, train, get a bunch of quality numbers, and poke the model with queries.

The starting model is a plain pre-norm GPT (RoPE, RMSNorm, SwiGLU, multi-head attention). Nothing exotic is built in
on purpose: the repo is organised so that GQA, delta-rule/linear attention, gated residuals, hybrid layer stacks,
etc. each take **one small file** and **one config line**.

```bash
pip install -e ".[data,plot,dev]"          # torch, numpy, pyyaml (+ optional datasets/tokenizers/tiktoken/matplotlib)

gpt adddir /data/gpt-exp/                   # optional: store data/ and runs/ in a separate dir instead of .

gpt data curate -c configs/data/shakespeare.yaml     # once
gpt data inspect shakespeare                          # what is in there? baselines, leakage, samples
gpt check -c configs/tiny.yaml                        # contract tests for the model config

gpt train -c configs/tiny.yaml -n base --gpu 0        # train on cuda:0 (auto-runs the full eval at the end)
gpt eval runs/base --deep                             # more numbers (logit lens, layer ablation)
gpt query runs/base -p "ROMEO:" -m 300                # sample;  --next 10 = next-token probs;  --score = per-token loss
gpt compare                                           # table over all runs
```

`gpt adddir` writes `.gpt-sandbox.json` in the current directory. Subsequent commands put `data/` and `runs/`
under that base directory; `gpt adddir --reset` (or deleting the file) goes back to the current directory.

## The 60-second mental model

Everything swappable is a **named component in a registry**. Configs pick components by name; plugin files
register them. `./plugins/*.py` is auto-imported.

| kind | contract | what you'd do with it |
|---|---|---|
| `attention` | `__init__(cfg, layer_idx, **kw)`, `forward(x, ctx) -> x` | GQA, MQA, linear / delta-rule, local, qk-norm, ... |
| `mlp` | same | GELU/SwiGLU variants, MoE, ... |
| `norm` | `__init__(dim, **kw)`, `forward(x)` | RMSNorm/LayerNorm variants |
| `posenc` | `embed(x, pos)`, `rotate(q, k, pos)` | learned, RoPE variants, ALiBi, none |
| `residual` | `forward(x, delta, ctx) -> x` | **how a sublayer output joins the stream**: gates, LayerScale, ReZero |
| `insert` | `forward(x, ctx) -> x` | **transform the residual stream at a named site** (see below) |
| `block` | subclass `BlockBase` | post-norm, parallel attn+mlp, skip connections |
| `optimizer`, `schedule` | functions | Muon-ish optimizers, WSD, ... |
| `metric` | `fn(ev) -> {name: float}` | any eval number you care about |
| `tokenizer`, `source`, `filter`, `task` | see `gpt list` | data side |

`gpt list` shows everything registered. `gpt new <kind> <name>` writes `plugins/<name>.py` as a **working copy
of the builtin** with `# >>> EDIT` markers, so you start from something that already passes `gpt check`.

### Config picks components

```yaml
model:
  n_layer: 8
  attention: {name: my_attention, some_kwarg: 3}     # kwargs go straight to your __init__
  mlp: swiglu
  inserts:
    - {site: block_end, name: my_gate, layers: all}   # attach modules to the residual stream, no model edit
```

Override anything from the command line (a value *replaces* what was at that path):

```bash
gpt train -c configs/tiny.yaml -n gqa2 --set 'model.attention={name: gqa, n_kv_head: 2}' train.max_steps=2000
```

### Per-layer / hybrid stacks

Any of `attention | mlp | norm | block | residual` can be a list:

```yaml
attention: [{name: kda}, {name: kda}, {name: kda}, {name: mha}]   # cycled over layers: 3:1 hybrid
attention:                                                         # or selectors; first match wins
  - {layers: "3::4", name: mha}      # layers 3, 7, 11, ...   (python slice strings, ints, "first", "last", "even", "odd")
  - {layers: all,    name: kda}
```

### The `ctx` object (how components talk without changing signatures)

Every `forward(x, ctx)` receives `ctx`:

* `ctx.posenc.rotate(q, k, ctx.positions)` - how attention gets RoPE
* `ctx.x0` the embedding output; `ctx.store` a dict for passing tensors between layers (value-residual, shared KV, U-Net skips)
* `ctx.log("name", tensor)` - diagnostics that land in `metrics.jsonl` on logging steps (free otherwise). Residual RMS per layer is logged automatically.
* `ctx.add_loss("name", tensor)` - auxiliary loss, added to the training loss and logged separately (sparsity penalties, balance losses)
* `ctx.layer_idx`, `ctx.training`, `ctx.cfg`

### Inserts: weird residual-stream experiments without touching the model

`model.inserts` entries attach a module at a named **site**:
`block_start | after_attn | block_end` (per layer, `layers:` selects which) and `embed | final` (model level).
"Add a gate at the end of every layer" is one tiny `nn.Module` plus:

```yaml
inserts: [{site: block_end, name: my_gate, layers: all, init: 0.0}]
```

Want the gate to train with its own learning rate / no weight decay? No code:

```yaml
optim:
  groups: [{match: "gate", lr_mult: 10, weight_decay: 0}]     # regex on parameter names, first match wins
```

If the sublayer *output* should be scaled instead of the stream, swap the `residual` component (it sits on every
`x + sublayer(x)` join). Need a different wiring entirely? Subclass `BlockBase` (`gpt new block`).

Components can define `init_weights()` (called after the default init) and mark a projection with
`self.proj._resid_out = True` to get GPT-2 depth-scaled init.

### Generation has no KV cache (on purpose)

Sampling recomputes the (cropped) context at every step, so *any* attention / recurrent / delta-rule component works
for `query` and generation metrics without writing a cache. It is slower, fine at sandbox scale.

## Data: curate once, reuse everywhere

```bash
gpt data curate -c configs/data/shakespeare.yaml
gpt data curate -n notes --files "~/notes/**/*.md" -t bpe:2048 --doc-split blank     # quick path
gpt data curate -c configs/data/synth_recall.yaml                                     # synthetic algorithmic data
gpt data retokenize shakespeare shakespeare_bpe -t bpe:1024                           # same cleaned docs, new tokenizer
gpt data list
```

Pipeline: `sources -> filters -> dedup -> hash-split by document -> fit tokenizer on train -> train.bin / val.bin`.
It stores the tokenizer, cleaned docs, and metadata next to the shards in `data/<name>/`. Each run copies the tokenizer,
so `query` works even if the dataset is gone.

* sources: `files`, `url`, `hf` (HuggingFace), `synthetic` (tasks: `modadd`, `copy`, `recall`; add your own)
* tokenizers: `char`, `byte`, `tiktoken` (gpt2), `hf_bpe` (trained on your corpus)
* filters: `min_chars`, `max_chars`, `ascii_ratio`, `alpha_ratio`, `strip`, `lowercase`, `collapse_whitespace`

`gpt data inspect NAME` prints sizes, vocab coverage, token frequencies, document-length histogram, **reference
losses** (uniform / unigram / bigram fit on train), a **train/val overlap check**, and decoded samples.

## Eval: numbers, not vibes

`gpt train` runs the full eval at the end (`runs/<name>/eval.json`); `gpt eval RUN` re-runs it any time, on any curated
dataset (`--data other`; val docs are re-tokenized with the run's tokenizer).

| group | numbers |
|---|---|
| `val` | `val_loss`, `val_ppl`, **`val_bpb`** (bits/byte - comparable across tokenizers), `val_acc1`, `val_acc5`, predictive entropy |
| `position` | loss by context position, **`ctx_gain`** (early minus late loss: how much longer context helps) |
| `baselines` | uniform/unigram/bigram val loss and `gain_vs_bigram` (did it learn anything beyond local statistics) |
| `induction` | repeated-random-token test: `induction_gap` / `induction_acc` (copy/induction ability; data-independent) |
| `calibration` | `ece`, `mean_confidence`, `overconfidence` |
| `generalization` | `train_loss_eval`, `gen_gap` |
| `generation` | distinct-1/2/3, unigram KL of samples (degeneration detector), sample text in `eval.json` |
| `task` | exact-match accuracy (`task_em`) on synthetic-task datasets |
| `throughput` | fwd and fwd+bwd tokens/s, peak memory |
| `params` | total / non-embedding params, FLOPs per token |
| `--deep` | `logit_lens` (loss reading the stream after each layer, residual RMS), `layer_ablation` (loss increase when skipping layer i) |

Validation uses fixed non-overlapping windows, identical for every run, so numbers are directly comparable.
Add your own metric with `gpt new metric NAME` (gets `ev.val_stats`: per-token nll/top-k/entropy).

## Doing experiments properly

* `gpt check -c CFG [--set ...]` - run after writing any component: shapes, **causality (future-token leak)**, batch
  independence, prefix consistency, gradient flow, init loss ~ ln V, overfit-one-batch, speed.
* `gpt info -c CFG` - per-layer table of what each layer actually uses and parameter counts (verify hybrids and selectors).
* `gpt sweep configs/sweeps/X.yaml` - variants x seeds, aggregated as mean ± std. **Run `noise_floor.yaml` first**:
  effects smaller than the seed noise are not effects. Param-count differences show up in the same table.
* Runs are reproducible: `runs/<name>/` holds the resolved config, metrics, checkpoints, eval, and a **snapshot of your
  plugin code**; `eval`/`query` load that snapshot, so editing a plugin later never breaks old runs.
* `gpt plot runs/a runs/b --keys loss,val_loss` (matplotlib), `gpt train --resume runs/a --set train.max_steps=5000`.

## Layout

```
configs/            tiny.yaml  small.yaml  recall.yaml  data/*.yaml  sweeps/*.yaml
plugins/            YOUR components (auto-imported)
src/gpt_sandbox/
  registry.py       @register / build
  config.py         defaults, base: inheritance, --set overrides, typo-catching validation
  model/            gpt.py (the whole model, ~150 lines), context.py (ctx), spec.py (per-layer selection)
  components/       builtin attention, mlp, norm, posenc, residual, block
  data/             sources, filters, tokenizers, synthetic tasks, curate, dataset, stats, inspect_data
  train/            trainer.py (plain loop), optim.py (param groups), schedules.py
  evals/            runner.py, metrics.py, generate.py
  check.py report.py sweep.py scaffold.py cli.py runs.py
tests/              end-to-end smoke tests (pytest)
```

## Limits (by design)

Single device, no DDP/FSDP; bf16 or fp32 only; no KV-cache generation; `torch.compile` is optional (`train.compile`)
and may graph-break on components that mutate `ctx`. Meant for sandbox-scale models (up to ~100M params on one GPU).
