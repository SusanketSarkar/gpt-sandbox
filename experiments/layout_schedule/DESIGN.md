# Depth-dependent attention/MLP schedules — experiment design

**Question.** Are attention and MLP sublayers doing fundamentally different kinds of
information processing at different depths? Is the conventional alternating
`Attn → MLP → Attn → MLP → …` arrangement actually optimal, or does a
depth-dependent computational schedule exist that is better (or cheaper)?

**Prior work.** Press, Smith & Levy (2020), *Improving Transformer Models by
Reordering their Sublayers* (arXiv:1911.03864, ACL 2020): random orderings sometimes
beat the interleaved baseline; successful variants tend to have **more self-attention at
the bottom (early) and more feed-forward at the top (late)**; the proposed "sandwich"
(attention block early, FFN block late) improves perplexity on LM benchmarks at no
parameter cost, but gains are not guaranteed on every task (e.g. machine translation).
We test the same question with a sublayer-granularity layout knob, controlled
parameter/FLOP matches, and structured synthetic tasks that isolate local / contextual /
multi-hop computation.

## The knob

`model.extra.layout` is a string over `{A, M}` with one character per sublayer, and
`n_layer` is set to its length. Each character becomes one `LayoutBlock`
(plugins/layout_block.py): one RMS-norm, one component (attention or MLP from the
registry), one residual join. The baseline is the layout `"AMAM…"` (16 or 24 chars),
so the baseline and every variant share identical init scaling (residual-output
projections scaled by 1/√(2·n_layer)) and code path — removing the "different init"
confound. A standard 8-layer block model is also run as a sanity check.

## Controls

- **Parameters matched** by construction in Family 1 (same multiset of sublayers).
- **FLOPs measured, not assumed.** `gpt.flops_per_token` assumed `n_layer` attention
  layers and would report identical attention cost for every variant. Replaced with a
  measured count (`torch.utils.flop_counter.FlopCounterMode` on one forward pass, plus
  the analytic QK^T/AV term that SDPA hides inside a fused kernel). Reported in the
  `params` group (`flops_per_token`, `n_attn_sublayers`) and shown in `compare`.
- **Noise floor measured first** (3 seeds, identical config) before reading any effect.
- **Random permutations** of the same multiset give a null distribution, so a designed
  schedule must beat the *spread* of random orderings, not just the baseline.
- ≥3 seeds per variant; effects within ~0.005 val_loss (measured seed noise) are not
  claimed.

## Family 1 — ordering only (identical multiset, 8A + 8M, 16 sublayers)

Only the order differs; parameters and FLOPs are identical by construction.

| name      | layout (16 sublayers)                      | shape of the schedule |
|-----------|--------------------------------------------|------------------------|
| baseline  | `AMAMAMAMAMAMAMAM`                         | alternating            |
| A         | `MAMAMAAMAAMMAMAM`                         | MLP-heavy ends, attention-heavy middle |
| B         | `AMAMAMMAMMAAMAMA`                         | attention-heavy ends, MLP-heavy middle |
| C         | `MMMMAAAAAAAA MMMM` → `MMMMAAAAAAAAMMMM`   | MLP-only ends, attention-only middle |
| D         | `AAAAMMMMMMMM AAAA` → `AAAAMMMMMMMMAAAA`   | attention-only ends, MLP-only middle |
| sandwich  | `AAAMAMAMAMAMMM MM` → `AAAAMAMAMAMAMMMM`   | attention early → MLP late (Press et al.) |
| rand0..5  | 6 random permutations of the same multiset | null distribution      |

## Family 2 — fewer attention sublayers (the efficiency claim)

Changes the counts, so parameters are matched by construction: dropping 4 attention
sublayers and adding 2 MLP sublayers keeps parameters ≈ equal (an MLP sublayer ≈ 2× the
parameters of an attention sublayer at these shapes). Then `d_ff` on some MLPs is tuned
until the measured FLOPs match the baseline within a few percent.

| name | layout | attention | MLP |
|------|--------|-----------|-----|
| base | 8A + 8M | 8 | 8 |
| f2a  | 4A + 10M | 4 | 10 |

## Depth ablation

The whole Family 1 is re-run at smaller depths (8 sublayers, 4A+4M; and 12 sublayers)
to check the ordering effect holds as a function of depth.

## Synthetic tasks (plugins/tasks_local_hop.py)

| task        | rule                                                        | isolates |
|-------------|-------------------------------------------------------------|----------|
| local_rule  | next token = fixed function of the previous two             | token-local feature extraction |
| recall      | one-hop copy (existing)                                     | contextual integration (1 hop) |
| hop2        | `a→b, b→c, ?a → c` (existing)                               | multi-hop reasoning |

The task metric splits each line at the first separator; each task uses the separator
exactly once per line.

## Deep probes (descriptive, per layer)

- **ctx_sensitivity** (plugins/ctx_sensitivity.py): capture each layer's residual stream
  at the last position, recompute with all earlier tokens replaced by random ones, report
  the relative change per layer. Low = token-local; a jump marks where context is
  integrated. (Approximate: the change at layer i also includes earlier layers' reaction.)
- **logit_lens**: loss if you read out the residual stream after sublayer i — shows where
  predictions form.
- **layer_ablation**: skip sublayer i entirely, report the loss increase — which
  sublayers matter.

## Data

- **edu_mix** (`/data/susanket/text/gpt/data/edu_mix`), a curated natural-language mix,
  for the FLOP/parameter-matched LM comparison (the "does it still hold on real text"
  check).
- **Synthetic tasks** for the mechanistic isolation.

## Compute

NVIDIA A100-80GB × 2 (GPU 1, GPU 2). Jumbo model: 80M params, d_model 512, 24
attention + 24 MLP sublayers (48 sublayers), seq_len 256, batch 128.

## Artifacts

- Runs: `/data/susanket/text/gpt/runs/layout_*`
- Queue logs: `experiments/layout_schedule/queue_{A,B}.log`
- Aggregation: `experiments/layout_schedule/analyze.py`
- Figures + summary: `experiments/layout_schedule/figs/`, `experiments/layout_schedule/summary.md`
- Blog: `experiments/layout_schedule/BLOG.md`
