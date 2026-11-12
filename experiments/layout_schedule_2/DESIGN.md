# layout_schedule_2 — the attention:MLP ratio question

**Research question.** What is the optimal proportion of attention vs MLP sublayers in a
decoder-only transformer, and how does changing this proportion affect language-modelling
performance, retrieval capabilities, and computational efficiency?

**Hypothesis under test.** Conventional stacks allocate ~2/3 of sublayer parameters to MLPs
(4x expansion) and ~1/3 to attention. Attention does global token-to-token information
exchange; MLPs do token-local nonlinear transformation. We test whether conventional
architectures **overallocate** to token-local compute: raising the attention proportion
should improve retrieval and contextual integration — up to an optimal ratio, beyond which
diminishing returns set in because MLP capacity becomes the bottleneck. The relationship
should be **nonlinear**.

## Why the previous experiment (layout_schedule) does not answer this

`layout_schedule` held the sublayer COUNTS fixed (16 = 8A + 8M) and only permuted their
order. It established that *order* matters (interleaved > blocky) but said nothing about
*how many* of each sublayer type there should be. This experiment varies the COUNTS.

## Fixed vs varied

| Quantity | Value | Note |
|---|---|---|
| total sublayers | 48 (jumbo) / 24 (depth ablation) | fixed |
| d_model | 512 (jumbo) / 256 (depth) / 128 (synth) | fixed |
| n_head | 8 | fixed |
| attention | mha, head_dim = d/8, RoPE, no bias | fixed |
| MLP | swiglu | fixed |
| seq_len / batch | 256 / 128 | fixed |
| optimizer | AdamW lr 6e-4 (1e-3 synth), wd 0.1, cosine, warmup 400 | fixed |
| **layout string** | **varied** | the independent variable |
| **d_ff (pm family only)** | **varied to match params** | control for param count |
| data | edu_mix (natural), synth tasks (probes) | per family |
| seeds | 0, 1, 2 | 3 per config |

## Ratio grid (user-specified)

48 sublayers — `nA` attention sublayers, `nM = 48 - nA` MLP sublayers:

| tag | nA | nM | MLP:Attn |
|---|---|---|---|
| r41 | 10 | 38 | 3.8:1 (≈4:1, "more MLP than conventional") |
| r31 | 12 | 36 | 3:1 |
| r21 | 16 | 32 | 2:1 |
| r11 | 24 | 24 | 1:1 (**conventional anchor**) |
| r12 | 32 | 16 | 1:2 |
| r13 | 36 | 12 | 1:3 |
| r14 | 38 | 10 | 1:4 ("more attention than conventional") |

Note: with a *conventional* 4x expansion and 1A:1M the param split is
attn:(4·d·d) vs mlp:(3·d·d_ff)=3·d·(8d/3)=8d² → 1:8 per sublayer, i.e. ~89% of sublayer
params in the MLP. The "2/3 MLP" figure in the hypothesis statement is the *parameter*
split under a 4x expansion with 1:1 counts: attn 4d², mlp 3d·4d=12d² → 1:3 → 75% MLP.
Either way the conventional anchor (r11) is heavily MLP-parameter-weighted.

## Families (in priority order)

1. **ratio_jumbo** — raw sweep, d_ff fixed at baseline default (1376). Param count varies
   ±19.6% (more MLPs = more params). 7 ratios × 4 arrangements (alt + 3 seeded random) × 3 seeds = **84 runs**.
2. **ratio_synth_{recall,hop2,local}** — same 7-ratio grid at 24 sublayers (nA 5..19),
   d=128, trained on the synthetic task datasets, `task` metric (EM) + long_recall measured.
   7 × 2 arrangements × 3 seeds × 3 tasks = **126 runs**.
3. **ratio_pm_jumbo** — param-matched control: d_ff tuned per ratio so total params match
   the r11 baseline within ±0.17% (1120..2344). 6 ratios × 2 arrangements × 3 seeds = **36 runs**.
4. **ratio_depth24** — depth ablation: 24 sublayers, d=256, same ratio grid. 7 × 2 × 3 = **42 runs**.
5. **ratio_fine_jumbo** — refinement of the coarse curve at nA ∈ {14, 18, 28, 34}. 4 × 2 × 3 = **24 runs**.
6. **ratio_jumbo_long** (stretch) — 12000-step training of anchor + extremes, to check the
   ordering persists under longer training. 5 × 1 × 3 = **15 runs**.

Total: **327 runs**. Budget ≈ 51 GPU-hr ≈ 13 h on 4× H100 (measured: jumbo 14.5 min,
depth 8 min, synth 5.2 min per run).

## Metrics per run

- `val_loss` (FineWeb-edu val, fixed windows, best checkpoint) — primary LM metric.
- `long_recall` — new eval metric: random-token sequences with a repeated token at distance
  d ∈ {8,16,32,64,128}; `recall_acc/d = P(p(d) > p(d+1))`, `recall_gap/d = Δp/d`. Direct
  probe of global token-to-token retrieval.
- `task_em` (synth families) — exact-match on one-hop recall (8-pair), multi-hop (20-pair,
  2-hop chain), and a local rule (doubled token) — the three task classes from the
  hypothesis' "retrieval vs token-local" framing.
- `params`, `params_nonemb`, `flops_per_token` (measured), `tok_per_s`, wall-clock — cost side.
- convergence: val-loss trajectory every 500 steps; train loss/gnorm every 100 steps
  (metrics.jsonl) — convergence speed + stability (gnorm spikes, non-finite loss).
- deep probes (selected checkpoints): `logit_lens` (depth of prediction), `layer_ablation`
  (which sublayer types carry the function), `ctx_sensitivity` (retrieval vs local).

## Decisions / notes

- **Seeds 0/1/2** everywhere (continuity with layout_schedule, whose measured noise floor
  was ±0.0026–0.0027 val-loss over 3 seeds at fixed layout — a ratio effect must clear that).
- **Arrangements**: `alt` = evenly distributed (best-known order from layout_schedule);
  `r0..r2` = random permutations (seeds 1000+nA*7+i) — brackets arrangement dependence.
- **Param-matched d_ff** computed exactly from the built baseline model (75,915,776 non-emb
  params): per-attention 4d²=1,048,576; per-MLP 3d·d_ff; norms (n+1)·d=25,088. Verified by
  building each model: all pm variants within ±0.17% of baseline.
- **best checkpoint** (lowest fixed-window val loss) is evaluated, not last — cosine schedule
  + fixed eval windows make this the unbiased choice (layout_schedule established the
  protocol).
- **Stale-lock stealing**: worker locks are claimed with atomic `mkdir`; locks older than
  90 min are stealable (a jumbo run takes <35 min).
- **Retries**: up to 3 attempts per run (crash/OOM), then marked failed.
- Old `layout_f1_jumbo/baseline-s{0,1,2}` runs (identical config to r11_alt, 3 seeds) are
  kept as extra 1:1 anchor points; long_recall to be added to them on a free GPU.
- `smoke2_synth` / `smoke2_depth` were full 16k/12k-step runs and were renamed into the
  official `ratio_synth_recall/r41_5A19M_alt-s0` and `ratio_depth24/r41_5A19M_alt-s0` slots.

## Early signal (before the sweep finishes)

- `ratio_synth_recall/r41_5A19M_alt-s0` (MLP-heavy, 5A+19M): long_recall acc ≈ 0.08–0.13 at
  all distances — poor long-range retrieval, consistent with the hypothesis' direction.
- Both extreme jumbo layouts (10A+38M, 38A+10M) pass all contract checks
  (causality, gradients, overfit, batch independence).
