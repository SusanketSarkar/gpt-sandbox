# Where is the residual stream fragile?

**Question:** Does the model's information density vary by depth?

**Hypothesis:** Gaussian noise injected into the residual stream *during training* hurts more in the
middle layers than at the ends, because middle layers carry the least redundant information.

- If it holds → the information-dense layers are in the middle (early = local work, late = readout).
- If it fails → noise acts as a generic regulariser (plausible on a ~1 MB overfitting corpus) — check
  `gen_gap` before concluding anything about layers.

## Setup

- **Insert:** `plugins/rms_noise.py` — `x ← x + σ·rms(x)·N(0,I)` at site `after_attn`,
  **only while `ctx.training`** (gated on `model.train()`; every eval path uses `model.eval()` so
  val_bpb / gen_gap / ablations are all measured on the clean model).
- σ is a fraction of the slice's own RMS → scale-invariant across depth and model size.
  σ=1.0 = noise std equals the stream's std.
- **Data:** `edu_mix` (1.56M tokens), same corpus for all runs.
- **Sizes:** M = 8 layers / 25M params / 10k steps; D = 16 layers / 77M params / 12k steps.
- **Measure:** val_bpb (clean eval), gen_gap, layer-ablation profile (`--deep`), logit lens (`--deep`),
  plus a test-time sensitivity profile (no retraining; noise at eval time only).

## Cells (114 runs total)

| cell | variants | seeds | runs | purpose |
|---|---|---|---|---|
| `depth_{m,d}_core` | plain + {first, mid, last} × σ∈{0.1, 0.3, 1.0} | 3 | 30 | the hypothesis test |
| `depth_{m,d}_profile_{a,b}` | every layer × σ∈{0.3, 1.0} (split a/b across GPUs) | 1 | 16+16 | full fragility curve |
| `depth_{m,d}_global` | all layers, σ∈{0.1, 0.3, 1.0} | 1 | 3 | pure-dose / regulariser control |

`gpt sweep` resumes: runs with `eval.json` are skipped. Orchestrator: `orchestrator.sh` (3 GPU workers).
Deep evals (ablation/lens) run automatically after each cell for plain + σ=1.0 anchors + global controls.

## Reading the results

1. **Fragility curve** (train-time): Δval_bpb vs layer. A middle peak = hypothesis.
2. **Anchor test**: first/mid/last paired over seeds, in units of the plain seed-noise floor.
3. **Shape test**: Spearman ρ of the curve against a centred "bump" prior, permutation-tested.
4. **Importance check**: does fragility track plain-model layer importance (ablation, lens) or output
   scale? Importance-tracking = density story; scale-tracking = bookkeeping story.
5. **Test-time sensitivity** (no retraining): Δval_bpb of the plain model when one slice is noised at
   eval time. The static dependency side of the same question.
6. **Regulariser checks**: gen_gap by variant, variants beating plain, global-vs-mid dose ratio.

## Files

- `analyze.py` — runs → `results.json` + `results.md` + `plots/`
- `build_blog.py` + `blog_template.html` + `narrative.json` → `index.html` (the blog)
- `sensitivity.py` — eval-time sensitivity profiles → `sensitivity.json`
- `monitor.sh` — one-shot progress snapshot
- `state/` — orchestrator + per-cell logs
- run dirs + checkpoints: `/data/susanket/text/gpt/runs/depth_*`
