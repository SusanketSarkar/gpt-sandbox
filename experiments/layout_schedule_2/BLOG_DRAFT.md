# Draft — "Is there an optimal attention:MLP ratio in a transformer decoder?"

> STATUS: PROCESS section drafted. FINDINGS section to be filled after the night's runs
> complete (core by ~22:45, all by ~08:45). Run `bash finalize.sh` to regenerate tables/figs,
> then fill in the FINDINGS placeholders below with real numbers.

## 1. Question

Conventional decoder-only transformers allocate roughly **two-thirds of sublayer compute to
MLPs and one-third to attention** (for a 4× MLP expansion). Attention does *global*
token-to-token mixing; MLPs do *token-local* nonlinear transformation.

**Hypothesis:** conventional models may *over-allocate* to token-local compute. Raising the
attention proportion (and cutting MLPs) may improve retrieval / contextual integration and
language modelling — up to an optimum, after which diminishing returns set in because there is
not enough MLP capacity left to transform features.

We ask: **is there a specific attention:MLP ratio that is optimal, and does it depend on the
task (LM vs. retrieval)?**

## 2. Method (what we actually ran)

**Unit = sublayer.** Every block runs exactly one op (attention *or* MLP), chosen by a layout
string. The 1:1 baseline is `"AM"*24` (48 sublayers). We sweep the A/M *multiset*, holding
width (d=512), attention config, MLP expansion, and total sublayers (48) **fixed**, so the only
thing that changes is the attention:MLP proportion.

| sweep | what | ratios (nA:nM) | arrangements | seeds | scale |
|---|---|---|---|---|---|
| ratio_jumbo | primary sweep | 4:1,3:1,2:1,1:1,1:2,1:3,1:4 | alt + 3 random | 3 | 48 sub, d=512 |
| ratio_pm_jumbo | **param-matched control** (vary d_ff to hold params constant) | 1:2,1:1,2:1 | alt | 3 | 48 sub, d=512 |
| ratio_fine_jumbo | refine the optimum | 1:1.5,1:2,2:1,1.5:1 | alt | 3 | 48 sub, d=512 |
| ratio_depth24 | **depth ablation** (does it hold at 2× less depth?) | 7 ratios | alt | 3 | 24 sub, d=256 |
| ratio_synth_{recall,hop2,local} | **retrieval tasks** (one-hop, multi-hop, local-distractor) | 5 ratios | alt | 3 | 24 sub, d=128 |
| ratio_jumbo_long | longer training (12k steps) | 1:2,1:1,2:1 | alt | 3 | 48 sub, d=512 |

- **3 seeds × 4 arrangements per ratio** (jumbo) controls for seed noise and layout *order*
  (the prior layout_schedule experiment showed order effects are second-order vs. the A/M ratio).
- **Param-matched control** separates "more attention" from "more total params" (the MLP-heavy
  side has more params; we hold params constant by shrinking d_ff on the attention-heavy side).
- **Depth ablation** checks the effect is not an artefact of the specific 48-sublayer scale.
- **Synthetic retrieval tasks** directly probe the "attention = retrieval" mechanism:
  - `synth_recall` — one-hop: find a unique token mentioned earlier, answer with it.
  - `synth_hop2` — multi-hop: A→B→C chained relations, answer C.
  - `synth_local` — local distractor: answer from nearby context (MLP-favoured), controls for
    "attention-heavy just memorises locally".
- **Metrics:** FineWeb val loss, one-hop/multi-hop/long-range recall accuracy, FLOPs/token,
  param count, convergence (loss/gnorm curves), calibration, induction heads, generation.

**Compute:** 327 runs × ~15.5 min (jumbo) on 4× RTX 4090, ~15 wall-hours. Priority order:
core sweep → retrieval → control → depth → refine → long.

## 3. Findings

<!-- FILL IN AFTER RUNS COMPLETE. Structure:
### 3.1 Does an optimal ratio exist for LM?  [fig1]
  - val_loss vs ratio curve (jumbo, param-matched, raw). Is it U-shaped? Where's the min?
  - Is the optimum on the attention-heavy side of 1:1, the MLP-heavy side, or AT 1:1?
  - Seed/arrangement noise vs. effect size (is the effect > noise?).
### 3.2 Param-matched: is it the ratio or just params?  [fig6]
  - Does the attention-heavy advantage survive when params are held constant?
### 3.3 Retrieval: does attention proportion help retrieval?  [fig3, fig4]
  - one-hop / multi-hop / long-range recall vs ratio. Attention-heavy better?
  - Does the local-distractor task show the opposite (MLP-favoured)?
  - -> task-dependence: LM optimum vs retrieval optimum.
### 3.4 Does it hold across depth?  [fig5]
  - 24-sublayer ablation: same direction of effect?
### 3.5 Convergence & stability  [fig2, fig7]
  - Do attention-heavy models train stably? gnorm/loss curves. Any instability at extremes?
### 3.6 Efficiency  [fig6]
  - FLOPs/params per ratio. Is the attention-heavy config more/less efficient per val-loss?
-->

## 4. Interpretation

<!-- WHY: attention = global mixing, MLP = local transform. If attention-heavy wins on
retrieval but not (or less) on LM, the two compute types are doing different jobs and the
"optimal" ratio is task-dependent. If attention-heavy wins on both, conventional models are
over-allocating to MLPs. If 1:1 (or MLP-heavy) wins, global communication alone cannot
compensate for insufficient nonlinear capacity. -->

## 5. Conclusion (proved / disproved / task-dependent)

<!-- One-paragraph answer to "is there a specific attn:mlp ratio that exists?". -->

## 6. Limitations

- Fixed d=512, seq_len=256, 6k steps (jumbo); small model — effects may shift at scale.
- One dataset (edu_mix); one architecture family (RoPE MHA + SwiGLU).
- Recall metric is noisy across seeds (use val_loss as primary signal).
- Per-layer mechanistic metrics (logit-lens/ablation) not available in this run's eval.

## 7. Reproduce

```
cd experiments/layout_schedule_2
python3 gen_ratio_sweeps.py   # (re)generate configs + runlist
bash launch.sh 4              # 4 GPU workers over the shared queue
bash status.sh                # progress
bash finalize.sh              # tables.md + summary.json + figs/*.png
```
Artifacts: `tables.md`, `summary.json`, `all_runs.json`, `figs/fig1..7*.png`, run dirs under
`/data/susanket/text/gpt/runs/ratio_*`.
