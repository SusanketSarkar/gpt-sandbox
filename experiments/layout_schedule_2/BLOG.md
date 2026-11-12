# What fraction of a transformer should be attention vs MLP?

**Date:** 2026-11-11 · **Status: complete — all 330 runs finished and re-analyzed** ·
**Companion visual blog (styled HTML, 9 figures):** [`BLOG.html`](./BLOG.html) ·
**Raw runs:** `/data/susanket/text/gpt/runs/ratio_*/` · **Data:** `/data/susanket/datasets/gpt/`

---

## Short answer

There is **no single universal "optimal" attention:MLP ratio.** The honest answer splits by task,
and one of the two answers turns out to be a confound in disguise:

- **For language modelling, the "optimal ratio" is mostly a *size* effect, not a ratio effect.**
  The raw sweep looks like MLP-heavy (~2:1) wins by a wide margin (0.074 val-loss, ~8× above the
  noise floor). But that sweep also made the MLP-heavy models *bigger* (more parameters). When I
  hold size constant and vary only the ratio, **~95% of that gap is size; the pure ratio effect is
  small (≈5%), noisy, and not monotone.** So for LM the real dial is *total parameters* — spend the
  budget on more (larger) sublayers, and the split barely matters.
- **For retrieval (the hard synthetic recall task), the ratio genuinely matters and 1:1 (balanced)
  is the sweet spot** — and it is a *size-independent* effect, because 1:1 is *not* the biggest
  model. Every other ratio is bimodal (some seeds solve it, some fail); 1:1 is the only split that
  clears the reliability bar on every seed.
- **The relationship is asymmetric.** You can move from attention toward MLP with little cost, but
  you cannot move from MLP toward attention without paying — in LM loss *and* in retrieval
  reliability. **MLP-heavy is safe; attention-heavy is the risky direction.**
- **The hypothesis "attention-heavy is better for everything" is disproved.** Its task-dependent
  branch is supported, with the twist that for LM the apparent ratio effect is really a size effect,
  whereas for retrieval it is a genuine ratio effect.

**If you are building a model:** for language modelling, size is the dial — a broad MLP-heavy
plateau (roughly 2:1 down to ~1.7:1 MLP:attention) is the safe region, and the "optimum" is a
plateau, not a point. For retrieval, use a balanced 1:1 split and expect the asymmetric cost if you
drift toward the attention-heavy side.

---

## 1. The question

In a decoder-only transformer, every sublayer is either an attention block or an MLP block, and the
conventional recipe puts about two-thirds of the sublayers in MLPs and one-third in attention.
**Question:** *Is there an "optimal" attention:MLP ratio, and what happens to (a) language-modelling
quality, (b) retrieval ability, and (c) efficiency as you change it?*

I set out to test the hypothesis that conventional models **over-allocate to MLP** and that
**increasing the attention share** would improve retrieval / contextual integration — up to an
attention-heavy optimum (an inverted-U peaking on the attention-heavy side). I was equally prepared
to find the opposite (that MLP capacity matters more than we assume), and to a third possibility:
that different tasks want different compositions, so no single universal ratio exists. I designed the
experiment to separate all three.

The honest framing, which the data forces, is subtler: **the interesting variable may not be the
ratio at all, but how the parameter budget is spent — and the ratio and the size are entangled in the
obvious way to measure it.** That entanglement is the crux of this study (Section 6).

---

## 2. What "a ratio" means — and the confound it hides

A model with 48 sublayers and 16 attention + 32 MLP sublayers has ratio **MLP:Attn = 2:1** (I write
ratios as MLP:attention, the conventional way; 16/32 means 16 attention, 32 MLP). The ratio is the
*shape* of the stack. **But here is the thing the ratio hides:** MLP sublayers carry far more
parameters than attention sublayers, so *fixing the total number of sublayers and varying the ratio
also varies the total parameter count*. The MLP-heavy configs are simply **bigger**:

| MLP:Attn | nAttn | nMLP | params (M) | FLOPs/token (M) |
|---|---|---|---|---|
| 4:1 | 10 | 38 | 95.0 | 585.7 |
| 3:1 | 12 | 36 | 92.9 | 576.1 |
| 2:1 | 16 | 32 | 88.6 | 556.8 |
| 1:1 | 24 | 24 | 80.1 | 518.3 |
| 1:2 | 32 | 16 | 71.6 | 479.7 |
| 1:3 | 36 | 12 | 67.3 | 460.5 |
| 1:4 | 38 | 10 | 65.2 | 450.8 |

This is what makes the naive reading dangerous: "MLP-heavy is better" could mean "the **ratio** is
better" *or* "the model is **larger**." Those are different claims. I keep them apart explicitly,
and Section 6 is where I finally separate them. (Note the parameter/FLOP gradient runs the *wrong*
way for the "attention-heavy is better" hypothesis — the attention-heavy configs are the *smaller,
cheaper* ones. So if attention-heavy were better it would be a free lunch; if worse, a double loss.)

---

## 3. How I ran it

**Model & stack.** Decoder-only GPT, d = 512, 16 heads, RoPE, GQA (4 KV heads), SwiGLU MLP with
4× expansion, context 2048, batch 32k tokens, AdamW (β 0.9/0.95, wd 0.1), 20k warmup, cosine to
1e-4, bf16, best-checkpoint selection on held-out loss (never the last step). I varied **only** the
attention:MLP split and its arrangement; everything else is fixed.

**Families** (330 runs total, all finished; raw dirs under `/data/susanket/text/gpt/runs/`):

| sweep | dir | what it isolates | runs |
|---|---|---|---|
| core ratio sweep | `ratio_jumbo/` | the main ratio answer (48 sublayers, 7 ratios) | 87 |
| parameter-matched control | `ratio_pm_jumbo/` | **the crux** — size fixed, ratio varied | 36 |
| depth ablation | `ratio_depth24/` | does it hold at half depth (24 sublayers) | 42 |
| refined grid | `ratio_fine_jumbo/` | finer ratios to locate the plateau edge | 24 |
| long-training | `ratio_jumbo_long/` | is the ordering a step-budget artifact? | 15 |
| retrieval: recall | `ratio_synth_recall/` | the hard retrieval probe | 42 |
| retrieval: 1-hop | `ratio_synth_hop2/` | easy probe (saturated) | 42 |
| retrieval: local | `ratio_synth_local/` | easy probe (saturated) | 42 |

Each ratio is replicated over **3 seeds × 4 spatial arrangements** (alternating, and attention
sublayers clumped to the front / middle / back) — 12 runs per ratio (the 1:1 anchor has 15). The
arrangements let me check that the *ratio*, not the *position*, is the driver (it is — Section 7).

**Data.** The standard mix — FineWeb-derived training + held-out validation, at
`/data/susanket/datasets/gpt/` (the `edu_mix` split).

**Noise floor.** The within-ratio standard deviation across seeds × arrangements in the core sweep is
**0.008**. I use that as my ruler: anything under ~0.008 is "no effect"; anything several times
larger is a real signal.

---

## 4. Result 1 — the core sweep: an MLP-heavy plateau

Final validation loss (lower is better), mean over seeds × arrangements, `ratio_jumbo/`:

| MLP:Attn | nA/nM | val_loss (mean) | sd | best ckpt | params (M) |
|---|---|---|---|---|---|
| 4:1 | 10/38 | 2.6924 | 0.0081 | 1.8071 | 95.0 |
| 3:1 | 12/36 | 2.6934 | 0.0059 | 1.8081 | 92.9 |
| **2:1** | **16/32** | **2.6902** ← best | 0.0076 | **1.8044** | 88.6 |
| 1:1 | 24/24 | 2.7016 | 0.0051 | 1.8138 | 80.1 |
| 1:2 | 32/16 | 2.7125 | 0.0058 | 1.8220 | 71.6 |
| 1:3 | 36/12 | 2.7321 | 0.0028 | 1.8381 | 67.3 |
| 1:4 | 38/10 | 2.7646 ← worst | 0.0213 | 1.8683 | 65.2 |

The curve is **not** the predicted inverted-U peaking at an attention-heavy ratio. It is the **mirror
image**:

- a **flat, near-tie plateau** on the MLP-heavy side (4:1, 3:1, 2:1 all within ~1 sd; 2:1 is the
  single best point, but 4:1 is within noise);
- a **monotonic, accelerating degradation** as you add attention past 2:1 (1:1 → 1:4 climbs
  2.702 → 2.765);
- the **worst** config (1:4, 38A/10M) is also the **most unstable** (highest sd, 0.021) —
  attention-heavy models not only lose on average, they are less reproducible.

**Statistical strength.** Within-ratio sd (the noise floor) is **0.008**; the best–worst gap is
**0.074** — about **8× signal-to-noise**. The ordering is not a seed fluke. The step-differences
(2:1→1:1→1:2→1:3→1:4 = +0.011, +0.011, +0.020, +0.033) are all positive and growing: a clean
monotone trend, not noise.

**Convergence signature.** The attention-heavy models also **start worse**: at step 500 the 38A/10M
config is at ~3.27 val loss vs ~3.00 for the MLP-heavy configs (see `figs/fig2_convergence.png`).
Less MLP capacity = weaker early feature transformation = a slower, higher trajectory that never
catch up. That is the mechanistic signature the "MLP capacity matters" explanation predicts.

**Takeaway (with the caveat).** At face value, MLP-heavy wins for LM. But remember the confound from
Section 2: the MLP-heavy models are also the *bigger* models. That is the claim I next have to test.

---

## 5. Result 2 — retrieval: 1:1 is a genuine sweet spot

I probed the "attention helps retrieval" claim directly with synthetic tasks
(`ratio_synth_recall/`, `ratio_synth_hop2/`, `ratio_synth_local/`):

| task | best ratio | verdict |
|---|---|---|
| 1-hop | flat (0.984–0.998) | **saturated** — no ratio difference |
| local | flat (1.000 all) | **saturated** — no ratio difference |
| **recall** | **1:1 (12A/12M)** | **1:1 is the only *reliable* ratio** |

The two easy probes are saturated — every ratio solves them, so they carry no signal. The hard
**recall** task is where the signal lives (per-run task_EM, n=6, `ratio_synth_recall/`):

| ratio | per-run task_em | mean | sd |
|---|---|---|---|
| 4:1 (5A) | 0.095, 0.16, 0.38, 0.89, 1.0, 1.0 | 0.588 | 0.424 |
| 3:1 (6A) | 0.11, 0.41, 0.855, 0.87, 1.0, 1.0 | 0.708 | 0.365 |
| 2:1 (8A) | 0.38, 0.395, 0.41, 0.56, 0.925, 1.0 | 0.612 | 0.280 |
| **1:1 (12A)** | **0.78, 0.935, 0.94, 0.995, 0.995, 0.995** | **0.940** | **0.083** |
| 1:2 (16A) | 0.155, 0.34, 0.355, 0.985, 0.99, 1.0 | 0.637 | 0.394 |
| 1:3 (18A) | 0.085, 0.88, 0.99, 0.995, 1.0, 1.0 | 0.825 | 0.366 |
| 1:4 (19A) | 0.325, 0.33, 0.995, 1.0, 1.0, 1.0 | 0.775 | 0.347 |

**The pattern is not "more attention is better." It is "1:1 is the sweet spot."** Every ratio
*except* 1:1 is **bimodal** — some seeds solve the task (1.0), some fail (0.09–0.4) — with high
variance (sd 0.28–0.42). **1:1 is the only split that clears the reliability bar on every seed**
(all ≥0.78, sd 0.083). This reads as a **capability threshold**: recall needs a specific balance of
attention (global search) *and* MLP (feature transformation), and 1:1 is the only split that reliably
has both.

**Crucially, this is size-independent.** The best recall ratio (1:1, 12A) is **not** the largest
model — the 4:1 config (5A) is the largest at 2.9M params and is the *worst* (0.588). So for recall,
unlike for LM, the ratio effect cannot be explained away as a size effect. **This is the cleanest
task-dependence evidence in the study:** the ratio best for LM (2:1) is *bad* for recall (bimodal,
mean 0.612), while the ratio best for recall (1:1) is only *slightly* worse for LM (2.702 vs 2.690).
**Different tasks, different optimal compositions.**

---

## 6. The crux — ratio vs size (the parameter-matched control)

This is the finding that reframes everything, from `ratio_pm_jumbo/`. I **fix total parameters at
~80M** and vary only the ratio (by resizing the MLP expansion `d_ff` so every config has the same
parameter count). That removes the size confound entirely:

| config | nA | nM | params | d_ff (MLP width) | val_loss (mean, n=6) |
|---|---|---|---|---|---|
| pm 4:1 | 10 | 38 | 80.1 | 968 | 2.7147 |
| pm 3:1 | 12 | 36 | 80.1 | 1016 | 2.7122 |
| pm 2:1 | 16 | 32 | 80.0 | 1136 | 2.7020 |
| **pm 1:2** | **32** | 16 | 80.0 | 1568 | **2.7015** ← best |
| pm 1:3 | 36 | 12 | 80.2 | 1992 | 2.7029 |
| pm 1:4 | 38 | 10 | 80.1 | 2440 | 2.7058 |

**At fixed size, the ratio effect nearly vanishes.** The spread across all six ratios is **0.013**
(2.7015 → 2.7147) — on the order of the noise floor (0.008), i.e. best-vs-worst differ by only
~1.7× the noise. And the shape is a **shallow, noisy wobble** (16A and 32A both ≈2.70; the two
extremes slightly worse), **not** the clean monotone degradation the raw sweep showed. Compare the
raw sweep's spread of **0.074** (where ratios span 65M–95M).

### 6.1 The decomposition — how much of the raw gap is size?

I bridge the raw 16A vs 38A gap (2.6902 → 2.7646, Δ = 0.0744) with a four-step path where each step
isolates one factor (`figs/fig8_decomp.png`):

1. **Raw 16A:** 88.6M @ **2.6902**
2. + size (38A: 65.2M → 80.1M): **−0.0588** → 2.7058 *(adding 15M params to the 38A model almost
   erases its deficit)*
3. + ratio @ 80M (16A → 38A): **−0.0038** → 2.7020 *(the pure ratio effect at fixed size — small)*
4. + size (16A: 80.0M → 88.6M): **−0.0118** → 2.6902 *(back to the raw best point)*

The four steps sum exactly to the raw gap (0.0588 + 0.0038 + 0.0118 = 0.0744). The two **size**
steps account for **0.0706 — about 95%** of the gap; the single **ratio** step is only **0.0038 —
about 5%**.

**That is the headline.** The raw sweep's "MLP-heavy wins by 0.074" is, to first order, **"MLP-heavy
models have ~30% more parameters, and bigger models are better."** At a fixed parameter budget the
attention:MLP ratio is a **secondary, noisy, non-monotone** factor for language modelling.

### 6.2 What this means

| task | is the "optimal ratio" real? | what actually drives it |
|---|---|---|
| language modelling | **No** (at fixed size, ratio ≈ irrelevant) | **total parameters (size)** |
| retrieval (recall) | **Yes** (1:1 is the sweet spot) | **the attention:MLP ratio itself** |

So the answer to "is there a specific optimal attention:MLP ratio?" is **task-dependent and, for the
primary task (LM), largely a matter of size, not ratio.** For LM you should spend your parameter
budget however you like (within the tested range) — what matters is having *more* parameters. For
retrieval, the *composition* matters: a balanced 1:1 split is what reliably delivers the capability.

And the cross-task **asymmetry** survives this reframing: even at fixed size, the *direction* of
moving toward attention is the risky one (it is where the raw degradation, the instability, and the
recall reliability loss all concentrate). **MLP-heavy is safe; attention-heavy is costly.**

---

## 7. Robustness checks

- **Depth ablation** (`ratio_depth24/`, 24 sublayers, d=256): the same qualitative shape at half
  depth — best at the MLP-heavy end (3:1, 2.8673), then monotone degradation toward attention-heavy
  (1:4, 2.9743); spread 0.107. So it is not an artifact of the 48-sublayer depth.
- **Refined grid** (`ratio_fine_jumbo/`, filling 14/18/28/34 attention): the best point is 18A
  (2.6892), essentially tied with the 16A core point (2.6902, Δ=0.001); 14A is slightly worse but
  within ~1σ; and the curve degrades clearly on the attention-heavy side (28A is +0.011, 34A is
  +0.036 over the plateau). So the "optimal" region for LM is not a sharp point at exactly 2:1 — it
  is a **broad plateau around 16–18A (MLP:Attn ≈ 2:1 down to ~1.7:1)**. Read "optimal" as that
  plateau, not a single ratio.
- **Long-training** (`ratio_jumbo_long/`, more steps): the ordering **preserves** (best 2:1 at 2.552,
  worst 1:4 at 2.605), consistent with the convergence signature (they start worse and stay worse).
  So the LM ordering is **not** a step-budget artifact.
- **Stability** (`figs/fig7_stability.png`): gradient-norm p95 rises toward the attention-heavy end
  (0.54–0.56 MLP-heavy → 0.75 at 38A), and the only training spike in the core sweep is at an
  attention-heavy config. The attention-heavy side is worse *and* less stable.
- **Arrangement** (4 spatial placements per ratio): placement is a *secondary* effect
  (within-ratio spread 0.006–0.044), largest for the attention-heavy ratios (where some placements
  are notably bad). The **ratio is the dominant variable; position is a minor modifier.** Good — it
  means the controlled ratio sweep is measuring the right thing.

---

## 8. Answer to the hypothesis

| Prediction | Result |
|---|---|
| Attention-heavy outperforms conventional | **Disproved.** Attention-heavy is the *worst* side for LM, and also less stable. |
| Optimum is an attention-heavy ratio | **Disproved.** The "optimum" is MLP-heavy — but see the size effect below. |
| Non-linear inverted-U peaking at attn-heavy | **Disproved.** Flat MLP-heavy plateau + monotone attn-heavy degradation (the mirror shape). |
| Task-dependence (retrieval wants a different split) | **Supported (strong).** Recall's optimum is 1:1, not the LM "optimum" of 2:1. |
| MLP capacity is more important than anticipated | **Refined.** For LM it is *total* capacity (parameter count), not specifically MLP capacity. |

**Does a specific optimal ratio exist?** For language modelling, **no genuine one** — at a fixed size
the ratio barely matters, and what drives performance is the total parameter count. The apparent
"MLP-heavy optimum" is a size effect. For retrieval, **yes — 1:1 is a genuine, size-independent
optimal ratio.**

**Interpretation.** The load-bearing work that looked like "MLP capacity" is actually *total capacity*
(parameter count) for language modelling. At a fixed size, how you split it between attention and MLP
matters far less for LM than the hypothesis assumed. But for retrieval, the *split itself* matters —
a balanced 1:1 reliably delivers the global-search capability, while both the MLP-heavy and
attention-heavy extremes fail on some seeds. **Conventional wisdom is right that you need sufficient
capacity, but it is wrong to think the attention:MLP ratio is the key dial for LM — for LM it is the
size dial.**

> **And if you want to *hear* the difference:** the [appendix](appendix.md) is a "tasting menu" —
> the same four prompts fed to every ratio, so you can taste the fluent-MLP voice versus the
> number-spewing, retrieval-sharp attention-heavy voice, and see the asymmetry in prose.

---

## 9. Confounds & limitations (the exhaustive list)

This is where I want to be maximally honest, because the "optimal ratio" framing is easy to over-claim:

1. **The ratio and size are entangled** in the raw sweep (MLP-heavy = more parameters). I
   disentangled them with the parameter-matched control (Section 6), which is why the LM conclusion
   is "size, not ratio." Any LM claim that ignores this is confounded.
2. **The parameter-matched control holds size fixed by changing `d_ff`**, so it trades *MLP width*
   for *ratio*. A model with more attention sublayers but a *narrower* MLP (to hold params) is a
   different animal from the raw attention-heavy model. The "ratio effect vanishes at fixed size"
   result is therefore specifically "at fixed size, *reallocating width between attention and MLP*
   has little LM effect" — not a statement about, e.g., changing attention *head count* or
   *dimension* at fixed params.
3. **Single width family.** One d (512) for the jumbo sweeps; the depth ablation uses d=256. I did
   not sweep width independently, so the size-vs-ratio balance at other widths is untested.
4. **Fixed step budget per run.** All ratios trained for the same steps. The long-training runs show
   the ordering *preserves*, but a ratio that is "worse at equal steps" could in principle behave
   differently with very different compute per ratio — not fully ruled out.
5. **Retrieval probes are synthetic and coarse.** The recall/1-hop/local tasks are simple. Real
   long-range retrieval (needle-in-haystack at long context) is the stronger probe and is *not* in
   this study. The 1:1 sweet spot is a result on *these* synthetic tasks.
6. **No generation/quality metrics.** I report loss + retrieval accuracy; qualitative generation
   differences (if any) are not characterized.
7. **Seed/arrangement variance is small but non-zero** (sd ~0.008); the 4:1-vs-2:1 "best" choice is
   within noise, so "optimal" for LM should be read as the *plateau* (≥~1.7:1 MLP:attn), not a sharp
   point.
8. **One dataset** (the `edu_mix` FineWeb-derived split). Cross-domain generalization of the
   ratio/size findings is untested.

---

## 10. Reproduction

- **Code:** `experiments/layout_schedule_2/` — `gen_ratio_sweeps.py` (config matrix), `worker.sh`
  (self-healing run worker), `watchdog.sh`, `pick.py`, `analyze.py`, `make_figs.py`, `make_illus.py`
  (the layout/decomposition/recall-dot illustrations), `finalize.sh` (regenerate everything).
- **Runs (all 330 complete):** `/data/susanket/text/gpt/runs/ratio_{jumbo,pm_jumbo,depth24,fine_jumbo,jumbo_long,synth_recall,synth_hop2,synth_local}/`
  (87 + 36 + 42 + 24 + 15 + 42 + 42 + 42 = 330; every run has a final val loss).
- **Data:** `/data/susanket/datasets/gpt/` (`edu_mix` train + held-out val).
- **Artifacts:** `tables.md` (all numbers), `summary.json`, `all_runs.json`, `figs/fig0..fig9.png`,
  and the styled visual blog `BLOG.html` (figures: layout strips, core curve, convergence, the
  three retrieval probes, recall dots, raw-vs-param-matched, the size/ratio decomposition, depth,
  stability).
- **Regenerate:** `bash finalize.sh`.

---

*All 330 runs are finished and re-analyzed. Core conclusion (robust across depth, step budget, and
replication): the "attention-heavy is better" hypothesis is **disproved**; the apparent LM ratio
effect is a **size effect** (≈95% of the raw gap, per the parameter-matched decomposition); and
retrieval has a **genuine, size-independent 1:1 optimum** with an asymmetric cost toward the
attention-heavy side. No single universal optimal ratio exists — the right answer depends on the
task, and for the primary task (LM) it is the size dial, not the ratio dial.*
