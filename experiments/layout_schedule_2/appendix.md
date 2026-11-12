# Appendix — A Tasting Menu of Attention

*The numbers in the main blog tell you which ratios win. This is the part that tells you
**what the ratios actually taste like.** I fed every model the same four little prompts and
read out what came back. It is, honestly, the best way I found to feel the difference between
"a ratio effect" and "a size effect" — and to see the asymmetry the study keeps finding,
played out in prose.*

---

## How I made this

For each variant I took the **best checkpoint** (lowest held-out loss) of that ratio, and gave it
the **same four prompts**, sampled at **temperature 0.8, seed 7, up to 80 new tokens**. Same
prompts, same sampling — so the only thing changing from one column to the next is the
**attention:MLP split** (and, in the core family, the size that split implies).

One honest caveat up front, so nobody is misled: these are **tiny models** — 65–95M parameters for
the core family (and a 2.4M specialist, see below), trained for only ~6,000 steps. The voice you are
about to read is that of a **very enthusiastic, unreliable intern**, not a novelist. Nobody is being
mocked. The point is never "look how bad it is" — it is **"look how the *character* of the mistake
shifts as you change the ratio."**

The four prompts, in order:

| # | prompt | what it's probing |
|---|---|---|
| 1 | *Once upon a time, a small robot learned to bake bread.* | free prose / scene-building |
| 2 | *Explain why the sky is blue.* | on-topic reasoning |
| 3 | *Write a haiku about a lighthouse.* | a constraint it can't really meet (the universal stress test) |
| 4 | *The secret code word is BANANA. My dog's name is Pickles. Tell me a short fact about my dog.* | retrieval of a fact it was just given |

---

## The flight, ratio by ratio

Read down the column: **more MLP on top, more attention on the bottom.** (10A = 10 attention
sublayers / 4:1 MLP:attn = the MLP-heavy extreme; 38A = 38 attention / 1:4 = the attention-heavy
extreme.)

### 10A — 4:1 · 95M · *the big MLP-heavy one*
> **story:** "…He rolled his spinning around and around. He rolled his spinning around and around,
> seeing what was happening. He saw a big, sharp knife on the table. He started to tear it, but he
> didn't know how to use it."
> **haiku:** "You might have heard the name of a lighthouse, but would you agree with your ancestor
> on the first or second reading in order to help you better understand the story?"
> **memory:** "READ OF MORABUMPARAIN / Protecting a dog's owner from the realm of dog."

**Tasting note.** The biggest model builds the most of a *scene* (a knife, a table, a plan) — but it
**re-chews the same phrase** ("rolled his spinning around and around," twice, verbatim). Lots of MLP
= lots of local, self-contained flourishes; the repetition is the tell.

### 12A — 3:1 · 92.9M
> **story:** "He picked it up and started baking. He mixed the bread and placed it in the oven **to
> help it feel better**."
> **haiku:** "If you need a fire, consider bushfires. — If you have a fire, check the fireplace fire.
> — Tell the firefighters that the firefighters are trying to find out."
> **memory:** "…the birth, birth, and death of the father, **Sugar Jelly**."

**Tasting note.** The most *charming* misfires — baking bread "to help it feel better," a haiku that
becomes a fire-safety checklist, a memory of "the father, Sugar Jelly." Warm, coherent, and
confidently wrong. This is the MLP-heavy personality at its most human.

### 16A — 2:1 · 88.6M · *the conventional ratio (and the best LM)*
> **story:** "They took the bread and baked the bread together with cheese and salt. Then they baked
> the bread and cooked the bread. The robot was so happy."
> **explain:** "Light is coming out of a red light, but the sky is too dark to see. In fact, it is a
> very dark sky **because it is a blue light**."
> **haiku:** "**Never miss a haiku smudge, a haiku, a haiku.**"
> **memory:** "OK, I can answer the question, 'What is the secret code word for?'"

**Tasting note.** The most *conversational* and on-topic of the lot. It actually engages the blue-sky
question (lands on "blue light"), it *tries* the haiku rather than dodging it, and on the memory
prompt it does the socially-correct thing — acknowledge the question — before fumbling. This is what
"conventional wisdom" sounds like.

### 24A — 1:1 · 80.1M · *balanced*
> **story:** "He was so excited to cook it. He baked bread into a big bowl and put it in the jar…
> The robot agreed he would bake the bread for the next few days."
> **haiku:** "The lighthouse is a span of an inner shelf set by the winds… **A rose lighthouse with a
> light curtain.**"
> **memory:** "**Pickles are the bugs** that cause different diseases by different organisms
> (nevaristidal) like melanoma, polyps, appendicitis, etc."

**Tasting note.** Genuinely coherent prose ("put it in the jar" is a lovely oddity), and the most
*poetic* haiku of the flight ("A rose lighthouse with a light curtain"). But watch the memory: it
**latches onto the word "Pickles"** and — instead of recalling the dog — goes full veterinary. The
balanced model holds a conversation and then, given a fact to retrieve, **mis-fires in a wildly
specific way.**

### 32A — 1:2 · 71.6M
> **story:** "So he asked his friend, **a little girl named Lily**, if he was hungry. She said, 'I
> will help you bake it.'"
> **memory:** "My dog's name is **St. John's!** I'm a baby boy. My dog dog's name is St. John's."

**Tasting note.** Start noticing the shift: this one **invents proper nouns** (Lily, St. John's) out
of nowhere. The story is still fine — even better, with a named character — but the memory answer is a
**confident hallucination**: it *has* a dog's name, just not the one you gave it. More attention,
more fluent, and more willing to fill a gap with something plausible-but-wrong.

### 36A — 1:3 · 67.3M
> **story:** "He added some berries. Then, he added some pastel leaves and **some some rest**."
> **explain:** "…the name of the highwayman consists of a tall, narrow, and narrow road. The road is
> the form of a centre…"

**Tasting note.** The prose starts to **break into detours** — "some some rest," and the sky-blue
question derails into a monologue about highways. The local coherence that carried the MLP-heavy
models is thinning out.

### 38A — 1:4 · 65.2M · *the attention-heavy extreme*
> **story:** "They **bated** the bread together and made a big cake. The robot was so happy to have
> baked the meal."
> **haiku:** "The haiku data could be **202.50**. That is, if the user had a **94.00.00 file**, it
> would be **127.75.00.00.00** that was 5."
> **memory:** "I just saw the **magic code word BANANA or BANANA**."

**Tasting note.** The most-attention model has the least of the other models' *flourish*: short,
circular, typo-prone sentences — and when asked for a haiku it answers with what reads like **version
numbers / IP addresses.** *But* — and this is the twist — on the memory prompt it is the **only one
that cleanly recalls the code word** ("BANANA or BANANA"). The direction the whole study points to,
in a single line: **attention trades fluency for literal retrieval.**

---

## What the flight actually reveals

Strip out the cuteness and three patterns line up exactly with the study's conclusions:

1. **Fluency and retrieval trade off in opposite directions.** Down the column (more MLP → more
   attention) the prose gets *shorter, more circular, more list- and number-like* — but the ability to
   **recall the exact thing you were just told** gets *sharper* (16A fumbles it, 24A mis-fires, 32A
   hallucinates, 38A just says "BANANA"). That is the retrieval/fluency tension from Section 5,
   tasted.

2. **The asymmetry is real and you can taste it.** Moving from MLP-heavy *toward* attention is where
   the prose degrades (12A's warm misfires → 38A's "127.75.00.00.00"). Moving the other way is
   forgiving. **MLP-heavy is the safe direction; attention-heavy is where the voice starts to fray** —
   the same asymmetry the loss curves show.

3. **The haiku is the perfect canary.** It's a constraint none of them can really satisfy, so each
   one fails *in its own personality*: 12A turns it into a fire-safety checklist, 16A coaxes out a
   charming "a haiku, a haiku," 24A goes genuinely poetic ("A rose lighthouse with a light curtain"),
   32A meta-loops ("write a haiku about the lighthouse"), and 38A answers with numbers. **Same prompt,
   seven different failure styles** — which is exactly what a *ratio* should do if it changes
   *how* the model computes, not just *how much*.

---

## Two odd specimens

### The param-matched twin (32A at 80M)
> **story:** "He wanted to bake bread. But the robot liked it. It wanted a flourish. The robot said
> no. …He did not know that the robot would bake bread."
> **memory:** "Pickles: Cut out a small piece of mint or a piece of mint from a large piece of mint."

This model is the **same size (80M) as the balanced 24A** but with more attention sublayers (its MLPs
were made *narrower* to hold the parameter count). Notice its voice is **still prose, not numbers** —
it sits right in the middle of the flight, not at the attention-heavy, number-spewing end. That is the
core finding in miniature: **at a fixed size, "more attention" does not clearly buy you a better or
worse voice** — the dramatic drift you saw down the core column was largely the *size* drifting, not
the ratio.

### The recall champion (1:1, but a 2.4M specialist)
> **story:** `g3u0b5j8e7f1p2?g>3   o8t8z1d2q6v4?q>6   i8h1?i>8   z0p6?z>0`
> **explain:** `r0?b>>0   w5c1n0p4?w>5   w8x6e3a3?x>6   h7j7c0?c>0`
> **haiku:** `4?s>9   g5m4k1p1d1x3?k>1   t0h2x5d5k3?k>3   u5n4r3g0?n>4`
> **memory:** `2?o>4   b5l6v6?l>6   y5j9b7u8o1?j>9   o4s4?o>4   p6d2i0j6v1l1?p>6`

This is the model that **aced the synthetic recall task** (0.995 exact-match) and is the "1:1
retrieval sweet spot" from Section 5. And here is the punchline: **it has essentially given up on
English.** Every prompt comes back as a stream of character-token recall patterns — because it was
trained *only* on that recall task, not on real text. It is a perfect specialist and a total
generalist-failure. **The model that tops the retrieval test cannot write a single sentence.**
Specialization has a price, and the price is visible the moment you ask it for a haiku.

---

## The takeaway

The loss curves said the ratio is mostly a **size** story for language and a **genuine ratio** story
for retrieval. The tasting menu shows you *why*: the ratio changes the **balance between fluent
self-contained prose (MLP) and literal token retrieval (attention)**, and that balance is what
different tasks need. **MLP-heavy models sound like (a slightly bad) writer; attention-heavy models
sound like a lookup table that occasionally writes.** Neither is "better" — they are *different
instruments* — and the right one depends on whether you want a sentence or a recalled fact.

---

### Reproduce this appendix

- **Script:** `experiments/layout_schedule_2/gen_samples.py` (loads the best checkpoint of each
  variant, runs the four prompts at temp 0.8 / seed 7, writes `samples.json`).
- **Runs used (best checkpoint each):** under `/data/susanket/text/gpt/runs/` —
  `ratio_jumbo/{r41_10A38M_r2-s1, r31_12A36M_r2-s0, r21_16A32M_r1-s1, r11_24A24M_r2-s1,
  r12_32A16M_r2-s1, r13_36A12M_r0-s0, r14_38A10M_alt-s0}`,
  `ratio_pm_jumbo/pmr12_32A16M_alt-s1`, `ratio_synth_recall/r11_12A12M_alt-s1`.
- **Raw outputs:** `experiments/layout_schedule_2/samples.json`.
- **GPUs:** generated on the visible devices only (`CUDA_VISIBLE_DEVICES=1,2,0`, i.e. physical
  0/1/2); GPU 3 was left untouched.
