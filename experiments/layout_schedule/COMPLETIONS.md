# Side-by-side next-word quality: which layout reads the best?

All ten Family-1 **jumbo** layouts (48 sublayers, 80.1M params, best checkpoint, seed 0),
run on the **same prompts**. This is the qualitative face of the `edu_mix` results in
`BLOG.md` / `BLOG.html` — here you can read what each architecture actually predicts.

**How to read this.**

- **Top-5** — the model's belief distribution over the next token (probability in %). A
  sharp, confident, *correct* top-1 is what you want; a smeared distribution is hesitation.
- **Greedy** — the argmax path, 16 new tokens. Deterministic: it is the model's single
  best-guess narrative, no sampling noise.
- **Canonical NLL** — teacher-forced average nats/token (lower = better) on the canonical
  correct continuation, plus the whole-string probability. The clean quantitative score.
- **Longer prose** — a 40-token sampled continuation (temperature 0.8, top-k 50, fixed seed)
  to show the *feel* and diversity, not just the argmax.

Models are ordered best→worst on the overall `edu_mix` validation loss from the main sweep,
so you can watch quality degrade down the table.

---

## “The sun rises in the”

*local knowledge (control) — canonical continuation: **east.**.*

### Top-5 next tokens

| model | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| baseline (AMAM…) | ` sky` (34%) | ` morning` (4%) | ` middle` (3%) | ` north` (2%) | ` night` (2%) |
| A · MLP ends | ` sky` (59%) | ` morning` (6%) | ` middle` (3%) | ` evening` (3%) | ` night` (3%) |
| B · attn ends | ` sky` (21%) | ` middle` (4%) | ` morning` (4%) | ` south` (3%) | ` north` (3%) |
| C · MLP-only ends | ` sky` (72%) | ` morning` (3%) | ` evening` (2%) | ` middle` (2%) | ` east` (1%) |
| D · attn-only ends | ` sky` (21%) | ` middle` (5%) | ` morning` (4%) | ` night` (3%) | ` north` (3%) |
| sandwich | ` sky` (37%) | ` morning` (3%) | ` east` (3%) | ` south` (3%) | ` night` (2%) |
| rand0 | ` sky` (28%) | ` morning` (5%) | ` middle` (4%) | ` evening` (3%) | ` night` (2%) |
| rand1 | ` sky` (43%) | ` middle` (5%) | ` morning` (4%) | ` east` (3%) | ` west` (2%) |
| rand2 | ` sky` (49%) | ` morning` (5%) | ` night` (3%) | ` middle` (3%) | ` evening` (3%) |
| rand3 | ` sky` (18%) | ` south` (7%) | ` north` (5%) | ` west` (5%) | ` east` (5%) |

### Greedy completion (16 tokens)

| model | completion |
|---|---|
| baseline (AMAM…) | sky, and the sky is bright. It is a bright, sunny day |
| A · MLP ends | sky and the sky is blue. It is a beautiful day. One |
| B · attn ends | sky, and the sun rises in the sky. The sun rises in |
| C · MLP-only ends | sky and the sky is bright. It is very bright and bright. |
| D · attn-only ends | sky, and the moon is in the sky. The sun is shining bright |
| sandwich | sky, and the sky is bright. The sky is bright and the sky is |
| rand0 | sky, and the sun rises in the sky. It is the most beautiful |
| rand1 | sky and the sky is shining. It is a beautiful day. |
| rand2 | sky and the sky is bright. It is a beautiful day. One |
| rand3 | sky and the sky is shining brightly. The sky is bright and the |

### Canonical-continuation score (lower NLL = better)

| model | NLL (nats/tok) | P(canonical) |
|---|---|---|
| baseline (AMAM…) | 2.580 | 0.0% |
| A · MLP ends | 2.676 | 0.0% |
| B · attn ends | 2.289 | 0.1% |
| C · MLP-only ends | 2.276 | 0.1% |
| D · attn-only ends | 2.633 | 0.0% |
| sandwich | 1.999 | 0.2% |
| rand0 | 2.304 | 0.1% |
| rand1 | 2.453 | 0.1% |
| rand2 | 2.705 | 0.0% |
| rand3 | 2.116 | 0.2% |


## “Spring, summer, autumn,”

*sequence completion — canonical continuation: **winter.**.*

### Top-5 next tokens

| model | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| baseline (AMAM…) | ` and` (29%) | ` summer` (20%) | ` spring` (7%) | ` winter` (6%) | ` fall` (3%) |
| A · MLP ends | ` and` (53%) | ` summer` (7%) | ` winter` (5%) | ` spring` (5%) | ` or` (2%) |
| B · attn ends | ` and` (27%) | ` aut` (14%) | ` spring` (12%) | ` summer` (9%) | ` fall` (5%) |
| C · MLP-only ends | ` and` (26%) | ` spring` (12%) | ` winter` (12%) | ` summer` (10%) | ` aut` (5%) |
| D · attn-only ends | ` summer` (25%) | ` spring` (16%) | ` winter` (16%) | ` and` (11%) | ` aut` (2%) |
| sandwich | ` aut` (21%) | ` summer` (19%) | ` and` (14%) | ` spring` (12%) | ` winter` (12%) |
| rand0 | ` and` (28%) | ` winter` (14%) | ` summer` (13%) | ` spring` (8%) | ` aut` (7%) |
| rand1 | ` and` (24%) | ` spring` (8%) | ` summer` (8%) | ` aut` (7%) | ` winter` (5%) |
| rand2 | ` and` (26%) | ` summer` (16%) | ` winter` (14%) | ` fall` (11%) | ` spring` (8%) |
| rand3 | ` summer` (28%) | ` spring` (17%) | ` and` (13%) | ` aut` (10%) | ` fall` (9%) |

### Greedy completion (16 tokens)

| model | completion |
|---|---|
| baseline (AMAM…) | and summer are all the same. The summer is the most important time of the |
| A · MLP ends | and autumn are the most common winter in the United States. The winter is |
| B · attn ends | and spring are all the same. The summer is the time of year when the |
| C · MLP-only ends | and winter are the most common of the summer. The winter is the most common |
| D · attn-only ends | summer, summer, summer, summer, summer, summer, summer, summer, |
| sandwich | autumn, autumn, autumn, autumn, autumn, aut |
| rand0 | and winter are the most common winter weather in the United States. The most common |
| rand1 | and autumn are the most common types of autumn. The autumn is |
| rand2 | and winter are the most important things to know about the weather. The weather |
| rand3 | summer, summer, summer, summer, summer, summer, summer, summer, |

### Canonical-continuation score (lower NLL = better)

| model | NLL (nats/tok) | P(canonical) |
|---|---|---|
| baseline (AMAM…) | 2.455 | 0.1% |
| A · MLP ends | 2.232 | 0.1% |
| B · attn ends | 2.208 | 0.1% |
| C · MLP-only ends | 1.995 | 0.3% |
| D · attn-only ends | 2.095 | 0.2% |
| sandwich | 1.942 | 0.3% |
| rand0 | 2.204 | 0.1% |
| rand1 | 2.421 | 0.1% |
| rand2 | 2.352 | 0.1% |
| rand3 | 2.309 | 0.1% |


## “The capital of France is”

*factual recall — canonical continuation: **Paris.**.*

### Top-5 next tokens

| model | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| baseline (AMAM…) | ` the` (26%) | ` a` (8%) | ` located` (7%) | ` in` (4%) | ` an` (2%) |
| A · MLP ends | ` the` (17%) | ` a` (10%) | ` located` (10%) | ` in` (3%) | ` known` (3%) |
| B · attn ends | ` the` (12%) | ` a` (7%) | ` located` (4%) | ` in` (3%) | ` known` (3%) |
| C · MLP-only ends | ` the` (19%) | ` a` (6%) | ` located` (4%) | ` in` (3%) | ` called` (2%) |
| D · attn-only ends | ` the` (11%) | ` a` (10%) | ` located` (5%) | ` in` (2%) | ` not` (2%) |
| sandwich | ` the` (16%) | ` located` (9%) | ` a` (5%) | ` in` (4%) | ` divided` (2%) |
| rand0 | ` the` (24%) | ` a` (7%) | ` located` (5%) | ` in` (3%) | ` an` (2%) |
| rand1 | ` the` (24%) | ` a` (6%) | ` located` (6%) | ` in` (3%) | ` C` (1%) |
| rand2 | ` the` (17%) | ` a` (8%) | ` located` (5%) | ` in` (4%) | ` one` (3%) |
| rand3 | ` the` (16%) | ` a` (8%) | ` in` (7%) | ` located` (6%) | ` known` (2%) |

### Greedy completion (16 tokens)

| model | completion |
|---|---|
| baseline (AMAM…) | the capital of France. It is the capital of France. It is the capital |
| A · MLP ends | the capital of the United States. It is the capital of the United States. |
| B · attn ends | the capital of the United States. It is the capital of France. It is |
| C · MLP-only ends | the capital of France. The capital of France is the capital of France. The |
| D · attn-only ends | the capital of the United Kingdom. The capital of France is the capital of the |
| sandwich | the largest and most populous country in the world. It is the largest |
| rand0 | the capital of France. It is the capital of France. It is the capital |
| rand1 | the capital of France. It is the capital of France. It is the capital |
| rand2 | the capital of France. It is the capital of France. It is the capital |
| rand3 | the capital of the United States. It is the capital of the United States, |

### Canonical-continuation score (lower NLL = better)

| model | NLL (nats/tok) | P(canonical) |
|---|---|---|
| baseline (AMAM…) | 3.605 | 0.0% |
| A · MLP ends | 3.684 | 0.0% |
| B · attn ends | 3.703 | 0.0% |
| C · MLP-only ends | 3.497 | 0.0% |
| D · attn-only ends | 3.652 | 0.0% |
| sandwich | 4.016 | 0.0% |
| rand0 | 3.614 | 0.0% |
| rand1 | 3.495 | 0.0% |
| rand2 | 3.567 | 0.0% |
| rand3 | 3.775 | 0.0% |


## “If you have 3 apples and someone gives you 4 more, you have”

*arithmetic — canonical continuation: **7**.*

### Top-5 next tokens

| model | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| baseline (AMAM…) | ` to` (16%) | ` 3` (16%) | ` 4` (11%) | ` 2` (7%) | ` a` (7%) |
| A · MLP ends | ` 3` (15%) | ` 4` (10%) | ` to` (10%) | ` 2` (7%) | ` a` (7%) |
| B · attn ends | ` 3` (11%) | ` a` (11%) | ` 4` (7%) | ` 2` (6%) | ` to` (5%) |
| C · MLP-only ends | ` to` (20%) | ` a` (14%) | ` 3` (5%) | ` 4` (4%) | ` 5` (4%) |
| D · attn-only ends | ` 4` (23%) | ` 5` (16%) | ` 3` (15%) | ` to` (6%) | ` 6` (4%) |
| sandwich | ` a` (10%) | ` 2` (7%) | ` 4` (7%) | ` 3` (6%) | ` 5` (5%) |
| rand0 | ` to` (14%) | ` 3` (10%) | ` 4` (9%) | ` a` (8%) | ` 5` (5%) |
| rand1 | ` to` (24%) | ` a` (12%) | ` the` (5%) | ` 3` (4%) | ` probably` (3%) |
| rand2 | ` 3` (12%) | ` a` (9%) | ` 4` (8%) | ` to` (7%) | ` 2` (6%) |
| rand3 | ` to` (10%) | ` 3` (9%) | ` 4` (9%) | ` a` (7%) | ` 2` (6%) |

### Greedy completion (16 tokens)

| model | completion |
|---|---|
| baseline (AMAM…) | to take a look at the apple. The apple is a very |
| A · MLP ends | 3 apples. The best way to get the best apples is to |
| B · attn ends | 3 apples and you have 3 apples. - 1 apple |
| C · MLP-only ends | to take a few of the apples and then you can get a little more |
| D · attn-only ends | 4 more. If you have 3 more, you have 4 more. |
| sandwich | a better chance of getting a good one. The best way to get a |
| rand0 | to eat them. You can also eat them in a variety of ways. |
| rand1 | to be careful. - If you have 2 apples and one apple |
| rand2 | 3 apples. The apples are made from a mixture of apples |
| rand3 | to wait for the apple to grow. You can also add the app |

### Canonical-continuation score (lower NLL = better)

| model | NLL (nats/tok) | P(canonical) |
|---|---|---|
| baseline (AMAM…) | 4.276 | 1.4% |
| A · MLP ends | 4.269 | 1.4% |
| B · attn ends | 4.315 | 1.3% |
| C · MLP-only ends | 4.808 | 0.8% |
| D · attn-only ends | 4.389 | 1.2% |
| sandwich | 3.979 | 1.9% |
| rand0 | 4.569 | 1.0% |
| rand1 | 5.562 | 0.4% |
| rand2 | 4.257 | 1.4% |
| rand3 | 4.009 | 1.8% |


## “The dog chased the cat around the yard. The cat ran away because the”

*coreference (discriminator) — canonical continuation: **dog**.*

### Top-5 next tokens

| model | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| baseline (AMAM…) | ` cat` (44%) | ` dog` (23%) | ` sun` (6%) | ` house` (1%) | ` sky` (1%) |
| A · MLP ends | ` cat` (58%) | ` dog` (21%) | ` sun` (2%) | ` house` (1%) | ` cats` (1%) |
| B · attn ends | ` cat` (65%) | ` dog` (24%) | ` sun` (1%) | ` ball` (0%) | ` car` (0%) |
| C · MLP-only ends | ` dog` (47%) | ` cat` (22%) | ` sun` (3%) | ` park` (2%) | ` water` (2%) |
| D · attn-only ends | ` dog` (46%) | ` cat` (38%) | ` sun` (1%) | ` door` (1%) | ` cats` (1%) |
| sandwich | ` cat` (39%) | ` dog` (25%) | ` house` (3%) | ` sun` (2%) | ` park` (2%) |
| rand0 | ` cat` (59%) | ` dog` (29%) | ` cats` (1%) | ` fur` (1%) | ` sun` (1%) |
| rand1 | ` dog` (55%) | ` cat` (10%) | ` house` (2%) | ` car` (2%) | ` ball` (1%) |
| rand2 | ` dog` (49%) | ` cat` (28%) | ` sun` (4%) | ` owner` (1%) | ` cats` (1%) |
| rand3 | ` dog` (49%) | ` cat` (22%) | ` owner` (3%) | ` sun` (2%) | ` grass` (1%) |

### Greedy completion (16 tokens)

| model | completion |
|---|---|
| baseline (AMAM…) | cat was so scared. The cat was scared and ran away. The |
| A · MLP ends | cat was scared. The cat was scared, but the cat was brave. |
| B · attn ends | cat was scared. The cat was sad and didn't know what to do. |
| C · MLP-only ends | dog was scared. The cat was very scared and didn't know what to do |
| D · attn-only ends | dog was scared. The dog chased the cat away. The dog chased |
| sandwich | cat was scared. The cat was scared too. The cat was scared too. |
| rand0 | cat was scared. The cat was scared and ran away. The cat was sad |
| rand1 | dog was scared. The dog ran away and the dog ran away. |
| rand2 | dog was so big and strong. The cat was scared and ran away. |
| rand3 | dog was scared. The dog ran away and the dog ran away. |

### Canonical-continuation score (lower NLL = better)

| model | NLL (nats/tok) | P(canonical) |
|---|---|---|
| baseline (AMAM…) | 1.335 | 6.9% |
| A · MLP ends | 1.647 | 3.7% |
| B · attn ends | 1.362 | 6.6% |
| C · MLP-only ends | 1.262 | 8.0% |
| D · attn-only ends | 1.093 | 11.2% |
| sandwich | 1.264 | 8.0% |
| rand0 | 1.251 | 8.2% |
| rand1 | 0.580 | 31.4% |
| rand2 | 0.874 | 17.4% |
| rand3 | 0.995 | 13.7% |


## “The cat sat on the mat. The dog sat on the”

*induction-head probe — canonical continuation: **mat**.*

### Top-5 next tokens

| model | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| baseline (AMAM…) | ` mat` (97%) | ` cou` (0%) | ` floor` (0%) | ` table` (0%) | ` cat` (0%) |
| A · MLP ends | ` mat` (97%) | ` cou` (1%) | ` ben` (1%) | ` table` (1%) | ` floor` (0%) |
| B · attn ends | ` mat` (98%) | ` floor` (0%) | ` grass` (0%) | ` screen` (0%) | ` table` (0%) |
| C · MLP-only ends | ` mat` (93%) | ` cou` (1%) | ` floor` (1%) | ` ben` (1%) | ` table` (0%) |
| D · attn-only ends | ` mat` (85%) | ` table` (4%) | ` floor` (2%) | ` grass` (1%) | ` cou` (1%) |
| sandwich | ` mat` (98%) | ` floor` (0%) | ` pat` (0%) | ` ground` (0%) | ` table` (0%) |
| rand0 | ` mat` (83%) | ` cou` (5%) | ` table` (2%) | ` floor` (2%) | ` grass` (1%) |
| rand1 | ` mat` (70%) | ` cou` (8%) | ` floor` (3%) | ` grass` (2%) | ` ben` (2%) |
| rand2 | ` mat` (87%) | ` cou` (4%) | ` grass` (1%) | ` ben` (1%) | ` floor` (1%) |
| rand3 | ` mat` (94%) | ` floor` (1%) | ` ben` (1%) | ` ground` (0%) | ` table` (0%) |

### Greedy completion (16 tokens)

| model | completion |
|---|---|
| baseline (AMAM…) | mat. The cat was happy. The cat sat on the mat. |
| A · MLP ends | mat. The cat was happy. The cat and the cat played together. They |
| B · attn ends | mat. The cat was happy. The cat was happy too. The cat was |
| C · MLP-only ends | mat. The cat was happy. The cat liked the cat. The cat liked |
| D · attn-only ends | mat. The mat was soft and warm. The cat liked to play with the |
| sandwich | mat and looked at the cat. The cat was happy and said, "Thank |
| rand0 | mat. The cat was happy. The cat liked the mat. The cat liked |
| rand1 | mat. The cat sat on the mat. The cat sat on the mat. |
| rand2 | mat. The cat was happy. The cat liked to play with the cat. |
| rand3 | mat. The cat was happy. It wanted to play with the mat. |

### Canonical-continuation score (lower NLL = better)

| model | NLL (nats/tok) | P(canonical) |
|---|---|---|
| baseline (AMAM…) | 0.035 | 96.6% |
| A · MLP ends | 0.035 | 96.6% |
| B · attn ends | 0.025 | 97.6% |
| C · MLP-only ends | 0.077 | 92.6% |
| D · attn-only ends | 0.159 | 85.3% |
| sandwich | 0.016 | 98.4% |
| rand0 | 0.182 | 83.4% |
| rand1 | 0.356 | 70.1% |
| rand2 | 0.141 | 86.8% |
| rand3 | 0.059 | 94.3% |

---

## Longer prose — the *feel*

Two prompts, 40 sampled tokens each (temperature 0.8, top-k 50, seed 1234),
so every model sees the identical prompt and the same sampling seed. Watch how the same
seed produces different text from different layouts.


### Prompt: “The experiment was designed to measure the effect of temperature on the…”

- **baseline (AMAM…):** body of one of the most sensitive, non-fat tissue cells in the body. "The results and results of the study were the first to show that the human body temperature was very high
- **A · MLP ends:** shape of the glass and evaluate the influence of gravity on the density of the glass. "The results and results show that the glass is very sensitive to temperature," explained David J. Wils
- **B · attn ends:** body of water on a number of surfaces with a change in the rate of heat waves. "The results and results of the study were the same as the results of the same experiment," said R
- **C · MLP-only ends:** body's energy levels and the energy, which is a critical tool for measuring the body's energy levels. The results show that the subjects who spent the first few minutes of the experiment were very sensitive
- **D · attn-only ends:** Earth's atmosphere and to evaluate the influence of gravity on the Earth's surface during the course of a study and to evaluate the relationship between the Earth's rotation and the planet's surface.
- **sandwich:** body of water particles and the energy of gases and in the air. The experiment was performed on a different surface in the atmosphere and measured the amount of moisture in the air. Instead
- **rand0:** body of a person. The experiments showed that when a person is overweight, the body is at a different and important role in the body. The body is divided into two major groups: -
- **rand1:** body and its structure and function. It tested how many different colours and colours were different than the different colours and colouring colors. The experiments were designed specifically to measure the amount of
- **rand2:** Earth's atmosphere and to evaluate the influence of gravity on the Earth's surface. The idea was to measure and measure the effect of temperature on the Earth's surface. The experiment was very similar
- **rand3:** body of humans and bears on the human body and in the animal kingdom. The idea was “to give us the power of the human body so that we can’t wait to experience them


### Prompt: “The dog chased the cat around the yard. The cat ran away because the…”

- **baseline (AMAM…):** car didn't move. But the dog was happy and ran around the yard. The cat ran and jumped around and chased the cat. The dog chased the cat and ran away.
- **A · MLP ends:** lion was sad. He had been so excited that he ran away and was never seen again. One day, the lion was walking outside when the sun came up. He saw the lion and
- **B · attn ends:** cat didn't play. But the dog was happy and ran away. After playing with the cat, the dog heard a loud noise outside. He had no way to help. It was a
- **C · MLP-only ends:** car didn't move. But the car was very strong. It bumped into a big puddle and the ball went into a tree. In the forest, the dog was very
- **D · attn-only ends:** cat didn't play with the other dogs. The cat ran away and was never seen again. One day, the cat saw a little bird. The bird asked the cat, "Can you
- **sandwich:** car was sad. Suddenly, the car stopped and got wet. Then, the cat ran to the car and said, "Mommy, can you help me clean up the car?" Mommy smiled
- **rand0:** cat didn't play with the other animal. The cat ran and banged on the ground. One day, the cat was running and chasing the cat. The cat was so happy.
- **rand1:** car didn't move. But the dog ran and got in the car. The dog ran as fast as he could, but the cat was getting very tired. The cat saw the dog and
- **rand2:** cat was sad and hurt. Soon, the dog met a wise old cat named Ben. Ben said, "Don't worry, dog. The dog won't hurt you."
- **rand3:** car didn't move. But the dog was faster and ran away. After a while, the dog said to the dog "this article the dog is very sleepy. Wait!"

---

## Scorecard

Best and worst layout on each prompt's canonical-continuation NLL (lower is better):

| prompt | best (NLL) | worst (NLL) | spread |
|---|---|---|---|
| local knowledge (control) | sandwich (1.999) | rand2 (2.705) | 0.706 |
| sequence completion | sandwich (1.942) | baseline (AMAM…) (2.455) | 0.513 |
| factual recall | rand1 (3.495) | sandwich (4.016) | 0.521 |
| arithmetic | sandwich (3.979) | rand1 (5.562) | 1.582 |
| coreference (discriminator) | rand1 (0.580) | A · MLP ends (1.647) | 1.067 |
| induction-head probe | sandwich (0.016) | rand1 (0.356) | 0.339 |

## What this shows

The honest read, in order of how visible the differences are:

1. **All ten layouts are fluent.** On a small model the per-prompt gaps are as modest as the
   `edu_mix` validation-loss gaps (±0.003 nats). No layout is *broken*; they differ in
   *character*, not in collapse. That is consistent with Family-1 being a small effect.

2. **Two visible failure modes stand out, and they are layout-specific:**
   - **D (attention-only ends) degenerates into repetition** on the sequence prompt —
     “summer, summer, summer, summer…” — a loop the average loss barely registers but the
     reader immediately sees.
   - **A (MLP-heavy ends) loops on the story prompts** (“The cat was scared. The cat was
     scared, but the cat was brave…”). A is also the weakest on the coreference prompt and,
     in the main sweep, the worst on both `edu_mix` and the one-hop `recall` task. Its
     qualitative and quantitative weaknesses point the same direction.

3. **The models agree where the next word is locally determined** (sun/east, seasons,
   France/Paris) and **split where it must be retrieved from earlier context** (the
   coreference prompt, where top-1 is *cat* for some layouts and *dog* for others). That
   agreement-vs-split pattern is exactly the “different jobs at different depths” signature:
   layout choice matters most for cross-token retrieval, least for local prediction.

4. **There is no single best layout per prompt.** `sandwich` wins the local/sequence/
   arithmetic prompts but loses factual recall; the randoms win coreference but lose
   arithmetic. The *winner depends on the task*, which is the qualitative echo of the main
   finding: the optimal schedule is task-dependent, and interleaving is the safe default.

> **Caveat:** a single prompt is a noisy probe. These completions are the *qualitative face*
> of the results; the reliable signal is the aggregate metric (validation loss, `recall` EM,
> induction accuracy) over thousands of examples, where the ordering of the layouts is stable.

