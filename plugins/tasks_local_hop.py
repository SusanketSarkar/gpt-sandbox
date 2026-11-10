"""Two more synthetic tasks for the layout-schedule study:

  local_rule  - the next symbol is a fixed function of the previous two: x_{t+1} = (x_t + x_{t-1}) mod k.
                Tests token-local feature computation (no long-range retrieval needed).
  hop2        - a shuffled chain of pairs x0->x1->x2->...; given x0, output the value two hops ahead.
                Tests multi-hop relational composition (harder than the 1-hop `recall` task).

Each yields one line; `fn.sep` marks where the answer starts (the task metric splits each line at
the first separator, so the separator must appear exactly once per line).
"""
from __future__ import annotations

import string

from gpt_sandbox import register


@register("task", "local_rule")
def local_rule(rng, k: int = 4, n_prompt: int = 4, n_answer: int = 6,
               alphabet: str = string.ascii_lowercase[:8]):
    """'a b c d > e f g h i j' with x_{t+1} = (x_t + x_{t-1}) mod k.

    Every answer symbol is a fixed function of the two preceding symbols, so the task is
    solvable purely from token-local computation - a probe of the 'early layers do local
    feature extraction' side of the layout hypothesis."""
    a, b = rng.randrange(k), rng.randrange(k)
    seq = [a, b]
    for _ in range(n_prompt + n_answer - 2):
        seq.append((seq[-1] + seq[-2]) % k)
    prompt = " ".join(alphabet[x] for x in seq[:n_prompt])
    answer = " ".join(alphabet[x] for x in seq[n_prompt:])
    return f"{prompt} > {answer}"


local_rule.sep = ">"


@register("task", "hop2")
def hop2(rng, min_links: int = 3, max_links: int = 6,
         alphabet: str = string.ascii_lowercase[:16]):
    """'ab bc cd ?a > c' - a chain x0->x1->x2->..., pairs shuffled in the line;
    query = chain start x0, answer = the value two hops ahead (x2).

    Composing two relations is a multi-hop task: a 1-hop retrieval (like `recall`) is
    insufficient, the model must follow x0 -> x1 -> x2."""
    n = rng.randint(min_links, max_links)
    chain = rng.sample(alphabet, n + 1)  # distinct symbols, so no pair collisions
    pairs = [f"{chain[i]}{chain[i + 1]}" for i in range(n)]
    rng.shuffle(pairs)
    return " ".join(pairs) + f" ?{chain[0]} > {chain[2]}"


hop2.sep = ">"
