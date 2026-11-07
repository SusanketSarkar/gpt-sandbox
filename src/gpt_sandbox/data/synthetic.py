"""Synthetic tasks: tiny algorithmic corpora that are great for architecture experiments
(does my attention variant do recall / copying / arithmetic?). Each task yields one line;
`fn.sep` marks where the answer starts so eval can compute exact-match accuracy.

Add your own: @register("task", "mytask"); def mytask(rng, **kw) -> str; mytask.sep = ">"
"""
from __future__ import annotations

import string

from ..registry import register


@register("task", "modadd")
def modadd(rng, p: int = 97):
    """'a+b=c' with c = (a+b) mod p. Tests memorisation/generalisation of a tiny algorithm."""
    a, b = rng.randrange(p), rng.randrange(p)
    return f"{a}+{b}={(a + b) % p}"


modadd.sep = "="


@register("task", "copy")
def copy_task(rng, min_len: int = 4, max_len: int = 16, alphabet: str = string.ascii_lowercase[:12]):
    """'xyz>xyz': copy a random string. Tests induction/copy ability."""
    s = "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))
    return f"{s}>{s}"


copy_task.sep = ">"


@register("task", "recall")
def recall(rng, min_pairs: int = 2, max_pairs: int = 8, keys: str = string.ascii_lowercase, values: str = string.digits):
    """Associative recall: 'a1b7c3?b>7' - key/value pairs, then a query key; answer = its value."""
    n = rng.randint(min_pairs, min(max_pairs, len(keys)))
    ks = rng.sample(keys, n)
    vs = [rng.choice(values) for _ in ks]
    q = rng.randrange(n)
    return "".join(k + v for k, v in zip(ks, vs)) + f"?{ks[q]}>{vs[q]}"


recall.sep = ">"
