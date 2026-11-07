"""Document filters/normalisers used by `gpt data curate`.
Contract: fn(doc: str, **kw) -> True (keep) | False (drop) | str (replace doc)."""
from __future__ import annotations

from ..registry import register


@register("filter", "min_chars")
def min_chars(doc, n: int = 20):
    return len(doc) >= n


@register("filter", "max_chars")
def max_chars(doc, n: int = 100_000, truncate: bool = False):
    if len(doc) <= n:
        return True
    return doc[:n] if truncate else False


@register("filter", "ascii_ratio")
def ascii_ratio(doc, min: float = 0.95):
    return sum(c < "\x80" for c in doc) / max(1, len(doc)) >= min


@register("filter", "alpha_ratio")
def alpha_ratio(doc, min: float = 0.5):
    return sum(c.isalpha() or c.isspace() for c in doc) / max(1, len(doc)) >= min


@register("filter", "strip")
def strip(doc):
    return doc.strip()


@register("filter", "lowercase")
def lowercase(doc):
    return doc.lower()


@register("filter", "collapse_whitespace")
def collapse_whitespace(doc):
    import re

    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", doc))
