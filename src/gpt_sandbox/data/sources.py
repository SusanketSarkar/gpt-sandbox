"""Document sources for `gpt data curate`. Contract: fn(**kw) -> Iterator[str] of raw documents.
Add your own: @register("source", "mine")."""
from __future__ import annotations

import glob
import random
import re
import urllib.request
from pathlib import Path

from ..registry import get, register

CACHE = Path(".cache/downloads")


def split_docs(text: str, mode: str = "whole"):
    """whole | blank (paragraphs) | lines | regex:<pattern>"""
    if mode == "whole":
        parts = [text]
    elif mode == "blank":
        parts = re.split(r"\n\s*\n", text)
    elif mode == "lines":
        parts = text.splitlines()
    elif mode.startswith("regex:"):
        parts = re.split(mode[6:], text)
    else:
        raise ValueError(f"doc_split must be whole|blank|lines|regex:<pat>, got {mode!r}")
    return [p for p in parts if p.strip()]


@register("source", "files")
def files(paths, doc_split: str = "whole", encoding: str = "utf-8"):
    """Local text files. `paths`: a glob or list of globs (** and ~ supported)."""
    paths = [paths] if isinstance(paths, str) else paths
    found = []
    for p in paths:
        found += sorted(glob.glob(str(Path(p).expanduser()), recursive=True))
    found = [f for f in found if Path(f).is_file()]
    if not found:
        raise FileNotFoundError(f"no files matched {paths}")
    for f in found:
        text = Path(f).read_text(encoding=encoding, errors="replace")
        yield from split_docs(text, doc_split)


@register("source", "url")
def url(url: str, doc_split: str = "whole"):
    """Download one text file (cached under .cache/downloads) and split it into docs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / re.sub(r"[^A-Za-z0-9_.-]", "_", url)[-150:]
    if not dest.exists():
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    yield from split_docs(dest.read_text(encoding="utf-8", errors="replace"), doc_split)


@register("source", "hf")
def hf(dataset: str, split: str = "train", text_field: str = "text", config: str | None = None,
       max_docs: int | None = None, streaming: bool = True):
    """HuggingFace datasets (pip install datasets). Example: dataset: roneneldan/TinyStories."""
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=streaming)
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            return
        yield ex[text_field]


@register("source", "synthetic")
def synthetic(task: str, n_docs: int = 2000, lines_per_doc: int = 64, seed: int = 0, **task_kw):
    """Procedural data from a registered `task` (modadd | copy | recall | yours)."""
    fn = get("task", task)
    rng = random.Random(seed)
    for _ in range(n_docs):
        yield "\n".join(fn(rng, **task_kw) for _ in range(lines_per_doc)) + "\n"
