"""A curated dataset on disk: data/<name>/{train.bin,val.bin,meta.json,tokenizer.json,docs_*.jsonl}."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from ..utils import read_json
from .tokenizers import fingerprint, tokenizer_from_dict


def list_datasets(data_dir="data") -> list[str]:
    d = Path(data_dir)
    return sorted(p.name for p in d.iterdir() if (p / "meta.json").exists()) if d.exists() else []


class Dataset:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = read_json(self.path / "meta.json")
        self.name = self.meta["name"]
        self._tok = None
        self._train = None
        self._val = None

    @classmethod
    def load(cls, name: str, data_dir="data") -> "Dataset":
        p = Path(data_dir) / name
        if not (p / "meta.json").exists():
            have = list_datasets(data_dir)
            raise FileNotFoundError(
                f"dataset {name!r} not found in {data_dir}/ (have: {have or 'none'}). "
                f"Create it with `gpt data curate --config configs/data/<x>.yaml`."
            )
        return cls(p)

    # ---- tokenizer / tokens ----
    @property
    def tokenizer(self):
        if self._tok is None:
            self._tok = tokenizer_from_dict(read_json(self.path / "tokenizer.json"))
        return self._tok

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.tokenizer)

    def _mm(self, split):
        f = self.path / f"{split}.bin"
        return np.memmap(f, dtype=np.dtype(self.meta["dtype"]), mode="r")

    @property
    def train(self):
        if self._train is None:
            self._train = self._mm("train")
        return self._train

    @property
    def val(self):
        if self._val is None:
            self._val = self._mm("val")
        return self._val

    @property
    def task_sep(self):
        return self.meta.get("task_sep")

    @property
    def tokens_per_byte(self) -> float:
        v = self.meta["splits"]["val"]
        return v["tokens"] / max(1, v["bytes"])

    # ---- batches ----
    def split_array(self, split):
        return {"train": self.train, "val": self.val}[split]

    def get_batch(self, split: str, batch_size: int, seq_len: int, rng: np.random.Generator):
        arr = self.split_array(split)
        hi = len(arr) - seq_len - 1
        if hi <= 0:
            raise ValueError(f"{split} split has only {len(arr)} tokens; need > seq_len+1 ({seq_len + 1}). "
                             f"Curate more data or lower model.seq_len.")
        ix = rng.integers(0, hi, size=batch_size)
        x = np.stack([arr[i : i + seq_len] for i in ix]).astype(np.int64)
        y = np.stack([arr[i + 1 : i + 1 + seq_len] for i in ix]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def chunks(self, split: str, seq_len: int, batch_size: int, max_batches: int | None = None):
        """Deterministic non-overlapping windows covering the split (identical for every run)."""
        arr = self.split_array(split)
        n = (len(arr) - 1) // seq_len
        if n < 1:
            raise ValueError(f"{split} split too small ({len(arr)} tokens) for seq_len={seq_len}")
        for bi, b in enumerate(range(0, n, batch_size)):
            if max_batches is not None and bi >= max_batches:
                return
            starts = np.arange(b, min(b + batch_size, n)) * seq_len
            x = np.stack([arr[s : s + seq_len] for s in starts]).astype(np.int64)
            y = np.stack([arr[s + 1 : s + 1 + seq_len] for s in starts]).astype(np.int64)
            yield torch.from_numpy(x), torch.from_numpy(y)

    # ---- raw docs / cross-tokenizer eval ----
    def docs(self, split: str):
        f = self.path / f"docs_{split}.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing (dataset curated with keep_docs: false)")
        with open(f) as fh:
            for line in fh:
                yield json.loads(line)

    def retokenized(self, tokenizer) -> "Dataset":
        """Same val docs, encoded with another tokenizer (so a run can be evaluated on any dataset)."""
        if fingerprint(tokenizer) == self.fingerprint:
            return self
        ds = copy.copy(self)
        ids = []
        for doc in self.docs("val"):
            ids.extend(tokenizer.encode(doc))
            ids.append(tokenizer.eos_id)
        ds._val = np.asarray(ids, dtype=np.int64)
        ds._tok = tokenizer
        ds.meta = copy.deepcopy(self.meta)
        ds.meta["splits"]["val"]["tokens"] = len(ids)
        ds.meta["retokenized"] = True
        return ds
