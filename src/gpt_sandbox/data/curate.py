"""`gpt data curate`: sources -> filters -> dedup -> split -> tokenizer -> .bin shards. Run once, reuse everywhere."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from ..registry import build, get, parse_spec
from ..utils import now_iso, write_json
from . import filters, sources, synthetic  # noqa: F401  (register builtins)
from . import tokenizers as tk

CURATE_DEFAULTS = {
    "name": None,
    "dir": "data",
    "sources": [],
    "filters": [],
    "tokenizer": {"name": "char"},
    "val_fraction": 0.05,
    "dedup": True,
    "eos": True,
    "max_docs": None,
    "keep_docs": True,  # keep docs_*.jsonl (enables `gpt data retokenize` and cross-tokenizer eval)
    "plugins": [],
}


def _is_val(doc: str, frac: float) -> int:
    h = int(hashlib.md5(doc.encode("utf-8")).hexdigest()[:8], 16) % 10000
    return int(h < frac * 10000)


def _stage_a(spec, work: Path):
    """Stream sources through filters + dedup into a single jsonl. Returns (n_docs, split_flags, report)."""
    flt = []
    for f in spec["filters"]:
        name, kw = parse_spec(f, "filter")
        flt.append((name, get("filter", name), kw))
    report = {"sources": {}, "filters_dropped": {n: 0 for n, _, _ in flt}, "dedup_dropped": 0, "empty_dropped": 0}
    seen, flags, n = set(), bytearray(), 0
    with open(work / "_all.jsonl", "w") as out:
        for si, s in enumerate(spec["sources"]):
            sname, skw = parse_spec(s, "source")
            report["sources"][f"{si}:{sname}"] = 0
            for doc in get("source", sname)(**skw):
                if spec["max_docs"] and n >= spec["max_docs"]:
                    break
                keep = True
                for fname, fn, kw in flt:
                    r = fn(doc, **kw)
                    if r is False or r is None:
                        report["filters_dropped"][fname] += 1
                        keep = False
                        break
                    if isinstance(r, str):
                        doc = r
                if not keep:
                    continue
                if not doc.strip():
                    report["empty_dropped"] += 1
                    continue
                if spec["dedup"]:
                    h = hashlib.md5(doc.encode("utf-8")).digest()
                    if h in seen:
                        report["dedup_dropped"] += 1
                        continue
                    seen.add(h)
                out.write(json.dumps(doc) + "\n")
                flags.append(_is_val(doc, spec["val_fraction"]))
                report["sources"][f"{si}:{sname}"] += 1
                n += 1
    return n, flags, report


def _fix_split(spec, work: Path, n: int, flags: bytearray) -> int:
    """Tiny corpora (one big file, few docs): hash split can leave val empty -> fall back to tail split."""
    if n == 1:  # a single document: split by characters
        doc = json.loads((work / "_all.jsonl").read_text().splitlines()[0])
        cut = int(len(doc) * (1 - spec["val_fraction"]))
        with open(work / "_all.jsonl", "w") as f:
            f.write(json.dumps(doc[:cut]) + "\n" + json.dumps(doc[cut:]) + "\n")
        flags[:] = bytearray([0, 1])
        print("  single document -> split by characters (train head / val tail)")
        return 2
    if sum(flags) == 0:
        k = max(1, round(spec["val_fraction"] * n))
        for i in range(n - k, n):
            flags[i] = 1
        print(f"  hash split left val empty -> using the last {k} docs as val")
    if sum(flags) == n:
        raise ValueError("every document landed in val; lower val_fraction")
    return n


def _iter_all(work: Path, flags, want: int | None):
    with open(work / "_all.jsonl") as f:
        for i, line in enumerate(f):
            if want is None or flags[i] == want:
                yield json.loads(line)


def _encode_split(docs, tok, out_bin: Path, dtype, eos: bool, docs_out: Path | None, label: str):
    n_docs = n_chars = n_bytes = n_tok = 0
    buf: list[int] = []
    dfh = open(docs_out, "w") if docs_out else None
    with open(out_bin, "wb") as fh:
        for doc in docs:
            buf.extend(tok.encode(doc))
            if eos:
                buf.append(tok.eos_id)
            n_docs += 1
            n_chars += len(doc)
            n_bytes += len(doc.encode("utf-8"))
            if dfh:
                dfh.write(json.dumps(doc) + "\n")
            if len(buf) > 2_000_000:
                fh.write(np.asarray(buf, dtype=dtype).tobytes())
                n_tok += len(buf)
                buf = []
        if buf:
            fh.write(np.asarray(buf, dtype=dtype).tobytes())
            n_tok += len(buf)
    if dfh:
        dfh.close()
    print(f"  {label}: {n_docs:,} docs  {n_tok:,} tokens  ({n_bytes / max(1, n_tok):.2f} bytes/token)")
    return {"docs": n_docs, "tokens": n_tok, "chars": n_chars, "bytes": n_bytes}


def _write_meta(out: Path, spec, tok, dtype, splits, report=None, task_sep=None, extra=None):
    meta = {
        "name": spec["name"], "created": now_iso(), "dtype": np.dtype(dtype).name,
        "tokenizer": {"spec": spec["tokenizer"], "vocab_size": tok.vocab_size, "eos_id": tok.eos_id,
                      "fingerprint": tk.fingerprint(tok)},
        "eos_between_docs": spec["eos"], "splits": splits, "val_fraction": spec["val_fraction"],
        "task_sep": task_sep, "report": report, "keep_docs": spec["keep_docs"],
    }
    meta.update(extra or {})
    write_json(out / "meta.json", meta)
    (out / "curate.yaml").write_text(yaml.safe_dump({k: v for k, v in spec.items()}, sort_keys=False))


def curate(spec: dict, overwrite: bool = False) -> Path:
    spec = {**CURATE_DEFAULTS, **spec}
    bad = set(spec) - set(CURATE_DEFAULTS)
    if bad:
        raise ValueError(f"unknown curate keys {sorted(bad)}; valid: {sorted(CURATE_DEFAULTS)}")
    if not spec["name"]:
        raise ValueError("curate spec needs a `name`")
    if not spec["sources"]:
        raise ValueError("curate spec needs at least one entry in `sources`")
    if isinstance(spec["tokenizer"], str):
        spec["tokenizer"] = tk.parse_tokenizer_arg(spec["tokenizer"])
    out = Path(spec["dir"]) / spec["name"]
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists. Use --overwrite to rebuild, or pick another --name.")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    try:
        print(f"[1/3] reading sources + filters -> {out}")
        n, flags, report = _stage_a(spec, out)
        if n == 0:
            raise ValueError(f"no documents left after filtering. Report: {report}")
        n = _fix_split(spec, out, n, flags)
        print(f"  kept {n:,} docs ({int(sum(flags)):,} val) | dropped: filters={report['filters_dropped']} "
              f"dedup={report['dedup_dropped']}")

        print(f"[2/3] fitting tokenizer {spec['tokenizer']}")
        tok = build("tokenizer", spec["tokenizer"])
        tok.fit(_iter_all(out, flags, 0))
        dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
        print(f"  vocab_size={tok.vocab_size}")

        print("[3/3] tokenizing")
        keep = spec["keep_docs"]
        splits = {
            "train": _encode_split(_iter_all(out, flags, 0), tok, out / "train.bin", dtype, spec["eos"],
                                   out / "docs_train.jsonl" if keep else None, "train"),
            "val": _encode_split(_iter_all(out, flags, 1), tok, out / "val.bin", dtype, spec["eos"],
                                 out / "docs_val.jsonl" if keep else None, "val  "),
        }
        write_json(out / "tokenizer.json", tok.to_dict())
        task_sep = None
        for s in spec["sources"]:
            sname, skw = parse_spec(s, "source")
            if sname == "synthetic":
                task_sep = getattr(get("task", skw["task"]), "sep", None)
        _write_meta(out, spec, tok, dtype, splits, report, task_sep)
    finally:
        (out / "_all.jsonl").unlink(missing_ok=True)
    print(f"done -> {out}   next: gpt data inspect {spec['name']}")
    return out


def retokenize(src: str, new_name: str, tokenizer_spec, data_dir="data", overwrite=False) -> Path:
    """Re-encode an existing dataset's cleaned docs with a different tokenizer (no re-download / re-filter)."""
    from .dataset import Dataset

    ds = Dataset.load(src, data_dir)
    if isinstance(tokenizer_spec, str):
        tokenizer_spec = tk.parse_tokenizer_arg(tokenizer_spec)
    out = Path(data_dir) / new_name
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    spec = {**CURATE_DEFAULTS, **yaml.safe_load((ds.path / "curate.yaml").read_text()),
            "name": new_name, "tokenizer": tokenizer_spec, "dir": str(data_dir)}
    tok = build("tokenizer", tokenizer_spec)
    tok.fit(ds.docs("train"))
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    splits = {
        s: _encode_split(ds.docs(s), tok, out / f"{s}.bin", dtype, spec["eos"],
                         out / f"docs_{s}.jsonl", s) for s in ("train", "val")
    }
    write_json(out / "tokenizer.json", tok.to_dict())
    _write_meta(out, spec, tok, dtype, splits, None, ds.task_sep, {"retokenized_from": src})
    print(f"done -> {out}")
    return out
