"""Tokenizers. Each dataset stores its tokenizer (tokenizer.json) so train/eval/query always agree.

Contract (to add your own: @register("tokenizer", "mine")):
    cls.fit(texts)             learn from an iterator of training docs (no-op if fixed)
    cls.encode(text) -> list[int]
    cls.decode(ids) -> str
    cls.to_dict() / cls.from_dict(d)   JSON-serialisable state
    cls.vocab_size, cls.eos_id
    cls.token_str(i) -> str    printable form of one token (for `gpt data inspect`)
"""
from __future__ import annotations

from ..registry import build, register
from ..utils import hash_obj

SHORTHANDS = {"char": {"name": "char"}, "byte": {"name": "byte"}, "gpt2": {"name": "tiktoken", "encoding": "gpt2"}}


def parse_tokenizer_arg(s: str) -> dict:
    """CLI shorthand: char | byte | gpt2 | bpe:4096"""
    if s in SHORTHANDS:
        return dict(SHORTHANDS[s])
    if s.startswith("bpe:"):
        return {"name": "hf_bpe", "vocab_size": int(s.split(":", 1)[1])}
    return {"name": s}


@register("tokenizer", "char")
class CharTokenizer:
    """One token per distinct character seen in training data (+ <unk>, <eos>)."""

    def __init__(self, chars: list[str] | None = None):
        self.chars = chars or []
        self._rebuild()

    def _rebuild(self):
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.unk_id = len(self.chars)
        self.eos_id = len(self.chars) + 1
        self.vocab_size = len(self.chars) + 2

    def fit(self, texts):
        seen = set()
        for t in texts:
            seen.update(t)
        self.chars = sorted(seen)
        self._rebuild()

    def encode(self, text):
        g = self.stoi.get
        return [g(c, self.unk_id) for c in text]

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            out.append(self.chars[i] if i < len(self.chars) else ("<eos>" if i == self.eos_id else "<unk>"))
        return "".join(out)

    def token_str(self, i):
        return repr(self.decode([i]))

    def to_dict(self):
        return {"name": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, d):
        return cls(d["chars"])


@register("tokenizer", "byte")
class ByteTokenizer:
    """UTF-8 bytes (0..255) + <eos>. No training; works on anything."""

    vocab_size = 257
    eos_id = 256

    def fit(self, texts):
        pass

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        b = bytes(int(i) for i in ids if int(i) < 256)
        return b.decode("utf-8", errors="replace")

    def token_str(self, i):
        return "<eos>" if i == 256 else repr(bytes([i]).decode("latin-1"))

    def to_dict(self):
        return {"name": "byte"}

    @classmethod
    def from_dict(cls, d):
        return cls()


@register("tokenizer", "tiktoken")
class TiktokenTokenizer:
    """OpenAI BPE vocabularies via tiktoken (pip install tiktoken). Default: gpt2 (50257 tokens)."""

    def __init__(self, encoding: str = "gpt2"):
        import tiktoken

        self.encoding = encoding
        self.enc = tiktoken.get_encoding(encoding)
        self.vocab_size = self.enc.n_vocab
        self.eos_id = self.enc.eot_token

    def fit(self, texts):
        pass

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids])

    def token_str(self, i):
        return "<eos>" if i == self.eos_id else repr(self.enc.decode([i]))

    def to_dict(self):
        return {"name": "tiktoken", "encoding": self.encoding}

    @classmethod
    def from_dict(cls, d):
        return cls(d["encoding"])


@register("tokenizer", "hf_bpe")
class HFByteLevelBPE:
    """Byte-level BPE trained on your corpus with HuggingFace `tokenizers` (pip install tokenizers)."""

    EOS = "<|eos|>"

    def __init__(self, vocab_size: int = 4096, state: str | None = None):
        from tokenizers import Tokenizer

        self.target_vocab = vocab_size
        self.tok = Tokenizer.from_str(state) if state else None
        self._sync()

    def _sync(self):
        if self.tok is not None:
            self.vocab_size = self.tok.get_vocab_size()
            self.eos_id = self.tok.token_to_id(self.EOS)

    def fit(self, texts):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab, special_tokens=[self.EOS],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
        )
        tok.train_from_iterator(texts, trainer)
        self.tok = tok
        self._sync()

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=False)

    def token_str(self, i):
        return repr(self.tok.decode([i], skip_special_tokens=False))

    def to_dict(self):
        return {"name": "hf_bpe", "vocab_size": self.target_vocab, "state": self.tok.to_str()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["vocab_size"], d["state"])


def tokenizer_from_dict(d: dict):
    spec = dict(d)
    name = spec.pop("name")
    from ..registry import get

    return get("tokenizer", name).from_dict({"name": name, **spec})


def fingerprint(tok) -> str:
    return hash_obj(tok.to_dict())
