from __future__ import annotations

import math
import re
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import components  # noqa: F401  (registers builtins)
from ..registry import build, get
from .build import layer_norm
from .context import Ctx
from .spec import layer_selected, select_spec

MODEL_SITES = ("embed", "final")  # inserts that live on the model, not on a block


class GPTOutput(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None  # ce + aux (what you backprop)
    ce: torch.Tensor | None  # plain next-token cross-entropy
    aux: dict
    ctx: Ctx


class GPT(nn.Module):
    """Decoder-only transformer assembled from registered components. See README "Mental model"."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.vocab_size, "model.vocab_size must be set (it is filled from the dataset by `gpt train`)"
        self.cfg = cfg
        V, d, n = cfg.vocab_size, cfg.d_model, cfg.n_layer
        self.tok_emb = nn.Embedding(V, d)
        self.posenc = build("posenc", select_spec(cfg.posenc, 0, n, "model.posenc"), cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [build("block", select_spec(cfg.block, i, n, "model.block"), cfg, i) for i in range(n)]
        )
        self.model_inserts = nn.ModuleDict({s: nn.ModuleList() for s in MODEL_SITES})
        self._attach_inserts()
        self.final_norm = layer_norm(cfg, n - 1, field="final_norm")
        self.lm_head = nn.Linear(d, V, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    # ---- inserts -------------------------------------------------------------------------------
    def _attach_inserts(self):
        n = self.cfg.n_layer
        for spec in self.cfg.inserts or []:
            spec = dict(spec)
            site = spec.pop("site", None)
            layers = spec.pop("layers", "all")
            if site is None or "name" not in spec:
                raise ValueError(f"each model.inserts entry needs `site` and `name`, got {spec}")
            if site in MODEL_SITES:
                self.model_inserts[site].append(self._make_insert(spec, -1, site))
                continue
            matched = False
            for i, blk in enumerate(self.blocks):
                if not layer_selected(layers, i, n):
                    continue
                if site not in blk.SITES:
                    raise ValueError(f"block {type(blk).__name__} has no insert site {site!r}; "
                                     f"block sites: {blk.SITES}, model sites: {MODEL_SITES}")
                if site not in blk.inserts:
                    blk.inserts[site] = nn.ModuleList()
                blk.inserts[site].append(self._make_insert(dict(spec), i, site))
                matched = True
            if not matched:
                raise ValueError(f"insert {spec['name']!r}: `layers: {layers}` matched no layer")

    def _make_insert(self, spec, layer_idx, site):
        m = build("insert", spec, self.cfg, layer_idx, site)
        for k, v in (("layer_idx", layer_idx), ("site", site)):  # convenience: always available on inserts
            if not hasattr(m, k):
                setattr(m, k, v)
        return m

    # ---- init ----------------------------------------------------------------------------------
    def _init_weights(self):
        std, n = self.cfg.init_std, self.cfg.n_layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, std)
        for m in self.modules():  # residual-output projections: GPT-2 style depth scaling
            if getattr(m, "_resid_out", False) and isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, std / math.sqrt(2 * n))
        for m in self.modules():  # custom per-module init has the last word
            if m is not self and hasattr(m, "init_weights") and callable(m.init_weights):
                m.init_weights()

    # ---- forward -------------------------------------------------------------------------------
    def readout(self, x):
        """residual stream -> logits (used by forward and by logit-lens style probes)."""
        return self.lm_head(self.final_norm(x))

    def _inserts(self, site, x, ctx):
        for m in self.model_inserts[site]:
            x = m(x, ctx)
        return x

    def forward(self, idx, targets=None, collect: bool = False, capture: bool = False) -> GPTOutput:
        B, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"sequence length {T} exceeds model.seq_len {self.cfg.seq_len}")
        pos = torch.arange(T, device=idx.device)
        ctx = Ctx(self.cfg, pos, training=self.training, collect=collect, capture=capture)
        ctx.posenc = self.posenc
        x = self.drop(self.posenc.embed(self.tok_emb(idx), pos))
        x = self._inserts("embed", x, ctx)
        ctx.x0 = x
        for i, blk in enumerate(self.blocks):
            ctx.layer_idx = i
            x = blk(x, ctx)
            if capture:
                ctx.resid.append(x)
            if collect:
                ctx.log(f"resid_rms/L{i}", x.float().pow(2).mean().sqrt())
        x = self._inserts("final", x, ctx)
        logits = self.readout(x)
        loss = ce = None
        if targets is not None:
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            loss = ce
            for v in ctx.aux.values():
                loss = loss + v
        return GPTOutput(logits, loss, ce, ctx.aux, ctx)

    # ---- bookkeeping ---------------------------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights counted once by .parameters()
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            wpe = getattr(self.posenc, "wpe", None)
            if wpe is not None:
                n -= wpe.weight.numel()
        return n

    def param_breakdown(self) -> list[tuple[str, int, int]]:
        """(pattern, count, n_params) grouping e.g. blocks.0.attn.qkv.weight and blocks.1.attn.qkv.weight."""
        groups: dict[str, list[int]] = {}
        for name, p in self.named_parameters():
            key = re.sub(r"\.\d+\.", ".*.", name)
            g = groups.setdefault(key, [0, 0])
            g[0] += 1
            g[1] += p.numel()
        return [(k, v[0], v[1]) for k, v in groups.items()]

    def flops_per_token(self, seq_len: int | None = None) -> float:
        """Rough training FLOPs/token: 6 * non-embedding params + attention score/value terms."""
        T = seq_len or self.cfg.seq_len
        n_ne = self.num_params(non_embedding=True)
        attn = 12 * self.cfg.n_layer * self.cfg.d_model * T  # fwd+bwd of QK^T and AV (dense causal upper bound)
        head = 6 * self.cfg.vocab_size * self.cfg.d_model
        return 6 * n_ne + attn + head


def build_model(model_cfg) -> GPT:
    return GPT(model_cfg)
