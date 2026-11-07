"""`gpt new <kind> <name>` writes plugins/<name>.py: a WORKING copy of the builtin for that kind with
`# >>> EDIT` markers where you make your change. Copy-then-edit beats writing from a blank page."""
from __future__ import annotations

from pathlib import Path

HEADER = '''"""{doc}"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_sandbox import register, Ctx, sdpa_causal, repeat_kv, layer_component, BlockBase, PosEncBase  # noqa: F401

'''

T = {}

T["attention"] = '''
@register("attention", "__NAME__")
class __CLS__(nn.Module):
    """Starts as a copy of builtin `mha`. Config: model.attention: {name: __NAME__, <your kwargs>}"""

    def __init__(self, cfg, layer_idx, n_head: int | None = None):  # >>> EDIT: add kwargs here (e.g. n_kv_head: int = 2)
        super().__init__()
        d = cfg.d_model
        self.n_head = n_head or cfg.n_head
        self.head_dim = cfg.head_dim or d // self.n_head
        inner = self.n_head * self.head_dim
        self.qkv = nn.Linear(d, 3 * inner, bias=cfg.bias)  # >>> EDIT: e.g. separate q / k / v projections
        self.proj = nn.Linear(inner, d, bias=cfg.bias)
        self.proj._resid_out = True  # marks the residual-output projection for depth-scaled init
        self.attn_drop, self.resid_drop = cfg.dropout, nn.Dropout(cfg.dropout)

    def forward(self, x, ctx: Ctx):  # x: (B, T, d) -> (B, T, d)
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = ctx.posenc.rotate(q, k, ctx.positions)  # rope is applied here (no-op for other posenc)
        # >>> EDIT: replace the next line for custom attention (linear/recurrent/delta-rule/local/...).
        #     sdpa_causal accepts fewer k/v heads than q heads (it repeats them).
        y = sdpa_causal(q, k, v, self.attn_drop, ctx.training)
        y = y.transpose(1, 2).reshape(B, T, -1)
        ctx.log("__NAME__/y_rms", y.float().pow(2).mean().sqrt())  # shows up in metrics.jsonl on log steps
        return self.resid_drop(self.proj(y))
'''

T["mlp"] = '''
@register("mlp", "__NAME__")
class __CLS__(nn.Module):
    """Starts as a copy of builtin `swiglu`."""

    def __init__(self, cfg, layer_idx, d_ff: int | None = None):
        super().__init__()
        d = cfg.d_model
        h = d_ff or cfg.d_ff or int(round(8 * d / 3 / 32)) * 32
        self.gate = nn.Linear(d, h, bias=cfg.bias)
        self.up = nn.Linear(d, h, bias=cfg.bias)
        self.down = nn.Linear(h, d, bias=cfg.bias)
        self.down._resid_out = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, ctx: Ctx):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))  # >>> EDIT
'''

T["norm"] = '''
@register("norm", "__NAME__")
class __CLS__(nn.Module):
    """Starts as RMSNorm. Used for norm1/norm2 in each block and for the final norm."""

    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # >>> EDIT
        return (x * self.weight.float()).to(dt)
'''

T["posenc"] = '''
@register("posenc", "__NAME__")
class __CLS__(PosEncBase):
    """Two hooks. `embed` is added to token embeddings; `rotate` is applied to q/k inside attention
    (attention modules call ctx.posenc.rotate). Implement whichever you need."""

    def __init__(self, cfg):
        super().__init__()

    def embed(self, x, positions):  # x: (B, T, d)
        return x  # >>> EDIT e.g. x + self.table(positions)

    def rotate(self, q, k, positions):  # q, k: (B, H, T, head_dim)
        return q, k  # >>> EDIT e.g. rotary / ALiBi-style tricks
'''

T["residual"] = '''
@register("residual", "__NAME__")
class __CLS__(nn.Module):
    """HOW a sublayer output joins the residual stream. Built once per (layer, site) with
    site in {"attn", "mlp"}. Starts as plain `x + delta`."""

    def __init__(self, cfg, layer_idx, site):
        super().__init__()
        self.layer_idx, self.site = layer_idx, site
        # >>> EDIT: add parameters, e.g. self.g = nn.Parameter(torch.zeros(()))

    def forward(self, x, delta, ctx: Ctx):
        return x + delta  # >>> EDIT e.g. x + torch.sigmoid(self.g) * delta
'''

T["insert"] = '''
@register("insert", "__NAME__")
class __CLS__(nn.Module):
    """Transforms the residual stream at a named site; attach with config, no model edits:

        model.inserts: [{site: block_end, name: __NAME__, layers: all}]

    sites: block_start | after_attn | block_end   (per layer; `layers` takes "all", 3, [0,5], "::2", "-1", ...)
           embed | final                           (model level; `layers` ignored)
    Starts as identity."""

    def __init__(self, cfg, layer_idx, site):  # >>> EDIT: add kwargs, they come from the config entry
        super().__init__()
        self.layer_idx, self.site = layer_idx, site

    def forward(self, x, ctx: Ctx):
        # ctx.x0 = embedding output, ctx.store = scratch dict shared across layers,
        # ctx.log(name, tensor) = diagnostics, ctx.add_loss(name, tensor) = auxiliary loss.
        return x  # >>> EDIT
'''

T["block"] = '''
@register("block", "__NAME__")
class __CLS__(BlockBase):
    """Starts as a copy of `prenorm`. Call run_inserts(site, x, ctx) at each site you want config-attachable."""

    def __init__(self, cfg, layer_idx):
        super().__init__(cfg, layer_idx)
        self.norm1 = layer_norm(cfg, layer_idx)
        self.attn = layer_component("attention", cfg, layer_idx)
        self.res_attn = layer_component("residual", cfg, layer_idx, "attn")
        self.norm2 = layer_norm(cfg, layer_idx)
        self.mlp = layer_component("mlp", cfg, layer_idx)
        self.res_mlp = layer_component("residual", cfg, layer_idx, "mlp")

    def forward(self, x, ctx: Ctx):  # >>> EDIT e.g. parallel attn+mlp, post-norm, extra skip from ctx.x0
        x = self.run_inserts("block_start", x, ctx)
        x = self.res_attn(x, self.attn(self.norm1(x), ctx), ctx)
        x = self.run_inserts("after_attn", x, ctx)
        x = self.res_mlp(x, self.mlp(self.norm2(x), ctx), ctx)
        return self.run_inserts("block_end", x, ctx)


from gpt_sandbox import layer_norm  # noqa: E402
'''

T["metric"] = '''
@register("metric", "__NAME__")
def __NAME__(ev):
    """ev: EvalContext -> ev.model, ev.ds, ev.cfg, ev.device, ev.amp(), ev.seq_len, ev.tokenizer,
    ev.val_batches(n), ev.train_batches(n), ev.val_stats (per-token nll/top1/top5/entropy/conf), ev.notes.
    Return a flat {name: float}. Add `__NAME__.deep = True` to run only under `gpt eval --deep`."""
    s = ev.val_stats
    return {"__NAME___example": float(s.nll.std())}  # >>> EDIT
'''

T["optimizer"] = '''
@register("optimizer", "__NAME__")
def __NAME__(groups, lr, betas=(0.9, 0.95), eps=1e-8, **kw):
    """groups: [{params, lr_mult, weight_decay, name}] built from optim.groups. Return a torch Optimizer whose
    param groups carry `lr_mult` (the trainer sets lr = optim.lr * lr_mult * schedule each step)."""
    pg = [{"params": g["params"], "lr": lr * g["lr_mult"], "weight_decay": g["weight_decay"],
           "lr_mult": g["lr_mult"], "name": g["name"]} for g in groups]
    return torch.optim.AdamW(pg, lr=lr, betas=tuple(betas), eps=eps)  # >>> EDIT (extra kwargs: optim.kwargs)
'''

T["schedule"] = '''
@register("schedule", "__NAME__")
def __NAME__(step, max_steps, warmup_steps):
    """Return a multiplier on optim.lr for this step."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0  # >>> EDIT
'''

T["task"] = '''
@register("task", "__NAME__")
def __NAME__(rng, n: int = 8):
    """One synthetic line. `rng` is a random.Random. The answer starts after `__NAME__.sep`."""
    a = "".join(rng.choice("abcdefgh") for _ in range(n))
    return f"{a}>{a[::-1]}"  # >>> EDIT


__NAME__.sep = ">"
'''

T["filter"] = '''
@register("filter", "__NAME__")
def __NAME__(doc, min_words: int = 5):
    """Return True to keep, False to drop, or a str to replace the doc."""
    return len(doc.split()) >= min_words  # >>> EDIT
'''

T["source"] = '''
@register("source", "__NAME__")
def __NAME__(path: str):
    """Yield raw documents (strings)."""
    with open(path) as f:
        for line in f:
            yield line  # >>> EDIT
'''

USE = {
    "attention": "gpt train -c CONFIG --set 'model.attention={name: __NAME__}'      (per-layer: model.attention=[{layers: '3::4', name: __NAME__}, {layers: all, name: mha}])",
    "mlp": "--set 'model.mlp={name: __NAME__}'",
    "norm": "--set 'model.norm={name: __NAME__}'   (and model.final_norm)",
    "posenc": "--set 'model.posenc={name: __NAME__}'",
    "residual": "--set 'model.residual={name: __NAME__}'",
    "insert": "--set 'model.inserts=[{site: block_end, name: __NAME__}]'",
    "block": "--set 'model.block={name: __NAME__}'",
    "metric": "gpt eval RUN --metrics val,__NAME__     (or add it to eval.metrics in the config)",
    "optimizer": "--set 'optim.name=__NAME__'",
    "schedule": "--set 'train.schedule={name: __NAME__}'",
    "task": "source: {name: synthetic, task: __NAME__, n_docs: 2000} in a data config",
    "filter": "filters: [{name: __NAME__}] in a data config",
    "source": "sources: [{name: __NAME__, path: ...}] in a data config",
}


def new_plugin(kind: str, name: str, plugin_dir="plugins") -> Path:
    if kind not in T:
        raise ValueError(f"no template for kind {kind!r}; choose from {sorted(T)}")
    ident = name.replace("-", "_")
    if not ident.isidentifier():
        raise ValueError(f"name {name!r} must be a valid identifier-ish string")
    cls = "".join(p.capitalize() for p in ident.split("_"))
    path = Path(plugin_dir) / f"{ident}.py"
    if path.exists():
        raise FileExistsError(f"{path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = HEADER.format(doc=f"{kind} plugin '{ident}' (scaffolded by `gpt new {kind} {ident}`).") + T[kind]
    body = body.replace("__NAME__", ident).replace("__CLS__", cls)
    path.write_text(body)
    print(f"wrote {path}\n\nuse it:\n  gpt check -c CONFIG {USE[kind].replace('__NAME__', ident)}" if kind not in ("metric", "task", "filter", "source", "optimizer", "schedule")
          else f"wrote {path}\n\nuse it:\n  {USE[kind].replace('__NAME__', ident)}")
    return path
