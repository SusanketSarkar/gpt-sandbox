"""Smoke test for the rms_noise insert:
  1. eval mode: model with sigma>0 is bit-identical to a sigma=0 twin (noise is training-only)
  2. train mode: streams up to the configured layer are identical to the twin; from that layer on
     they diverge, with divergence std ~= sigma * rms(stream) right at the insert site
  3. noise_std is logged on collection steps
  4. the insert is attached only to the configured layer
Run: python experiments/depth_fragility/smoke.py
"""
import sys
sys.path.insert(0, "/home/susanket/gpt-sandbox/src")
sys.path.insert(0, "/home/susanket/gpt-sandbox")

import torch

from gpt_sandbox.config import load_config
from gpt_sandbox.model.gpt import GPT
from gpt_sandbox.plugins import default_plugin_paths, load_plugins

load_plugins(default_plugin_paths())
BASE = "/home/susanket/gpt-sandbox/configs/depth/depth_m.yaml"
LAYER, SIGMA = 3, 0.3


def make_model(sigma):
    ovr = {} if sigma == 0.0 else {"model.inserts": [{"site": "after_attn", "name": "rms_noise",
                                                       "layers": LAYER, "sigma": sigma}]}
    cfg = load_config(BASE, ovr)
    cfg.model.vocab_size = 256
    torch.manual_seed(0)  # identical weights for both twins
    return GPT(cfg.model)


def capture(model, x, seed):
    """Capture the stream entering each block and entering each block's MLP (post-attn, post-insert)."""
    torch.manual_seed(seed)
    model.train()
    into, premlp = {}, {}
    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_pre_hook(lambda m, inp, i=i: into.__setitem__(i, inp[0].detach().clone())))
        handles.append(blk.mlp.register_forward_pre_hook(lambda m, inp, i=i: premlp.__setitem__(i, inp[0].detach().clone())))
    with torch.no_grad():
        out = model(x, collect=True)
    for h in handles:
        h.remove()
    return into, premlp, out.ctx.stats


clean = make_model(0.0)
noisy = make_model(SIGMA)
x = torch.randint(0, 256, (2, 32))

# 4. layer selection
assert "after_attn" not in noisy.blocks[0].inserts, "insert attached to unconfigured layer"
assert len(noisy.blocks[LAYER].inserts["after_attn"]) == 1
print(f"layer selection OK (insert only on block {LAYER})")

# 1. eval mode: identity
clean.eval(); noisy.eval()
with torch.no_grad():
    d = (noisy(x).logits - clean(x).logits).abs().max().item()
print(f"eval-mode: max |dlogit| vs clean twin = {d:.3e} (expect 0)")
assert d == 0.0, "noise leaked into eval mode"

# 2. train mode: identical up to the site, diverging after, correct scale
ic, pc, _ = capture(clean, x, seed=123)
in_n, pn_n, stats = capture(noisy, x, seed=123)
for i in range(LAYER):
    assert torch.equal(ic[i], in_n[i]), f"streams differ before the insert (block {i})"
    assert torch.equal(pc[i], pn_n[i]), f"post-attn stream differs before the insert (block {i})"
print(f"train-mode: blocks 0..{LAYER-1} bit-identical to clean twin")
assert not torch.equal(pc[LAYER], pn_n[LAYER]), "no divergence at the insert site"
diff = (pn_n[LAYER] - pc[LAYER]).float()
rms = pc[LAYER].float().pow(2).mean().sqrt().item()
emp = diff.std().item()
print(f"at block {LAYER}: stream rms={rms:.4f}, expected noise std={SIGMA*rms:.4f}, empirical={emp:.4f}")
assert abs(emp - SIGMA * rms) / (SIGMA * rms) < 0.1, "noise scale wrong"
assert not torch.equal(pc[LAYER + 1], pn_n[LAYER + 1]), "divergence did not propagate"
print("train-mode: divergence starts exactly at the insert site and propagates")

# 3. logging
assert any("noise_std" in k for k in stats), f"noise_std not logged; stats={stats}"
print("logging OK:", {k: round(float(v), 5) for k, v in stats.items() if "noise_std" in k})
print("\nALL SMOKE TESTS PASSED")
