"""Generate writing samples for the appendix: same prompts, every ratio/variant, best checkpoint each.
Loads the shared component plugins ONCE (they're identical across runs), then builds each model from its
own config.yaml + ckpt_best.pt. Uses the visible GPUs only (CUDA_VISIBLE_DEVICES=1,2,0 -> physical 0,1,2;
GPU 3 is left alone). Writes samples.json next to this file."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
from gpt_sandbox.config import load_config
from gpt_sandbox.model.gpt import GPT
from gpt_sandbox.data.tokenizers import tokenizer_from_dict
from gpt_sandbox.utils import amp_factory, get_device, read_json
from gpt_sandbox.plugins import load_plugins
from gpt_sandbox.evals.generate import generate

RUNS = Path("/data/susanket/text/gpt/runs")
SEED = 7
TEMP = 0.8
MAX_NEW = 80

MODELS = [
    ("10A · MLP:Attn 4:1",  "core",  "ratio_jumbo/r41_10A38M_r2-s1"),
    ("12A · MLP:Attn 3:1",  "core",  "ratio_jumbo/r31_12A36M_r2-s0"),
    ("16A · MLP:Attn 2:1",  "core",  "ratio_jumbo/r21_16A32M_r1-s1"),
    ("24A · MLP:Attn 1:1",  "core",  "ratio_jumbo/r11_24A24M_r2-s1"),
    ("32A · MLP:Attn 1:2",  "core",  "ratio_jumbo/r12_32A16M_r2-s1"),
    ("36A · MLP:Attn 1:3",  "core",  "ratio_jumbo/r13_36A12M_r0-s0"),
    ("38A · MLP:Attn 1:4",  "core",  "ratio_jumbo/r14_38A10M_alt-s0"),
    ("32A · param-matched (80M, max attn)", "variant", "ratio_pm_jumbo/pmr12_32A16M_alt-s1"),
    ("12A · recall champion (1:1, synth_recall)", "variant", "ratio_synth_recall/r11_12A12M_alt-s1"),
]

PROMPTS = [
    ("story",   "Once upon a time, a small robot learned to bake bread."),
    ("explain", "Explain why the sky is blue."),
    ("haiku",   "Write a haiku about a lighthouse."),
    ("memory",  "The secret code word is BANANA. My dog's name is Pickles. Tell me a short fact about my dog."),
]


def build(run_dir: Path):
    cfg = load_config(run_dir / "config.yaml")
    info = read_json(run_dir / "run.json")
    state = torch.load(run_dir / "ckpt_best.pt", map_location="cpu", weights_only=False)
    dev = get_device("auto")
    model = GPT(cfg.model)
    model.load_state_dict(state["model"])
    model.to(dev).eval()
    tok = tokenizer_from_dict(read_json(run_dir / "tokenizer.json"))
    amp, _ = amp_factory(dev, cfg.train.precision)
    return model, tok, dev, amp, state.get("step", -1), info


def main():
    # register the shared components exactly once (identical files across all these runs)
    first_plugins = RUNS / MODELS[0][2] / "plugins"
    load_plugins([first_plugins])

    out = {"seed": SEED, "temp": TEMP, "max_new": MAX_NEW, "models": []}
    for label, family, rdir in MODELS:
        run_dir = RUNS / rdir
        model, tok, dev, amp, step, info = build(run_dir)
        entry = {"label": label, "family": family, "run_dir": rdir, "step": step,
                 "params_M": round(info.get("params", 0) / 1e6, 1),
                 "nA": None, "samples": []}
        for pname, prompt in PROMPTS:
            ids = tok.encode(prompt)
            ids = [tok.eos_id] + ids  # document-start (matches training)
            x = torch.tensor([ids], device=dev)
            full = generate(model, x, MAX_NEW, TEMP, amp=amp)
            text = tok.decode(full[0, len(ids):].tolist())
            entry["samples"].append({"prompt": pname, "text": text.strip()})
        out["models"].append(entry)
        print(f"  done {label:42s} ({len(entry['samples'])} samples)", flush=True)
        del model
        torch.cuda.empty_cache()
    (Path(__file__).parent / "samples.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {Path(__file__).parent/'samples.json'}")


if __name__ == "__main__":
    main()
