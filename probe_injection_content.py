"""
What does a full-strength gated injection actually carry?

probe_injection_strength.py rules out the simplest explanation: inside the
gated path ||ip_out|| / ||text_out|| is 2.63, i.e. the injection is louder
than the text branch it is added to, and 94% of what the same adapter
produces when every region is handed the same reference.  The signal is not
being drowned.  So the question becomes what it encodes.

Two measurements, both self-interpreting - neither needs a second pipeline to
compare against, which matters because the global IP-Adapter baseline runs on
diffusers' own attention processor rather than this code path.

1. Attention uniformity.  The IP branch attends over the reference's 257
   image tokens.  Report H(attn) / log(257).  At 1.0 the distribution is
   uniform and the output is exactly the mean of the value vectors - a
   category average with no instance information left in it, at whatever
   magnitude the projection gives.

2. Reference sensitivity.  Generate the same sample twice, changing only the
   reference to a different instance of the same category, and measure
   ||ip_out(A) - ip_out(A')|| / ||ip_out(A)|| layer by layer.  If the
   injection carried the assigned instance this is large; if it carried the
   category it is near zero.  Everything else - prompt, seed, latents, layout
   - is held fixed, so the difference isolates the reference.

Usage:
    python probe_injection_content.py --limit 4
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, ".")

import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=4)
ap.add_argument("--dataset", default="datasets/mixed_eval_v1/mixed_eval_v1.json")
ap.add_argument("--ip-scale", type=float, default=0.8)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--size", type=int, default=768)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="outputs/injection_content.json")
args = ap.parse_args()

import cross_attention as CA

STATE = {"module": None, "step": 0, "run": None}
ENTROPY = []            # uniformity records
TRACE = defaultdict(dict)   # (run) -> key -> ip_out norm-vector summary
_orig_ip_attn = CA.apply_ip_adapter_attention
_orig_norm_cap = CA.apply_ip_norm_cap


def ip_attn_probe(module, query, ip_hidden_states, *a, **kw):
    STATE["module"] = module
    out, probs = _orig_ip_attn(module, query, ip_hidden_states, *a, **kw)
    if probs is not None:
        with torch.no_grad():
            p = probs.float().clamp_min(1e-9)
            h = -(p * p.log()).sum(-1).mean()          # nats, averaged
            n = p.shape[-1]
        ENTROPY.append({"run": STATE["run"], "step": STATE["step"],
                        "layer": getattr(module, "_probe_name", "?"),
                        "uniformity": float(h / math.log(n)), "n_tokens": n})
    return out, probs


def norm_cap_probe(text_out, ip_out, cap_ratio, cap_eps=1e-6):
    m = STATE["module"]
    key = (STATE["step"], getattr(m, "_probe_name", "?"))
    with torch.no_grad():
        TRACE[STATE["run"]][key] = ip_out.detach().float().flatten().cpu()
    return _orig_norm_cap(text_out, ip_out, cap_ratio, cap_eps)


CA.apply_ip_adapter_attention = ip_attn_probe
CA.apply_ip_norm_cap = norm_cap_probe

from experiments.config import ExperimentConfig, ABLATION_VARIANTS
from experiments.generators import create_generator

config = ExperimentConfig()
config.num_inference_steps = args.steps
config.height = config.width = args.size
config.guidance_scale = args.cfg
variant = next(v for v in ABLATION_VARIANTS if v.name == "ours")
variant.ip_scale = args.ip_scale
gen = create_generator("ours", config, variant)
gen.setup()
p = gen.pipe

p._ip_router_enabled = False
p._ip_disable_up = False
p._ip_layer_scales = lambda _n: 1.0
p._ip_timestep_mode = "cosine"
p._ip_timestep_start = 0.0
p._ip_timestep_end = 0.0
p._ip_norm_cap_ratio = 0.0
p._ip_entropy_gate = 0.0
p._ip_use_value_only = False

for name, mod in p.unet.named_modules():
    if hasattr(mod, "processor") and hasattr(mod.processor, "to_k_ip"):
        mod._probe_name = name

_orig_step = p.scheduler.step


def counting_step(*a, **kw):
    STATE["step"] += 1
    return _orig_step(*a, **kw)


p.scheduler.step = counting_step

samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]
samples = samples[:args.limit]
by_cat = defaultdict(list)
for s in json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]:
    by_cat[s["metadata"]["category"]].append(s)


def run(sample, refs, tag):
    STATE["run"], STATE["step"] = tag, 0
    TRACE[tag].clear()
    gen.generate(prompt=sample["full_prompt"], reference_images=refs,
                 split_ratio=sample["split_ratio"],
                 regional_prompts=sample["regional_prompts"], seed=args.seed,
                 num_inference_steps=args.steps, height=args.size,
                 width=args.size, guidance_scale=args.cfg)


sens = defaultdict(list)
for s in samples:
    cat = s["metadata"]["category"]
    alt = next(o for o in by_cat[cat] if o["sample_id"] != s["sample_id"])
    refs_a = [Image.open(x).convert("RGB") for x in s["reference_image_paths"]]
    refs_b = list(refs_a)
    refs_b[0] = Image.open(alt["reference_image_paths"][0]).convert("RGB")

    run(s, refs_a, "A")
    run(s, refs_b, "B")

    common = set(TRACE["A"]) & set(TRACE["B"])
    for key in common:
        a, b = TRACE["A"][key], TRACE["B"][key]
        if a.shape != b.shape:
            continue
        na = float(a.norm())
        if na > 0:
            sens[key[1]].append(float((a - b).norm()) / na)
    print(f"  {s['sample_id']}: {len(common)} matched layer/step slots",
          flush=True)


def block_of(name):
    for k in ("down_blocks", "mid_block", "up_blocks"):
        if name.startswith(k):
            return k
    return "other"


mean = lambda v: sum(v) / len(v) if v else float("nan")

print(f"\n1. Attention uniformity  H(attn)/log(257)   1.0 = uniform = "
      f"category mean")
u = defaultdict(list)
for r in ENTROPY:
    u[block_of(r["layer"])].append(r["uniformity"])
    u["ALL"].append(r["uniformity"])
for blk in ("down_blocks", "mid_block", "up_blocks", "ALL"):
    print(f"   {blk:<14}{mean(u[blk]):.4f}   (n={len(u[blk])})")

print(f"\n2. Reference sensitivity  ||ip_out(A)-ip_out(A')|| / ||ip_out(A)||")
b = defaultdict(list)
for layer, vals in sens.items():
    b[block_of(layer)].extend(vals)
    b["ALL"].extend(vals)
for blk in ("down_blocks", "mid_block", "up_blocks", "ALL"):
    print(f"   {blk:<14}{mean(b[blk]):.4f}   (n={len(b[blk])})")

Path(args.out).write_text(json.dumps(
    {"uniformity": {k: mean(v) for k, v in u.items()},
     "reference_sensitivity": {k: mean(v) for k, v in b.items()}}, indent=1))
print(f"\nWritten to {args.out}")
