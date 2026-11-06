"""
Why does a full-strength gated adapter not write identity?

Section 4.4 establishes the effect and stops there.  This probe measures the
quantity that decides it.  Every cross-attention layer returns

    text_out + ip_out                        (cross_attention.py, main_forward)

so the adapter's influence at that layer is the ratio r = ||ip_out|| /
||text_out||.  If r is large the injection is loud and the failure is about
*what* it carries; if r is small the injection is simply drowned before it can
carry anything.  The ratio is logged per layer and per denoising step for two
configurations that differ only in where the constraint is applied:

    global    one reference, no regional decomposition   (lifts DINO-ID +0.538)
    regional  per-region references, attention-gated     (lifts DINO-ID +0.014)

Both run the same adapter weights at the same lambda, so any difference in r
is a property of the gating, not of the adapter.

Instrumentation is read-only: apply_ip_adapter_attention is wrapped to record
which module is active, and apply_ip_norm_cap - which already receives both
text_out and ip_out immediately before the sum - is wrapped to record their
norms.  Neither wrapper changes a value.

Usage:
    python probe_injection_strength.py --limit 4
"""

import argparse
import json
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
ap.add_argument("--out", default="outputs/injection_probe.json")
args = ap.parse_args()

import cross_attention as CA

# ── read-only instrumentation ────────────────────────────────────────────────
STATE = {"module": None, "step": 0, "config": None}
RECORDS = []
_orig_ip_attn = CA.apply_ip_adapter_attention
_orig_norm_cap = CA.apply_ip_norm_cap


def ip_attn_probe(module, *a, **kw):
    STATE["module"] = module
    return _orig_ip_attn(module, *a, **kw)


def norm_cap_probe(text_out, ip_out, cap_ratio, cap_eps=1e-6):
    m = STATE["module"]
    name = getattr(m, "_probe_name", "?") if m is not None else "?"
    with torch.no_grad():
        t = float(text_out.float().norm())
        i = float(ip_out.float().norm())
    RECORDS.append({"config": STATE["config"], "step": STATE["step"],
                    "layer": name, "text": t, "ip": i,
                    "ratio": i / t if t > 0 else float("nan")})
    return _orig_norm_cap(text_out, ip_out, cap_ratio, cap_eps)


CA.apply_ip_adapter_attention = ip_attn_probe
CA.apply_ip_norm_cap = norm_cap_probe

# ── pipeline ─────────────────────────────────────────────────────────────────
from experiments.config import ExperimentConfig, ABLATION_VARIANTS
from experiments.generators import create_generator

config = ExperimentConfig()
config.num_inference_steps = args.steps
config.height = args.size
config.width = args.size
config.guidance_scale = args.cfg

variant = next(v for v in ABLATION_VARIANTS if v.name == "ours")
variant.ip_scale = args.ip_scale
gen = create_generator("ours", config, variant)
gen.setup()
p = gen.pipe

# same gates as the full-strength ablation of Section 4.4
p._ip_router_enabled = False
p._ip_disable_up = False
p._ip_layer_scales = lambda _n: 1.0
p._ip_timestep_mode = "cosine"
p._ip_timestep_start = 0.0
p._ip_timestep_end = 0.0
p._ip_norm_cap_ratio = 0.0
p._ip_entropy_gate = 0.0
p._ip_use_value_only = False

# name every attention module so records are attributable to a block
for name, mod in p.unet.named_modules():
    if hasattr(mod, "processor") and hasattr(mod.processor, "to_k_ip"):
        mod._probe_name = name

# count denoising steps without touching the sampler's behaviour
_orig_step = p.scheduler.step


def counting_step(*a, **kw):
    STATE["step"] += 1
    return _orig_step(*a, **kw)


p.scheduler.step = counting_step

samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]
samples = samples[:args.limit]

for s in samples:
    refs = [Image.open(x).convert("RGB") for x in s["reference_image_paths"]]

    # regional: one reference per region, gated
    STATE["config"], STATE["step"] = "regional_gated", 0
    gen.generate(prompt=s["full_prompt"], reference_images=refs,
                 split_ratio=s["split_ratio"],
                 regional_prompts=s["regional_prompts"], seed=args.seed,
                 num_inference_steps=args.steps, height=args.size,
                 width=args.size, guidance_scale=args.cfg)

    # global: the same adapter, same lambda, one reference, no decomposition
    STATE["config"], STATE["step"] = "global", 0
    n = len(s["split_ratio"].split(","))
    gen.generate(prompt=s["full_prompt"], reference_images=[refs[0]] * n,
                 split_ratio=s["split_ratio"],
                 regional_prompts=" BREAK ".join(
                     [s["regional_prompts"].split("BREAK")[0].strip()] * n),
                 seed=args.seed, num_inference_steps=args.steps,
                 height=args.size, width=args.size, guidance_scale=args.cfg)
    print(f"  {s['sample_id']}: {len(RECORDS)} records so far", flush=True)


# ── aggregate ────────────────────────────────────────────────────────────────
def block_of(name):
    for k in ("down_blocks", "mid_block", "up_blocks"):
        if name.startswith(k):
            return k
    return "other"


agg = defaultdict(list)
for r in RECORDS:
    agg[(r["config"], block_of(r["layer"]))].append(r["ratio"])
    agg[(r["config"], "ALL")].append(r["ratio"])

by_step = defaultdict(list)
for r in RECORDS:
    by_step[(r["config"], r["step"] * 5 // max(args.steps, 1))].append(r["ratio"])

mean = lambda v: sum(v) / len(v) if v else float("nan")
print(f"\n||ip_out|| / ||text_out||  per attention layer  "
      f"({len(RECORDS)} measurements)")
print(f"{'block':<14}{'regional (gated)':>20}{'global':>14}{'ratio':>10}")
for blk in ("down_blocks", "mid_block", "up_blocks", "ALL"):
    a = mean(agg[("regional_gated", blk)])
    b = mean(agg[("global", blk)])
    print(f"{blk:<14}{a:>20.4f}{b:>14.4f}{(a/b if b else float('nan')):>10.2f}")

print(f"\nby denoising phase (0 = first fifth of the trajectory)")
print(f"{'phase':<14}{'regional (gated)':>20}{'global':>14}")
for q in range(5):
    a = mean(by_step[("regional_gated", q)])
    b = mean(by_step[("global", q)])
    print(f"{q:<14}{a:>20.4f}{b:>14.4f}")

out = {"n_records": len(RECORDS),
       "by_block": {f"{c}/{b}": mean(v) for (c, b), v in agg.items()},
       "by_phase": {f"{c}/{q}": mean(v) for (c, q), v in by_step.items()}}
Path(args.out).write_text(json.dumps(out, indent=1))
print(f"\nWritten to {args.out}")
