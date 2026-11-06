"""
Does the base blend explain why attention-gated regional injection fails?

Inside the regional pipeline every cross-attention layer computes the region's
output with the reference injected, then mixes it with a *base* forward that
carries no reference at all:

    out = out * (1 - base) + out_base * base            (cross_attention.py)

with base = 0.5 by default.  Half of every layer's regional output therefore
comes from a pass that never saw the reference, and the attenuation compounds
over ~70 attention layers and 30 denoising steps.  The injection itself is not
weak - measured at 3.3x the norm of the text branch - so this blend, not the
injection, is the candidate explanation for Section 4.4.

The test holds everything else fixed and sweeps only the blend.  If the
dilution is the mechanism, identity transfer should rise sharply as base goes
to zero; if Section 4.4's effect is a property of the injection site itself,
it should not move.

Metric matches sdxl_gating_probe.py: whole-image DINOv2 similarity to
reference A, and specificity against same-category distractors, so the numbers
sit on the same scale as the SDXL/FLUX comparison in the Limitations.

Usage:
    python probe_base_dilution.py --limit 12
    python probe_base_dilution.py --limit 12 --bases 0.5,0.25,0.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, ".")

import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=12)
ap.add_argument("--bases", default="0.5,0.25,0.0")
ap.add_argument("--dataset", default="datasets/mixed_eval_v1/mixed_eval_v1.json")
ap.add_argument("--ip-scale", type=float, default=0.8)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--size", type=int, default=768)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="outputs/base_dilution")
args = ap.parse_args()

BASES = [float(x) for x in args.bases.split(",")]
OUT = ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)

from experiments.config import ExperimentConfig, ABLATION_VARIANTS
from experiments.generators import create_generator

cfg = ExperimentConfig()
cfg.num_inference_steps = args.steps
cfg.height = cfg.width = args.size
cfg.guidance_scale = args.cfg
variant = next(v for v in ABLATION_VARIANTS if v.name == "ours")
variant.ip_scale = args.ip_scale
gen = create_generator("ours", cfg, variant)
gen.setup()
p = gen.pipe

# the full-strength gating configuration of Section 4.4
p._ip_router_enabled = False
p._ip_disable_up = False
p._ip_layer_scales = lambda _n: 1.0
p._ip_timestep_mode = "cosine"
p._ip_timestep_start = 0.0
p._ip_timestep_end = 0.0
p._ip_norm_cap_ratio = 0.0
p._ip_entropy_gate = 0.0
p._ip_use_value_only = False

samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]
samples = samples[:args.limit]

for b in BASES:
    d = OUT / f"base{int(round(b*100)):03d}"
    d.mkdir(parents=True, exist_ok=True)
    todo = [s for s in samples if not (d / f"{s['sample_id']}.png").exists()]
    print(f"\nbase_ratio={b}: {len(todo)} to render", flush=True)
    for s in todo:
        refs = [Image.open(x).convert("RGB") for x in s["reference_image_paths"]]
        img = gen.generate(prompt=s["full_prompt"], reference_images=refs,
                           split_ratio=s["split_ratio"],
                           regional_prompts=s["regional_prompts"],
                           seed=args.seed, num_inference_steps=args.steps,
                           height=args.size, width=args.size,
                           guidance_scale=args.cfg, base_ratio=b)
        img.save(d / f"{s['sample_id']}.png")
    print(f"  base_ratio={b} done", flush=True)

del gen
torch.cuda.empty_cache()

# ── score, same metric as sdxl_gating_probe.py ───────────────────────────────
from transformers import AutoImageProcessor, AutoModel

proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()


@torch.no_grad()
def embed(img):
    x = proc(images=img, return_tensors="pt").to("cuda")
    v = dino(**x).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(v, dim=-1)


allsamples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]
by_cat = {}
for s in allsamples:
    by_cat.setdefault(s["metadata"]["category"], []).append(s)

rows = {}
for b in BASES:
    d = OUT / f"base{int(round(b*100)):03d}"
    ids, specs = [], []
    for s in samples:
        f = d / f"{s['sample_id']}.png"
        if not f.exists():
            continue
        e = embed(Image.open(f).convert("RGB"))
        ref = embed(Image.open(s["reference_image_paths"][0]).convert("RGB"))
        sim = float(e @ ref.T)
        distr = [Image.open(o["reference_image_paths"][0]).convert("RGB")
                 for o in by_cat[s["metadata"]["category"]]
                 if o["sample_id"] != s["sample_id"]][:4]
        ids.append(sim)
        if distr:
            dm = torch.cat([embed(x) for x in distr])
            specs.append(sim - float((e @ dm.T).mean()))
    m = lambda v: sum(v) / len(v) if v else float("nan")
    rows[b] = {"dino": m(ids), "spec": m(specs), "n": len(ids)}

# text-only floor from the existing RPG-only renders, same metric
floor_ids = []
rpg = ROOT / "outputs/mixed_eval_v1/rpg_only/images"
for s in samples:
    f = rpg / f"{s['sample_id']}.png"
    if f.exists():
        floor_ids.append(float(embed(Image.open(f).convert("RGB")) @
                               embed(Image.open(s["reference_image_paths"][0])
                                     .convert("RGB")).T))
floor = sum(floor_ids) / len(floor_ids) if floor_ids else float("nan")

print(f"\ntext-only floor (RPG-only, same metric): {floor:.4f}")
print(f"{'base_ratio':<12}{'DINO-ID':>10}{'lift':>10}{'specificity':>14}{'n':>5}")
for b in BASES:
    r = rows[b]
    print(f"{b:<12}{r['dino']:>10.4f}{r['dino']-floor:>+10.4f}"
          f"{r['spec']:>14.4f}{r['n']:>5}")

(OUT / "summary.json").write_text(json.dumps(
    {"floor": floor, "rows": {str(k): v for k, v in rows.items()}}, indent=1))
print(f"\nWritten to {OUT/'summary.json'}")
