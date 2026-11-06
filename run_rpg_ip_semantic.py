"""RPG's own regional pipeline, with the adapter gated by box AND semantics.

The paper's ladder tests attention gating on the Conceptrol pipeline, which is
driven by one global prompt.  RPG's regional text attention -- the thing that
makes RPG's compositions look like photographs -- was never combined with it.
This runs the cascade the fusion plan describes, inside RPG:

    stage I   RPG            -> the planner's rectangle, M_spatial
    stage II  Conceptrol     -> the region's own text attention, M_semantic
    stage III regional IP    -> the adapter writes only where both agree

--semantic-gate 0 reproduces the existing box-only rows exactly.
"""
import argparse, json, os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--out-root", default="outputs/hh_frame_medium")
ap.add_argument("--variant", default="rpg_ip_sem")
ap.add_argument("--ip-scale", type=float, default=0.6)
ap.add_argument("--semantic-gate", type=float, default=1.0,
                help="0 = the planner's rectangle alone (the existing rows); "
                     "1 = rectangle intersected with the semantic map")
ap.add_argument("--semantic-mode", choices=["soft", "otsu"], default="soft")
ap.add_argument("--base-ratio", type=float, default=0.3)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

OUT = ROOT / args.out_root / args.variant / "images"
OUT.mkdir(parents=True, exist_ok=True)

from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

print("Loading RPG (RegionalDiffusionXL)...", flush=True)
pipe = RegionalDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

samples = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["samples"]
if args.limit:
    samples = samples[:args.limit]
# one adapter: RegionalDiffusionXLPipeline routes regional_ip_adapter_images to
# the regions itself ("SingleAdapter Mode"), and rejects a list of adapters
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                     weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
pipe.set_ip_adapter_scale(args.ip_scale)

# full-strength injection, the same gate settings the paper's ablation uses;
# without these the router resolves the per-region ip_scale to 0 and the
# adapter branch is never entered
pipe._ip_router_enabled = False
pipe._ip_disable_up = False
pipe._ip_layer_scales = lambda _n: 1.0
pipe._ip_timestep_mode = "cosine"
pipe._ip_timestep_start = 0.0
pipe._ip_timestep_end = 0.0
pipe._ip_norm_cap_ratio = 0.0
pipe._ip_entropy_gate = 0.0
pipe._ip_use_value_only = False

pipe._ip_semantic_gate = args.semantic_gate
pipe._ip_semantic_mode = args.semantic_mode
print(f"semantic gate={args.semantic_gate} mode={args.semantic_mode} "
      f"ip_scale={args.ip_scale} -> {args.variant} | {len(samples)} samples",
      flush=True)

done = fail = 0
t0 = time.time()
for k, s in enumerate(samples):
    sid = s["sample_id"]
    dst = OUT / f"{sid}.png"
    if dst.exists():
        done += 1
        continue
    try:
        refs = [Image.open(ROOT / p).convert("RGB")
                for p in s["reference_image_paths"]]
        img = pipe(
            prompt=s["regional_prompts"],
            base_prompt=s["full_prompt"],
            split_ratio=s["split_ratio"],
            base_ratio=args.base_ratio,
            regional_ip_adapter_images=refs,
            height=args.size, width=args.size,
            num_inference_steps=args.steps, guidance_scale=args.cfg,
            seed=args.seed,
            negative_prompt="low quality, blurry, distorted",
        ).images[0]
        img.save(dst)
        done += 1
        if (k + 1) % 5 == 0 or k == 0:
            print(f"[{k+1}/{len(samples)}] {sid}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{k+1}/{len(samples)}] FAIL {sid}: {e}", flush=True)

n = len(list(OUT.glob("*.png")))
print(f"PNG count: {n}/{len(samples)}  ({done} ok, {fail} failed)  ->  {OUT}",
      flush=True)
