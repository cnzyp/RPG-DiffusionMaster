"""
MS-Diffusion (ICLR 2025) baseline on Mixed-Eval-v1.

MS-Diffusion is the closest published comparison to ours: same SDXL-base-1.0
backbone, no per-concept optimisation, and it takes reference images plus
bounding boxes at inference - exactly the inputs our planner produces.  Run
from the authors' released code (third_party/MS-Diffusion) and checkpoint.

Protocol is matched to ours wherever the method allows it: 768x768, 30
DPM-Solver++ steps on the Karras sigma schedule, CFG 7.5, seed 42, and the
same negative prompt.  The one method-specific knob, the injection scale, is
swept over {0.6 (the authors' default), 0.8 (matched to our lambda)} and the
stronger run is the one reported, which is the same rule we applied to
Conceptrol and IP-FaceID.

Layout: Mixed-Eval-v1 splits the canvas into equal vertical strips
("1,1" or "1,1,1"), so region i becomes the box [i/n, 0, (i+1)/n, 1].

Usage:
    python regen_msdiffusion.py --scale 0.6
    python regen_msdiffusion.py --scale 0.8
    python regen_msdiffusion.py --scale 0.6 --limit 4 --swap-test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
MSDIR = ROOT / "third_party" / "MS-Diffusion"
if not MSDIR.exists():
    sys.exit(f"missing {MSDIR} - git clone the MS-Diffusion repo there first")
sys.path.insert(0, str(MSDIR))

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from msdiffusion.models.projection import Resampler
from msdiffusion.models.model import MSAdapter
from msdiffusion.utils import get_phrase_idx, get_eot_idx

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=0.6,
                help="MS-Diffusion injection scale; 0.6 is the authors' default")
ap.add_argument("--dataset", default="datasets/mixed_eval_v1/mixed_eval_v1.json")
ap.add_argument("--out-root", default="outputs/mixed_eval_v1")
ap.add_argument("--variant", default="")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--size", type=int, default=768)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--inset", action="store_true",
                help="use the authors' tighter, non-adjacent box shape")
ap.add_argument("--swap-test", action="store_true",
                help="render one sample twice with different references and exit")
args = ap.parse_args()

H = W = args.size
NEG = "low quality, blurry, distorted"
device, dtype = "cuda", torch.float16

variant = args.variant or (f"msdiffusion_s{args.scale:g}".replace(".", "")
                           + ("_inset" if args.inset else ""))
OUT = ROOT / args.out_root / variant / "images"
OUT.mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download

MS_CKPT = Path(snapshot_download(
    repo_id="doge1516/MS-Diffusion",
    allow_patterns=["ms_adapter.bin", "*.json"])) / "ms_adapter.bin"
IMAGE_ENCODER = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

samples = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["samples"]
if args.limit:
    samples = samples[:args.limit]
todo = [s for s in samples if not (OUT / f"{s['sample_id']}.png").exists()]
print(f"variant={variant} scale={args.scale} | total {len(samples)} | "
      f"remaining {len(todo)}", flush=True)
if not todo and not args.swap_test:
    print("nothing to do")
    raise SystemExit(0)

print("Loading SDXL + CLIP-G + MS-Adapter ...", flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype, add_watermarker=False).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)
pipe.set_progress_bar_config(disable=True)

def load_clip_g_vision():
    """Build CLIP-G's vision tower from the safetensors shards on disk.

    The repo ships its shard index as `pytorch_model.bin.index.json`, i.e. it
    maps onto `.bin` shards, while the safetensors shards carry the older
    `pytorch_model-*.safetensors` names and no matching index.  A plain
    `from_pretrained` therefore follows the .bin map and starts a ~10 GB
    re-download of weights we already have in the other format.  MS-Diffusion
    only ever uses the vision tower and its projection, so load those keys
    directly.
    """
    from safetensors.torch import load_file
    from transformers import CLIPVisionConfig

    snap = Path(snapshot_download(
        repo_id=IMAGE_ENCODER,
        allow_patterns=["config.json", "preprocessor_config.json",
                        "pytorch_model-*.safetensors", "*.index.json"]))
    cfg = CLIPVisionConfig(
        **json.loads((snap / "config.json").read_text())["vision_config"])
    model = CLIPVisionModelWithProjection(cfg)

    sd = {}
    for shard in sorted(snap.glob("pytorch_model-*-of-*.safetensors")):
        for k, v in load_file(str(shard)).items():
            if k.startswith(("vision_model.", "visual_projection.")):
                sd[k] = v
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        sys.exit(f"CLIP-G vision weights incomplete: {len(result.missing_keys)} "
                 f"missing, e.g. {result.missing_keys[:3]}")
    print(f"  CLIP-G vision tower: {len(sd)} tensors from "
          f"{len(list(snap.glob('pytorch_model-*-of-*.safetensors')))} shards",
          flush=True)
    return model.to(device, dtype=dtype).eval()


image_processor = CLIPImageProcessor()
image_encoder = load_clip_g_vision()

NUM_TOKENS = 16
image_proj_model = Resampler(
    dim=1280, depth=4, dim_head=64, heads=20, num_queries=NUM_TOKENS,
    embedding_dim=image_encoder.config.hidden_size,
    output_dim=pipe.unet.config.cross_attention_dim, ff_mult=4,
    latent_init_mode="grounding",
    phrase_embeddings_dim=pipe.text_encoder.config.projection_dim,
).to(device, dtype=dtype)
ms_model = MSAdapter(pipe.unet, image_proj_model, ckpt_path=str(MS_CKPT),
                     device=device, num_tokens=NUM_TOKENS)
ms_model.to(device, dtype=dtype)
print("ready\n", flush=True)


def boxes_for(split_ratio):
    """Normalised [x1, y1, x2, y2] per region.

    Default is the planner's own layout: equal, adjacent, full-height strips,
    i.e. exactly the regions our sub-canvases occupy.  --inset instead shrinks
    each strip to 80% of its width and to the vertical band [0.15, 0.85],
    leaving a gap between boxes and a margin around them, which is the shape
    the MS-Diffusion authors use in their released examples.  Adjacent
    full-height boxes leave the model no background region, so the tighter
    form is run as well and the stronger result is the one reported.
    """
    n = len(split_ratio.split(","))
    if not args.inset:
        return [[i / n, 0.0, (i + 1) / n, 1.0] for i in range(n)]
    half = 0.4 / n
    return [[(i + 0.5) / n - half, 0.15, (i + 0.5) / n + half, 0.85]
            for i in range(n)]


def phrase_for(regional_prompt, full_prompt):
    """Head noun phrase of the sub-prompt, e.g. "a teenage girl".

    The authors' get_phrase_idx matches on whitespace-split words and strips
    trailing punctuation from the prompt side only, so any phrase carrying an
    internal comma can never match and silently yields the span [0, 0].
    Taking the first comma-delimited chunk keeps the phrase punctuation-free
    and verbatim in the global prompt.
    """
    chunks = [c.strip() for c in regional_prompt.split(",") if c.strip()]
    return chunks[0] if chunks else regional_prompt.strip()


def phrases_idx(tokenizer, phrases, prompt):
    res, seen = [], {}
    for p in phrases:
        k = seen.get(p, 0)
        seen[p] = k + 1
        res.append(get_phrase_idx(tokenizer, p, prompt, num=k)[0])
    return res


def render(sample, refs):
    prompt = sample["full_prompt"]
    regional = [r.strip() for r in sample["regional_prompts"].split("BREAK")]
    phrases = [phrase_for(r, prompt) for r in regional]
    boxes = [boxes_for(sample["split_ratio"])]

    imgs = ms_model.generate(
        pipe=pipe, pil_images=[refs], num_samples=1,
        num_inference_steps=args.steps, seed=args.seed,
        prompt=[prompt], negative_prompt=[NEG], scale=args.scale,
        guidance_scale=args.cfg,
        image_encoder=image_encoder, image_processor=image_processor,
        boxes=boxes, image_proj_type="resampler", image_encoder_type="clip",
        phrases=[phrases], drop_grounding_tokens=[0],
        phrase_idxes=[phrases_idx(pipe.tokenizer, phrases, prompt)],
        eot_idxes=[[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases)],
        height=H, width=W,
    )
    return imgs[0]


def load_refs(sample):
    return [Image.open(ROOT / p).convert("RGB").resize((512, 512))
            for p in sample["reference_image_paths"]]


if args.swap_test:
    # The adapter must actually read the references: two different reference
    # sets on the same prompt have to produce different pixels.
    s = samples[0]
    a = np.asarray(render(s, load_refs(s)).convert("RGB"), np.float32)
    other = next(x for x in samples[1:]
                 if x["metadata"]["category"] == s["metadata"]["category"])
    b = np.asarray(render(s, load_refs(other)).convert("RGB"), np.float32)
    d = float(np.abs(a - b).mean())
    print("=" * 60)
    print(f"swap test: mean |diff| = {d:.4f}")
    print("VERDICT:", "STILL BROKEN - references ignored" if d < 1e-6
          else "reference conditioning is ACTIVE")
    print("=" * 60)
    raise SystemExit(0)

done, failed = 0, []
t0 = time.perf_counter()
for s in todo:
    sid = s["sample_id"]
    t = time.perf_counter()
    try:
        render(s, load_refs(s)).save(OUT / f"{sid}.png")
        done += 1
        print(f"[{done:3d}/{len(todo)}] {sid}  {time.perf_counter()-t:.1f}s",
              flush=True)
    except Exception as e:
        failed.append((sid, repr(e)))
        print(f"[FAIL] {sid}: {e}", flush=True)

print(f"\nDone: {done} generated, {len(failed)} failed, "
      f"{time.perf_counter()-t0:.0f}s total")
for sid, e in failed:
    print("  ", sid, e)
