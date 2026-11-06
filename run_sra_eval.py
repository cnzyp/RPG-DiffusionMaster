"""
SRA: attention-level gating with a semantic-AND-box mask.

The paper's injection-site ladder tests attention gating with a rectangular
mask and finds it recovers 1.7% of the identity the same adapter delivers
from the sampling loop, an amount a paired test cannot separate from zero.
The obvious objection is that the mask was too crude: a box includes
background pixels inside it and cannot follow a subject's outline.  This
script tests the strongest form of that objection.

The mask becomes the intersection of the two spatial signals available
without training:

    M_final(i) = M_box(i)  *  M_sem(i)

M_box comes from the planner's layout, M_sem from Conceptrol's per-concept
text cross-attention at the layer its authors identify.  The box keeps the
concept from spreading, which Conceptrol alone does not do (its Margin on
Mixed-Eval-v1 is +0.004); the semantic map keeps the reference's texture off
the background inside the box, which a rectangle alone does not do.

Implementation is deliberately non-invasive.  With `global_masking=True` the
upstream processor computes the concept mask once at its chosen layer, stores
it in a module-level list, and reuses a resized copy at every other layer.
Replacing that list with one that intersects on write applies the box
everywhere the mask is used, without touching the vendored attention code.

Everything else -- backbone, adapter weights, resolution, scheduler, steps,
CFG, seed, and the corrected concept-span locator -- is identical to
run_conceptrol_official_eval.py, so the two rows differ only in the mask.

Usage:
    python run_sra_eval.py --size 1024 --scale 0.8 --variant sra_1024
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
CONCEPTROL_DIR = (ROOT / "third_party/Conceptrol").resolve()
if not CONCEPTROL_DIR.exists():
    sys.exit(f"Conceptrol checkout not found at {CONCEPTROL_DIR}")
sys.path.insert(0, str(CONCEPTROL_DIR / "gradio_src"))

import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="datasets/mixed_eval_v1/mixed_eval_v1.json")
ap.add_argument("--out-root", default="outputs/mixed_eval_v1")
ap.add_argument("--variant", default="sra_1024")
ap.add_argument("--scale", type=float, default=0.8)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--box-feather", type=float, default=0.0,
                help="fraction of the canvas width over which the layout box "
                     "ramps instead of stepping.  0 is the hard rectangle; "
                     "0.2 gives the kind of soft spatial bias RPG and "
                     "MS-Diffusion use, which never draws a boundary")
ap.add_argument("--adapter", choices=["plus", "base"], default="plus",
                help="'plus' is IP-Adapter Plus: 16 image tokens from the "
                     "ViT patch grid, which carry the reference's pose, "
                     "viewpoint and framing along with its identity, so the "
                     "subject is reproduced in the pose it was photographed "
                     "in.  'base' is the 4-token projection of the pooled "
                     "CLIP embedding, which keeps far less spatial detail and "
                     "should let the prompt choose the pose.")
ap.add_argument("--mask-refs", action="store_true",
                help="segment each reference and composite it onto one shared "
                     "ground before encoding.  A box mask confines each "
                     "reference's tokens to its half of the canvas, so each "
                     "half inherits that reference's own background and "
                     "grading, and the two disagree at the boundary.  Giving "
                     "both references the same ground removes what they "
                     "disagree about.")
ap.add_argument("--shared-bg", type=float, default=0.0,
                help="with --mask-refs, blur radius of the shared ground, "
                     "taken from the first reference.  0 uses flat grey.")
ap.add_argument("--ref-feather", type=float, default=0.0,
                help="with --mask-refs, blur the segmentation alpha so the "
                     "subject fades into the ground instead of being pasted "
                     "onto it with a hard silhouette edge.")
ap.add_argument("--init-from", default="",
                help="directory of stage-1 images, one <sample_id>.png each.  "
                     "Start from that image noised back to --strength of the "
                     "trajectory instead of from noise.  Paired with the "
                     "single-canvas attention route this is the only "
                     "combination in which nothing re-composes the canvas: "
                     "stage 1 fixes the scene, its lighting and its framing, "
                     "and the adapter only writes identity into it.")
ap.add_argument("--strength", type=float, default=0.6,
                help="fraction of the trajectory to re-run with the adapter.")
ap.add_argument("--skip-blocks", default="none",
                choices=["none", "style", "layout", "both"],
                help="withhold the adapter from the SDXL cross-attention "
                     "blocks that InstantStyle identifies as carrying style "
                     "(up_blocks.0.attentions.1) and spatial layout "
                     "(down_blocks.2.attentions.1).  The mask says WHERE on "
                     "the canvas the reference may write; this says WHERE IN "
                     "THE NETWORK.  Withholding the style block should keep "
                     "the subject's identity while leaving the reference's "
                     "lighting, grading and depth of field behind.")
ap.add_argument("--mask-mode", choices=["soft", "otsu"], default="soft",
                help="'soft' multiplies the box by Conceptrol's normalised "
                     "attention; 'otsu' binarises the attention first, as the "
                     "SRA proposal specifies")
args = ap.parse_args()

OUT = ROOT / args.out_root / args.variant / "images"
OUT.mkdir(parents=True, exist_ok=True)

_CKPT_NAME = ("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
              if args.adapter == "plus"
              else "sdxl_models/ip-adapter_sdxl_vit-h.safetensors")
from huggingface_hub import snapshot_download
IP_CKPT = Path(snapshot_download(
    repo_id="h94/IP-Adapter",
    allow_patterns=[_CKPT_NAME, "models/image_encoder/*"])) / _CKPT_NAME

from diffusers.schedulers import DPMSolverMultistepScheduler
from ip_adapter import ConceptrolIPAdapterPlusXL, ConceptrolIPAdapterXL
from ip_adapter.attention_processor import ConceptrolAttnProcessor
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
import ip_adapter.attention_processor as ap_mod

print("Loading SDXL...", flush=True)
pipe = StableDiffusionXLCustomPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    add_watermarker=False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)

print("Attaching Conceptrol...", flush=True)
_CLS, _NTOK = ((ConceptrolIPAdapterPlusXL, 16) if args.adapter == "plus"
               else (ConceptrolIPAdapterXL, 4))
adapter = _CLS(pipe, "", str(IP_CKPT), "cuda", num_tokens=_NTOK,
               global_masking=True)
print(f"adapter={args.adapter} ({_NTOK} image tokens) "
      f"ckpt={IP_CKPT.name}", flush=True)


# ── concept-span locator, identical to the Conceptrol row ────────────────────
def _find_subseq(hay, needle, start=0):
    if not needle:
        return -1
    for i in range(start, len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return -1


def locate_spans(prompt, subjects, offset=1):
    toks = adapter.tokenizer.tokenize(prompt)
    spans, cursor = [], 0
    for sub in subjects:
        st = adapter.tokenizer.tokenize(sub)
        pos = _find_subseq(toks, st, cursor)
        if pos < 0:
            pos = _find_subseq(toks, st, 0)
        if pos < 0:
            pos = _find_subseq(toks, st[:1], cursor)
            if pos < 0:
                return None
            st = st[:1]
        spans.append((pos + offset, pos + len(st) + offset))
        cursor = pos + len(st)
    return spans


def patched_load_textual_concept(prompt, subjects):
    spans = locate_spans(prompt, subjects)
    if spans is None:
        raise ValueError(f"could not locate subjects {subjects!r}")
    for proc in adapter.pipe.unet.attn_processors.values():
        if isinstance(proc, ConceptrolAttnProcessor):
            proc.textual_concept_idxs = spans
    return spans


adapter.load_textual_concept = patched_load_textual_concept


def subjects_for(sample):
    subs = [p.strip() for p in sample["regional_prompts"].split("BREAK")]
    return [s.split(",")[0].strip() for s in subs]


# ── the one mechanism this script adds ───────────────────────────────────────



def otsu(x):
    """Otsu threshold on a flat tensor in [0,1], as the SRA proposal specifies."""
    hist = torch.histc(x.float(), bins=64, min=0.0, max=1.0)
    total = hist.sum()
    if total <= 0:
        return 0.5
    p = hist / total
    idx = torch.arange(64, device=x.device, dtype=torch.float32) / 63.0
    w0 = torch.cumsum(p, 0)
    m0 = torch.cumsum(p * idx, 0)
    mT = m0[-1]
    w1 = 1.0 - w0
    denom = w0 * w1
    var = torch.where(denom > 0, (mT * w0 - m0) ** 2 / denom.clamp(min=1e-8),
                      torch.zeros_like(denom))
    return float(idx[int(torch.argmax(var))])


def box_at(i, side, device, dtype):
    """Layout box for concept i, built directly at the layer's resolution.

    Constructing it per size avoids interpolating a half-precision mask, which
    is what SDXL's attention layers hand us and what cuDNN declines to resize.
    """
    key = (i, side)
    if key not in _BOX_CACHE:
        parts = [float(x) for x in _SPLIT.split(",")]
        tot = sum(parts)
        cum = [sum(parts[:k]) / tot for k in range(len(parts) + 1)]
        c0 = cum[i] * side
        c1 = cum[i + 1] * side
        x = torch.arange(side, dtype=torch.float32) + 0.5
        f = args.box_feather * side
        if f <= 0:
            prof = ((x >= c0) & (x < c1)).float()
        else:
            # smooth ramps at both edges; the box biases rather than partitions
            rise = torch.clamp((x - (c0 - f)) / (2 * f), 0.0, 1.0)
            fall = torch.clamp(((c1 + f) - x) / (2 * f), 0.0, 1.0)
            prof = torch.minimum(rise, fall)
            prof = prof * prof * (3 - 2 * prof)      # smoothstep
        m = prof.view(1, 1, 1, side).expand(1, 1, side, side).clone()
        _BOX_CACHE[key] = m
    return _BOX_CACHE[key].to(device=device, dtype=dtype)


def intersect_with_box(i, mask):
    """Multiply the concept mask by its layout box at the layer's own size.

    Installed as ip_adapter.attention_processor.region_mask_hook, so it runs
    wherever the mask scales the adapter output rather than only where
    upstream happens to cache it.
    """
    global HOOK_CALLS
    if mask is None or _SPLIT is None or i >= _N_CONCEPTS:
        return mask
    n = mask.shape[1]
    side = int(round(n ** 0.5))
    if side * side != n:
        return mask
    v = mask.view(mask.shape[0], 1, side, side)
    if args.mask_mode == "otsu":
        thr = otsu(v.reshape(-1))
        v = (v > thr).to(v.dtype)
    v = v * box_at(i, side, v.device, v.dtype)
    HOOK_CALLS += 1
    return v.reshape(mask.shape[0], -1, 1)


_BOX_CACHE = {}
_SPLIT = None
_N_CONCEPTS = 0
HOOK_CALLS = 0


def reset_globals(n_concepts, split_ratio):
    global _SPLIT, _N_CONCEPTS, _BOX_CACHE
    _SPLIT, _N_CONCEPTS, _BOX_CACHE = split_ratio, n_concepts, {}
    ap_mod.region_mask_hook = intersect_with_box
    ap_mod.global_concept_mask = []
    ap_mod.attn_mask_logs.clear()
    ap_mod.text_attn_map_logs.clear()
    ap_mod.image_attn_map_logs.clear()


# ── per-block injection policy ───────────────────────────────────────────────
STYLE_BLOCK  = "up_blocks.0.attentions.1"
LAYOUT_BLOCK = "down_blocks.2.attentions.1"
_skip = {"none": [], "style": [STYLE_BLOCK], "layout": [LAYOUT_BLOCK],
         "both": [STYLE_BLOCK, LAYOUT_BLOCK]}[args.skip_blocks]

if _skip:
    _orig_set_scale = adapter.set_scale

    def _set_scale_with_policy(scale):
        _orig_set_scale(scale)
        n = 0
        for name, proc in adapter.pipe.unet.attn_processors.items():
            if hasattr(proc, "scale") and any(b in name for b in _skip):
                proc.scale = 0.0
                n += 1
        return n

    adapter.set_scale = _set_scale_with_policy
    _n = _set_scale_with_policy(args.scale)
    _tot = sum(1 for p in adapter.pipe.unet.attn_processors.values()
               if hasattr(p, "scale"))
    print(f"skip-blocks={args.skip_blocks}: adapter withheld from "
          f"{_n}/{_tot} cross-attention layers", flush=True)
    if _n == 0:
        sys.exit("skip-blocks matched no layer -- refusing to run a no-op")


# ── reference preprocessing ──────────────────────────────────────────────────
_REMBG = None


def load_refs(paths):
    ims = [Image.open(ROOT / p).convert("RGB") for p in paths]
    if not args.mask_refs:
        return ims
    global _REMBG
    if _REMBG is None:
        from rembg import new_session
        _REMBG = new_session("u2net")
    from rembg import remove
    import numpy as _np
    from PIL import ImageFilter as _IF
    backdrop = (ims[0].filter(_IF.GaussianBlur(args.shared_bg))
                if args.shared_bg > 0 else None)
    out = []
    for im in ims:
        m = _np.asarray(remove(im, session=_REMBG, only_mask=True),
                        _np.float32) / 255.0
        if m.mean() < 0.04:            # segmentation found nothing
            out.append(im); continue
        if args.ref_feather > 0:
            m = _np.asarray(Image.fromarray((m * 255).astype(_np.uint8))
                            .filter(_IF.GaussianBlur(args.ref_feather)),
                            _np.float32) / 255.0
        a = _np.asarray(im, _np.float32)
        bg = (_np.asarray(backdrop.resize(im.size), _np.float32)
              if backdrop is not None else _np.full_like(a, 127.0))
        m3 = m[..., None]
        out.append(Image.fromarray((a * m3 + bg * (1 - m3)).astype(_np.uint8)))
    return out


# ── stage-1 warm start ───────────────────────────────────────────────────────
# The pipeline exposes denoising_end but no denoising_start, so the trajectory
# is truncated by wrapping set_timesteps: the pipeline sets the full schedule,
# and we drop the steps stage 1 already covered.  prepare_latents multiplies
# whatever it is handed by init_noise_sigma unconditionally, so the latent is
# pre-divided to survive the round trip.
START_I = 0
if args.init_from:
    START_I = max(0, min(int(round((1.0 - args.strength) * args.steps)),
                         args.steps - 1))
    _orig_sst = pipe.scheduler.set_timesteps

    def _set_timesteps_trunc(*a, **kw):
        _orig_sst(*a, **kw)
        sch = pipe.scheduler
        if START_I > 0 and len(sch.timesteps) > START_I:
            sch.timesteps = sch.timesteps[START_I:]
            if getattr(sch, "sigmas", None) is not None:
                sch.sigmas = sch.sigmas[START_I:]
            sch.num_inference_steps = len(sch.timesteps)
            sch.model_outputs = [None] * sch.config.solver_order
            sch.lower_order_nums = 0
            sch._step_index = None
            sch._begin_index = None

    pipe.scheduler.set_timesteps = _set_timesteps_trunc
    # SDXL's fp16 VAE overflows on encode; without this the stage-1 latent is
    # all NaN and the decoded image is black.  Only done on this path so the
    # from-noise rows keep the weights they were measured with.
    pipe.upcast_vae()
    print(f"init-from: strength={args.strength} -> skipping the first "
          f"{START_I}/{args.steps} steps (VAE upcast)", flush=True)


def stage1_latents(sid, gen):
    """Stage-1 image, noised to the timestep the truncated schedule starts at."""
    src = Path(args.init_from) / f"{sid}.png"
    if not src.exists():
        raise FileNotFoundError(f"stage-1 image missing: {src}")
    im = Image.open(src).convert("RGB").resize((args.size, args.size))
    import numpy as _np
    px = (torch.from_numpy(_np.asarray(im, _np.float32) / 127.5 - 1.0)
          .permute(2, 0, 1)[None].to("cuda"))
    vd = pipe.vae.encoder.conv_in.weight.dtype
    with torch.no_grad():
        lat = (pipe.vae.encode(px.to(vd)).latent_dist.sample(generator=gen)
               * pipe.vae.config.scaling_factor).to(torch.float16)
    sch = pipe.scheduler
    sch.set_timesteps(args.steps, device="cuda")     # truncated by the wrapper
    noise = torch.randn(lat.shape, generator=gen, device="cuda",
                        dtype=torch.float16)
    lat = sch.add_noise(lat, noise, sch.timesteps[:1]).to(torch.float16)
    ins = float(getattr(sch, "init_noise_sigma", 1.0)) or 1.0
    return lat / ins            # prepare_latents multiplies this straight back


# ── generate ─────────────────────────────────────────────────────────────────
samples = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["samples"]
if args.limit:
    samples = samples[:args.limit]

print(f"SRA mask={args.mask_mode} scale={args.scale} size={args.size} "
      f"-> {args.variant} | {len(samples)} samples", flush=True)

done = fail = 0
t0 = time.time()
for k, s in enumerate(samples):
    sid = s["sample_id"]
    dst = OUT / f"{sid}.png"
    if dst.exists():
        done += 1
        continue
    try:
        subs = subjects_for(s)
        refs = load_refs(s["reference_image_paths"])
        if len(subs) != len(refs):
            raise ValueError(f"{len(subs)} subjects vs {len(refs)} refs")

        reset_globals(len(refs), s["split_ratio"])
        extra = {}
        if args.init_from:
            extra["latents"] = stage1_latents(
                sid, torch.Generator(device="cuda").manual_seed(args.seed))
        images = adapter.generate(
            pil_images=refs, prompt=s["full_prompt"],
            negative_prompt="low quality, blurry, distorted",
            subjects=subs, scale=args.scale, num_samples=1, seed=args.seed,
            num_inference_steps=args.steps, height=args.size, width=args.size,
            guidance_scale=args.cfg, **extra)
        images[0].save(dst)
        done += 1
        if (k + 1) % 10 == 0 or k == 0:
            print(f"[{k+1}/{len(samples)}] {sid}  {time.time()-t0:.0f}s",
                  flush=True)
    except Exception as e:
        fail += 1
        print(f"[{k+1}] ERR {sid}: {e}", flush=True)

print(f"mask hook fired {HOOK_CALLS} times", flush=True)
print(f"PNG count: {len(list(OUT.glob('*.png')))}/{len(samples)}  "
      f"({done} ok, {fail} failed)  ->  {OUT}", flush=True)
