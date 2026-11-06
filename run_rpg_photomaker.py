"""RPG with identity carried in the text stream instead of an image branch.

Every configuration measured so far injects identity through IP-Adapter's image
cross-attention, a branch that runs alongside the text.  A CLIP image embedding
bundles identity with the reference photograph's pose, framing and background,
so the branch cannot deliver the first without imposing the rest: turn it up and
the composition collapses to the reference's bust framing, turn it down and no
identity arrives.  Inside RPG, measured at every strength from 0.6 to 3.0, no
identity arrives at any setting.

PhotoMaker fuses identity into the *text token* of the class noun.  There is no
image branch, so nothing competes with the prompt for framing or pose, and RPG's
regional text attention already routes each region's tokens to that region's
pixels -- which makes the identity regional with no mask at all.

FuseModule writes the k-th identity into the k-th trigger position, so one
concatenated regional prompt with one trigger per region carries one identity
per region.

    python run_rpg_photomaker.py --dataset datasets/dreambench2/hh_medium.json \
        --variant rpg_pm --limit 4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "third_party/PhotoMaker"))

import torch
from PIL import Image
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--out-root", default="outputs/hh_frame_medium")
ap.add_argument("--variant", default="rpg_pm")
ap.add_argument("--trigger", default="img")
ap.add_argument("--base-ratio", type=float, default=0.3)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--lora-scale", type=float, default=0.0,
                help="PhotoMaker ships a LoRA trained on portraits.  Measured "
                     "on 24 samples it does not carry identity -- identity "
                     "comes from the fusion -- it only drags the framing from "
                     "the prompt's medium shot to a bust (9.5%% -> 20.5%% face "
                     "area).  Default off.")
ap.add_argument("--pm-version", choices=["v1", "v2"], default="v1",
                help="v2 adds an ArcFace branch, which is stronger on identity "
                     "but needs a detectable face in every reference")
ap.add_argument("--hybrid", action="store_true",
                help="two injection sites in one image: the text-stream fusion "
                     "on person regions, RPG's regional IP-Adapter on the rest. "
                     "The fusion cannot carry object identity (its encoder was "
                     "trained on faces); the adapter can, in principle.")
ap.add_argument("--ip-scale", type=float, default=0.6,
                help="adapter strength on the non-person regions of --hybrid")
ap.add_argument("--humans-only", action="store_true",
                help="fuse only into person-like class nouns.  The encoder is "
                     "trained on faces; writing an identity into 'flamingo' or "
                     "'vase' damages those subjects (DINO-ID 0.609 -> 0.442 on "
                     "animal-object pairs), so leave them to the text.")
ap.add_argument("--fuse-strength", type=float, default=1.0,
                help="blend between the plain class token (0) and the fused "
                     "identity token (1), a knob on identity that leaves the "
                     "backbone untouched")
ap.add_argument("--base-ckpt", default="",
                help="path to a single-file checkpoint to use instead of SDXL-base.  RPG's README recommends a photoreal CivitAI checkpoint for photorealism; everything in this study was rendered on SDXL-base, so this makes the backbone a controlled variable rather than an assumption.")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--no-fuse", action="store_true",
                help="control: identical prompts and weights with the identity "
                     "fusion switched off, so any difference is the fusion")
args = ap.parse_args()

OUT = ROOT / args.out_root / args.variant / "images"
OUT.mkdir(parents=True, exist_ok=True)

from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from photomaker.model import PhotoMakerIDEncoder
from photomaker.model_v2 import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
from transformers import CLIPImageProcessor

print(f"Loading RPG on {args.base_ckpt or 'SDXL-base'}...", flush=True)
if args.base_ckpt:
    pipe = RegionalDiffusionXLPipeline.from_single_file(
        args.base_ckpt, torch_dtype=torch.float16,
        use_safetensors=True).to("cuda")
else:
    pipe = RegionalDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True,
        variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)

if args.pm_version == "v1":
    ckpt = hf_hub_download("TencentARC/PhotoMaker", "photomaker-v1.bin")
else:
    ckpt = hf_hub_download("TencentARC/PhotoMaker-V2", "photomaker-v2.bin")
print(f"Loading PhotoMaker {args.pm_version} from {ckpt}", flush=True)
state = torch.load(ckpt, map_location="cpu", weights_only=False)
assert sorted(state.keys()) == ["id_encoder", "lora_weights"], sorted(state.keys())

id_encoder = (PhotoMakerIDEncoder() if args.pm_version == "v1"
              else PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken())
id_encoder.load_state_dict(state["id_encoder"], strict=True)
id_encoder = id_encoder.to("cuda", dtype=torch.float16).eval()
pipe.load_lora_weights(state["lora_weights"], adapter_name="photomaker")
pipe.fuse_lora(lora_scale=args.lora_scale)
id_image_processor = CLIPImageProcessor()

_ARC = None
if args.pm_version == "v2":
    from insightface.app import FaceAnalysis
    _ARC = FaceAnalysis(name="buffalo_l",
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _ARC.prepare(ctx_id=0, det_size=(640, 640))


def arcface_embeds(images):
    """[N, 512] ArcFace vectors; raises when a reference has no detectable face."""
    import numpy as _np
    out = []
    for im in images:
        fs = _ARC.get(_np.asarray(im.convert("RGB"))[:, :, ::-1])
        if not fs:
            raise ValueError("no detectable face in a reference (v2 needs one)")
        f = max(fs, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        out.append(torch.from_numpy(f.normed_embedding))
    return torch.stack(out)

# PhotoMaker's own pipeline adds a trigger token, then strips it before encoding
# and fuses into the CLASS NOUN that preceded it.  Adding the token here would
# push its id past the text encoder's embedding table (a device-side assert), and
# it is unnecessary: the class noun is what we need to locate, and it is already
# in the prompt.  So no trigger, no vocabulary change -- just find the noun.
MAXLEN = pipe.tokenizer_2.model_max_length
print(f"region length {MAXLEN}, fusing into the class noun directly", flush=True)


def class_token_pos(region_prompt, concept):
    """Index of the class noun's last token inside this region's 77-token block."""
    ids = pipe.tokenizer_2(region_prompt, padding="max_length", max_length=MAXLEN,
                           truncation=True, return_tensors="pt").input_ids[0].tolist()
    cid = pipe.tokenizer_2(concept, add_special_tokens=False).input_ids
    for i in range(len(ids) - len(cid) + 1):
        if ids[i:i + len(cid)] == cid:
            return i + len(cid) - 1
    raise ValueError(f"class noun {concept!r} not found in {region_prompt!r}")

PERSON_NOUNS = {"man", "woman", "girl", "boy", "person", "child", "lady",
                "gentleman", "player", "basketball player", "guy", "kid",
                "male", "female", "teenager", "adult", "elderly man",
                "elderly woman", "spider-man", "space ranger"}


def is_person(concept):
    c = concept.lower().strip()
    return c in PERSON_NOUNS or any(w in c.split() for w in ("man", "woman",
                                                             "girl", "boy",
                                                             "person", "player"))


_REFS = {"images": None, "concepts": None}


def fuse_regional_identities(prompt_embeds, region_prompts):
    """Write subject i's identity into region i's trigger token."""
    refs = _REFS["images"]
    if refs is None or args.no_fuse:
        return prompt_embeds
    # with base_ratio set, RPG prepends the base prompt as chunk 0; the regions
    # are the trailing chunks, and only those carry a reference
    n_reg = len(refs)
    offset = len(region_prompts) - n_reg
    if offset < 0:
        raise ValueError(f"{len(refs)} references for {len(region_prompts)} chunks")

    # one class-noun position per region, in region order
    mask = torch.zeros(prompt_embeds.shape[:2], dtype=torch.bool,
                       device=prompt_embeds.device)
    keep = []
    for i, concept in enumerate(_REFS["concepts"]):
        if args.humans_only and not is_person(concept):
            continue
        rp = region_prompts[offset + i]
        mask[0, (offset + i) * MAXLEN + class_token_pos(rp, concept)] = True
        keep.append(i)
    if not keep:
        return prompt_embeds
    refs = [refs[i] for i in keep]
    if int(mask.sum()) != len(keep):
        raise ValueError(f"{int(mask.sum())} class tokens for {len(keep)} subjects")

    px = id_image_processor(refs, return_tensors="pt").pixel_values
    px = px.unsqueeze(0).to("cuda", dtype=torch.float16)      # [1, N, 3, H, W]
    src = prompt_embeds.to(torch.float16)
    with torch.no_grad():
        if args.pm_version == "v2":
            ids = arcface_embeds(refs).unsqueeze(0).to("cuda", dtype=torch.float16)
            out = id_encoder(px, src.clone(), mask, ids)
        else:
            out = id_encoder(px, src.clone(), mask)
    w = args.fuse_strength
    if w != 1.0:
        out = (1.0 - w) * src + w * out
    return out.to(prompt_embeds.dtype)


if args.hybrid:
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
    # full-strength injection: RPG's router otherwise resolves the per-region
    # scale to zero and the adapter branch is never entered
    pipe._ip_router_enabled = False
    pipe._ip_disable_up = False
    pipe._ip_layer_scales = lambda _n: 1.0
    pipe._ip_timestep_mode = "cosine"
    pipe._ip_timestep_start = 0.0
    pipe._ip_timestep_end = 0.0
    pipe._ip_norm_cap_ratio = 0.0
    pipe._ip_entropy_gate = 0.0
    pipe._ip_use_value_only = False
    print(f"hybrid: adapter on non-person regions at {args.ip_scale}", flush=True)

pipe._regional_id_fuser = fuse_regional_identities
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

samples = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["samples"]
if args.limit:
    samples = samples[:args.limit]
print(f"fuse={'OFF' if args.no_fuse else 'ON'} -> {args.variant} | "
      f"{len(samples)} samples", flush=True)


done = fail = 0
t0 = time.time()
for k, s in enumerate(samples):
    sid = s["sample_id"]
    dst = OUT / f"{sid}.png"
    if dst.exists():
        done += 1
        continue
    try:
        regions = [r.strip() for r in s["regional_prompts"].split("BREAK")]
        names = s["concept_names"]
        if len(regions) != len(names):
            raise ValueError(f"{len(regions)} regions vs {len(names)} concepts")
        _REFS["images"] = [Image.open(ROOT / p).convert("RGB")
                           for p in s["reference_image_paths"]]
        _REFS["concepts"] = names

        extra = {}
        if args.hybrid:
            # the adapter writes only where the fusion cannot: per-region scale
            # 0 on people, full strength on objects and animals
            pipe._ip_scale = [0.0 if is_person(c) else args.ip_scale
                              for c in names]
            extra["regional_ip_adapter_images"] = _REFS["images"]

        img = pipe(
            prompt=" BREAK ".join(regions),
            base_prompt=s["full_prompt"],
            split_ratio=s["split_ratio"],
            base_ratio=args.base_ratio,
            height=args.size, width=args.size,
            num_inference_steps=args.steps, guidance_scale=args.cfg,
            seed=args.seed,
            negative_prompt="low quality, blurry, distorted",
            **extra,
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
