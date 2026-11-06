"""Give each face its own full-resolution pass.

Identity score across every method we have tried is almost entirely explained by
how much of the frame the face occupies (Pearson r = 0.949 over seven methods).
A wide environmental shot puts a face at roughly 0.5% of a 1024x1024 canvas,
about 70x70 pixels; a bust-framed portrait puts it at 13%, about 370x370.  The
methods that score well on identity do so by framing portraits, which is exactly
what makes them look pasted.

That constraint is about pixels on the face, not about the injection site.  So
spend pixels there directly: keep the wide composition, then crop each face,
upsample it to the full canvas, regenerate it with its own identity adapter, and
composite it back.  The face gets a portrait's pixel budget while the image
keeps a wide shot's framing.

This is a second pass over finished images, not a generator: it takes whatever
produced the composition and only changes who the faces belong to.

    python refine_faces.py --dataset datasets/dreambench2/hh_photo_spatial.json \
        --src outputs/hh_photo/rpg_only/images \
        --out outputs/hh_photo/rpg_facerefine/images --strength 0.45
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

import numpy as np
import torch
from PIL import Image, ImageFilter

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--src", required=True, help="directory of finished images")
ap.add_argument("--out", required=True)
ap.add_argument("--strength", type=float, default=0.45,
                help="img2img strength on the face crop.  Low keeps the "
                     "original face's pose and expression and only nudges "
                     "identity; high repaints the face outright.")
ap.add_argument("--scale", type=float, default=0.9)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--cfg", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--crop-pad", type=float, default=1.1,
                help="face box is expanded by this fraction on each side "
                     "before upsampling, so the crop carries neck and hair")
ap.add_argument("--paste-feather", type=int, default=48,
                help="blur radius of the composite alpha, in pixels of the "
                     "original image")
ap.add_argument("--tile", type=int, default=1024)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

FACEID_W = ROOT / "models/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sdxl.bin"
FACEID_LORA = ROOT / "models/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sdxl_lora.safetensors"
for p in (FACEID_W, FACEID_LORA):
    if not p.exists():
        sys.exit(f"missing weights: {p}")

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

print("Loading SDXL img2img...", flush=True)
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    add_watermarker=False).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)
pipe.set_progress_bar_config(disable=True)

pipe.load_ip_adapter(str(FACEID_W.parent), subfolder="",
                     weight_name=FACEID_W.name, image_encoder_folder=None)
pipe.set_ip_adapter_scale(args.scale)
pipe.load_lora_weights(str(FACEID_LORA.parent), weight_name=FACEID_LORA.name)
pipe.fuse_lora()
ADAPTER_DTYPE = next(
    pipe.unet.encoder_hid_proj.image_projection_layers[0].parameters()).dtype
print(f"adapter dtype: {ADAPTER_DTYPE}", flush=True)

from insightface.app import FaceAnalysis
_face = FaceAnalysis(name="buffalo_l",
                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
_face.prepare(ctx_id=0, det_size=(640, 640))
_clip = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter", subfolder="models/image_encoder",
    torch_dtype=torch.float16).to("cuda")
_clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


def detect(im):
    return _face.get(np.asarray(im.convert("RGB"))[:, :, ::-1])


def embeds_for(im):
    fs = detect(im)
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
    ch = _clip_proc(images=im, return_tensors="pt")["pixel_values"].to("cuda").half()
    with torch.no_grad():
        ch = _clip(pixel_values=ch, output_hidden_states=True).hidden_states[-2]
    ch = ch.to(ADAPTER_DTYPE)
    idv = torch.from_numpy(f.normed_embedding).to("cuda", dtype=ch.dtype).unsqueeze(0)
    return (torch.cat([torch.zeros_like(idv), idv], 0).unsqueeze(1),
            torch.cat([torch.zeros_like(ch), ch], 0).unsqueeze(1))


def square_box(bbox, W, H, pad):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (0.5 + pad)
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
    s = min(x1 - x0, y1 - y0)          # keep it square so the resize is clean
    return x0, y0, x0 + s, y0 + s


samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]
if args.limit:
    samples = samples[:args.limit]
SRC = Path(args.src)
print(f"refining faces: strength={args.strength} scale={args.scale} "
      f"tile={args.tile} | {len(samples)} samples", flush=True)

done = fail = 0
t0 = time.time()
for k, s in enumerate(samples):
    sid = s["sample_id"]
    src, dst = SRC / f"{sid}.png", OUT / f"{sid}.png"
    if dst.exists():
        done += 1
        continue
    try:
        if not src.exists():
            raise FileNotFoundError(src)
        img = Image.open(src).convert("RGB")
        W, H = img.size
        faces = detect(img)
        refs = [Image.open(ROOT / p).convert("RGB")
                for p in s["reference_image_paths"]]
        if len(faces) < len(refs):
            raise ValueError(f"{len(faces)} faces in the image, "
                             f"{len(refs)} references")
        # left to right, so region order matches the planner's layout
        faces = sorted(faces, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)[:len(refs)]

        out = img.copy()
        for f, rp in zip(faces, refs):
            e = embeds_for(rp)
            if e is None:
                raise ValueError("no face in a reference")
            id_cfg, clip_cfg = e
            pipe.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_cfg
            pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut = True
            pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut_scale = 1.0

            bx = square_box(f.bbox, W, H, args.crop_pad)
            crop = out.crop(bx).resize((args.tile, args.tile), Image.LANCZOS)
            new = pipe(
                prompt="a photograph of a person's face, natural light, sharp",
                negative_prompt="blurry, distorted, deformed, low quality",
                image=crop, strength=args.strength,
                ip_adapter_image_embeds=[id_cfg],
                num_inference_steps=args.steps, guidance_scale=args.cfg,
                generator=torch.Generator("cuda").manual_seed(args.seed),
            ).images[0]

            side = bx[2] - bx[0]
            new = new.resize((side, side), Image.LANCZOS)
            # feathered alpha so the repainted crop blends into the untouched
            # body and background rather than showing its own boundary
            m = Image.new("L", (side, side), 0)
            pad = int(side * 0.12)
            m.paste(255, (pad, pad, side - pad, side - pad))
            m = m.filter(ImageFilter.GaussianBlur(args.paste_feather))
            region = out.crop(bx)
            out.paste(Image.composite(new, region, m), bx)

        out.save(dst)
        done += 1
        if (k + 1) % 5 == 0 or k == 0:
            print(f"[{k+1}/{len(samples)}] {sid}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{k+1}/{len(samples)}] FAIL {sid}: {e}", flush=True)

n = len(list(OUT.glob("*.png")))
print(f"PNG count: {n}/{len(samples)}  ({done} ok, {fail} failed)  ->  {OUT}",
      flush=True)
