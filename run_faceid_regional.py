"""Regional ArcFace identity: one FaceID adapter per subject, spatially masked.

Every configuration in this study so far injects a CLIP *image* embedding, which
carries the reference photograph's pose, framing and background along with the
subject's identity.  That bundling is why identity, scene layout and pose all
compete for the same early steps: you cannot ask for the identity early without
also getting the photograph it came from.

ArcFace is trained to be invariant to pose, expression and lighting, so an
ArcFace embedding carries identity and nothing else.  Injected from step 0 it
should give strong identity without dictating how the subject stands, leaving
pose and composition to the text.

Two adapters are loaded from the same FaceID PlusV2 weights, one per subject,
and diffusers' native `ip_adapter_masks` confines each to its own region.

    python run_faceid_regional.py --dataset datasets/dreambench2/hh_photo.json \
        --out-root outputs/hh_photo --variant faceid_regional --scale 0.8
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
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="datasets/dreambench2/hh_photo.json")
ap.add_argument("--out-root", default="outputs/hh_photo")
ap.add_argument("--variant", default="faceid_regional")
ap.add_argument("--scale", type=float, default=0.8)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--cfg", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--mask-feather", type=float, default=0.0,
                help="fraction of canvas width over which each region mask "
                     "ramps instead of stepping, so a subject may cross the "
                     "boundary")
ap.add_argument("--scales", default="",
                help="per-adapter scales 'a,b' overriding --scale, for "
                     "isolating whether each adapter reaches the canvas")
ap.add_argument("--shortcut-scale", type=float, default=1.0,
                help="weight of the ArcFace vector added to the projection "
                     "output (FaceID PlusV2's shortcut).  0 disables it and "
                     "leaves only the CLIP branch.")
ap.add_argument("--face-masks", action="store_true",
                help="with --init-from, mask to the faces actually present in "
                     "the stage-1 image instead of to half the canvas.  A "
                     "half-canvas mask spreads the identity signal over grass "
                     "and walls; the face is a few percent of the frame, so "
                     "that is where it has to go.")
ap.add_argument("--face-pad", type=float, default=0.6,
                help="fraction of the detected face box to pad the mask by, so "
                     "hair and jaw are included.")
ap.add_argument("--init-from", default="",
                help="directory of stage-1 images.  ArcFace identity is "
                     "pose-invariant, so unlike a CLIP image embedding it "
                     "should still land when injected late: stage 1 fixes two "
                     "people, their poses and the scene, and the adapter only "
                     "changes who they are.")
ap.add_argument("--strength", type=float, default=0.45,
                help="with --init-from, img2img strength.")
ap.add_argument("--no-lora", action="store_true",
                help="skip the FaceID LoRA.  It is trained on single-subject "
                     "portraits and biases the model toward one dominant "
                     "face, which is a problem when two identities are "
                     "active at once.")
ap.add_argument("--no-mask", action="store_true",
                help="control: load both adapters but apply no masks, so both "
                     "identities compete over the whole canvas")
args = ap.parse_args()

OUT = ROOT / args.out_root / args.variant / "images"
OUT.mkdir(parents=True, exist_ok=True)

FACEID_W = ROOT / "models/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sdxl.bin"
FACEID_LORA = ROOT / "models/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sdxl_lora.safetensors"
for p in (FACEID_W, FACEID_LORA):
    if not p.exists():
        sys.exit(f"missing weights: {p}")

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

print("Loading SDXL...", flush=True)
# Build the img2img class directly rather than via from_pipe: from_pipe left the
# adapter (and the UNet) in fp32, which ran ~100x slower per image.
_CLS = (StableDiffusionXLImg2ImgPipeline if args.init_from
        else StableDiffusionXLPipeline)
pipe = _CLS.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    add_watermarker=False).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)
pipe.set_progress_bar_config(disable=True)

# One adapter per subject, same weights.  Two instances so each identity has its
# own projection and its own mask; a single adapter fed two embeddings would
# blend them before the mask could separate them.
N_ADAPTERS = 2
print("Loading FaceID PlusV2 x2...", flush=True)
pipe.load_ip_adapter(
    [str(FACEID_W.parent)] * N_ADAPTERS,
    subfolder=[""] * N_ADAPTERS,
    weight_name=[FACEID_W.name] * N_ADAPTERS,
    image_encoder_folder=None)
_SCALES = ([float(x) for x in args.scales.split(",")] if args.scales
           else [args.scale] * N_ADAPTERS)
assert len(_SCALES) == N_ADAPTERS, f"--scales needs {N_ADAPTERS} values"
pipe.set_ip_adapter_scale(_SCALES)
print(f"per-adapter scales: {_SCALES}", flush=True)
if not args.no_lora:
    # offline mode refuses to guess the file name, so name it explicitly
    pipe.load_lora_weights(str(FACEID_LORA.parent), weight_name=FACEID_LORA.name)
    pipe.fuse_lora()
print(f"FaceID LoRA: {'skipped' if args.no_lora else 'fused'}", flush=True)
# load_ip_adapter attaches the projection and the per-layer to_k_ip/to_v_ip in
# whatever dtype it chooses; feed it embeddings in that same dtype rather than
# casting the modules, which would change the txt2img rows already measured.
ADAPTER_DTYPE = next(
    pipe.unet.encoder_hid_proj.image_projection_layers[0].parameters()).dtype
print(f"adapter dtype: {ADAPTER_DTYPE}", flush=True)

print("Loading insightface + CLIP ViT-H...", flush=True)
from insightface.app import FaceAnalysis
_face = FaceAnalysis(name="buffalo_l",
                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
_face.prepare(ctx_id=0, det_size=(640, 640))
_clip = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter", subfolder="models/image_encoder",
    torch_dtype=torch.float16).to("cuda")
_clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
_mask_proc = IPAdapterMaskProcessor()


def arcface(im):
    """512-d identity vector, or None when no face is detected."""
    bgr = np.asarray(im.convert("RGB"))[:, :, ::-1]
    faces = _face.get(bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return torch.from_numpy(f.normed_embedding).float()


def clip_hidden(im):
    px = _clip_proc(images=im, return_tensors="pt")["pixel_values"].to("cuda").half()
    with torch.no_grad():
        return _clip(pixel_values=px, output_hidden_states=True).hidden_states[-2]


def embeds_for(im):
    a = arcface(im)
    if a is None:
        return None, None
    ch = clip_hidden(im).to(ADAPTER_DTYPE)
    idv = a.to("cuda", dtype=ch.dtype).unsqueeze(0)                    # [1,512]
    id_cfg = torch.cat([torch.zeros_like(idv), idv], 0).unsqueeze(1)   # [2,1,512]
    clip_cfg = torch.cat([torch.zeros_like(ch), ch], 0).unsqueeze(1)   # [2,1,257,1280]
    return id_cfg, clip_cfg


def region_masks(split, n, H, W):
    parts = [float(x) for x in split.split(",")][:n]
    tot = sum(parts)
    cum = [sum(parts[:i]) / tot for i in range(n + 1)]
    xs = np.arange(W, dtype=np.float32) + 0.5
    out = []
    for i in range(n):
        c0, c1 = cum[i] * W, cum[i + 1] * W
        f = args.mask_feather * W
        if f <= 0:
            prof = ((xs >= c0) & (xs < c1)).astype(np.float32)
        else:
            rise = np.clip((xs - (c0 - f)) / (2 * f), 0, 1)
            fall = np.clip(((c1 + f) - xs) / (2 * f), 0, 1)
            prof = np.minimum(rise, fall)
            prof = prof * prof * (3 - 2 * prof)
        out.append(Image.fromarray(
            (np.tile(prof[None, :], (H, 1)) * 255).astype(np.uint8)))
    return out


def stage1_face_masks(init_img, n):
    """Masks around the faces stage 1 actually drew, left to right."""
    bgr = np.asarray(init_img.convert("RGB"))[:, :, ::-1]
    faces = _face.get(bgr)
    if len(faces) < n:
        return None
    faces = sorted(faces, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)[:n]
    W, H = init_img.size
    out = []
    for f in faces:
        x0, y0, x1, y1 = f.bbox
        pw, ph = (x1 - x0) * args.face_pad, (y1 - y0) * args.face_pad
        m = np.zeros((H, W), np.float32)
        m[max(0, int(y0 - ph)):min(H, int(y1 + ph)),
          max(0, int(x0 - pw)):min(W, int(x1 + pw))] = 1.0
        out.append(Image.fromarray((m * 255).astype(np.uint8)))
    return out


samples = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["samples"]
if args.limit:
    samples = samples[:args.limit]
print(f"regional FaceID scale={args.scale} feather={args.mask_feather} "
      f"masks={'OFF' if args.no_mask else 'ON'} -> {args.variant} | "
      f"{len(samples)} samples", flush=True)

done = fail = noface = 0
t0 = time.time()
for k, s in enumerate(samples):
    sid = s["sample_id"]
    dst = OUT / f"{sid}.png"
    if dst.exists():
        done += 1
        continue
    try:
        refs = [Image.open(ROOT / p).convert("RGB")
                for p in s["reference_image_paths"]][:N_ADAPTERS]
        if len(refs) != N_ADAPTERS:
            raise ValueError(f"{len(refs)} references, expected {N_ADAPTERS}")
        ids, clips = [], []
        for im in refs:
            i_, c_ = embeds_for(im)
            if i_ is None:
                raise ValueError("no face detected in a reference")
            ids.append(i_); clips.append(c_)

        # FaceID PlusV2's projection reads its CLIP branch from a module buffer.
        # `shortcut` is what makes it V2: with it off the ArcFace vector never
        # reaches the output and the result is driven by the CLIP branch alone,
        # which is the pose-carrying signal we are trying to get away from.
        for j, c_ in enumerate(clips):
            proj = pipe.unet.encoder_hid_proj.image_projection_layers[j]
            proj.clip_embeds = c_
            proj.shortcut = True
            proj.shortcut_scale = args.shortcut_scale

        init_img = None
        if args.init_from:
            ip_ = Path(args.init_from) / f"{sid}.png"
            if not ip_.exists():
                raise FileNotFoundError(f"stage-1 image missing: {ip_}")
            init_img = Image.open(ip_).convert("RGB").resize(
                (args.size, args.size))

        ck = {}
        if not args.no_mask:
            ms = None
            if args.face_masks and init_img is not None:
                ms = stage1_face_masks(init_img, N_ADAPTERS)
                if ms is None:
                    raise ValueError("stage 1 shows fewer faces than subjects")
            if ms is None:
                ms = region_masks(s["split_ratio"], N_ADAPTERS,
                                  args.size, args.size)
            proc = _mask_proc.preprocess(ms, height=args.size, width=args.size)
            ck["ip_adapter_masks"] = [proc[j:j + 1] for j in range(N_ADAPTERS)]

        call = dict(
            prompt=s["full_prompt"],
            negative_prompt="low quality, blurry, distorted, deformed, extra limbs",
            ip_adapter_image_embeds=ids,
            cross_attention_kwargs=ck or None,
            num_inference_steps=args.steps, guidance_scale=args.cfg,
            generator=torch.Generator("cuda").manual_seed(args.seed))
        if args.init_from:
            call["image"] = init_img
            call["strength"] = args.strength
        else:
            call["height"] = call["width"] = args.size
        img = pipe(**call).images[0]
        img.save(dst)
        done += 1
        if (k + 1) % 5 == 0 or k == 0:
            print(f"[{k+1}/{len(samples)}] {sid}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        fail += 1
        if fail == 1 and os.environ.get("FID_TRACE"):
            import traceback; traceback.print_exc()
        if "no face" in str(e):
            noface += 1
        print(f"[{k+1}/{len(samples)}] FAIL {sid}: {e}", flush=True)

n = len(list(OUT.glob("*.png")))
print(f"PNG count: {n}/{len(samples)}  ({done} ok, {fail} failed, "
      f"{noface} of them no-face)  ->  {OUT}", flush=True)
