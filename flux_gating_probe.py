"""
Does attention-level gating write instance identity on a DiT backbone?

Section 4.4 establishes on SDXL + IP-Adapter that masking an image adapter
*inside* cross-attention changes the image substantially yet leaves per-region
identity at the level of a text-only baseline.  The obvious objection is that
this is a property of SDXL's UNet cross-attention, or of IP-Adapter, rather
than of the injection site.  This probe re-runs the same question on a
different architecture family and a different adapter:

    backbone   FLUX.1-schnell        (rectified-flow MMDiT, not a UNet)
    adapter    OminiControl subject  (LoRA-based, not IP-Adapter)
    gating     Conceptrol            (the authors' own FLUX implementation)

Three configurations per prompt, one reference image each:

    text_only     no reference at all - the floor
    ominicontrol  adapter at full strength, ungated
    conceptrol    same adapter, gated to the subject's text tokens

Scope: Conceptrol's FLUX pipeline accepts a single reference and gates on text
tokens, with no spatial regions, so this probe tests identity transfer only.
It says nothing about multi-concept isolation on FLUX, and the paper reports
it under that restriction.  Metrics are whole-image rather than per-region for
the same reason: there is no layout to crop to.

Usage:
    python flux_gating_probe.py --limit 30
    python flux_gating_probe.py --limit 30 --metrics-only
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
CDIR = ROOT / "third_party" / "Conceptrol"
if not CDIR.exists():
    sys.exit(f"missing {CDIR}")
sys.path.insert(0, str(CDIR))

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=30)
ap.add_argument("--dataset", default="datasets/mixed_eval_v1/mixed_eval_v1.json")
ap.add_argument("--out", default="outputs/flux_probe")
ap.add_argument("--steps", type=int, default=4,
                help="FLUX.1-schnell is distilled for 1-4 steps; the "
                     "Conceptrol demo uses 20, which schnell does not need")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--cfg", type=float, default=3.5)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--metrics-only", action="store_true")
ap.add_argument("--int8", action="store_true",
                help="int8 transformer; needs >16 GB, see notes below")
ap.add_argument("--offload", action="store_true",
                help="bf16 + sequential CPU offload instead of int8 quantisation")
args = ap.parse_args()

OUT = ROOT / args.out
CONFIGS = ["text_only", "ominicontrol", "conceptrol"]
for c in CONFIGS:
    (OUT / c).mkdir(parents=True, exist_ok=True)

samples = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["samples"]
samples = samples[:args.limit]


def head_phrase(regional_prompt):
    chunks = [c.strip() for c in regional_prompt.split(",") if c.strip()]
    return chunks[0] if chunks else regional_prompt.strip()


# ── generation ───────────────────────────────────────────────────────────────
if not args.metrics_only:
    from optimum.quanto import freeze, qint4, qint8, quantize
    from omini_control.conceptrol import Conceptrol
    from omini_control.flux_conceptrol_pipeline import FluxConceptrolPipeline

    print("Loading FLUX.1-schnell ...", flush=True)
    pipe = FluxConceptrolPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

    # The LoRA has to go on before quantisation: quanto swaps each Linear for
    # a QLinear whose state dict no longer carries the `.weight` keys the LoRA
    # loader resolves against.
    pipe.load_lora_weights("Yuanshi/OminiControl",
                           weight_name="omini/subject_512.safetensors",
                           adapter_name="subject")

    if args.offload:
        # bf16 throughout, streamed submodule by submodule.  Slower, but no
        # quantisation error at all.
        print("Enabling sequential CPU offload (bf16) ...", flush=True)
        pipe.enable_sequential_cpu_offload()
    else:
        # 23.8 GB of bf16 transformer does not fit in 16 GB; int8 brings it to
        # ~12 GB.  Every configuration shares the same quantised weights, so
        # the comparison between them is unaffected.
        # Everything has to be resident: Conceptrol's pipeline builds its
        # condition tensors by hand and does not honour accelerate's offload
        # hooks, so enable_model_cpu_offload() fails with a CPU/CUDA type
        # mismatch.  That makes the memory budget the binding constraint, and
        # on a 16 GB card an int8 transformer alone saturates it: measured at
        # 15.9 of 15.9 GiB resident and 638 s per denoising step, unchanged
        # when T5 was dropped to int4.  int4 on the transformer brings it to
        # ~6 GB and restores normal speed.
        #
        # Quantising the module that carries the subject LoRA is exactly the
        # thing that could manufacture a false negative here, so the design
        # carries its own control: the `ominicontrol` configuration is the
        # same adapter at full strength with no gating, and it is expected to
        # transfer identity strongly.  If it does, the injection path
        # survived quantisation and a null result for `conceptrol` means
        # something; if it does not, the quantisation broke the adapter and
        # the probe is void.  That check is reported alongside the result.
        w = qint8 if args.int8 else qint4
        print(f"Quantising transformer ({w}) and T5 (int4) ...", flush=True)
        quantize(pipe.transformer, weights=w)
        freeze(pipe.transformer)
        quantize(pipe.text_encoder_2, weights=qint4)
        freeze(pipe.text_encoder_2)
        pipe.to("cuda")
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM after load: {(total-free)/2**30:.1f} / "
              f"{total/2**30:.1f} GiB", flush=True)
    print("ready\n", flush=True)

    todo = [s for s in samples
            if not all((OUT / c / f"{s['sample_id']}.png").exists()
                       for c in CONFIGS)]
    print(f"{len(todo)} of {len(samples)} samples remaining", flush=True)

    for i, s in enumerate(todo, 1):
        sid = s["sample_id"]
        prompt = s["full_prompt"]
        subject = head_phrase(s["regional_prompts"].split("BREAK")[0].strip())
        ref = Image.open(ROOT / s["reference_image_paths"][0]).convert("RGB")
        t = time.perf_counter()

        for name in CONFIGS:
            dst = OUT / name / f"{sid}.png"
            if dst.exists():
                continue
            if name == "text_only":
                # same sampler, same seed, no reference: the floor every
                # injected configuration has to beat
                pipe.load_conceptrol(Conceptrol({"name": "ominicontrol"}))
                img = pipe(prompt=prompt, image=ref, subject=subject,
                           num_inference_steps=args.steps,
                           guidance_scale=args.cfg, condition_scale=0.0,
                           height=args.size, width=args.size,
                           seed=args.seed).images[0]
            else:
                pipe.load_conceptrol(Conceptrol({"name": name}))
                img = pipe(prompt=prompt, image=ref, subject=subject,
                           num_inference_steps=args.steps,
                           guidance_scale=args.cfg, condition_scale=args.scale,
                           height=args.size, width=args.size,
                           seed=args.seed).images[0]
            img.save(dst)

        print(f"[{i:3d}/{len(todo)}] {sid}  {time.perf_counter()-t:.1f}s",
              flush=True)

    del pipe
    torch.cuda.empty_cache()

# ── metrics ──────────────────────────────────────────────────────────────────
print("\nScoring with DINOv2 ...", flush=True)
from transformers import AutoImageProcessor, AutoModel

proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()


@torch.no_grad()
def embed(img):
    x = proc(images=img, return_tensors="pt").to("cuda")
    v = dino(**x).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(v, dim=-1)


by_cat = {}
for s in samples:
    by_cat.setdefault(s["metadata"]["category"], []).append(s)

res = {c: {"id": [], "spec": [], "dpix": []} for c in CONFIGS}
base_px = {}
for s in samples:
    sid, cat = s["sample_id"], s["metadata"]["category"]
    ref = Image.open(ROOT / s["reference_image_paths"][0]).convert("RGB")
    e_ref = embed(ref)
    # distractors: first references of other samples in the same category
    distr = [Image.open(ROOT / o["reference_image_paths"][0]).convert("RGB")
             for o in by_cat[cat] if o["sample_id"] != sid][:4]
    e_dis = torch.cat([embed(d) for d in distr]) if distr else None

    for c in CONFIGS:
        p = OUT / c / f"{sid}.png"
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        e = embed(img)
        sim = float(e @ e_ref.T)
        res[c]["id"].append(sim)
        if e_dis is not None:
            res[c]["spec"].append(sim - float((e @ e_dis.T).mean()))
        px = np.asarray(img, np.float32)
        if c == "text_only":
            base_px[sid] = px
        elif sid in base_px:
            res[c]["dpix"].append(float(np.abs(px - base_px[sid]).mean()))

print(f"\n{'config':<14}{'DINO-ID':>10}{'specificity':>13}{'d_pix vs text':>15}{'n':>5}")
mean = lambda v: sum(v) / len(v) if v else float("nan")
summary = {}
for c in CONFIGS:
    r = res[c]
    summary[c] = {"dino_id": mean(r["id"]), "specificity": mean(r["spec"]),
                  "dpix_vs_text_only": mean(r["dpix"]), "n": len(r["id"])}
    d = summary[c]
    print(f"{c:<14}{d['dino_id']:>10.4f}{d['specificity']:>13.4f}"
          f"{d['dpix_vs_text_only']:>15.2f}{d['n']:>5}")

(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(f"\nWritten to {OUT/'summary.json'}")
