"""Two subjects on a DiT backbone.

Every configuration in this study so far runs on SDXL, a UNet whose image
conditioning enters through a parallel cross-attention branch.  The paper's
ladder ends by noting that the one remaining injection site -- writing a
reference into the sequence the model already attends over -- needs a backbone
whose tokens carry semantics, i.e. a DiT.

FLUX is that backbone, and OminiControl already writes a reference into the
sequence as extra image tokens.  Its pipeline capped conditions at one; the
machinery underneath was already list-shaped, so lifting the cap and giving
each subject its own position band is enough to ask the question this study
could not ask on SDXL: does putting both references in the sequence give two
identities without the paste-like collapse?

    python run_flux_multi.py --dataset datasets/dreambench2/hh_medium.json --limit 4
"""
import argparse, json, os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "third_party/Conceptrol/gradio_src"))

import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--out-root", default="outputs/flux_multi")
ap.add_argument("--variant", default="omini2")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--cfg", type=float, default=3.5)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--single", action="store_true",
                help="control: pass only the first reference, so any "
                     "difference is the second condition")
ap.add_argument("--offload", action="store_true",
                help="bf16 with sequential CPU offload instead of int4.  Slow "
                     "per step but starts immediately, so a failing "
                     "configuration can be diagnosed without paying an hour "
                     "of quantisation first.")
ap.add_argument("--trace", action="store_true",
                help="print the full traceback on the first failure")
ap.add_argument("--no-cond", action="store_true",
                help="control: condition_scale 0, the text-only floor")
args = ap.parse_args()

OUT = ROOT / args.out_root / args.variant / "images"
OUT.mkdir(parents=True, exist_ok=True)

from optimum.quanto import freeze, qint4, quantize
from omini_control.conceptrol import Conceptrol
from omini_control.flux_conceptrol_pipeline import FluxConceptrolPipeline

print("Loading FLUX.1-schnell ...", flush=True)
pipe = FluxConceptrolPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
# the LoRA must go on before quantisation: quanto replaces each Linear with a
# QLinear whose state dict no longer carries the keys the loader resolves
pipe.load_lora_weights("Yuanshi/OminiControl",
                       weight_name="omini/subject_512.safetensors",
                       adapter_name="subject")
if args.offload:
    print("Sequential CPU offload (bf16) ...", flush=True)
    pipe.enable_sequential_cpu_offload()
else:
    print("Quantising transformer and T5 to int4 ...", flush=True)
    quantize(pipe.transformer, weights=qint4); freeze(pipe.transformer)
    quantize(pipe.text_encoder_2, weights=qint4); freeze(pipe.text_encoder_2)
    pipe = pipe.to("cuda")
pipe.load_conceptrol(Conceptrol({"name": "ominicontrol"}))

samples = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["samples"]
if args.limit:
    samples = samples[:args.limit]
print(f"{args.variant}: {len(samples)} samples, "
      f"{'text only' if args.no_cond else ('1 ref' if args.single else '2 refs')}",
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
        if args.single:
            refs = refs[:1]
        img = pipe(
            prompt=s["full_prompt"],
            image=refs,
            subject=s["concept_names"][0],
            condition_scale=0.0 if args.no_cond else args.scale,
            height=args.size, width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed,
        ).images[0]
        img.save(dst)
        done += 1
        print(f"[{k+1}/{len(samples)}] {sid}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{k+1}/{len(samples)}] FAIL {sid}: {type(e).__name__}: {e}",
              flush=True)
        if args.trace and fail == 1:
            import traceback; traceback.print_exc()

n = len(list(OUT.glob("*.png")))
print(f"PNG count: {n}/{len(samples)}  ({done} ok, {fail} failed)  ->  {OUT}",
      flush=True)
