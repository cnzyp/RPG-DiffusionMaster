"""
Re-score existing SDXL runs with the metric the FLUX probe uses.

flux_gating_probe.py measures whole-image DINOv2 similarity to reference A and
specificity against same-category distractors, because Conceptrol's FLUX
pipeline has no layout to crop to.  Section 4.4's numbers are per-region and
two-reference, so they cannot be compared with it directly.  This script
applies the FLUX metric to SDXL renders that already exist, which makes the
two backbones comparable without generating anything.

    rpg_only          text-only regional diffusion, no reference   (floor)
    ip_global_real    IP-Adapter on reference A, ungated, global   (upper)
    ours_gating_fixed  same adapter, full strength, masked per region

The first two are the SDXL analogue of the FLUX probe's `text_only` and
`ominicontrol`.  The third is the configuration Section 4.4 reports as failing.
Reference A is the only one injected in ip_global_real, so all three are scored
against reference A alone.

Usage:
    python sdxl_gating_probe.py
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parent
DS = ROOT / "datasets/mixed_eval_v1/mixed_eval_v1.json"
OUT = ROOT / "outputs/mixed_eval_v1"
VARIANTS = ["rpg_only", "ip_global_real", "ours_gating_fixed", "ours_tau050"]

samples = json.loads(DS.read_text(encoding="utf-8"))["samples"]

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

res = {v: {"id": [], "spec": []} for v in VARIANTS}
for s in samples:
    sid, cat = s["sample_id"], s["metadata"]["category"]
    e_ref = embed(Image.open(ROOT / s["reference_image_paths"][0]).convert("RGB"))
    distr = [Image.open(ROOT / o["reference_image_paths"][0]).convert("RGB")
             for o in by_cat[cat] if o["sample_id"] != sid][:4]
    e_dis = torch.cat([embed(d) for d in distr]) if distr else None

    for v in VARIANTS:
        p = OUT / v / "images" / f"{sid}.png"
        if not p.exists():
            continue
        e = embed(Image.open(p).convert("RGB"))
        sim = float(e @ e_ref.T)
        res[v]["id"].append(sim)
        if e_dis is not None:
            res[v]["spec"].append(sim - float((e @ e_dis.T).mean()))

mean = lambda x: sum(x) / len(x) if x else float("nan")
floor_id = mean(res["rpg_only"]["id"])
floor_sp = mean(res["rpg_only"]["spec"])

print(f"\nSDXL, whole-image vs reference A ({len(samples)} samples)")
print(f"{'variant':<20}{'DINO-ID':>10}{'lift':>10}{'specificity':>14}{'lift':>10}{'n':>5}")
summary = {}
for v in VARIANTS:
    i, sp = mean(res[v]["id"]), mean(res[v]["spec"])
    summary[v] = {"dino_id": i, "specificity": sp, "dino_lift": i - floor_id,
                  "spec_lift": sp - floor_sp, "n": len(res[v]["id"])}
    print(f"{v:<20}{i:>10.4f}{i-floor_id:>+10.4f}{sp:>14.4f}{sp-floor_sp:>+10.4f}"
          f"{len(res[v]['id']):>5}")

(OUT / "sdxl_gating_probe.json").write_text(json.dumps(summary, indent=1))
print(f"\nWritten to {OUT/'sdxl_gating_probe.json'}")
