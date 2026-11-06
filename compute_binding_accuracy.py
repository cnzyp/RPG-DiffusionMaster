"""
Region-level text-binding accuracy: did the requested concept land in the
region it was assigned to?

Margin answers whether a region resembles its own reference more than another
sample's, but it cannot separate two different failures that score alike: a
region that renders the wrong subject, and a region that renders nothing the
prompt asked for.  MS-Diffusion at N=3 shows both - concepts dropped, concepts
displaced into a neighbouring strip - and Margin reports only that the number
got worse.

This metric is threshold-free.  For every region i we take its crop and score
it with CLIP against each of the sample's own regional sub-prompts.  Region i
is bound correctly when the crop scores highest on sub-prompt i.  Chance is
1/N, so a method that ignores layout entirely lands near 0.5 at N=2 and 0.33
at N=3, and a method that places every concept correctly lands at 1.0.  The
comparison is within one image and within one sample, so no threshold and no
cross-sample calibration is involved.

Usage:
    python compute_binding_accuracy.py
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/mixed_eval_v1"
DS = ROOT / "datasets/mixed_eval_v1/mixed_eval_v1.json"
H = W = 768

VARIANTS = [
    ("rpg_only", "RPG-only"),
    ("ip_global_real", "IP-Adapter (global)"),
    ("conceptrol_official_s08", "Conceptrol"),
    ("ours_gating_fixed", "Attention gating"),
    ("msdiffusion_s08_inset", "MS-Diffusion"),
    ("ours_methodA_v2", "Ours, tau=1"),
    ("ours_tau070", "Ours, tau=0.7"),
    ("ours_tau050", "Ours"),
    ("ours_tau030", "Ours, tau=0.3"),
    ("ours_tau000", "Ours, tau=0"),
    ("ours_tau070_1024", "Ours @1024"),
    ("msdiff_1024", "MS-Diffusion @1024"),
    ("conceptrol_1024", "Conceptrol @1024"),
    ("ipglobal_1024", "IP-Adapter (global) @1024"),
    ("gating_1024", "Attention gating @1024"),
    ("rpgonly_1024", "RPG-only @1024"),
    ("ours_tau000_1024", "ours_tau000_1024"),
    ("ours_tau030_1024", "ours_tau030_1024"),
    ("ours_tau050_1024", "ours_tau050_1024"),
    ("ours_tau100_1024", "ours_tau100_1024"),
    ("ours_fullcanvas_1024", "ours_fullcanvas_1024"),
]

from PIL import Image
from experiments.metrics.identity_fidelity import (
    create_region_masks_from_split_ratio)
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
samples = json.loads(DS.read_text(encoding="utf-8"))["samples"]


import numpy as np


def crop_to_mask(img, mask):
    ys, xs = np.nonzero(np.asarray(mask))
    return img.crop((int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1))


@torch.no_grad()
def score_matrix(crops, prompts):
    """[n_crops, n_prompts] cosine similarities."""
    ii = proc(images=crops, return_tensors="pt").to(device)
    ti = proc(text=prompts, return_tensors="pt", padding=True,
              truncation=True, max_length=77).to(device)
    ie = torch.nn.functional.normalize(clip.get_image_features(**ii), dim=-1)
    te = torch.nn.functional.normalize(clip.get_text_features(**ti), dim=-1)
    return (ie @ te.T).float().cpu()


res = {v: {2: [], 3: []} for v, _ in VARIANTS}
per_sample = {v: {} for v, _ in VARIANTS}
for s in samples:
    sid = s["sample_id"]
    rc = s["metadata"].get("region_count", len(s["reference_image_paths"]))
    sub = [p.strip() for p in s["regional_prompts"].split("BREAK")]
    masks = create_region_masks_from_split_ratio(s["split_ratio"], H, W,
                                                direction="horizontal")
    for var, _ in VARIANTS:
        p = OUT / var / "images" / f"{sid}.png"
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB").resize((W, H))
        crops = [crop_to_mask(img, m) for m in masks]
        M = score_matrix(crops, sub)
        correct = sum(1 for i in range(len(crops)) if int(M[i].argmax()) == i)
        res[var][rc].append(correct / len(crops))
        per_sample[var][sid] = correct / len(crops)

mean = lambda x: sum(x) / len(x) if x else float("nan")
print(f"\nRegion-level text-binding accuracy "
      f"(chance = 1/N; 0.50 at N=2, 0.33 at N=3)")
print(f"{'method':<22}{'N=2':>9}{'N=3':>9}{'all':>9}{'vs chance':>12}")
summary = {}
for var, name in VARIANTS:
    a2, a3 = mean(res[var][2]), mean(res[var][3])
    allv = mean(res[var][2] + res[var][3])
    chance = (0.5 * len(res[var][2]) + (1 / 3) * len(res[var][3])) / \
             max(len(res[var][2]) + len(res[var][3]), 1)
    summary[name] = {"n2": a2, "n3": a3, "all": allv, "chance": chance}
    print(f"{name:<22}{a2:>9.3f}{a3:>9.3f}{allv:>9.3f}{allv - chance:>+12.3f}")

(OUT / "binding_accuracy.json").write_text(json.dumps(summary, indent=1))
(OUT / "binding_accuracy_per_sample.json").write_text(
    json.dumps(per_sample, indent=1))

# ── paired tests against Ours ────────────────────────────────────────────────
import math

ours = per_sample["ours_tau050"]
print(f"\nPaired vs Ours (+ favours Ours)")
print(f"{'comparison':<24}{'diff':>9}{'95% CI':>20}{'p':>9}{'wins':>9}")
for var, name in VARIANTS:
    if var == "ours_tau050":
        continue
    ids = sorted(set(ours) & set(per_sample[var]))
    if not ids:
        continue
    d = [ours[i] - per_sample[var][i] for i in ids]
    n = len(d)
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n) if n else float("nan")
    t = mu / se if se else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    ci = f"[{mu-1.96*se:+.3f},{mu+1.96*se:+.3f}]"
    tag = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"{'Ours vs ' + name:<24}{mu:>+9.3f}{ci:>20}{p:>9.4f}"
          f"{sum(1 for x in d if x > 0):>5}/{n:<3} {tag}")
print(f"\nWritten to {OUT/'binding_accuracy.json'}")
