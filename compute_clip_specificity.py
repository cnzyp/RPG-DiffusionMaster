"""
CLIP-I Identity Specificity metric.

IP-Adapter is trained to transfer the reference's *CLIP image features* into
the output. Therefore CLIP-I (image-image similarity), not ArcFace (face
geometry) or DINO (category), is the axis where reference injection should
show measurable value -- especially for OBJECTS where text is least specific.

For each generated region crop:
    specificity = CLIP-I(crop, assigned_ref)
                  - mean_{d in same-category, other samples} CLIP-I(crop, d)

Expected if the method works:
  - Ours > RPG-only          (visual injection adds appearance the text lacks)
  - Ours > Ours(no-gating)   (correct per-region ref beats a blended ref)
  - Effect strongest on OBJECT category.

Usage:
    python compute_clip_specificity.py
"""

import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(Path(__file__).parent)
sys.path.insert(0, ".")

DATASET   = "datasets/mixed_eval_v1/mixed_eval_v1.json"
IMG_ROOT  = "outputs/mixed_eval_v1"
VARIANTS  = ["ours", "rpg_only", "ours_no_region_gating", "ip_adapter_only", "ours_methodA_v2", "ours_fullcanvas", "ours_tau050", "ours_tau000", "ours_tau030", "ours_tau070", "ours_gating_fair", "conceptrol_official_s1", "conceptrol_official_s08", "ip_global_real", "ip_faceid_real_s08", "ip_faceid_real_s06", "msdiffusion_s06", "msdiffusion_s08", "msdiffusion_s08_inset", "ours_gating_fixed", "ours_tau070_1024", "msdiff_1024", "conceptrol_1024", "ipglobal_1024", "gating_1024", "rpgonly_1024", "ours_tau000_1024", "ours_tau030_1024", "ours_tau050_1024", "ours_tau100_1024", "ours_fullcanvas_1024", "ours_tau070_scaling1024", "msdiffusion_scaling_1024", "ipfaceid_1024", "sra_soft_1024"]

from experiments.metrics.identity_fidelity import IdentityFidelityMetric
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP on {device}...")
clip = IdentityFidelityMetric(metric_type="clip", device=device)


def feat(pil_img):
    """Return L2-normed CLIP image feature (numpy, 768)."""
    return clip.extract_features(pil_img.convert("RGB")).squeeze(0).cpu().numpy().astype(np.float32)


def bbox_from_split(split_str, ri, direction="horizontal"):
    parts = [float(x) for x in split_str.split(",")]
    total = sum(parts)
    cum = [sum(parts[:i]) / total for i in range(len(parts) + 1)]
    if direction == "horizontal":
        return (cum[ri], 0.0, cum[ri+1], 1.0)
    return (0.0, cum[ri], 1.0, cum[ri+1])


def crop_region(img, bbox, H, W):
    x0, y0, x1, y1 = bbox
    return img.crop((int(x0*W), int(y0*H), int(x1*W), int(y1*H)))


data = json.loads(Path(DATASET).read_text(encoding="utf-8"))
samples = data["samples"]
print(f"Dataset: {len(samples)} samples\n")

# ── Precompute reference (ref-A) CLIP features once ──────────────────────────
print("Embedding reference images (ref-A) with CLIP...")
assigned_feat = {}     # (sid, ri) -> feat
pool = {}              # category -> list of (sid, feat)
for s in samples:
    cat = s["metadata"]["category"]
    for ri, p in enumerate(s["reference_image_paths"]):
        e = feat(Image.open(p))
        assigned_feat[(s["sample_id"], ri)] = e
        pool.setdefault(cat, []).append((s["sample_id"], e))

pool_mat = {c: np.stack([e for _, e in lst]) for c, lst in pool.items()}
pool_sid = {c: np.array([sid for sid, _ in lst]) for c, lst in pool.items()}
for c, lst in pool.items():
    print(f"  {c:8s}: {len(lst)} refs in distractor pool")
print()


def distractor_mean(cat, crop_f, cur_sid):
    sims = pool_mat[cat] @ crop_f
    mask = pool_sid[cat] != cur_sid
    return float(sims[mask].mean())


def evaluate(variant):
    img_dir = Path(IMG_ROOT) / variant / "images"
    recs = []
    for s in samples:
        sid, cat = s["sample_id"], s["metadata"]["category"]
        ip = img_dir / f"{sid}.png"
        if not ip.exists():
            continue
        gen = Image.open(ip).convert("RGB")
        W, H = gen.size
        for ri in range(len(s["reference_image_paths"])):
            a = assigned_feat[(sid, ri)]
            crop = crop_region(gen, bbox_from_split(s["split_ratio"], ri), H, W)
            c = feat(crop)
            sa = float(c @ a)
            sd = distractor_mean(cat, c, sid)
            recs.append({"sid": sid, "cat": cat, "ri": ri,
                         "assigned": sa, "distractor": sd,
                         "specificity": sa - sd})
    return recs


def agg(recs, cat=None):
    sub = [r for r in recs if cat is None or r["cat"] == cat]
    if not sub:
        return None
    return {"n": len(sub),
            "assigned":    float(np.mean([r["assigned"]    for r in sub])),
            "distractor":  float(np.mean([r["distractor"]  for r in sub])),
            "specificity": float(np.mean([r["specificity"] for r in sub]))}


results = {}
for v in VARIANTS:
    if not (Path(IMG_ROOT) / v / "images").exists():
        print(f"skip {v} (no images)"); continue
    print(f"Evaluating {v} ...")
    recs = evaluate(v)
    results[v] = {c: agg(recs, c) for c in [None, "person", "object", "mixed"]}
    Path(f"{IMG_ROOT}/{v}/clip_specificity.json").write_text(
        json.dumps({"summary": results[v]}, indent=2, default=str), encoding="utf-8")

# ── Report ──────────────────────────────────────────────────────────────────
print("\n" + "="*78)
print("CLIP-I SPECIFICITY = CLIP-I(crop, assigned) - CLIP-I(crop, same-cat distractor)")
print("Higher = output reproduces the SPECIFIC reference's appearance")
print("="*78)
for scope, label in [("object","OBJECT"), ("person","PERSON"), ("mixed","MIXED"), (None,"ALL")]:
    print(f"\n[{label}]")
    print(f"  {'variant':24s}  {'n':>4s}  {'assigned':>9s}  {'distract':>9s}  {'SPECIFICITY':>12s}")
    for v in results:
        a = results[v][scope]
        if a is None:
            continue
        print(f"  {v:24s}  {a['n']:4d}  {a['assigned']:9.4f}  {a['distractor']:9.4f}  {a['specificity']:+12.4f}")
print("\n" + "="*78)
print("KEY TESTS:")
for scope in ["object", "person", None]:
    lbl = scope or "all"
    o = results.get("ours",{}).get(scope); r = results.get("rpg_only",{}).get(scope)
    ng = results.get("ours_no_region_gating",{}).get(scope)
    if o and r:
        line = f"  [{lbl:7s}] Ours {o['specificity']:+.4f}  vs RPG-only {r['specificity']:+.4f}  -> DELTA {o['specificity']-r['specificity']:+.4f}"
        if ng:
            line += f"   | Ours vs no-gating {o['specificity']-ng['specificity']:+.4f}"
        print(line)
print("="*78)
