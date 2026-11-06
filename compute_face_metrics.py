"""
ArcFace-based face identity metric.

Computes per-region face similarity between reference images and generated
image crops. This is the RIGHT metric for personalization papers:
  - DINOv2 → measures semantic category (both are "person" → high score)
  - ArcFace → measures FACE IDENTITY (only the same person → high score)

Usage:
    python compute_face_metrics.py --variant ours
    python compute_face_metrics.py --variant rpg_only
"""

import argparse, json, os, traceback
from pathlib import Path
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(Path(__file__).parent)

parser = argparse.ArgumentParser()
parser.add_argument("--variant",     default="ours",
                    choices=["ours", "rpg_only", "ip_adapter_only", "ours_no_region_gating",
                             "conceptrol", "photomaker", "ip_faceid", "instantid",
                             "ours_adaptive", "ours_adaptive_768", "ours_1024",
                             "ours_methodA_v2", "ours_fullcanvas", "ours_tau050", "ours_tau000", "ours_tau030", "ours_tau070",
                             "ours_gating_fair",
                             "conceptrol_official_s1", "conceptrol_official_s08",
                             "ip_global_real",
                             "ip_faceid_real_s08", "ip_faceid_real_s06",
                             "msdiffusion_s06", "msdiffusion_s08", "msdiffusion_s08_inset",
                             "ours_gating_fixed", "ours_tau070_1024", "msdiff_1024", "conceptrol_1024", "ipglobal_1024", "gating_1024", "rpgonly_1024", "ours_tau000_1024", "ours_tau030_1024", "ours_tau050_1024", "ours_tau100_1024", "ours_fullcanvas_1024", "ours_tau070_scaling1024", "msdiffusion_scaling_1024", "ipfaceid_1024", "sra_soft_1024"])
parser.add_argument("--dataset",     default="datasets/mixed_eval_v1/mixed_eval_v1.json")
parser.add_argument("--images-root", default="outputs/mixed_eval_v1")
parser.add_argument("--height",      type=int, default=768)
parser.add_argument("--width",       type=int, default=768)
args = parser.parse_args()

# ── Load ArcFace ──────────────────────────────────────────────────────────────
from insightface.app import FaceAnalysis

print("Loading ArcFace (buffalo_l)...")
face_app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("ArcFace ready.")


def get_face_embedding(pil_img):
    """Returns ArcFace embedding or None if no face detected."""
    img_np = np.array(pil_img.convert("RGB"))
    faces = face_app.get(img_np)
    if not faces:
        return None
    # Return embedding of largest face
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return largest.normed_embedding  # (512,) normalized


def cosine_sim(a, b):
    return float(np.dot(a, b))  # both already normed


def crop_region(pil_img, bbox_norm, H, W):
    """Crop image to bbox. bbox_norm = (x0, y0, x1, y1) normalized [0,1]."""
    x0, y0, x1, y1 = bbox_norm
    left   = int(x0 * W)
    top    = int(y0 * H)
    right  = int(x1 * W)
    bottom = int(y1 * H)
    return pil_img.crop((left, top, right, bottom))


def bbox_from_split(split_ratio_str, region_idx, direction="horizontal"):
    """Convert split_ratio string (e.g. '1,1' or '1,2,1') to normalized bbox."""
    parts = [float(x) for x in split_ratio_str.split(",")]
    total = sum(parts)
    cumulative = [sum(parts[:i]) / total for i in range(len(parts) + 1)]
    if direction == "horizontal":
        x0 = cumulative[region_idx]
        x1 = cumulative[region_idx + 1]
        return (x0, 0.0, x1, 1.0)
    else:
        y0 = cumulative[region_idx]
        y1 = cumulative[region_idx + 1]
        return (0.0, y0, 1.0, y1)


# ── Paths ─────────────────────────────────────────────────────────────────────
images_dir   = Path(args.images_root) / args.variant / "images"
out_dir      = Path(args.images_root) / args.variant
results_path = out_dir / "face_metrics_per_sample.json"
summary_path = out_dir / "face_metrics_summary.json"

data    = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
samples = data["samples"]
print(f"Dataset: {len(samples)} samples | Variant: {args.variant}")

# Resume support
existing = {}
if results_path.exists():
    try:
        for item in json.loads(results_path.read_text(encoding="utf-8")):
            existing[item["sample_id"]] = item
        print(f"Resuming — {len(existing)} already done")
    except Exception:
        pass

results, skipped = list(existing.values()), []

for idx, sample in enumerate(samples):
    sid = sample["sample_id"]

    if sid in existing:
        print(f"[{idx+1:3d}/150] SKIP {sid}")
        continue

    img_path = images_dir / f"{sid}.png"
    if not img_path.exists():
        print(f"[{idx+1:3d}/150] MISS {sid}")
        skipped.append(sid)
        continue

    try:
        gen_img  = Image.open(img_path).convert("RGB")
        W, H     = gen_img.size
        split    = sample["split_ratio"]
        n_regions = len(sample["reference_image_paths"])

        region_sims_a, region_sims_b = [], []
        detected_regions = 0

        for ri in range(n_regions):
            bbox   = bbox_from_split(split, ri, direction="horizontal")
            crop   = crop_region(gen_img, bbox, H, W)

            # Reference A embedding
            ref_a  = Image.open(sample["reference_image_paths"][ri]).convert("RGB")
            extra_paths = sample.get("reference_image_paths_extra",
                                     sample["reference_image_paths"])
            ref_b  = Image.open(extra_paths[ri]).convert("RGB")

            emb_crop = get_face_embedding(crop)
            emb_a    = get_face_embedding(ref_a)
            emb_b    = get_face_embedding(ref_b)

            if emb_crop is None or emb_a is None:
                region_sims_a.append(None)
                region_sims_b.append(None)
                continue

            detected_regions += 1
            sim_a = cosine_sim(emb_crop, emb_a)
            region_sims_a.append(sim_a)

            if emb_b is not None:
                sim_b = cosine_sim(emb_crop, emb_b)
                region_sims_b.append(sim_b)
            else:
                region_sims_b.append(None)

        valid_a   = [s for s in region_sims_a if s is not None]
        valid_b   = [s for s in region_sims_b if s is not None]
        mean_a    = float(np.mean(valid_a)) if valid_a else None
        mean_b    = float(np.mean(valid_b)) if valid_b else None
        mean_both = float(np.mean(valid_a + valid_b)) if (valid_a or valid_b) else None

        record = {
            "sample_id":          sid,
            "category":           sample["metadata"]["category"],
            "region_count":       n_regions,
            "face_id_mean_ref_a": mean_a,
            "face_id_mean_ref_b": mean_b,
            "face_id_mean":       mean_both,
            "detected_regions":   detected_regions,
            "total_regions":      n_regions,
        }

        existing[sid] = record
        results.append(record)
        results_path.write_text(
            json.dumps(list(existing.values()), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        status = f"FaceID={mean_a:.3f}" if mean_a is not None else "no-face"
        print(f"[{idx+1:3d}/150] OK   {sid:30s}  {status}  "
              f"detected={detected_regions}/{n_regions}")

    except Exception as e:
        print(f"[{idx+1:3d}/150] ERR  {sid}: {e}")
        traceback.print_exc()
        skipped.append(sid)
        continue

# ── Summary ───────────────────────────────────────────────────────────────────
def avg_nn(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None

def summary_for(cat_filter=None):
    subset = [r for r in results
              if (cat_filter is None or r["category"] == cat_filter)
              and r.get("face_id_mean") is not None]
    if not subset:
        return None
    return {
        "n":           len(subset),
        "face_id":     avg_nn([r["face_id_mean"]    for r in subset]),
        "face_id_a":   avg_nn([r["face_id_mean_ref_a"] for r in subset]),
        "face_id_b":   avg_nn([r["face_id_mean_ref_b"] for r in subset]),
        "detect_rate": np.mean([r["detected_regions"]/max(r["total_regions"],1)
                                for r in subset]),
    }

summary = {
    "variant":  args.variant,
    "all":      summary_for(None),
    "person":   summary_for("person"),
    "object":   summary_for("object"),
    "mixed":    summary_for("mixed"),
    "skipped":  len(skipped),
}
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("\n" + "="*60)
print(f"DONE  variant={args.variant}")
for cat in ["all", "person", "mixed"]:
    s = summary.get(cat)
    if s:
        print(f"  [{cat:6s}]  n={s['n']:3d}  FaceID={s['face_id']:.4f}  "
              f"detect={s['detect_rate']:.2f}")
print(f"Results: {results_path}")
