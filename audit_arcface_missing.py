"""How much of the reported ArcFace advantage depends on dropping missing faces.

compute_face_metrics.py averages over the regions where a face was detected and
skips samples where none was.  A method that renders one face instead of two is
then scored on the one it rendered, and a method that renders none is excluded
from the mean entirely -- so failing to produce a subject costs nothing.

This recomputes the same score under both policies: drop (as reported) and
zero (a region with no face scores 0, which is what a reader assumes).
"""
import argparse, json, os, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from PIL import Image
from insightface.app import FaceAnalysis

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="datasets/mixed_eval_v1/mixed_eval_v1.json")
ap.add_argument("--root", default="outputs/mixed_eval_v1")
ap.add_argument("--variants", nargs="+", required=True)
ap.add_argument("--category", default="person")
args = ap.parse_args()

app = FaceAnalysis(name="buffalo_l",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def emb(im):
    fs = app.get(np.asarray(im.convert("RGB"))[:, :, ::-1])
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
    return f.normed_embedding


S = [s for s in json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]
     if s.get("metadata", {}).get("category") == args.category]
ref = {}
for s in S:
    for p in s["reference_image_paths"]:
        if p not in ref:
            ref[p] = emb(Image.open(p))

print(f"category={args.category}  samples={len(S)}\n")
print(f"{'variant':<26}{'drop (reported)':>17}{'zero':>9}{'detected':>11}")
for v in args.variants:
    d = Path(args.root) / v / "images"
    kept, zeroed, n_reg, n_det = [], [], 0, 0
    for s in S:
        q = d / f"{s['sample_id']}.png"
        if not q.exists():
            continue
        im = Image.open(q).convert("RGB")
        W, H = im.size
        parts = [float(x) for x in s["split_ratio"].split(",")]
        tot = sum(parts)
        cum = [sum(parts[:i])/tot for i in range(len(parts)+1)]
        for i, rp in enumerate(s["reference_image_paths"]):
            r = ref.get(rp)
            if r is None:
                continue
            n_reg += 1
            crop = im.crop((int(W*cum[i]), 0, int(W*cum[i+1]), H))
            e = emb(crop)
            if e is None:
                zeroed.append(0.0)
            else:
                n_det += 1
                sim = float(np.dot(e, r))
                kept.append(sim); zeroed.append(sim)
    if n_reg:
        print(f"{v:<26}{np.mean(kept) if kept else float('nan'):>17.4f}"
              f"{np.mean(zeroed):>9.4f}{n_det}/{n_reg:>10}")
