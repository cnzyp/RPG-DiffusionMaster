"""ArcFace similarity on detected faces: identity measured independently of framing.

DINO-ID crops a region and compares it whole to the reference, so a subject that
fills its region scores higher than the same subject standing in a scene.  Across
seven methods that framing effect explains almost all of the variance
(r = 0.949 between face area and DINO-ID), which makes DINO-ID unusable for
deciding whether a wide shot carries the right person.

ArcFace normalises every detected face to 112x112 before embedding, so this score
asks only whether the face is the reference's face.  Faces are matched to
references left to right, the same order the layout assigns regions.
"""
import argparse, json, os, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from PIL import Image
from insightface.app import FaceAnalysis

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--variants", nargs="+", required=True, help="label=imagedir")
args = ap.parse_args()

app = FaceAnalysis(name="buffalo_l",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def faces(im):
    return app.get(np.asarray(im.convert("RGB"))[:, :, ::-1])


S = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]
ref_emb = {}
for s in S:
    for p in s["reference_image_paths"]:
        if p in ref_emb:
            continue
        fs = faces(Image.open(p))
        ref_emb[p] = (max(fs, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                      .normed_embedding if fs else None)

print(f"{'variant':<26}{'ArcFace':>9}{'faces/img':>11}{'face area %':>13}{'n':>5}")
for spec in args.variants:
    lab, root = spec.split("=", 1)
    root = Path(root)
    sims, nf, area, n = [], 0, 0.0, 0
    for s in S:
        q = root / f"{s['sample_id']}.png"
        if not q.exists():
            continue
        im = Image.open(q).convert("RGB")
        fs = sorted(faces(im), key=lambda f: (f.bbox[0]+f.bbox[2])/2)
        n += 1; nf += len(fs)
        A = im.width * im.height
        area += sum((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in fs) / A
        rps = s["reference_image_paths"]
        for i, rp in enumerate(rps):
            r = ref_emb.get(rp)
            if r is None:
                continue
            if i < len(fs):
                sims.append(float(np.dot(fs[i].normed_embedding, r)))
            else:
                sims.append(0.0)      # a subject with no face rendered scores zero
    if n:
        print(f"{lab:<26}{np.mean(sims):>9.4f}{nf/n:>11.2f}{100*area/n:>12.2f}%{n:>5}")
