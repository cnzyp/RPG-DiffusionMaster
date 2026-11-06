"""DINO identity on subject-tight crops.

The reported DINO-ID crops each region and compares it whole to the reference.
When the subject fills the region (a bust-framed portrait) the crop is almost
all subject; when the subject stands in a scene it is mostly background, so the
same identity scores lower for reasons that have nothing to do with identity.
This recomputes it on the segmented subject's bounding box on both sides.
"""
import argparse, json, os
import numpy as np, torch
from pathlib import Path
from PIL import Image
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from transformers import AutoModel, AutoImageProcessor
from rembg import new_session, remove

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--variants", nargs="+", required=True)   # label=path pairs
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
sess = new_session("u2net")

def subject_crop(im, pad=0.08):
    m = np.asarray(remove(im, session=sess, only_mask=True), np.float32) / 255.0
    ys, xs = np.where(m > 0.5)
    if len(xs) < 64:
        return im
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    px, py = int(w * pad), int(h * pad)
    return im.crop((max(0, x0 - px), max(0, y0 - py),
                    min(im.width, x1 + px), min(im.height, y1 + py)))

@torch.no_grad()
def feat(im):
    t = proc(images=im, return_tensors="pt").to(dev)
    f = model(**t).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(f, dim=-1)

S = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]
refc = {}
for s in S:
    for p in s["reference_image_paths"]:
        if p not in refc:
            refc[p] = feat(subject_crop(Image.open(p).convert("RGB")))

print(f"{'variant':<24}{'DINO(region)':>14}{'DINO(subject)':>15}")
for spec in args.variants:
    lab, root = spec.split("=", 1)
    reg_s, sub_s, n = 0.0, 0.0, 0
    for s in S:
        q = Path(root) / f"{s['sample_id']}.png"
        if not q.exists():
            continue
        im = Image.open(q).convert("RGB")
        W, k = im.width, len(s["reference_image_paths"])
        for i, rp in enumerate(s["reference_image_paths"]):
            reg = im.crop((int(W * i / k), 0, int(W * (i + 1) / k), im.height))
            reg_s += float(feat(reg) @ refc[rp].T)
            sub_s += float(feat(subject_crop(reg)) @ refc[rp].T)
            n += 1
    if n:
        print(f"{lab:<24}{reg_s/n:>14.4f}{sub_s/n:>15.4f}")
