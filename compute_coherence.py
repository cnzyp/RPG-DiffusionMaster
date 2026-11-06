"""
Does the late global release actually remove the panel structure?

The release point tau is defended in Section 4.8 by a pixel count - how much
of the canvas the global phase repaints - and by figures.  Neither measures
the thing the release exists for: whether the regions read as one scene
rather than as adjacent panels.  This script measures it without human
raters, on renders that already exist.

The measure is a step in the low-frequency field at the region boundary,
normalised by steps of the same width everywhere else in the same image:

    LFD = || L(x_b + d) - L(x_b - d) ||  /  mean_x || L(x + d) - L(x - d) ||

with L the image blurred to remove subject detail.  LFD = 1 means the
boundary is indistinguishable from any other column; LFD > 1 means the two
sides disagree about lighting and background in a way the rest of the image
does not.  The per-image normalisation is what makes it comparable across
samples: a busy scene has large steps everywhere and is not penalised.

Endpoints, because neither number means anything on its own:

    collage   region i taken from a *different sample* of the same category,
              rendered by the same generator, pasted at the same boundaries.
              Independent by construction; the incoherent end.
    tau=0     one full-canvas pass, no regional decomposition; the coherent
              end.

A metric that cannot separate those two is measuring nothing, and the rows
between them cannot be read.  That check is the point of the script.

A first version of this measured mean colour in a top strip per region and
differenced adjacent regions.  It failed the check - the collage scored
*more* coherent than pure sub-canvas - for two reasons worth recording: the
top strip contains heads and hair, so the statistic tracked subject
difference rather than illumination; and that collage was assembled from
tau=0 renders, which are the most homogeneous images in the set precisely
because global injection washes every region toward one appearance.  The
floor was built out of unusually similar parts.  Both faults are fixed here:
blurring removes the subject, per-image normalisation removes scene
variability, and the collage is assembled from the same variant it bounds.

Usage:
    python compute_coherence.py
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

ap = argparse.ArgumentParser()
ap.add_argument("--dataset",
                default="datasets/mixed_eval_v1/mixed_eval_v1.json",
                help="scored samples; a subset restricts every row to it, so "
                     "the baselines stay paired with the variant under test")
ap.add_argument("--extra", action="append", default=[], metavar="LABEL=DIR",
                help="additional variant to score, repeatable")
cli = ap.parse_args()

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/mixed_eval_v1"
DS = ROOT / cli.dataset

SIGMA = 24       # px, blur that removes subject detail but keeps illumination
OFFSET = 48      # px, half-width of the step being probed (> the 64 px blend
                 # band's half-width, so the step is read outside the feather)
EXCLUDE = 96     # px around a true boundary excluded from the baseline

VARIANTS = [
    ("collage (different samples)", "COLLAGE"),
    ("Ours, tau=1", "ours_methodA_v2"),
    ("Ours, tau=0.7", "ours_tau070"),
    ("Ours, tau=0.5", "ours_tau050"),
    ("Ours, tau=0.3", "ours_tau030"),
    ("Ours, tau=0 (one global pass)", "ours_tau000"),
    ("RPG-only (text, full canvas)", "rpg_only"),
] + [(e.split("=", 1)[0], e.split("=", 1)[1]) for e in cli.extra]
COLLAGE_FROM = "ours_methodA_v2"   # same generator as the row it bounds

samples = json.loads(DS.read_text(encoding="utf-8"))["samples"]
by_cat = defaultdict(list)
for s in samples:
    by_cat[s["metadata"]["category"]].append(s["sample_id"])
rng = random.Random(20260821)


def bounds(split, w):
    parts = [float(x) for x in split.split(",")]
    tot = sum(parts)
    cum = [sum(parts[:i]) / tot for i in range(len(parts) + 1)]
    return [int(round(c * w)) for c in cum]


def lowfreq(a):
    """Blur along both axes; subject detail out, illumination field in."""
    b = gaussian_filter1d(a, SIGMA, axis=0, mode="nearest")
    return gaussian_filter1d(b, SIGMA, axis=1, mode="nearest")


def lfd(a, edges):
    """Step at the interior boundaries over the mean step elsewhere."""
    L = lowfreq(a)
    w = L.shape[1]
    xs = np.arange(OFFSET, w - OFFSET)
    step = np.linalg.norm(L[:, xs + OFFSET] - L[:, xs - OFFSET],
                          axis=-1).mean(axis=0)          # per column
    interior = edges[1:-1]
    if not interior:
        return float("nan")
    keep = np.ones(len(xs), bool)
    for x in interior:
        keep &= np.abs(xs - x) > EXCLUDE
    base = step[keep].mean()
    if base <= 0 or not keep.any():
        return float("nan")
    at = np.mean([step[np.argmin(np.abs(xs - x))] for x in interior])
    return float(at / base)


def load(variant, sid):
    p = OUT / variant / "images" / f"{sid}.png"
    if not p.exists():
        return None
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def build_collage(s, w, edges):
    cat = s["metadata"]["category"]
    pool = [x for x in by_cat[cat] if x != s["sample_id"]]
    canvas = None
    for i in range(len(edges) - 1):
        donor = load(COLLAGE_FROM, rng.choice(pool))
        if donor is None or donor.shape[1] != w:
            return None
        if canvas is None:
            canvas = np.zeros_like(donor)
        canvas[:, edges[i]:edges[i + 1]] = donor[:, edges[i]:edges[i + 1]]
    return canvas


def stats(v):
    n = len(v)
    mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return mu, (sd / math.sqrt(n) if n else float("nan")), n


def paired_p(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("nan")
    if not se:
        return mu, 1.0
    t = mu / se
    return mu, 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


vals = {label: {} for label, _ in VARIANTS}
for s in samples:
    sid = s["sample_id"]
    ref = load(COLLAGE_FROM, sid)
    if ref is None:
        continue
    w = ref.shape[1]
    edges = bounds(s["split_ratio"], w)
    for label, var in VARIANTS:
        a = build_collage(s, w, edges) if var == "COLLAGE" else load(var, sid)
        if a is None or a.shape[1] != w:
            continue
        v = lfd(a, edges)
        if not math.isnan(v):
            vals[label][sid] = v

print(f"\nBoundary coherence, LFD  (1.0 = boundary looks like any other "
      f"column)")
print(f"{'variant':<32}{'LFD':>9}{'+/-':>8}{'n':>5}")
rows = {}
for label, _ in VARIANTS:
    if not vals[label]:
        print(f"{label:<32}{'--':>9}{'--':>8}{0:>5}")
        continue
    mu, se, n = stats(list(vals[label].values()))
    rows[label] = dict(lfd=mu, se=se, n=n)
    print(f"{label:<32}{mu:>9.3f}{se:>8.3f}{n:>5}")

FLOOR = "collage (different samples)"
CEIL = "Ours, tau=0 (one global pass)"
ok = False
if FLOOR in rows and CEIL in rows:
    ids = sorted(set(vals[FLOOR]) & set(vals[CEIL]))
    mu, p = paired_p([vals[FLOOR][i] for i in ids], [vals[CEIL][i] for i in ids])
    ok = p < 0.001 and mu > 0
    print(f"\nendpoint check   collage - global = {mu:+.3f}, p={p:.2e}, "
          f"n={len(ids)}   {'SEPARATED' if ok else 'NOT SEPARATED'}")

if ok:
    print()
    for a, b in [("Ours, tau=1", "Ours, tau=0.7"),
                 ("Ours, tau=1", "Ours, tau=0.5"),
                 ("Ours, tau=1", "Ours, tau=0.3"),
                 ("Ours, tau=0.5", "Ours, tau=0 (one global pass)"),
                 ("collage (different samples)", "Ours, tau=1")]:
        ids = sorted(set(vals[a]) & set(vals[b]))
        mu, p = paired_p([vals[a][i] for i in ids], [vals[b][i] for i in ids])
        tag = "***" if p < 0.001 else "**" if p < 0.01 else \
              "*" if p < 0.05 else "n.s."
        print(f"{a:>28} - {b:<32} {mu:+.3f}  p={p:.4f}  {tag}")
else:
    print("\nendpoints do not separate: the rows above are not interpretable "
          "and must not be reported as a coherence result.")

out = ROOT / "outputs/coherence.json"
out.write_text(json.dumps(rows, indent=1))
print(f"\nWritten to {out}")
