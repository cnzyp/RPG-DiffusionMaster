"""
Compute FID and KID for every method on Mixed-Eval-v1, from one file list.

Why KID: FID compares 150 generated images against 750 reference photographs.
At that sample size FID's Gaussian estimator carries a large, sample-size
dependent bias, which is why the paper treats it as a sanity check rather
than a criterion.  KID uses an unbiased MMD estimator and is the standard
choice in this regime, so it belongs beside the FID column.

Why this script supersedes compute_fid.py: clean-fid globs once per entry of
its EXTENSIONS set, and that set holds both "png" and "PNG".  NTFS is
case-insensitive, so on Windows every image is matched twice and both sets
are silently doubled (the run log reads "Found 300 images" for a 150-image
folder).  FID is nearly invariant to exact duplication - the feature mean and
covariance are unchanged and only the 1/(N-1) normalisation shifts, measured
at 0.17% here - but KID is not: the unbiased MMD estimator then finds m
identical pairs at the maximal kernel value inside each within-set sum, which
inflates the score.  Both metrics are recomputed here from a de-duplicated
glob.

Usage:
    conda run -n rpg-faceid python compute_fid_kid.py
    conda run -n rpg-faceid python compute_fid_kid.py --fresh   # ignore cache
"""

import json
import sys
from pathlib import Path

import torch
from cleanfid import fid as cleanfid
from cleanfid import utils as cleanfid_utils

# `fid.py` does `from cleanfid.utils import *`, so the name must be rebound in
# both namespaces for the restriction to take effect.
cleanfid.EXTENSIONS = {"png"}
cleanfid_utils.EXTENSIONS = {"png"}

ROOT = Path(__file__).parent
OUT_BASE = ROOT / "outputs/mixed_eval_v1"
REAL_TMP = ROOT / "outputs/_fid_real_tmp"

if not REAL_TMP.exists() or not any(REAL_TMP.glob("*.png")):
    raise SystemExit(f"{REAL_TMP} is empty - run compute_fid.py first")

n_real = len(list(REAL_TMP.glob("*.png")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Real images: {n_real}   device: {device}\n")

METHODS = [
    # Table 1
    "rpg_only", "ip_global_real", "ip_faceid_real_s06", "ip_faceid_real_s08",
    "instantid", "conceptrol_official_s08", "conceptrol_official_s1",
    "msdiffusion_s06", "msdiffusion_s08", "msdiffusion_s08_inset",
    "ours_tau050",
    # tau sweep (Table 5)
    "ours_tau000", "ours_tau030", "ours_tau070", "ours_methodA_v2",
    # ablations
    "ours_gating_fair", "ours_fullcanvas",
    # native-resolution protocol
    "ours_tau070_1024", "msdiff_1024", "conceptrol_1024", "ipglobal_1024",
    "gating_1024", "rpgonly_1024", "ipfaceid_1024",
]

RESULTS = OUT_BASE / "fid_kid_dedup.json"
results = {}
if RESULTS.exists() and "--fresh" not in sys.argv:
    try:
        results = json.loads(RESULTS.read_text())
        print(f"Loaded {len(results)} cached entries\n")
    except Exception:
        results = {}

for method in METHODS:
    gen_dir = OUT_BASE / method / "images"
    if not gen_dir.exists():
        print(f"[{method}] SKIP - no images dir")
        continue
    if results.get(method):
        print(f"[{method}] cached  FID {results[method]['fid']}  "
              f"KID {results[method]['kid']}")
        continue

    n_gen = len(list(gen_dir.glob("*.png")))
    print(f"[{method}] {n_gen} generated vs {n_real} real ...", flush=True)
    entry = {}
    for name, fn in (("fid", cleanfid.compute_fid), ("kid", cleanfid.compute_kid)):
        try:
            score = fn(str(REAL_TMP), str(gen_dir), mode="clean",
                       num_workers=0, batch_size=16, device=device)
            entry[name] = (round(float(score) * 1000, 3) if name == "kid"
                           else round(float(score), 2))
        except Exception as e:
            print(f"  {name.upper()} FAILED: {e}")
            entry[name] = None
    results[method] = entry
    print(f"  FID {entry['fid']}   KID x1000 {entry['kid']}")
    RESULTS.write_text(json.dumps(results, indent=1))

print("\n--- de-duplicated, lower is better ---")
print(f"  {'method':<26} {'FID':>8} {'KIDx1000':>10}")
for m in METHODS:
    e = results.get(m)
    if e:
        print(f"  {m:<26} {e['fid']:>8} {e['kid']:>10}")
print(f"\nWritten to {RESULTS}")
