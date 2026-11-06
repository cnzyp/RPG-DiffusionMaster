"""
Paired significance tests for every comparison the paper makes.

The tables report means over 150 samples with no dispersion, so a reader
cannot tell a real effect from noise - and at least one claim in the text did
not survive testing.  Every variant is scored on the same samples with the
same seed, so the comparisons are paired, which is both the right test and a
much tighter one than treating the runs as independent.

Reports, per comparison: the paired mean difference, a 95% CI on it, the
t statistic, a two-sided p value, the Wilcoxon signed-rank p as a
distribution-free check, and how many of the 150 samples each side wins.

Usage:
    python compute_significance.py
    python compute_significance.py --latex
"""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "outputs/mixed_eval_v1"

ap = argparse.ArgumentParser()
ap.add_argument("--latex", action="store_true", help="emit LaTeX table rows")
args = ap.parse_args()

METRICS = [("identity_mean", "DINO-ID", True),
           ("leakage_margin", "Margin", True),
           ("clip_full", "CLIP-T", True),
           ("leakage_mean", "Leakage", False)]

# (variant, label) compared against `ours_tau050` unless stated otherwise
BASELINES = [
    ("rpg_only", "RPG-only"),
    ("ip_global_real", "IP-Adapter (global)"),
    ("ip_faceid_real_s06", "IP-FaceID"),
    ("conceptrol_official_s08", "Conceptrol"),
    ("msdiffusion_s08_inset", "MS-Diffusion"),
    ("ours_gating_fair", "Attention gating (buggy)"),
    ("ours_gating_fixed", "Attention gating"),
    ("ours_fullcanvas", "Noise fusion, full-canvas"),
    ("ours_methodA_v2", "Ours, tau=1"),
]
OURS = "ours_tau050"

# the paper's core claim: attention gating is indistinguishable from text-only
EXTRA = [("ours_gating_fixed", "rpg_only", "Attention gating vs text-only"),
         ("ours_gating_fixed", "ours_gating_fair", "fixed vs buggy gating")]


def load(variant):
    p = R / variant / "metrics_per_sample.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    rows = rows if isinstance(rows, list) else rows.get("samples", rows)
    return {r["sample_id"]: r for r in rows}


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def wilcoxon_p(diffs):
    """Signed-rank test, normal approximation with tie correction."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 10:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if d[i] > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mu) / sigma
    return 2 * (1 - norm_cdf(abs(z)))


def compare(A, B, key):
    ids = sorted(set(A) & set(B))
    d = [A[i][key] - B[i][key] for i in ids]
    n = len(d)
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = mu / se if se else float("inf")
    return {"n": n, "diff": mu, "ci": (mu - 1.96 * se, mu + 1.96 * se),
            "t": t, "p": 2 * (1 - norm_cdf(abs(t))),
            "wilcoxon_p": wilcoxon_p(d),
            "wins": sum(1 for x in d if x > 0)}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


ours = load(OURS)
if ours is None:
    raise SystemExit(f"missing {OURS}")

out = {}
for key, label, higher_better in METRICS:
    print(f"\n{'='*94}\n{label}   (positive difference favours the "
          f"{'first' if higher_better else 'second'} method)\n{'='*94}")
    print(f"{'comparison':<34}{'diff':>9}{'95% CI':>20}{'t':>7}"
          f"{'p':>10}{'wilcox':>9}{'wins':>9}")
    for var, name in BASELINES:
        B = load(var)
        if B is None:
            print(f"{name:<34}  (no per-sample data)")
            continue
        r = compare(ours, B, key)
        out.setdefault(label, {})[name] = r
        ci = f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]"
        print(f"{'Ours vs ' + name:<34}{r['diff']:>+9.4f}{ci:>20}{r['t']:>7.2f}"
              f"{r['p']:>10.4f}{r['wilcoxon_p']:>9.4f}"
              f"{str(r['wins']) + '/' + str(r['n']):>9}  {stars(r['p'])}")
    for a, b, name in EXTRA:
        A, B = load(a), load(b)
        if A is None or B is None:
            continue
        r = compare(A, B, key)
        out.setdefault(label, {})[name] = r
        ci = f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]"
        print(f"{name:<34}{r['diff']:>+9.4f}{ci:>20}{r['t']:>7.2f}"
              f"{r['p']:>10.4f}{r['wilcoxon_p']:>9.4f}"
              f"{str(r['wins']) + '/' + str(r['n']):>9}  {stars(r['p'])}")

(R / "significance.json").write_text(json.dumps(out, indent=1, default=str))
print(f"\nWritten to {R/'significance.json'}")

if args.latex:
    print("\n% paired vs Ours, 150 samples; *** p<0.001, ** p<0.01, * p<0.05")
    for var, name in BASELINES:
        B = load(var)
        if B is None:
            continue
        cells = []
        for key, label, _ in METRICS[:3]:
            r = compare(ours, B, key)
            cells.append(f"{r['diff']:+.3f}$^{{{stars(r['p'])}}}$")
        print(f"{name} & " + " & ".join(cells) + r" \\")
