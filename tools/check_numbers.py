"""Check the generated manuscript's numbers against each other.

Run after tools/make_pr_tex.py, on what a reviewer will actually see:

    python tools/check_numbers.py

This exists because the paper shipped a wrong-signed claim for weeks: a
paragraph quoted 768-resolution numbers while pointing at a table of 1024
ones, and nothing caught it.  The four checks below are each aimed at a class
of error that has actually occurred here, not at hypothetical ones.

  1. A stated difference must equal the two numbers it is drawn from.
  2. A confidence interval must contain its own point estimate.
  3. Every decimal in the abstract or the contribution list must appear
     somewhere in the body or the supplement.
  4. A number the prose attributes to a named metric must be one of the values
     printed for that metric in some table, or a difference of two of them.

Check 4 flags legitimate numbers too: the paper reports sub-experiments
(DreamBench-2 subsets, the 30-sample repairs, the 24-sample face-area study)
whose values appear only in prose.  Those are listed in PROSE_ONLY with the
place they come from, so a genuinely new flag stands out.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / 'paper_latex'
TOL = 0.0011

# Values the abstract or contributions state that are derived, not quoted.
DERIVED = {
    '0.83': 'cost ratio, 9.88s / 11.92s per image (Table: timing)',
}

# Numbers quoted beside a metric name that are measured in prose only.
PROSE_ONLY = {
    ('DINO-ID', 0.789): 'tau-variant probe with the adapter off (Limitations)',
    ('DINO-ID', 0.609): 'DreamBench-2 animal-object pairs, text alone',
    ('DINO-ID', 0.611): 'DreamBench-2 animal-object pairs, combined sites',
    ('Margin', -0.106): 'inset-layout repair, 30 samples (Limitations)',
    ('Margin', -0.237): 'background-pass repair, 30 DreamBench-2 samples',
    ('ArcFace', 0.396): 'per-face full-resolution pass, 17 detectable faces',
    ('DINO-ID', 0.412): 'reads as DINO-ID but is the Margin in the same clause',
    ('DINO-ID', 1.9): 'per cent, not a DINO-ID',
    ('ArcFace', 1.7): 'per cent, not an ArcFace',
    ('DINO-ID', 0.15): 'a threshold, not a measurement',
    ('DINO-ID', 0.14): 'a p-value',
    ('ArcFace', 0.51): 'a p-value',
    ('DINO-ID', 0.69): 'inset repair scored on its own boxes (Limitations)',
    ('DINO-ID', 0.083): 'SDXL crop conditioning per sub-canvas (Limitations)',
    ('DINO-ID', 0.033): 'background-discard variant (supplement)',
}

METRICS = ('DINO-ID', 'Margin', 'CLIP-T', 'ArcFace-P', 'ArcFace')

problems = []
checked = 0


def load(name):
    return HERE.joinpath(name).read_bytes().decode('utf-8').replace('\r\n', '\n')


def flat(s):
    return ' '.join(s.split())


def to_f(s):
    s = s.replace('$', '').replace('{', '').replace('}', '').replace(' ', '')
    s = s.replace('\\,', '').replace('~', '').replace('\u2212', '-')
    s = s.replace('\\textbf', '').replace('\\mathbf', '')
    if s[:2] in ('+-', '-+'):
        s = s[1:]
    return float(s)


# The prose writes a number as $0.720$, $+0.072$, $-$0.082 or bare, so the
# pattern has to consume the closing $ as well as the opening one -- without
# that it matched "$0.720" and then choked on the "$" before "against", and
# the whole arithmetic check silently matched nothing.
N = r'\$?[-+]?\$?\$?\s*[0-9]+(?:\.[0-9]+)?\$?'
SIGNED = r'\$?[-+]\$?\$?\s*[0-9]+\.[0-9]+\$?'


# --- 1. "A against B ... +C" ------------------------------------------------
AGAINST = re.compile(
    r'(' + N + r')\s*\\?\s*against\s+(?:our\s+|the\s+)?(' + N + r')'
    r'(?P<mid>.{0,90}?)(?:difference of\s+|is\s+|,\s*)(?P<diff>' + SIGNED + r')',
    re.S)


def check_against(text, where):
    """The paper always states a difference as ours minus the baseline, but
    either side may be ours: "0.720 against 0.648 ... +0.072" puts ours first,
    "0.651 against our 0.647, -0.004" puts ours second.  So the magnitude is
    always checked and the sign only where the text says which side is ours."""
    global checked
    for m in AGAINST.finditer(text):
        mid = m.group('mid')
        # an intervening math expression means the trailing number belongs to
        # some other clause, not to this comparison
        if 'against' in mid or '$' in mid:
            continue
        try:
            a, b, d = to_f(m.group(1)), to_f(m.group(2)), to_f(m.group('diff'))
        except ValueError:
            continue
        checked += 1
        if abs(abs(a - b) - abs(d)) > 0.0015:
            problems.append(
                f'[{where}] stated difference does not follow:\n'
                f'    {flat(m.group(0))}\n'
                f'    the two differ by {abs(a - b):.3f}, '
                f'but the text says {abs(d):.3f}')
            continue
        ours_second = re.search(r'against\s+our\b', m.group(0))
        expect = (b - a) if ours_second else (a - b)
        if expect * d < 0:
            problems.append(
                f'[{where}] difference has the wrong sign:\n'
                f'    {flat(m.group(0))}\n'
                f'    ours minus baseline is {expect:+.3f}, '
                f'text says {d:+.3f}')


# --- 2. "+C, 95% CI [lo, hi]" ----------------------------------------------
CI = re.compile(
    r'(?P<pt>' + SIGNED + r')\s*\$?,?\s*(?:95\\%\s*CI\s*|CI\s*)'
    r'\$?\[\s*(?P<lo>' + N + r')\s*,\s*(?P<hi>' + N + r')\s*\]', re.S)


def check_ci(text, where):
    global checked
    for m in CI.finditer(text):
        try:
            pt, lo, hi = (to_f(m.group('pt')), to_f(m.group('lo')),
                          to_f(m.group('hi')))
        except ValueError:
            continue
        checked += 1
        if lo > hi:
            problems.append(f'[{where}] CI bounds reversed: {flat(m.group(0))}')
        elif not lo - 1e-9 <= pt <= hi + 1e-9:
            problems.append(
                f'[{where}] CI does not contain its estimate:\n'
                f'    {flat(m.group(0))}\n'
                f'    {pt:+.3f} is outside [{lo:+.3f}, {hi:+.3f}]')


# --- 3. front-matter numbers must occur in the body ------------------------
def check_front(main, supp):
    global checked
    abstract = main[main.index('\\begin{abstract}'):main.index('\\end{abstract}')]
    i = main.index('\\begin{enumerate}')
    contrib = main[i:main.index('\\end{enumerate}', i)]
    body = main[main.index('\\section{Related Work}'):]
    known = set()
    for t in (body, supp):
        for tok in re.findall(r'[0-9]+(?:\.[0-9]+)?', t):
            known.add(tok)
            known.add(tok[1:] if tok.startswith('0.') else '0' + tok
                      if tok.startswith('.') else tok)
    for where, blk in (('abstract', abstract), ('contributions', contrib)):
        for tok in sorted(set(re.findall(r'[0-9]+\.[0-9]+', blk))):
            checked += 1
            if tok in known or tok in DERIVED:
                continue
            problems.append(
                f'[{where}] {tok} appears here but nowhere in the body')


# --- 4. a metric-attributed number must be in that metric's column ---------
def strip_tex(s):
    s = re.sub(r'\\textbf|\\mathbf|\\emph|\\text(super|sub)script', '', s)
    return re.sub(r'\$|\\,|\\!|~|\{|\}', '', s).strip()


def metric_columns(text, cols):
    for m in re.finditer(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', text, re.S):
        tb = re.search(r'\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}',
                       m.group(0), re.S)
        if not tb:
            continue
        # the rules are not rows, and \toprule shares the header's line
        body = re.sub(r'\\(top|mid|bottom|c?mid)rule(\([^)]*\))?(\{[^}]*\})?',
                      ' ', tb.group(1))
        rows = [r for r in re.split(r'\\\\', body) if r.strip()]
        if not rows:
            continue
        idx = {}
        for j, h in enumerate(strip_tex(c) for c in rows[0].split('&')):
            h = re.sub(r'\\uparrow|\\downarrow|\\times 10\^3|\s', '', h)
            if h in METRICS and h not in idx.values():
                idx[j] = h
        for r in rows[1:]:
            cs = [strip_tex(c) for c in r.split('&')]
            for j, name in idx.items():
                if j < len(cs):
                    cols[name].update(
                        float(v) for v in re.findall(r'[-+]?[0-9]+\.[0-9]+', cs[j]))
    return cols


def check_metrics(main, supp):
    global checked
    cols = metric_columns(main, {k: set() for k in METRICS})
    cols = metric_columns(supp, cols)
    diffs = {k: {round(abs(a - b), 6) for a in v for b in v}
             for k, v in cols.items()}
    for where, text in (('main', main), ('supp', supp)):
        prose = re.sub(r'\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}', ' ',
                       text, flags=re.S)
        for name in METRICS:
            nm = re.escape(name) + (r'(?!-P)' if name == 'ArcFace' else '')
            for pat in (nm + r'[^.;()]{0,18}?(?P<v>' + SIGNED.replace('[-+]', '[-+]?') + r')',
                        r'(?P<v>' + SIGNED.replace('[-+]', '[-+]?') + r')\s*' + nm):
                for m in re.finditer(pat, prose):
                    try:
                        x = to_f(m.group('v'))
                    except ValueError:
                        continue
                    checked += 1
                    if (name, x) in PROSE_ONLY or (name, abs(x)) in PROSE_ONLY:
                        continue
                    if any(abs(x - y) <= TOL for y in cols[name]):
                        continue
                    if any(abs(abs(x) - d) <= TOL for d in diffs[name]):
                        continue
                    problems.append(
                        f'[{where}] {name} = {x} is not a printed {name} value, '
                        f'nor a difference of two, nor listed in PROSE_ONLY\n'
                        f'    ...{flat(m.group(0))}...')


def main():
    m, s = load('main_pr.tex'), load('supp_pr.tex')
    for where, text in (('main', m), ('supp', s)):
        check_against(text, where)
        check_ci(text, where)
    check_front(m, s)
    check_metrics(m, s)

    print(f'{checked} numeric claims checked')
    if problems:
        seen, uniq = set(), []
        for p in problems:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        print(f'\n{len(uniq)} to look at:\n')
        for p in uniq:
            print(p + '\n')
        return 1
    print('no inconsistency found')
    return 0


sys.exit(main())
