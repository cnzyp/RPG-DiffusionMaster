"""Assemble the LaTeX source archive Editorial Manager needs.

Pattern Recognition's own section says "Upload editable source files
(.doc/.docx or .tex). PDF is not an acceptable source file", so the PDF alone
is not a submission -- the sources have to go up with it.

What goes in is discovered by following \\input, \\includegraphics and
\\bibliography out of the manuscript rather than listed by hand, so a figure
added later cannot be left out of the archive.  elsarticle.cls is not
included: it ships with every TeX distribution and Editorial Manager has it.

The .bbl is included deliberately.  Editorial Manager runs LaTeX but not
always BibTeX, and a missing .bbl turns every citation into [?].

    python tools/make_source_zip.py            # blind manuscript
    python tools/make_source_zip.py --named    # named manuscript
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / 'paper_latex'
OUT = TEX / 'submission'

ap = argparse.ArgumentParser()
ap.add_argument('--named', action='store_true',
                help='package main_pr_named.tex instead of the blind build')
args = ap.parse_args()

MAIN = 'main_pr_named' if args.named else 'main_pr'


def read(p):
    return p.read_bytes().decode('utf-8').replace('\r\n', '\n')


def collect(stem, seen):
    """Every file `stem`.tex needs, following \\input recursively."""
    src = TEX / f'{stem}.tex'
    if not src.exists() or src in seen:
        return
    seen.add(src)
    t = read(src)
    for name in re.findall(r'\\input\{([^}]*)\}', t):
        p = TEX / (name if name.endswith('.tex') else name + '.tex')
        if p.exists():
            collect(p.stem, seen)
        else:
            print(f'  WARNING: \\input{{{name}}} not found')
    for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', t):
        p = TEX / name
        if p.exists():
            seen.add(p)
        elif (TEX / (name + '.png')).exists():
            seen.add(TEX / (name + '.png'))
        else:
            print(f'  WARNING: image {name} not found')
    for base in re.findall(r'\\bibliography\{([^}]*)\}', t):
        for b in base.split(','):
            p = TEX / (b.strip() + '.bib')
            if p.exists():
                seen.add(p)

def main():
    files = set()
    collect(MAIN, files)
    # only the top-level document has a bibliography; the \input'd figure
    # files do not, and warning about those would bury a real miss
    bbl = TEX / f'{MAIN}.bbl'
    if bbl.exists():
        files.add(bbl)
    else:
        print(f'  WARNING: {bbl.name} missing -- citations will come out as [?]')
    missing_pdf = not (TEX / f'{MAIN}.pdf').exists()
    if missing_pdf:
        print(f'  WARNING: {MAIN}.pdf has not been built')

    # Editorial Manager refuses a LaTeX archive with subfolders: "The system
    # cannot process a zipfile containing one or more subfolders."  So the
    # archive is flat, and the .tex inside it has its paper_figures/ prefixes
    # stripped to match.  The working copy in the repo keeps its directories.
    flat = {}
    for f in sorted(files):
        if f.name in flat:
            raise SystemExit(
                f'flattening collides: {f} and {flat[f.name]} share a name')
        flat[f.name] = f

    def deprefix(text):
        return re.sub(r'(\\includegraphics(?:\[[^\]]*\])?\{)[^}]*/([^}]*\})',
                      r'\1\2', text)

    zpath = OUT / f'{MAIN}_source.zip'
    OUT.mkdir(parents=True, exist_ok=True)
    rewritten = 0
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, f in sorted(flat.items()):
            if f.suffix == '.tex':
                before = read(f)
                after = deprefix(before)
                rewritten += before != after
                z.writestr(name, after)
            else:
                z.write(f, name)
    if rewritten:
        print(f'  flattened: image paths rewritten in {rewritten} .tex file(s)')

    total = sum(f.stat().st_size for f in files)
    print(f'{zpath}')
    print(f'  {len(files)} files, {total / 1e6:.1f} MB raw, '
          f'{zpath.stat().st_size / 1e6:.1f} MB zipped\n')
    for f in sorted(files):
        print(f'    {f.relative_to(TEX).as_posix():44s} '
              f'{f.stat().st_size / 1024:8.0f} KB')
    return 0


sys.exit(main())
