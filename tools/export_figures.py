"""Export the manuscript's figures as separate numbered files, plus captions.

Pattern Recognition asks for figures "provided within the text and as separate
files, numbered and cited; captions provided", and separately that captions be
supplied "separately, not attached to the figure".  The manuscript satisfies
the first half by construction; this script produces the second.

Output lands in paper_latex/submission/figures/:

    Fig1.png ... FigN.pdf     one file per figure, in citation order
    captions.txt              the caption text, numbered to match

Two of the figures are drawn in TikZ and have no source file of their own, so
they are compiled standalone at the size they appear in the manuscript.  The
rest are copied from paper_figures/ unchanged -- re-encoding a PNG would only
lose something.

    python tools/export_figures.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / 'paper_latex'
OUT = TEX / 'submission' / 'figures'

STANDALONE = r"""\documentclass[preview,border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{tikz,amsmath,amssymb,bm,graphicx}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,shapes.geometric,%
decorations.pathreplacing}
\input{fig_tikz_defs}
\begin{document}
\input{<<FIG>>}
\end{document}
"""
# %-formatting is not usable here: the template is LaTeX and LaTeX is full of %.


def balanced(text, start):
    """The contents of the {...} group beginning at text[start] == '{'."""
    depth, i = 0, start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValueError('unbalanced group')


# Greek and the handful of operators the captions use.  Without these the
# plain-text caption loses every symbol it depends on -- "at one switch point
# ." instead of "at one switch point tau" -- which is worse than useless to
# whoever reads these separately from the figures.
SYMBOLS = {
    r'\tau': 'τ', r'\Omega': 'Ω', r'\omega': 'ω', r'\lambda': 'λ',
    r'\varepsilon': 'ε', r'\epsilon': 'ε', r'\theta': 'θ', r'\rho': 'ρ',
    r'\Delta': 'Δ', r'\delta': 'δ', r'\sigma': 'σ', r'\mu': 'μ',
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\cap': '∩',
    r'\times': '×', r'\le': '≤', r'\ge': '≥', r'\to': '→',
    r'\pm': '±', r'\approx': '≈', r'\ldots': '...', r'\dots': '...',
    r'\%': '%', r'\&': '&', r'\_': '_',
}


def caption_of(block):
    i = block.find('\\caption')
    if i < 0:
        return ''
    body, _ = balanced(block, block.index('{', i))
    # \caption{% opens with a comment to the end of the line.  An escaped \%
    # is a per cent sign, not a comment, and eating from there to the end of
    # the line silently truncates the sentence it sits in.
    body = re.sub(r'(?m)(?<!\\)%.*$', '', body)
    body = re.sub(r'\\label\{[^}]*\}', ' ', body)
    body = re.sub(r'\\ref\{[^}]*\}', '<ref>', body)
    # \tfrac12 and \frac{1}{2} both have to survive as 1/2; stripping the
    # command alone would leave "12"
    body = re.sub(r'\\[dt]?frac\s*(?:\{([^{}]*)\}|(\d))\s*(?:\{([^{}]*)\}|(\d))',
                  lambda m: f'{m.group(1) or m.group(2)}/{m.group(3) or m.group(4)}',
                  body)
    for tex, ch in SYMBOLS.items():
        body = body.replace(tex + ' ', ch).replace(tex, ch)
    # font switches: unwrap, repeatedly, so nested ones come out too
    for _ in range(4):
        body = re.sub(r'\\(textbf|emph|textit|texttt|mathbf|mathrm|text)'
                      r'\{([^{}]*)\}', r'\2', body)
    body = re.sub(r'\\[,;:!> ]', ' ', body)          # spacing commands
    body = re.sub(r'\\[a-zA-Z]+\*?', ' ', body)      # anything left
    body = body.replace('{', '').replace('}', '').replace('$', '')
    body = body.replace('~', ' ').replace('---', ' -- ')
    return ' '.join(body.split())


def main():
    src = (TEX / 'main_pr.tex').read_bytes().decode('utf-8').replace('\r\n', '\n')
    OUT.mkdir(parents=True, exist_ok=True)

    captions, n, made = [], 0, []
    for m in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}', src, re.S):
        block = m.group(0)
        n += 1
        imgs = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}', block)
        inputs = [f for f in re.findall(r'\\input\{([^}]*)\}', block)
                  if 'tikz' in f]
        captions.append(f'Figure {n}. {caption_of(block)}')

        if inputs:                                   # drawn, not imported
            stem = f'_sa_fig{n}'
            (TEX / f'{stem}.tex').write_text(
                STANDALONE.replace('<<FIG>>', inputs[0]), encoding='utf-8')
            r = subprocess.run(['pdflatex', '-interaction=nonstopmode',
                                f'{stem}.tex'], cwd=TEX,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            pdf = TEX / f'{stem}.pdf'
            if not pdf.exists():
                print(f'  Fig{n}: FAILED to compile {inputs[0]}')
                continue
            shutil.copyfile(pdf, OUT / f'Fig{n}.pdf')
            made.append(f'Fig{n}.pdf   <- {inputs[0]} (TikZ, compiled)')
            for ext in ('.tex', '.pdf', '.aux', '.log'):
                (TEX / f'{stem}{ext}').unlink(missing_ok=True)
        elif imgs:
            s = TEX / imgs[0]
            if not s.exists():
                print(f'  Fig{n}: missing {imgs[0]}')
                continue
            dst = OUT / f'Fig{n}{s.suffix}'
            shutil.copyfile(s, dst)
            made.append(f'{dst.name}   <- {imgs[0]}')
        else:
            print(f'  Fig{n}: no image and no TikZ input')

    (OUT / 'captions.txt').write_text(
        'Figure captions\n'
        'The Injection Site Determines What Identity Survives in '
        'Training-Free Multi-Concept Personalization\n\n'
        '<ref> marks a cross-reference to a section or table of the '
        'manuscript.\n\n' + '\n\n'.join(captions) + '\n',
        encoding='utf-8')

    for line in made:
        print('  ' + line)
    print(f'\n{len(made)} of {n} figures written to {OUT}')
    print(f'  captions.txt   {len(captions)} captions')
    return 0 if len(made) == n else 1


sys.exit(main())
