"""Build the editable submission documents Editorial Manager asks for.

Pattern Recognition requires the highlights "in a separate editable file ...
Please use 'Highlights' in the file name", and a cover letter.  Editorial
Manager takes .docx for both, so they are written as .docx here rather than
as the .txt drafts that were easier to edit.

The highlights' own rule -- 3 to 5 bullets, at most 85 characters including
spaces -- is checked rather than trusted, because the limit is enforced on
upload and a 90-character bullet comes back as a rejected submission item.

    python tools/build_submission_docs.py
"""
import datetime
import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'paper_latex' / 'submission'

TITLE = ('The Injection Site Determines What Identity Survives in '
         'Training-Free Multi-Concept Personalization')

BULLETS = [
    'Where a reference is injected, not how strongly, decides what identity survives',
    'Attention-level masking recovers 1.7% of the identity the sampling loop delivers',
    'A reference-swap control tells a weak adapter from a dead one; gating is dead',
    'Sub-canvas fusion beats a corpus-trained baseline on identity, at 0.83x its cost',
    'Identity gain tracks face area at r=0.949, so framing must be reported alongside',
]


def check_bullets():
    assert 3 <= len(BULLETS) <= 5, f'{len(BULLETS)} bullets, the rule is 3 to 5'
    bad = [(len(b), b) for b in BULLETS if len(b) > 85]
    if bad:
        for n, b in bad:
            print(f'  TOO LONG ({n} chars): {b}')
        raise SystemExit('highlights exceed 85 characters')
    for b in BULLETS:
        print(f'  {len(b):3d}/85  {b}')


def body_font(doc, name='Times New Roman', size=11):
    st = doc.styles['Normal']
    st.font.name = name
    st.font.size = Pt(size)


def highlights():
    doc = Document()
    body_font(doc)
    doc.add_paragraph(TITLE).runs[0].bold = True
    doc.add_paragraph('Highlights').runs[0].bold = True
    for b in BULLETS:
        doc.add_paragraph(b, style='List Bullet')
    p = OUT / 'Highlights.docx'
    doc.save(p)
    return p


def cover_letter(date_str):
    src = (OUT / 'cover_letter.txt').read_text(encoding='utf-8')
    # The first line is the date.  It starts out as a placeholder and is
    # replaced by today's; re-running must restamp it rather than refuse,
    # since the letter is rebuilt every time its body changes.
    lines = src.replace('\r\n', '\n').split('\n')
    first = lines[0].strip()
    assert ('[update to the day you actually submit]' in first
            or re.fullmatch(r'\d{1,2} [A-Z][a-z]+ \d{4}', first)), \
        f'the cover letter should start with a date line, found: {first!r}'
    lines[0] = date_str
    text = '\n'.join(lines)

    doc = Document()
    body_font(doc)
    for para in text.split('\n\n'):
        para = para.rstrip()
        if not para:
            continue
        p = doc.add_paragraph(' '.join(para.split()))
        # the letter's two numbered lists are indented in the source; keep
        # them indented here, or they read as ordinary prose that happens to
        # start with a digit
        if re.match(r'\s+\d+\.\s', para):
            p.paragraph_format.left_indent = Pt(24)
            p.paragraph_format.first_line_indent = Pt(-12)
    p = OUT / 'Cover_letter.docx'
    doc.save(p)
    (OUT / 'cover_letter.txt').write_text(text, encoding='utf-8')
    return p


AUTHORS = ['Yaping Zhang', 'Mengqin Sun', 'Kaicheng Shi', 'Shaoying Yang']


def declaration_of_interests(date_str):
    """Editorial Manager lists this as its own required item type.

    The statement already appears in the manuscript and on the title page,
    but EM wants a file of its own, and a submission missing a required item
    is sent back.  The wording and the two boxes are Elsevier's template.
    """
    doc = Document()
    body_font(doc)
    doc.add_paragraph('Declaration of interests').runs[0].bold = True
    doc.add_paragraph(TITLE).runs[0].italic = True

    doc.add_paragraph(
        '☒ The authors declare that they have no known competing '
        'financial interests or personal relationships that could have '
        'appeared to influence the work reported in this paper.')
    doc.add_paragraph(
        '☐ The authors declare the following financial '
        'interests/personal relationships which may be considered as '
        'potential competing interests:')
    doc.add_paragraph()
    doc.add_paragraph('Authors: ' + ', '.join(AUTHORS))
    doc.add_paragraph('Date: ' + date_str)
    p = OUT / 'Declaration_of_interests.docx'
    doc.save(p)
    return p


def main():
    today = datetime.date.today()
    date_str = f'{today.day} {today:%B %Y}'
    print('Highlights:')
    check_bullets()
    h = highlights()
    c = cover_letter(date_str)
    d = declaration_of_interests(date_str)
    print(f'\nwritten {h}')
    print(f'written {c}   (dated {date_str})')
    print(f'written {d}')
    return 0


sys.exit(main())
