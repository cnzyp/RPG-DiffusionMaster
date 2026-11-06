"""Pull the speaker notes out of the deck into a standalone script.

One source: the notes live in the .pptx, so practising from the script and
presenting from the deck cannot drift.  Timing is estimated at 240 Chinese
characters per minute, which is an unhurried speaking pace.
"""
from pathlib import Path

from pptx import Presentation

DECK = Path('report/论文汇报.pptx')
OUT = Path('report/汇报稿.md')
CPM = 240

prs = Presentation(DECK)
titles, notes = [], []
for sl in prs.slides:
    t = ''
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            t = sh.text_frame.text.strip().split('\n')[0]
            break
    titles.append(t)
    notes.append(sl.notes_slide.notes_text_frame.text.strip()
                 if sl.has_notes_slide else '')

total = sum(len(n) for n in notes)
lines = [
    '# 论文汇报稿',
    '',
    '《注入位点决定身份能否保留》—— 投稿 Pattern Recognition',
    '',
    f'**共 {len(titles)} 页，正文约 {total} 字，按 {CPM} 字/分钟估算约 '
    f'{total / CPM:.0f} 分钟**（不含提问）。',
    '',
    '> 讲稿与 PPT 的备注页是同一份内容，改了 PPT 重新生成即可，两边不会对不上。',
    '',
    '---',
    '',
]

for i, (t, n) in enumerate(zip(titles, notes), 1):
    mins = len(n) / CPM
    lines.append(f'## 第 {i} 页 · {t}')
    lines.append('')
    lines.append(f'*约 {mins:.1f} 分钟*')
    lines.append('')
    for para in n.split('\n'):
        if para.strip():
            lines.append(para.strip())
            lines.append('')
    lines.append('---')
    lines.append('')

OUT.write_text('\n'.join(lines), encoding='utf-8')
print(f'written {OUT}  ({total} chars, ~{total / CPM:.0f} min)')
