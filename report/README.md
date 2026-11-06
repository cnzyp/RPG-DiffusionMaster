# Advisor report

- `论文汇报.pptx` — 13 slides, 16:9. Every slide carries the spoken script in
  its notes pane.
- `汇报稿.md` — the same script standalone, per slide, with timings. ~21 min
  at 240 characters/minute, excluding questions.

The script lives in the deck's notes; `汇报稿.md` is generated from it, so the
two cannot drift. Edit the deck (or `makedeck.js`), then:

    node makedeck.js . .          # rebuild the deck
    python make_script.py         # regenerate 汇报稿.md from its notes

`slide_ladder.png` and `slide_pipeline.png` are the paper's own TikZ figures
exported wide for the slides, not redrawn — so what the advisor sees is what
the reviewer will see.

Every number in the deck was read off the built manuscript rather than from
memory. Two things in it are worth raising explicitly rather than hoping they
do not come up:

- slide 8 reports the MS-Diffusion comparison error that was caught and fixed
  before submission (768 numbers left in a table that had moved to 1024, which
  made the text and the table disagree on who won isolation);
- slide 10 reports that the late release's scene-coherence benefit is not
  visible — measured across all 150 samples, the boundary step drops by at
  most 5%.
