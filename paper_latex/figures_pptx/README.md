# Editable figures

`fig2_ladder.pptx` and `fig3_pipeline.pptx` are PowerPoint versions of the
paper's two drawn figures, in native shapes: every box, arrow and label can be
dragged, retyped and recoloured.

## The one rule that matters

**The slide is already the final size the journal gives the figure** —
13.7 cm wide, which is `\linewidth` in the single-column Pattern Recognition
layout. So a font size set in PowerPoint is that font size on the printed
page.

Do not redraw these on a normal 16:9 slide. That canvas is 33.9 cm wide, so
everything on it shrinks 2.5x when placed in the paper: 9pt text arrives at
3.6pt. That is exactly how the original `fig2_method.png` failed — it was
authored 15.5 inches wide with 7.2–9.5pt labels, which reached the page at
2.5–3.3pt.

Keep labels at **8pt minimum** (Pattern Recognition's own caption size), 9pt
preferred.

## Exporting back into the paper

    File > Save As > PDF

One slide, sized exactly, so the PDF needs no cropping. Then in `main.tex`:

    \includegraphics[width=\linewidth]{your_figure.pdf}

Check the export by opening it at 400%: text must stay sharp. If it is fuzzy,
something became a bitmap.

## Palette

Stroke / fill pairs, chosen to stay distinguishable in greyscale — many
reviewers print in black and white, so never rely on colour alone to
distinguish two things. Every element carries a text label for that reason.

| Role | Stroke | Fill |
|---|---|---|
| region 1 / cross-attention | `2F6FB0` | `EAF1F8` |
| region 2 / fusion | `3AA66A` | `E9F5EE` |
| IP-Adapter | `E08A1E` | `FBF0DD` |
| text / planner | `7A5AA6` | `F0EBF6` |
| site markers, tau | `B94040` | white |
| neutral | `6B6B6B` | `F4F4F4` |

Font is Times New Roman, matching the paper body. Line weights are 0.5–0.75pt;
PowerPoint's default is heavier and reads clumsy at this scale.

## Numbers — copy, do not retype

    ladder     1.7% (p=0.35)  /  65%  /  68%  /  100%
    site 4     ArcFace +0.244 vs +0.254, at 62% of the face area
    tau        default 0.7, of 30 steps
    seam       1.059 (no adapter) / 1.103 (ours) / 1.37 (collage)

Two things in figure 2 are load-bearing and easy to break by accident:

- the four sites must sit on **one** denoising step, not four separate
  diagrams — that they are alternatives at one point is the figure's argument;
- **site 4 must not get a bar.** It is measured by ArcFace, not as a share of
  the identity gain, so a bar would put it on an axis it is not on.

## Regenerating

    node makefigs.js <output-dir>

Edits made in PowerPoint are lost if this is re-run over them; it is a
starting point, not a build step.

> **Note.** The paper's figures were subsequently redrawn in TikZ
> (`paper_latex/fig_ladder_tikz.tex`, `fig_method_tikz.tex`), which is now the
> source of truth. These PowerPoint files are the earlier layout, kept in case
> a hand-edit is ever wanted; they are not what the paper builds from.
