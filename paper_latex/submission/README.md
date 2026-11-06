# Pattern Recognition submission

Everything in the tables below was read from the journal's own Guide for
Authors on **2026-11-06**, not from a secondary source. An earlier version of
this file quoted a circulating Chinese guide because the Guide itself could
not be fetched; three of its claims were wrong and are corrected at the
bottom.

- Guide for Authors: https://www.sciencedirect.com/journal/pattern-recognition/publish/guide-for-authors
- Submission system: https://www.editorialmanager.com/pr
- Article type: **Full Length Article**

## Build everything

```
python tools/make_pr_tex.py --named     # regenerates both manuscripts + supplement
python tools/check_numbers.py           # 112 numeric claims cross-checked
python tools/export_figures.py          # separate figure files + captions.txt
python tools/build_submission_docs.py   # Highlights.docx, Cover_letter.docx
python tools/make_source_zip.py --named # the .tex archive EM wants
cd paper_latex/submission && pdflatex title_page
```

`main_pr.tex`, `main_pr_named.tex` and `supp_pr.tex` are **generated**. Edit
`main.tex` and regenerate; everything that differs between the CVPR and
journal versions lives in `tools/make_pr_tex.py`, so the two cannot drift.

Manuscript and supplement cite each other through `xr`, so the build order is
main, supp, main, supp. `tools/` has no build script for this; use the one in
the session notes or run the four passes by hand.

## Journal requirements, checked

| Requirement (verbatim from the Guide) | This submission |
|---|---|
| single-column, double-spaced, fully-justified, numbered pages | yes |
| 20–35 pages incl. figures, tables, references, bio-sketches, appendices | **46** — over, see below |
| "Appendices are not included in the page limit" | not currently used |
| Times New Roman; text incl. tables 10pt; footnotes and captions 8pt | yes |
| Margins 4.3cm, 4.8cm, 4.3cm, 4.8cm (T, R, B, L) | yes, A4, set by `geometry` |
| Upload editable source files (.tex); "PDF is not an acceptable source file" | `main_pr_named_source.zip` |
| Tables as editable text, no vertical rules or shading | yes, booktabs |
| Figures within the text **and as separate files**, numbered and cited | `figures/Fig1..Fig6` |
| Captions supplied separately, not attached to the figure | `figures/captions.txt` |
| Title 14pt, ideally 10–15 words | 14pt, 13 words |
| Title page: full names, affiliations, corresponding contact, 8pt | `title_page.pdf` |
| Abstract up to 250 words | 248 |
| 1–7 keywords | 6 |
| Highlights mandatory: 3–5 bullets, ≤85 characters each, separate editable file | `Highlights.docx`, 5 bullets, 77–80 |
| Conclusions longer than and different from the abstract | 536 words against 248 |
| Sections numbered 1, 1.1, 1.1.1; cross-referenced by number | yes |
| References numbered in order, square brackets | `elsarticle-num` |
| CRediT statement | on the title page and in the named manuscript |
| Review model: "This journal follows a **single anonymized** review process" | see below |

Build status: **0 errors, 0 undefined citations or references, 0 overfull
boxes** in all three documents. Manuscript 46 pages, supplement 28.

## Three things still to decide

### 1. Blind or named

The Guide says single anonymized — reviewers are anonymous, authors are not —
so **`main_pr_named.pdf` is the one to submit**. Both builds exist and are
identical apart from the byline and the CRediT block; `main_pr.pdf` is the
fallback if Editorial Manager's own submission steps say otherwise, which is
the only authority above the Guide.

### 2. Length: 46 pages against 35

Compressing prose returns 9–11% of what it touches, measured three times.
Only relocating whole units returns all of it, and 16 subsections have already
gone to the supplement.

What is newly available: the Guide says **"Appendices are not included in the
page limit"**, so material moved into a lettered appendix of the manuscript
stops counting, while staying in the same document as the argument that needs
it. The cost is that "they will appear only in the digital format of the
paper." Appendices are formatted A, B, …, with Eq. (A.1), Table A.1, Fig. A.1.

Submitting at 46 is still a defensible choice — what is left is argument, not
apparatus — but it is now a choice with a sanctioned alternative.

### 3. The repository

`github.com/cnzyp/RPG-DiffusionMaster` is public but `origin/main` is still
upstream RPG code; `paper/tau05-default` has never been pushed. **The Data
availability statement in the named manuscript and on the title page is
therefore not true as written.** Either push the branch and make it the
default, or change the statement to describe what is actually available.

## Files to upload

| Editorial Manager item | File |
|---|---|
| Manuscript (PDF for review) | `paper_latex/main_pr_named.pdf` |
| Manuscript source (required) | `submission/main_pr_named_source.zip` |
| Title page | `submission/title_page.pdf` |
| Highlights | `submission/Highlights.docx` |
| Cover letter | `submission/Cover_letter.docx` |
| Declaration of competing interests | `submission/Declaration_of_interests.docx` |
| Figures, separate files | `submission/figures/Fig1.png` … `Fig6.png`/`.pdf` |
| Figure captions | `submission/figures/captions.txt` |
| Supplementary material | `paper_latex/supp_pr.pdf` |

The supplement reads `main_pr.aux` through `xr`, so it is uploaded as a built
PDF, not as source to be compiled by Editorial Manager.

Editorial Manager lists three items as **required** for a Full Length
Article: Manuscript, Highlights, and Declaration of competing interests.
The last one needs a file of its own even though the statement also appears
in the manuscript and on the title page.

Editorial Manager's LaTeX processor refuses an archive with subfolders --
"The system cannot process a zipfile containing one or more subfolders" --
so `make_source_zip.py` writes a flat archive and strips the
`paper_figures/` prefix from the `.tex` inside it. The working copy in the
repository keeps its directories.

It also warns that **at revision stage PDFs are not accepted** for the
Manuscript, Table and Title page item types -- editable files are wanted
then, because they are used for typesetting. For a LaTeX submission the
source archive serves that purpose; see Elsevier's LaTeX submission guidance
linked from the attach-files screen.

## Left for proof stage, not blocking

- `refs.bib` uses short venue names (CVPR, NeurIPS). Elsevier spells
  conference names out and abbreviates journal names per the LTWA.
- The Guide recommends a title in the form 'Topic - application'. Ours is not,
  and it is a recommendation rather than a rule.
- Graphical abstract: optional, not prepared.
- Suggested reviewers: the authors declined to name any. If Editorial Manager
  makes the field mandatory it is filled in the web form.

## Three corrections to the circulating Chinese guide

- It gives `\documentclass[preprint,review,3p,times,sort&compress]{elsarticle}`.
  `preprint`, `review` and `3p` are **mutually exclusive** layout options and
  must not be passed together. This build uses `[review,10pt,times]`.
- It says Highlights are "85 words in total". The Guide's spec is **85
  characters, including spaces, per bullet**.
- It gives a 150-word abstract limit; the Guide's figure is **250**.

It was also the source of the double-blind claim, which the Guide contradicts.
