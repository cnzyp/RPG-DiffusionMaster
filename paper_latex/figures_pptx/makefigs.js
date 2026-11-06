// Editable PowerPoint versions of the paper's two drawn figures.
//
// The slide is the FINAL SIZE the journal gives the figure: 13.7cm = 5.4in
// wide.  That is the whole point -- a font size set here is that font size on
// the printed page, so nothing can shrink to 3pt the way the original
// matplotlib figure did.  Export with File > Save As > PDF and the page comes
// out at exactly the right size, no cropping.
//
//   node makefigs.js <output-dir>

const pptxgen = require('pptxgenjs');
const path = require('path');
const OUT = process.argv[2] || '.';

const W = 5.4;                       // 13.7 cm, the journal's text width
const F = 'Times New Roman';         // matches the paper body

// The paper's palette.  Each pair survives conversion to greyscale.
const C = {
  blue: '2F6FB0', blueBg: 'EAF1F8',
  green: '3AA66A', greenBg: 'E9F5EE',
  orange: 'E08A1E', orangeBg: 'FBF0DD',
  purple: '7A5AA6', purpleBg: 'F0EBF6',
  red: 'B94040',
  grey: '6B6B6B', greyBg: 'F4F4F4',
  ink: '1A1A1A',
};

// ---------------------------------------------------------------- helpers --
function box(s, x, y, w, h, stroke, fill, text, size) {
  s.addShape('roundRect', {
    x, y, w, h, rectRadius: 0.02,
    fill: { color: fill }, line: { color: stroke, width: 0.75 },
  });
  if (text) {
    s.addText(text, {
      x, y, w, h, fontSize: size || 8, fontFace: F, color: C.ink,
      align: 'center', valign: 'middle', isTextBox: true, margin: 0,
    });
  }
}

function label(s, x, y, w, text, size, color, align) {
  s.addText(text, {
    x, y, w, h: 0.16, fontSize: size || 8, fontFace: F,
    color: color || C.ink, align: align || 'center', valign: 'middle',
    isTextBox: true, margin: 0,
  });
}

function arrow(s, x1, y1, x2, y2) {
  s.addShape('line', {
    x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    line: { color: C.grey, width: 0.75, endArrowType: 'triangle' },
    flipH: x2 < x1, flipV: y2 < y1,
  });
}

// A numbered injection-site marker.
function site(s, x, y, n) {
  const d = 0.15;
  s.addShape('ellipse', {
    x: x - d / 2, y: y - d / 2, w: d, h: d,
    fill: { color: 'FFFFFF' }, line: { color: C.red, width: 0.75 },
  });
  s.addText(String(n), {
    x: x - d / 2, y: y - d / 2, w: d, h: d, fontSize: 7, bold: true,
    fontFace: F, color: C.red, align: 'center', valign: 'middle',
    isTextBox: true, margin: 0,
  });
}

// ============================================================ FIGURE 2 =====
// The ladder: four places to put the reference, and what each recovers.
function figureLadder() {
  const p = new pptxgen();
  p.defineLayout({ name: 'FIG', width: W, height: 3.45 });
  p.layout = 'FIG';
  const s = p.addSlide();

  const yT = 0.62, yI = 1.28, hB = 0.30;   // text row, image row, box height

  // site 4 sits above the text-token path, in its own band
  site(s, 1.95, 0.16, 4);
  label(s, 2.10, 0.08, 2.20, 'identity into the class-noun token', 7, C.red, 'l');

  box(s, 0.05, yT - hB / 2, 0.70, hB, C.grey, C.greyBg, 'prompt');
  box(s, 0.05, yI - hB / 2, 0.70, hB, C.grey, C.greyBg, 'references');
  box(s, 0.88, yT - hB / 2, 0.80, hB, C.purple, C.purpleBg, 'text\nencoder', 7.5);
  box(s, 0.88, yI - hB / 2, 0.80, hB, C.orange, C.orangeBg, 'IP-Adapter\nencoder', 7.5);
  label(s, 1.74, yT - 0.08, 0.62, 'text tokens', 7.5);
  s.addText([{ text: 'image tokens ' }, { text: 'e' }, { text: 'i', options: { subscript: true } }], {
    x: 1.74, y: yI - 0.08, w: 0.75, h: 0.16, fontSize: 7.5, fontFace: F,
    color: C.ink, align: 'center', valign: 'middle', isTextBox: true, margin: 0,
  });

  arrow(s, 0.75, yT, 0.88, yT);
  arrow(s, 0.75, yI, 0.88, yI);
  arrow(s, 1.68, yT, 1.74, yT);
  arrow(s, 1.68, yI, 1.74, yI);

  // the U-Net, with cross-attention called out inside it
  box(s, 2.58, 0.42, 0.95, 1.20, C.grey, C.greyBg, null);
  label(s, 2.58, 0.46, 0.95, 'U-Net', 8);
  box(s, 2.66, 0.80, 0.79, 0.30, C.blue, C.blueBg, 'cross-\nattention', 7);
  arrow(s, 2.49, yT, 2.66, 0.86);
  arrow(s, 2.49, yI, 2.66, 1.05);

  site(s, 3.05, 1.28, 1);
  label(s, 2.45, 1.64, 1.22, 'mask the adapter', 7, C.red);

  s.addText([{ text: 'ε' }, { text: 'θ', options: { subscript: true } }], {
    x: 3.56, y: 0.94, w: 0.22, h: 0.16, fontSize: 8, fontFace: F,
    color: C.ink, align: 'center', valign: 'middle', isTextBox: true, margin: 0,
  });
  arrow(s, 3.53, 1.02, 3.56, 1.02);

  box(s, 4.05, 0.42, 1.28, 0.32, C.green, C.greenBg, 'fuse over the\nfull canvas', 7);
  box(s, 4.05, 1.30, 1.28, 0.32, C.green, C.greenBg, 'fuse per\nsub-canvas', 7);
  arrow(s, 3.78, 0.98, 4.05, 0.60);
  arrow(s, 3.78, 1.06, 4.05, 1.44);
  site(s, 3.94, 0.68, 2);
  site(s, 3.94, 1.36, 3);

  // ---- the ladder itself -------------------------------------------------
  const bx = 2.42, bw = 2.30;          // bars start, length of a 100% bar
  const rungs = [
    ['①  attention, planner’s box', 1.7, '1.7%  (p=0.35)', C.red],
    ['①  attention, box ∩ semantic', 65, '65%', C.red],
    ['②  noise fusion, full canvas', 68, '68%', C.green],
    ['③  noise fusion, sub-canvas', 100, '100%', C.green],
  ];
  rungs.forEach(([name, pct, val, col], i) => {
    const y = 1.98 + i * 0.22;
    s.addText(name, {
      x: 0.05, y: y - 0.08, w: 2.30, h: 0.16, fontSize: 7.5, fontFace: F,
      color: C.ink, align: 'right', valign: 'middle', isTextBox: true, margin: 0,
    });
    s.addShape('rect', {
      x: bx, y: y - 0.045, w: Math.max(bw * pct / 100, 0.012), h: 0.09,
      fill: { color: col, transparency: 72 }, line: { color: col, width: 0.6 },
    });
    s.addText(val, {
      x: bx + bw * pct / 100 + 0.04, y: y - 0.08, w: 0.58, h: 0.16,
      fontSize: 6.5, fontFace: F, color: C.ink, align: 'left',
      valign: 'middle', isTextBox: true, margin: 0, bold: pct === 100,
    });
  });
  s.addText('share of the identity gain the same adapter delivers from the sampling loop', {
    x: 1.40, y: 2.84, w: 3.32, h: 0.14, fontSize: 6, fontFace: F,
    color: C.grey, align: 'right', valign: 'middle', isTextBox: true, margin: 0,
  });
  s.addText([
    { text: '④  text stream', options: { bold: true } },
    { text: ' — a different axis: ArcFace +0.244 against +0.254 for site 3, reached at 62% of the face area and with no image branch at all' },
  ], {
    x: 0.05, y: 3.02, w: 5.30, h: 0.36, fontSize: 7.5, fontFace: F,
    color: C.ink, align: 'left', valign: 'top', isTextBox: true, margin: 0,
  });

  return p.writeFile({ fileName: path.join(OUT, 'fig2_ladder.pptx') });
}

// ============================================================ FIGURE 3 =====
// The endpoint: sub-canvas early, full canvas late.
function figurePipeline() {
  const p = new pptxgen();
  p.defineLayout({ name: 'FIG', width: W, height: 2.75 });
  p.layout = 'FIG';
  const s = p.addSlide();

  const hB = 0.28;

  // planning row
  box(s, 0.05, 0.14 - 0.02, 0.62, hB, C.grey, C.greyBg, 'prompt', 7.5);
  box(s, 0.05, 0.52 - 0.02, 0.62, hB, C.grey, C.greyBg, 'refs rᵢ', 7.5);
  box(s, 0.80, 0.14 - 0.02, 0.78, hB, C.purple, C.purpleBg, 'MLLM\nplanner', 7);
  box(s, 0.80, 0.52 - 0.02, 0.78, hB, C.orange, C.orangeBg, 'IP-Adapter\nPlus', 7);
  label(s, 1.66, 0.20, 1.30, 'Ωᵢ regions, prompts pᵢ', 7.5, C.ink, 'l');
  label(s, 1.66, 0.58, 1.30, 'tokens eᵢ', 7.5, C.ink, 'l');
  arrow(s, 0.67, 0.26, 0.80, 0.26);
  arrow(s, 0.67, 0.64, 0.80, 0.64);
  arrow(s, 0.60, 0.82, 0.60, 1.02);

  // early phase
  label(s, 0.30, 0.92, 2.40, 'early (t > τ): each region on its own sub-canvas', 7, C.ink, 'l');
  box(s, 0.18, 1.12, 0.86, 0.30, C.blue, C.blueBg, 'Ω₁: U-Net\n+ e₁', 7);
  box(s, 0.18, 1.60, 0.86, 0.30, C.green, C.greenBg, 'Ω₂: U-Net\n+ e₂', 7);
  box(s, 1.30, 1.36, 0.74, 0.30, C.grey, C.greyBg, 'fuse εᵢ,\nfeathered', 7);
  arrow(s, 1.04, 1.27, 1.30, 1.45);
  arrow(s, 1.04, 1.75, 1.30, 1.57);

  // the intermediate: two identities, one seam
  s.addShape('rect', { x: 2.22, y: 1.33, w: 0.18, h: 0.36, fill: { color: C.blueBg }, line: { color: C.grey, width: 0.5 } });
  s.addShape('rect', { x: 2.40, y: 1.33, w: 0.18, h: 0.36, fill: { color: C.greenBg }, line: { color: C.grey, width: 0.5 } });
  arrow(s, 2.04, 1.51, 2.22, 1.51);
  label(s, 1.80, 1.74, 1.20, 'identities written,\nseams visible', 6, C.grey);

  s.addShape('roundRect', {
    x: 0.10, y: 1.06, w: 2.56, h: 0.92, rectRadius: 0.02,
    fill: { type: 'none' }, line: { color: C.grey, width: 0.6, dashType: 'dash' },
  });

  // the switch
  s.addShape('line', {
    x: 2.78, y: 0.98, w: 0, h: 1.18,
    line: { color: C.red, width: 0.75, dashType: 'dash' },
  });
  label(s, 2.30, 0.86, 0.96, 'switch at τ', 6.5, C.red);

  // late phase
  label(s, 3.00, 0.92, 2.30, 'late (t ≤ τ): one pass, global prompt', 7, C.ink, 'l');
  box(s, 3.00, 1.36, 1.20, 0.30, C.red, 'FFFFFF', 'U-Net, full canvas\n+ p, {eᵢ}', 6.5);
  s.addShape('rect', { x: 4.52, y: 1.33, w: 0.36, h: 0.36, fill: { color: C.greenBg }, line: { color: '000000', width: 0.6 } });
  s.addShape('rect', { x: 4.58, y: 1.40, w: 0.12, h: 0.24, fill: { color: C.blueBg }, line: { type: 'none' } });
  arrow(s, 2.58, 1.51, 3.00, 1.51);
  arrow(s, 4.20, 1.51, 4.52, 1.51);
  label(s, 4.10, 1.74, 1.20, 'one scene: shared\nlight and background', 6, C.grey);

  s.addShape('roundRect', {
    x: 2.92, y: 1.06, w: 2.06, h: 0.92, rectRadius: 0.02,
    fill: { type: 'none' }, line: { color: C.red, width: 0.6, dashType: 'dash' },
  });

  // the denoising timeline the whole panel is laid out along
  s.addShape('line', {
    x: 0.10, y: 2.36, w: 5.20, h: 0,
    line: { color: '000000', width: 0.7, endArrowType: 'triangle' },
  });
  label(s, 0.10, 2.44, 1.00, 't = T (noise)', 6, C.grey, 'l');
  label(s, 4.30, 2.44, 1.00, 't = 0 (image)', 6, C.grey, 'r');
  s.addShape('line', { x: 2.78, y: 2.28, w: 0, h: 0.16, line: { color: C.red, width: 0.75 } });
  label(s, 2.58, 2.44, 0.40, 'τ', 6.5, C.red);

  return p.writeFile({ fileName: path.join(OUT, 'fig3_pipeline.pptx') });
}

figureLadder()
  .then(figurePipeline)
  .then(() => console.log('written fig2_ladder.pptx and fig3_pipeline.pptx'))
  .catch((e) => { console.error(e); process.exit(1); });
