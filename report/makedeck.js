// Advisor progress report on the Pattern Recognition submission.
//
// Every number is taken from the built manuscript, not from memory.  The two
// mechanism figures are the paper's own, exported from TikZ.
//
//   node makedeck.js <output-dir> <figure-dir>

const pptxgen = require('pptxgenjs');
const path = require('path');
const OUT = process.argv[2] || '.';
const FIG = process.argv[3] || '.';

const F = 'Microsoft YaHei';          // ships with Windows, handles CJK + Latin
const C = {
  ink: '1A1A1A', mute: '6B6B6B', faint: '9AA0A6',
  blue: '2F6FB0', blueBg: 'EAF1F8',
  green: '2E7D55', greenBg: 'E9F5EE',
  red: 'B94040', redBg: 'FBEDED',
  rule: 'D8DCE2', band: 'F7F9FC',
};

const p = new pptxgen();
p.defineLayout({ name: 'W', width: 13.333, height: 7.5 });
p.layout = 'W';
p.author = 'Yaping Zhang';
p.title = '注入位点决定身份能否保留';

const M = 0.72;                        // left margin
const WIDE = 13.333 - 2 * M;

// ---------------------------------------------------------------- helpers --
function slide(title, kicker) {
  const s = p.addSlide();
  if (title) {
    s.addText(title, {
      x: M, y: 0.44, w: WIDE, h: 0.56, fontSize: 28, bold: true,
      fontFace: F, color: C.ink, isTextBox: true, margin: 0, valign: 'middle',
    });
  }
  if (kicker) {
    s.addText(kicker, {
      x: M, y: 1.04, w: WIDE, h: 0.34, fontSize: 14, fontFace: F,
      color: C.mute, isTextBox: true, margin: 0, valign: 'middle',
    });
  }
  return s;
}

function body(s, items, opts = {}) {
  const y = opts.y || 1.72;
  const size = opts.size || 17;
  const gap = opts.gap || 0.62;
  items.forEach((it, i) => {
    const runs = typeof it === 'string' ? [{ text: it }] : it;
    s.addText(runs, {
      x: opts.x || M, y: y + i * gap, w: opts.w || WIDE, h: gap - 0.06,
      fontSize: size, fontFace: F, color: C.ink, isTextBox: true,
      margin: 0, valign: 'top', lineSpacing: size * 1.45,
    });
  });
}

// A number worth looking at, with its label underneath.
function stat(s, x, y, w, value, label, color) {
  s.addText(value, {
    x, y, w, h: 0.74, fontSize: 40, bold: true, fontFace: F,
    color: color || C.ink, align: 'center', valign: 'middle',
    isTextBox: true, margin: 0,
  });
  s.addText(label, {
    x, y: y + 0.74, w, h: 0.62, fontSize: 12.5, fontFace: F, color: C.mute,
    align: 'center', valign: 'top', isTextBox: true, margin: 0,
  });
}

function card(s, x, y, w, h, fill, line) {
  s.addShape('roundRect', {
    x, y, w, h, rectRadius: 0.05,
    fill: { color: fill }, line: { color: line || C.rule, width: 0.75 },
  });
}

function note(s, t) { s.addNotes(t); }

// ============================================================ 1. title =====
{
  const s = p.addSlide();
  s.addShape('rect', { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: '15283C' }, line: { type: 'none' } });
  s.addText('注入位点决定身份能否保留', {
    x: M, y: 2.35, w: WIDE, h: 0.9, fontSize: 40, bold: true, fontFace: F,
    color: 'FFFFFF', isTextBox: true, margin: 0,
  });
  s.addText('The Injection Site Determines What Identity Survives in\nTraining-Free Multi-Concept Personalization', {
    x: M, y: 3.30, w: WIDE, h: 1.0, fontSize: 19, fontFace: F,
    color: 'AFC4DA', isTextBox: true, margin: 0, lineSpacing: 28,
  });
  s.addShape('line', { x: M, y: 4.52, w: 2.4, h: 0, line: { color: '4E7CA8', width: 1.5 } });
  s.addText('投稿 Pattern Recognition（CAS 一区 · 双盲）     ·     汇报人：张亚苹', {
    x: M, y: 4.78, w: WIDE, h: 0.4, fontSize: 15, fontFace: F,
    color: 'AFC4DA', isTextBox: true, margin: 0,
  });
  note(s, `老师好，我汇报一下这篇投 Pattern Recognition 的论文。

题目是《注入位点决定身份能否保留》。一句话概括：训练免调的多概念个性化生成里，大家都会在扩散过程的某个地方加空间约束，但"加在哪"这个选择从来没有被单独隔离出来研究过。我们把它隔离出来了，发现主流做法几乎是无效的。

整个汇报分四部分：问题和空白、核心发现、方法和结果、投稿状态和下一步。大概二十分钟。`);
}

// ============================================================ 2. problem ===
{
  const s = slide('问题：一张图里放多个"指定的"概念', '不是"一个女人和一盆植物"，而是"这个女人和这盆植物"');
  body(s, [
    [{ text: '输入：', options: { bold: true } },
     { text: '一句话 + 每个概念一张参考照片（推理时才给，不能为每个概念训练）' }],
    [{ text: '要同时满足三件事：', options: { bold: true } }],
  ], { y: 1.78, gap: 0.56 });

  const items = [
    ['身份', '每个区域画的是\n给定的那个实例'],
    ['隔离', '概念之间不串味\n左边的脸不跑到右边'],
    ['构图', '按提示词的布局\n不被参考图绑架'],
  ];
  items.forEach(([t, d], i) => {
    const x = M + i * 4.0;
    card(s, x, 3.05, 3.65, 1.5, C.band);
    s.addText(t, { x, y: 3.18, w: 3.65, h: 0.42, fontSize: 19, bold: true, fontFace: F, color: C.blue, align: 'center', isTextBox: true, margin: 0 });
    s.addText(d, { x, y: 3.62, w: 3.65, h: 0.82, fontSize: 13.5, fontFace: F, color: C.mute, align: 'center', isTextBox: true, margin: 0, lineSpacing: 19 });
  });

  s.addText('现有方法做不到三者兼得：纯文本方法（RPG）构图好但不像；全局图像适配器（IP-Adapter）像但串味、构图被毁。', {
    x: M, y: 5.0, w: WIDE, h: 0.6, fontSize: 15, fontFace: F, color: C.ink,
    isTextBox: true, margin: 0, lineSpacing: 23,
  });
  note(s, `先说问题。

多概念个性化生成，关键词是"指定的"。不是生成"一个女人和一盆植物"，而是"这个女人和这盆植物"——每个概念用户给一张参考照片，而且是推理的时候才给，不能为每个新概念去训练。

要同时满足三件事：身份要像、概念之间不能串味、构图还得听提示词的。

现有方法做不到三者兼得。纯文本的布局方法比如 RPG，构图很好但它根本看不到参考图，画出来不像。全局的图像适配器比如 IP-Adapter，像是像了，但一张参考图的特征会铺满整个画布，另一个概念直接消失，构图也被参考照片绑架了。`);
}

// ============================================================ 3. gap ======
{
  const s = slide('空白：所有人都在加空间约束，但"加在哪"没人单独研究过', '文献报告的是架构，不是机制');
  body(s, [
    '训练免调的方法都要把"这个概念只能出现在这个区域"这件事告诉模型。',
    [{ text: '但每篇论文同时换掉了很多东西', options: { bold: true } },
     { text: '——骨干网络、适配器、注入强度、布局来源、采样器。' }],
    '所以读者无法知道：效果的差异来自"约束加在哪"，还是来自其它变量。',
  ], { y: 1.80, gap: 0.56 });

  card(s, M, 3.75, WIDE, 1.72, C.blueBg, C.blue);
  s.addText('我们的做法：把其它全部钉死', {
    x: M + 0.35, y: 3.95, w: WIDE - 0.7, h: 0.4, fontSize: 19, bold: true,
    fontFace: F, color: C.blue, isTextBox: true, margin: 0,
  });
  s.addText('骨干 SDXL-base-1.0 · 适配器 IP-Adapter Plus · 注入强度 λ=0.8 · 布局规划器 · 随机种子 42 · 分辨率 1024\n只变一件事：参考图从哪里进入扩散过程。', {
    x: M + 0.35, y: 4.38, w: WIDE - 0.7, h: 0.9, fontSize: 15, fontFace: F,
    color: C.ink, isTextBox: true, margin: 0, lineSpacing: 24,
  });
  s.addText('这是这篇论文唯一的实验设计，后面所有结论都建立在它上面。', {
    x: M, y: 5.72, w: WIDE, h: 0.4, fontSize: 15, italic: true, fontFace: F,
    color: C.mute, isTextBox: true, margin: 0,
  });
  note(s, `空白在哪里。

训练免调的方法，都得把"这个概念只能出现在这个区域"告诉模型。但每篇论文在提出自己方法的时候，同时换掉了很多东西——骨干网络、适配器、注入强度、布局从哪来、用什么采样器。

结果就是，读者没办法知道效果的差异到底来自"约束加在哪里"，还是来自其它那些变量。文献报告的是一个一个的架构，不是机制。

我们的做法很简单：把其它全部钉死。骨干都用 SDXL-base-1.0，适配器都用 IP-Adapter Plus，注入强度都是 0.8，同一个布局规划器，同一个随机种子 42，同一个分辨率 1024。只变一件事：参考图从哪里进入扩散过程。

这是整篇论文唯一的实验设计，后面所有结论都建立在它上面。`);
}

// ============================================================ 4. ladder ===
{
  const s = slide('核心发现：四个注入位点，差一个数量级', '同一个去噪步上的四种选择，不是四条不同的管线');
  s.addImage({ path: path.join(FIG, 'slide_ladder.png'), x: M, y: 1.62, w: WIDE, h: 5.31 });
  note(s, `这是整篇论文的核心图，我重点讲。

上半部分是一个去噪步。提示词经过文本编码器变成 text tokens，参考图经过 IP-Adapter 编码器变成 image tokens，两路都进 U-Net 的 cross-attention，出来一个噪声预测 epsilon。注意所有模块都标了雪花——全部是冻结的，我们不训练任何东西。

四个红圈就是四个可以把"空间约束"放进去的位置：
位点一，在 cross-attention 内部给适配器加 mask——这是目前绝大多数训练免调方法的做法；
位点二，让适配器正常跑完，在噪声层面按区域融合；
位点三，每个区域在自己的子画布上跑；
位点四，把身份写进文本流里的类名词 token。

下半部分是结果。条形是"同一个适配器从采样循环能给出的身份增益，各个位点能回收多少"。

位点一只回收了 1.7%，而且 p 等于 0.35——统计上和零没区别。就算用我们能做出的最强 mask，也就是规划器的框和区域自己的语义注意力取交集，也只到 65%。
位点二是 68%，位点三是 100%。

第一个和第二个台阶之间差了一个数量级。这就是论文的核心发现：主流做法不是"效果差一点"，是基本不传递身份。

位点四我特意没给条形——它测的是 ArcFace 人脸相似度，不是同一个量纲，给了条就是把它放到一个它不在的坐标轴上。`);
}

// ============================================================ 5. not weak =
{
  const s = slide('不是"注入太弱"——三个排除性实验', '审稿人一定会问的问题，我们提前堵死');
  const rows = [
    ['把强度拉到 2.5 倍', '图像被改写的幅度和我们一样大，身份响应仍然是零'],
    ['换更强的 mask', '框 ∩ 语义注意力，65%，仍比采样循环少三分之一'],
    ['扫掉管线的基础混合', '扫到零，输出不变'],
  ];
  rows.forEach(([t, d], i) => {
    const y = 1.88 + i * 1.12;
    card(s, M, y, WIDE, 0.94, 'FFFFFF');
    s.addText(t, { x: M + 0.3, y: y + 0.08, w: 3.9, h: 0.78, fontSize: 17, bold: true, fontFace: F, color: C.ink, valign: 'middle', isTextBox: true, margin: 0 });
    s.addText(d, { x: M + 4.3, y: y + 0.08, w: WIDE - 4.6, h: 0.78, fontSize: 15, fontFace: F, color: C.mute, valign: 'middle', isTextBox: true, margin: 0 });
  });
  card(s, M, 5.35, WIDE, 1.0, C.redBg, C.red);
  s.addText('结论：问题不在注入强度，在注入位置。mask 能限制身份写到哪里，但不能凭空产生身份。', {
    x: M + 0.35, y: 5.5, w: WIDE - 0.7, h: 0.7, fontSize: 17, bold: true,
    fontFace: F, color: C.red, valign: 'middle', isTextBox: true, margin: 0,
  });
  note(s, `这一页是防守。任何审稿人看到"1.7%"，第一反应都是"你是不是注入强度给小了"。我们做了三个实验把这条路堵死。

第一，把注入强度拉到我们自己用的 2.5 倍。在"换参考图"的检验下，它的图像被改写的幅度和我们一样大——说明适配器确实在工作、确实在改图——但身份响应还是零。

第二，换更强的 mask。用规划器的框和区域自己的文本注意力取交集，这是我们能构造出的最强空间 mask，到 65%，仍然比采样循环少三分之一。

第三，把管线自带的基础混合扫到零，输出没有变化。

所以结论是：问题不在强度，在位置。一个 mask 能限制身份写到哪里，但不能凭空产生身份。`);
}

// ============================================================ 6. method ===
{
  const s = slide('方法：阶梯的终点 —— 子画布 + 晚期释放', '不改适配器，把空间约束移到采样循环里');
  s.addImage({ path: path.join(FIG, 'slide_pipeline.png'), x: M, y: 1.62, w: WIDE, h: 5.07 });
  note(s, `既然位置是关键，方法就顺理成章了：不动适配器，把空间约束移到采样循环里。

上面一行是规划：一个多模态大模型把提示词拆成区域和子提示词，并且把第 i 张参考图指派给第 i 个区域；IP-Adapter Plus 把每张参考图编码成 tokens。

下面是采样，沿着去噪时间轴分成两段，分界点是 tau。

早期，t 大于 tau：每个区域在自己的子画布上，用完全没改过的、满强度的适配器去噪，每个区域出一个噪声预测，用带羽化重叠的方式融合，然后走一个调度器步。这个过程迭代到 t 等于 tau。

晚期，t 小于等于 tau：切换成一次全画布的前向，用全局提示词和所有的 tokens。

右边的缩略图都是真实渲染结果，不是示意图——两张参考图、我们输出的左右两半、还有同样提示词同样种子下纯文本 RPG 的结果。可以看到纯文本的结果里，两个参考对象一个都没有。

这里我要特别说明一点：我原来的图在这个位置画了一个"有接缝"的中间结果和一个"一个完整场景"的输出，声称晚期释放把接缝合上了。我后来把全部 150 个样本都量了一遍，tau=1 到 tau=0.7 的边界跳变最大降幅只有 5%，肉眼完全看不出来。所以这个前后对比我删掉了，图注也改成只说设计意图、并指向局限性那一节。论文正文一直是诚实的，是图在夸大。`);
}

// ============================================================ 7. results ==
{
  const s = slide('主结果：Mixed-Eval-v1（150 样本，1024 分辨率）', '所有方法同骨干、同采样配置、同种子');
  const head = ['方法', 'DINO-ID ↑', 'Margin ↑', 'CLIP-T ↑', 'ArcFace ↑'];
  const rows = [
    ['RPG-only（纯文本）', '0.352', '+0.076', '0.650', '0.129'],
    ['IP-Adapter（全局）', '0.341', '−0.082', '0.615', '0.089'],
    ['Conceptrol（注意力门控）', '0.387', '+0.004', '0.630', '0.165'],
    ['Attention gating（满强度）', '0.358', '+0.098', '0.652', '0.122'],
    ['MS-Diffusion（训练过）', '0.648', '+0.437', '0.651', '0.219'],
    ['本文方法', '0.720', '+0.488', '0.647', '0.266'],
  ];
  const tbl = [head.map((h) => ({
    text: h, options: { bold: true, color: 'FFFFFF', fill: { color: '15283C' }, fontSize: 14 },
  }))];
  rows.forEach((r, i) => {
    const last = i === rows.length - 1;
    tbl.push(r.map((cell, j) => ({
      text: cell,
      options: {
        bold: last, fontSize: 14,
        color: last ? '15283C' : C.ink,
        fill: { color: last ? C.blueBg : 'FFFFFF' },
        align: j === 0 ? 'left' : 'center',
      },
    })));
  });
  s.addTable(tbl, {
    x: M, y: 1.74, w: WIDE, colW: [4.2, 1.9, 1.9, 1.9, 2.0],
    border: { type: 'solid', color: C.rule, pt: 0.5 },
    fontFace: F, rowH: 0.42, valign: 'middle',
  });
  s.addText('Margin = 区域裁片与"自己那张"参考图的相似度，减去与"别人那张"的相似度。为正才说明概念真的被隔离开了。', {
    x: M, y: 5.05, w: WIDE, h: 0.5, fontSize: 13.5, fontFace: F, color: C.mute,
    isTextBox: true, margin: 0, lineSpacing: 20,
  });
  s.addText('身份翻一倍（0.352 → 0.720），隔离度翻六倍（+0.076 → +0.488），而构图质量（CLIP-T）没有下降。', {
    x: M, y: 5.62, w: WIDE, h: 0.5, fontSize: 17, bold: true, fontFace: F,
    color: C.green, isTextBox: true, margin: 0,
  });
  note(s, `主结果表。150 个样本，所有方法同一个骨干、同一套采样配置、同一个种子、都在 1024 分辨率。

先解释 Margin 这一列，它是我们提出的隔离度指标：每个区域的裁片，跟"指派给它的那张参考图"的相似度，减去跟"别人那张"的相似度。为正才说明概念真的被分开了。

看 IP-Adapter 全局那一行，Margin 是负的 0.082——每个区域反而更像错的那张参考图，这就是串味。Conceptrol 是正的 0.004，基本等于没隔离，它典型的失败模式是把两个参考图融成一个人。

我们的方法，DINO-ID 从纯文本的 0.352 到 0.720，翻了一倍；Margin 从 +0.076 到 +0.488，翻了六倍；而 CLIP-T 是 0.647，和纯文本的 0.650 基本持平——也就是说构图质量没有被牺牲。`);
}

// ============================================================ 8. vs MS ====
{
  const s = slide('和训练过的模型比：MS-Diffusion', '它用了一个带主体标注的语料训练过，我们的骨干是冻结的');
  stat(s, M, 1.80, 2.7, '+0.072', '身份 DINO-ID\np < 0.001', C.green);
  stat(s, M + 2.9, 1.80, 2.7, '+0.047', '人脸身份 ArcFace\np < 0.001', C.green);
  stat(s, M + 5.8, 1.80, 2.7, '+0.051', '隔离度 Margin\np < 0.001', C.green);
  stat(s, M + 8.7, 1.80, 2.7, '−0.004', '文本对齐 CLIP-T\np = 0.005', C.red);

  card(s, M, 3.92, WIDE, 1.18, C.band);
  s.addText('四个可做配对检验的指标里，我们赢三个、输一个；而且是在 0.83 倍的单图成本下做到的（9.88 秒 vs 11.92 秒）。', {
    x: M + 0.35, y: 4.08, w: WIDE - 0.7, h: 0.86, fontSize: 17, fontFace: F,
    color: C.ink, valign: 'middle', isTextBox: true, margin: 0, lineSpacing: 26,
  });

  card(s, M, 5.28, WIDE, 1.28, 'FFF8E8', 'D9862B');
  s.addText([
    { text: '⚠  一个要向老师报告的修正：', options: { bold: true, color: 'A8651B' } },
    { text: '这段对比原先写的是 768 分辨率的数字，主表换到 1024 后没同步，导致正文说"隔离度我们输"、表格说"我们赢"。我从每样本数据重算后已经改正——1024 下我们领先 +0.051，p=1.8e−8。', options: { color: '6B4A12' } },
  ], {
    x: M + 0.35, y: 5.44, w: WIDE - 0.7, h: 0.96, fontSize: 14.5, fontFace: F,
    valign: 'middle', isTextBox: true, margin: 0, lineSpacing: 22,
  });
  note(s, `最有信息量的对比是和 MS-Diffusion。它是用一个带主体标注的语料训练过的多主体模型，我们的骨干是完全冻结的。两边同骨干、同分辨率、同采样配置、同布局。

四个能做配对统计检验的指标：身份我们领先 0.072，人脸身份领先 0.047，隔离度领先 0.051，三个都是 p 小于 0.001。文本对齐我们输 0.004，p 等于 0.005。

三胜一负，而且是在 0.83 倍的单图成本下——9.88 秒对 11.92 秒。

这里有一件事要专门向老师报告。这段对比原先引用的是 768 分辨率的实验数字，后来主表换成 1024 之后没有同步更新，结果造成正文写着"隔离度这一项我们输"，而同一页的表格显示我们赢。我从每个样本的原始数据重新算了一遍，1024 下我们是领先 0.051，p 等于 1.8 乘以 10 的负 8 次方。现在已经改正，论文里所有相关段落都统一了。

这是个很容易被审稿人抓到的硬伤，幸好在投稿前发现了。`);
}

// ============================================================ 9. measure ==
{
  const s = slide('三个测量贡献', '论文的分量有一半在这里，不在方法上');
  const items = [
    ['① 换参考图检验（reference-swap）',
     '相似度分不出"弱的适配器"和"死的适配器"。我们把某个区域的参考图换成同类别的另一个实例，其它全不变，问输出跟不跟。',
     '一个不看参考图的基线在这个检验下恰好是零——这给了这个测量一个相似度指标给不了的正确性校验。门控适配器大幅改写了图像，但朝新参考图移动的幅度和零没区别（p=0.79）。'],
    ['② 身份特异性（identity specificity）',
     '区分"复现了这一张参考图"和"只是画对了类别"。',
     '我们的方法是纯文本基线的 2.8 倍。'],
    ['③ 取景混淆（framing confound）',
     '七个方法上，身份增益和人脸占画面的面积相关系数 r = 0.949。',
     '也就是说现有的身份分数有很大一部分在测"脸拍得多大"。我们因此同时报告取景，并给出一个与取景无关的替代指标。'],
  ];
  items.forEach(([t, d1, d2], i) => {
    const y = 1.70 + i * 1.72;
    s.addShape('line', { x: M, y, w: WIDE, h: 0, line: { color: C.rule, width: 0.75 } });
    s.addText(t, { x: M, y: y + 0.12, w: WIDE, h: 0.38, fontSize: 17, bold: true, fontFace: F, color: C.blue, isTextBox: true, margin: 0 });
    s.addText(d1 + ' ' + d2, { x: M, y: y + 0.54, w: WIDE, h: 1.02, fontSize: 13.5, fontFace: F, color: C.mute, isTextBox: true, margin: 0, lineSpacing: 20 });
  });
  note(s, `除了方法，论文的分量有一半在这三个测量上。我认为这是比方法本身更能留下来的东西。

第一个，换参考图检验。相似度指标有个根本缺陷：它分不出"一个很弱的适配器"和"一个已经死掉的适配器"。所以我们加了一个它表达不了的控制：把某个区域的参考图换成同类别的另一个实例，其它全部不变，然后问输出跟不跟着变。

这个检验最漂亮的地方是，一个根本不看参考图的基线，在它下面恰好是零——这给了这个测量一个正确性校验，是任何相似度分数都给不了的。在这个检验下，满强度的门控适配器大幅改写了图像，但朝它新拿到的那张参考图移动的幅度，和零没有统计差别，p 等于 0.79。

第二个，身份特异性，区分"复现了这一张"和"只是画对了类别"，我们是纯文本基线的 2.8 倍。

第三个我觉得最有意思。我们发现在七个方法上，身份增益和人脸占画面的面积，相关系数高达 0.949。也就是说现在文献里报的身份分数，有很大一部分其实在测"脸拍得有多大"。我们因此同时报告取景，并给了一个与取景无关的替代指标。`);
}

// ============================================================ 10. limits ==
{
  const s = slide('我们自己报告的边界', '负面结果和正面结果一样写清楚');
  const items = [
    ['面板结构去不掉', '三次尝试全部失败。溯源发现接缝来自两张参考图各自的背景被忠实复现，不是融合算法的问题——把适配器强度设为零、其它全同，接缝就消失了。'],
    ['晚期释放的场景融合效果不可见', '全部 150 个样本上，边界跳变最大只降 5%，肉眼分辨不出。设计意图成立，但效果没有测出来。'],
    ['结论限定在 U-Net + IP-Adapter', '在 FLUX 这类 DiT 骨干上做了一个探针，门控保留了 85.8% 的信号，而 SDXL 上只有 4.7%。所以这个结论可能不迁移到 Transformer 骨干。'],
    ['分辨率是硬约束', '优势只在骨干的原生分辨率下成立。768 下六个区域每个只有 128 像素宽，优势消失。'],
  ];
  items.forEach(([t, d], i) => {
    const y = 1.72 + i * 1.22;
    card(s, M, y, WIDE, 1.06, 'FFFFFF');
    s.addText(t, { x: M + 0.3, y: y + 0.1, w: 4.3, h: 0.86, fontSize: 15.5, bold: true, fontFace: F, color: C.ink, valign: 'middle', isTextBox: true, margin: 0, lineSpacing: 21 });
    s.addText(d, { x: M + 4.75, y: y + 0.1, w: WIDE - 5.05, h: 0.86, fontSize: 13, fontFace: F, color: C.mute, valign: 'middle', isTextBox: true, margin: 0, lineSpacing: 19 });
  });
  note(s, `这一页是我们自己报告的边界。我的判断是，这类论文的可信度主要靠负面结果写得够不够清楚。

第一，面板结构去不掉。我们试了三种办法，全部失败。但后来溯源到了原因：把适配器强度设成零、其它全部一样，接缝就消失了。所以接缝不是融合算法造成的，是两张参考图各自的背景被忠实复现了，两个被复现的场景在中间相遇。这个归因还预测了一个修复方法，而且预测是对的——把主体分割出来放到同一个背景上，接缝就合上了。

第二，晚期释放的场景融合效果，我们测不出来。全部 150 个样本，边界跳变最大只降 5%。设计意图是成立的，但效果没有测出来，我们就这么写。

第三，结论限定在 U-Net 加 IP-Adapter 这个组合上。我们在 FLUX 这种 DiT 骨干上做了个探针，同样是门控，它保留了 85.8% 的信号，而 SDXL 上只有 4.7%。所以这个结论很可能不迁移到 Transformer 骨干，我们明确写了。

第四，分辨率。优势只在骨干的原生分辨率下成立，768 下六个区域每个只有 128 像素宽，优势就没了。`);
}

// ============================================================ 11. status ==
{
  const s = slide('投稿状态', 'Pattern Recognition · Full Length Article · 双盲');
  const done = [
    '稿件 52 页，补充材料 27 页，参考文献 44 篇',
    '单栏双倍行距、Times 10pt、图注 8pt、连续行号 —— 都按指南',
    '全文匿名，零身份信息；Title Page 单独成文件',
    '摘要 255 词、关键词 6 个、CRediT、利益冲突、资助声明齐全',
    'Highlights 5 条、投稿信、四位作者简介全部完成',
  ];
  s.addText('已完成', { x: M, y: 1.72, w: 5.9, h: 0.4, fontSize: 18, bold: true, fontFace: F, color: C.green, isTextBox: true, margin: 0 });
  done.forEach((t, i) => {
    s.addText('✓  ' + t, { x: M, y: 2.18 + i * 0.58, w: 6.1, h: 0.52, fontSize: 13.5, fontFace: F, color: C.ink, isTextBox: true, margin: 0, lineSpacing: 19 });
  });

  s.addText('待办', { x: M + 6.6, y: 1.72, w: 5.3, h: 0.4, fontSize: 18, bold: true, fontFace: F, color: 'A8651B', isTextBox: true, margin: 0 });
  const todo = [
    ['代码仓库还没推', 'GitHub 上默认分支还是上游原作者的代码，Data availability 那句话要等推完才成立'],
    ['页数超限', '期刊要求 20–35 页（含图表参考文献），我们 52 页'],
    ['用户研究未做', '场景连贯性目前没有有效的自动指标，只能靠人评'],
  ];
  todo.forEach(([t, d], i) => {
    const y = 2.18 + i * 1.18;
    s.addText('•  ' + t, { x: M + 6.6, y, w: 5.3, h: 0.36, fontSize: 14.5, bold: true, fontFace: F, color: C.ink, isTextBox: true, margin: 0 });
    s.addText(d, { x: M + 6.95, y: y + 0.36, w: 4.95, h: 0.72, fontSize: 12.5, fontFace: F, color: C.mute, isTextBox: true, margin: 0, lineSpacing: 18 });
  });
  note(s, `投稿状态。

目标是 Pattern Recognition，Full Length Article，双盲评审。

已经完成的：稿件 52 页，补充材料 27 页，参考文献 44 篇，在期刊要求的 35 到 55 之间。排版完全按指南——单栏、双倍行距、Times 10pt、图注 8pt、连续行号。全文匿名，我检查过稿件里零身份信息，所有作者信息单独放在 Title Page 文件里。摘要 255 词，关键词 6 个，CRediT、利益冲突、资助声明都齐。Highlights 五条、投稿信、四位作者简介也都写好了。

待办三件。

第一件最要紧：代码仓库还没推。GitHub 上那个仓库默认分支还是上游原作者的代码，我们这一年的工作一行都没推上去。稿件里的 Data availability 要等推完才成立。

第二件是页数。期刊要求 20 到 35 页，包含图表和参考文献，我们现在 52 页。这个我下一页详细说。

第三件，用户研究还没做。场景连贯性目前没有有效的自动指标——我们试过两种设计都失败了——只能靠人评。`);
}

// ============================================================ 12. pages ===
{
  const s = slide('页数：从 104 压到 52，还超限', '我想听听老师的意见');
  const steps = [
    ['会议双栏版本直接套期刊模板', '104'],
    ['改成期刊要求的 10pt Times、8pt 图注', '77'],
    ['九个实验小节移到补充材料', '61'],
    ['再移两节 + 面板分析 + 删 12 篇引用', '53'],
    ['结论和实验设置压缩', '52'],
  ];
  steps.forEach(([t, n], i) => {
    const y = 1.78 + i * 0.66;
    s.addText(t, { x: M, y, w: 8.6, h: 0.56, fontSize: 15, fontFace: F, color: C.ink, valign: 'middle', isTextBox: true, margin: 0 });
    s.addText(n + ' 页', { x: M + 8.8, y, w: 1.5, h: 0.56, fontSize: 16, bold: i === steps.length - 1, fontFace: F, color: i === steps.length - 1 ? C.red : C.mute, align: 'right', valign: 'middle', isTextBox: true, margin: 0 });
  });
  card(s, M, 5.22, WIDE, 1.42, C.band);
  s.addText([
    { text: '两次认真的文字压缩，回报率都是 9%。', options: { bold: true } },
    { text: ' 搬走整节有效，压句子基本无效。剩下的内容全是论证，再砍就是删结论而不是搬附录。我的建议是按 52 页投，让编辑判断——如果要求压缩，那时候会知道他们在意哪部分，比现在盲砍准。' },
  ], {
    x: M + 0.35, y: 5.4, w: WIDE - 0.7, h: 1.08, fontSize: 14.5, fontFace: F,
    color: C.ink, valign: 'middle', isTextBox: true, margin: 0, lineSpacing: 22,
  });
  note(s, `页数这件事我单独讲一页，因为我想听听老师的意见。

期刊要求 20 到 35 页，而且明确说包含图、表、参考文献、作者简介和附录。

我们从 104 页压到了 52。路径是：先把会议双栏版本套进期刊模板是 104 页；改成期刊要求的 10pt Times 和 8pt 图注，降到 77；把九个实验小节移到补充材料——补充材料是单独上传的，不计入页数——降到 61；再移两节加上面板分析，删掉 12 篇只在相关工作里出现一次的引用，降到 53；最后压缩结论和实验设置，52 页。

这里有个我量出来的经验：认真压缩文字，两次的回报率都是 9%。搬走一个整节是 100% 有效，压句子基本无效。

现在剩下的内容全是论证——注入位点阶梯、主对比、三个测量、局限性。再砍就是删掉一个结论，而不是搬走一段附录了。

我的建议是按 52 页投出去，让编辑判断。如果编辑要求压缩，那时候我们会知道他们在意的是哪一部分，比现在盲砍要准。但这个我想听老师的意见。`);
}

// ============================================================ 13. ask =====
{
  const s = slide('想请老师指导的三个问题', '');
  const qs = [
    ['页数要不要再压？', '按 52 页投，还是先砍到 35 页再投？砍的话需要决定牺牲哪条论证线——目前候选是把 text-stream 这个操作点整体移到补充材料（约 3 页），但那样正文就只报告一个操作点了。'],
    ['用户研究做不做？', '场景连贯性是我们唯一说不清的轴。做用户研究需要 15–20 个人评。不做的话，这一条就只能定性描述。'],
    ['代码公开到什么程度？', '是把完整实验代码和 benchmark 都放出去，还是先放 benchmark 和评测脚本？'],
  ];
  qs.forEach(([t, d], i) => {
    const y = 1.70 + i * 1.72;
    card(s, M, y, WIDE, 1.5, C.blueBg, C.blue);
    s.addText((i + 1) + '.  ' + t, { x: M + 0.35, y: y + 0.14, w: WIDE - 0.7, h: 0.4, fontSize: 18, bold: true, fontFace: F, color: '15283C', isTextBox: true, margin: 0 });
    s.addText(d, { x: M + 0.35, y: y + 0.56, w: WIDE - 0.7, h: 0.82, fontSize: 13.5, fontFace: F, color: C.ink, isTextBox: true, margin: 0, lineSpacing: 20 });
  });
  note(s, `最后，三个想请老师指导的问题。

第一，页数要不要再压。按 52 页投，还是先砍到 35 页再投。如果要砍，需要决定牺牲哪条论证线。目前的候选是把 text-stream 这个操作点整体移到补充材料，大概能省 3 页，但那样正文就只报告一个操作点了，摘要、teaser 图和图 2 都要跟着改。

第二，用户研究做不做。场景连贯性是我们唯一说不清的轴——自动指标我们试了两种设计都失败了。做用户研究需要 15 到 20 个人来评。不做的话，这一条就只能定性描述。

第三，代码公开到什么程度。是把完整的实验代码和 benchmark 都放出去，还是先放 benchmark 和评测脚本。

我汇报完了，请老师指正。`);
}

p.writeFile({ fileName: path.join(OUT, '论文汇报.pptx') })
  .then(() => console.log('written 论文汇报.pptx'))
  .catch((e) => { console.error(e); process.exit(1); });
