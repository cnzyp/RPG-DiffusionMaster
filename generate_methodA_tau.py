"""
Method A v3 — sub-canvas identity locking + LATE GLOBAL RELEASE.

Identity/coherence decoupling across timesteps:
  * early  (step_idx <  tau*STEPS): per-region sub-canvas denoising with
      full-strength IP-Adapter  -> locks each reference's identity
      (identical to Method A v2)
  * late   (step_idx >= tau*STEPS): ONE full-canvas pass conditioned on the
      GLOBAL prompt (+ all reference embeds) -> unifies lighting, background,
      perspective and seams, removing the "pasted panels" look.

tau = 1.0 reproduces Method A v2 exactly (pure sub-canvas).
tau = 0.0 is pure full-canvas global generation (no regional identity).

tau = 0.7 is the paper default and reproduces every "Ours" row in the paper.
It was located by intervention: identity is flat between tau=1 and tau=0.5
while region-level binding falls away below 0.7, and the two axes not
flattening together is what fixes the value.

No scene-coherence benefit is claimed for any tau.  Two automatic measures
of scene coherence were attempted and both failed their own validation, and
across all 150 samples the release changes the boundary step by at most 5%,
which is not visible.  At tau=0.7 the global phase repaints 3.7/255 of the
canvas on average; that is reported as an observation, not as a benefit.

Output goes to outputs/mixed_eval_v1/ours_tau<tau*100>/, so each tau keeps
its own directory and existing renders are skipped rather than overwritten.

Usage:
    python generate_methodA_tau.py                      # tau = 0.7, as in the paper
    python generate_methodA_tau.py --tau 0.5
    python generate_methodA_tau.py --tau 0.7 --limit 20 --out-suffix _pilot
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir(Path(__file__).parent)
torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("--tau", type=float, default=0.7,
                help="fraction of steps in sub-canvas mode before the global "
                     "release; 0.7 is the paper's default, located by "
                     "intervention.  No scene-coherence benefit is claimed "
                     "for any value: two automatic measures of it were "
                     "attempted and both failed their own validation.")
ap.add_argument("--base-ckpt", default="",
                help="path to a single-file checkpoint to use instead of SDXL-base.  RPG's README recommends a photoreal CivitAI checkpoint for photorealism; everything in this study was rendered on SDXL-base, so this makes the backbone a controlled variable rather than an assumption.")
ap.add_argument("--limit", type=int, default=0, help="only first N samples (0=all)")
ap.add_argument("--out-suffix", default="", help="suffix appended to output dir name")
ap.add_argument("--dataset", default="datasets/mixed_eval_v1/mixed_eval_v1.json")
ap.add_argument("--size", type=int, default=768,
                help="square canvas side. At 768 with N=2 each sub-canvas is "
                     "384 wide, below the range SDXL was trained on; 1024 "
                     "gives each region 512 and is the backbone's native size")
ap.add_argument("--layout", choices=["strips", "inset"], default="strips",
                help="'strips' (the paper) partitions the canvas into "
                     "adjacent full-height regions, so no part of it is ever "
                     "denoised as background.  'inset' shrinks each region to "
                     "a tight box and carries everything outside the boxes "
                     "with one global pass conditioned on the full prompt")
ap.add_argument("--inset-w", type=float, default=0.8,
                help="inset layout: fraction of its strip's width each box keeps")
ap.add_argument("--inset-v", type=float, default=0.15,
                help="inset layout: vertical margin above and below each box")
ap.add_argument("--bg-prompt", default="",
                help="inset layout: prompt for the pass that carries the area "
                     "outside the boxes.  Empty uses the sample's global "
                     "prompt, which names the subjects and therefore asks the "
                     "background pass to render them too")
ap.add_argument("--mask-refs", action="store_true",
                help="segment each reference and composite it on neutral grey "
                     "before encoding.  The text-only control shows the "
                     "boundary is not produced by the fusion rule but by the "
                     "references disagreeing about scene; removing their "
                     "scenes removes what they disagree about")
ap.add_argument("--init-from", default="",
                help="directory of stage-1 images, one <sample_id>.png each.  "
                     "Instead of starting from noise, start from that image "
                     "noised back to --strength of the trajectory.  A "
                     "text-only render already holds a coherent scene with "
                     "its own lighting and depth of field; the adapter then "
                     "only has to write identity into it, rather than compose "
                     "a photograph and carry an identity at the same time.")
ap.add_argument("--strength", type=float, default=0.75,
                help="with --init-from, the fraction of the trajectory to "
                     "re-run.  1.0 discards the stage-1 image entirely; low "
                     "values keep its composition but leave the adapter too "
                     "few steps to write an identity (Section: identity is "
                     "committed in the first third).")
ap.add_argument("--warm-start", type=float, default=0.0,
                help="fraction of steps to run as ONE full-canvas text-only "
                     "pass on the global prompt before the regional phase "
                     "begins.  Layout is decided in the first fifth of the "
                     "trajectory; without this the regional passes own that "
                     "window and each invents its own scene from its own "
                     "reference.  A warm start spends it on the global prompt "
                     "instead, so the sub-canvases inherit one scene.")
ap.add_argument("--ref-feather", type=float, default=0.0,
                help="with --mask-refs, blur the segmentation alpha by this "
                     "radius so the subject fades into the backdrop instead "
                     "of being pasted onto it with a hard silhouette edge.")
ap.add_argument("--shared-bg", type=float, default=0.0,
                help="with --mask-refs, composite every subject onto ONE "
                     "shared backdrop instead of flat grey: the first "
                     "reference blurred by this radius, which reads as a "
                     "defocused scene carrying no competing subject.  The "
                     "adapter then transfers the same context for every "
                     "region, so the regions have nothing to disagree about "
                     "and the result still has a scene.  0 keeps flat grey")
ap.add_argument("--ip-scale", type=float, default=0.8,
                help="adapter strength. 0 makes each sub-canvas text-only, "
                     "which isolates whether the boundary needs a reference "
                     "to appear or is a property of the fusion rule itself")
ap.add_argument("--quantize-mask", type=float, default=0.0,
                help="StreamMultiDiffusion's quantised mask, adapted to a "
                     "tiling layout.  At each step the feathered weight is "
                     "thresholded against the sampler's own noise level, so "
                     "the overlap band belongs to a single region while the "
                     "noise is high and is shared only once the two sides "
                     "already agree on global structure.  0 disables it; the "
                     "value scales the threshold")
ap.add_argument("--fusion-mask", choices=["box", "subject"], default="box",
                help="shape of the blend weight. 'box' (the paper) changes "
                     "weight along a column, so the seam it leaves is a "
                     "straight line. 'subject' uses the region's own "
                     "silhouette, segmented from a first pass, and carries "
                     "everything outside every silhouette with one global "
                     "pass -- the sub-canvas windows are unchanged, so the "
                     "identity each region can write is unchanged")
ap.add_argument("--mask-dir", default="masks/subject_1024",
                help="where build_subject_masks.py wrote the silhouettes")
ap.add_argument("--overlap", type=int, default=8,
                help="latent columns each sub-canvas extends into its "
                     "neighbours.  This is the receptive field of a regional "
                     "pass: at 0 it sees only its own region and must invent "
                     "a background, at lw/N it sees the whole canvas and the "
                     "variant becomes full-canvas noise fusion.  The paper's "
                     "two fusion rungs are the two ends of this one knob")
ap.add_argument("--size-cond", choices=["region", "crop"], default="region",
                help="what SDXL's size conditioning tells each sub-canvas it "
                     "is painting. 'region' (the paper) declares a complete "
                     "image of the sub-canvas's own size, so every region "
                     "composes its own framing and its own background. "
                     "'crop' declares the same sub-canvas as a crop of the "
                     "full canvas at its true offset, which is the semantics "
                     "SDXL's crop conditioning was trained for")
args = ap.parse_args()

TAU = args.tau
tag = f"ours_tau{int(round(TAU*100)):03d}{args.out_suffix}"
OUT = Path(f"outputs/mixed_eval_v1/{tag}/images")
OUT.mkdir(parents=True, exist_ok=True)

IP_SCALE = args.ip_scale
H = W    = args.size
STEPS    = 30
GUID     = 7.5
OVERLAP  = args.overlap
NEG      = "low quality, blurry, distorted"
SEED     = 42
device, dtype = "cuda", torch.float16

samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["samples"]
if args.limit:
    samples = samples[:args.limit]
todo = [s for s in samples if not (OUT / f"{s['sample_id']}.png").exists()]
print(f"tau={TAU} -> {tag} | total {len(samples)} | remaining {len(todo)}", flush=True)
if not todo:
    print("nothing to do"); raise SystemExit(0)

print(f"Loading {'SDXL-base' if not args.base_ckpt else args.base_ckpt} "
      "+ IP-Adapter Plus (ViT-H)...", flush=True)
if args.base_ckpt:
    pipe = StableDiffusionXLPipeline.from_single_file(
        args.base_ckpt, torch_dtype=dtype, use_safetensors=True)
else:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype, use_safetensors=True, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True)
pipe.to(device)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                     weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                     image_encoder_folder="models/image_encoder")
pipe.set_ip_adapter_scale(IP_SCALE)
pipe.image_encoder.to(device=device, dtype=dtype)
pipe.upcast_vae()

vae_sf = pipe.vae_scale_factor
lh, lw = H // vae_sf, W // vae_sf
do_cfg = GUID > 1.0
proj_dim = pipe.text_encoder_2.config.projection_dim


_REMBG = None


def load_ref(path, backdrop=None):
    """Reference image, optionally with its scene removed.

    IP-Adapter transfers whatever the reference contains, background included.
    When two regions are driven by photographs shot in different places, each
    sub-canvas faithfully reproduces its own, and the disagreement is what the
    boundary shows.  Compositing the segmented subject onto neutral grey
    leaves the adapter nothing to disagree about.
    """
    im = Image.open(path).convert("RGB")
    if not args.mask_refs:
        return im
    global _REMBG
    if _REMBG is None:
        from rembg import new_session
        _REMBG = new_session("u2net")
    from rembg import remove
    m = np.asarray(remove(im, session=_REMBG, only_mask=True),
                   np.float32) / 255.0
    if m.mean() < 0.04:          # segmentation found nothing; keep the original
        return im
    a = np.asarray(im, np.float32)
    if backdrop is not None:
        bg = np.asarray(backdrop.resize(im.size), np.float32)
    else:
        bg = np.full_like(a, 127.0)
    if args.ref_feather > 0:
        # A binary silhouette composites as a cut-out, and the adapter
        # transfers the cut-out edge along with the subject.  Softening the
        # alpha spreads the handover over a band instead of a step.
        from PIL import ImageFilter as _IF
        m = np.asarray(
            Image.fromarray((m * 255).astype(np.uint8))
            .filter(_IF.GaussianBlur(args.ref_feather)),
            np.float32) / 255.0
    m3 = m[..., None]
    return Image.fromarray((a * m3 + bg * (1 - m3)).astype(np.uint8))


def region_cols(split, lw, overlap):
    parts = [float(x) for x in split.split(",")]; tot = sum(parts)
    cum = [sum(parts[:i]) / tot for i in range(len(parts) + 1)]
    spans = []
    for i in range(len(parts)):
        c0 = int(round(cum[i] * lw)); c1 = int(round(cum[i + 1] * lw))
        e0 = max(0, c0 - (overlap if i > 0 else 0))
        e1 = min(lw, c1 + (overlap if i < len(parts) - 1 else 0))
        spans.append((e0, e1))
    return spans


def feather_weight(w, left_ov, right_ov):
    x = torch.ones(w, device=device, dtype=dtype)
    if left_ov > 0:
        x[:left_ov] = torch.linspace(0, 1, left_ov + 2, device=device, dtype=dtype)[1:-1]
    if right_ov > 0:
        x[w - right_ov:] = torch.linspace(1, 0, right_ov + 2, device=device, dtype=dtype)[1:-1]
    return x.view(1, 1, 1, w)


# scene-only prompts for the pass outside the boxes.  A background prompt that
# names the subjects makes that pass try to render them, which is what the
# sample's own global prompt does; these describe a place and nothing else.
BG_BY_CATEGORY = {
    "person": "a photograph of an empty sunlit park, soft natural light, "
              "blurred green foliage, no people",
    "object": "a photograph of an empty light wooden table, soft diffuse "
              "studio light, plain grey background",
    "mixed":  "a photograph of an empty bright room, soft natural window "
              "light, plain wall, no people",
}


def strip_concepts(s):
    """The sample's own prompt with its subject nouns removed.

    DreamBench-2 prompts name a setting as well as the subjects---"a man and
    a girl sitting together in a sunlit park"---and the setting is exactly
    what the background pass should paint.  Removing the concept names, the
    articles that precede them and any conjunction left dangling turns the
    prompt into a description of the scene alone.  Falls back to the
    category default when the sample carries no concept list.
    """
    import re
    names = s.get("concept_names")
    if not names:
        return BG_BY_CATEGORY[s["metadata"]["category"]]
    p = s["full_prompt"]
    for n in sorted(names, key=len, reverse=True):
        p = re.sub(r"(a|an|the)\s+" + re.escape(n) + r"", "", p,
                   flags=re.I)
        p = re.sub(r"" + re.escape(n) + r"", "", p, flags=re.I)
    p = re.sub(r"\s*(and|next to|alongside|with|together)\s*", " ", p,
               flags=re.I)
    p = re.sub(r"\s{2,}", " ", p).strip(" ,;")
    return p if len(p.split()) >= 3 else BG_BY_CATEGORY[s["metadata"]["category"]]


def subject_weight(sid, i, lh, lw, blur=1.5):
    """Region i's silhouette at latent resolution, softened at its edge.

    Returns None when the mask is missing, so the caller can fall back to the
    rectangle rather than silently dropping the region.
    """
    p = Path(args.mask_dir) / f"{sid}_r{i}.png"
    if not p.exists():
        return None
    m = Image.open(p).convert("L").resize((lw, lh), Image.BILINEAR)
    a = np.asarray(m, np.float32) / 255.0
    if blur > 0:
        a = gaussian_filter(a, blur)
    a = np.clip(a, 0.0, 1.0)
    return torch.from_numpy(a).to(device=device, dtype=dtype)[None, None]


def region_boxes(split, lh, lw, inset_w, inset_v):
    """Tight, non-adjacent boxes: each strip shrunk in width and in height.

    The strips of `region_cols` tile the canvas, so nothing is ever denoised
    as background and each region composes its own.  These boxes leave a gap
    between neighbours and a margin above and below, and the region outside
    them is carried by a single global pass (Section: layout ablation).
    """
    parts = [float(x) for x in split.split(",")]
    tot = sum(parts)
    cum = [sum(parts[:i]) / tot for i in range(len(parts) + 1)]
    r0 = int(round(inset_v * lh))
    r1 = int(round((1.0 - inset_v) * lh))
    out = []
    for i in range(len(parts)):
        lo, hi = cum[i], cum[i + 1]
        mid, half = (lo + hi) / 2, (hi - lo) * inset_w / 2
        c0 = int(round((mid - half) * lw))
        c1 = int(round((mid + half) * lw))
        out.append((r0, r1, c0, c1))
    return out


def feather_2d(h, w, ov):
    """Ramp in from all four sides over `ov` latent units."""
    ry = torch.ones(h, device=device, dtype=dtype)
    rx = torch.ones(w, device=device, dtype=dtype)
    k = min(ov, h // 2)
    if k > 0:
        ramp = torch.linspace(0, 1, k + 2, device=device, dtype=dtype)[1:-1]
        ry[:k] = ramp
        ry[h - k:] = ramp.flip(0)
    k = min(ov, w // 2)
    if k > 0:
        ramp = torch.linspace(0, 1, k + 2, device=device, dtype=dtype)[1:-1]
        rx[:k] = ramp
        rx[w - k:] = ramp.flip(0)
    return (ry.view(1, 1, h, 1) * rx.view(1, 1, 1, w))


def generate(s):
    region_prompts = [p.strip() for p in s["regional_prompts"].split("BREAK")]
    paths = s["reference_image_paths"]
    backdrop = None
    if args.mask_refs and args.shared_bg > 0:
        from PIL import ImageFilter
        backdrop = (Image.open(paths[0]).convert("RGB")
                    .filter(ImageFilter.GaussianBlur(args.shared_bg)))
    refs = [load_ref(p, backdrop) for p in paths]
    spans = region_cols(s["split_ratio"], lw, OVERLAP)
    n = len(refs)

    boxes = (region_boxes(s["split_ratio"], lh, lw, args.inset_w, args.inset_v)
             if args.layout == "inset" else None)

    # ---- per-region conditioning (early phase) ----
    reg = []
    for k, (rp, ref) in enumerate(zip(region_prompts, refs)):
        if boxes is None:
            e0, e1 = spans[k]
            r0, r1 = 0, lh
        else:
            r0, r1, e0, e1 = boxes[k]
        rw_px, rh_px = (e1 - e0) * vae_sf, (r1 - r0) * vae_sf
        pe, npe, ppe, nppe = pipe.encode_prompt(
            prompt=rp, prompt_2=None, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg, negative_prompt=NEG)
        ipe = pipe.prepare_ip_adapter_image_embeds([ref], None, device, 1, do_cfg)
        ipe = [e.to(device=device, dtype=dtype) for e in ipe]
        if args.size_cond == "crop":
            # this sub-canvas is a crop of the full canvas, taken at its true
            # column offset - the conditioning SDXL was trained with
            tid = pipe._get_add_time_ids(
                (H, W), (0, e0 * vae_sf), (H, W), dtype=dtype,
                text_encoder_projection_dim=proj_dim).to(device)
        else:
            # a complete image of the region's own size: each region is asked
            # to compose its own framing, hence its own background
            tid = pipe._get_add_time_ids(
                (rh_px, rw_px), (0, 0), (rh_px, rw_px), dtype=dtype,
                text_encoder_projection_dim=proj_dim).to(device)
        sw = (subject_weight(s["sample_id"], k, lh, lw)
              if args.fusion_mask == "subject" else None)
        if sw is not None:
            # the window is unchanged; only the shape of the blend weight is
            fw = sw[:, :, r0:r1, e0:e1]
        elif boxes is None:
            fw = feather_weight(e1 - e0,
                                OVERLAP if k > 0 else 0,
                                OVERLAP if k < n - 1 else 0)
        else:
            fw = feather_2d(r1 - r0, e1 - e0, OVERLAP)
        reg.append(dict(pe=pe, npe=npe, ppe=ppe, nppe=nppe, ipe=ipe, tid=tid,
                        span=(e0, e1), rows=(r0, r1), fw=fw))

    # ---- global conditioning (late phase): full prompt + ALL references ----
    g_pe, g_npe, g_ppe, g_nppe = pipe.encode_prompt(
        prompt=s["full_prompt"], prompt_2=None, device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg, negative_prompt=NEG)
    # single loaded IP adapter -> encode each ref separately and average the
    # conditional embeds, so the late global pass still carries every identity
    per_ref = []
    for ref in refs:
        e = pipe.prepare_ip_adapter_image_embeds([ref], None, device, 1, do_cfg)[0]
        per_ref.append(e.to(device=device, dtype=dtype))
    g_ipe = [torch.stack(per_ref, dim=0).mean(dim=0)]
    g_tid = pipe._get_add_time_ids((H, W), (0, 0), (H, W), dtype=dtype,
                                   text_encoder_projection_dim=proj_dim).to(device)

    # ---- background conditioning (early phase, inset layout only) ----
    if (boxes is not None or args.fusion_mask == "subject") and args.bg_prompt:
        if args.bg_prompt == "strip":
            # keep the scene the prompt actually describes and drop the
            # subjects from it, so the background pass paints the setting
            # rather than attempting the people who belong in the boxes
            bp = strip_concepts(s)
        elif args.bg_prompt == "auto":
            bp = BG_BY_CATEGORY[s["metadata"]["category"]]
        else:
            bp = args.bg_prompt
        b_pe, b_npe, b_ppe, b_nppe = pipe.encode_prompt(
            prompt=bp, prompt_2=None, device=device,
            num_images_per_prompt=1, do_classifier_free_guidance=do_cfg,
            negative_prompt=NEG)
    else:
        b_pe, b_npe, b_ppe, b_nppe = g_pe, g_npe, g_ppe, g_nppe

    pipe.scheduler.set_timesteps(STEPS, device=device)
    g = torch.Generator(device=device).manual_seed(SEED)
    latents = (torch.randn((1, pipe.unet.config.in_channels, lh, lw),
                           generator=g, device=device, dtype=dtype)
               * pipe.scheduler.init_noise_sigma).to(dtype)

    switch_at = int(round(TAU * STEPS))   # first step index using global mode

    start_i = 0
    if args.init_from:
        src = Path(args.init_from) / f"{s['sample_id']}.png"
        if not src.exists():
            raise FileNotFoundError(f"stage-1 image missing: {src}")
        init = Image.open(src).convert("RGB").resize((W, H))
        px = (torch.from_numpy(np.asarray(init, np.float32) / 127.5 - 1.0)
              .permute(2, 0, 1)[None].to(device))
        # upcast_vae() leaves post_quant_conv and the decoder entry in
        # fp16 while the encoder is fp32, so the decode-side dtype is
        # the wrong one to encode with.
        vd = pipe.vae.encoder.conv_in.weight.dtype
        with torch.no_grad():
            lat0 = (pipe.vae.encode(px.to(vd)).latent_dist.sample(generator=g)
                    * pipe.vae.config.scaling_factor).to(dtype)
        start_i = int(round((1.0 - args.strength) * STEPS))
        start_i = max(0, min(start_i, STEPS - 1))
        noise0 = torch.randn(lat0.shape, generator=g, device=device, dtype=dtype)
        latents = pipe.scheduler.add_noise(
            lat0, noise0, pipe.scheduler.timesteps[start_i:start_i + 1]).to(dtype)
    warm_at = int(round(args.warm_start * STEPS))   # regional phase starts here
    if warm_at >= switch_at:
        warm_at = 0                       # never let the warm start eat the whole run

    sig = getattr(pipe.scheduler, "sigmas", None)
    for i, t in enumerate(pipe.scheduler.timesteps):
        if i < start_i:
            continue                      # stage 1 already covered these
        # noise level in [0,1], monotonically decreasing, so the quantised
        # mask opens up as the trajectory settles
        beta_t = (float(sig[i] / sig[0]) if sig is not None and len(sig) > i
                  and float(sig[0]) > 0 else 1.0 - i / max(1, STEPS - 1))
        full_scaled = pipe.scheduler.scale_model_input(
            torch.cat([latents] * 2) if do_cfg else latents, t)

        if i < warm_at:
            # ---------- WARM: one full-canvas text-only pass (layout) ----------
            # The adapter is off, so this phase is the global prompt deciding
            # where things go and what the scene is, on one canvas.  The
            # regional phase then starts from a latent that already holds it.
            pe = torch.cat([g_npe, g_pe]) if do_cfg else g_pe
            te = torch.cat([g_nppe, g_ppe]) if do_cfg else g_ppe
            ti = torch.cat([g_tid, g_tid]) if do_cfg else g_tid
            added = {"text_embeds": te, "time_ids": ti,
                     "image_embeds": [torch.zeros_like(g_ipe[0])]}
            npred = pipe.unet(full_scaled, t, encoder_hidden_states=pe,
                              added_cond_kwargs=added, return_dict=False)[0]
            if do_cfg:
                nu, nt = npred.chunk(2); npred = nu + GUID * (nt - nu)
            noise = npred.to(dtype)
        elif i < switch_at:
            # ---------- EARLY: per-region sub-canvas (identity locking) ----------
            acc = torch.zeros_like(latents)
            wsum = torch.zeros((1, 1, lh, lw), device=device, dtype=dtype)
            for r in reg:
                e0, e1 = r["span"]; r0, r1 = r["rows"]
                sub = full_scaled[:, :, r0:r1, e0:e1]
                pe = torch.cat([r["npe"], r["pe"]]) if do_cfg else r["pe"]
                te = torch.cat([r["nppe"], r["ppe"]]) if do_cfg else r["ppe"]
                ti = torch.cat([r["tid"], r["tid"]]) if do_cfg else r["tid"]
                added = {"text_embeds": te, "time_ids": ti, "image_embeds": r["ipe"]}
                npred = pipe.unet(sub, t, encoder_hidden_states=pe,
                                  added_cond_kwargs=added, return_dict=False)[0]
                if do_cfg:
                    nu, nt = npred.chunk(2); npred = nu + GUID * (nt - nu)
                w = r["fw"]
                if args.quantize_mask > 0:
                    # 1[w > beta(t)] with beta falling as the sampler denoises:
                    # early the band is owned outright, late it is blended
                    w = (w > (args.quantize_mask * beta_t)).to(dtype)
                r["last"] = npred
                acc[:, :, r0:r1, e0:e1] += npred * w
                wsum[:, :, r0:r1, e0:e1] += w
            if boxes is None and args.fusion_mask != "subject":
                if args.quantize_mask > 0:
                    # where the threshold left a pixel unclaimed, give it to the
                    # region with the largest soft weight rather than to noise
                    bare = (wsum <= 1e-6)
                    if bool(bare.any()):
                        best = torch.zeros_like(acc)
                        top = torch.full_like(wsum, -1.0)
                        for r in reg:
                            e0, e1 = r["span"]; r0, r1 = r["rows"]
                            wv = r["fw"].expand(1, 1, r1 - r0, e1 - e0)
                            sel = wv > top[:, :, r0:r1, e0:e1]
                            top[:, :, r0:r1, e0:e1] = torch.where(
                                sel, wv, top[:, :, r0:r1, e0:e1])
                            best[:, :, r0:r1, e0:e1] = torch.where(
                                sel, r["last"], best[:, :, r0:r1, e0:e1])
                        acc = torch.where(bare, best, acc)
                        wsum = torch.where(bare, torch.ones_like(wsum), wsum)
                noise = (acc / wsum.clamp(min=1e-6)).to(dtype)
            else:
                # the regions do not tile the canvas -- either because the
                # boxes are inset or because the weights follow silhouettes --
                # so everything outside them is carried by one full-canvas pass
                # on the global prompt with the adapter switched off.  The background is then shared by
                # construction instead of being invented once per region.
                pe = torch.cat([b_npe, b_pe]) if do_cfg else b_pe
                te = torch.cat([b_nppe, b_ppe]) if do_cfg else b_ppe
                ti = torch.cat([g_tid, g_tid]) if do_cfg else g_tid
                added = {"text_embeds": te, "time_ids": ti,
                         "image_embeds": [torch.zeros_like(g_ipe[0])]}
                bg = pipe.unet(full_scaled, t, encoder_hidden_states=pe,
                               added_cond_kwargs=added, return_dict=False)[0]
                if do_cfg:
                    nu, nt = bg.chunk(2); bg = nu + GUID * (nt - nu)
                w = wsum.clamp(max=1.0)
                noise = (bg * (1.0 - w) + acc).to(dtype)
        else:
            # ---------- LATE: single full-canvas pass (global coherence) ----------
            pe = torch.cat([g_npe, g_pe]) if do_cfg else g_pe
            te = torch.cat([g_nppe, g_ppe]) if do_cfg else g_ppe
            ti = torch.cat([g_tid, g_tid]) if do_cfg else g_tid
            added = {"text_embeds": te, "time_ids": ti, "image_embeds": g_ipe}
            npred = pipe.unet(full_scaled, t, encoder_hidden_states=pe,
                              added_cond_kwargs=added, return_dict=False)[0]
            if do_cfg:
                nu, nt = npred.chunk(2); npred = nu + GUID * (nt - nu)
            noise = npred.to(dtype)

        latents = pipe.scheduler.step(noise, t, latents, return_dict=False)[0].to(dtype)

    lat = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


for i, s in enumerate(todo):
    sid = s["sample_id"]
    try:
        generate(s).save(OUT / f"{sid}.png")
        torch.cuda.empty_cache()
        print(f"[{i+1:3d}/{len(todo)}] OK {sid}", flush=True)
    except Exception as e:
        import traceback; print(f"[{i+1:3d}/{len(todo)}] ERR {sid}: {e}", flush=True)
        traceback.print_exc()

print("DONE ->", OUT, flush=True)
