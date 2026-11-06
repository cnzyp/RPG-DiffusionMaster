# The Injection Site Determines What Identity Survives in Training-Free Multi-Concept Personalization

Code, benchmarks and evaluation harness for the paper of that name, submitted
to *Pattern Recognition*.

Generating one image containing several **specific** concepts, each given by a
reference photograph, requires a training-free method to impose a spatial
constraint somewhere in the diffusion process. Published methods differ in
*where*, but also in backbone, adapter, layout source and sampler, so the site
itself has never been isolated. Holding all of those fixed and varying only
the site, we find that the construction most training-free work uses — masking
the adapter inside cross-attention — recovers **1.7%** of the identity the same
adapter delivers from the sampling loop, indistinguishable from zero at any
injection strength, while a per-region sub-canvas recovers all of it.

Nothing here is fine-tuned. Every model is frozen.

## What is in this repository

| | |
|---|---|
| `generate_methodA_tau.py` | the method: per-region sub-canvas denoising with an unmodified IP-Adapter, released to one full-canvas pass at a switch point τ |
| `datasets/mixed_eval_v1/` | **Mixed-Eval-v1**, 150 two- and three-region samples across person, object, animal and mixed categories |
| `datasets/scaling_bench/` | **Scaling-Bench**, 90 samples at N = 4, 5 and 6 regions |
| `datasets/ref_swap/` | the reference-swap control: each region's reference exchanged for another instance of the same category |
| `compute_*.py` | the metrics: DINO-ID, leakage margin, CLIP-T, ArcFace, region-level binding accuracy, FID/KID, significance tests |
| `*_gating_probe.py`, `probe_*.py` | the injection-site probes, including the rectified-flow DiT probe |
| `paper_latex/` | the manuscript source |

The dataset directories hold **manifests**, not images: each sample names its
prompts, its split ratio and the reference photographs it uses. The reference
images come from their original public sources under the terms of those
sources, and `datasets/README.md` says where.

## Reproducing the paper

```bash
pip install -r requirements.txt

# render the method on Mixed-Eval-v1 at the paper's default
python generate_methodA_tau.py --tau 0.7

# score it
python compute_clip_specificity.py      # DINO-ID and the leakage margin
python compute_face_metrics.py  --variant ours_tau070_1024
python compute_binding_accuracy.py      # region-level binding
python compute_significance.py          # paired tests over the same samples
```

Every number in the paper is at 1024×1024; the variant names carrying the
`_1024` suffix are the ones the main table reports. At 768 the comparison
with a trained model is substantially closer, for a reason the paper
identifies, so the resolution a number was computed at always matters.

## The user study

Not included. It has not been run yet, and its build script would regenerate
the condition assignment, so publishing it would tell a prospective rater
which side is which. It will be added once the study is complete. No user
study is reported in the paper.

## Built on RPG

This work builds on **RPG** (Yang et al., *Mastering Text-to-Image Diffusion:
Recaptioning, Planning, and Generating with Multimodal LLMs*, ICML 2024) and
this repository is a fork of
[YangLing0818/RPG-DiffusionMaster](https://github.com/YangLing0818/RPG-DiffusionMaster).
RPG supplies the MLLM region planner and the regional diffusion pipeline; the
paper's contribution is where the reference is injected relative to that
layout, and what each site costs. The upstream project's own README is
preserved as [`README_RPG.md`](README_RPG.md), and its license applies to the
code it covers.

```bibtex
@inproceedings{yang2024mastering,
  title     = {Mastering Text-to-Image Diffusion: Recaptioning, Planning,
               and Generating with Multimodal LLMs},
  author    = {Yang, Ling and Yu, Zhaochen and Meng, Chenlin and Xu, Minkai
               and Ermon, Stefano and Cui, Bin},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}
```

The method also uses IP-Adapter (Ye et al., 2023) unmodified, and is evaluated
against MS-Diffusion (Wang et al., ICLR 2025) and Conceptrol (He et al., 2025)
run from their authors' own code. DreamBench-2 is built from DreamBench++
(Peng et al., ICLR 2025).
