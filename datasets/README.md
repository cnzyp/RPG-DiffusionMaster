# Benchmarks and manifests

These directories hold **manifests**, not images. Each sample names its
prompts, its split ratio and the reference photographs it uses; the
photographs themselves come from their original public sources, under the
terms of those sources, and are not redistributed here.

## Released with the paper

### `mixed_eval_v1/` — Mixed-Eval-v1

150 two- and three-region samples across person, object, animal and mixed
category pairs. The benchmark the main results table is computed on. Each
entry carries `sample_id`, `full_prompt`, `regional_prompts`, `split_ratio`
and one `reference_image_path` per region.

References are drawn from DreamBench++ (Peng et al., ICLR 2025) and from
public photographs named in each entry's `metadata`.

### `scaling_bench/` — Scaling-Bench

90 samples at N = 4, 5 and 6 regions, 30 each, extending the same task past
the three regions Mixed-Eval-v1 covers. Used for the per-region degradation
rates and the binding-accuracy lead as regions are added.

### `ref_swap/` — the reference-swap control

`ref_orig.json` and `ref_swap.json` are the same prompts and layouts with
each region's reference exchanged for another instance of the same category;
`swap_key.json` records which reference replaced which. The pairing is what
makes the test work: a reference-free baseline sits at exactly zero by
construction, so any non-zero response is attributable to the reference.

## Benchmarks built from other people's work

### `dreambench2/` — DreamBench-2

150 two-concept samples over 30 concept pairs (8 human-human, 8 human-animal,
8 human-object, 6 animal-object) with 5 prompts each, constructed from
DreamBench++ references. Used for the cross-dataset check in the
supplementary material.

### `dreambench_plus/`

Manifests for DreamBench++ itself.
Source: https://github.com/yuangpeng/dreambench_plus

> Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai,
> Chunrui Han, Zheng Ge, Xiangyu Zhang and Shu-Tao Xia.
> *DreamBench++: A Human-Aligned Benchmark for Personalized Image
> Generation.* ICLR 2025.

### `t2i_compbench/`

Compositional prompts from T2I-CompBench.
Source: https://github.com/Karine-Huang/T2I-CompBench

## Earlier manifests

`personalized_comp/`, `personalized_comp_v2/`, `scene_bench_v1/`,
`shared_scene/` and the underscore-prefixed files are from development and
from experiments reported in the supplementary material. They are kept so
that every script in the repository runs, but Mixed-Eval-v1 and
Scaling-Bench are the two benchmarks the paper releases.

## Citing

The paper's own bibliography in `paper_latex/refs.bib` carries the verified
entry for every work named here. Please cite from there rather than from a
summary.
