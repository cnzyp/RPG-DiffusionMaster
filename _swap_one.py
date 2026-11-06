import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; sys.path.insert(0,".")
from PIL import Image
from experiments.config import ExperimentConfig, ABLATION_VARIANTS
from experiments.generators import create_generator
which=sys.argv[1]; out=sys.argv[2]
cfg=ExperimentConfig(); cfg.num_inference_steps=30; cfg.height=cfg.width=768; cfg.guidance_scale=7.5
v=next(x for x in ABLATION_VARIANTS if x.name=="ours"); v.ip_scale=0.8
gen=create_generator("ours",cfg,v); gen.setup(); p=gen.pipe
p._ip_router_enabled=False; p._ip_disable_up=False; p._ip_layer_scales=lambda _n:1.0
p._ip_timestep_mode="cosine"; p._ip_timestep_start=0.0; p._ip_timestep_end=0.0
p._ip_norm_cap_ratio=0.0; p._ip_entropy_gate=0.0; p._ip_use_value_only=False
S=json.load(open("datasets/mixed_eval_v1/mixed_eval_v1.json",encoding="utf-8"))["samples"]
s=[x for x in S if x["metadata"]["category"]=="object"][0]
alt=[x for x in S if x["metadata"]["category"]=="object"][1]
refs=[Image.open(x).convert("RGB") for x in s["reference_image_paths"]]
if which=="B": refs[0]=Image.open(alt["reference_image_paths"][0]).convert("RGB")
img=gen.generate(prompt=s["full_prompt"],reference_images=refs,split_ratio=s["split_ratio"],
    regional_prompts=s["regional_prompts"],seed=42,num_inference_steps=30,height=768,width=768,guidance_scale=7.5)
img.save(out); print("saved",out)
