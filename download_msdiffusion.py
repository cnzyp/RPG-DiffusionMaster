"""
Fetch the two weight sets MS-Diffusion needs that are not already cached.

SDXL-base-1.0 is already in the local HF cache and is not re-downloaded.
CLIP-G ships two full copies of the weights (a transformers-format shard pair
and an open_clip single file); only the transformers pair is pulled, which
halves the download.
"""

from huggingface_hub import snapshot_download

print("1/2  doge1516/MS-Diffusion  (ms_adapter.bin, ~1.84 GB)", flush=True)
p = snapshot_download(
    repo_id="doge1516/MS-Diffusion",
    allow_patterns=["ms_adapter.bin", "*.json"],
)
print("     ->", p, flush=True)

print("2/2  laion/CLIP-ViT-bigG-14-laion2B-39B-b160k  (~10.2 GB)", flush=True)
q = snapshot_download(
    repo_id="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    allow_patterns=[
        "config.json",
        "preprocessor_config.json",
        "pytorch_model-*.safetensors",
        "*.index.json",
        "tokenizer*.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ],
)
print("     ->", q, flush=True)
print("\nDONE")
