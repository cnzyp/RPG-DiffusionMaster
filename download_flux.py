"""
Fetch FLUX.1-schnell in diffusers layout, plus the OminiControl subject LoRA.

The repo carries the weights twice: a single-file checkpoint
(flux1-schnell.safetensors, 23.8 GB) for the reference implementation and a
sharded diffusers layout.  Only the latter is pulled, along with the original
ae.safetensors being skipped in favour of vae/.  34.1 GB in total.
"""

import time

from huggingface_hub import snapshot_download


def fetch(repo_id, patterns, label, tries=12):
    """snapshot_download, retried.

    Multi-GB transfers here drop mid-stream often enough that a single
    attempt is not usable; the partial blobs are kept on disk, so each retry
    resumes rather than restarting.
    """
    for attempt in range(1, tries + 1):
        try:
            path = snapshot_download(repo_id=repo_id, allow_patterns=patterns,
                                     max_workers=4)
            print(f"     -> {path}", flush=True)
            return path
        except Exception as e:
            if attempt == tries:
                raise
            wait = min(30 * attempt, 180)
            print(f"     [{label}] attempt {attempt} failed "
                  f"({type(e).__name__}); resuming in {wait}s", flush=True)
            time.sleep(wait)


print("1/2  black-forest-labs/FLUX.1-schnell  (~34.1 GB, diffusers layout)",
      flush=True)
fetch("black-forest-labs/FLUX.1-schnell", [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "transformer/*",
    "vae/*",
], "flux")

print("2/2  Yuanshi/OminiControl  (subject_512 LoRA, ~0.05 GB)", flush=True)
fetch("Yuanshi/OminiControl", ["omini/subject_512.safetensors", "*.json"],
      "omini")
print("\nDONE")
