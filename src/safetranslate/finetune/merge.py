"""Merges a LoRA adapter into its base model, as a fallback for serving.

Usage: uv run --group train python -m safetranslate.finetune.merge ADAPTER_DIR OUT_DIR

vLLM normally serves the adapter on top of the base model. If that fails, the merged
copy (adapter folded into the base weights) is served like any ordinary model.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def lora_weights(adapter: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Each adapted weight's small (A, B) matrices, keyed by the base model's weight name."""
    config = json.loads((adapter / "adapter_config.json").read_text())
    with safe_open(adapter / "adapter_model.safetensors", "pt") as f:
        weights = {k: f.get_tensor(k) for k in f.keys()}
    pairs = {
        key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight"):
            (weights[key], weights[key.replace("lora_A", "lora_B")])
        for key in weights if ".lora_A." in key
    }
    return pairs, config["lora_alpha"] / config["r"]


def merge(adapter: Path, out: Path, base: Path | None = None) -> None:
    """Writes base + adapter one weight file at a time, so it needs ~5 GB of RAM, not ~16.

    Each change (scale * B @ A) is computed only when its weight is reached: computing
    all of them up front would take ~27 GB.
    """
    config = json.loads((adapter / "adapter_config.json").read_text())
    base = base or Path(snapshot_download(config["base_model_name_or_path"]))
    pairs, scale = lora_weights(adapter)
    out.mkdir(parents=True, exist_ok=True)
    merged = 0
    for shard in sorted(base.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for name, tensor in tensors.items():
            if name in pairs:
                a, b = pairs[name]
                tensors[name] = (tensor.float() + scale * b.float() @ a.float()).to(tensor.dtype)
                merged += 1
        save_file(tensors, out / shard.name, metadata={"format": "pt"})
    # Every adapted weight must have been found, or the merged model is silently wrong
    if merged != len(pairs):
        raise ValueError(f"merged {merged} of {len(pairs)} adapted weights")
    for extra in base.glob("*.json"):
        shutil.copy(extra, out / extra.name)
    print(f"merged {merged} weights into {out}")


if __name__ == "__main__":
    merge(Path(sys.argv[1]), Path(sys.argv[2]))
