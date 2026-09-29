import json
from pathlib import Path

import pytest

# The merge needs PyTorch, which is in the optional "train" group (not installed in CI)
torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file  # noqa: E402

from safetranslate.finetune.merge import merge  # noqa: E402

WEIGHT = "model.layers.0.self_attn.q_proj.weight"


def make_files(tmp_path: Path, adapted: str) -> tuple[Path, Path]:
    """A tiny base model (one adapted and one untouched weight) and a rank-2 adapter."""
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    save_file({WEIGHT: torch.ones(4, 3), "model.embed_tokens.weight": torch.ones(5, 3)},
              base / "model-00001-of-00001.safetensors")
    (base / "config.json").write_text("{}")
    save_file({f"base_model.model.{adapted}".replace(".weight", ".lora_A.weight"): torch.ones(2, 3),
               f"base_model.model.{adapted}".replace(".weight", ".lora_B.weight"): torch.ones(4, 2)},
              adapter / "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_text(json.dumps({"lora_alpha": 4, "r": 2}))
    return base, adapter


def test_merge_adds_scaled_adapter_to_matching_weights_only(tmp_path: Path) -> None:
    base, adapter = make_files(tmp_path, WEIGHT)
    merge(adapter, tmp_path / "out", base)
    merged = load_file(tmp_path / "out" / "model-00001-of-00001.safetensors")
    # 1 + (alpha / r = 2) * (B @ A: every entry 2) = 5
    assert torch.equal(merged[WEIGHT], torch.full((4, 3), 5.0))
    assert torch.equal(merged["model.embed_tokens.weight"], torch.ones(5, 3))
    assert (tmp_path / "out" / "config.json").exists()


def test_merge_fails_if_an_adapted_weight_is_missing(tmp_path: Path) -> None:
    base, adapter = make_files(tmp_path, "model.layers.9.mlp.up_proj.weight")
    with pytest.raises(ValueError, match="merged 0 of 1"):
        merge(adapter, tmp_path / "out", base)
