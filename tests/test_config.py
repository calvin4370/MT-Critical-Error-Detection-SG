import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from safetranslate.config import DataConfig, load_data_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "data.yaml"


def _raw_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_repo_config_loads() -> None:
    config = load_data_config(CONFIG_PATH)
    assert config.languages == ["zh", "ms", "ta"]
    assert config.training_corpora[0].max_pairs_per_language is None
    assert config.filters.length_ratio_percentiles == (1, 99)


@pytest.mark.parametrize(
    ("text", "is_junk"),
    [
        ("Pengguna:Malurian123/Vietnam", True),
        ("See https://example.com for details.", True),
        ("Retrieved November 22, 2010.", True),
        ("用户:张三", True),
        ("The meeting starts at 2:30 pm.", False),
        ("Wash your hands and/or use sanitiser!", False),
        ("请勤洗手。", False),
    ],
)
def test_junk_patterns(text: str, is_junk: bool) -> None:
    patterns = load_data_config(CONFIG_PATH).filters.junk_patterns
    assert any(re.search(p, text) for p in patterns) == is_junk


def test_invalid_regex_rejected() -> None:
    raw = _raw_config()
    raw["filters"]["junk_patterns"] = ["(unclosed"]
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw)


@pytest.mark.parametrize("bounds", [[99, 1], [5, 5], [-1, 50], [1, 101]])
def test_bad_percentiles_rejected(bounds: list[float]) -> None:
    raw = _raw_config()
    raw["filters"]["length_ratio_percentiles"] = bounds
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw)


def test_unknown_dataset_rejected() -> None:
    raw = _raw_config()
    raw["test_sets"] = ["ntrex", "flores_devtset"]
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw)


def test_misspelled_key_rejected() -> None:
    raw = _raw_config()
    raw["overlap"] = {"ngram_sise": 13}
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw)


def test_validation_and_test_sets_must_not_overlap() -> None:
    raw = _raw_config()
    raw["validation_sets"] = ["ntrex"]
    with pytest.raises(ValidationError):
        DataConfig.model_validate(raw)
