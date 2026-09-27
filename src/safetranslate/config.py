"""Typed, validated settings loaded from the YAML files in configs/."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from safetranslate.data.schema import Language, Split

# Known dataset names; a new dataset is added here together with its loader
TrainingCorpusName = Literal["wikimedia"]
EvalSetName = Literal["flores_dev", "flores_devtest", "ntrex", "wmt24pp", "tico19"]


class _StrictModel(BaseModel):
    # Misspelled keys in the YAML (e.g. "ngram_sise") should fail, not be ignored
    model_config = ConfigDict(extra="forbid")


class TrainingCorpus(_StrictModel):
    """A training corpus and how many pairs to take from it.

    Attributes:
        name: Corpus name.
        max_pairs_per_language: Cap on pairs per language; None means no cap.
    """

    name: TrainingCorpusName
    max_pairs_per_language: int | None = Field(default=None, gt=0)


class FilterConfig(_StrictModel):
    """Rules for dropping noisy training pairs.

    Attributes:
        junk_patterns: Regexes; a pair is dropped if either side matches any of them.
        length_ratio_percentiles: Low and high percentiles of the character-length
            ratio in professional translations (FLORES+ dev) to keep.
        strip_citation_marks: Whether to remove marks like "[12]".
    """

    junk_patterns: list[str]
    length_ratio_percentiles: tuple[float, float]
    strip_citation_marks: bool = True

    @field_validator("junk_patterns")
    @classmethod
    def _patterns_compile(cls, patterns: list[str]) -> list[str]:
        # Catch broken regexes at load time rather than midway through a data build
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as err:
                raise ValueError(f"invalid regex {pattern!r}: {err}") from err
        return patterns

    @field_validator("length_ratio_percentiles")
    @classmethod
    def _percentiles_ordered(cls, bounds: tuple[float, float]) -> tuple[float, float]:
        low, high = bounds
        if not 0 <= low < high <= 100:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        return bounds


class OverlapConfig(_StrictModel):
    """How training pairs that overlap test sentences are detected.

    Attributes:
        ngram_size: Drop a training pair sharing a word sequence this long with any
            test sentence. Exact matches after normalisation are always dropped.
    """

    ngram_size: int = Field(gt=0)


class DataConfig(_StrictModel):
    """Settings for building the datasets (Phase 1).

    Attributes:
        languages: Target languages to build data for.
        data_dir: Folder for raw downloads and processed output.
        training_corpora: Corpora used for training.
        validation_sets: Sets used for tuning only.
        test_sets: Sets used for final results only.
        filters: Noise filters for training pairs.
        overlap: Overlap-removal settings.
    """

    languages: list[Language] = Field(min_length=1)
    data_dir: Path
    training_corpora: list[TrainingCorpus] = Field(min_length=1)
    validation_sets: list[EvalSetName]
    test_sets: list[EvalSetName] = Field(min_length=1)
    filters: FilterConfig
    overlap: OverlapConfig

    @model_validator(mode="after")
    def _no_validation_test_overlap(self) -> "DataConfig":
        # A set used for tuning must never also produce the reported results
        shared = set(self.validation_sets) & set(self.test_sets)
        if shared:
            raise ValueError(f"sets used for both validation and test: {sorted(shared)}")
        return self


def load_data_config(path: str | Path) -> DataConfig:
    """Reads and validates a data config YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated config.
    """
    # safe_load only builds plain types; yaml.load can construct arbitrary objects
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DataConfig.model_validate(raw)


class LLMEndpoint(_StrictModel):
    """An OpenAI-compatible chat API (a local vLLM server, OpenRouter, SageMaker...).

    Attributes:
        base_url: API address, e.g. "http://localhost:8000/v1".
        model: Model name as the server knows it.
        api_key_env: Environment variable holding the API key; None for local servers.
    """

    base_url: str
    model: str
    api_key_env: str | None = None


class AlterConfig(_StrictModel):
    """Settings for making the altered (known-error) data (Phase 2).

    Attributes:
        processed_dir: Folder with train/validation/test.jsonl from Phase 1.
        out_dir: Folder for the altered data and its report.
        seed: Random seed for picking sentences and categories.
        per_category: How many sentences to alter per category per language, per split.
        multi_error_share: Share of altered sentences that get 2-3 errors.
        max_workers: How many LLM requests to send in parallel.
        alter_llm: Model that makes the alterations.
        verify_llm: Different model that confirms each error.
    """

    processed_dir: Path
    out_dir: Path
    seed: int
    per_category: dict[Split, int]
    multi_error_share: float = Field(ge=0, le=1)
    max_workers: int = Field(gt=0)
    alter_llm: LLMEndpoint
    verify_llm: LLMEndpoint


def load_alter_config(path: str | Path) -> AlterConfig:
    """Reads and validates an alteration config YAML file."""
    with open(path, encoding="utf-8") as f:
        return AlterConfig.model_validate(yaml.safe_load(f))
