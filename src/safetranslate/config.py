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
        extra_body: Server-specific request options, e.g. turning off Qwen3's thinking.
    """

    base_url: str
    model: str
    api_key_env: str | None = None
    extra_body: dict = {}


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


class HarnessSystem(LLMEndpoint):
    """A model the harness can test, plus how fast it may be called.

    Attributes:
        max_workers: Parallel requests; 1 for rate-limited free APIs.
        seconds_between_requests: Pause after each request, to stay under rate limits.
    """

    max_workers: int = Field(default=16, gt=0)
    seconds_between_requests: float = Field(default=0, ge=0)


class HarnessConfig(_StrictModel):
    """Settings for the evaluation harness (Phase 3).

    Attributes:
        processed_dir: Phase 1 output (test sentences with professional references).
        altered_dir: Phase 2 output (examples with known errors).
        out_dir: Where predictions, translations and metrics are written.
        seed: Random seed for sampling.
        systems: Models that can be tested, by name.
    """

    processed_dir: Path
    altered_dir: Path
    out_dir: Path
    seed: int
    systems: dict[str, HarnessSystem]


def load_harness_config(path: str | Path) -> HarnessConfig:
    """Reads and validates a harness config YAML file."""
    with open(path, encoding="utf-8") as f:
        return HarnessConfig.model_validate(yaml.safe_load(f))


class FinetuneConfig(_StrictModel):
    """Settings for LoRA fine-tuning (Phases 4-5).

    Attributes:
        base_model: Hugging Face model to adapt.
        train_file: Training examples (JSONL).
        validation_file: Examples for measuring loss during training (JSONL).
        validation_limit: How many validation examples to use, to keep evaluation quick.
        out_dir: Where checkpoints and the final adapter are written.
        max_length: Longest example in tokens; longer ones are cut.
        lora_rank: Size of the LoRA add-on matrices.
        lora_alpha: Scaling of the LoRA update (usually 2 x rank).
        lora_dropout: Dropout inside the LoRA layers, against overfitting.
        learning_rate: Step size for training.
        epochs: Passes over the training data.
        batch_size: Examples per GPU step.
        gradient_accumulation: Steps summed before each update (effective batch =
            batch_size x gradient_accumulation).
        eval_every: Evaluate and save a checkpoint every this many updates.
    """

    base_model: str
    train_file: Path
    validation_file: Path
    validation_limit: int = Field(gt=0)
    out_dir: Path
    max_length: int = Field(gt=0)
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0, lt=1)
    learning_rate: float = Field(gt=0)
    epochs: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    eval_every: int = Field(gt=0)


def load_finetune_config(path: str | Path) -> FinetuneConfig:
    """Reads and validates a fine-tuning config YAML file."""
    with open(path, encoding="utf-8") as f:
        return FinetuneConfig.model_validate(yaml.safe_load(f))


class WorkflowConfig(_StrictModel):
    """Settings for running and measuring the LangGraph workflow (Phase 6).

    Attributes:
        harness_config: Harness config whose systems (models) the workflow can use.
        translator: System name for the translate node.
        evaluator: System name for the evaluate node.
        checker: A different system that measures errors slipping through.
        max_attempts: Translations tried before escalating to a human.
        sample_per_language: Test sentences per language to run.
        out_dir: Where results are written.
    """

    harness_config: Path
    translator: str
    evaluator: str
    checker: str
    max_attempts: int = Field(gt=0)
    sample_per_language: int = Field(gt=0)
    out_dir: Path


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    """Reads and validates a workflow config YAML file."""
    with open(path, encoding="utf-8") as f:
        return WorkflowConfig.model_validate(yaml.safe_load(f))
