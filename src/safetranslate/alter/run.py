"""Runs Phase 2 in three stages: alter, verify, finalize.

Usage:
    python -m safetranslate.alter.run configs/alter.yaml alter [--limit N] [--out-dir DIR]
    python -m safetranslate.alter.run configs/alter.yaml verify [--out-dir DIR]
    python -m safetranslate.alter.run configs/alter.yaml finalize [--out-dir DIR]

--limit N alters only N randomly chosen sentences per split and language (for a pilot).
"""

import argparse
import json
import random
from functools import partial
from pathlib import Path

from openai import OpenAI

from safetranslate.alter.checks import automatic_failure, finalize
from safetranslate.alter.generate import (
    Alteration,
    Verdict,
    alteration_prompt,
    ask_json,
    make_client,
    run_resumable,
    verification_prompt,
)
from safetranslate.alter.plan import eligible_categories, plan
from safetranslate.config import AlterConfig, load_alter_config
from safetranslate.data.schema import Category, ErrorItem, Record

LANGUAGES = ["zh", "ms", "ta"]


def alter_one(client: OpenAI, model: str, record: Record, categories: list[Category]) -> dict:
    """Asks for one alteration; a single-category request that can't be made tries others."""
    result = ask_json(client, model, alteration_prompt(record, categories), Alteration)
    if not result.applicable and len(categories) == 1:
        for fallback in eligible_categories(record.source):
            if fallback == categories[0]:
                continue
            categories = [fallback]
            result = ask_json(client, model, alteration_prompt(record, categories), Alteration)
            if result.applicable:
                break
    return {
        "origin_id": record.id,
        "split": record.split,
        "target_lang": record.target_lang,
        "source": record.source,
        "original": record.target,
        "requested": categories,
        "applicable": result.applicable,
        "altered": result.altered_translation,
        "errors": [e.model_dump() for e in result.errors],
        "model": model,
    }


def verify_one(client: OpenAI, model: str, alteration: dict) -> dict:
    """An alteration is valid only if the verifier confirms every one of its errors."""
    valid = all(
        ask_json(client, model, verification_prompt(alteration, ErrorItem(**e)), Verdict).valid
        for e in alteration["errors"]
    )
    return {"valid": valid, "model": model}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def alter(config: AlterConfig, out_dir: Path, limit: int | None) -> None:
    client = make_client(config.alter_llm)
    rng = random.Random(config.seed)
    jobs = []
    for split, per_category in config.per_category.items():
        path = config.processed_dir / f"{split}.jsonl"
        records = [Record.model_validate_json(line) for line in path.open(encoding="utf-8")]
        for lang in LANGUAGES:
            tasks = plan(
                [r for r in records if r.target_lang == lang],
                per_category,
                config.multi_error_share,
                rng,
            )
            if limit is not None:
                tasks = rng.sample(tasks, min(limit, len(tasks)))
            jobs += [
                (f"{split}-{record.id}", partial(alter_one, client, config.alter_llm.model, record, cats))
                for record, cats in tasks
            ]
    failed = run_resumable(jobs, out_dir / "alterations.jsonl", config.max_workers)
    print(f"{len(jobs)} alteration tasks, {failed} failed (rerun to retry them)")


def verify(config: AlterConfig, out_dir: Path) -> None:
    client = make_client(config.verify_llm)
    # Only alterations that pass the automatic checks are worth a verifier call
    jobs = [
        (alt["task_id"], partial(verify_one, client, config.verify_llm.model, alt))
        for alt in read_jsonl(out_dir / "alterations.jsonl")
        if automatic_failure(alt, config.min_similarity) is None
    ]
    failed = run_resumable(jobs, out_dir / "verdicts.jsonl", config.max_workers)
    print(f"{len(jobs)} verification tasks, {failed} failed (rerun to retry them)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("stage", choices=["alter", "verify", "finalize"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    config = load_alter_config(args.config)
    out_dir = args.out_dir or config.out_dir

    if args.stage == "alter":
        alter(config, out_dir, args.limit)
    elif args.stage == "verify":
        verify(config, out_dir)
    else:
        verdicts = {v["task_id"]: v["valid"] for v in read_jsonl(out_dir / "verdicts.jsonl")}
        alterations = read_jsonl(out_dir / "alterations.jsonl")
        print(json.dumps(finalize(alterations, verdicts, config.min_similarity, out_dir), indent=2))


if __name__ == "__main__":
    main()
