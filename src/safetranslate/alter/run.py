"""Runs Phase 2 in three stages: alter, verify, finalize.

Usage:
    python -m safetranslate.alter.run configs/alter.yaml alter [--limit N] [--out-dir DIR]
    python -m safetranslate.alter.run configs/alter.yaml verify [--splits ...] [--out-dir DIR]
    python -m safetranslate.alter.run configs/alter.yaml finalize [--splits ...] [--out-dir DIR]

--limit N alters only N randomly chosen sentences per split and language (for a pilot).
--splits: which splits are verified (verify) or must be verified to be kept (finalize);
    default all. Unverified splits keep examples that pass the automatic checks.
"""

import argparse
import json
import random
from functools import partial
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from safetranslate.alter.checks import EditError, apply_edits, automatic_failure, finalize
from safetranslate.alter.generate import (
    Alteration,
    Verdict,
    alteration_prompt,
    alteration_schema,
    ask_json,
    make_client,
    run_resumable,
    verification_prompt,
)
from safetranslate.alter.plan import eligible_categories, plan
from safetranslate.config import AlterConfig, load_alter_config
from safetranslate.data.schema import Category, ErrorItem, Record

LANGUAGES = ["zh", "ms", "ta"]


MAX_ATTEMPTS = 3
FEEDBACK = {
    "find_not_found": 'The "find" text does not appear in the translation. Copy "find" '
    "exactly, character for character, from the translation, not from the English or "
    "the example.",
    "no_change": '"replace_with" must be different from "find".',
    "missing_english_not_found": '"missing_english" must be copied exactly from the '
    "English sentence.",
    "empty_replacement": 'Only removed_information may delete text; "replace_with" must '
    "contain the changed words.",
    "edits_overlap": "Your edit changed a phrase that was already edited; choose another "
    "part of the sentence.",
    "invalid_reply": "Your reply was not valid; keep it short.",
}


def try_category(
    client: OpenAI,
    model: str,
    extra_body: dict,
    record: Record,
    translation: str,
    category: Category,
    protected: tuple[str, ...] = (),
) -> tuple[str, list[ErrorItem], list[dict]] | str:
    """Asks for one error of one category, retrying with feedback when an edit fails.

    Returns:
        (new translation, its errors, the raw edits), or the reason it failed.
    """
    keep = f'\nDo not change these already-edited phrases: {list(protected)}' if protected else ""
    feedback, reason = "", ""
    for _ in range(MAX_ATTEMPTS):
        prompt = alteration_prompt(
            record.source, translation, record.target_lang, category, keep + feedback
        )
        try:
            result = ask_json(
                client, model, prompt, Alteration, alteration_schema([category]), extra_body
            )
        except ValidationError:
            reason = "invalid_reply"
            feedback = f"\nYour previous attempt was rejected: {FEEDBACK[reason]}"
            continue
        if not result.applicable or not result.edits:
            return "not_applicable"
        try:
            new_translation, errors = apply_edits(
                record.source, translation, result.edits, protected
            )
            return new_translation, errors, [e.model_dump() for e in result.edits]
        except EditError as err:
            reason = str(err)
            feedback = f"\nYour previous attempt was rejected: {FEEDBACK.get(reason, reason)}"
    return reason


def alter_one(
    client: OpenAI, model: str, record: Record, categories: list[Category], extra_body: dict = {}
) -> dict:
    """Creates the requested errors one call at a time, each on the already-edited text.

    A single-category request that can't be made tries the sentence's other categories.
    """
    options = [categories]
    if len(categories) == 1:
        options += [[c] for c in eligible_categories(record.source) if c != categories[0]]

    failure: str | None = "not_applicable"
    for option in options:
        translation, errors, edits, failure = record.target, [], [], None
        for category in option:
            protected = tuple(e.span for e in errors if e.category != "removed_information")
            outcome = try_category(
                client, model, extra_body, record, translation, category, protected
            )
            if isinstance(outcome, str):
                failure = outcome
                break
            translation, new_errors, new_edits = outcome
            errors, edits = errors + new_errors, edits + new_edits
        if failure != "not_applicable":
            break
    return {
        "origin_id": record.id,
        "split": record.split,
        "target_lang": record.target_lang,
        "source": record.source,
        "original": record.target,
        "requested": option,
        "applicable": failure != "not_applicable",
        "edits": edits,
        "failure": None if failure == "not_applicable" else failure,
        "altered": translation if failure is None else "",
        "errors": [e.model_dump() for e in errors] if failure is None else [],
        "model": model,
    }


def verify_one(client: OpenAI, model: str, alteration: dict) -> dict:
    """Valid only if the blind verifier finds a meaning change covering every claimed category."""
    verdict = ask_json(client, model, verification_prompt(alteration), Verdict)
    claimed = {e["category"] for e in alteration["errors"]}
    valid = verdict.meaning_changed and claimed <= set(verdict.categories)
    return {"valid": valid, "verdict": verdict.model_dump(), "model": model}


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
            llm = config.alter_llm
            jobs += [
                (f"{split}-{record.id}", partial(alter_one, client, llm.model, record, cats, llm.extra_body))
                for record, cats in tasks
            ]
    failed = run_resumable(jobs, out_dir / "alterations.jsonl", config.max_workers)
    print(f"{len(jobs)} alteration tasks, {failed} failed (rerun to retry them)")


def verify(config: AlterConfig, out_dir: Path, splits: tuple[str, ...]) -> None:
    client = make_client(config.verify_llm)
    # Only alterations that pass the automatic checks are worth a verifier call
    jobs = [
        (alt["task_id"], partial(verify_one, client, config.verify_llm.model, alt))
        for alt in read_jsonl(out_dir / "alterations.jsonl")
        if automatic_failure(alt) is None and alt["split"] in splits
    ]
    failed = run_resumable(jobs, out_dir / "verdicts.jsonl", config.max_workers)
    print(f"{len(jobs)} verification tasks, {failed} failed (rerun to retry them)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("stage", choices=["alter", "verify", "finalize"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()
    config = load_alter_config(args.config)
    out_dir = args.out_dir or config.out_dir

    if args.stage == "alter":
        alter(config, out_dir, args.limit)
    elif args.stage == "verify":
        verify(config, out_dir, tuple(args.splits))
    else:
        verdicts = {v["task_id"]: v["valid"] for v in read_jsonl(out_dir / "verdicts.jsonl")}
        alterations = read_jsonl(out_dir / "alterations.jsonl")
        print(json.dumps(finalize(alterations, verdicts, out_dir, tuple(args.splits)), indent=2))


if __name__ == "__main__":
    main()
