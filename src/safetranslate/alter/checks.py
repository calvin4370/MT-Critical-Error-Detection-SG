"""Automatic checks on alterations, and turning the ones that pass into final data."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from safetranslate.alter.generate import Edit
from safetranslate.data.schema import ErrorItem, EvaluatorExample

SPOT_CHECK_PER_GROUP = 50


class EditError(Exception):
    """An LLM-proposed edit that can't be applied; the message is the reason."""


def apply_edits(
    source: str, original: str, edits: list[Edit], protected: tuple[str, ...] = ()
) -> tuple[str, list[ErrorItem]]:
    """Applies the LLM's find-and-replace edits to the original translation.

    The LLM decides what to change; applying it in code guarantees the text really
    changes, nothing else changes, and each error's span is known exactly.

    Args:
        protected: Earlier errors' text, which these edits must leave intact.

    Raises:
        EditError: If an edit can't be applied as described.
    """
    if len({e.find for e in edits}) < len(edits):
        raise EditError("duplicate_edits")
    text, errors = original, []
    for edit in edits:
        if edit.find not in text:
            raise EditError("find_not_found")
        if edit.find == edit.replace_with:
            raise EditError("no_change")
        text = text.replace(edit.find, edit.replace_with, 1)
        if edit.category == "removed_information":
            # Nothing is left to highlight in the translation, so quote the missing English
            if not edit.missing_english or edit.missing_english not in source:
                raise EditError("missing_english_not_found")
            span = edit.missing_english
        elif not edit.replace_with:
            raise EditError("empty_replacement")
        else:
            span = edit.replace_with
        errors.append(ErrorItem(category=edit.category, span=span, description=edit.description))
    # A later edit could have overwritten an earlier one
    spans = [*protected, *(e.span for e in errors if e.category != "removed_information")]
    if any(span not in text for span in spans):
        raise EditError("edits_overlap")
    return text, errors


def automatic_failure(alteration: dict) -> str | None:
    """Returns why an alteration fails the automatic checks, or None if it passes."""
    if not alteration["applicable"]:
        return "not_applicable"
    if alteration["failure"]:
        return alteration["failure"]
    if sorted(e["category"] for e in alteration["errors"]) != sorted(alteration["requested"]):
        return "wrong_categories"
    return None


def finalize(
    alterations: list[dict],
    verdicts: dict[str, bool],
    out_dir: Path,
    verified_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict:
    """Writes the final examples per split, a spot-check sample and a report.

    Each passing alteration gives two examples: the altered translation with its errors,
    and the original translation with no errors.

    Returns:
        The report that is also written to report.json.
    """
    examples: dict[str, list[EvaluatorExample]] = defaultdict(list)
    outcomes: Counter[str] = Counter()
    passed_by_group: dict[str, list[dict]] = defaultdict(list)
    for alt in alterations:
        failure = automatic_failure(alt)
        needs_verdict = alt["split"] in verified_splits
        if failure is None and needs_verdict and not verdicts.get(alt["task_id"], False):
            failure = "verifier_rejected" if alt["task_id"] in verdicts else "not_verified"
        group = f"{alt['split']}/{alt['target_lang']}/{'+'.join(sorted(alt['requested']))}"
        outcomes[f"{group}: {failure or 'passed'}"] += 1
        if failure:
            continue
        passed_by_group[f"{alt['target_lang']}/{alt['requested'][0]}"].append(alt)
        common = {
            "origin_id": alt["origin_id"],
            "split": alt["split"],
            "target_lang": alt["target_lang"],
            "source": alt["source"],
        }
        examples[alt["split"]] += [
            EvaluatorExample(
                id=alt["task_id"], translation=alt["altered"], errors=alt["errors"], **common
            ),
            EvaluatorExample(
                id=f"{alt['task_id']}-original", translation=alt["original"], errors=[], **common
            ),
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_examples in examples.items():
        (out_dir / f"{split}.jsonl").write_text(
            "".join(e.model_dump_json() + "\n" for e in split_examples), encoding="utf-8"
        )
    # A fixed-seed sample per language and category for Claude's manual review
    rng = random.Random(0)
    sample = [
        alt
        for group in sorted(passed_by_group)
        for alt in rng.sample(
            passed_by_group[group], min(SPOT_CHECK_PER_GROUP, len(passed_by_group[group]))
        )
    ]
    (out_dir / "spot_check.jsonl").write_text(
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in sample), encoding="utf-8"
    )
    report = {
        "examples_per_split": {s: len(e) for s, e in examples.items()},
        "outcomes": dict(sorted(outcomes.items())),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
