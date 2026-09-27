"""Automatic checks on alterations, and turning the ones that pass into final data."""

import difflib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from safetranslate.data.schema import ErrorItem, EvaluatorExample

SPOT_CHECK_PER_GROUP = 50


def automatic_failure(alteration: dict, min_similarity: float) -> str | None:
    """Returns why an alteration fails the automatic checks, or None if it passes."""
    if not alteration["applicable"]:
        return "not_applicable"
    original, altered = alteration["original"], alteration["altered"]
    if altered.strip() == original.strip():
        return "unchanged"
    errors = [ErrorItem(**e) for e in alteration["errors"]]
    if sorted(e.category for e in errors) != sorted(alteration["requested"]):
        return "wrong_categories"
    for error in errors:
        quoted_from = alteration["source"] if error.category == "removed_information" else altered
        if error.span not in quoted_from:
            return "span_not_found"
    # A big rewrite means other, unlabelled changes may have slipped in
    if difflib.SequenceMatcher(None, original, altered).ratio() < min_similarity:
        return "too_different"
    return None


def finalize(
    alterations: list[dict], verdicts: dict[str, bool], min_similarity: float, out_dir: Path
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
        failure = automatic_failure(alt, min_similarity)
        if failure is None and not verdicts.get(alt["task_id"], False):
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
