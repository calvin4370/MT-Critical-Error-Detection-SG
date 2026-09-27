import json
import random
from pathlib import Path

import pytest

from safetranslate.alter.checks import EditError, apply_edits, automatic_failure, finalize
from safetranslate.alter.generate import Edit, run_resumable
from safetranslate.alter.plan import eligible_categories, plan
from safetranslate.alter.run import alter_one, verify_one
from safetranslate.data.schema import Record
from fakes import FakeClient


def _record(i: int, source: str) -> Record:
    return Record(
        id=f"r{i}", dataset="d", split="test", target_lang="ms", source=source, target="t"
    )


def test_eligible_categories_screen_the_english() -> None:
    assert eligible_categories("Take 500 mg twice a day.") == [
        "wrong_quantity", "flipped_meaning", "added_information"
    ]
    assert "wrong_name" in eligible_categories("The minister visited Johor early today.")
    assert "removed_information" in eligible_categories(
        "Wash your hands with soap before eating any food."
    )
    assert eligible_categories("References") == []


def test_plan_balances_categories_and_uses_each_sentence_once() -> None:
    records = [_record(i, f"Patients in Johor took {i} tablets after every meal.") for i in range(50)]
    tasks = plan(records, per_category=5, multi_error_share=0.2, rng=random.Random(0))
    first_categories = [cats[0] for _, cats in tasks]
    assert all(first_categories.count(c) == 5 for c in set(first_categories))
    assert len({r.id for r, _ in tasks}) == len(tasks)
    assert sum(len(cats) > 1 for _, cats in tasks) == round(len(tasks) * 0.2)


ALTERATION = {
    "task_id": "test-r1",
    "origin_id": "r1",
    "split": "test",
    "target_lang": "ms",
    "source": "Take 500 mg twice a day.",
    "original": "Ambil 500 mg dua kali sehari.",
    "requested": ["wrong_quantity"],
    "applicable": True,
    "failure": None,
    "altered": "Ambil 500 g dua kali sehari.",
    "errors": [{"category": "wrong_quantity", "span": "500 g", "description": "mg -> g"}],
}


def test_automatic_failure() -> None:
    assert automatic_failure(ALTERATION) is None
    assert automatic_failure({**ALTERATION, "failure": "no_change"}) == "no_change"
    assert automatic_failure({**ALTERATION, "requested": ["wrong_name"]}) == "wrong_categories"


def _edit(category: str, find: str, replace_with: str, missing: str = "") -> Edit:
    return Edit(
        category=category, find=find, replace_with=replace_with,
        missing_english=missing, description="d",
    )


def test_apply_edits_replaces_only_the_chosen_text() -> None:
    altered, errors = apply_edits(
        "Take 500 mg twice a day.",
        "Ambil 500 mg dua kali sehari.",
        [_edit("wrong_quantity", "500 mg", "500 g"),
         _edit("removed_information", " dua kali", "", missing="twice")],
    )
    assert altered == "Ambil 500 g sehari."
    assert [(e.category, e.span) for e in errors] == [
        ("wrong_quantity", "500 g"), ("removed_information", "twice")
    ]


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (_edit("wrong_quantity", "5 kg", "50 kg"), "find_not_found"),
        (_edit("wrong_quantity", "500 mg", "500 mg"), "no_change"),
        (_edit("removed_information", "sehari", "", missing="daily"), "missing_english_not_found"),
        (_edit("wrong_quantity", "500 mg", ""), "empty_replacement"),
    ],
)
def test_apply_edits_rejects_bad_edits(edit: Edit, reason: str) -> None:
    with pytest.raises(EditError, match=reason):
        apply_edits("Take 500 mg twice a day.", "Ambil 500 mg dua kali sehari.", [edit])


def test_apply_edits_rejects_duplicate_edits() -> None:
    duplicate = _edit("wrong_quantity", "500 mg", "500 g")
    with pytest.raises(EditError, match="duplicate_edits"):
        apply_edits("Take 500 mg.", "Ambil 500 mg.", [duplicate, duplicate])


def test_alter_one_falls_back_to_another_category() -> None:
    record = Record(
        id="r1", dataset="d", split="test", target_lang="ms",
        source="Take 500 mg twice a day.", target="Ambil 500 mg dua kali sehari.",
    )
    client = FakeClient([
        {"applicable": False, "edits": []},
        {"applicable": True, "edits": [{"category": "wrong_quantity", "find": "500 mg",
                                        "replace_with": "500 g", "description": "mg -> g"}]},
    ])
    result = alter_one(client, "m", record, ["flipped_meaning"])
    assert result["requested"] == ["wrong_quantity"]
    assert result["altered"] == "Ambil 500 g dua kali sehari."
    assert result["failure"] is None


def test_verify_one_needs_every_error_confirmed() -> None:
    two_errors = {**ALTERATION, "errors": ALTERATION["errors"] * 2}
    assert not verify_one(FakeClient([{"valid": True}, {"valid": False}]), "m", two_errors)["valid"]


def test_run_resumable_skips_finished_tasks(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    run_resumable([("a", lambda: {"x": 1})], out, max_workers=2)
    calls = []
    run_resumable(
        [("a", lambda: calls.append("a") or {"x": 1}), ("b", lambda: {"x": 2})], out, 2
    )
    assert calls == []
    assert [json.loads(line)["task_id"] for line in out.open()] == ["a", "b"]


def test_finalize_writes_altered_and_original_examples(tmp_path: Path) -> None:
    rejected = {**ALTERATION, "task_id": "test-r2", "origin_id": "r2"}
    report = finalize([ALTERATION, rejected], {"test-r1": True, "test-r2": False}, tmp_path)
    examples = [json.loads(line) for line in (tmp_path / "test.jsonl").open()]
    assert [len(e["errors"]) for e in examples] == [1, 0]
    assert report["outcomes"] == {
        "test/ms/wrong_quantity: passed": 1,
        "test/ms/wrong_quantity: verifier_rejected": 1,
    }


def test_alter_one_retries_a_failed_edit_with_feedback() -> None:
    record = Record(
        id="r1", dataset="d", split="test", target_lang="ms",
        source="Take 500 mg twice a day.", target="Ambil 500 mg dua kali sehari.",
    )
    edit = {"category": "wrong_quantity", "replace_with": "500 g", "description": "mg -> g"}
    client = FakeClient([
        {"applicable": True, "edits": [{**edit, "find": "500 milligrams"}]},  # not in the text
        {"applicable": True, "edits": [{**edit, "find": "500 mg"}]},
    ])
    result = alter_one(client, "m", record, ["wrong_quantity"])
    assert result["altered"] == "Ambil 500 g dua kali sehari."
    assert result["failure"] is None


def test_alter_one_applies_multiple_errors_one_call_each() -> None:
    record = Record(
        id="r1", dataset="d", split="test", target_lang="ms",
        source="Do not take 500 mg twice a day.", target="Jangan ambil 500 mg dua kali sehari.",
    )
    client = FakeClient([
        {"applicable": True, "edits": [{"category": "wrong_quantity", "find": "500 mg",
                                        "replace_with": "50 mg", "description": "d"}]},
        {"applicable": True, "edits": [{"category": "flipped_meaning", "find": "Jangan ambil",
                                        "replace_with": "Ambil", "description": "d"}]},
    ])
    result = alter_one(client, "m", record, ["wrong_quantity", "flipped_meaning"])
    assert result["altered"] == "Ambil 50 mg dua kali sehari."
    assert [e["category"] for e in result["errors"]] == ["wrong_quantity", "flipped_meaning"]
