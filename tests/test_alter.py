import json
import random
from pathlib import Path
from types import SimpleNamespace

from safetranslate.alter.checks import automatic_failure, finalize
from safetranslate.alter.generate import run_resumable
from safetranslate.alter.plan import eligible_categories, plan
from safetranslate.alter.run import alter_one, verify_one
from safetranslate.data.schema import Record


def _record(i: int, source: str) -> Record:
    return Record(
        id=f"r{i}", dataset="d", split="test", target_lang="ms", source=source, target="t"
    )


class FakeClient:
    """Stands in for the OpenAI client, replying with queued JSON strings in order."""

    def __init__(self, replies: list[dict]) -> None:
        self.replies = [json.dumps(r) for r in replies]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_: object) -> SimpleNamespace:
        message = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_eligible_categories_screen_the_english() -> None:
    assert eligible_categories("Take 500 mg twice a day.") == [
        "wrong_quantity", "flipped_meaning", "added_information"
    ]
    assert "wrong_name" in eligible_categories("The minister visited Johor today.")
    assert "removed_information" in eligible_categories(
        "Wash your hands with soap before eating any food."
    )


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
    "altered": "Ambil 500 g dua kali sehari.",
    "errors": [{"category": "wrong_quantity", "span": "500 g", "description": "mg -> g"}],
}


def test_automatic_failure() -> None:
    assert automatic_failure(ALTERATION, 0.5) is None
    assert automatic_failure({**ALTERATION, "altered": ALTERATION["original"]}, 0.5) == "unchanged"
    bad_span = [{**ALTERATION["errors"][0], "span": "5 kg"}]
    assert automatic_failure({**ALTERATION, "errors": bad_span}, 0.5) == "span_not_found"
    rewritten = {**ALTERATION, "altered": "Sila makan 500 g ubat itu dengan air suam."}
    assert automatic_failure({**rewritten, "errors": [{**bad_span[0], "span": "500 g"}]}, 0.9) == (
        "too_different"
    )


def test_alter_one_falls_back_to_another_category() -> None:
    record = Record(
        id="r1", dataset="d", split="test", target_lang="ms",
        source="Take 500 mg twice a day.", target="Ambil 500 mg dua kali sehari.",
    )
    client = FakeClient([
        {"applicable": False, "altered_translation": "", "errors": []},
        {"applicable": True, "altered_translation": "Ambil 500 g dua kali sehari.",
         "errors": [{"category": "wrong_quantity", "span": "500 g", "description": "mg -> g"}]},
    ])
    result = alter_one(client, "m", record, ["flipped_meaning"])
    assert result["requested"] == ["wrong_quantity"]
    assert result["applicable"]


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
    report = finalize([ALTERATION, rejected], {"test-r1": True, "test-r2": False}, 0.5, tmp_path)
    examples = [json.loads(line) for line in (tmp_path / "test.jsonl").open()]
    assert [len(e["errors"]) for e in examples] == [1, 0]
    assert report["outcomes"] == {
        "test/ms/wrong_quantity: passed": 1,
        "test/ms/wrong_quantity: verifier_rejected": 1,
    }
