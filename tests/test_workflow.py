from fakes import FakeClient
from safetranslate.config import LLMEndpoint
from safetranslate.data.schema import ErrorItem
from safetranslate.workflow.graph import (
    build_graph,
    check_document,
    pair_document,
    split_document,
    translate_document,
)

ENDPOINT = LLMEndpoint(base_url="http://x", model="m")
ERROR = ErrorItem(category="wrong_quantity", span="500 g", description="mg -> g").model_dump()


def _graph(replies: list[dict]):
    # Translator and evaluator share one fake client, so replies alternate in call order
    client = FakeClient(replies)
    return build_graph((client, ENDPOINT), (client, ENDPOINT), max_attempts=3)


def test_clean_translation_is_published_first_time() -> None:
    graph = _graph([{"translation": "Ambil 500 mg."}, {"errors": []}])
    state = graph.invoke({"source": "Take 500 mg.", "lang": "ms"})
    assert state["status"] == "published"
    assert state["attempts"] == 1
    assert state.get("notifications", []) == []


def test_error_is_retried_with_a_notification_then_published() -> None:
    graph = _graph([
        {"translation": "Ambil 500 g."}, {"errors": [ERROR]},
        {"translation": "Ambil 500 mg."}, {"errors": []},
    ])
    state = graph.invoke({"source": "Take 500 mg.", "lang": "ms"})
    assert state["status"] == "published"
    assert state["translation"] == "Ambil 500 mg."
    assert state["attempts"] == 2
    assert "wrong_quantity" in state["notifications"][0]
    assert len(state["history"]) == 2


def test_persistent_errors_are_escalated_after_three_attempts() -> None:
    graph = _graph([{"translation": "Ambil 500 g."}, {"errors": [ERROR]}] * 3)
    state = graph.invoke({"source": "Take 500 mg.", "lang": "ms"})
    assert state["status"] == "escalated"
    assert state["attempts"] == 3
    assert "human review" in state["notifications"][-1]


def test_split_document_keeps_paragraphs_and_sentences() -> None:
    text = "Wash your hands. Stay home!\n\nSee a doctor if feverish."
    assert split_document(text) == [
        (0, "Wash your hands."), (0, "Stay home!"), (1, "See a doctor if feverish."),
    ]


def test_check_only_skips_the_translator_and_does_not_retry() -> None:
    # Only the evaluator is called: one reply, and errors don't trigger a retry
    graph = _graph([{"errors": [ERROR]}])
    state = graph.invoke({"source": "Take 500 mg.", "translation": "Ambil 500 g.", "lang": "ms", "check_only": True})
    assert state["status"] == "checked"
    assert state["translation"] == "Ambil 500 g."
    assert state["history"][0]["errors"] == [ERROR]


def test_chinese_sentences_split_without_spaces() -> None:
    assert split_document("洗手。待在家里！") == [(0, "洗手。"), (0, "待在家里！")]


def test_pair_document_pairs_sentences_when_counts_match() -> None:
    pairs, counts = pair_document("Wash hands. Stay home.\nSee a doctor.", "洗手。待在家。\n看医生。")
    assert counts == (2, 2)
    assert [(p["source"], p["translation"], p["unit"]) for p in pairs] == [
        ("Wash hands.", "洗手。", "sentence"), ("Stay home.", "待在家。", "sentence"),
        ("See a doctor.", "看医生。", "sentence"),
    ]


def test_pair_document_falls_back_to_paragraph_then_whole_text() -> None:
    # The MT merged two sentences: the paragraph is checked as one pair
    pairs, _ = pair_document("Wash hands. Stay home.", "Cuci tangan dan duduk di rumah.")
    assert pairs == [{"source": "Wash hands. Stay home.", "translation": "Cuci tangan dan duduk di rumah.", "unit": "paragraph"}]
    # Paragraph counts differ: the whole text is one pair
    pairs, counts = pair_document("A.\nB.", "A.")
    assert counts == (2, 1)
    assert [p["unit"] for p in pairs] == ["text"]


def test_check_document_returns_each_pair_with_its_unit() -> None:
    graph = _graph([{"errors": []}])
    result = check_document(graph, "Take 500 mg.", "Ambil 500 mg.", "ms", max_concurrency=1)
    assert result["segments"][0]["status"] == "checked"
    assert result["segments"][0]["unit"] == "sentence"
    assert result["paragraphs"] == (1, 1)


def test_translate_document_reassembles_in_order() -> None:
    graph = _graph([{"translation": "A."}, {"errors": []}])
    # One sentence keeps the fake client's reply order deterministic
    result = translate_document(graph, "Wash your hands.", "ms", max_concurrency=1)
    assert result["translation"] == "A."
    assert result["segments"][0]["status"] == "published"


def test_run_and_check_measure_the_workflow(tmp_path, monkeypatch) -> None:
    import json

    import yaml

    from safetranslate.config import WorkflowConfig
    from safetranslate.data.schema import Record
    from safetranslate.workflow import run as workflow_run

    monkeypatch.chdir(tmp_path)  # MLflow writes mlflow.db here
    (tmp_path / "processed").mkdir()
    record = Record(id="ntrex-ms-000001", dataset="ntrex", split="test", target_lang="ms",
                    source="Take 500 mg.", target="Ambil 500 mg.")
    (tmp_path / "processed" / "test.jsonl").write_text(record.model_dump_json() + "\n")
    harness = {"processed_dir": str(tmp_path / "processed"), "altered_dir": str(tmp_path),
               "out_dir": str(tmp_path / "h"), "seed": 0,
               "systems": {"fake": {"base_url": "http://x", "model": "m"}}}
    (tmp_path / "harness.yaml").write_text(yaml.safe_dump(harness))
    config = WorkflowConfig(harness_config=tmp_path / "harness.yaml", translator="fake", evaluator="fake",
                            checker="fake", max_attempts=3, sample_per_language=5, out_dir=tmp_path / "w")
    # First attempt has an error, the retry is clean; the checker agrees on both
    fake = FakeClient([
        {"translation": "Ambil 500 g."}, {"errors": [ERROR]},
        {"translation": "Ambil 500 mg."}, {"errors": []},
        {"errors": [ERROR]}, {"errors": []},
    ])
    monkeypatch.setattr(workflow_run, "make_client", lambda _: fake)
    workflow_run.run(config)
    workflow_run.check(config)
    metrics = json.loads((tmp_path / "w" / "metrics_checked_by_fake.json").read_text())
    assert metrics["all"]["fixed_by_retry"]["rate"] == 1.0
    assert metrics["all"]["errors_single_pass"]["rate"] == 1.0
    assert metrics["all"]["errors_reaching_readers"]["rate"] == 0.0
