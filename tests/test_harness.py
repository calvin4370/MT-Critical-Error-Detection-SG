import pytest

from fakes import FakeClient
from safetranslate.config import LLMEndpoint
from safetranslate.data.schema import ErrorItem, EvaluatorExample
from safetranslate.harness.metrics import evaluator_metrics, rogan_gladen, wilson
from safetranslate.harness.models import judge, translate

ENDPOINT = LLMEndpoint(base_url="http://x", model="m")
QUANTITY = ErrorItem(category="wrong_quantity", span="500 g", description="d")


def _example(i: int, errors: list[ErrorItem]) -> EvaluatorExample:
    return EvaluatorExample(
        id=f"e{i}", origin_id="r", split="test", target_lang="ms",
        source="s", translation="t", errors=errors,
    )


def test_wilson_interval() -> None:
    result = wilson(8, 10)
    assert result["rate"] == 0.8
    assert result["low"] == pytest.approx(0.490, abs=1e-3)
    assert result["high"] == pytest.approx(0.943, abs=1e-3)
    assert wilson(0, 0)["rate"] is None


def test_evaluator_metrics() -> None:
    examples = [_example(1, [QUANTITY]), _example(2, [QUANTITY]), _example(3, []), _example(4, [])]
    wrong_category = ErrorItem(category="wrong_name", span="x", description="d")
    predictions = {"e1": [QUANTITY], "e2": [], "e3": [wrong_category], "e4": []}
    metrics = evaluator_metrics(examples, predictions)
    assert metrics["all"]["recall"]["rate"] == 0.5
    assert metrics["all"]["false_positive_rate"]["rate"] == 0.5
    assert metrics["all"]["precision"] == 0.5
    assert metrics["ms/wrong_quantity"]["category_accuracy"]["rate"] == 1.0


def test_rogan_gladen() -> None:
    assert rogan_gladen(0.10, sensitivity=0.80, specificity=0.95) == pytest.approx(0.0667, abs=1e-3)
    assert rogan_gladen(0.01, 0.80, 0.95) == 0.0  # clipped, not negative
    with pytest.raises(ValueError):
        rogan_gladen(0.1, 0.5, 0.5)


def test_judge_and_translate_parse_replies() -> None:
    client = FakeClient([
        {"errors": [QUANTITY.model_dump()]},
        {"translation": "Basuh tangan anda."},
    ])
    assert judge(client, ENDPOINT, "Take 500 mg.", "Ambil 500 g.", "ms") == [QUANTITY]
    assert translate(client, ENDPOINT, "Wash your hands.", "ms") == "Basuh tangan anda."


def _harness_setup(tmp_path, monkeypatch, replies):
    import json as _json

    from safetranslate.config import HarnessConfig
    from safetranslate.harness import run as harness_run

    monkeypatch.chdir(tmp_path)  # MLflow writes mlflow.db here, not in the project
    fake = FakeClient(replies)  # one shared client, so replies are used in order
    monkeypatch.setattr(harness_run, "make_client", lambda _: fake)
    (tmp_path / "altered").mkdir()
    (tmp_path / "processed").mkdir()
    config = HarnessConfig(
        processed_dir=tmp_path / "processed", altered_dir=tmp_path / "altered",
        out_dir=tmp_path / "out", seed=0,
        systems={"fake": {"base_url": "http://x", "model": "m"}},
    )
    return config, harness_run, _json


def test_run_evaluator_scores_and_logs(tmp_path, monkeypatch) -> None:
    replies = [{"errors": [QUANTITY.model_dump()]}, {"errors": []}]
    config, harness_run, json = _harness_setup(tmp_path, monkeypatch, replies)
    examples = [_example(1, [QUANTITY]), _example(2, [])]
    (tmp_path / "altered" / "test.jsonl").write_text(
        "".join(e.model_dump_json() + "\n" for e in examples)
    )
    harness_run.run_evaluator(config, "fake", sample=None)
    metrics = json.loads((tmp_path / "out" / "fake" / "evaluator" / "metrics.json").read_text())
    assert metrics["all"]["recall"]["rate"] == 1.0
    assert metrics["all"]["false_positive_rate"]["rate"] == 0.0
    assert (tmp_path / "mlflow.db").exists()


def test_run_translate_then_judge(tmp_path, monkeypatch) -> None:
    from safetranslate.data.schema import Record

    replies = [{"translation": "Ambil 500 g."}, {"errors": [QUANTITY.model_dump()]}]
    config, harness_run, json = _harness_setup(tmp_path, monkeypatch, replies)
    record = Record(id="ntrex-ms-000001", dataset="ntrex", split="test", target_lang="ms",
                    source="Take 500 mg.", target="Ambil 500 mg.")
    (tmp_path / "processed" / "test.jsonl").write_text(record.model_dump_json() + "\n")
    # spBLEU downloads a tokenizer model on first use; keep the test offline
    monkeypatch.setattr(harness_run, "translation_scores", lambda h, r: {"spbleu": 0.0, "chrf": 0.0})
    harness_run.run_translate(config, "fake", sample=None)
    harness_run.run_judge_translations(config, "fake", "fake")
    metrics = json.loads((tmp_path / "out" / "fake" / "metrics_judged_by_fake.json").read_text())
    assert metrics["ms/ntrex"]["critical_error_rate"]["rate"] == 1.0
    assert "chrf" in metrics["all"]


def test_sample_examples_keeps_originals() -> None:
    import random

    from safetranslate.harness.run import sample_examples

    altered = [_example(i, [QUANTITY]) for i in range(5)]
    originals = [_example(i, []).model_copy(update={"id": f"e{i}-original"}) for i in range(5)]
    chosen = sample_examples(altered + originals, 2, random.Random(0))
    assert len(chosen) == 4
    assert {e.id for e in chosen if not e.errors} == {f"{e.id}-original" for e in chosen if e.errors}


def test_binary_metrics() -> None:
    from safetranslate.harness.metrics import binary_metrics

    m = binary_metrics([True, True, False, False], [True, False, True, False])
    assert m["recall"]["rate"] == 0.5
    assert m["false_positive_rate"]["rate"] == 0.5
    assert m["precision"] == 0.5


def test_load_wmt21_joins_labels_and_removes_spaces(tmp_path) -> None:
    import io
    import tarfile

    from safetranslate.data.real_errors import load_wmt21

    folder = tmp_path / "wmt21_ced"
    folder.mkdir()
    (folder / "test_blind.tsv").write_text("7\tWash your hands.\t请 洗 手 。\n")
    gold = b"en-zh\tREFERENCE\t7\tERR\n"
    with tarfile.open(folder / "goldlabels.tar.gz", "w:gz") as tar:
        info = tarfile.TarInfo("enzh_majority_test_goldlabels/goldlabels.txt")
        info.size = len(gold)
        tar.addfile(info, io.BytesIO(gold))
    [example] = load_wmt21(tmp_path)
    assert example.translation == "请洗手。"
    assert example.critical
