import json
from pathlib import Path

from safetranslate.config import HarnessConfig, WorkflowConfig
from safetranslate.data.schema import Record
from safetranslate.harness.report import translator_error_section, workflow_section


def _config(tmp_path: Path) -> HarnessConfig:
    processed = tmp_path / "processed"
    processed.mkdir()
    records = [
        Record(id=f"r{i}", split="test", dataset="flores_devtest", target_lang="ms", source="Hi.", target="Hai.")
        for i in range(4)
    ]
    (processed / "test.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in records))
    return HarnessConfig(processed_dir=processed, altered_dir=tmp_path, out_dir=tmp_path / "out", seed=0, systems={})


def _judged(config: HarnessConfig, system: str, flags: dict[str, bool]) -> None:
    path = config.out_dir / system / "judged_by_sealion-evaluator.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"task_id": i, "errors": [{}] if f else []}) + "\n" for i, f in flags.items()))


def test_translator_errors_compare_models_on_the_same_sentences(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _judged(config, "a", {"r0": True, "r1": False, "r2": False, "r3": True})
    _judged(config, "b", {"r0": True, "r1": True})  # only r0, r1 in common
    section = translator_error_section(config, ["a", "b"])
    assert "| ms | a | 2 | 50.0%" in section
    assert "| ms | b | 2 | 100.0%" in section


def test_translator_errors_correct_for_a_better_than_chance_judge(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _judged(config, "a", {"r0": True, "r1": False, "r2": False, "r3": False})
    accuracy = config.out_dir / "sealion-evaluator" / "evaluator" / "metrics.json"
    accuracy.parent.mkdir(parents=True)
    accuracy.write_text(json.dumps({"ms": {"recall": {"rate": 0.8}, "false_positive_rate": {"rate": 0.1}}}))
    # (0.25 + 0.9 - 1) / (0.8 + 0.9 - 1) = 0.214...
    assert "| 21.4% |" in translator_error_section(config, ["a"])


def test_workflow_section_reads_the_checker_metrics(tmp_path: Path) -> None:
    rate = {"rate": 0.1, "low": 0.05, "high": 0.2, "n": 10}
    metrics = {"all": {"sentences": 10, "published_first_try": rate, "fixed_by_retry": rate, "escalated": rate,
                       "mean_attempts": 1.3, "mean_seconds": 2.0, "errors_single_pass": rate, "errors_reaching_readers": rate}}
    (tmp_path / "metrics_checked_by_gemma.json").write_text(json.dumps(metrics))
    workflow = WorkflowConfig(harness_config=tmp_path / "h.yaml", translator="t", evaluator="e", checker="gemma",
                              max_attempts=3, sample_per_language=1, out_dir=tmp_path)
    section = workflow_section(workflow)
    assert "| all | 10 | 10.0% (5.0%-20.0%)" in section
    assert "Not run yet" in workflow_section(None)
