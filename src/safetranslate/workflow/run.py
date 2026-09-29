"""Runs the workflow on test sentences and measures it.

Usage:
    python -m safetranslate.workflow.run configs/workflow.yaml run
    python -m safetranslate.workflow.run configs/workflow.yaml check

run: every sampled test sentence goes through the graph (needs the translator/evaluator
    server). Each sentence's first attempt is kept, so "single pass" and "with the
    workflow" can be compared on the same sentences with no extra translation calls.
check: a different model (checker) looks for critical errors in the first attempt and in
    the final output (needs the checker's server), measuring what the workflow fixes and
    what slips through.
"""

import json
import random
import sys
import time
from functools import partial

from safetranslate.alter.generate import make_client, run_resumable
from safetranslate.alter.run import read_jsonl
from safetranslate.config import WorkflowConfig, load_harness_config, load_workflow_config
from safetranslate.data.schema import Record
from safetranslate.harness.metrics import wilson
from safetranslate.harness.models import judge
from safetranslate.harness.run import log_to_mlflow, sample_records
from safetranslate.workflow.graph import build_graph


def run(config: WorkflowConfig) -> None:
    harness = load_harness_config(config.harness_config)
    translator, evaluator = harness.systems[config.translator], harness.systems[config.evaluator]
    graph = build_graph(
        (make_client(translator), translator), (make_client(evaluator), evaluator), config.max_attempts
    )
    records = [Record.model_validate_json(l) for l in (harness.processed_dir / "test.jsonl").open(encoding="utf-8")]
    records = sample_records(records, config.sample_per_language, random.Random(harness.seed))

    def job(record: Record) -> dict:
        start = time.perf_counter()
        state = graph.invoke({"source": record.source, "lang": record.target_lang})
        return {
            "status": state["status"],
            "attempts": state["attempts"],
            "first_translation": state["history"][0]["translation"],
            "final_translation": state["translation"],
            "history": state["history"],
            "notifications": state.get("notifications", []),
            "seconds": time.perf_counter() - start,
        }

    jobs = [(r.id, partial(job, r)) for r in records]
    failed = run_resumable(jobs, config.out_dir / "runs.jsonl", evaluator.max_workers)
    print(f"{len(jobs)} sentences, {failed} failed (rerun to retry them)")


def check(config: WorkflowConfig) -> None:
    harness = load_harness_config(config.harness_config)
    checker = harness.systems[config.checker]
    client = make_client(checker)
    records = {r.id: r for r in map(Record.model_validate_json, (harness.processed_dir / "test.jsonl").open(encoding="utf-8"))}
    runs = {r["task_id"]: r for r in read_jsonl(config.out_dir / "runs.jsonl")}

    def job(task_id: str) -> dict:
        record, result = records[task_id], runs[task_id]
        first = judge(client, checker, record.source, result["first_translation"], record.target_lang)
        final = judge(client, checker, record.source, result["final_translation"], record.target_lang)
        return {"first_has_error": bool(first), "final_has_error": bool(final)}

    checked_path = config.out_dir / f"checked_by_{config.checker}.jsonl"
    run_resumable([(i, partial(job, i)) for i in runs], checked_path, checker.max_workers)
    checked = {r["task_id"]: r for r in read_jsonl(checked_path)}

    metrics = {}
    for lang in ["all", "zh", "ms", "ta"]:
        ids = [i for i in checked if lang == "all" or records[i].target_lang == lang]
        if not ids:
            continue
        published = [i for i in ids if runs[i]["status"] == "published"]
        metrics[lang] = {
            "sentences": len(ids),
            "published_first_try": wilson(sum(runs[i]["attempts"] == 1 and runs[i]["status"] == "published" for i in ids), len(ids)),
            "fixed_by_retry": wilson(sum(runs[i]["attempts"] > 1 and runs[i]["status"] == "published" for i in ids), len(ids)),
            "escalated": wilson(sum(runs[i]["status"] == "escalated" for i in ids), len(ids)),
            "mean_attempts": sum(runs[i]["attempts"] for i in ids) / len(ids),
            "mean_seconds": sum(runs[i]["seconds"] for i in ids) / len(ids),
            # What a reader would get without the workflow (first attempt, always published)
            "errors_single_pass": wilson(sum(checked[i]["first_has_error"] for i in ids), len(ids)),
            # What a reader gets with it (escalated sentences go to a human, not the reader)
            "errors_reaching_readers": wilson(sum(checked[i]["final_has_error"] for i in published), len(ids)),
        }
    out = config.out_dir / f"metrics_checked_by_{config.checker}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    flat = {f"{g}/{k}": (v["rate"] if isinstance(v, dict) else v) for g, m in metrics.items() for k, v in m.items()}
    params = {"stage": "workflow", "translator": config.translator, "evaluator": config.evaluator,
              "checker": config.checker, "max_attempts": config.max_attempts}
    log_to_mlflow(f"workflow-{config.evaluator}-checked-by-{config.checker}", params, flat, out)
    print(json.dumps(metrics.get("all", {}), indent=2))


if __name__ == "__main__":
    workflow_config = load_workflow_config(sys.argv[1])
    {"run": run, "check": check}[sys.argv[2]](workflow_config)
