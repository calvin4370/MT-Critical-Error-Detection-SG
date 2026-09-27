"""Runs the evaluation harness and logs results to MLflow.

Usage:
    python -m safetranslate.harness.run CONFIG evaluator SYSTEM [--sample N]
    python -m safetranslate.harness.run CONFIG translate SYSTEM [--sample N]
    python -m safetranslate.harness.run CONFIG judge-translations SYSTEM --judge JUDGE
    python -m safetranslate.harness.run CONFIG real-errors SYSTEM [--sample N]

evaluator: SYSTEM judges the Phase 2 test examples (known errors), without a reference.
translate: SYSTEM translates the Phase 1 test sentences.
judge-translations: JUDGE counts critical errors in SYSTEM's translations (no reference,
    matching how JUDGE's accuracy is measured); rates are corrected with JUDGE's
    evaluator metrics if present. References are used only for spBLEU/chrF.
real-errors: SYSTEM judges real machine translations with human critical-error labels
    (WMT21 Chinese, IndicMT Eval Tamil), without a reference.
--sample N: N altered examples per language and category (plus their error-free
    originals), N sentences per language, or N real translations per dataset; the same
    seed gives the same sample.
"""

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

import mlflow

from safetranslate.alter.generate import make_client, run_resumable
from safetranslate.alter.run import read_jsonl
from safetranslate.config import HarnessConfig, HarnessSystem, load_harness_config
from safetranslate.data import real_errors
from safetranslate.data.schema import ErrorItem, EvaluatorExample, LabelledTranslation, Record
from safetranslate.harness.metrics import (
    binary_metrics,
    evaluator_metrics,
    rogan_gladen,
    translation_scores,
    wilson,
)
from safetranslate.harness.models import judge, translate


def paced(system: HarnessSystem, job: Callable[[], dict]) -> dict:
    """Runs a job, then pauses to stay under the system's rate limit."""
    result = job()
    time.sleep(system.seconds_between_requests)
    return result


def sample_examples(
    examples: list[EvaluatorExample], n: int, rng: random.Random
) -> list[EvaluatorExample]:
    """n altered examples per language and category, plus each one's error-free original."""
    groups: dict[str, list[EvaluatorExample]] = defaultdict(list)
    for example in sorted((e for e in examples if e.errors), key=lambda e: e.id):
        groups[f"{example.target_lang}/{example.errors[0].category}"].append(example)
    chosen = {e.id for group in groups.values() for e in rng.sample(group, min(n, len(group)))}
    chosen |= {f"{i}-original" for i in chosen}
    return [e for e in examples if e.id in chosen]


def sample_records(records: list[Record], n: int, rng: random.Random) -> list[Record]:
    """n test sentences per language."""
    by_lang: dict[str, list[Record]] = defaultdict(list)
    for record in sorted(records, key=lambda r: r.id):
        by_lang[record.target_lang].append(record)
    return [r for group in by_lang.values() for r in rng.sample(group, min(n, len(group)))]


def log_to_mlflow(run_name: str, params: dict, metrics: dict[str, float], artifact: Path) -> None:
    """Records one harness run so all results can be compared in one place."""
    # A local SQLite file: no server needed, and the recommended MLflow backend
    mlflow.set_tracking_uri(f"sqlite:///{Path('mlflow.db').resolve()}")
    mlflow.set_experiment("harness")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        mlflow.log_artifact(str(artifact))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def run_evaluator(config: HarnessConfig, name: str, sample: int | None) -> None:
    system, data = config.systems[name], config.altered_dir / "test.jsonl"
    client = make_client(system)
    examples = [EvaluatorExample.model_validate_json(line) for line in data.open(encoding="utf-8")]
    if sample:
        examples = sample_examples(examples, sample, random.Random(config.seed))
    out = config.out_dir / name / "evaluator"

    def job(example: EvaluatorExample) -> dict:
        errors = judge(client, system, example.source, example.translation, example.target_lang)
        return {"errors": [e.model_dump() for e in errors]}

    jobs = [(e.id, partial(paced, system, partial(job, e))) for e in examples]
    run_resumable(jobs, out / "predictions.jsonl", system.max_workers)

    predictions = {
        r["task_id"]: [ErrorItem(**e) for e in r["errors"]]
        for r in read_jsonl(out / "predictions.jsonl")
    }
    scored = [e for e in examples if e.id in predictions]
    metrics = evaluator_metrics(scored, predictions)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    flat = {f"{g}/{k}": (v["rate"] if isinstance(v, dict) else v) for g, m in metrics.items() for k, v in m.items()}
    params = {"stage": "evaluator", "system": name, "model": system.model,
              "sample": sample, "examples": len(scored), "data_sha256": file_hash(data)}
    log_to_mlflow(f"evaluator-{name}", params, flat, out / "metrics.json")
    print(json.dumps(metrics.get("all", {}), indent=2))


def run_translate(config: HarnessConfig, name: str, sample: int | None) -> None:
    system, data = config.systems[name], config.processed_dir / "test.jsonl"
    client = make_client(system)
    records = [Record.model_validate_json(line) for line in data.open(encoding="utf-8")]
    if sample:
        records = sample_records(records, sample, random.Random(config.seed))

    def job(record: Record) -> dict:
        return {"translation": translate(client, system, record.source, record.target_lang)}

    jobs = [(r.id, partial(paced, system, partial(job, r))) for r in records]
    failed = run_resumable(jobs, config.out_dir / name / "translations.jsonl", system.max_workers)
    print(f"{len(jobs)} translation tasks, {failed} failed (rerun to retry them)")


def run_judge_translations(config: HarnessConfig, name: str, judge_name: str) -> None:
    judge_system, data = config.systems[judge_name], config.processed_dir / "test.jsonl"
    client = make_client(judge_system)
    records = {r.id: r for r in map(Record.model_validate_json, data.open(encoding="utf-8"))}
    out = config.out_dir / name
    translations = {r["task_id"]: r["translation"] for r in read_jsonl(out / "translations.jsonl")}

    def job(record_id: str) -> dict:
        record = records[record_id]
        errors = judge(
            client, judge_system, record.source, translations[record_id], record.target_lang
        )
        return {"errors": [e.model_dump() for e in errors]}

    jobs = [(i, partial(paced, judge_system, partial(job, i))) for i in translations]
    judged_path = out / f"judged_by_{judge_name}.jsonl"
    run_resumable(jobs, judged_path, judge_system.max_workers)

    # The judge's own accuracy, measured on examples with known errors, corrects its counts
    judge_metrics_path = config.out_dir / judge_name / "evaluator" / "metrics.json"
    judge_metrics = json.loads(judge_metrics_path.read_text()) if judge_metrics_path.exists() else {}
    judged = {r["task_id"]: r["errors"] for r in read_jsonl(judged_path)}
    groups: dict[str, list[str]] = defaultdict(list)
    for record_id in judged:
        record = records[record_id]
        groups["all"].append(record_id)
        groups[record.target_lang].append(record_id)
        groups[f"{record.target_lang}/{record.dataset}"].append(record_id)

    metrics = {}
    for group, ids in groups.items():
        rate = wilson(sum(bool(judged[i]) for i in ids), len(ids))
        lang = group.split("/")[0]
        accuracy = judge_metrics.get(lang, {})
        corrected = None
        if accuracy and rate["rate"] is not None:
            sensitivity = accuracy["recall"]["rate"]
            specificity = 1 - accuracy["false_positive_rate"]["rate"]
            corrected = rogan_gladen(rate["rate"], sensitivity, specificity)
        metrics[group] = {
            "critical_error_rate": rate,
            "corrected_critical_error_rate": corrected,
            **translation_scores([translations[i] for i in ids], [records[i].target for i in ids]),
        }
    (out / f"metrics_judged_by_{judge_name}.json").write_text(json.dumps(metrics, indent=2))
    flat = {f"{g}/{k}": (v["rate"] if isinstance(v, dict) else v) for g, m in metrics.items() for k, v in m.items()}
    params = {"stage": "translator", "system": name, "judge": judge_name,
              "judge_model": judge_system.model, "sentences": len(judged), "data_sha256": file_hash(data)}
    log_to_mlflow(f"translator-{name}-judged-by-{judge_name}", params, flat,
                  out / f"metrics_judged_by_{judge_name}.json")
    print(json.dumps(metrics.get("all", {}), indent=2))


def run_real_errors(config: HarnessConfig, name: str, sample: int | None) -> None:
    system, raw = config.systems[name], config.processed_dir.parent / "raw"
    client = make_client(system)
    real_errors.download(raw)
    datasets = {"wmt21_ced": real_errors.load_wmt21(raw), "indicmt": real_errors.load_indicmt(raw)}
    rng = random.Random(config.seed)
    if sample:
        datasets = {k: rng.sample(v, min(sample, len(v))) for k, v in datasets.items()}
    out = config.out_dir / name / "real_errors"

    def job(example: LabelledTranslation) -> dict:
        errors = judge(client, system, example.source, example.translation, example.target_lang)
        return {"errors": [e.model_dump() for e in errors]}

    examples = [e for group in datasets.values() for e in group]
    jobs = [(e.id, partial(paced, system, partial(job, e))) for e in examples]
    run_resumable(jobs, out / "predictions.jsonl", system.max_workers)

    flagged = {r["task_id"]: bool(r["errors"]) for r in read_jsonl(out / "predictions.jsonl")}
    metrics = {}
    for dataset, group in datasets.items():
        scored = [e for e in group if e.id in flagged]
        metrics[dataset] = binary_metrics([e.critical for e in scored], [flagged[e.id] for e in scored])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    flat = {f"{g}/{k}": (v["rate"] if isinstance(v, dict) else v) for g, m in metrics.items() for k, v in m.items()}
    params = {"stage": "real_errors", "system": name, "model": system.model, "sample": sample,
              "examples": len(flagged)}
    log_to_mlflow(f"real-errors-{name}", params, flat, out / "metrics.json")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "stage", choices=["evaluator", "translate", "judge-translations", "real-errors"]
    )
    parser.add_argument("system")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--judge")
    args = parser.parse_args()
    config = load_harness_config(args.config)
    if args.stage == "evaluator":
        run_evaluator(config, args.system, args.sample)
    elif args.stage == "translate":
        run_translate(config, args.system, args.sample)
    elif args.stage == "judge-translations":
        run_judge_translations(config, args.system, args.judge)
    else:
        run_real_errors(config, args.system, args.sample)


if __name__ == "__main__":
    main()
