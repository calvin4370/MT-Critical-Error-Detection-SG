"""Builds comparison tables (Markdown) from the harness outputs.

Usage: uv run python -m safetranslate.harness.report configs/harness.yaml [configs/workflow.yaml]

Models are compared only on the items every one of them was run on, because each
model's own metric files can cover different numbers of sentences.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from safetranslate.alter.run import read_jsonl
from safetranslate.config import HarnessConfig, WorkflowConfig, load_harness_config, load_workflow_config
from safetranslate.data import real_errors
from safetranslate.data.schema import Record
from safetranslate.harness.metrics import binary_metrics, rogan_gladen, translation_scores, wilson

LANGUAGES = ["zh", "ms", "ta"]
# The fine-tuned evaluator counts critical errors in every model's translations
JUDGE = "sealion-evaluator"


def percent(rate: dict) -> str:
    """Formats a Wilson rate as '42.0% (38.1-46.0)'."""
    if rate["rate"] is None:
        return "-"
    return f"{rate['rate']:.1%} ({rate['low']:.1%}-{rate['high']:.1%})"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def translation_section(config: HarnessConfig, systems: list[str]) -> str:
    records = {r.id: r for r in map(Record.model_validate_json, (config.processed_dir / "test.jsonl").open(encoding="utf-8"))}
    translations = {
        s: {r["task_id"]: r["translation"] for r in read_jsonl(config.out_dir / s / "translations.jsonl")}
        for s in systems
    }
    comet = {
        s: {r["task_id"]: r["comet22"] for r in read_jsonl(config.out_dir / s / "comet_scores.jsonl")}
        for s in systems
    }
    rows = []
    for lang in LANGUAGES:
        # Only sentences every model translated, so the comparison is like-for-like
        common = set.intersection(*(set(t) for t in translations.values()))
        ids = sorted(i for i in common if records[i].target_lang == lang)
        for s in systems:
            scores = translation_scores([translations[s][i] for i in ids], [records[i].target for i in ids])
            have_comet = ids and all(i in comet[s] for i in ids)
            comet_score = f"{sum(comet[s][i] for i in ids) / len(ids):.3f}" if have_comet else "-"
            rows.append([lang, s, str(len(ids)), f"{scores['spbleu']:.1f}", f"{scores['chrf']:.1f}", comet_score])
    return "## Translation quality (same sentences for every model)\n\n" + table(
        ["Language", "Model", "Sentences", "spBLEU", "chrF", "COMET-22"], rows
    )


def real_error_section(config: HarnessConfig, systems: list[str]) -> str:
    raw = config.processed_dir.parent / "raw"
    labels = {e.id: e for e in real_errors.load_wmt21(raw) + real_errors.load_indicmt(raw)}
    flagged = {
        s: {r["task_id"]: bool(r["errors"]) for r in read_jsonl(config.out_dir / s / "real_errors" / "predictions.jsonl")}
        for s in systems
    }
    systems = [s for s in systems if flagged[s]]
    common = set.intersection(*(set(f) for f in flagged.values())) if systems else set()
    rows = []
    for dataset in ["wmt21_ced", "indicmt"]:
        ids = sorted(i for i in common if labels[i].dataset == dataset)
        for s in systems:
            m = binary_metrics([labels[i].critical for i in ids], [flagged[s][i] for i in ids])
            precision = f"{m['precision']:.1%}" if m["precision"] is not None else "-"
            rows.append([dataset, s, str(len(ids)), percent(m["recall"]), percent(m["false_positive_rate"]), precision])
    return "## Detecting real critical errors (human-labelled; same items for every model)\n\n" + table(
        ["Dataset", "Model", "Items", "Recall (95% CI)", "False-positive rate (95% CI)", "Precision"], rows
    )


def synthetic_section(config: HarnessConfig, systems: list[str]) -> str:
    rows = []
    for s in systems:
        path = config.out_dir / s / "evaluator" / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        for group in ["all", *LANGUAGES]:
            if group in metrics:
                m = metrics[group]
                rows.append([s, group, percent(m["recall"]), percent(m["false_positive_rate"]), percent(m["category_accuracy"])])
    if not rows:
        return "## Detecting synthetic critical errors\n\nNot run yet."
    return "## Detecting synthetic critical errors (known answers)\n\n" + table(
        ["Model", "Group", "Recall (95% CI)", "False-positive rate (95% CI)", "Category accuracy"], rows
    )


def corrected(rate: dict, accuracy: dict) -> str:
    """The rate corrected for the judge's measured accuracy, or "-" if it can't be."""
    sensitivity = accuracy.get("recall", {}).get("rate")
    false_positive_rate = accuracy.get("false_positive_rate", {}).get("rate")
    if None in (rate["rate"], sensitivity, false_positive_rate) or sensitivity <= false_positive_rate:
        return "-"
    return f"{rogan_gladen(rate['rate'], sensitivity, 1 - false_positive_rate):.1%}"


def translator_error_section(config: HarnessConfig, systems: list[str], judge: str = JUDGE) -> str:
    title = f"## Critical errors in each model's translations (judged by {judge}; same sentences for every model)"
    judged = {
        s: {r["task_id"]: bool(r["errors"]) for r in read_jsonl(config.out_dir / s / f"judged_by_{judge}.jsonl")}
        for s in systems
    }
    systems = [s for s in systems if judged[s]]
    if not systems:
        return f"{title}\n\nNot run yet."
    records = {r.id: r for r in map(Record.model_validate_json, (config.processed_dir / "test.jsonl").open(encoding="utf-8"))}
    accuracy_path = config.out_dir / judge / "evaluator" / "metrics.json"
    accuracy = json.loads(accuracy_path.read_text()) if accuracy_path.exists() else {}
    common = set.intersection(*(set(judged[s]) for s in systems))
    rows = []
    for lang in LANGUAGES:
        ids = [i for i in common if records[i].target_lang == lang]
        for s in systems:
            rate = wilson(sum(judged[s][i] for i in ids), len(ids))
            rows.append([lang, s, str(len(ids)), percent(rate), corrected(rate, accuracy.get(lang, {}))])
    return f"{title}\n\n" + table(
        ["Language", "Model", "Sentences", "Flagged by the judge (95% CI)", "Corrected for the judge's accuracy"], rows
    )


def workflow_section(workflow: WorkflowConfig | None) -> str:
    title = "## The translation workflow (translate, check, retry, escalate)"
    path = workflow.out_dir / f"metrics_checked_by_{workflow.checker}.json" if workflow else None
    if not path or not path.exists():
        return f"{title}\n\nNot run yet."
    metrics = json.loads(path.read_text())
    rows = [
        [group, str(m["sentences"]), percent(m["published_first_try"]), percent(m["fixed_by_retry"]),
         percent(m["escalated"]), f"{m['mean_attempts']:.2f}", percent(m["errors_single_pass"]),
         percent(m["errors_reaching_readers"])]
        for group, m in metrics.items()
    ]
    return (
        f"{title}\n\nTranslator: {workflow.translator}; evaluator: {workflow.evaluator}; errors counted by an "
        f"independent checker ({workflow.checker}). \"Single pass\" is the first translation published as is; "
        "escalated sentences go to a person instead of readers.\n\n"
        + table(
            ["Group", "Sentences", "Published first time", "Fixed by retry", "Escalated", "Mean attempts",
             "Errors, single pass", "Errors reaching readers"],
            rows,
        )
    )


def main() -> None:
    config = load_harness_config(sys.argv[1])
    workflow_path = Path(sys.argv[2] if len(sys.argv) > 2 else "configs/workflow.yaml")
    workflow = load_workflow_config(workflow_path) if workflow_path.exists() else None
    systems = [s for s in config.systems if (config.out_dir / s).exists()]
    translators = [s for s in systems if (config.out_dir / s / "translations.jsonl").exists()]
    sections = [
        "# Results",
        translation_section(config, translators),
        real_error_section(config, systems),
        synthetic_section(config, systems),
        translator_error_section(config, translators),
        workflow_section(workflow),
    ]
    out = config.out_dir / "results.md"
    out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(out.read_text())


if __name__ == "__main__":
    main()
