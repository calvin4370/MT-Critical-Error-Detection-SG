"""Builds comparison tables (Markdown) from the harness outputs.

Usage: uv run python -m safetranslate.harness.report configs/harness.yaml

Models are compared only on the items every one of them was run on, because each
model's own metric files can cover different numbers of sentences.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from safetranslate.alter.run import read_jsonl
from safetranslate.config import HarnessConfig, load_harness_config
from safetranslate.data import real_errors
from safetranslate.data.schema import Record
from safetranslate.harness.metrics import binary_metrics, translation_scores

LANGUAGES = ["zh", "ms", "ta"]


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


def main() -> None:
    config = load_harness_config(sys.argv[1])
    systems = [s for s in config.systems if (config.out_dir / s).exists()]
    translators = [s for s in systems if (config.out_dir / s / "translations.jsonl").exists()]
    sections = [
        "# Results",
        translation_section(config, translators),
        real_error_section(config, systems),
        synthetic_section(config, systems),
    ]
    out = config.out_dir / "results.md"
    out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(out.read_text())


if __name__ == "__main__":
    main()
