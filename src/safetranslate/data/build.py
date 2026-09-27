"""Builds train/validation/test JSONL files and a report from configs/data.yaml.

Usage: uv run --env-file .env python -m safetranslate.data.build configs/data.yaml
"""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from safetranslate.config import DataConfig, load_data_config
from safetranslate.data import flores, ntrex, tico19, wikimedia, wmt24pp
from safetranslate.data.filters import filter_pairs, ratio_bounds
from safetranslate.data.overlap import remove_overlap
from safetranslate.data.schema import Language, Record


def download_all(config: DataConfig) -> None:
    """Downloads every raw file the config needs, skipping files already present."""
    raw = config.data_dir / "raw"
    ntrex.download(raw / "ntrex")
    wmt24pp.download(raw / "wmt24pp")
    flores.download(raw / "flores")
    for lang in config.languages:
        tico19.download(raw / "tico19", lang)
        wikimedia.download(raw / "wikimedia", lang)


def load_eval_set(name: str, raw: Path, lang: Language) -> list[Record]:
    """Loads one validation or test set by its config name."""
    loaders = {
        "ntrex": lambda: ntrex.load(raw / "ntrex", lang),
        "tico19": lambda: tico19.load(raw / "tico19", lang),
        "wmt24pp": lambda: wmt24pp.load(raw / "wmt24pp", lang),
        "flores_dev": lambda: flores.load(raw / "flores", lang, "dev"),
        "flores_devtest": lambda: flores.load(raw / "flores", lang, "devtest"),
    }
    return loaders[name]()


def build(config: DataConfig) -> dict:
    """Builds the processed files from already-downloaded raw data.

    Returns:
        The report that is also written to processed/report.json.
    """
    raw, out = config.data_dir / "raw", config.data_dir / "processed"
    splits: dict[str, list[Record]] = {"train": [], "validation": [], "test": []}
    report: dict = {"languages": {}}

    for lang in config.languages:
        validation = [r for s in config.validation_sets for r in load_eval_set(s, raw, lang)]
        test = [r for s in config.test_sets for r in load_eval_set(s, raw, lang)]
        # Ratio limits come from professional translations in the validation set
        bounds = ratio_bounds(validation, config.filters.length_ratio_percentiles)

        train: list[Record] = []
        lang_report: dict = {"ratio_bounds": bounds, "training_corpora": {}}
        for corpus in config.training_corpora:
            loaded = {"wikimedia": wikimedia.load}[corpus.name](raw / corpus.name, lang)
            filtered, filter_counts = filter_pairs(
                loaded, config.filters.junk_patterns, bounds, config.filters.strip_citation_marks
            )
            kept, overlapping = remove_overlap(
                filtered, validation + test, config.overlap.ngram_size
            )
            cap = corpus.max_pairs_per_language
            if cap is not None and len(kept) > cap:
                # Fixed seed so a capped build is still reproducible
                kept = random.Random(0).sample(kept, cap)
            lang_report["training_corpora"][corpus.name] = {
                "loaded": len(loaded),
                "filter_counts": dict(filter_counts),
                "dropped_by_overlap": dict(overlapping),
                "kept": len(kept),
            }
            train += kept

        splits["train"] += train
        splits["validation"] += validation
        splits["test"] += test
        report["languages"][lang] = lang_report

    out.mkdir(parents=True, exist_ok=True)
    report["files"] = {}
    for split, records in splits.items():
        path = out / f"{split}.jsonl"
        path.write_text(
            "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
        )
        report["files"][path.name] = {
            "records": len(records),
            "per_dataset_and_language": dict(Counter(f"{r.dataset}/{r.target_lang}" for r in records)),
            # Fingerprint so experiments can record exactly which data version they used
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    data_config = load_data_config(sys.argv[1])
    download_all(data_config)
    print(json.dumps(build(data_config), indent=2))
