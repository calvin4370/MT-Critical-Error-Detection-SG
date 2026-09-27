"""FLORES+: 997 dev + 1,012 devtest sentences (Wikinews, Wikijunior, Wikivoyage).

The dataset is gated on Hugging Face, so downloading needs HF_TOKEN in the
environment (e.g. `uv run --env-file .env ...`).
"""

import json
import os
from pathlib import Path
from typing import Literal

from safetranslate.data.download import fetch
from safetranslate.data.schema import Language, Record

# Pinned to a commit so the data cannot silently change upstream
BASE_URL = "https://huggingface.co/datasets/openlanguagedata/flores_plus/resolve/5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06"
CODES = {"en": "eng_Latn", "zh": "cmn_Hans", "ms": "zsm_Latn", "ta": "tam_Taml"}
FloresSplit = Literal["dev", "devtest"]
# dev is only used for tuning; devtest is part of the final test data
RECORD_SPLIT = {"dev": "validation", "devtest": "test"}


def download(raw_dir: Path) -> None:
    """Downloads English and all target files for both splits, skipping existing ones."""
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    for split in RECORD_SPLIT:
        for code in CODES.values():
            name = f"{split}/{code}.jsonl"
            fetch(f"{BASE_URL}/{name}", raw_dir / name, headers)


def _read(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row for row in map(json.loads, f)}


def load(raw_dir: Path, lang: Language, split: FloresSplit) -> list[Record]:
    """Pairs English and target sentences that share the same id."""
    english = _read(raw_dir / split / f"{CODES['en']}.jsonl")
    targets = _read(raw_dir / split / f"{CODES[lang]}.jsonl")
    return [
        Record(
            id=f"flores_{split}-{lang}-{sentence_id:06d}",
            dataset=f"flores_{split}",
            split=RECORD_SPLIT[split],
            target_lang=lang,
            source=english[sentence_id]["text"].strip(),
            target=row["text"].strip(),
            domain=row["domain"],
            doc_id=row["url"],
        )
        for sentence_id, row in sorted(targets.items())
    ]
