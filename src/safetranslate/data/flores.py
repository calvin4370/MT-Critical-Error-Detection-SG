"""FLORES+: 997 dev + 1,012 devtest sentences (Wikinews, Wikijunior, Wikivoyage).

The dataset is gated on Hugging Face, so downloading needs HUGGINGFACE_TOKEN in the
environment (e.g. `uv run --env-file .env ...`).
"""

import json
import os
import urllib.request
from pathlib import Path
from typing import Literal

from safetranslate.data.schema import Language, Record

BASE_URL = "https://huggingface.co/datasets/openlanguagedata/flores_plus/resolve/main"
CODES = {"en": "eng_Latn", "zh": "cmn_Hans", "ms": "zsm_Latn", "ta": "tam_Taml"}
FloresSplit = Literal["dev", "devtest"]
# dev is only used for tuning; devtest is part of the final test data
RECORD_SPLIT = {"dev": "validation", "devtest": "test"}


def download(raw_dir: Path) -> None:
    """Downloads English and all target files for both splits, skipping existing ones."""
    request_headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}
    for split in RECORD_SPLIT:
        (raw_dir / split).mkdir(parents=True, exist_ok=True)
        for code in CODES.values():
            path = raw_dir / split / f"{code}.jsonl"
            if not path.exists():
                request = urllib.request.Request(
                    f"{BASE_URL}/{split}/{code}.jsonl", headers=request_headers
                )
                with urllib.request.urlopen(request) as response:
                    path.write_bytes(response.read())


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
