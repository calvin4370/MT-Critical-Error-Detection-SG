"""TICO-19: 3,071 COVID-19 health sentences professionally translated from English."""

import csv
import urllib.request
from pathlib import Path

from safetranslate.data.schema import Language, Record

BASE_URL = "https://huggingface.co/datasets/gmnlp/tico19/resolve/main"
PARTS = ["dev", "test"]


def download(raw_dir: Path, lang: Language) -> None:
    """Downloads both TICO-19 files for one language, skipping ones already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        path = raw_dir / f"{part}.en-{lang}.tsv"
        if not path.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{part}/{path.name}", path)


def load(raw_dir: Path, lang: Language) -> list[Record]:
    """Loads TICO-19's dev and test files together; all of it is used as test data."""
    records = []
    for part in PARTS:
        # csv's default quoting undoes the doubled quotes ("") found in the Tamil file
        with open(raw_dir / f"{part}.en-{lang}.tsv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                records.append(
                    Record(
                        id=f"tico19-{lang}-{len(records) + 1:06d}",
                        dataset="tico19",
                        split="test",
                        target_lang=lang,
                        source=row["sourceString"].strip(),
                        target=row["targetString"].strip(),
                        domain="health",
                        doc_id=row["stringID"].split(":")[0],
                    )
                )
    return records
