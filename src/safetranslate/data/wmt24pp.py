"""WMT24++: 998 segments (news, social, speech, literary) with post-edited references."""

import json
import urllib.request
from pathlib import Path

from safetranslate.data.schema import Language, Record

BASE_URL = "https://huggingface.co/datasets/google/wmt24pp/resolve/main"
# WMT24++ has no Malay
FILES = {"zh": "en-zh_CN.jsonl", "ta": "en-ta_IN.jsonl"}


def download(raw_dir: Path) -> None:
    """Downloads the Chinese and Tamil files, skipping ones already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        path = raw_dir / name
        if not path.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)


def load(raw_dir: Path, lang: Language) -> list[Record]:
    """Loads one language, dropping bad sources; returns [] for Malay."""
    if lang not in FILES:
        return []
    records = []
    with open(raw_dir / FILES[lang], encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # Flagged rows include the canary (a leak-detection marker) and broken sources
            if row["is_bad_source"]:
                continue
            records.append(
                Record(
                    id=f"wmt24pp-{lang}-{len(records) + 1:06d}",
                    dataset="wmt24pp",
                    split="test",
                    target_lang=lang,
                    source=row["source"].strip(),
                    target=row["target"].strip(),
                    domain=row["domain"],
                    doc_id=row["document_id"],
                )
            )
    return records
