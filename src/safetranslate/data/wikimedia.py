"""wikimedia (OPUS): Wikipedia articles translated by editors; our training corpus."""

import urllib.request
import zipfile
from pathlib import Path

from safetranslate.data.schema import Language, Record

BASE_URL = "https://object.pouta.csc.fi/OPUS-wikimedia/v20260327/moses"


def download(raw_dir: Path, lang: Language) -> None:
    """Downloads the zipped English/target pair for one language if not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"en-{lang}.txt.zip"
    if not path.exists():
        urllib.request.urlretrieve(f"{BASE_URL}/{path.name}", path)


def load(raw_dir: Path, lang: Language) -> list[Record]:
    """Pairs line N of the English file with line N of the target file, unfiltered."""
    with zipfile.ZipFile(raw_dir / f"en-{lang}.txt.zip") as z:
        sources = z.read(f"wikimedia.en-{lang}.en").decode("utf-8").splitlines()
        targets = z.read(f"wikimedia.en-{lang}.{lang}").decode("utf-8").splitlines()
    if len(sources) != len(targets):
        raise ValueError(f"{len(sources)} English lines but {len(targets)} {lang} lines")
    return [
        Record(
            id=f"wikimedia-{lang}-{i:07d}",
            dataset="wikimedia",
            split="train",
            target_lang=lang,
            source=src.strip(),
            target=tgt.strip(),
        )
        for i, (src, tgt) in enumerate(zip(sources, targets), start=1)
        # Empty pairs can't be Records, so the "drop empty pairs" rule is applied here
        if src.strip() and tgt.strip()
    ]
