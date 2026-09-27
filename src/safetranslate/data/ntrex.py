"""NTREX-128: 1,997 news sentences professionally translated from English."""

import urllib.request
from pathlib import Path

from safetranslate.data.schema import Language, Record

BASE_URL = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128"
SOURCE_FILE = "newstest2019-src.eng.txt"
TARGET_FILES = {
    "zh": "newstest2019-ref.zho-CN.txt",
    "ms": "newstest2019-ref.msa.txt",
    "ta": "newstest2019-ref.tam.txt",
}


def download(raw_dir: Path) -> None:
    """Downloads the English and target files, skipping ones already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in [SOURCE_FILE, *TARGET_FILES.values()]:
        path = raw_dir / name
        if not path.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)


def load(raw_dir: Path, lang: Language) -> list[Record]:
    """Pairs line N of the English file with line N of the target file."""
    sources = (raw_dir / SOURCE_FILE).read_text(encoding="utf-8").splitlines()
    targets = (raw_dir / TARGET_FILES[lang]).read_text(encoding="utf-8").splitlines()
    # Files are aligned by line number, so a count mismatch means every pair is wrong
    if len(sources) != len(targets):
        raise ValueError(f"{len(sources)} English lines but {len(targets)} {lang} lines")
    return [
        Record(
            id=f"ntrex-{lang}-{i:06d}",
            dataset="ntrex",
            split="test",
            target_lang=lang,
            source=src.strip(),
            target=tgt.strip(),
            domain="news",
        )
        for i, (src, tgt) in enumerate(zip(sources, targets), start=1)
    ]
