import zipfile
from pathlib import Path

from safetranslate.data import wikimedia


def test_load_pairs_lines_and_skips_empty(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "en-ms.txt.zip", "w") as z:
        z.writestr("wikimedia.en-ms.en", "Wash your hands.\n\nStay home.\n")
        z.writestr("wikimedia.en-ms.ms", "Basuh tangan anda.\nKosong\nDuduk di rumah.\n")
    records = wikimedia.load(tmp_path, "ms")
    assert [(r.source, r.target) for r in records] == [
        ("Wash your hands.", "Basuh tangan anda."),
        ("Stay home.", "Duduk di rumah."),
    ]
    assert records[0].split == "train"
