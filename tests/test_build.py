import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from safetranslate.config import load_data_config
from safetranslate.data.build import build

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "data.yaml"


def test_build_writes_splits_and_report(tmp_path: Path) -> None:
    # The fixtures folder mirrors the data/raw layout, so it can stand in for downloads
    shutil.copytree(FIXTURES, tmp_path / "raw")
    (tmp_path / "raw" / "wikimedia").mkdir()
    with zipfile.ZipFile(tmp_path / "raw" / "wikimedia" / "en-ms.txt.zip", "w") as z:
        # Fixture FLORES+ dev ratios are 1.125 and 1.5, so the allowed range is [1.125, 1.5]
        z.writestr("wikimedia.en-ms.en", "Wash your hands.\nDrink water daily.\nRest well.[3]\nSee also\n")
        z.writestr(
            "wikimedia.en-ms.ms",
            "Basuh tangan anda.\nMinum air setiap hari.\nRehat cukup.\nPada akhir bulan Jun 2012.\n",
        )

    config = load_data_config(CONFIG_PATH).model_copy(
        update={
            "data_dir": tmp_path,
            "languages": ["ms"],
            "validation_sets": ["flores_dev"],
            "test_sets": ["ntrex"],
        }
    )
    report = build(config)

    out = tmp_path / "processed"
    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    # "Wash your hands." is also an NTREX test sentence, so the overlap check removes it
    assert [r["source"] for r in train] == ["Drink water daily.", "Rest well."]
    wikimedia_report = report["languages"]["ms"]["training_corpora"]["wikimedia"]
    assert wikimedia_report["filter_counts"] == {"length_ratio": 1}  # "See also"
    assert wikimedia_report["dropped_by_overlap"] == {"exact": 1}
    assert report["files"]["test.jsonl"]["records"] == 2
    assert report["files"]["train.jsonl"]["sha256"] == hashlib.sha256(
        (out / "train.jsonl").read_bytes()
    ).hexdigest()
    assert (out / "report.json").exists()
