"""Real machine translations with human critical-error labels, for testing evaluators.

WMT21 Critical Error Detection (Chinese): 1,000 test translations labelled ERR/NOT.
IndicMT Eval (Tamil): 1,475 translations with MQM error spans; counted as critical when
an Accuracy error (mistranslation, omission, addition, untranslated) is High or Very High.
"""

import json
import re
import tarfile
from pathlib import Path

from safetranslate.data.download import fetch
from safetranslate.data.schema import LabelledTranslation

# Pinned to commits so the data cannot silently change upstream
WMT21_URL = "https://raw.githubusercontent.com/sheffieldnlp/mlqe-pe/2a670a1140416cf80507b5a829659383c878feb8/data"
INDICMT_URL = "https://raw.githubusercontent.com/AI4Bharat/IndicMT-Eval/d8a8666b7e79975d0c3490e433e3a2b98961e86e/Dataset/Indic%20MT%20Eval"
INDICMT_PARTS = ["train", "val", "test"]
CRITICAL_SEVERITIES = {"High", "Very High"}
# WMT21's Chinese output is tokenised with spaces between words; real Chinese has none
SPACE_BETWEEN_CJK = re.compile(r"(?<=[^\x00-\x7F]) (?=[^\x00-\x7F])")


def download(raw_dir: Path) -> None:
    """Downloads both datasets, skipping files already present."""
    fetch(f"{WMT21_URL}/catastrophic_errors/enzh_majority_test_blind.tsv",
          raw_dir / "wmt21_ced" / "test_blind.tsv")
    fetch(f"{WMT21_URL}/catastrophic_errors_goldlabels/enzh_majority_test_goldlabels.tar.gz",
          raw_dir / "wmt21_ced" / "goldlabels.tar.gz")
    for part in INDICMT_PARTS:
        fetch(f"{INDICMT_URL}/Tam_{part}.jsonl", raw_dir / "indicmt" / f"Tam_{part}.jsonl")


def load_wmt21(raw_dir: Path) -> list[LabelledTranslation]:
    """Joins the blind test file with its gold labels by row id."""
    with tarfile.open(raw_dir / "wmt21_ced" / "goldlabels.tar.gz") as tar:
        gold_file = tar.extractfile("enzh_majority_test_goldlabels/goldlabels.txt")
        labels = {line.split("\t")[2]: line.split("\t")[3].strip()
                  for line in gold_file.read().decode("utf-8").splitlines()}
    examples = []
    for line in (raw_dir / "wmt21_ced" / "test_blind.tsv").read_text(encoding="utf-8").splitlines():
        row_id, source, translation = line.split("\t")[:3]
        examples.append(LabelledTranslation(
            id=f"wmt21_ced-zh-{row_id}", dataset="wmt21_ced", target_lang="zh",
            source=source.strip(), translation=SPACE_BETWEEN_CJK.sub("", translation.strip()),
            critical=labels[row_id] == "ERR",
        ))
    return examples


def load_indicmt(raw_dir: Path) -> list[LabelledTranslation]:
    """All Tamil annotations (train/val/test are used together, for evaluation only)."""
    examples = []
    for part in INDICMT_PARTS:
        path = raw_dir / "indicmt" / f"Tam_{part}.jsonl"
        for i, row in enumerate(map(json.loads, path.open(encoding="utf-8"))):
            critical = any(
                span["span_type"].startswith("Accuracy")
                and span["span_severity"] in CRITICAL_SEVERITIES
                for span in row["completion"]
            )
            examples.append(LabelledTranslation(
                id=f"indicmt-ta-{part}-{i:05d}", dataset="indicmt", target_lang="ta",
                source=row["src"].strip(), translation=row["translation"].strip(),
                critical=critical,
            ))
    return examples
