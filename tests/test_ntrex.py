from pathlib import Path

import pytest

from safetranslate.data import ntrex

# Tiny copies of the NTREX files; the Tamil file is deliberately one line short
FIXTURES = Path(__file__).parent / "fixtures" / "ntrex"


def test_load_pairs_lines() -> None:
    records = ntrex.load(FIXTURES, "zh")
    assert len(records) == 2
    assert records[1].source == "Take 500 mg twice a day."
    assert records[1].target == "每天两次，每次服用500毫克。"
    assert records[1].id == "ntrex-zh-000002"
    assert records[1].split == "test"


def test_line_count_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ntrex.load(FIXTURES, "ta")
