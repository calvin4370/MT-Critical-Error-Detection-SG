from pathlib import Path

from safetranslate.data import flores, tico19, wmt24pp

FIXTURES = Path(__file__).parent / "fixtures"


def test_tico19_joins_parts_and_undoes_doubled_quotes() -> None:
    records = tico19.load(FIXTURES / "tico19", "ta")
    assert [r.source for r in records] == ["Wash your hands.", "Stay home."]
    assert records[0].target == 'அவர் "கைகளைக் கழுவவும்" என்றார்.'
    assert records[1].doc_id == "PubMed_7"


def test_wmt24pp_drops_bad_sources_and_has_no_malay() -> None:
    records = wmt24pp.load(FIXTURES / "wmt24pp", "zh")
    assert len(records) == 1
    assert records[0].target == "请洗手。"  # the post-edited reference
    assert wmt24pp.load(FIXTURES / "wmt24pp", "ms") == []


def test_flores_pairs_by_id_and_marks_dev_as_validation() -> None:
    records = flores.load(FIXTURES / "flores", "ms", "dev")
    assert [(r.source, r.target) for r in records] == [
        ("Wash your hands.", "Basuh tangan anda."),
        ("Stay home.", "Duduk di rumah."),
    ]
    assert records[0].split == "validation"
