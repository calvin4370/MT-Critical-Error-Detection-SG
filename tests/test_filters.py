from safetranslate.data.filters import filter_pairs, ratio_bounds
from safetranslate.data.schema import Record

JUNK = [r"(?:^|\s)(?:User|Pengguna):\S", r"↑"]


def _record(source: str, target: str) -> Record:
    return Record(
        id="x", dataset="wikimedia", split="train", target_lang="ms", source=source, target=target
    )


def test_ratio_bounds_uses_percentiles() -> None:
    # Ratios 1.0, 2.0, ..., 5.0
    reference = [_record("ab", "ab" * n) for n in range(1, 6)]
    assert ratio_bounds(reference, (0, 100)) == (1.0, 5.0)
    assert ratio_bounds(reference, (25, 75)) == (2.0, 4.0)


def test_filter_pairs_applies_each_rule() -> None:
    records = [
        _record("Wash your hands.[1]", "Basuh tangan anda.[1]"),  # kept, marks stripped
        _record("[12]", "[12]"),  # empty after stripping
        _record("Vietnam", "Vietnam"),  # identical
        _record("Vietnam", "Pengguna:Malurian123/Vietnam"),  # junk on one side only
        _record("See also", "Pada akhir bulan Jun 2012, kontraktor memulakan kerja."),  # ratio
    ]
    kept, dropped = filter_pairs(records, JUNK, bounds=(0.5, 2.0))
    assert [r.target for r in kept] == ["Basuh tangan anda."]
    assert dropped == {"empty": 1, "identical": 1, "junk": 1, "length_ratio": 1}
