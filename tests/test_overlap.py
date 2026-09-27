from safetranslate.data.overlap import normalise, remove_overlap
from safetranslate.data.schema import Record

TEST_SENTENCE = "The incubation period is typically between 2 and 14 days, though longer cases exist."


def _record(source: str, split: str = "train") -> Record:
    return Record(
        id="x", dataset="d", split=split, target_lang="ms", source=source, target="t"
    )


def test_normalise() -> None:
    assert normalise("  Wash  your HANDS! ") == "wash your hands"


def test_remove_overlap() -> None:
    held_out = [_record("Wash your hands.", "test"), _record(TEST_SENTENCE, "test")]
    train = [
        _record("wash your hands"),  # exact after normalising
        _record("Experts say the incubation period is typically between 2 and 14 days, "
                "though longer cases exist in rare patients."),  # shares 13 words
        _record("Wash your hands with soap."),  # different sentence, kept
    ]
    kept, dropped = remove_overlap(train, held_out, ngram_size=13)
    assert [r.source for r in kept] == ["Wash your hands with soap."]
    assert dropped == {"exact": 1, "ngram": 1}
