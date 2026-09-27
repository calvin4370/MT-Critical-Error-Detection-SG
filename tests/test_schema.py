import pytest
from pydantic import ValidationError

from safetranslate.data.schema import Record

VALID = {
    "id": "ntrex-ms-000001",
    "dataset": "ntrex",
    "split": "test",
    "target_lang": "ms",
    "source": "Wash your hands.",
    "target": "Basuh tangan anda.",
}


def test_valid_record() -> None:
    record = Record(**VALID)
    assert record.domain is None
    assert record.doc_id is None


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("target_lang", "id"),  # Indonesian, not one of our languages
        ("split", "dev"),  # must be "validation"
        ("source", ""),
        ("target", ""),
    ],
)
def test_invalid_values_rejected(field: str, bad_value: str) -> None:
    with pytest.raises(ValidationError):
        Record(**{**VALID, field: bad_value})


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Record(**VALID, langauge="ms")
