"""Common record format that every dataset loader must produce."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

# The three target languages we support for translation.
# zh = Chinese, ms = Malay, ta = Tamil
Language = Literal["zh", "ms", "ta"]

Split = Literal["train", "validation", "test"]


class Record(BaseModel):
    """One English sentence / passage and its translation.

    Attributes:
        id: Unique record ID, e.g. "wikimedia-ms-000001".
        dataset: Dataset the pair came from, e.g. "ntrex".
        split: Whether the pair is used for training, validation or testing.
        target_lang: Language of the translation.
        source: The English text.
        target: The translation.
        domain: Topic area where the dataset provides one, e.g. "wikinews".
        doc_id: Document the sentence belongs to, where the dataset provides one.
    """

    # Reject unknown fields
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: Split
    target_lang: Language
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    domain: str | None = None
    doc_id: str | None = None


Category = Literal[
    "wrong_quantity", "wrong_name", "flipped_meaning", "removed_information", "added_information"
]


class ErrorItem(BaseModel):
    """One critical error, as the evaluator reports it.

    Attributes:
        category: Which of the five critical-error categories it is.
        span: Exact quoted text of the error. Quoted from the translation, except for
            removed_information, where it quotes the English that is missing.
        description: Short explanation, e.g. "Source says 500 mg; translation says 500 g".
    """

    category: Category
    span: str = Field(min_length=1)
    description: str


class EvaluatorExample(BaseModel):
    """An English sentence, a translation of it, and the translation's critical errors.

    Attributes:
        id: Unique example ID.
        origin_id: ID of the Record the example was made from.
        split: Whether the example is used for training, validation or testing.
        target_lang: Language of the translation.
        source: The English text.
        translation: The translation being judged.
        errors: Its critical errors; empty means the translation has none.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    origin_id: str
    split: Split
    target_lang: Language
    source: str
    translation: str
    errors: list[ErrorItem]
