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
