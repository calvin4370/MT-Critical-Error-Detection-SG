"""Removes training pairs whose English overlaps validation or test sentences."""

import re
from collections import Counter

from safetranslate.data.schema import Record


def normalise(text: str) -> str:
    """Lowercases, replaces punctuation with spaces and collapses whitespace."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = text.split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def remove_overlap(
    train: list[Record], held_out: list[Record], ngram_size: int
) -> tuple[list[Record], Counter[str]]:
    """Drops training pairs matching a held-out English sentence.

    A pair is dropped if its normalised English equals a held-out sentence (catches short
    sentences) or shares any ngram_size-word sequence with one (catches near-copies).

    Returns:
        The kept records, and how many were dropped by each check.
    """
    held_out_texts = {normalise(r.source) for r in held_out}
    held_out_ngrams = set().union(*(_ngrams(t, ngram_size) for t in held_out_texts))
    kept: list[Record] = []
    dropped: Counter[str] = Counter()
    for record in train:
        text = normalise(record.source)
        if text in held_out_texts:
            dropped["exact"] += 1
        elif _ngrams(text, ngram_size) & held_out_ngrams:
            dropped["ngram"] += 1
        else:
            kept.append(record)
    return kept, dropped
