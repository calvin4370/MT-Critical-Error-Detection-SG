"""Noise filters for training pairs (rules decided in Phase 1)."""

import re
from collections import Counter

from safetranslate.data.schema import Record

CITATION_MARK = re.compile(r"\[\d+\]")


def length_ratio(record: Record) -> float:
    """Target length divided by English length, in characters (works for Chinese too)."""
    return len(record.target) / len(record.source)


def ratio_bounds(reference: list[Record], percentiles: tuple[float, float]) -> tuple[float, float]:
    """Returns the ratios at the given percentiles of professional translations."""
    ratios = sorted(length_ratio(r) for r in reference)
    low, high = (ratios[round(p / 100 * (len(ratios) - 1))] for p in percentiles)
    return low, high


def filter_pairs(
    records: list[Record],
    junk_patterns: list[str],
    bounds: tuple[float, float],
    strip_citation_marks: bool = True,
) -> tuple[list[Record], Counter[str]]:
    """Cleans and filters training pairs.

    Returns:
        The kept records, and how many records each rule dropped.
    """
    junk = re.compile("|".join(junk_patterns))
    kept: list[Record] = []
    dropped: Counter[str] = Counter()
    for record in records:
        if strip_citation_marks:
            source = CITATION_MARK.sub("", record.source).strip()
            target = CITATION_MARK.sub("", record.target).strip()
            if not source or not target:
                dropped["empty"] += 1
                continue
            record = record.model_copy(update={"source": source, "target": target})
        if record.source == record.target:
            dropped["identical"] += 1
        elif junk.search(record.source) or junk.search(record.target):
            dropped["junk"] += 1
        elif not bounds[0] <= length_ratio(record) <= bounds[1]:
            dropped["length_ratio"] += 1
        else:
            kept.append(record)
    return kept, dropped
