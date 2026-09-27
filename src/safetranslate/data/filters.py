"""Noise filters for training pairs (rules decided in Phase 1)."""

import re
from collections import Counter

import opencc

from safetranslate.data.schema import Record

CITATION_MARK = re.compile(r"\[\d+\]")
# Test sets use Simplified Chinese. t2s only detects Traditional characters; tw2sp also
# converts Taiwan vocabulary, but would rewrite correct mainland words if run on
# already-Simplified text (e.g. 文件 -> 文档), so it's only used on Traditional text
TRADITIONAL_CHARS = opencc.OpenCC("t2s")
TO_SIMPLIFIED = opencc.OpenCC("tw2sp")
TAIWAN_QUOTES = str.maketrans({"「": "“", "」": "”", "『": "‘", "』": "’"})


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
        The kept records, and per-rule counts: how many each rule dropped, plus how
        many Chinese targets were converted to Simplified (before the drop rules).
    """
    junk = re.compile("|".join(junk_patterns))
    kept: list[Record] = []
    counts: Counter[str] = Counter()
    for record in records:
        if strip_citation_marks:
            source = CITATION_MARK.sub("", record.source).strip()
            target = CITATION_MARK.sub("", record.target).strip()
            if not source or not target:
                counts["empty"] += 1
                continue
            record = record.model_copy(update={"source": source, "target": target})
        if record.target_lang == "zh":
            target = record.target
            if TRADITIONAL_CHARS.convert(target) != target:
                counts["converted_to_simplified"] += 1
                target = TO_SIMPLIFIED.convert(target)
            record = record.model_copy(update={"target": target.translate(TAIWAN_QUOTES)})
        if record.source == record.target:
            counts["identical"] += 1
        elif junk.search(record.source) or junk.search(record.target):
            counts["junk"] += 1
        elif not bounds[0] <= length_ratio(record) <= bounds[1]:
            counts["length_ratio"] += 1
        else:
            kept.append(record)
    return kept, counts
