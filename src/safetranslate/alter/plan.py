"""Chooses which sentences to alter and with which error categories."""

import random
import re
from typing import get_args

from safetranslate.data.schema import Category, Record

CATEGORIES: list[Category] = list(get_args(Category))

# Screening runs on the English source, so no Chinese/Malay/Tamil-specific rules are needed
NUMBER = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|"
    r"hundred|thousand|million|billion|half|twice|percent)\b",
    re.IGNORECASE,
)
NAME = re.compile(r"\s[A-Z][a-z]+")  # a capitalised word that isn't the first word
MIN_WORDS_FOR_REMOVAL = 8


def eligible_categories(source: str) -> list[Category]:
    """Returns the categories an English sentence can plausibly take, rarest first."""
    categories: list[Category] = []
    if NUMBER.search(source):
        categories.append("wrong_quantity")
    if NAME.search(source):
        categories.append("wrong_name")
    if len(source.split()) >= MIN_WORDS_FOR_REMOVAL:
        categories.append("removed_information")
    return categories + ["flipped_meaning", "added_information"]


def plan(
    records: list[Record], per_category: int, multi_error_share: float, rng: random.Random
) -> list[tuple[Record, list[Category]]]:
    """Picks up to per_category sentences for each category, each sentence used once.

    Rare categories pick first, so common ones don't use up the sentences they need.
    """
    pool = records[:]
    rng.shuffle(pool)
    used: set[str] = set()
    tasks: list[tuple[Record, list[Category]]] = []
    for category in CATEGORIES:
        picks = [
            r for r in pool if r.id not in used and category in eligible_categories(r.source)
        ][:per_category]
        for record in picks:
            used.add(record.id)
            tasks.append((record, [category]))

    # Some examples get 1-2 extra errors so the evaluator learns to report a list
    for record, categories in rng.sample(tasks, round(len(tasks) * multi_error_share)):
        extra = [c for c in eligible_categories(record.source) if c not in categories]
        categories += rng.sample(extra, min(len(extra), rng.randint(1, 2)))
    return tasks
