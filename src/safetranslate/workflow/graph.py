"""The translation workflow: translate, check, retry with feedback, escalate.

For each sentence:
    translate -> evaluate -> no critical errors?  -> publish
                          -> errors, attempts left -> notify, retry translate with feedback
                          -> errors, no attempts left -> escalate to a human

Checking a translation someone already has (check_only) starts at evaluate and ends
there: there's no translator to retry.

A document is split into sentences, each runs through the graph in parallel, and the
results are put back together in order.
"""

import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from safetranslate.config import LLMEndpoint
from safetranslate.data.schema import ErrorItem, Language
from safetranslate.harness.models import judge, translate


class SegmentState(TypedDict, total=False):
    """Everything the graph knows about one sentence as it moves through the steps.

    Attributes:
        source: The English sentence.
        lang: Target language.
        translation: The latest translation.
        errors: Critical errors the evaluator found in the latest translation.
        attempts: Translations made so far.
        history: Every attempt, with its errors (for the UI and for measuring retries).
        notifications: Messages for the officer, e.g. "attempt 1 had wrong_quantity".
        check_only: The translation was supplied: only check it.
        status: "published", "escalated" or "checked" once finished.
    """

    source: str
    lang: Language
    translation: str
    errors: list[ErrorItem]
    attempts: int
    history: list[dict]
    notifications: list[str]
    check_only: bool
    status: Literal["published", "escalated", "checked"]


def build_graph(
    translator: tuple[OpenAI, LLMEndpoint], evaluator: tuple[OpenAI, LLMEndpoint], max_attempts: int = 3
):
    """Builds the per-sentence graph with the given translator and evaluator models."""

    def translate_node(state: SegmentState) -> SegmentState:
        # On a retry, the previous attempt's errors are passed back as feedback
        text = translate(*translator, state["source"], state["lang"], state.get("errors"))
        return {"translation": text, "attempts": state.get("attempts", 0) + 1}

    def evaluate_node(state: SegmentState) -> SegmentState:
        errors = judge(*evaluator, state["source"], state["translation"], state["lang"])
        attempt = {"translation": state["translation"], "errors": [e.model_dump() for e in errors]}
        return {"errors": errors, "history": state.get("history", []) + [attempt]}

    def start(state: SegmentState) -> str:
        return "evaluate" if state.get("check_only") else "translate"

    def route(state: SegmentState) -> str:
        if state.get("check_only"):
            return "checked"
        if not state["errors"]:
            return "publish"
        return "notify" if state["attempts"] < max_attempts else "escalate"

    def notify_node(state: SegmentState) -> SegmentState:
        kinds = ", ".join(sorted({e.category for e in state["errors"]}))
        message = f"Attempt {state['attempts']} had critical errors ({kinds}); retrying."
        return {"notifications": state.get("notifications", []) + [message]}

    def publish_node(state: SegmentState) -> SegmentState:
        return {"status": "published"}

    def checked_node(state: SegmentState) -> SegmentState:
        return {"status": "checked"}

    def escalate_node(state: SegmentState) -> SegmentState:
        message = f"Still had critical errors after {state['attempts']} attempts; needs human review."
        return {"status": "escalated", "notifications": state.get("notifications", []) + [message]}

    graph = StateGraph(SegmentState)
    graph.add_node("translate", translate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("notify", notify_node)
    graph.add_node("publish", publish_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("checked", checked_node)
    graph.add_conditional_edges(START, start, ["translate", "evaluate"])
    graph.add_edge("translate", "evaluate")
    graph.add_conditional_edges("evaluate", route, ["publish", "notify", "escalate", "checked"])
    graph.add_edge("notify", "translate")
    graph.add_edge("publish", END)
    graph.add_edge("escalate", END)
    graph.add_edge("checked", END)
    return graph.compile()


# A sentence ends at . ! ? followed by a space, or at Chinese 。！？ (no space follows them)
SENTENCE_END = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+")


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n") if p.strip()]


def sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(paragraph) if s.strip()]


def split_document(text: str) -> list[tuple[int, str]]:
    """Splits text into (paragraph number, sentence) pairs, keeping paragraph order."""
    return [(number, s) for number, p in enumerate(paragraphs(text)) for s in sentences(p)]


def pair_document(english: str, translation: str) -> tuple[list[dict], tuple[int, int]]:
    """Pairs English with its translation for checking, without ever guessing.

    Paragraphs are paired by line breaks (MT tools keep them). Within a paragraph,
    sentences are paired one-to-one if the counts match; if the MT merged or split
    sentences, the whole paragraph is one pair. If the paragraph counts differ, the whole
    text is one pair.

    Returns:
        Pairs ({"source", "translation", "unit": sentence/paragraph/text}) in order, and
        the (English, translation) paragraph counts.
    """
    en, tr = paragraphs(english), paragraphs(translation)
    if len(en) != len(tr):
        return [{"source": english.strip(), "translation": translation.strip(), "unit": "text"}], (len(en), len(tr))
    pairs = []
    for e, t in zip(en, tr):
        es, ts = sentences(e), sentences(t)
        if len(es) == len(ts):
            pairs += [{"source": a, "translation": b, "unit": "sentence"} for a, b in zip(es, ts)]
        else:
            pairs.append({"source": e, "translation": t, "unit": "paragraph"})
    return pairs, (len(en), len(tr))


def translate_document(graph, text: str, lang: Language, max_concurrency: int = 8) -> dict:
    """Runs every sentence through the graph in parallel and reassembles the document.

    Returns:
        The translated text (paragraphs preserved) and each sentence's final state.
    """
    pieces = split_document(text)
    states = graph.batch(
        [{"source": sentence, "lang": lang} for _, sentence in pieces],
        config={"max_concurrency": max_concurrency},
    )
    paragraphs: dict[int, list[str]] = {}
    for (number, _), state in zip(pieces, states):
        paragraphs.setdefault(number, []).append(state["translation"])
    joiner = "" if lang == "zh" else " "  # Chinese has no spaces between sentences
    return {
        "translation": "\n".join(joiner.join(sentences) for sentences in paragraphs.values()),
        "segments": states,
    }


def check_document(graph, english: str, translation: str, lang: Language, max_concurrency: int = 8) -> dict:
    """Checks someone's own translation for critical errors, pair by pair, in parallel.

    Returns:
        Each pair's final state (with its "unit"), and the paragraph counts.
    """
    pairs, counts = pair_document(english, translation)
    states = graph.batch(
        [{"source": p["source"], "translation": p["translation"], "lang": lang, "check_only": True} for p in pairs],
        config={"max_concurrency": max_concurrency},
    )
    return {"segments": [s | {"unit": p["unit"]} for p, s in zip(pairs, states)], "paragraphs": counts}
