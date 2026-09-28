"""The translation workflow: translate, check, retry with feedback, escalate.

For each sentence:
    translate -> evaluate -> no critical errors?  -> publish
                          -> errors, attempts left -> notify, retry translate with feedback
                          -> errors, no attempts left -> escalate to a human

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
        status: "published" or "escalated" once finished.
    """

    source: str
    lang: Language
    translation: str
    errors: list[ErrorItem]
    attempts: int
    history: list[dict]
    notifications: list[str]
    status: Literal["published", "escalated"]


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

    def route(state: SegmentState) -> str:
        if not state["errors"]:
            return "publish"
        return "notify" if state["attempts"] < max_attempts else "escalate"

    def notify_node(state: SegmentState) -> SegmentState:
        kinds = ", ".join(sorted({e.category for e in state["errors"]}))
        message = f"Attempt {state['attempts']} had critical errors ({kinds}); retrying."
        return {"notifications": state.get("notifications", []) + [message]}

    def publish_node(state: SegmentState) -> SegmentState:
        return {"status": "published"}

    def escalate_node(state: SegmentState) -> SegmentState:
        message = f"Still had critical errors after {state['attempts']} attempts; needs human review."
        return {"status": "escalated", "notifications": state.get("notifications", []) + [message]}

    graph = StateGraph(SegmentState)
    graph.add_node("translate", translate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("notify", notify_node)
    graph.add_node("publish", publish_node)
    graph.add_node("escalate", escalate_node)
    graph.add_edge(START, "translate")
    graph.add_edge("translate", "evaluate")
    graph.add_conditional_edges("evaluate", route, ["publish", "notify", "escalate"])
    graph.add_edge("notify", "translate")
    graph.add_edge("publish", END)
    graph.add_edge("escalate", END)
    return graph.compile()


# A sentence ends at . ! ? or Chinese 。！？, followed by space or the end of the line
SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


def split_document(text: str) -> list[tuple[int, str]]:
    """Splits text into (paragraph number, sentence) pairs, keeping paragraph order."""
    pieces = []
    for number, paragraph in enumerate(p for p in text.split("\n") if p.strip()):
        pieces += [(number, s.strip()) for s in SENTENCE_END.split(paragraph.strip()) if s.strip()]
    return pieces


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
