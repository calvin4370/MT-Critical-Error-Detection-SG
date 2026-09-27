"""The two jobs a model can be tested on: judging a translation, and translating."""

from openai import OpenAI
from pydantic import BaseModel

from safetranslate.alter.generate import DEFINITIONS, LANGUAGE_NAMES, ask_json
from safetranslate.config import LLMEndpoint
from safetranslate.data.schema import ErrorItem, Language


class Judgement(BaseModel):
    """What a judge returns: the translation's critical errors, empty if none."""

    errors: list[ErrorItem]


class Translation(BaseModel):
    """What a translator returns."""

    translation: str


def judge_prompt(source: str, translation: str, lang: Language, reference: str | None) -> str:
    """Asks for every critical error in a translation, and nothing else."""
    categories = "\n".join(f"- {c}: {d}" for c, d in DEFINITIONS.items())
    reference_note = (
        f"\nA professional reference translation, for comparison: {reference}\n"
        "Different wording from the reference is fine; only differences in meaning count.\n"
        if reference
        else ""
    )
    return f"""You are checking a {LANGUAGE_NAMES[lang]} translation of English for critical errors:
errors that change what a reader would understand or do. Clumsy wording, typos and
style are NOT critical errors.

English: {source}
Translation: {translation}
{reference_note}
Critical error categories (what the translation did compared with the English):
{categories}

List every critical error. For each: its category; "span" quoting the erroneous text
exactly from the translation (for removed_information, quote the English words that are
missing instead); and a short description. If there are none, return an empty list.
"""


def judge(
    client: OpenAI,
    endpoint: LLMEndpoint,
    source: str,
    translation: str,
    lang: Language,
    reference: str | None = None,
) -> list[ErrorItem]:
    """Returns the critical errors a model finds in one translation."""
    prompt = judge_prompt(source, translation, lang, reference)
    return ask_json(
        client, endpoint.model, prompt, Judgement, extra_body=endpoint.extra_body
    ).errors


def translate(client: OpenAI, endpoint: LLMEndpoint, source: str, lang: Language) -> str:
    """Returns a model's translation of one English text."""
    prompt = (
        f"Translate this English text into {LANGUAGE_NAMES[lang]}. Keep all information and "
        f"meaning; add nothing.\n\nEnglish: {source}"
    )
    return ask_json(
        client, endpoint.model, prompt, Translation, extra_body=endpoint.extra_body
    ).translation
