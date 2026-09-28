"""The two jobs a model can be tested on: judging a translation, and translating."""

from openai import OpenAI
from pydantic import BaseModel, Field

from safetranslate.alter.generate import DEFINITIONS, LANGUAGE_NAMES, ask_json
from safetranslate.config import LLMEndpoint
from safetranslate.data.schema import ErrorItem, Language


# A single sentence won't have more; the cap stops runaway lists that hit the length limit
MAX_ERRORS = 10


class Judgement(BaseModel):
    """What a judge returns: the translation's critical errors, empty if none."""

    errors: list[ErrorItem] = Field(max_length=MAX_ERRORS)


class Translation(BaseModel):
    """What a translator returns."""

    translation: str


def judge_prompt(source: str, translation: str, lang: Language) -> str:
    """Asks for every critical error in a translation, and nothing else."""
    categories = "\n".join(f"- {c}: {d}" for c, d in DEFINITIONS.items())
    return f"""You are checking a {LANGUAGE_NAMES[lang]} translation of English for critical errors:
errors that change what a reader would understand or do. Clumsy wording, typos and
style are NOT critical errors.

English: {source}
Translation: {translation}

Critical error categories (what the translation did compared with the English):
{categories}

List every critical error. For each: its category; "span" quoting the erroneous text
exactly from the translation (for removed_information, quote the English words that are
missing instead); and a short description. If there are none, return an empty list.
"""


def judge(
    client: OpenAI, endpoint: LLMEndpoint, source: str, translation: str, lang: Language
) -> list[ErrorItem]:
    """Returns the critical errors a model finds in one translation.

    No reference translation is given: the evaluator won't have one in real use, and the
    judge's measured accuracy (used to correct its counts) must come from the same setup.
    """
    prompt = judge_prompt(source, translation, lang)
    return ask_json(
        client, endpoint.model, prompt, Judgement, extra_body=endpoint.extra_body
    ).errors


def translate(
    client: OpenAI,
    endpoint: LLMEndpoint,
    source: str,
    lang: Language,
    feedback: list[ErrorItem] | None = None,
) -> str:
    """Returns a model's translation of one English text.

    Args:
        feedback: Critical errors found in a previous attempt, to avoid this time.
    """
    prompt = (
        f"Translate this English text into {LANGUAGE_NAMES[lang]}. Keep all information and "
        f"meaning; add nothing.\n\nEnglish: {source}"
    )
    if feedback:
        problems = "\n".join(f"- {e.category}: {e.description}" for e in feedback)
        prompt += f"\n\nA previous translation had these critical errors; avoid them:\n{problems}"
    return ask_json(
        client, endpoint.model, prompt, Translation, extra_body=endpoint.extra_body
    ).translation
