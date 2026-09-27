"""LLM calls: making alterations and verifying them. Works with any OpenAI-compatible API."""

import json
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from safetranslate.config import LLMEndpoint
from safetranslate.data.schema import Category, ErrorItem, Record

LANGUAGE_NAMES = {"zh": "Simplified Chinese", "ms": "Malay", "ta": "Tamil"}
DEFINITIONS: dict[Category, str] = {
    "wrong_quantity": "change a number or unit (e.g. 500 mg -> 500 g, 2 -> 20, 14 days -> 4 days)",
    "wrong_name": "replace a name (person, place, organisation, medicine) with a different real one",
    "flipped_meaning": "reverse the meaning, e.g. add or remove a negation (do -> do not)",
    "removed_information": "delete a piece of information, such as a warning, condition or detail",
    "added_information": "add a plausible piece of information that is not in the English",
}
Parsed = TypeVar("Parsed", bound=BaseModel)


class Alteration(BaseModel):
    """What the alteration LLM returns."""

    applicable: bool
    altered_translation: str
    errors: list[ErrorItem]


class Verdict(BaseModel):
    """What the verifier LLM returns."""

    valid: bool


def make_client(endpoint: LLMEndpoint) -> OpenAI:
    """Creates a client for an OpenAI-compatible server."""
    api_key = os.environ[endpoint.api_key_env] if endpoint.api_key_env else "not-needed"
    return OpenAI(base_url=endpoint.base_url, api_key=api_key)


def ask_json(client: OpenAI, model: str, prompt: str, schema: type[Parsed]) -> Parsed:
    """Sends one prompt and parses the reply into the given pydantic model."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        # Constrains the server to produce JSON matching the schema, so parsing can't fail
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        },
        # Qwen3's "thinking" mode writes long reasoning first; not needed for small edits
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return schema.model_validate_json(response.choices[0].message.content or "")


def alteration_prompt(record: Record, categories: list[Category]) -> str:
    """Asks for a translation containing the requested critical errors, and nothing else."""
    wanted = "\n".join(f"- {c}: {DEFINITIONS[c]}" for c in categories)
    return f"""You are creating test data for a translation error detector.

English: {record.source}
Correct {LANGUAGE_NAMES[record.target_lang]} translation: {record.target}

Rewrite the correct translation so it contains exactly these critical errors (one each):
{wanted}

Rules:
- Change as little as possible; keep everything else identical and fluent.
- For each error, "span" must quote the changed text exactly as it appears in your
  rewritten translation. For removed_information, quote the English words that are now
  missing instead.
- "description" briefly states what the English says and what the translation now says.
- If any requested error cannot be made naturally for this sentence, set "applicable" to
  false and leave the other fields empty.
"""


def verification_prompt(alteration: dict, error: ErrorItem) -> str:
    """Asks a second model whether one claimed error is real."""
    return f"""You are checking test data for a translation error detector.

English: {alteration["source"]}
Correct translation: {alteration["original"]}
Altered translation: {alteration["altered"]}

Claimed critical error in the altered translation:
- category: {error.category} ({DEFINITIONS[error.category]})
- span: {error.span}
- description: {error.description}

Is this claim correct: does the altered translation really differ from the correct one in
this way, so that a reader would understand something different from the English?
Answer with "valid": true or false.
"""


def run_resumable(
    tasks: Iterable[tuple[str, Callable[[], dict | None]]], out_path: Path, max_workers: int
) -> int:
    """Runs (task_id, job) pairs in parallel, appending each result to a JSONL file.

    Task IDs already in the file are skipped, so an interrupted run can be restarted.
    Jobs that raise are not written, so they are retried on the next run.

    Returns:
        How many jobs failed.
    """
    done = set()
    if out_path.exists():
        done = {json.loads(line)["task_id"] for line in out_path.open(encoding="utf-8")}
    todo = [(task_id, job) for task_id, job in tasks if task_id not in done]

    def safe(item: tuple[str, Callable[[], dict | None]]) -> dict | None:
        task_id, job = item
        try:
            result = job()
        except Exception as err:  # one bad reply shouldn't stop an overnight run
            print(f"{task_id} failed: {err}")
            return None
        return {"task_id": task_id, **result} if result is not None else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(max_workers) as pool, out_path.open("a", encoding="utf-8") as f:
        for result in pool.map(safe, todo):
            if result is None:
                failed += 1
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    return failed
