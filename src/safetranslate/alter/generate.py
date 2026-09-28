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
EXAMPLES: dict[Category, str] = {
    "wrong_quantity": 'find "500 mg", replace_with "50 mg"',
    "wrong_name": 'find "Johor", replace_with "Melaka"',
    "flipped_meaning": 'find "Jangan ambil", replace_with "Ambil"',
    "removed_information": 'find " sehari", replace_with "", missing_english "a day"',
    "added_information": 'find "sehari", replace_with "sehari selepas makan"',
}
Parsed = TypeVar("Parsed", bound=BaseModel)


class Edit(BaseModel):
    """One find-and-replace edit that creates one critical error."""

    category: Category
    find: str
    replace_with: str
    missing_english: str = ""
    description: str


class Alteration(BaseModel):
    """What the alteration LLM returns: the edits, not the rewritten translation."""

    applicable: bool
    edits: list[Edit]


class Verdict(BaseModel):
    """What the verifier returns, judged blind (without seeing the claimed error).

    The reason comes first so the model works it out before committing to a verdict.
    """

    reason: str
    meaning_changed: bool
    categories: list[Category]


def make_client(endpoint: LLMEndpoint) -> OpenAI:
    """Creates a client for an OpenAI-compatible server."""
    api_key = os.environ[endpoint.api_key_env] if endpoint.api_key_env else "not-needed"
    # Extra retries with back-off ride out rate limits (429) and brief server errors (500)
    return OpenAI(base_url=endpoint.base_url, api_key=api_key, max_retries=6)


def ask_json(
    client: OpenAI,
    model: str,
    prompt: str,
    schema: type[Parsed],
    json_schema: dict | None = None,
    extra_body: dict | None = None,
) -> Parsed:
    """Sends one prompt and parses the reply into the given pydantic model.

    Args:
        json_schema: Stricter schema to enforce than the model's own, if given.
        extra_body: Server-specific request options (see LLMEndpoint.extra_body).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # Low randomness helps the model copy text exactly
        temperature=0.3,
        # Stops a runaway reply instead of letting it fill the context
        max_tokens=1024,
        # Constrains the server to produce JSON matching the schema, so parsing can't fail
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": json_schema or schema.model_json_schema(),
            },
        },
        extra_body=extra_body or {},
    )
    return schema.model_validate_json(extract_json(response.choices[0].message.content or ""))


def extract_json(text: str) -> str:
    """Cuts the JSON object out of a reply.

    Servers that don't strictly enforce the schema (e.g. Gemma via Google's API) wrap it
    in reasoning ("...</thought>") or markdown code fences.
    """
    text = text.split("</thought>")[-1]
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def alteration_schema(categories: list[Category]) -> dict:
    """The Alteration schema, narrowed to at most one edit per requested category."""
    schema = Alteration.model_json_schema()
    schema["properties"]["edits"]["maxItems"] = len(categories)
    schema["$defs"]["Edit"]["properties"]["category"] = {"enum": list(categories)}
    return schema


def alteration_prompt(
    source: str, translation: str, lang: str, category: Category, feedback: str = ""
) -> str:
    """Asks for one find-and-replace edit that creates one critical error."""
    return f"""You are creating test data for a translation error detector. Your job is to
CREATE an error by editing a translation. Do not look for existing errors.

English: {source}
{LANGUAGE_NAMES[lang]} translation: {translation}

Create exactly one critical error of this kind, with one edit:
- {category}: {DEFINITIONS[category]}

The edit is a find-and-replace on the {LANGUAGE_NAMES[lang]} translation:
- "find": a short phrase copied EXACTLY, character for character, from the
  {LANGUAGE_NAMES[lang]} translation above. Never copy it from the English or the example.
- "replace_with": what that phrase becomes, in {LANGUAGE_NAMES[lang]}. It must be
  different from "find".
- For removed_information: "replace_with" is "" (delete the phrase), and
  "missing_english" quotes, exactly, the English words that are now missing. Leave
  "missing_english" as "" for the other categories.
- "description": what the English says and what the translation now says.

Example for a different sentence (do not reuse its words). English: "Do not take more
than 500 mg a day in Johor." Malay: "Jangan ambil lebih daripada 500 mg sehari di Johor."
- {category}: {EXAMPLES[category]}

If this error cannot be made naturally in this sentence, set "applicable" to false and
"edits" to [].
{feedback}"""


def verification_prompt(alteration: dict) -> str:
    """Asks a second model, blind to the claimed error, whether the meaning changed.

    Showing the claim and asking "is this correct?" made the verifier agree with
    everything, so it only sees the two translations and decides for itself.
    """
    categories = "\n".join(f"- {c}: {d}" for c, d in DEFINITIONS.items())
    return f"""Compare two translations of the same English sentence.

English: {alteration["source"]}
Translation A: {alteration["original"]}
Translation B: {alteration["altered"]}

Does Translation B change what a reader would understand, compared with Translation A?
Differences in wording, style, spelling or punctuation that keep the same meaning do NOT
count. First explain the difference in "reason". Then set "meaning_changed", and list in
"categories" each kind of meaning change B contains (empty if none):
{categories}
"""


def read_jsonl(path: Path) -> list[dict]:
    """Reads a JSONL file, skipping a line cut off by a crash or power cut."""
    rows = []
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def run_resumable(
    tasks: Iterable[tuple[str, Callable[[], dict | None]]], out_path: Path, max_workers: int
) -> int:
    """Runs (task_id, job) pairs in parallel, appending each result to a JSONL file.

    Task IDs already in the file are skipped, so an interrupted run can be restarted.
    Jobs that raise are not written, so they are retried on the next run.

    Returns:
        How many jobs failed.
    """
    done = {row["task_id"] for row in read_jsonl(out_path)}
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
    # End a line cut off by a crash, so the next result isn't glued onto it
    if out_path.exists() and not out_path.read_bytes().endswith(b"\n") and out_path.stat().st_size:
        with out_path.open("a", encoding="utf-8") as f:
            f.write("\n")
    failed = 0
    with ThreadPoolExecutor(max_workers) as pool, out_path.open("a", encoding="utf-8") as f:
        for result in pool.map(safe, todo):
            if result is None:
                failed += 1
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    return failed
