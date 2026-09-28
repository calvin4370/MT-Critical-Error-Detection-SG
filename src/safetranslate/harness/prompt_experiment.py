"""Compares judge prompt designs on items with known answers.

Usage: uv run python -m safetranslate.harness.prompt_experiment configs/harness.yaml SYSTEM [N]

Base SEA-LION flagged every translation when asked to "list every critical error" in a
forced JSON list. The variants test whether deciding first, and calibration wording,
fix that. N items per group (real: 2 datasets; synthetic: altered + originals).
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from safetranslate.alter.generate import ask_json, make_client
from safetranslate.config import load_harness_config
from safetranslate.data import real_errors
from safetranslate.data.schema import ErrorItem, EvaluatorExample
from safetranslate.harness.metrics import binary_metrics
from safetranslate.harness.models import MAX_ERRORS, Judgement, judge_prompt

CALIBRATION = """
Most professional translations contain no critical errors. Report an error only if the
meaning a reader takes away is different from the English. Different wording, synonyms,
word order, transliterated names and small grammatical differences are NOT errors.
"""


class DecideFirst(BaseModel):
    """The model commits to yes/no before it may list anything."""

    has_critical_error: bool
    errors: list[ErrorItem] = Field(max_length=MAX_ERRORS)


def flags(client, system, variant: str, source: str, translation: str, lang: str) -> bool:
    """Whether a prompt variant says the translation has a critical error."""
    prompt = judge_prompt(source, translation, lang)
    if variant == "A":
        reply = ask_json(client, system.model, prompt, Judgement, extra_body=system.extra_body)
        return bool(reply.errors)
    if variant == "C":
        prompt += CALIBRATION
    prompt += '\nFirst decide "has_critical_error" (true or false); list errors only if true.'
    reply = ask_json(client, system.model, prompt, DecideFirst, extra_body=system.extra_body)
    return reply.has_critical_error


def main() -> None:
    config = load_harness_config(sys.argv[1])
    system = config.systems[sys.argv[2]]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    rng = random.Random(config.seed)
    raw = config.processed_dir.parent / "raw"

    # Items as (group, source, translation, lang, truly_has_error)
    items = []
    for name, examples in [("wmt21_ced", real_errors.load_wmt21(raw)), ("indicmt", real_errors.load_indicmt(raw))]:
        items += [(name, e.source, e.translation, e.target_lang, e.critical) for e in rng.sample(examples, n)]
    test = [EvaluatorExample.model_validate_json(l) for l in (config.altered_dir / "test.jsonl").open(encoding="utf-8")]
    altered = rng.sample([e for e in test if e.errors], n)
    by_id = {e.id: e for e in test}
    originals = [by_id[f"{e.id}-original"] for e in altered if f"{e.id}-original" in by_id]
    items += [("synthetic", e.source, e.translation, e.target_lang, bool(e.errors)) for e in altered + originals]

    client = make_client(system)
    results = {}
    for variant in ["A", "B", "C"]:
        with ThreadPoolExecutor(system.max_workers) as pool:
            said = list(pool.map(lambda it: flags(client, system, variant, *it[1:4]), items))
        for group in ["wmt21_ced", "indicmt", "synthetic"]:
            idx = [i for i, it in enumerate(items) if it[0] == group]
            m = binary_metrics([items[i][4] for i in idx], [said[i] for i in idx])
            results[f"{variant}/{group}"] = {k: (v["rate"] if isinstance(v, dict) else v) for k, v in m.items()}
            print(f"{variant} {group:10} recall {m['recall']['rate']:.0%}  false-positive rate {m['false_positive_rate']['rate']:.0%}")
    out = config.out_dir / sys.argv[2] / "prompt_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
