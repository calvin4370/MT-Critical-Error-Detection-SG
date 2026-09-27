import json
from pathlib import Path

from safetranslate.config import load_finetune_config
from safetranslate.data.schema import ErrorItem, EvaluatorExample
from safetranslate.finetune.evaluator import read_examples, to_chat
from safetranslate.harness.models import judge_prompt

EXAMPLE = EvaluatorExample(
    id="e1", origin_id="r1", split="train", target_lang="ms",
    source="Take 500 mg twice a day.", translation="Ambil 500 g dua kali sehari.",
    errors=[ErrorItem(category="wrong_quantity", span="500 g", description="mg -> g")],
)


def test_to_chat_uses_the_harness_prompt_and_json_answer() -> None:
    chat = to_chat(EXAMPLE)
    assert chat["prompt"][0]["content"] == judge_prompt(EXAMPLE.source, EXAMPLE.translation, "ms")
    assert json.loads(chat["completion"][0]["content"])["errors"][0]["span"] == "500 g"


def test_read_examples_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text((EXAMPLE.model_dump_json() + "\n") * 3)
    assert len(read_examples(path)) == 3
    assert len(read_examples(path, limit=2)) == 2


def test_repo_finetune_config_loads() -> None:
    config = load_finetune_config(Path(__file__).parents[1] / "configs" / "finetune_evaluator.yaml")
    assert config.lora_alpha == 2 * config.lora_rank
