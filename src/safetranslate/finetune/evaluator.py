"""QLoRA fine-tuning of the evaluator (Phase 4).

Usage:
    uv run --group train python -m safetranslate.finetune.evaluator CONFIG [--limit N]

--limit N trains on only N examples: a quick smoke test before the full run.
Re-running resumes from the latest checkpoint in out_dir.
"""

import argparse
import os
from pathlib import Path

from safetranslate.config import FinetuneConfig, load_finetune_config
from safetranslate.data.schema import EvaluatorExample
from safetranslate.harness.models import Judgement, judge_prompt


def to_chat(example: EvaluatorExample) -> dict:
    """One training example: exactly the harness's judge prompt, and the JSON answer.

    Training on the same prompt the harness uses means the model is tested the way it
    was trained.
    """
    prompt = judge_prompt(example.source, example.translation, example.target_lang)
    answer = Judgement(errors=example.errors).model_dump_json()
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": answer}],
    }


def read_examples(path: Path, limit: int | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        examples = [to_chat(EvaluatorExample.model_validate_json(line)) for line in f]
    return examples[:limit] if limit else examples


def train(config: FinetuneConfig, limit: int | None) -> None:
    # Heavy libraries are imported here: they live in the optional "train" group
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{Path('mlflow.db').resolve()}")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "finetune-evaluator")


    # QLoRA: the frozen base is stored in 4-bit so an 8B model trains on a 12 GB GPU
    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, quantization_config=quantisation, dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Truncating would cut off the JSON answer at the end, so over-long examples are dropped
    def fits(chat: dict) -> bool:
        tokens = tokenizer.apply_chat_template(chat["prompt"] + chat["completion"], tokenize=True)
        return len(tokens) <= config.max_length

    train_chats = read_examples(config.train_file, limit)
    eval_chats = read_examples(config.validation_file, config.validation_limit)
    train_data = Dataset.from_list([c for c in train_chats if fits(c)])
    eval_data = Dataset.from_list([c for c in eval_chats if fits(c)])
    print(f"training on {len(train_data)} examples ({len(train_chats) - len(train_data)} too long)")

    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    out = config.out_dir if not limit else config.out_dir.with_name(config.out_dir.name + "-smoke")
    args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=config.max_length,
        # Loss only on the JSON answer, not on the prompt the model is given
        completion_only_loss=True,
        # Recompute activations instead of storing them: slower, but fits in 12 GB
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=config.eval_every,
        save_strategy="steps",
        save_steps=config.eval_every,
        save_total_limit=2,
        report_to=["mlflow"],
        run_name=f"evaluator-{out.name}",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora,
    )
    has_checkpoint = any(out.glob("checkpoint-*")) if out.exists() else False
    trainer.train(resume_from_checkpoint=has_checkpoint)
    trainer.save_model(str(out / "adapter"))
    print(f"adapter saved to {out / 'adapter'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    train(load_finetune_config(args.config), args.limit)


if __name__ == "__main__":
    main()
