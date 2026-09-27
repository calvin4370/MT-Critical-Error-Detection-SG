.PHONY: install test data alter verify finalize

# Sets up .venv from uv.lock so every machine gets identical package versions
install:
	uv sync

test:
	uv run pytest

# Downloads raw data (FLORES+ needs HF_TOKEN from .env) and builds data/processed/
data:
	uv run --env-file .env python -m safetranslate.data.build configs/data.yaml

# Phase 2 stages; need the LLM server from configs/alter.yaml running
alter:
	uv run python -m safetranslate.alter.run configs/alter.yaml alter

verify:
	uv run python -m safetranslate.alter.run configs/alter.yaml verify

finalize:
	uv run python -m safetranslate.alter.run configs/alter.yaml finalize
