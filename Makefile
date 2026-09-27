.PHONY: install test data

# Sets up .venv from uv.lock so every machine gets identical package versions
install:
	uv sync

test:
	uv run pytest

# Downloads raw data (FLORES+ needs HUGGINGFACE_TOKEN from .env) and builds data/processed/
data:
	uv run --env-file .env python -m safetranslate.data.build configs/data.yaml
