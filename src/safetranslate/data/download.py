"""Shared file download used by every dataset loader."""

import shutil
import urllib.request
from pathlib import Path


def fetch(url: str, path: Path, headers: dict[str, str] | None = None) -> None:
    """Downloads url to path, skipping it if path already exists."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name and rename only when complete, so an interrupted
    # download never leaves a partial file that looks finished
    partial = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request) as response, partial.open("wb") as f:
        shutil.copyfileobj(response, f)
    partial.rename(path)
