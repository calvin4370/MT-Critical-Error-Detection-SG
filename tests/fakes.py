"""Test doubles shared by the test files."""

import json
from types import SimpleNamespace


class FakeClient:
    """Stands in for the OpenAI client, replying with queued JSON strings in order."""

    def __init__(self, replies: list[dict]) -> None:
        self.replies = [json.dumps(r) for r in replies]
        self.calls: list[dict] = []  # what each request sent, for tests to inspect
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
