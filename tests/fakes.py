"""A fake LLM client for offline node tests.

Nodes reach the model through `core.llm.get_client()`; tests swap in `FakeLLM`.
Responses are queued per schema rather than by call order, so a node that makes
calls in the wrong order fails.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from core.llm import LLMResult


class FakeLLM:
    """Returns pre-built model instances, keyed by the schema requested."""

    def __init__(self, **by_schema: Any) -> None:
        # value may be a single model, or a list consumed one call at a time
        self._queues: dict[str, list[BaseModel]] = {
            name: list(value) if isinstance(value, list) else [value]
            for name, value in by_schema.items()
        }
        self.calls: list[tuple[str, str, str]] = []   # (schema, system, user)

    async def structured(self, schema: type[BaseModel], system: str, user: str,
                         **kwargs: Any) -> LLMResult:
        name = schema.__name__
        self.calls.append((name, system, user))
        queue = self._queues.get(name)
        if not queue:
            raise AssertionError(
                f"FakeLLM got an unexpected {name} call. Queued: {sorted(self._queues)}"
            )
        value = queue.pop(0) if len(queue) > 1 else queue[0]
        return LLMResult(value=value, latency_ms=0.0, api_ms=0.0, attempts=1,
                         cached=False, model="fake", raw=value.model_dump_json())

    async def aclose(self) -> None:
        return None

    def calls_for(self, schema_name: str) -> list[tuple[str, str, str]]:
        return [c for c in self.calls if c[0] == schema_name]
