"""Structured-output LLM client for Groq.

Each call runs a LangChain chain:
`ChatPromptTemplate | ChatGroq.with_structured_output(schema, method="json_mode", include_raw=True)`.

Around the chain:
  * a semaphore (4 concurrent calls) to stay within free-tier rate limits,
  * retries with backoff on transient errors,
  * a disk cache keyed by a hash of the full request, so repeated questions and
    eval re-runs make no API calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, DefaultAsyncHttpxClient, RateLimitError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

def _backoff(exc: Exception, attempt: int) -> float:
    """Use the provider's `retry-after` header if present; otherwise full jitter."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    raw = header.get("retry-after") or header.get("x-ratelimit-reset-tokens")
    if raw:
        try:
            return min(20.0, float(str(raw).rstrip("s")) + 0.25)
        except ValueError:
            pass
    return random.uniform(0, min(8.0, 2**attempt))   # full jitter, as a fallback


# Transient failures worth retrying. `APITimeoutError` subclasses
# `APIConnectionError`, so timeouts are retried too.
_RETRYABLE = (RateLimitError, APIConnectionError)

# Per-request timeout, and SDK-level retries. SDK retries are disabled because
# they would multiply with `_raw_call`'s own retries and the repair attempt,
# letting a single call block for a very long time.
REQUEST_TIMEOUT_S = 45.0
SDK_RETRIES = 0

# Groq free tier: 8000 tokens per minute per model, which is also the maximum size
# of a single request. Prompt sizes are budgeted against it in agents/budget.py.
TPM_LIMIT = 8000


class LLMError(RuntimeError):
    """Raised when a call fails after retries, or output never validates."""


class ProviderJSONError(LLMError):
    """Groq rejected the generation as invalid JSON (400 json_validate_failed).

    Handled by the repair retry rather than treated as a fatal 400.
    """

    def __init__(self, message: str, failed_generation: str = "") -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


@dataclass
class LLMResult:
    """A parsed response plus timing and token metrics."""

    value: BaseModel
    latency_ms: float  # wall clock incl. time queued behind the semaphore
    api_ms: float  # time spent in the API call(s)
    attempts: int  # 1 = parsed first try; 2 = needed the repair retry
    cached: bool
    model: str
    raw: str
    prompt_tokens: int = 0      # from the provider's usage data
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def queued_ms(self) -> float:
        """How long this call sat waiting for a concurrency slot."""
        return max(0.0, self.latency_ms - self.api_ms)

    @property
    def repaired(self) -> bool:
        return self.attempts > 1


def _strip_fence(text: str) -> str:
    """gpt-oss sometimes wraps JSON in a markdown fence despite json_object mode."""
    return _FENCE.sub("", text).strip()


# Prompts arrive already rendered, so this template only places them: system,
# user, and on a repair attempt the failed reply plus the validation error.
_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{instructions}"),
    ("human", "{user}"),
    MessagesPlaceholder("repair", optional=True),
])


def _unpack(out: dict[str, Any]) -> tuple[str, BaseModel | None, Exception | None,
                                          int, int]:
    """(raw, parsed, parsing_error, prompt_tokens, completion_tokens) from an
    `include_raw=True` result.

    `raw` is kept for `LLMResult.raw` and for the repair turn. Token counts come
    from `usage_metadata` (reasoning tokens included).
    """
    message: AIMessage = out["raw"]
    content = message.content if isinstance(message.content, str) else ""
    usage = message.usage_metadata or {}
    return (_strip_fence(content), out.get("parsed"), out.get("parsing_error"),
            int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0))


def _repair_error(schema: type[BaseModel], raw: str, error: Exception) -> Exception:
    """The error message shown to the model in the repair turn.

    Prefers Pydantic's validation error, since LangChain's `OutputParserException`
    repeats the whole completion, which is already in the previous turn.
    """
    if isinstance(error, OutputParserException):
        try:
            schema.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            return exc
    return error


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        fast_model: str | None = None,
        *,
        concurrency: int = 4,
        max_retries: int = 4,
        cache_dir: str | Path = "data/.llm_cache",
        api_key: str | None = None,
        max_completion_tokens: int = 2400,
        reasoning_effort: str = "low",
    ) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY is not set - copy .env.example to .env and fill it in.")

        self.model = model or os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
        self.fast_model = fast_model or os.getenv("FAST_MODEL_NAME", "openai/gpt-oss-20b")
        # `fast=True` routes to FAST_MODEL_NAME, which has its own rate-limit bucket.
        # Used for the per-branch sufficiency check. ALLOW_FAST_MODEL=0 disables it.
        self.allow_fast = os.getenv("ALLOW_FAST_MODEL", "1").strip().lower() not in {
            "0", "false", "no"
        }
        self.max_retries = max_retries
        # Completion tokens count against the same per-minute budget. gpt-oss
        # includes reasoning tokens in the completion, so the cap must leave room
        # for them (too low a cap truncates the JSON). reasoning_effort="low"
        # reduces completion size and latency.
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        # LLM_CACHE=0 disables cache reads but keeps writes, so the eval can run a
        # cold pass and then a cached pass.
        self.cache_reads = os.getenv("LLM_CACHE", "1").strip().lower() not in {
            "0", "false", "no"
        }
        self._api_key = key
        # One shared HTTP connection pool for every ChatGroq this client builds.
        self._http = DefaultAsyncHttpxClient()
        self._chains: dict[tuple, Runnable] = {}
        self._sem = asyncio.Semaphore(concurrency)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- cache ----------

    def _cache_key(self, model: str, system: str, user: str, schema: type[T], temp: float) -> str:
        """Hash the full request, including the JSON schema, so schema changes miss."""
        blob = json.dumps(
            [model, system, user, schema.model_json_schema(), temp],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _cache_read(self, key: str) -> str | None:
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    def _cache_write(self, key: str, raw: str) -> None:
        try:
            (self._cache_dir / f"{key}.json").write_text(raw, encoding="utf-8")
        except OSError:
            pass  # a cache miss is never fatal

    # ---------- chain ----------

    def _chat_model(self, model: str, max_tokens: int, reasoning_effort: str) -> ChatGroq:
        """Build the chat model for a chain. Tests override this method."""
        return ChatGroq(
            model=model,
            api_key=self._api_key,
            timeout=REQUEST_TIMEOUT_S,
            max_retries=SDK_RETRIES,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            http_async_client=self._http,
        )

    def _chain(self, model: str, schema: type[T], temperature: float,
               max_tokens: int, reasoning_effort: str) -> Runnable:
        """`_PROMPT | ChatGroq.with_structured_output(schema)`, cached per configuration.

        - `json_mode`: `response_format={"type": "json_object"}`, with the JSON
          schema written into the system message by `structured()`.
        - `include_raw=True`: the raw message is needed for the repair turn, token
          counts and `LLMResult.raw`; parse failures come back as `parsing_error`.
        - `temperature` is bound per call because `ChatGroq` turns a constructor
          temperature of 0 into 1e-8.
        """
        key = (model, schema, temperature, max_tokens, reasoning_effort)
        chain = self._chains.get(key)
        if chain is None:
            chat = self._chat_model(model, max_tokens, reasoning_effort)
            chain = self._chains[key] = _PROMPT | chat.with_structured_output(
                schema, method="json_mode", include_raw=True, temperature=temperature,
            )
        return chain

    # ---------- core call ----------

    async def _raw_call(
        self, model: str, chain: Runnable, inputs: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        """Invoke the chain once, retrying transient errors with backoff.

        Returns (chain output, api_ms). `api_ms` excludes time spent waiting for
        the semaphore. `ChatGroq` re-raises the Groq SDK's exceptions, so they are
        handled directly here.
        """
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    started = time.perf_counter()
                    out = await chain.ainvoke(inputs)
                    api_ms = (time.perf_counter() - started) * 1000
                return out, api_ms
            except _RETRYABLE as exc:
                last = exc
                await asyncio.sleep(_backoff(exc, attempt))
            except APIStatusError as exc:
                if exc.status_code >= 500:
                    last = exc
                    await asyncio.sleep(random.uniform(0, min(8.0, 2**attempt)))
                else:
                    body = getattr(exc, "body", None) or {}
                    err = body.get("error", {}) if isinstance(body, dict) else {}
                    if err.get("code") == "json_validate_failed":
                        raise ProviderJSONError(
                            f"{model} produced JSON the provider rejected",
                            str(err.get("failed_generation") or ""),
                        ) from exc
                    raise LLMError(
                        f"{model} rejected the request ({exc.status_code}): {exc}"
                    ) from exc
        raise LLMError(f"{model} failed after {self.max_retries} attempts: {last}")

    async def structured(
        self,
        schema: type[T],
        system: str,
        user: str,
        *,
        fast: bool = False,
        temperature: float = 0.0,
        use_cache: bool = True,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        """Call the model and validate its JSON against `schema`.

        On a validation failure, retries once with the model's output and the
        validation error. A second failure raises `LLMError`.
        """
        model = self.fast_model if (fast and self.allow_fast) else self.model
        key = self._cache_key(model, system, user, schema, temperature)

        if use_cache and self.cache_reads:
            hit = self._cache_read(key)
            if hit is not None:
                try:
                    return LLMResult(
                        value=schema.model_validate_json(hit),
                        latency_ms=0.0,
                        api_ms=0.0,
                        attempts=1,
                        cached=True,
                        model=model,
                        raw=hit,
                    )
                except ValidationError:
                    pass  # stale entry from an older schema; fall through and re-call

        # json_mode does not add the schema to the prompt, so it is appended here.
        instructions = (
            f"{system}\n\n"
            "Respond with a single JSON object and nothing else. "
            "It must validate against this JSON Schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        inputs = {"instructions": instructions, "user": user}
        chain = self._chain(model, schema, temperature,
                            max_tokens or self.max_completion_tokens,
                            reasoning_effort or self.reasoning_effort)

        started = time.perf_counter()
        try:
            out, api_ms = await self._raw_call(model, chain, inputs)
            raw, value, err, prompt_tok, completion_tok = _unpack(out)
        except ProviderJSONError as provider_err:
            raw, value, err = provider_err.failed_generation, None, provider_err
            api_ms, prompt_tok, completion_tok = 0.0, 0, 0
        if value is None:
            # One repair attempt with the failed output and the validation error.
            err = _repair_error(schema, raw, err)
            repair = [
                AIMessage(content=raw),
                HumanMessage(content=(
                    f"That response did not validate:\n{err}\n\n"
                    "Return corrected JSON only. No prose, no markdown fence."
                )),
            ]
            out2, repair_ms = await self._raw_call(model, chain, {**inputs, "repair": repair})
            raw2, value, err2, ptok2, ctok2 = _unpack(out2)
            elapsed = (time.perf_counter() - started) * 1000
            if value is None:
                err2 = _repair_error(schema, raw2, err2)
                raise LLMError(
                    f"{model} produced invalid JSON twice for {schema.__name__}: {err2}"
                ) from err2
            if use_cache:
                self._cache_write(key, value.model_dump_json())
            return LLMResult(value, elapsed, api_ms + repair_ms, 2, False, model, raw2,
                             prompt_tok + ptok2, completion_tok + ctok2)

        elapsed = (time.perf_counter() - started) * 1000
        if use_cache:
            self._cache_write(key, value.model_dump_json())
        return LLMResult(value, elapsed, api_ms, 1, False, model, raw,
                         prompt_tok, completion_tok)

    async def aclose(self) -> None:
        await self._http.aclose()


# ------------------------------------------------------------ process client --
#
# Process-wide client used by the nodes; tests swap it with `set_client`. Created
# lazily because it requires GROQ_API_KEY.

_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def set_client(client) -> None:
    """Install a client (or a fake, in tests)."""
    global _client
    _client = client


def reset_client() -> None:
    global _client
    _client = None
