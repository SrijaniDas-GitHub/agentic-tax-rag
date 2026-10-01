"""Offline tests for the LangChain chain inside `LLMClient.structured()`.

Two levels of faking:

* `ScriptedGroq` - the real `ChatGroq` (json_mode binding and output parser) with
  completions from `GenericFakeChatModel`. Covers parsing, repair, usage and cache.
* An httpx `MockTransport` under the real Groq SDK. Covers the request body and
  SDK errors (429 with `retry-after`, 5xx, 400 `json_validate_failed`).

No network or API key needed.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from pydantic import Field

from agents.contracts import Sufficiency
from core import llm

SYSTEM = "Judge the passages. Amounts look like $14,600 and JSON like {\"a\": 1}."
USER = "Sub-query: standard deduction {single}\nPassages: [p.96] $14,600"

GOOD = {"sufficient": True, "answer_chunk_id": "us_p17_2024:96:0",
        "reason": "p.96 states $14,600 for a single filer."}
BAD = {"sufficient": "maybe"}   # valid JSON, invalid schema: needs the repair turn


def reply(payload: dict, prompt_tokens: int, completion_tokens: int) -> AIMessage:
    return AIMessage(content=json.dumps(payload), usage_metadata={
        "input_tokens": prompt_tokens, "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })


class ScriptedGroq(ChatGroq):
    """ChatGroq with the network call replaced by a GenericFakeChatModel."""

    replies: Any = None
    requests: list = Field(default_factory=list)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append((messages, kwargs))
        return self.replies._generate(messages, stop=stop)


def scripted_client(tmp_path, *replies: AIMessage) -> tuple[llm.LLMClient, ScriptedGroq]:
    client = llm.LLMClient(api_key="offline", cache_dir=tmp_path)
    fake = ScriptedGroq(model="scripted", api_key="offline",
                        replies=GenericFakeChatModel(messages=iter(replies)))
    client._chat_model = lambda model, max_tokens, reasoning_effort: fake
    return client, fake


# ---------------------------------------------------------------- the chain --

async def test_parses_on_the_first_try(tmp_path):
    client, fake = scripted_client(tmp_path, reply(GOOD, 900, 60))

    result = await client.structured(Sufficiency, SYSTEM, USER)

    assert result.value == Sufficiency(**GOOD)
    assert (result.attempts, result.cached, result.repaired) == (1, False, False)
    assert json.loads(result.raw) == GOOD
    [(messages, kwargs)] = fake.requests
    # json_mode + a temperature of exactly 0 (ChatGroq would have made it 1e-8)
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0.0
    assert [m.type for m in messages] == ["system", "human"]
    assert messages[0].content.startswith(SYSTEM + "\n\nRespond with a single JSON object")
    assert json.dumps(Sufficiency.model_json_schema(), indent=2) in messages[0].content
    assert messages[1].content == USER


async def test_usage_tokens_reach_the_result(tmp_path):
    """Token counts come from the provider's usage metadata."""
    client, _ = scripted_client(tmp_path, reply(GOOD, 1234, 56))

    result = await client.structured(Sufficiency, SYSTEM, USER)

    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (
        1234, 56, 1290)


async def test_one_repair_then_success(tmp_path):
    client, fake = scripted_client(tmp_path, reply(BAD, 900, 40), reply(GOOD, 1000, 50))

    result = await client.structured(Sufficiency, SYSTEM, USER)

    assert result.value == Sufficiency(**GOOD)
    assert result.attempts == 2 and result.repaired
    assert (result.prompt_tokens, result.completion_tokens) == (1900, 90)   # both calls
    first, second = (m for m, _ in fake.requests)
    assert [m.type for m in second] == ["system", "human", "ai", "human"]
    assert second[:2] == first
    assert json.loads(second[2].content) == BAD
    repair = second[3].content
    assert repair.startswith("That response did not validate:\n")
    assert repair.endswith("Return corrected JSON only. No prose, no markdown fence.")
    # Pydantic's error, not LangChain's wrapper, which repeats the whole completion
    assert "2 validation errors for Sufficiency" in repair   # bad bool, missing reason
    assert "Failed to parse" not in repair


async def test_two_failures_raise_llm_error_and_cache_nothing(tmp_path):
    client, fake = scripted_client(tmp_path, reply(BAD, 900, 40), reply(BAD, 950, 40))

    with pytest.raises(llm.LLMError, match="invalid JSON twice for Sufficiency"):
        await client.structured(Sufficiency, SYSTEM, USER)

    assert len(fake.requests) == 2          # one repair, not a loop
    assert not list(tmp_path.glob("*.json"))


async def test_a_cache_hit_never_calls_the_model(tmp_path):
    client, fake = scripted_client(tmp_path, reply(GOOD, 900, 60))
    await client.structured(Sufficiency, SYSTEM, USER)          # fills the cache

    def no_model(*_: Any) -> None:
        raise AssertionError("a cache hit must not build a chain")

    replay = llm.LLMClient(api_key="offline", cache_dir=tmp_path)
    replay._chat_model = no_model
    result = await replay.structured(Sufficiency, SYSTEM, USER)

    assert result.cached and result.value == Sufficiency(**GOOD)
    assert (result.prompt_tokens, result.completion_tokens) == (0, 0)
    assert len(fake.requests) == 1


# --------------------------------------------------- the wire, via the SDK --

def completion(content: str) -> dict:
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 800, "completion_tokens": 70, "total_tokens": 870}}


def wired_client(tmp_path, *responses: httpx.Response):
    """A real LLMClient -> ChatGroq -> Groq SDK, answered by a MockTransport."""
    bodies: list[dict] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return queue.pop(0)

    client = llm.LLMClient(api_key="offline", cache_dir=tmp_path)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, bodies


@pytest.fixture
def no_sleep(monkeypatch):
    """Record every backoff the client computes, and sleep none of them."""
    waits: list[float] = []
    real_backoff = llm._backoff
    monkeypatch.setattr(llm, "_backoff",
                        lambda exc, attempt: waits.append(real_backoff(exc, attempt)) or 0.0)
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: 0.0)
    return waits


async def test_the_request_body_is_the_sdk_request_plus_chatgroq_defaults(tmp_path):
    """model, messages, max_tokens, reasoning_effort, response_format and
    temperature, plus five fields ChatGroq adds with the API's default values."""
    client, bodies = wired_client(tmp_path, httpx.Response(200, json=completion(
        json.dumps(GOOD))))

    result = await client.structured(Sufficiency, SYSTEM, USER, fast=True)

    assert (result.prompt_tokens, result.completion_tokens) == (800, 70)
    [body] = bodies
    messages = body.pop("messages")
    assert body == {
        "model": client.fast_model,
        "max_tokens": 2400,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        # added by ChatGroq._default_params; all API defaults
        "n": 1, "stream": False, "stop": None, "reasoning_format": None,
        "service_tier": "on_demand",
    }
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1] == {"role": "user", "content": USER}


async def test_provider_json_rejection_goes_to_the_repair_turn(tmp_path, no_sleep):
    rejected = httpx.Response(400, json={"error": {
        "message": "Failed to validate JSON", "type": "invalid_request_error",
        "code": "json_validate_failed", "failed_generation": '{"sufficient": tr',
    }})
    client, bodies = wired_client(
        tmp_path, rejected, httpx.Response(200, json=completion(json.dumps(GOOD))))

    result = await client.structured(Sufficiency, SYSTEM, USER)

    assert result.attempts == 2 and result.value == Sufficiency(**GOOD)
    assert (result.prompt_tokens, result.completion_tokens) == (800, 70)   # 1st billed 0
    repair = bodies[1]["messages"]
    assert repair[2] == {"role": "assistant", "content": '{"sufficient": tr'}
    assert "produced JSON the provider rejected" in repair[3]["content"]


async def test_a_rate_limit_waits_the_providers_retry_after(tmp_path, no_sleep):
    limited = httpx.Response(429, headers={"retry-after": "3"},
                             json={"error": {"message": "Rate limit", "code": "rate_limit"}})
    client, bodies = wired_client(
        tmp_path, limited, httpx.Response(200, json=completion(json.dumps(GOOD))))

    result = await client.structured(Sufficiency, SYSTEM, USER)

    assert result.attempts == 1 and len(bodies) == 2
    assert no_sleep == [3.25]           # the header, not the jitter


async def test_a_5xx_is_retried_and_a_4xx_is_not(tmp_path, no_sleep):
    client, bodies = wired_client(
        tmp_path, httpx.Response(503, json={"error": {"message": "busy"}}),
        httpx.Response(200, json=completion(json.dumps(GOOD))))
    assert (await client.structured(Sufficiency, SYSTEM, USER)).value == Sufficiency(**GOOD)
    assert len(bodies) == 2

    client, bodies = wired_client(
        tmp_path, httpx.Response(413, json={"error": {"message": "Requested 8103"}}))
    with pytest.raises(llm.LLMError, match=r"rejected the request \(413\)"):
        await client.structured(Sufficiency, SYSTEM, USER, use_cache=False)
    assert len(bodies) == 1


def test_chatgroq_is_built_with_one_retry_policy_and_a_bounded_request(tmp_path):
    """SDK retries disabled and the request timeout set on the ChatGroq client."""
    client = llm.LLMClient(api_key="offline", cache_dir=tmp_path)
    chat = client._chat_model("openai/gpt-oss-120b", 2400, "low")

    assert chat.max_retries == llm.SDK_RETRIES == 0
    assert chat.request_timeout == llm.REQUEST_TIMEOUT_S
    assert chat.async_client._client.max_retries == 0     # what the SDK will obey
    assert chat.async_client._client.timeout == llm.REQUEST_TIMEOUT_S
    assert (chat.max_tokens, chat.reasoning_effort) == (2400, "low")
