"""Search agent: filtered retrieval plus a sufficiency self-check, up to 2 attempts.

Attempt 1 uses the planner's sub-query as is. If the self-check finds the answer
missing, it also suggests a new query, written using the vocabulary of the
passages that were retrieved, and attempt 2 runs that.

Every search is filtered by country and tax year. Without the filter, packs
contain a large share of passages from the wrong country or year.
"""

from __future__ import annotations

import time

from agents import budget
from agents.contracts import EvidencePack, Passage, SubQuery, Sufficiency
from agents.prompts import load, render
from core import retrieval
from core.llm import get_client

MAX_ATTEMPTS = 2       # a third attempt mostly adds latency
PACK_SIZE = retrieval.TOP_K


def format_passages(passages: list[Passage], *, limit: int, chars: int,
                    figures_only: bool = False) -> str:
    """Format passages for a prompt, each headed by its chunk_id for citation.

    `limit` and `chars` are required so every call site sizes itself against
    `agents/budget.py`. Truncated passages are marked.
    """
    blocks = []
    for p in passages[:limit]:
        text = p.text if len(p.text) <= chars else p.text[:chars] + "\n[...truncated]"
        blocks.append(
            f"[{p.chunk_id}] {p.doc_title} | {p.country} {p.tax_year_label} | page {p.page}\n"
            f"{text}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "(nothing retrieved)"


def retrieve(sub_query: SubQuery, query: str, k: int = PACK_SIZE) -> list[Passage]:
    """Hybrid search filtered to the sub-query's country and tax year."""
    return retrieval.search(
        query,
        country=sub_query.country,
        tax_year_key=sub_query.tax_year_key,
        k=k,
        mode="hybrid",
    )


async def check_sufficiency(sub_query: SubQuery,
                            passages: list[Passage]) -> tuple[Sufficiency, int, str]:
    """Ask the LLM whether a passage contains the answer, and which one.

    Returns the verdict, tokens used, and the model used. Rate limits are per
    model, so the eval tracks token usage per model.
    """
    if not passages:
        return Sufficiency(sufficient=False, reason="Retrieval returned nothing.",
                           missing=sub_query.question), 0, ""

    result = await get_client().structured(
        Sufficiency,
        load("sufficiency_system"),
        render(
            "sufficiency_user",
            question=sub_query.question,
            country=sub_query.country,
            tax_year_label=sub_query.tax_year_label,
            passages=format_passages(passages,
                                     limit=budget.SUFFICIENCY_PASSAGES,
                                     chars=budget.SUFFICIENCY_CHARS,
                                     figures_only=True),
        ),
        # Runs once per branch, so it uses FAST_MODEL_NAME, which has its own
        # rate-limit bucket. Disable with ALLOW_FAST_MODEL=0.
        fast=True,
        max_tokens=budget.SUFFICIENCY_MAX_COMPLETION,
    )
    verdict: Sufficiency = result.value

    # A "sufficient" verdict must name a chunk that is actually in the pack.
    known = {p.chunk_id for p in passages}
    if verdict.sufficient and verdict.answer_chunk_id not in known:
        return Sufficiency(
            sufficient=False,
            answer_chunk_id=None,
            reason=(f"Self-check claimed sufficiency but named "
                    f"{verdict.answer_chunk_id!r}, which is not in the pack."),
            missing=verdict.missing or sub_query.question,
        ), result.total_tokens, result.model
    return verdict, result.total_tokens, result.model


async def run_search_agent(sub_query: SubQuery, *,
                           max_attempts: int = MAX_ATTEMPTS,
                           k: int = PACK_SIZE) -> tuple[EvidencePack, list[dict]]:
    """Retrieve and self-check for one sub-query, retrying once with a new query.

    Returns the evidence pack and trace rows for each attempt and rewrite. If no
    attempt succeeds, the best pack is still returned with `sufficient=False` and
    a reason saying what is missing.
    """
    trace: list[dict] = []
    query = sub_query.question
    best: tuple[list[Passage], Sufficiency, int] | None = None
    tried: list[str] = []

    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        passages = retrieve(sub_query, query, k=k)
        retrieved_ms = (time.perf_counter() - started) * 1000
        tried.append(query)

        verdict, tokens, model = await check_sufficiency(sub_query, passages)
        trace.append({
            "step": "search.attempt",
            "sub_query": f"{sub_query.country}/{sub_query.tax_year_key}",
            "attempt": attempt,
            "query": query,
            "n_passages": len(passages),
            # All ids, so the eval can score recall per attempt.
            "chunk_ids": [p.chunk_id for p in passages],
            "scores": [round(p.score, 5) for p in passages],
            "sufficient": verdict.sufficient,
            "reason": verdict.reason,
            "missing": verdict.missing,
            "answer_chunk_id": verdict.answer_chunk_id,
            "retrieve_ms": round(retrieved_ms, 1),
            "tokens": tokens,
            "model": model,
        })

        # Keep the newest pack unless an earlier attempt already succeeded.
        if best is None or not best[1].sufficient:
            best = (passages, verdict, attempt)

        if verdict.sufficient or attempt == max_attempts:
            break

        # The self-check returns the replacement query; stop if it gave none.
        if not verdict.suggested_query or verdict.suggested_query.strip() == query:
            break
        trace.append({
            "step": "search.rewrite",
            "sub_query": f"{sub_query.country}/{sub_query.tax_year_key}",
            "from": query,
            "to": verdict.suggested_query,
            "why": verdict.missing or verdict.reason,
        })
        query = verdict.suggested_query.strip()

    passages, verdict, _ = best  # type: ignore[misc]
    reason = verdict.reason
    if not verdict.sufficient and verdict.missing:
        reason = f"{reason} Missing: {verdict.missing}"

    pack = EvidencePack(
        sub_query=sub_query,
        passages=passages,
        sufficient=verdict.sufficient,
        reason=reason,
        attempts=len(tried),
        queries_tried=tried,
        answer_chunk_id=verdict.answer_chunk_id,
    )
    return pack, trace
