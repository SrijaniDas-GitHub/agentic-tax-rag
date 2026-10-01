"""Agent state and reducers.

Shared by both orchestrators: `runner.apply` uses the `REDUCERS` table, and
LangGraph reads the `Annotated` metadata directly.

    evidence, trace  - appended to by every search branch  -> operator.add
    timings          - written by branches and other nodes -> merge_timings
    everything else  - written by exactly one node          -> last write wins
"""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Any, TypedDict

from agents.contracts import EvidencePack, Plan, SubQuery


def merge_timings(left: dict, right: dict) -> dict:
    """Shallow-merge two timing dicts. Each search branch writes its own key."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class AgentState(TypedDict, total=False):
    """State for one turn. `total=False` because nodes return partial updates."""

    # --- inputs -------------------------------------------------------------
    user_query: str
    carried_filters: dict | None   # resolved filters from the previous turn
    today: date                    # injected so relative years are testable

    # --- accumulated across concurrent branches ------------------------------
    evidence: Annotated[list[EvidencePack], operator.add]
    trace: Annotated[list[dict], operator.add]
    timings: Annotated[dict, merge_timings]

    # --- written by exactly one node each -------------------------------------
    plan: Plan | None
    verification: dict | None
    # Separate from `evidence`, whose append reducer would duplicate the packs.
    verified_evidence: list[EvidencePack]
    draft_answer: dict | None
    grounding_report: dict | None
    final_answer: str | None
    citations: list[dict]


class SearchState(TypedDict):
    """Input for one search branch (also the LangGraph `Send` payload)."""

    sub_query: SubQuery


def initial_state(user_query: str, *, today: date,
                  carried_filters: dict | None = None) -> AgentState:
    """Initial state; accumulators start empty rather than None."""
    return AgentState(
        user_query=user_query,
        carried_filters=carried_filters,
        today=today,
        evidence=[],
        trace=[],
        timings={},
        plan=None,
        verification=None,
        draft_answer=None,
        grounding_report=None,
        final_answer=None,
        citations=[],
    )


# Reducer table used by `runner.apply`.
REDUCERS: dict[str, Any] = {
    "evidence": operator.add,
    "trace": operator.add,
    "timings": merge_timings,
}
