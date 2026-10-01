"""Hand-written async orchestrator (ORCHESTRATOR=runner).

Calls the nodes in order, applies routing, and merges partial state updates
using the reducers declared in `state.py`. `agents/graph.py` is the LangGraph
equivalent built from the same nodes and routers.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date

from agents.nodes import (
    clarify_node,
    ground_check_node,
    plan_node,
    refuse_node,
    search_node,
    synthesize_node,
    verify_node,
)
from agents.routing import route_after_plan, route_after_verify
from agents.state import REDUCERS, AgentState, initial_state


def apply(state: AgentState, update: dict) -> AgentState:
    """Merge a node's partial update into state using the declared reducers."""
    merged = dict(state)
    for key, value in update.items():
        reducer = REDUCERS.get(key)
        merged[key] = reducer(merged.get(key), value) if reducer else value
    return merged  # type: ignore[return-value]


async def fan_out(state: AgentState) -> dict:
    """Run one search branch per sub-query concurrently and merge their updates."""
    started = time.perf_counter()
    sub_queries = state["plan"].sub_queries
    updates = await asyncio.gather(
        *(search_node({"sub_query": sq}) for sq in sub_queries)
    )
    merged: dict = {"evidence": [], "trace": [], "timings": {}}
    for update in updates:
        for key, acc in merged.items():
            merged[key] = REDUCERS[key](acc, update.get(key, type(acc)()))
    merged["timings"]["fan_out_ms"] = round((time.perf_counter() - started) * 1000, 1)
    merged["trace"].append({"step": "fan_out", "branches": len(sub_queries)})
    return merged


async def run(user_query: str, *, today: date | None = None,
              carried_filters: dict | None = None) -> AgentState:
    """Run one turn end to end.

    `today` is passed in rather than read inside nodes, so relative years
    ("last year") are deterministic in tests.
    """
    started = time.perf_counter()
    state = initial_state(user_query, today=today or date.today(),
                          carried_filters=carried_filters)

    state = apply(state, await plan_node(state))

    route = route_after_plan(state)
    if route == "clarify":
        state = apply(state, await clarify_node(state))
    elif route == "refuse":
        state = apply(state, await refuse_node(state))
    else:
        state = apply(state, await fan_out(state))
        state = apply(state, await verify_node(state))
        if route_after_verify(state) == "refuse":
            state = apply(state, await refuse_node(state))
        else:
            state = apply(state, await synthesize_node(state))
            state = apply(state, await ground_check_node(state))

    total_ms = round((time.perf_counter() - started) * 1000, 1)
    return apply(state, {"timings": {"total_ms": total_ms}})
