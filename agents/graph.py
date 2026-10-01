"""LangGraph orchestrator (ORCHESTRATOR=graph).

Uses the same nodes, routers and state as `runner.py`:

    runner.py                          graph.py
    ---------                          --------
    `if/elif` over the routers         `add_conditional_edges`, same router functions
    `fan_out`: `asyncio.gather`        `Send("search", {"sub_query": sq})`, one per branch
    `apply`: the REDUCERS dict         the `Annotated[...]` metadata on `AgentState`

The fan-out is an edge rather than a node here, so unlike the runner there is no
`fan_out_ms` timing or `fan_out` trace row.
"""

from __future__ import annotations

import time
from datetime import date
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.nodes import (
    clarify_node,
    ground_check_node,
    plan_node,
    refuse_node,
    search_node,
    synthesize_node,
    verify_node,
)
from agents.routing import (
    CLARIFY,
    FAN_OUT,
    REFUSE,
    SYNTHESIZE,
    route_after_plan,
    route_after_verify,
)
from agents.state import AgentState, initial_state

PLAN = "plan"
SEARCH = "search"
VERIFY = "verify"
GROUND_CHECK = "ground_check"


def fan_out(state: AgentState) -> list[Send]:
    """One `search` task per sub-query; the state reducers merge the results."""
    return [Send(SEARCH, {"sub_query": sq}) for sq in state["plan"].sub_queries]


def after_plan(state: AgentState) -> str | list[Send]:
    """`route_after_plan`, with its `fan_out` result turned into Sends."""
    route = route_after_plan(state)
    return fan_out(state) if route == FAN_OUT else route


def build_graph():
    """Build and compile the graph."""
    g = StateGraph(AgentState)

    g.add_node(PLAN, plan_node)
    g.add_node(SEARCH, search_node)
    g.add_node(VERIFY, verify_node)
    g.add_node(SYNTHESIZE, synthesize_node)
    g.add_node(GROUND_CHECK, ground_check_node)
    g.add_node(CLARIFY, clarify_node)
    g.add_node(REFUSE, refuse_node)

    g.add_edge(START, PLAN)
    # Listing every destination lets LangGraph validate and draw the fan-out edge.
    g.add_conditional_edges(PLAN, after_plan, [CLARIFY, REFUSE, SEARCH])
    g.add_edge(SEARCH, VERIFY)
    g.add_conditional_edges(VERIFY, route_after_verify,
                            {SYNTHESIZE: SYNTHESIZE, REFUSE: REFUSE})
    g.add_edge(SYNTHESIZE, GROUND_CHECK)
    g.add_edge(GROUND_CHECK, END)
    g.add_edge(CLARIFY, END)
    g.add_edge(REFUSE, END)

    return g.compile()


@lru_cache(maxsize=1)
def graph():
    return build_graph()


async def run(user_query: str, *, today: date | None = None,
              carried_filters: dict | None = None) -> AgentState:
    """Run one turn. Same signature and returned state as `runner.run`."""
    started = time.perf_counter()
    state = initial_state(user_query, today=today or date.today(),
                          carried_filters=carried_filters)
    final = await graph().ainvoke(state)
    total_ms = round((time.perf_counter() - started) * 1000, 1)
    return {**final, "timings": {**final.get("timings", {}), "total_ms": total_ms}}
