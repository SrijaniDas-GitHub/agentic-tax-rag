"""Routing decisions, as pure functions that return node names.

Both orchestrators use these: the runner in an if/elif chain, the LangGraph
graph via `add_conditional_edges`. Keep them pure (no LLM calls, no I/O) so the
two stay in sync and the functions stay trivially testable.
"""

from __future__ import annotations

from agents.contracts import Plan
from agents.state import AgentState

# Node names shared by both orchestrators.
CLARIFY = "clarify"
REFUSE = "refuse"
FAN_OUT = "fan_out"
SYNTHESIZE = "synthesize"


def route_after_plan(state: AgentState) -> str:
    """plan -> clarify | refuse | fan_out.

    An answerable plan with no sub-queries is refused rather than sent on, since
    synthesizing from an empty evidence pack invites an answer from model memory.
    """
    plan: Plan | None = state.get("plan")
    if plan is None:
        return REFUSE

    if plan.intent == "needs_clarification":
        return CLARIFY
    if plan.intent == "out_of_scope":
        return REFUSE
    if not plan.sub_queries:
        return REFUSE
    return FAN_OUT


def route_after_verify(state: AgentState) -> str:
    """verify -> synthesize | refuse.

    Synthesize if at least one sub-query has sufficient evidence; the synthesizer
    is told which legs are missing. Refuse if none do.
    """
    verification = state.get("verification") or {}
    if not verification.get("packs"):
        return REFUSE
    if not verification.get("sufficient"):
        return REFUSE
    return SYNTHESIZE
