"""LangGraph orchestrator tests, run offline with a fake LLM.

  * the graph has the expected nodes and edges;
  * each terminal path (answer, clarify, refuse) returns the same state as
    `runner.run`, apart from timings and the runner's `fan_out` trace row;
  * branches run concurrently and merge in sub-query order;
  * without a reducer, LangGraph rejects concurrent writes to a channel.
"""

from __future__ import annotations

import asyncio
import operator
import time
from datetime import date
from typing import Annotated, TypedDict

import pytest
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents import graph as graph_mod
from agents import runner
from agents.contracts import (
    AnswerClaim,
    DraftAnswer,
    DraftPlan,
    EvidencePack,
    Passage,
    PlannedSubQuery,
)
from agents.run import select_runner
from core.llm import reset_client, set_client
from eval.parity import comparable, first_difference
from tests.fakes import FakeLLM

TODAY = date(2026, 9, 20)

FIGURES = {2023: "$13,850", 2024: "$14,600"}


@pytest.fixture(autouse=True)
def _no_real_llm():
    yield
    reset_client()


def fake_search(delays: dict[int, float] | None = None, sufficient: bool = True):
    """Fake search agent returning one passage per sub-query.

    `delays` lets a later sub-query finish first, to check merge order.
    """
    async def run_search_agent(sq):
        await asyncio.sleep((delays or {}).get(sq.tax_year_key, 0))
        passage = Passage(
            chunk_id=f"us_p17_{sq.tax_year_key}:98:0",
            text=f"| Single | {FIGURES.get(sq.tax_year_key, '$1')} |",
            country=sq.country, tax_year_label=sq.tax_year_label,
            doc_id=f"us_p17_{sq.tax_year_key}", doc_title=f"Pub 17 ({sq.tax_year_key})",
            page=98, printed_page=96, score=0.5,
        )
        pack = EvidencePack(sub_query=sq, passages=[passage], sufficient=sufficient,
                            reason="ok" if sufficient else "no table", attempts=1)
        return pack, [{"step": "search.attempt", "sub_query": sq.tax_year_key, "attempt": 1}]
    return run_search_agent


def compare_years_plan() -> DraftPlan:
    return DraftPlan(intent="compare_years", sub_queries=[
        PlannedSubQuery(question="US standard deduction single 2023", country="US",
                        year_text="2023"),
        PlannedSubQuery(question="US standard deduction single 2024", country="US",
                        year_text="2024"),
    ])


async def both(question: str) -> tuple[dict, dict]:
    return (await runner.run(question, today=TODAY),
            await graph_mod.run(question, today=TODAY))


def assert_same(a: dict, b: dict) -> None:
    assert first_difference(comparable(a), comparable(b)) is None


# ------------------------------------------------------------------- shape --

def test_the_graph_has_the_expected_nodes_and_edges():
    drawn = graph_mod.build_graph().get_graph()
    assert set(drawn.nodes) == {"__start__", "plan", "search", "verify", "synthesize",
                                "ground_check", "clarify", "refuse", "__end__"}
    edges = {(e.source, e.target, e.conditional) for e in drawn.edges}
    assert edges == {
        ("__start__", "plan", False),
        ("plan", "clarify", True), ("plan", "refuse", True), ("plan", "search", True),
        ("search", "verify", False),
        ("verify", "synthesize", True), ("verify", "refuse", True),
        ("synthesize", "ground_check", False),
        ("ground_check", "__end__", False),
        ("clarify", "__end__", False), ("refuse", "__end__", False),
    }


def test_fan_out_sends_one_search_state_per_sub_query():
    class FakePlan:
        sub_queries = ("a", "b", "c")

    sends = graph_mod.fan_out({"plan": FakePlan()})
    assert [(s.node, s.arg) for s in sends] == [
        ("search", {"sub_query": "a"}), ("search", {"sub_query": "b"}),
        ("search", {"sub_query": "c"}),
    ]


# ------------------------------------------------------------------ parity --

async def test_an_answered_comparison_is_identical_under_both_runners(monkeypatch):
    """The second leg finishes first; results still merge in dispatch order."""
    monkeypatch.setattr("agents.nodes.run_search_agent", fake_search({2023: 0.03}))
    set_client(FakeLLM(
        DraftPlan=compare_years_plan(),
        DraftAnswer=DraftAnswer(claims=[
            AnswerClaim(text="It was $13,850 for 2023.", chunk_ids=["us_p17_2023:98:0"]),
            AnswerClaim(text="It is $14,600 for 2024.", chunk_ids=["us_p17_2024:98:0"]),
        ]),
    ))

    ran, graphed = await both("How did the US standard deduction change from 2023 to 2024?")

    assert_same(ran, graphed)
    assert graphed["grounding_report"]["ok"] is True
    assert "$14,600" in graphed["final_answer"]
    assert [p.sub_query.tax_year_key for p in graphed["evidence"]] == [2023, 2024]
    # The fan_out row is the only trace difference.
    assert [r["step"] for r in ran["trace"]].count("fan_out") == 1
    assert [r["step"] for r in graphed["trace"]].count("fan_out") == 0
    assert "search[US/2023]_ms" in graphed["timings"] and "total_ms" in graphed["timings"]


async def test_a_clarification_is_identical_under_both_runners():
    set_client(FakeLLM(DraftPlan=DraftPlan(intent="needs_clarification",
                                           clarifying_question="Which tax year?")))
    ran, graphed = await both("what's the standard deduction")
    assert_same(ran, graphed)
    assert graphed["final_answer"].startswith("Which tax year?")
    assert "evidence" in graphed and graphed["evidence"] == []


async def test_a_refusal_after_verify_is_identical_under_both_runners(monkeypatch):
    """Every leg insufficient -> `route_after_verify` -> refuse, in both."""
    monkeypatch.setattr("agents.nodes.run_search_agent", fake_search(sufficient=False))
    set_client(FakeLLM(DraftPlan=compare_years_plan()))
    ran, graphed = await both("How did the US standard deduction change from 2023 to 2024?")
    assert_same(ran, graphed)
    assert graphed["trace"][-1]["reason"] == "insufficient_evidence"


async def test_the_graph_runs_branches_concurrently(monkeypatch):
    """Two 100 ms legs should take ~100 ms, not ~200 ms."""
    monkeypatch.setattr("agents.nodes.run_search_agent",
                        fake_search({2023: 0.1, 2024: 0.1}))
    set_client(FakeLLM(DraftPlan=compare_years_plan(),
                       DraftAnswer=DraftAnswer(claims=[])))
    started = time.perf_counter()
    await graph_mod.run("q", today=TODAY)
    assert time.perf_counter() - started < 0.19


# ---------------------------------------------------------------- reducers --

async def test_without_a_reducer_the_fan_out_is_refused():
    """Two branches writing a plain list channel in one step raise InvalidUpdateError;
    with an `operator.add` reducer they merge."""
    class Bare(TypedDict, total=False):
        items: list
        evidence: list

    class Reduced(TypedDict, total=False):
        items: list
        evidence: Annotated[list, operator.add]

    def build(schema):
        g = StateGraph(schema)
        g.add_node("branch", lambda s: {"evidence": [s["item"]]})
        g.add_conditional_edges(START, lambda s: [Send("branch", {"item": i})
                                                  for i in s["items"]], ["branch"])
        g.add_edge("branch", END)
        return g.compile()

    with pytest.raises(InvalidUpdateError):
        await build(Bare).ainvoke({"items": [1, 2], "evidence": []})
    assert (await build(Reduced).ainvoke({"items": [1, 2], "evidence": []}))["evidence"] == [1, 2]


# -------------------------------------------------------------- the switch --

def test_orchestrator_selects_the_runner(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR", "graph")
    assert select_runner() == (graph_mod.run, "graph")
    monkeypatch.setenv("ORCHESTRATOR", "runner")
    assert select_runner() == (runner.run, "runner")
    monkeypatch.delenv("ORCHESTRATOR")
    assert select_runner()[1] == "runner"


def test_a_typo_in_orchestrator_fails_rather_than_falling_back(monkeypatch):
    """An unknown value raises instead of silently using the runner."""
    monkeypatch.setenv("ORCHESTRATOR", "grpah")
    with pytest.raises(SystemExit):
        select_runner()
