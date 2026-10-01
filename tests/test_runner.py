"""Tests for the runner's state merging (`apply`) and concurrent fan-out."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from agents.contracts import EvidencePack, Passage, SubQuery
from agents.runner import apply, fan_out
from agents.state import initial_state, merge_timings

TODAY = date(2026, 9, 20)


def sub_query(country="US", key=2024, label="2024"):
    return SubQuery(question=f"q {country} {key}", country=country,
                    tax_year_key=key, tax_year_label=label)


def pack(sq):
    return EvidencePack(
        sub_query=sq,
        passages=[Passage(chunk_id=f"{sq.country}:{sq.tax_year_key}", text="x",
                          country=sq.country, tax_year_label=sq.tax_year_label,
                          doc_id="d", doc_title="D", page=1, printed_page=None, score=1.0)],
        sufficient=True, reason="ok", attempts=1,
    )


# ------------------------------------------------------------------- reducers --

def test_appending_fields_accumulate_and_scalar_fields_replace():
    state = initial_state("q", today=TODAY)
    state = apply(state, {"trace": [{"step": "a"}], "final_answer": "first"})
    state = apply(state, {"trace": [{"step": "b"}], "final_answer": "second"})

    assert [row["step"] for row in state["trace"]] == ["a", "b"]
    assert state["final_answer"] == "second"


def test_timings_merge_rather_than_replace():
    assert merge_timings({"a": 1.0}, {"b": 2.0}) == {"a": 1.0, "b": 2.0}

    state = initial_state("q", today=TODAY)
    state = apply(state, {"timings": {"plan_ms": 1.0}})
    state = apply(state, {"timings": {"verify_ms": 2.0}})
    assert state["timings"] == {"plan_ms": 1.0, "verify_ms": 2.0}


def test_apply_does_not_mutate_the_state_it_was_given():
    state = initial_state("q", today=TODAY)
    before = state["trace"]
    apply(state, {"trace": [{"step": "a"}]})
    assert state["trace"] is before and before == []


# -------------------------------------------------------------------- fan_out --

async def test_fan_out_runs_one_branch_per_sub_query_and_merges_their_lists(monkeypatch):
    seen: list[SubQuery] = []

    async def fake_search_node(state):
        # The branch receives only its SearchState payload.
        assert set(state) == {"sub_query"}
        sq = state["sub_query"]
        seen.append(sq)
        await asyncio.sleep(0)
        return {"evidence": [pack(sq)],
                "trace": [{"step": "search", "sq": sq.country}],
                "timings": {f"search[{sq.country}/{sq.tax_year_key}]_ms": 1.0}}

    monkeypatch.setattr("agents.runner.search_node", fake_search_node)

    sub_queries = [sub_query("US", 2023, "2023"), sub_query("US", 2024, "2024"),
                   sub_query("UK", 2024, "2024-25")]

    class FakePlan:
        pass

    plan = FakePlan()
    plan.sub_queries = sub_queries
    update = await fan_out({"plan": plan})

    assert len(seen) == 3
    assert len(update["evidence"]) == 3
    assert len([r for r in update["trace"] if r["step"] == "search"]) == 3
    assert update["trace"][-1] == {"step": "fan_out", "branches": 3}
    assert "fan_out_ms" in update["timings"]
    # one timing key per branch, plus the fan-out's own
    assert len(update["timings"]) == 4


async def test_fan_out_with_no_sub_queries_returns_empty_accumulators():
    class FakePlan:
        sub_queries = ()

    update = await fan_out({"plan": FakePlan()})
    assert update["evidence"] == []
    assert update["trace"] == [{"step": "fan_out", "branches": 0}]


@pytest.mark.parametrize("branches", [2, 3])
async def test_branches_are_concurrent_not_sequential(monkeypatch, branches):
    """N branches of 50ms each should take about 50ms in total, not N x 50ms."""
    async def slow_search_node(state):
        await asyncio.sleep(0.05)
        return {"evidence": [], "trace": [], "timings": {}}

    monkeypatch.setattr("agents.runner.search_node", slow_search_node)

    class FakePlan:
        pass

    plan = FakePlan()
    plan.sub_queries = [sub_query("US", 2020 + i, str(2020 + i)) for i in range(branches)]

    update = await fan_out({"plan": plan})
    assert update["timings"]["fan_out_ms"] < 50 * branches
