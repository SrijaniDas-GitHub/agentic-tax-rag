"""Tests for the routing functions shared by both orchestrators."""

from __future__ import annotations

from agents.contracts import Plan, SubQuery
from agents.routing import route_after_plan, route_after_verify

SQ = SubQuery(question="q", country="US", tax_year_key=2024, tax_year_label="2024")


def plan_state(**kwargs) -> dict:
    return {"plan": Plan(**kwargs)}


# ----------------------------------------------------------- route_after_plan --

def test_answerable_intents_fan_out():
    for intent in ("lookup", "compare_years", "compare_countries"):
        assert route_after_plan(plan_state(intent=intent, sub_queries=[SQ])) == "fan_out"


def test_clarify_and_refuse_are_separate_destinations():
    assert route_after_plan(plan_state(intent="needs_clarification")) == "clarify"
    assert route_after_plan(plan_state(intent="out_of_scope")) == "refuse"


def test_an_answerable_intent_with_no_sub_queries_refuses():
    assert route_after_plan(plan_state(intent="lookup", sub_queries=[])) == "refuse"


def test_a_missing_plan_refuses_rather_than_raising():
    assert route_after_plan({}) == "refuse"


# --------------------------------------------------------- route_after_verify --

def test_one_sufficient_pack_is_enough_to_synthesize():
    assert route_after_verify({"verification": {"packs": 2, "sufficient": 1}}) == "synthesize"


def test_no_sufficient_pack_refuses():
    assert route_after_verify({"verification": {"packs": 2, "sufficient": 0}}) == "refuse"


def test_no_packs_at_all_refuses():
    assert route_after_verify({"verification": {"packs": 0, "sufficient": 0}}) == "refuse"
    assert route_after_verify({}) == "refuse"
