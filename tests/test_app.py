"""Tests for what carries from one turn to the next, and UI text escaping."""

from __future__ import annotations

from agents.carry import carry
from agents.contracts import Plan, SubQuery


def test_the_app_and_the_eval_share_one_carry_rule():
    import eval.run_eval as harness
    from core.paths import REPO_ROOT

    assert harness.carry is carry
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "from agents.carry import carry" in source
    assert "def carry(" not in source


def _sub(country: str, key: int, label: str) -> SubQuery:
    return SubQuery(question=f"{country} {label}", country=country, tax_year_key=key,
                    tax_year_label=label, year_text=label)


def test_carries_the_resolved_filters_and_the_previous_question_as_subject():
    plan = Plan(intent="compare_countries",
                sub_queries=[_sub("US", 2024, "2024"), _sub("UK", 2024, "2024-25")])
    carried = carry({"plan": plan, "user_query": "US deduction vs UK allowance?"})

    assert carried["previous_question"] == "US deduction vs UK allowance?"
    assert carried["countries"] == ["UK", "US"]
    assert carried["tax_years"] == ["2024", "2024-25"]
    assert carried["resolved"] == [
        {"country": "US", "tax_year_key": 2024, "tax_year_label": "2024"},
        {"country": "UK", "tax_year_key": 2024, "tax_year_label": "2024-25"},
    ]


def test_a_clarification_carries_nothing():
    assert carry({"plan": Plan(intent="needs_clarification",
                               clarifying_question="Which year?")}) is None


def test_a_refusal_carries_nothing_even_though_it_has_sub_queries():
    """out_of_scope plans still list sub-queries; they must not carry over."""
    plan = Plan(intent="out_of_scope", refusal_reason="personalised_advice",
                sub_queries=[_sub("US", 2024, "2024")])
    assert plan.sub_queries, "the premise of this test"
    assert carry({"plan": plan}) is None


def test_no_plan_carries_nothing():
    assert carry({}) is None
    assert carry({"plan": None}) is None


# ------------------------------------------------ what the planner is shown --

from agents.planner import NO_PREVIOUS_TURN, carried_context


def test_no_carry_renders_the_exact_string_the_eval_plans_under():
    """This string is part of every planner cache key."""
    assert carried_context(None) == NO_PREVIOUS_TURN == "(No previous turn.)"
    assert carried_context({}) == NO_PREVIOUS_TURN


def test_carried_context_names_the_subject_and_overrides_the_clarify_rule():
    """A carried country must count as stated, so the planner does not ask for it."""
    plan = Plan(intent="lookup", sub_queries=[_sub("US", 2024, "2024")])
    text = carried_context(carry({
        "plan": plan,
        "user_query": "What is the 2024 standard deduction for a single filer?",
    }))

    assert "What is the 2024 standard deduction for a single filer?" in text
    assert "US 2024" in text
    assert "context only" in text
    assert "do not ask for it" in text


def test_dollar_figures_are_escaped_so_markdown_does_not_typeset_them():
    from app.render import no_math

    q07 = "is $14,600 [1]. The UK is £12,570 [2]. The US ($14,600) is larger"
    assert no_math(q07) == r"is \$14,600 [1]. The UK is £12,570 [2]. The US (\$14,600) is larger"
    assert no_math("£12,570, no dollars") == "£12,570, no dollars"
