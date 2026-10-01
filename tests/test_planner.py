"""Tests for `resolve_plan`, which turns the LLM's draft plan into resolved tax years.

`today` and `known` (corpus coverage) are passed in, so no LLM, index or clock
is involved.
"""

from __future__ import annotations

from datetime import date

from agents.contracts import DraftPlan, PlannedSubQuery
from agents.planner import coverage, resolve_plan

TODAY = date(2026, 9, 20)
KNOWN = {"US": {2023: "2023", 2024: "2024"}, "UK": {2023: "2023-24", 2024: "2024-25"}}


def draft(*legs, intent="lookup", **kwargs):
    return DraftPlan(
        intent=intent,
        sub_queries=[PlannedSubQuery(question=q, country=c, year_text=y) for q, c, y in legs],
        **kwargs,
    )


def resolved(plan):
    return [(s.country, s.tax_year_key, s.tax_year_label) for s in plan.sub_queries]


# --------------------------------------------------------- the year, resolved --

def test_each_country_gets_its_own_label_for_the_same_key():
    """US 2024 is labelled "2024"; UK 2024 is "2024-25"."""
    plan = resolve_plan(draft(("q", "US", "2024"), ("q", "UK", "2024")),
                        today=TODAY, known=KNOWN)
    assert resolved(plan) == [("US", 2024, "2024"), ("UK", 2024, "2024-25")]


def test_a_uk_span_label_resolves_to_the_year_it_starts_in():
    plan = resolve_plan(draft(("q", "UK", "2024-25")), today=TODAY, known=KNOWN)
    assert resolved(plan) == [("UK", 2024, "2024-25")]


def test_today_is_injected_so_relative_years_do_not_rot():
    """Relative years resolve against the injected `today`."""
    plan = resolve_plan(draft(("q", "US", "last year")),
                        today=date(2025, 6, 1), known=KNOWN)
    assert resolved(plan) == [("US", 2024, "2024")]


# ------------------------------------------------ two exceptions, two routes --

def test_ambiguous_year_clarifies():
    """"FY24" is ambiguous, so the plan asks for clarification."""
    plan = resolve_plan(draft(("q", "US", "FY24")), today=TODAY, known=KNOWN)
    assert plan.intent == "needs_clarification"
    assert plan.sub_queries == []


def test_unresolvable_year_refuses():
    """No year in the text, so the plan refuses."""
    plan = resolve_plan(draft(("q", "US", "sometime")), today=TODAY, known=KNOWN)
    assert plan.intent == "out_of_scope"
    assert plan.refusal_reason == "unresolvable_year"


def test_the_two_routes_are_not_collapsed_into_one():
    ambiguous = resolve_plan(draft(("q", "US", "FY24")), today=TODAY, known=KNOWN)
    unreadable = resolve_plan(draft(("q", "US", "sometime")), today=TODAY, known=KNOWN)
    assert ambiguous.intent != unreadable.intent


# ------------------------------------------------------------------ coverage --

def test_a_resolvable_year_outside_the_corpus_is_refused_by_reason():
    """2026-27 is a valid UK tax year that the corpus does not hold."""
    plan = resolve_plan(draft(("q", "UK", "2026-27")), today=TODAY, known=KNOWN)
    assert plan.intent == "out_of_scope"
    assert plan.refusal_reason == "year_not_in_corpus"
    assert "2024-25" in plan.refusal_detail          # says what it DOES hold


def test_coverage_shrinks_with_the_corpus_not_with_the_code():
    one_year = {"US": {2024: "2024"}}
    assert resolve_plan(draft(("q", "US", "2024")), today=TODAY,
                        known=one_year).intent == "lookup"
    assert resolve_plan(draft(("q", "US", "2023")), today=TODAY,
                        known=one_year).refusal_reason == "year_not_in_corpus"


def test_coverage_is_read_from_the_manifest():
    assert coverage() == {"UK": {2023: "2023-24", 2024: "2024-25"},
                          "US": {2023: "2023", 2024: "2024"}}


# ------------------------------------------------------------- plan plumbing --

def test_duplicate_legs_are_dropped():
    """Two legs on the same (country, year) are merged into one."""
    plan = resolve_plan(draft(("a", "US", "2024"), ("b", "US", "2024")),
                        today=TODAY, known=KNOWN)
    assert len(plan.sub_queries) == 1
    assert plan.sub_queries[0].question == "a"


def test_an_assumption_note_reaches_the_plan_qualified_by_its_leg():
    """Assumption notes are attached to the sub-query and listed on the plan."""
    d = DraftPlan(intent="compare_countries", sub_queries=[
        PlannedSubQuery(question="q", country="UK", year_text="2024",
                        year_assumption="You did not give a year for the UK figure."),
    ])
    plan = resolve_plan(d, today=TODAY, known=KNOWN)
    assert plan.sub_queries[0].assumption_note.startswith("You did not give")
    assert plan.assumption_notes == [
        "UK 2024-25: You did not give a year for the UK figure."
    ]


def test_clarification_and_refusal_intents_pass_straight_through():
    clarify = resolve_plan(DraftPlan(intent="needs_clarification",
                                     clarifying_question="Which year?"),
                           today=TODAY, known=KNOWN)
    assert clarify.clarifying_question == "Which year?"

    refuse = resolve_plan(DraftPlan(intent="out_of_scope",
                                    out_of_scope_reason="Asks which option is better for them."),
                          today=TODAY, known=KNOWN)
    assert refuse.refusal_reason == "personalised_advice"


def test_a_clarification_with_no_question_still_asks_something():
    plan = resolve_plan(DraftPlan(intent="needs_clarification"), today=TODAY, known=KNOWN)
    assert plan.clarifying_question and plan.clarifying_question.endswith("?")


# ------------------------------------------- q12: the refusal reason, from code --
#
# A real planner response for q12: out_of_scope with no sub-queries and the year
# only in free text. The refusal reason should still be `year_not_in_corpus`.

Q12 = "What is the UK Personal Allowance for the 2026-27 tax year?"
Q12_DRAFT = ('{"intent":"out_of_scope","sub_queries":[],"clarifying_question":null,'
             '"out_of_scope_reason":"The requested tax year 2026-27 is not covered by '
             'the available data."}')


def test_q12s_cached_refusal_gets_its_reason_from_the_resolver():
    """The year is resolved from the user's question, and the detail matches the
    sub-query path's refusal for the same year."""
    plan = resolve_plan(DraftPlan.model_validate_json(Q12_DRAFT), today=TODAY,
                        known=KNOWN, question=Q12)
    via_sub_query = resolve_plan(draft(("q", "UK", "2026-27")), today=TODAY, known=KNOWN)

    assert plan.intent == "out_of_scope"
    assert plan.refusal_reason == "year_not_in_corpus"
    assert plan.refusal_detail == via_sub_query.refusal_detail
    assert plan.refusal_detail.startswith("I do not hold UK 2026-27.")


def test_without_the_question_the_old_mapping_is_unchanged():
    """Without `question`, the planner's reason maps to `other`."""
    plan = resolve_plan(DraftPlan.model_validate_json(Q12_DRAFT), today=TODAY, known=KNOWN)
    assert plan.refusal_reason == "other"


def test_an_advice_refusal_stays_advice_even_when_it_names_a_missing_year():
    """Personal-advice refusals keep that reason even if the year is missing."""
    d = DraftPlan(intent="out_of_scope",
                  out_of_scope_reason="Asks which option is better for their situation.")
    plan = resolve_plan(d, today=TODAY, known=KNOWN,
                        question="Should I itemise or take the standard deduction in 2026?")
    assert plan.refusal_reason == "personalised_advice"


def test_a_refusal_with_no_resolvable_year_still_says_other():
    d = DraftPlan(intent="out_of_scope", out_of_scope_reason="Canada is not covered.")
    plan = resolve_plan(d, today=TODAY, known=KNOWN,
                        question="What is the Canadian basic personal amount?")
    assert plan.refusal_reason == "other"
    assert plan.refusal_detail == "Canada is not covered."


def test_a_refusal_naming_a_year_we_hold_is_not_turned_into_a_coverage_refusal():
    """2024 is held, so the planner's reason is kept."""
    d = DraftPlan(intent="out_of_scope", out_of_scope_reason="Canada is not covered.")
    plan = resolve_plan(d, today=TODAY, known=KNOWN,
                        question="What is the Canadian basic personal amount for 2024?")
    assert plan.refusal_reason == "other"


def test_an_ambiguous_year_in_a_refusal_is_left_to_the_planner():
    d = DraftPlan(intent="out_of_scope", out_of_scope_reason="Not covered.")
    plan = resolve_plan(d, today=TODAY, known=KNOWN, question="US standard deduction FY24?")
    assert plan.refusal_reason == "other"


def test_the_refusal_names_the_country_the_question_names():
    """The refusal names the country from the question, not the first in coverage."""
    known = {"UK": KNOWN["UK"], "US": KNOWN["US"]}
    d = DraftPlan(intent="out_of_scope", out_of_scope_reason="2026 is not covered.")
    us = resolve_plan(d, today=TODAY, known=known,
                      question="What is the US standard deduction for 2026?")
    assert us.refusal_reason == "year_not_in_corpus"
    assert us.refusal_detail.startswith("I do not hold US 2026.")
    # "us" the pronoun is not the US: with US listed first, only the UK is named.
    known = {"US": KNOWN["US"], "UK": KNOWN["UK"]}
    uk = resolve_plan(d, today=TODAY, known=known,
                      question="Tell us the UK higher rate for 2026-27.")
    assert uk.refusal_detail.startswith("I do not hold UK 2026-27.")


# ------------------------------------------------- q07: a leg with no year --
#
# A real planner response for q07: the year is on the UK leg only and the US leg's
# `year_text` is empty.

Q07 = ("How does the US standard deduction for a single filer compare with the UK "
       "Personal Allowance for 2024?")
Q07_DRAFT = ('{"intent":"compare_countries","sub_queries":[{"question":"What is the standard '
             'deduction for a single filer in the US?","country":"US","year_text":"",'
             '"year_assumption":"You did not give a year for the US figure, so I used the most '
             'recent year mentioned in the question, 2024."},{"question":"What is the Personal '
             'Allowance in the UK?","country":"UK","year_text":"2024","year_assumption":null}],'
             '"clarifying_question":null,"out_of_scope_reason":null}')


def test_q07s_blank_leg_takes_its_year_from_the_question():
    """The question names 2024 once; it applies to both legs."""
    plan = resolve_plan(DraftPlan.model_validate_json(Q07_DRAFT), today=TODAY,
                        known=KNOWN, question=Q07)

    assert plan.intent == "compare_countries"
    assert plan.refusal_reason is None
    assert resolved(plan) == [("US", 2024, "2024"), ("UK", 2024, "2024-25")]


def test_a_year_taken_from_the_question_is_said_out_loud():
    """With no `year_assumption` from the model, the code adds the note."""
    d = draft(("US standard deduction single", "US", ""), ("UK Personal Allowance", "UK", "2024"),
              intent="compare_countries")
    plan = resolve_plan(d, today=TODAY, known=KNOWN, question=Q07)

    us = next(sq for sq in plan.sub_queries if sq.country == "US")
    assert us.tax_year_key == 2024
    assert us.assumption_note and "2024" in us.assumption_note


def test_a_blank_leg_is_not_lent_a_year_when_the_question_names_two():
    """With two years in the question, the blank leg is not assigned either."""
    question = ("How did the US standard deduction change from 2023 to 2024, and how "
                "does it compare to the UK Personal Allowance?")
    d = draft(("US standard deduction", "US", "2023"), ("US standard deduction", "US", "2024"),
              ("UK Personal Allowance", "UK", ""), intent="compare_countries")
    plan = resolve_plan(d, today=TODAY, known=KNOWN, question=question)

    assert plan.intent in {"out_of_scope", "needs_clarification"}
    assert plan.sub_queries == []


def test_without_the_question_a_blank_leg_still_refuses():
    """Without the question there is no year to use, so the plan refuses."""
    plan = resolve_plan(DraftPlan.model_validate_json(Q07_DRAFT), today=TODAY, known=KNOWN)
    assert plan.intent == "out_of_scope"
    assert plan.refusal_reason == "unresolvable_year"


def test_a_blank_year_refusal_says_no_year_was_given_and_does_not_quote_it():
    """The refusal says no year was given rather than quoting an empty string."""
    no_year = "How does the US standard deduction compare with the UK Personal Allowance?"
    for question in (None, no_year):
        plan = resolve_plan(DraftPlan.model_validate_json(Q07_DRAFT), today=TODAY,
                            known=KNOWN, question=question)
        assert plan.refusal_reason == "unresolvable_year", question
        assert "''" not in plan.refusal_detail and '""' not in plan.refusal_detail, question
        assert "US" in plan.refusal_detail, question


# -------------------------------------------- q05: a region the user stated --

from agents.planner import REGION_STATED_NOTE, named_region, region_stated_note

Q05 = "Did the top rate of income tax in Scotland change between 2023-24 and 2024-25?"
# Real clarifying questions for q05, q10 and q09. q10 names no region, so it should
# still be asked; q09 is about the year.
Q05_ASKED = ("Do you want the top rate of income tax for Scotland specifically, or for "
             "the rest of the UK?")
Q10 = "I earn £50,000 a year - which tax band am I in for 2024-25?"
Q10_ASKED = "Are you asking about the tax band for England/Wales/Northern Ireland or for Scotland?"
Q09 = "What's the standard deduction?"
Q09_ASKED = "Which tax year would you like the standard deduction information for (e.g., 2023 or 2024)?"


def clarifying(question):
    return DraftPlan(intent="needs_clarification", clarifying_question=question)


def test_named_region_reads_each_region_in_its_canonical_spelling():
    assert named_region(Q05) == "Scotland"
    assert named_region("tax in northern ireland") == "Northern Ireland"
    assert named_region("and for the Rest of the UK?") == "rest of the UK"
    assert named_region("Wales 2024-25") == "Wales"
    assert named_region(Q10) is None
    assert named_region("Scotlandish") is None


def test_the_region_guard_fires_for_q05():
    assert region_stated_note(clarifying(Q05_ASKED), Q05) == \
        REGION_STATED_NOTE.format(region="Scotland")


def test_the_region_guard_does_not_fire_for_q10_which_names_no_region():
    assert region_stated_note(clarifying(Q10_ASKED), Q10) is None


def test_the_region_guard_does_not_fire_for_a_year_question():
    assert region_stated_note(clarifying(Q09_ASKED), Q09) is None
    # A region in the question does not matter if the clarification is about the year.
    assert region_stated_note(clarifying("Which tax year do you mean?"),
                              "Scottish top rate in Scotland?") is None


def test_the_region_guard_only_reads_clarifications():
    assert region_stated_note(draft(("q", "UK", "2024-25")), Q05) is None
