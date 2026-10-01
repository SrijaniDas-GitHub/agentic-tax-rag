"""Node tests, each node called directly with a state dict.

Nodes that call an LLM use `FakeLLM`. Retrieval-backed tests are skipped when
the index has not been built.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents.contracts import (
    AnswerClaim,
    DraftAnswer,
    DraftPlan,
    EvidencePack,
    Passage,
    Plan,
    PlannedSubQuery,
    SubQuery,
    Sufficiency,
)
from agents.nodes import (
    clarify_node,
    ground_check_node,
    plan_node,
    refuse_node,
    search_node,
    synthesize_node,
    verify_node,
)
from agents.planner import resolve_plan
from agents.state import initial_state
from core.llm import reset_client, set_client
from core.paths import BM25_PATH, CHROMA_DIR
from tests.fakes import FakeLLM

TODAY = date(2026, 9, 20)

needs_index = pytest.mark.skipif(
    not (CHROMA_DIR.exists() and BM25_PATH.exists()),
    reason="no index - run `uv run python -m ingest.build_index`",
)


@pytest.fixture(autouse=True)
def _no_real_llm():
    """Reset the LLM client after each test."""
    yield
    reset_client()


def state_with(**overrides):
    state = initial_state("q", today=TODAY)
    state.update(overrides)
    return state


def passage(chunk_id="us_p17_2024:98:0", text="| Single | $14,600 |",
            country="US", label="2024", page=98, printed=96, doc="IRS Publication 17 (2024)"):
    return Passage(chunk_id=chunk_id, text=text, country=country, tax_year_label=label,
                   doc_id="us_p17_2024", doc_title=doc, page=page, printed_page=printed,
                   score=0.5)


def pack(sufficient=True, passages=None, country="US", key=2024, label="2024", attempts=1):
    return EvidencePack(
        sub_query=SubQuery(question="standard deduction single 2024", country=country,
                           tax_year_key=key, tax_year_label=label),
        passages=passages if passages is not None else [passage()],
        sufficient=sufficient,
        reason="ok" if sufficient else "no rate schedule in the pack",
        attempts=attempts,
    )


# ------------------------------------------------------------------ plan_node --

async def test_plan_node_resolves_the_year_itself_and_never_takes_one_from_the_model():
    """The model returns `year_text`; the key comes from the resolver."""
    set_client(FakeLLM(DraftPlan=DraftPlan(
        intent="lookup",
        sub_queries=[PlannedSubQuery(question="UK Personal Allowance",
                                     country="UK", year_text="2024-25")],
    )))
    update = await plan_node(state_with(user_query="UK personal allowance 2024-25"))
    plan: Plan = update["plan"]

    assert plan.intent == "lookup"
    assert [(s.country, s.tax_year_key, s.tax_year_label) for s in plan.sub_queries] == [
        ("UK", 2024, "2024-25")
    ]
    assert update["trace"][0]["year_texts"] == ["2024-25"]
    assert "plan_ms" in update["timings"]


async def test_plan_node_returns_a_partial_update_only():
    """A node returns only the fields it changed."""
    set_client(FakeLLM(DraftPlan=DraftPlan(intent="out_of_scope",
                                           out_of_scope_reason="asks for personal advice")))
    update = await plan_node(state_with(user_query="should I itemise?"))
    assert set(update) == {"plan", "timings", "trace"}
    assert "user_query" not in update and "evidence" not in update


async def test_plan_node_routes_an_ambiguous_year_to_clarify_not_refuse():
    """An ambiguous year ("FY24") clarifies; an unreadable one refuses (next test)."""
    set_client(FakeLLM(DraftPlan=DraftPlan(
        intent="lookup",
        sub_queries=[PlannedSubQuery(question="standard deduction", country="US",
                                     year_text="FY24")],
    )))
    plan = (await plan_node(state_with(user_query="standard deduction FY24")))["plan"]
    assert plan.intent == "needs_clarification"
    assert plan.clarifying_question and "FY24" in plan.clarifying_question


async def test_plan_node_refuses_an_unreadable_year():
    set_client(FakeLLM(DraftPlan=DraftPlan(
        intent="lookup",
        sub_queries=[PlannedSubQuery(question="standard deduction", country="US",
                                     year_text="whenever")],
    )))
    plan = (await plan_node(state_with(user_query="standard deduction whenever")))["plan"]
    assert plan.intent == "out_of_scope"
    assert plan.refusal_reason == "unresolvable_year"


async def test_plan_node_refuses_a_year_the_corpus_does_not_hold():
    """2026-27 is a valid UK tax year that the corpus does not hold."""
    set_client(FakeLLM(DraftPlan=DraftPlan(
        intent="lookup",
        sub_queries=[PlannedSubQuery(question="UK Personal Allowance",
                                     country="UK", year_text="2026-27")],
    )))
    plan = (await plan_node(state_with(user_query="UK personal allowance 2026-27")))["plan"]
    assert plan.intent == "out_of_scope"
    assert plan.refusal_reason == "year_not_in_corpus"
    assert "2026-27" in (plan.refusal_detail or "")


async def test_plan_node_passes_the_question_so_q12s_reason_comes_from_code():
    """A planner refusal with no sub-queries still gets `year_not_in_corpus`."""
    from tests.test_planner import Q12, Q12_DRAFT

    set_client(FakeLLM(DraftPlan=DraftPlan.model_validate_json(Q12_DRAFT)))
    plan = (await plan_node(state_with(user_query=Q12)))["plan"]
    assert plan.refusal_reason == "year_not_in_corpus"
    assert plan.refusal_detail.startswith("I do not hold UK 2026-27.")


async def test_plan_node_passes_the_question_so_q07s_blank_leg_is_resolved():
    """q07's draft with an empty US `year_text` becomes two legs, not a refusal.
    The trace keeps the model's original `year_texts`."""
    from tests.test_planner import Q07, Q07_DRAFT

    set_client(FakeLLM(DraftPlan=DraftPlan.model_validate_json(Q07_DRAFT)))
    update = await plan_node(state_with(user_query=Q07))
    plan = update["plan"]

    assert plan.intent == "compare_countries"
    assert [(s.country, s.tax_year_key) for s in plan.sub_queries] == [("US", 2024), ("UK", 2024)]
    assert update["trace"][0]["year_texts"] == ["", "2024"]


# -------------------------------------------------- plan_node: region re-plan --

def _q05_answered():
    return DraftPlan(intent="compare_years", sub_queries=[
        PlannedSubQuery(question="top rate of income tax in Scotland 2023-24",
                        country="UK", year_text="2023-24"),
        PlannedSubQuery(question="top rate of income tax in Scotland 2024-25",
                        country="UK", year_text="2024-25"),
    ])


async def test_a_region_clarification_of_a_question_naming_it_re_plans_once():
    """The question names Scotland but the planner asks which region. One re-plan,
    with the note appended to the user message; the first call is unchanged."""
    from agents.planner import NO_PREVIOUS_TURN, REGION_STATED_NOTE
    from agents.prompts import render
    from tests.test_planner import Q05, Q05_ASKED

    fake = FakeLLM(DraftPlan=[DraftPlan(intent="needs_clarification",
                                        clarifying_question=Q05_ASKED),
                              _q05_answered()])
    set_client(fake)
    update = await plan_node(state_with(user_query=Q05))

    calls = fake.calls_for("DraftPlan")
    assert len(calls) == 2
    first_user = render("planner_user", carried=NO_PREVIOUS_TURN, question=Q05)
    assert calls[0][2] == first_user                       # first call unchanged
    note = REGION_STATED_NOTE.format(region="Scotland")
    assert calls[1][2] == f"{first_user}\n\n{note}"
    assert calls[1][1] == calls[0][1]                     # system prompt untouched

    plan = update["plan"]
    assert plan.intent == "compare_years"
    assert [(s.country, s.tax_year_key) for s in plan.sub_queries] == [("UK", 2023), ("UK", 2024)]

    row = update["trace"][0]
    assert row["replan"]["reason"] == "region_stated"
    assert row["replan"]["first_clarifying_question"] == Q05_ASKED
    assert row["replan"]["note"] == note
    assert row["draft_intent"] == "compare_years"
    assert row["llm"]["calls"] == 2


async def test_the_region_re_plan_fires_at_most_once():
    """If the model clarifies again, that answer stands."""
    from tests.test_planner import Q05, Q05_ASKED

    fake = FakeLLM(DraftPlan=DraftPlan(intent="needs_clarification",
                                       clarifying_question=Q05_ASKED))
    set_client(fake)
    update = await plan_node(state_with(user_query=Q05))

    assert len(fake.calls_for("DraftPlan")) == 2
    assert update["plan"].intent == "needs_clarification"
    assert update["trace"][0]["replan"]["reason"] == "region_stated"


@pytest.mark.parametrize("which", ["q10", "q09"])
async def test_the_region_re_plan_does_not_fire_for_a_fair_clarification(which):
    """q10 names no region, so asking is right; q09's question is about the year."""
    from tests import test_planner as tp

    question, asked = {"q10": (tp.Q10, tp.Q10_ASKED), "q09": (tp.Q09, tp.Q09_ASKED)}[which]
    fake = FakeLLM(DraftPlan=DraftPlan(intent="needs_clarification",
                                       clarifying_question=asked))
    set_client(fake)
    update = await plan_node(state_with(user_query=question))

    assert len(fake.calls_for("DraftPlan")) == 1
    assert update["plan"].clarifying_question == asked
    assert update["trace"][0]["replan"] is None
    assert "calls" not in update["trace"][0]["llm"]


# ---------------------------------------------------------------- search_node --

@needs_index
async def test_search_node_takes_a_SearchState_and_returns_an_appendable_update():
    """The input is `{"sub_query": sq}` and the update is list-shaped for the reducer."""
    set_client(FakeLLM(Sufficiency=Sufficiency(
        sufficient=True, answer_chunk_id=None, reason="found it",
        suggested_query="2024 Tax Rate Schedule X")))
    sq = SubQuery(question="2024 standard deduction single", country="US",
                  tax_year_key=2024, tax_year_label="2024")

    update = await search_node({"sub_query": sq})

    assert list(update["evidence"]) and isinstance(update["evidence"], list)
    assert update["trace"][-1]["step"] == "search"
    # Sufficiency claimed with a chunk_id that is not in the pack -> not sufficient.
    assert update["evidence"][0].sufficient is False


@needs_index
async def test_search_node_never_dispatches_without_both_filters():
    """Every passage matches the sub-query's country and year."""
    set_client(FakeLLM(Sufficiency=Sufficiency(
        sufficient=False, reason="no",
        suggested_query="2023 Tax Rate Schedule X single")))
    sq = SubQuery(question="top tax rate", country="US", tax_year_key=2023,
                  tax_year_label="2023")

    update = await search_node({"sub_query": sq})
    passages = update["evidence"][0].passages

    assert passages, "expected a non-empty pack"
    assert {p.country for p in passages} == {"US"}
    assert {p.tax_year_label for p in passages} == {"2023"}


# ---------------------------------------------------------------- verify_node --

async def test_verify_node_drops_off_target_passages_and_counts_them():
    """Passages from the wrong country or year are dropped and counted."""
    stray = passage(chunk_id="uk_rates_2024:6:0", country="UK", label="2024-25",
                    printed=None, page=6, doc="gov.uk rates")
    update = await verify_node(state_with(evidence=[pack(passages=[passage(), stray])]))

    assert update["verification"]["off_target_dropped"] == 1
    assert [p.chunk_id for p in update["verified_evidence"][0].passages] == ["us_p17_2024:98:0"]


async def test_verify_node_does_not_write_back_to_the_appending_field():
    """`evidence` has an append reducer, so cleaned packs go in `verified_evidence`."""
    update = await verify_node(state_with(evidence=[pack()]))
    assert "evidence" not in update
    assert update["verified_evidence"]


async def test_verify_node_reports_what_is_missing_not_just_how_many():
    update = await verify_node(state_with(evidence=[pack(sufficient=False, attempts=2)]))
    gap = update["verification"]["insufficient"][0]
    assert gap["sub_query"] == "US 2024"
    assert gap["attempts"] == 2
    assert "rate schedule" in gap["reason"]


# ------------------------------------------------------------ synthesize_node --

async def test_synthesize_node_puts_assumption_notes_in_the_prompt():
    """Plan-level and sub-query assumption notes reach the synthesizer prompt."""
    fake = FakeLLM(DraftAnswer=DraftAnswer(
        claims=[AnswerClaim(text="The standard deduction is $14,600.",
                            chunk_ids=["us_p17_2024:98:0"])]))
    set_client(fake)

    noted = pack()
    noted.sub_query.assumption_note = "You did not give a year for the UK leg; used 2024-25."
    plan = Plan(intent="lookup", sub_queries=[noted.sub_query],
                assumption_notes=["planner-level note"])

    update = await synthesize_node(state_with(user_query="q", plan=plan,
                                              verified_evidence=[noted]))

    _, _, user_prompt = fake.calls_for("DraftAnswer")[0]
    assert "used 2024-25" in user_prompt
    assert "planner-level note" in user_prompt
    assert update["draft_answer"]["assumption_notes"][0] == "planner-level note"


# ---------------------------------------------------------- ground_check_node --

async def test_ground_check_passes_a_number_that_is_in_the_cited_passage():
    draft = {"claims": [{"text": "The 2024 standard deduction for a single filer is $14,600.",
                         "chunk_ids": ["us_p17_2024:98:0"]}],
             "caveats": [], "assumption_notes": []}
    update = await ground_check_node(state_with(draft_answer=draft,
                                                verified_evidence=[pack()]))

    assert update["grounding_report"]["ok"] is True
    assert update["grounding_report"]["checked"] == 1     # 2024 is whitelisted as a year
    assert "[1]" in update["final_answer"]
    assert update["citations"][0]["label"] == "IRS Publication 17 (2024), p.96"


async def test_ground_check_fails_a_number_cited_to_the_wrong_passage():
    """'29,200' appears in two passages with different meanings; only the cited
    passage counts."""
    deduction = passage(chunk_id="us_p17_2024:98:0", text="| Married filing jointly | $29,200 |")
    threshold = passage(chunk_id="us_p17_2023:9:0", text="| MFJ, one spouse 65+ | $29,200 |",
                        label="2023")
    evidence = [pack(passages=[deduction]),
                pack(passages=[threshold], key=2023, label="2023")]

    good = {"claims": [{"text": "It is $29,200.", "chunk_ids": ["us_p17_2024:98:0"]}],
            "caveats": [], "assumption_notes": []}
    bad = {"claims": [{"text": "It is $29,200.", "chunk_ids": ["us_p17_2024:11:0"]}],
           "caveats": [], "assumption_notes": []}

    assert (await ground_check_node(
        state_with(draft_answer=good, verified_evidence=evidence)))["grounding_report"]["ok"]

    report = (await ground_check_node(
        state_with(draft_answer=bad, verified_evidence=evidence)))["grounding_report"]
    assert report["ok"] is False
    kinds = {v["kind"] for v in report["violations"]}
    assert kinds == {"ungrounded_number", "unknown_citation"}


async def test_ground_check_marks_the_violation_in_the_answer_rather_than_hiding_it():
    draft = {"claims": [{"text": "The allowance is £99,999.", "chunk_ids": []}],
             "caveats": [], "assumption_notes": []}
    update = await ground_check_node(state_with(draft_answer=draft,
                                                verified_evidence=[pack()]))
    assert "[UNVERIFIED FIGURE]" in update["final_answer"]
    assert "could not be verified" in update["final_answer"]


async def test_ground_check_cites_uk_sections_not_pages():
    """UK passages cite a section (§5), not a page."""
    uk = passage(chunk_id="uk_rates_2024:5:0", text="Higher rate 40% £37,701 to £125,140",
                 country="UK", label="2024-25", page=5, printed=None, doc="gov.uk rates")
    draft = {"claims": [{"text": "The higher rate is 40%.", "chunk_ids": ["uk_rates_2024:5:0"]}],
             "caveats": [], "assumption_notes": []}
    update = await ground_check_node(state_with(
        draft_answer=draft,
        verified_evidence=[pack(passages=[uk], country="UK", label="2024-25")]))

    assert update["citations"][0]["label"] == "gov.uk rates, §5"
    assert "p.5" not in update["final_answer"]


# ----------------------------------------------------------- clarify / refuse --

async def test_clarify_node_asks_one_question_and_makes_no_llm_call():
    plan = Plan(intent="needs_clarification",
                clarifying_question="Which tax year do you mean?")
    update = await clarify_node(state_with(plan=plan))
    assert update["final_answer"].startswith("Which tax year do you mean?")
    assert update["final_answer"].count("?") == 1


async def test_refuse_node_names_the_reason_and_what_the_corpus_does_hold():
    plan = Plan(intent="out_of_scope", refusal_reason="personalised_advice",
                refusal_detail="The user asked which option is better for them.")
    answer = (await refuse_node(state_with(plan=plan)))["final_answer"]
    assert "personalised tax advice" in answer
    assert "rates and thresholds" in answer      # offers the figures instead
    assert "US 2023, 2024" in answer


async def test_an_advice_refusal_offers_the_factual_questions_instead():
    plan = resolve_plan(DraftPlan(
        intent="out_of_scope",
        sub_queries=[PlannedSubQuery(question="What is the standard deduction?",
                                     country="US", year_text="")],
        out_of_scope_reason="I cannot provide personalized tax advice."),
        today=TODAY, question="Should I itemise?")
    answer = (await refuse_node(state_with(plan=plan)))["final_answer"]

    assert "- What is the standard deduction?" in answer
    assert "personalized" not in answer      # the planner's restatement is not echoed
    assert answer.index("What is the standard deduction?") < answer.index("I hold")


async def test_refuse_node_handles_the_insufficient_evidence_route():
    """Insufficient evidence: say what is missing rather than answering."""
    verification = {"packs": 1, "sufficient": 0,
                    "insufficient": [{"sub_query": "UK 2024-25", "question": "q",
                                      "reason": "no rate table in the pack", "attempts": 2}]}
    answer = (await refuse_node(state_with(plan=None,
                                           verification=verification)))["final_answer"]
    assert "won't fill that gap from memory" in answer
    assert "no rate table in the pack" in answer
