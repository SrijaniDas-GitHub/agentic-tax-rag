"""Offline tests for the eval harness: token pacer, scorers and report rendering."""

from __future__ import annotations

from datetime import date

import pytest

from agents.contracts import EvidencePack, Passage, Plan, SubQuery
from eval.run_eval import (
    TokenPacer,
    aggregate,
    behaviour_detail,
    observed_behavior,
    page_of,
    render_report,
    score_question,
    tokens_by_model,
)

MODEL = "openai/gpt-oss-120b"
FAST = "openai/gpt-oss-20b"


# --------------------------------------------------------------------- pacer --

def test_pacer_lets_the_first_turn_straight_through():
    """A full bucket means no wait before question 1."""
    pacer = TokenPacer()
    pacer.expected = {MODEL: TokenPacer.SEED}
    assert pacer.delay() == 0.0


def test_pacer_waits_only_for_what_has_to_refill():
    """The wait is proportional to the shortfall, not a fixed minute."""
    pacer = TokenPacer()
    pacer.record(MODEL, 3800)
    wait = pacer.delay()
    assert 0 < wait < 10, wait
    # 7200 available - 3800 spent = 3400; short 400 at 120 tok/s.
    assert wait == pytest.approx(400 / pacer.rate, rel=0.2)


def test_pacer_never_waits_longer_than_a_full_bucket():
    """Even a turn that overdraws completely refills within the window."""
    pacer = TokenPacer()
    pacer.record(MODEL, 20_000)
    assert pacer.delay() <= 60.0 + 1


def test_pacer_meters_each_model_separately():
    """Each model has its own bucket; draining one does not affect the other."""
    pacer = TokenPacer()
    pacer.record(MODEL, 7000)
    drained = pacer.delay()
    fresh = TokenPacer()
    fresh.record(FAST, 7000)
    assert drained == pytest.approx(fresh.delay(), rel=0.05)

    both = TokenPacer()
    both.record(MODEL, 7000)
    both.record(FAST, 100)
    # The wait is set by the worst bucket, not by their sum or their average.
    assert both.delay() == pytest.approx(drained, abs=1.0)


def test_pacer_expectation_grows_to_the_largest_turn_seen_but_stops_at_the_limit():
    pacer = TokenPacer()
    pacer.record(MODEL, 100)
    assert pacer.expected[MODEL] == TokenPacer.SEED   # never below the seed

    pacer.record(MODEL, 5000)
    assert pacer.expected[MODEL] == 5000

    # Capped at the bucket size; waiting cannot help a turn larger than that.
    pacer.record(MODEL, 9000)
    assert pacer.expected[MODEL] == pacer.limit


# ------------------------------------------------------------------ scorers --

def test_page_of_matches_the_gold_page_format():
    assert page_of("us_p17_2024:98:0") == "us_p17_2024|98"
    assert page_of("uk_rates_2024:5:1") == "uk_rates_2024|5"
    # An explicit `page` takes precedence over the one in the chunk_id.
    assert page_of("us_p17_2024:98:0", 96) == "us_p17_2024|96"


def test_observed_behaviour_is_read_off_the_terminal_node():
    """Behaviour comes from which node ran, not from the answer text."""
    assert observed_behavior([{"step": "plan"}, {"step": "clarify"}]) == "clarify"
    assert observed_behavior([{"step": "plan"}, {"step": "refuse"}]) == "refuse"
    assert observed_behavior([{"step": "synthesize"}, {"step": "ground_check"}]) == "answer"
    assert observed_behavior([{"step": "plan"}]) == "error"


def test_refusal_reason_is_checked_against_the_enum():
    gold = {"expected_behavior": "refuse", "refusal_because": "year_not_in_corpus"}
    state = {"trace": [{"step": "refuse", "reason": "year_not_in_corpus"}]}
    assert behaviour_detail(gold, state) == ("year_not_in_corpus", True)

    wrong = {"trace": [{"step": "refuse", "reason": "personalised_advice"}]}
    _, ok = behaviour_detail(gold, wrong)
    assert ok is False


def test_clarification_topic_is_checked_by_keyword():
    """q10 must ask about Scotland, not about the year."""
    gold = {"expected_behavior": "clarify", "clarification_about": "scotland_or_rest_of_uk"}
    right = {"trace": [{"step": "clarify", "question": "Are you in Scotland or elsewhere?"}]}
    wrong = {"trace": [{"step": "clarify", "question": "Which tax year do you mean?"}]}
    assert behaviour_detail(gold, right)[1] is True
    assert behaviour_detail(gold, wrong)[1] is False


def test_tokens_are_attributed_to_the_model_that_spent_them():
    """Tokens are summed per model."""
    trace = [
        {"step": "plan", "llm": {"tokens": 1800, "model": MODEL}},
        {"step": "search.attempt", "tokens": 900, "model": FAST},
        {"step": "search.attempt", "tokens": 950, "model": FAST},
        {"step": "synthesize", "llm": {"tokens": 2100, "model": MODEL}},
    ]
    assert tokens_by_model(trace) == {MODEL: 3900, FAST: 1850}


# ------------------------------------------------------- scoring one question --

def _passage(chunk_id: str, page: int, text: str = "The amount is $14,600.") -> Passage:
    return Passage(chunk_id=chunk_id, text=text, country="US", tax_year_label="2024",
                   doc_id=chunk_id.split(":")[0], doc_title="IRS Publication 17 (2024)",
                   page=page, printed_page=page - 2, url="", score=0.5)


def _answered_state(*, attempt1_pages: list[str], final_chunk: str, final_page: int) -> dict:
    sub = SubQuery(question="standard deduction 2024", country="US", tax_year_key=2024,
                   tax_year_label="2024", year_text="2024")
    pack = EvidencePack(sub_query=sub, passages=[_passage(final_chunk, final_page)],
                        sufficient=True, reason="found", attempts=2,
                        queries_tried=["q1", "q2"], answer_chunk_id=final_chunk)
    return {
        "plan": Plan(intent="lookup", sub_queries=[sub]),
        "verified_evidence": [pack],
        "final_answer": "The standard deduction was $14,600 [1].",
        "grounding_report": {"ok": True, "checked": 1, "grounded": 1,
                             "violations": [], "numbers": []},
        "timings": {"total_ms": 6900.0},
        "trace": [
            {"step": "plan", "llm": {"tokens": 1800, "model": MODEL}},
            {"step": "search.attempt", "sub_query": "US/2024", "attempt": 1,
             "chunk_ids": [f"{p.split('|')[0]}:{p.split('|')[1]}:0" for p in attempt1_pages],
             "tokens": 800, "model": FAST},
            {"step": "search.attempt", "sub_query": "US/2024", "attempt": 2,
             "chunk_ids": [final_chunk], "tokens": 800, "model": FAST},
            {"step": "synthesize", "llm": {"tokens": 2000, "model": MODEL}},
            {"step": "ground_check"},
        ],
    }


GOLD_Q01 = {
    "id": "q01", "type": "lookup", "question": "What is the 2024 standard deduction?",
    "expected_filters": [{"country": "US", "tax_year_key": 2024, "question": "..."}],
    "gold_pages": ["us_p17_2024|98"], "expected_answer_contains": ["14,600"],
    "expected_behavior": "answer",
}


def test_scores_a_clean_answer():
    state = _answered_state(attempt1_pages=["us_p17_2024|98"],
                            final_chunk="us_p17_2024:98:0", final_page=98)
    score = score_question(GOLD_Q01, state)
    assert score.filters_ok is True
    assert score.contains_ok is True
    assert score.sub_queries[0]["hit_attempt1"] is True
    assert score.sub_queries[0]["hit_final"] is True
    assert score.violations == []


def test_recall_is_scored_per_attempt_so_the_retry_loop_is_a_number():
    """Attempt 1 misses the gold page and attempt 2 finds it."""
    state = _answered_state(attempt1_pages=["us_p17_2024|22", "us_p17_2024|9"],
                            final_chunk="us_p17_2024:98:0", final_page=98)
    row = score_question(GOLD_Q01, state).sub_queries[0]
    assert row["hit_attempt1"] is False
    assert row["hit_final"] is True
    assert row["attempts"] == 2

    agg = aggregate([score_question(GOLD_Q01, state)])
    assert agg["recall@8_attempt1"] == 0.0
    assert agg["recall@8"] == 1.0
    assert agg["retries_fired"] == 1
    assert agg["retries_recovered"] == 1


def test_filters_are_compared_as_pairs_not_as_question_text():
    """Sub-query wording varies between runs; only the filter pair is compared."""
    state = _answered_state(attempt1_pages=["us_p17_2024|98"],
                            final_chunk="us_p17_2024:98:0", final_page=98)
    state["plan"].sub_queries[0].question = "wildly different phrasing, same filter"
    assert score_question(GOLD_Q01, state).filters_ok is True


def test_wrong_year_dispatched_fails_filter_precision():
    state = _answered_state(attempt1_pages=["us_p17_2024|98"],
                            final_chunk="us_p17_2024:98:0", final_page=98)
    state["plan"].sub_queries[0].tax_year_key = 2023
    assert score_question(GOLD_Q01, state).filters_ok is False


def test_missing_figure_is_named_not_just_counted():
    state = _answered_state(attempt1_pages=["us_p17_2024|98"],
                            final_chunk="us_p17_2024:98:0", final_page=98)
    state["final_answer"] = "The standard deduction went up."
    score = score_question(GOLD_Q01, state)
    assert score.contains_ok is False
    assert score.missing_contains == ["14,600"]


def test_behavioural_questions_are_excluded_from_filter_precision():
    """Clarify/refuse questions do not count towards filter precision."""
    gold = {"id": "q11", "type": "out_of_scope", "question": "Should I itemise?",
            "expected_filters": [], "gold_pages": [], "expected_answer_contains": [],
            "expected_behavior": "refuse", "refusal_because": "personalised_advice"}
    state = {"plan": Plan(intent="out_of_scope", refusal_reason="personalised_advice"),
             "final_answer": "I can't advise.", "timings": {"total_ms": 2400.0},
             "trace": [{"step": "plan"}, {"step": "refuse",
                                          "reason": "personalised_advice"}]}
    score = score_question(gold, state)
    assert score.filters_ok is None
    assert score.behaviour_ok is True
    assert aggregate([score])["filter_precision"] == 0.0   # no answerable questions at all


def test_aggregate_counts_violations_by_subkind():
    """Violations are counted by subkind."""
    state = _answered_state(attempt1_pages=["us_p17_2024|98"],
                            final_chunk="us_p17_2024:98:0", final_page=98)
    state["grounding_report"] = {
        "ok": False, "checked": 3, "grounded": 1, "numbers": [],
        "violations": [
            {"claim": 0, "kind": "ungrounded_number", "subkind": "derived_arithmetic",
             "detail": "$750"},
            {"claim": 1, "kind": "ungrounded_number", "subkind": "invented_figure",
             "detail": "$99"},
            {"claim": 1, "kind": "unknown_citation", "detail": "nope:1:0"},
        ],
    }
    agg = aggregate([score_question(GOLD_Q01, state)])
    assert agg["grounding_violations"] == 3
    assert agg["violations_by_kind"] == {
        "derived_arithmetic": 1, "invented_figure": 1, "unknown_citation": 1
    }


def test_a_failed_turn_is_a_result_not_a_crashed_harness():
    """An exception in one question is recorded as an error, not raised."""
    import asyncio

    import eval.run_eval as harness

    async def boom(question, **kwargs):
        raise RuntimeError("provider down")

    original = harness.run_pass.__globals__  # run_pass imports `run` inside itself
    import agents.runner

    saved = agents.runner.run
    agents.runner.run = boom
    try:
        scores = asyncio.run(harness.run_pass([GOLD_Q01], paced=False, label="t"))
    finally:
        agents.runner.run = saved
    assert original is not None
    assert scores[0].observed_behavior == "error"
    assert "provider down" in scores[0].error
    assert scores[0].behaviour_ok is False


# ------------------------------------------------------------------- report --

def test_report_renders_the_metric_table_and_both_runs():
    state = _answered_state(attempt1_pages=["us_p17_2024|98"],
                            final_chunk="us_p17_2024:98:0", final_page=98)
    scores = [score_question(GOLD_Q01, state)]
    payload = {
        "label": "baseline",
        "generated": "2026-09-20 00:00 UTC",
        "config": {"orchestrator": "runner", "allow_fast_model": "1"},
        "aggregate": aggregate(scores),
        "latency": {"cold_p50": 6.9, "cold_max": 11.5, "cached_p50": 0.4,
                    "cached_max": 0.5, "paced_s": 42.0},
        "questions": [__import__("dataclasses").asdict(s) for s in scores],
    }
    other = {**payload, "label": "diversify"}
    text = render_report({"baseline": payload, "diversify": other})

    assert "| Filter precision |" in text
    assert "| Grounding violations |" in text
    assert "| p50 latency (cold) |" in text
    # Cold and cached latency are reported with the pacing explanation.
    assert "per turn, not per minute" in text
    assert "0.4" in text and "6.9" in text
    # Both runs appear in the comparison.
    assert "diversify_by_page" in text
    assert "derived_arithmetic" in text
    assert "No LLM judge" in text


def test_report_refuses_to_render_from_nothing():
    with pytest.raises(SystemExit):
        render_report({})


def test_eval_today_is_injected_not_read_off_the_clock():
    """Relative years depend on `today`, so the eval uses a fixed date."""
    from eval.run_eval import EVAL_TODAY

    assert isinstance(EVAL_TODAY, date)


# ---------------------------------------------------------------- follow-ups --

def _answered(year: int, figure: str, question: str) -> dict:
    """An answered US state for `year`, citing Table 10-1 on PDF page 98."""
    chunk = f"us_p17_{year}:98:3"
    sub = SubQuery(question=f"standard deduction single {year}", country="US",
                   tax_year_key=year, tax_year_label=str(year), year_text=str(year))
    passage = Passage(chunk_id=chunk, text=f"Single ${figure}", country="US",
                      tax_year_label=str(year), doc_id=f"us_p17_{year}",
                      doc_title=f"IRS Publication 17 ({year})", page=98, printed_page=96,
                      url="", score=0.5)
    return {
        "user_query": question,
        "plan": Plan(intent="lookup", sub_queries=[sub]),
        "verified_evidence": [EvidencePack(sub_query=sub, passages=[passage], sufficient=True,
                                           reason="found", attempts=1, queries_tried=["q"])],
        "final_answer": f"The standard deduction was ${figure} [1].",
        "grounding_report": {"ok": True, "checked": 1, "grounded": 1, "violations": []},
        "timings": {"total_ms": 1000.0},
        "trace": [
            {"step": "plan", "llm": {"tokens": 1000, "model": MODEL}},
            {"step": "search.attempt", "sub_query": f"US/{year}", "attempt": 1,
             "chunk_ids": [chunk], "tokens": 500, "model": FAST},
            {"step": "ground_check"},
        ],
    }


def test_the_followup_set_is_separate_and_points_at_the_gold_set():
    """Follow-ups are separate from the 12 gold questions and reference turn 1 by id."""
    from eval.retrieval_eval import load_gold
    from eval.run_eval import load_followups

    gold = {q["id"]: q for q in load_gold()}
    rows = load_followups()
    assert len(gold) == 12
    assert [r["id"] for r in rows] == ["f01"]
    f01 = rows[0]
    assert f01["previous_turn"] == "q01" and f01["previous_turn"] in gold
    assert f01["question"] == "and what about 2023?"
    assert [(f["country"], f["tax_year_key"]) for f in f01["expected_filters"]] == [("US", 2023)]
    assert f01["expected_answer_contains"] == ["13,850"]
    assert f01["expected_behavior"] == "answer"
    assert not {r["id"] for r in rows} & set(gold)


def test_a_followup_runs_turn_1_carries_like_the_app_and_scores_turn_2():
    import asyncio

    import agents.runner
    import eval.run_eval as harness
    from agents.carry import carry
    from eval.retrieval_eval import load_gold

    gold = {q["id"]: q for q in load_gold()}
    f01 = harness.load_followups()[0]
    turn1 = _answered(2024, "14,600", gold["q01"]["question"])
    seen: list[tuple[str, object]] = []

    async def fake_run(question, *, today=None, carried_filters=None):
        seen.append((question, carried_filters))
        return turn1 if question == gold["q01"]["question"] else \
            _answered(2023, "13,850", question)

    saved = agents.runner.run
    agents.runner.run = fake_run
    try:
        [score] = asyncio.run(harness.run_pass([f01], paced=False, label="t", previous=gold))
    finally:
        agents.runner.run = saved

    assert seen == [(gold["q01"]["question"], None), ("and what about 2023?", carry(turn1))]
    assert score.id == "f01" and score.previous_turn == "q01"
    assert score.turn1_behavior == "answer"
    assert score.carried["previous_question"] == gold["q01"]["question"]
    assert score.filters_ok is True and score.contains_ok is True
    assert score.sub_queries[0]["hit_final"] is True     # us_p17_2023|98
    assert score.total_ms == 1000.0                        # turn 2 only


def test_a_plain_pass_carries_nothing():
    """Without `previous`, nothing is carried between questions."""
    import asyncio

    import agents.runner
    import eval.run_eval as harness

    seen = []

    async def fake_run(question, *, today=None, carried_filters=None):
        seen.append(carried_filters)
        return _answered(2024, "14,600", question)

    saved = agents.runner.run
    agents.runner.run = fake_run
    try:
        [score] = asyncio.run(harness.run_pass([GOLD_Q01], paced=False, label="t"))
    finally:
        agents.runner.run = saved
    assert seen == [None]
    assert score.previous_turn is None and score.carried is None


def test_the_followup_results_file_is_not_a_gold_set_run(tmp_path, monkeypatch):
    """Follow-up results use a different question set and are loaded separately."""
    import json

    import eval.run_eval as harness

    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    for label in ("baseline", "followups"):
        (tmp_path / f"{label}.json").write_text(json.dumps(
            {"label": label, "generated": "x", "config": {}, "latency": {}, "questions": []}),
            encoding="utf-8")
    assert list(harness.load_results()) == ["baseline"]
    assert harness.load_followup_results()["label"] == "followups"


def test_an_only_run_is_kept_on_disk_but_left_out_of_the_comparison(tmp_path, monkeypatch):
    """`--only` spot checks are saved but excluded from the comparison."""
    import json

    import eval.run_eval as harness

    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    for label, only in (("baseline", ""), ("diversify", ""), ("q07_fix", "q07")):
        (tmp_path / f"{label}.json").write_text(
            json.dumps(_payload(label, {"only": only})), encoding="utf-8")

    assert list(harness.load_results()) == ["baseline", "diversify"]
    assert (tmp_path / "q07_fix.json").exists()
    report = harness.render_report(harness.load_results())
    assert "q07_fix" not in report


def test_an_only_run_under_the_default_label_is_refused_before_it_runs(monkeypatch, capsys):
    """`--only` without `--label` would overwrite the baseline, so it is rejected."""
    import pytest

    import eval.run_eval as harness

    def must_not_run(*args, **kwargs):
        raise AssertionError("the eval started")

    monkeypatch.setattr(harness, "run_pass", must_not_run)
    with pytest.raises(SystemExit) as exit_:
        harness.main(["--only", "q07"])
    assert exit_.value.code == 2
    assert "--label" in capsys.readouterr().err


def _payload(label: str, config: dict | None = None) -> dict:
    import dataclasses

    scores = [score_question(GOLD_Q01, _answered_state(
        attempt1_pages=["us_p17_2024|98"], final_chunk="us_p17_2024:98:0", final_page=98))]
    return {
        "label": label, "generated": "2026-09-21 05:59 UTC",
        "config": {"orchestrator": "runner", "allow_fast_model": "1", **(config or {})},
        "aggregate": aggregate(scores),
        "latency": {"cold_p50": 2.6, "cold_max": 5.0, "cached_p50": 0.2,
                    "cached_max": 0.3, "paced_s": 0.0},
        "questions": [dataclasses.asdict(s) for s in scores],
    }


def _followup_section_of(text: str) -> str:
    """The diversification section has its own "Not run." text, so scope to this one."""
    return text.split("## Follow-ups", 1)[1].split("\n## ", 1)[0]


def test_report_says_the_followups_have_not_run_until_they_have():
    section = _followup_section_of(render_report({"baseline": _payload("baseline")}))
    assert "**Not run.**" in section
    assert "--followups" in section


def test_report_renders_the_followup_rows_once_they_exist():
    followups = _payload("followups", {"commit": "abc1234", "cold": True})
    followups["questions"][0].update(id="f01", previous_turn="q01", turn1_behavior="answer",
                                     carried={"resolved": [{"country": "US",
                                                            "tax_year_key": 2024,
                                                            "tax_year_label": "2024"}]})
    text = render_report({"baseline": _payload("baseline")}, followups)
    assert "**Not run.**" not in _followup_section_of(text)
    assert "| f01 | q01 answer | US 2024 |" in text
    assert "`abc1234`" in text


def test_report_says_the_committed_runs_predate_the_planner_changes():
    """Runs saved without a commit hash are flagged as predating the planner changes."""
    text = render_report({"baseline": _payload("baseline"), "diversify": _payload("diversify")})
    assert "| `baseline` | `850ae31` |" in text
    assert "| `diversify` | `850ae31` |" in text
    assert "measured before two planner changes" in text

    fresh = render_report({"baseline": _payload("baseline", {"commit": "def5678"})})
    assert "| `baseline` | `def5678` |" in fresh
    assert "measured before two planner changes" not in fresh


def test_every_new_results_file_records_its_commit():
    import re

    from eval.run_eval import git_commit

    assert re.fullmatch(r"[0-9a-f]{7,}(-dirty)?|unknown", git_commit())
