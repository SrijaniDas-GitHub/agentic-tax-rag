"""Graph nodes, shared by both orchestrators.

    async def node(state) -> dict          # returns a partial update

Nodes are module-level functions that take only state and never mutate it.
`search_node` takes a `SearchState` (one sub-query), which is also the LangGraph
`Send` payload. Nodes get the LLM client from `core.llm.get_client()`, which
tests replace with a fake.
"""

from __future__ import annotations

import time

from agents import budget
from agents import citations as cite
from agents.contracts import AnswerClaim, DraftAnswer, DraftPlan, EvidencePack, Passage, Plan
from agents.planner import (
    carried_context,
    coverage_prompt,
    coverage_sentence,
    region_stated_note,
    resolve_plan,
)
from agents.prompts import load, render
from agents.search_agent import format_passages, run_search_agent
from agents.state import AgentState, SearchState
from core.llm import get_client
from guardrails.numeric_grounding import check_answer

__all__ = [
    "clarify_node",
    "ground_check_node",
    "plan_node",
    "refuse_node",
    "search_node",
    "synthesize_node",
    "verify_node",
]


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _llm(result) -> dict:
    return {"cached": result.cached, "attempts": result.attempts,
            "api_ms": round(result.api_ms, 1),
            "tokens": result.total_tokens, "model": result.model}


# --------------------------------------------------------------------- plan --

async def plan_node(state: AgentState) -> dict:
    """Decompose the question with one LLM call, then resolve tax years in code.

    A second call is made only if the draft asks about a UK region the question
    already names (`agents.planner.region_stated_note`).
    """
    started = time.perf_counter()

    client = get_client()
    system = render("planner_system", coverage=coverage_prompt())
    user = render("planner_user", carried=carried_context(state.get("carried_filters")),
                  question=state["user_query"])
    result = await client.structured(DraftPlan, system, user)
    draft: DraftPlan = result.value
    llm = _llm(result)

    # Re-plan once with a note if the model asked about a region the user named.
    # If it still asks, that answer stands.
    replan = None
    note = region_stated_note(draft, state["user_query"])
    if note:
        retry = await client.structured(DraftPlan, system, f"{user}\n\n{note}")
        replan = {"reason": "region_stated", "note": note,
                  "first_draft_intent": draft.intent,
                  "first_clarifying_question": draft.clarifying_question,
                  "llm": _llm(retry)}
        draft = retry.value
        # Token counters read this row, so it includes both calls.
        llm = {"cached": llm["cached"] and replan["llm"]["cached"],
               "attempts": llm["attempts"] + replan["llm"]["attempts"],
               "api_ms": round(llm["api_ms"] + replan["llm"]["api_ms"], 1),
               "tokens": llm["tokens"] + replan["llm"]["tokens"],
               "model": llm["model"], "calls": 2}

    plan = resolve_plan(draft, today=state["today"], question=state["user_query"])

    return {
        "plan": plan,
        "timings": {"plan_ms": _ms(started)},
        "trace": [{
            "step": "plan",
            "draft_intent": draft.intent,
            "intent": plan.intent,
            "sub_queries": [
                {
                    "question": sq.question,
                    "country": sq.country,
                    "tax_year_key": sq.tax_year_key,
                    "tax_year_label": sq.tax_year_label,
                    "year_text": sq.year_text,
                    "assumption_note": sq.assumption_note,
                }
                for sq in plan.sub_queries
            ],
            "year_texts": [p.year_text for p in draft.sub_queries],
            "refusal_reason": plan.refusal_reason,
            "clarifying_question": plan.clarifying_question,
            "replan": replan,
            "llm": llm,
        }],
    }


# ------------------------------------------------------------------- search --

async def search_node(state: SearchState) -> dict:
    """Search for one sub-query. Branches are merged by the state reducers."""
    started = time.perf_counter()
    sub_query = state["sub_query"]
    pack, trace = await run_search_agent(sub_query)
    label = f"{sub_query.country}/{sub_query.tax_year_key}"

    return {
        "evidence": [pack],
        "timings": {f"search[{label}]_ms": _ms(started)},
        "trace": trace + [{
            "step": "search",
            "sub_query": label,
            "attempts": pack.attempts,
            "sufficient": pack.sufficient,
            "queries_tried": pack.queries_tried,
            "reason": pack.reason,
        }],
    }


# ------------------------------------------------------------------- verify --

def _off_target(pack: EvidencePack) -> list[Passage]:
    """Passages whose country or year is not the one this sub-query asked for."""
    return [
        p for p in pack.passages
        if p.country != pack.sub_query.country
        or p.tax_year_label != pack.sub_query.tax_year_label
    ]


async def verify_node(state: AgentState) -> dict:
    """Check the evidence packs before synthesis. No LLM.

    Drops any passage whose country or tax year does not match its sub-query
    (should never happen with filtered search), and records which sub-queries
    have sufficient evidence for `route_after_verify`.
    """
    started = time.perf_counter()
    packs: list[EvidencePack] = list(state.get("evidence") or [])

    dropped = 0
    cleaned: list[EvidencePack] = []
    for pack in packs:
        strays = {p.chunk_id for p in _off_target(pack)}
        if strays:
            dropped += len(strays)
            keep = [p for p in pack.passages if p.chunk_id not in strays]
            pack = pack.model_copy(update={"passages": keep})
        cleaned.append(pack)

    sufficient = [p for p in cleaned if p.sufficient]
    insufficient = [p for p in cleaned if not p.sufficient]

    verification = {
        "packs": len(cleaned),
        "sufficient": len(sufficient),
        "insufficient": [
            {"sub_query": f"{p.sub_query.country} {p.sub_query.tax_year_label}",
             "question": p.sub_query.question,
             "reason": p.reason,
             "attempts": p.attempts}
            for p in insufficient
        ],
        "off_target_dropped": dropped,
        "total_passages": sum(len(p.passages) for p in cleaned),
    }

    return {
        # `evidence` has an append reducer, so the cleaned packs go in a separate
        # field rather than being written back to it.
        "verification": verification,
        "verified_evidence": cleaned,
        "timings": {"verify_ms": _ms(started)},
        "trace": [{"step": "verify", **verification}],
    }


# --------------------------------------------------------------- synthesize --

def _evidence_block(packs: list[EvidencePack]) -> str:
    blocks = []
    for pack in packs:
        gap = "" if pack.sufficient else f" - SEARCH REPORTED THIS INSUFFICIENT: {pack.reason}"
        head = (f"### Sub-query: {pack.sub_query.question}\n"
                f"(country {pack.sub_query.country}, "
                f"tax year {pack.sub_query.tax_year_label}{gap})")
        shown = format_passages(budget.for_synthesis(pack),
                                limit=budget.SYNTHESIS_PASSAGES,
                                chars=budget.SYNTHESIS_CHARS)
        blocks.append(f"{head}\n\n{shown}")
    return "\n\n==========\n\n".join(blocks)


async def synthesize_node(state: AgentState) -> dict:
    """Write the answer as claims, each bound to the chunk_ids that support it.

    Assumption notes are passed to the prompt and included in the final answer.
    """
    started = time.perf_counter()
    plan: Plan = state["plan"]
    packs: list[EvidencePack] = list(state.get("verified_evidence")
                                     or state.get("evidence") or [])

    # Plan-level notes are prefixed with country and year, so check containment
    # to avoid printing a sub-query's note twice.
    notes = list(plan.assumption_notes)
    for pack in packs:
        note = pack.sub_query.assumption_note
        if note and not any(note in existing for existing in notes):
            notes.append(note)

    assumptions = (
        "Assumptions already made for you (state them in a claim if they affect the "
        "answer):\n" + "\n".join(f"- {n}" for n in notes) + "\n"
        if notes else ""
    )

    result = await get_client().structured(
        DraftAnswer,
        load("synthesize_system"),
        render("synthesize_user", question=state["user_query"],
               assumptions=assumptions, evidence=_evidence_block(packs)),
    )
    draft: DraftAnswer = result.value

    return {
        "draft_answer": {
            "claims": [c.model_dump() for c in draft.claims],
            "caveats": draft.caveats,
            "assumption_notes": notes,
        },
        "timings": {"synthesize_ms": _ms(started)},
        "trace": [{
            "step": "synthesize",
            "claims": len(draft.claims),
            "caveats": len(draft.caveats),
            "assumption_notes": notes,
            "llm": {"cached": result.cached, "attempts": result.attempts,
                    "api_ms": round(result.api_ms, 1),
                    "tokens": result.total_tokens, "model": result.model},
        }],
    }


# -------------------------------------------------------------- ground_check --

def render_answer(claims: list[AnswerClaim], passages: list[Passage],
                  caveats: list[str], notes: list[str],
                  report) -> tuple[str, list[dict]]:
    """Render claims as prose with [n] markers, plus assumptions, caveats and sources.

    Citation labels are formatted in code (`agents/citations.py`), not by the model.
    """
    by_id = {p.chunk_id: p for p in passages}
    cited = [by_id[cid] for c in claims for cid in c.chunk_ids if cid in by_id]
    citation_list, marker = cite.numbered(cited)

    ungrounded = {v["claim"] for v in report.violations if v["kind"] == "ungrounded_number"}

    lines: list[str] = []
    for i, claim in enumerate(claims):
        markers = sorted({marker[cid] for cid in claim.chunk_ids if cid in marker})
        # Normalize the trailing full stop and put markers before it.
        body = claim.text.rstrip().rstrip(".")
        suffix = " " + " ".join(f"[{n}]" for n in markers) if markers else ""
        flag = " [UNVERIFIED FIGURE]" if i in ungrounded else ""
        lines.append(f"{body}{suffix}.{flag}")

    parts = [" ".join(lines)]

    if notes:
        parts.append("Assumptions:\n" + "\n".join(f"- {n}" for n in notes))
    if caveats:
        parts.append("Note:\n" + "\n".join(f"- {c}" for c in caveats))
    if citation_list:
        parts.append("Sources:\n" + "\n".join(f"[{c['n']}] {c['label']}" for c in citation_list))
    if not report.ok:
        parts.append(
            "Grounding check: "
            f"{len(report.violations)} figure(s) could not be verified against the passage "
            "cited for them and are marked above."
        )
    return "\n\n".join(parts), citation_list


async def ground_check_node(state: AgentState) -> dict:
    """Run the numeric grounding check and render the final answer.

    Numbers are checked against the passages each claim cites. Unverified figures
    are flagged in the answer, and every number is listed in `grounding_report`.
    """
    started = time.perf_counter()
    draft = state.get("draft_answer") or {"claims": [], "caveats": [], "assumption_notes": []}
    claims = [AnswerClaim(**c) for c in draft["claims"]]

    packs: list[EvidencePack] = list(state.get("verified_evidence")
                                     or state.get("evidence") or [])
    passages = [p for pack in packs for p in pack.passages]

    report = check_answer(claims, passages)
    answer, citation_list = render_answer(
        claims, passages, draft.get("caveats", []), draft.get("assumption_notes", []), report
    )

    return {
        "grounding_report": report.as_dict(),
        "final_answer": answer,
        "citations": citation_list,
        "timings": {"ground_check_ms": _ms(started)},
        "trace": [{
            "step": "ground_check",
            "ok": report.ok,
            "checked": report.checked,
            "grounded": report.grounded,
            "violations": report.violations,
        }],
    }


# ------------------------------------------------------------ clarify/refuse --

async def clarify_node(state: AgentState) -> dict:
    """Return the clarifying question from the plan. No LLM call."""
    started = time.perf_counter()
    plan: Plan | None = state.get("plan")
    question = (plan.clarifying_question if plan and plan.clarifying_question
                else "Could you tell me which country and tax year you mean?")
    answer = f"{question}\n\n(I hold {coverage_sentence()}.)"
    return {
        "final_answer": answer,
        "timings": {"clarify_ms": _ms(started)},
        "trace": [{"step": "clarify", "question": question}],
    }


_REFUSALS = {
    "personalised_advice": (
        "I can't tell you which option is better for you - that's personalised tax "
        "advice, and it depends on figures this system doesn't have and isn't "
        "qualified to weigh. What I can do is show you the relevant rates and "
        "thresholds from the source documents, so ask me for those and decide with "
        "an adviser."
    ),
    "year_not_in_corpus": (
        "I don't have that tax year in my corpus, so I can't answer from a source. "
        "I won't answer it from memory - that is exactly the case where a "
        "confident-sounding wrong figure does the most damage."
    ),
    "country_not_in_corpus": (
        "I don't hold any documents for that country, so there is nothing I can cite."
    ),
    "unresolvable_year": (
        "I couldn't work out which tax year you mean, and I won't guess: the same "
        "shorthand means different years in different countries."
    ),
    "insufficient_evidence": (
        "I found documents for the right country and year, but not the specific "
        "figure you asked for, and I won't fill that gap from memory."
    ),
    "other": "That question is outside what this corpus can answer.",
}


async def refuse_node(state: AgentState) -> dict:
    """Refuse with a specific reason and list what the corpus does cover."""
    started = time.perf_counter()
    plan: Plan | None = state.get("plan")
    reason = (plan.refusal_reason if plan and plan.refusal_reason else None)

    verification = state.get("verification") or {}
    if reason is None and verification.get("insufficient"):
        reason = "insufficient_evidence"
    reason = reason or "other"

    parts = [_REFUSALS.get(reason, _REFUSALS["other"])]
    if plan and plan.refusal_detail:
        parts.append(plan.refusal_detail)
    for gap in verification.get("insufficient", []):
        parts.append(f"Specifically missing for {gap['sub_query']}: {gap['reason']}")

    # Only add the coverage sentence if the detail does not already include it.
    held = coverage_sentence()
    if not any(held in part for part in parts):
        parts.append(f"I hold {held}.")

    return {
        "final_answer": "\n\n".join(parts),
        "timings": {"refuse_ms": _ms(started)},
        "trace": [{"step": "refuse", "reason": reason,
                   "detail": plan.refusal_detail if plan else None}],
    }
