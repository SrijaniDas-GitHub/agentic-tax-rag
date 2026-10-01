"""Streamlit chat UI with a trace panel.

    uv run streamlit run app/main.py

The trace panel shows the plan, per-attempt queries and retrieved chunks,
sufficiency verdicts, grounding report, timings and token usage, all read from
`AgentState`. Citations come from `state["citations"]` (`agents/citations.py`).

The embedding model and indexes are loaded once via `st.cache_resource`, since
the first load takes several seconds. `use_local_hf_cache()` must run before
torch is imported (see core/paths.py).
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # `streamlit run app/main.py` puts app/ on sys.path, not the repo root.
    sys.path.insert(0, str(REPO_ROOT))

from core.paths import use_local_hf_cache

use_local_hf_cache()

import streamlit as st

from agents.carry import carry
from agents.planner import coverage_sentence
from app.render import no_math
from core.llm import TPM_LIMIT

TODAY = date.today()

# One example per intent type, taken from the gold set (eval/gold_set.jsonl).
EXAMPLES = [
    ("lookup", "q01", "What is the 2024 standard deduction for a single filer?",
     ("One country, one year, one figure. Cites `p.96` - the printed page, not the "
      "PDF index.")),
    ("compare_years", "q06",
     "Did the top tax bracket percentage for a single person change between 2023 and 2024?",
     ("**The retry demo.** Both legs fail attempt 1 - the corpus says *Tax Rate "
      "Schedules*, not *tax bracket percentage* - and the self-check rewrites into "
      "the document's own vocabulary. Open the trace.")),
    ("compare_countries", "q07",
     ("How does the US standard deduction for a single filer compare with the UK "
      "Personal Allowance for 2024?"),
     ("Two countries, one year, two hard filters. 203 US chunks against 24 UK ones: "
      "without the per-sub-query country filter the UK leg is crowded out.")),
    ("needs_clarification", "q10",
     "I earn £50,000 a year - which tax band am I in for 2024-25?",
     ("Country and year are unambiguous; the answer is not. At £50,000 the rest "
      "of the UK is in the 40% band and Scotland in the 42%. Asks **one** question.")),
    ("out_of_scope", "q11",
     ("Should I take the standard deduction or itemise my deductions to minimise my "
      "tax bill?"),
     ("The no-advice boundary: a planner *intent* routed to its own node, not a "
      "disclaimer stapled to an answer. Refuses the recommendation, offers the "
      "figures.")),
]


# ------------------------------------------------------------------ warm-up --

@st.cache_resource(show_spinner="Loading the embedding model and both indexes...")
def warm() -> float:
    """Load the embedding model and indexes once per process."""
    from agents.run import warm_up

    return warm_up()


# ------------------------------------------------------------------- panels --

def citation_chips(citations: list[dict]) -> None:
    """Render citations from `state['citations']` as text chips."""
    if not citations:
        return
    chips = []
    for c in citations:
        label = f"[{c['n']}] {c['label']}"
        chips.append(f"[`{label}`]({c['url']})" if c.get("url") else f"`{label}`")
    st.markdown("&nbsp;&nbsp;".join(chips), unsafe_allow_html=True)


def _plan_panel(state: dict) -> None:
    plan = state.get("plan")
    if plan is None:
        st.info("No plan - the turn failed before the planner returned.")
        return
    st.markdown(f"**intent** `{plan.intent}`")
    if plan.clarifying_question:
        st.markdown(f"**clarifying question** — {no_math(plan.clarifying_question)}")
    if plan.refusal_reason:
        st.markdown(f"**refusal reason** `{plan.refusal_reason}` — "
                    f"{no_math(plan.refusal_detail or '')}")
    replan = next((r.get("replan") for r in state.get("trace") or []
                   if r.get("step") == "plan"), None)
    if replan:
        st.markdown(f"**re-planned once** — the first draft asked "
                    f"*\"{no_math(replan['first_clarifying_question'] or '')}\"* "
                    f"of a question that already names the region; re-asked with: "
                    f"*{no_math(replan['note'])}*")

    for sq in plan.sub_queries:
        st.markdown(
            f"- `{sq.country}/{sq.tax_year_key}` ({sq.tax_year_label}) "
            f"&larr; year_text `{sq.year_text}` &nbsp; — &nbsp; {no_math(sq.question)}"
            + (f"\n\n  &nbsp;&nbsp;*{no_math(sq.assumption_note)}*"
               if sq.assumption_note else "")
        )
    st.caption("`tax_year_key` came from `core.year_resolver`, not from the model: "
               "`DraftPlan` has no field for it.")
    st.code(json.dumps(plan.model_dump(), indent=2, default=str), language="json")


def _search_panel(state: dict) -> None:
    """Per-attempt queries, retrieved chunks with scores, verdicts, attempt counts."""
    trace = state.get("trace") or []
    attempts = [r for r in trace if r.get("step") == "search.attempt"]
    rewrites = {(r["sub_query"], r["from"]): r for r in trace
                if r.get("step") == "search.rewrite"}
    if not attempts:
        st.info("No searches were dispatched - the turn clarified or refused first.")
        return

    packs = {f"{p.sub_query.country}/{p.sub_query.tax_year_key}": p
             for p in (state.get("verified_evidence") or state.get("evidence") or [])}

    for label in dict.fromkeys(r["sub_query"] for r in attempts):
        rows = [r for r in attempts if r["sub_query"] == label]
        pack = packs.get(label)
        head = f"**`{label}`** — {len(rows)} attempt(s)"
        if pack is not None:
            head += " — " + ("sufficient" if pack.sufficient else "**INSUFFICIENT**")
        st.markdown(head)

        for row in rows:
            verdict = "sufficient" if row["sufficient"] else "insufficient"
            st.markdown(
                f"&nbsp;&nbsp;**attempt {row['attempt']}** &nbsp; `{row['query']}` "
                f"&nbsp; → &nbsp; **{verdict}** "
                f"&nbsp; <sub>{row['retrieve_ms']:.0f} ms retrieve · "
                f"{row['tokens']} tok · {row.get('model', '?')}</sub>",
                unsafe_allow_html=True,
            )
            st.caption(f"self-check: {no_math(row['reason'])}"
                       + (f"  ·  missing: {no_math(row['missing'])}" if row.get("missing") else "")
                       + (f"  ·  answer in `{row['answer_chunk_id']}`"
                          if row.get("answer_chunk_id") else ""))

            scored = list(zip(row.get("chunk_ids", []), row.get("scores", [])))
            if scored:
                st.dataframe(
                    [{"rank": i, "chunk_id": cid, "rrf_score": score,
                      "answer": "<--" if cid == row.get("answer_chunk_id") else ""}
                     for i, (cid, score) in enumerate(scored, start=1)],
                    hide_index=True, use_container_width=True,
                )
            rewrite = rewrites.get((label, row["query"]))
            if rewrite:
                st.warning(
                    f"**rewrite** `{rewrite['from']}` → `{rewrite['to']}`\n\n"
                    f"because: {rewrite['why']}"
                )
    st.caption(
        "Attempt 1 searches the planner's sub-query verbatim - no pre-retrieval "
        "rewrite, so a genuine vocabulary miss is visible rather than quietly "
        "repaired. Attempt 2 rewrites *with evidence*: it has seen the failed "
        "query, the verdict and the passages that came back instead."
    )


def _grounding_panel(state: dict) -> None:
    report = state.get("grounding_report")
    if not report:
        st.info("Nothing was synthesized, so there was nothing to ground.")
        return

    if report["ok"]:
        st.success(f"PASS — {report['grounded']}/{report['checked']} checkable figures "
                   f"found in the passage cited for **that claim**.")
    else:
        st.error(f"FAIL — {len(report['violations'])} violation(s); "
                 f"{report['grounded']}/{report['checked']} verified. "
                 "The figures are marked in the answer, never silently dropped.")
        for v in report["violations"]:
            st.markdown(f"- `{v.get('subkind') or v['kind']}` — {no_math(v['detail'])}")

    if report["numbers"]:
        st.dataframe(
            [{"claim": n["claim"], "number": n["text"],
              "normalised": n["value"], "%": n["is_percent"],
              "whitelisted": n["reason"] if n["whitelisted"] else "",
              "grounded in": n["matched_chunk_id"] or
                             ("" if n["whitelisted"] else "— NOT FOUND —")}
             for n in report["numbers"]],
            hide_index=True, use_container_width=True,
        )
    st.caption(
        "Checked against the passages **that claim** cites, not the pack as a "
        "whole: `29,200` is the 2024 married-filing-jointly standard deduction "
        "*and* the 2023 filing threshold for a 65-or-older spouse, so a pack-wide "
        "check passes a wrong answer carrying a real citation. Years, tax-year "
        "labels, page references and table numbers are whitelisted; a "
        "percentage is its own token, so a bare `45` cannot ground `45%`."
    )


def _timing_panel(state: dict) -> None:
    timings = state.get("timings") or {}
    rows = sorted(timings.items(), key=lambda kv: (kv[0] == "total_ms", kv[0]))
    st.dataframe([{"stage": k, "ms": round(v, 1)} for k, v in rows],
                 hide_index=True, use_container_width=True)

    trace = state.get("trace") or []
    by_model: dict[str, int] = {}
    calls = 0
    for row in trace:
        llm = row.get("llm")
        if isinstance(llm, dict):
            by_model[llm.get("model", "?")] = by_model.get(llm.get("model", "?"), 0) \
                + llm.get("tokens", 0)
            calls += llm.get("calls", 1)   # a region re-plan is two calls in one row
        elif row.get("tokens"):
            model = row.get("model") or "?"
            by_model[model] = by_model.get(model, 0) + row["tokens"]
            calls += 1

    total = sum(by_model.values())
    st.markdown(f"**{calls} LLM calls, {total} tokens** across "
                f"{len(by_model)} rate-limit bucket(s):")
    for model, tokens in sorted(by_model.items()):
        st.markdown(f"- `{model}` — {tokens} tokens "
                    f"({tokens * 100 // TPM_LIMIT}% of its {TPM_LIMIT}/min free-tier bucket)")
    st.caption(
        f"The free tier meters {TPM_LIMIT} tokens/minute **per model**, which is "
        "why the per-branch sufficiency check runs on the second one. The budget "
        "is met per turn, not per minute: two cold turns inside the same minute "
        "spend the difference in backoff. Repeats replay from the disk cache in "
        "~0.4 s."
    )


def trace_panel(state: dict) -> None:
    """A single expander with one tab per trace section."""
    with st.expander("Trace", expanded=False):
        tabs = st.tabs(["Plan", "Search & retrieval", "Grounding", "Timings", "Raw"])
        with tabs[0]:
            _plan_panel(state)
        with tabs[1]:
            _search_panel(state)
        with tabs[2]:
            _grounding_panel(state)
        with tabs[3]:
            _timing_panel(state)
        with tabs[4]:
            st.code("\n".join(json.dumps(r, default=str) for r in state.get("trace") or []),
                    language="json")


# --------------------------------------------------------------------- turn --

def answer(question: str) -> dict:
    from agents.run import select_runner

    run, _ = select_runner()

    return asyncio.run(run(question, today=TODAY,
                           carried_filters=st.session_state.get("carried")))


# --------------------------------------------------------------------- page --

def main() -> None:
    st.set_page_config(page_title="Tax RAG", page_icon="::", layout="wide")
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("carried", None)
    st.session_state.setdefault("pending", None)

    warmed = warm()

    with st.sidebar:
        st.markdown("### Corpus")
        st.markdown(coverage_sentence())
        st.caption("Four documents, two countries. Nothing outside this is answerable, "
                   "and a year outside it is refused rather than guessed.")

        st.markdown("### Carried filters")
        if st.session_state["carried"]:
            st.json(st.session_state["carried"], expanded=False)
            st.caption("Passed to the next turn's planner, so *“and what about "
                       "2023?”* keeps the country it already resolved.")
        else:
            st.caption("None yet - set after the first answered turn.")

        st.markdown("### Session")
        st.caption(f"Warm-up {warmed:.1f}s, paid once and off every turn's clock.")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["turns"] = []
            st.session_state["carried"] = None
            st.rerun()

    st.title("Agentic RAG over US & UK tax documents")
    st.caption(
        "Every figure is checked against the passage cited for the sentence it "
        "appears in. Ambiguity is asked about, not guessed; a year outside the "
        "corpus is refused, not filled in from memory."
    )

    st.markdown("**Try one of these — one per intent type:**")
    columns = st.columns(len(EXAMPLES))
    for column, (intent, gold_id, question, _) in zip(columns, EXAMPLES):
        with column:
            if st.button(f"{intent}\n\n`{gold_id}`", key=f"ex_{gold_id}",
                         use_container_width=True):
                st.session_state["pending"] = question
    for column, (_, _, _, blurb) in zip(columns, EXAMPLES):
        column.caption(blurb)

    for turn in st.session_state["turns"]:
        with st.chat_message("user"):
            st.markdown(no_math(turn["question"]))
        with st.chat_message("assistant"):
            st.markdown(no_math(turn["answer"]))
            citation_chips(turn["state"].get("citations") or [])
            trace_panel(turn["state"])

    typed = st.chat_input("Ask about US or UK income tax...")
    question = st.session_state.pop("pending", None) or typed
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(no_math(question))
    with st.chat_message("assistant"):
        with st.spinner("Planning, searching, checking..."):
            state = answer(question)
        st.markdown(no_math(state.get("final_answer") or "(no answer produced)"))
        citation_chips(state.get("citations") or [])
        trace_panel(state)

    st.session_state["turns"].append({
        "question": question,
        "answer": state.get("final_answer") or "(no answer produced)",
        "state": state,
    })
    # Only answered turns carry context to the next turn.
    carried = carry(state)
    if carried:
        st.session_state["carried"] = carried
        # Rerun so the sidebar shows the updated carried filters.
        st.rerun()


main()
