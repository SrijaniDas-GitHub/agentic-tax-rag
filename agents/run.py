"""CLI: `uv run python -m agents.run "How did the US standard deduction change ..."`.

The orchestrator is selected with the ORCHESTRATOR environment variable:

    ORCHESTRATOR=runner   (default)  hand-written async executor, agents/runner.py
    ORCHESTRATOR=graph               LangGraph StateGraph, agents/graph.py

The embedding model and indexes are loaded before timing starts, since the
first load takes a few seconds. Pass `--cold` to include it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date

from agents.state import AgentState
from core.llm import TPM_LIMIT


def use_utf8_stdout() -> None:
    """Force UTF-8 output; Windows consoles default to cp1252, which cannot print `£`."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def warm_up() -> float:
    """Load the embedding model and open both indexes. Returns seconds spent."""
    started = time.perf_counter()
    from core import embedding, retrieval

    embedding.model()
    retrieval.collection()
    retrieval.bm25_index()
    retrieval.chunks_by_id()
    return time.perf_counter() - started


def orchestrator() -> str:
    """`runner` or `graph`; any other value is an error."""
    choice = os.getenv("ORCHESTRATOR", "runner").strip().lower() or "runner"
    if choice not in {"runner", "graph"}:
        raise SystemExit(f"ORCHESTRATOR={choice!r}: expected 'runner' or 'graph'.")
    return choice


def select_runner():
    """Return the selected `run` function and its name.

    Imported at call time so tests can patch `agents.runner.run` / `agents.graph.run`.
    """
    if orchestrator() == "graph":
        from agents.graph import run as graph_run
        return graph_run, "graph"
    from agents.runner import run as runner_run
    return runner_run, "runner"


def print_report(state: AgentState, *, show_trace: bool) -> None:
    print("\n" + "=" * 72)
    print(state.get("final_answer") or "(no answer produced)")
    print("=" * 72)

    plan = state.get("plan")
    if plan is not None:
        print(f"\nintent: {plan.intent}")
        for sq in plan.sub_queries:
            note = f"  [{sq.assumption_note}]" if sq.assumption_note else ""
            print(f"  - {sq.country}/{sq.tax_year_key} ({sq.tax_year_label}) "
                  f"<- {sq.year_text!r}: {sq.question}{note}")

    for pack in state.get("verified_evidence") or state.get("evidence") or []:
        mark = "ok " if pack.sufficient else "NO "
        print(f"\n{mark}{pack.sub_query.country}/{pack.sub_query.tax_year_key} "
              f"attempts={pack.attempts} passages={len(pack.passages)}")
        for i, query in enumerate(pack.queries_tried, start=1):
            print(f"     attempt {i}: {query}")

    report = state.get("grounding_report")
    if report:
        print(f"\ngrounding: {'PASS' if report['ok'] else 'FAIL'} "
              f"({report['grounded']}/{report['checked']} numbers verified)")
        for violation in report["violations"]:
            print(f"     ! {violation['detail']}")

    timings = state.get("timings") or {}
    print("\ntimings (ms):")
    for key in sorted(timings, key=lambda k: (k == "total_ms", k)):
        print(f"     {key:<28} {timings[key]:>8.1f}")

    # Token usage against the free-tier per-minute limit (core.llm.TPM_LIMIT).
    rows = state.get("trace") or []
    spent = sum(row.get("tokens", 0) for row in rows)
    spent += sum(row["llm"].get("tokens", 0) for row in rows if isinstance(row.get("llm"), dict))
    calls = sum(1 for row in rows if "tokens" in row or isinstance(row.get("llm"), dict))
    print(f"\nllm: {calls} calls, {spent} tokens "
          f"({spent * 100 // TPM_LIMIT}% of the {TPM_LIMIT}/min free-tier budget)")

    if show_trace:
        print("\ntrace:")
        for row in state.get("trace") or []:
            print("   " + json.dumps(row, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the tax bot one question.")
    parser.add_argument("question")
    parser.add_argument("--today", help="ISO date injected into the year resolver "
                                        "(default: the real today).")
    parser.add_argument("--trace", action="store_true", help="Print every trace row.")
    parser.add_argument("--json", action="store_true", help="Dump the whole state as JSON.")
    parser.add_argument("--cold", action="store_true",
                        help="Skip the warm-up, so the first query pays the model load.")
    args = parser.parse_args(argv)

    use_utf8_stdout()
    run, which = select_runner()

    warm_s = 0.0
    if not args.cold:
        warm_s = warm_up()
        print(f"[warm-up {warm_s:.2f}s: embedding model + indexes loaded, off the clock]",
              file=sys.stderr)
    print(f"[orchestrator: {which}]", file=sys.stderr)

    today = date.fromisoformat(args.today) if args.today else date.today()
    state = asyncio.run(run(args.question, today=today))

    if args.json:
        print(json.dumps(state, default=str, indent=2))
    else:
        print_report(state, show_trace=args.trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
