"""End-to-end evaluation: the 12 gold questions through either orchestrator, no LLM judge.

    uv run python -m eval.run_eval --label baseline
    uv run python -m eval.run_eval --label diversify --diversify
    uv run python -m eval.run_eval --report-only        # re-render from saved runs
    uv run python -m eval.run_eval --followups          # the two-turn set, on its own

Every metric is deterministic: set equality on filters, page membership for
recall, substring match for figures, the numeric guardrail for grounding, and the
trace for behaviour. At n=12 an LLM judge would add variance without adding
information.

1. Pacing. Groq's free tier allows 8000 tokens per minute per model, and a
   three-branch turn uses ~14k across two models. `TokenPacer` waits between
   turns until the next one fits, so rate-limit backoff does not end up inside
   measured latency. The wait is reported separately.
2. Two passes. A cold pass (`LLM_CACHE=0`) measures real latency; a cached
   pass replays it. Both p50s are reported.
3. Recall per attempt. Gold pages are scored against the planner's actual
   sub-queries, for attempt 1 and for the final pack, so the retry loop's
   contribution is measurable. `eval/retrieval_eval.py` scores retrieval alone.
4. Filters as pairs. Filters are compared as (country, tax_year_key) pairs,
   not as text, since the planner's wording varies between runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from agents.carry import carry
from agents.run import orchestrator
from core.llm import TPM_LIMIT
from core.paths import REPO_ROOT
from eval.retrieval_eval import load_gold

RESULTS_DIR = REPO_ROOT / "eval" / "results"
REPORT_PATH = REPO_ROOT / "eval" / "report.md"

# Two-turn questions, kept separate from the gold set. Each row names the gold
# question used as turn 1 (`previous_turn`); turn 2 runs with what
# `agents.carry.carry` takes from turn 1.
FOLLOWUPS_PATH = REPO_ROOT / "eval" / "followups.jsonl"
FOLLOWUPS_LABEL = "followups"

# Results files written before the commit hash was recorded in their config. They
# were measured before the region re-plan and the resolver-backed q12 refusal were
# added, and the report says so.
MEASURED_AT_BEFORE_COMMITS_WERE_RECORDED = {"baseline": "850ae31", "diversify": "850ae31"}

# Fixed date so relative years ("last year") give the same results on any day.
EVAL_TODAY = date(2026, 9, 20)

# Targets used to mark each metric pass/fail in the report.
TARGETS = {
    "filter_precision": 0.9,
    "recall@8": 0.8,
    "answer_contains": 0.8,
    "grounding_violations": 0,
    "behavioural": 4,
    "p50_latency_s": 15.0,
}


# ------------------------------------------------------------- daily quota --
#
# Besides the per-minute limit, the free tier has a 200,000 tokens/day limit that
# does not recover during a run. When it is hit, the run stops. The provider uses
# the same 429 code for both limits, so the check is on the error message.

_DAILY_QUOTA_MARKERS = ("tokens per day", "TPD")


def is_daily_quota_error(message: str) -> bool:
    return any(marker.lower() in message.lower() for marker in _DAILY_QUOTA_MARKERS)


# ------------------------------------------------------------------- pacing --

class TokenPacer:
    """Client-side model of Groq's per-model token bucket, checked between turns.

    The bucket refills continuously (8000 tokens, ~133/s per model), so the wait
    is proportional to the shortfall rather than a fixed minute. Waiting between
    turns keeps rate-limit backoff out of the measured latency.
    """

    # Assumed cost of the next turn before anything is recorded. Raised to the
    # largest turn observed during the run.
    SEED = 3800

    def __init__(self, limit: int = TPM_LIMIT, safety: float = 0.9) -> None:
        self.limit = limit * safety
        self.rate = self.limit / 60.0        # tokens per second, conservative by `safety`
        self.available: dict[str, float] = {}
        self.stamp: dict[str, float] = {}
        self.expected: dict[str, int] = {}
        self.slept = 0.0

    def _refill(self, model: str, now: float) -> float:
        last = self.stamp.get(model, now)
        avail = min(self.limit, self.available.get(model, self.limit) + (now - last) * self.rate)
        self.available[model], self.stamp[model] = avail, now
        return avail

    def record(self, model: str, tokens: int) -> None:
        """Debit the bucket, never below empty (the provider does not carry debt)."""
        if not model or tokens <= 0:
            return
        now = time.monotonic()
        self.available[model] = max(0.0, self._refill(model, now) - tokens)
        # Capped at the limit: waiting cannot help a turn larger than the bucket.
        self.expected[model] = min(self.limit,
                                   max(self.expected.get(model, self.SEED), tokens))

    def delay(self) -> float:
        """Seconds until every bucket the next turn will touch can afford it."""
        now = time.monotonic()
        wait = 0.0
        for model, expect in self.expected.items():
            short = expect - self._refill(model, now)
            if short > 0:
                wait = max(wait, short / self.rate)
        return wait

    async def pace(self) -> float:
        wait = self.delay()
        if wait > 0:
            await asyncio.sleep(wait)
            self.slept += wait
        return wait


# ------------------------------------------------------------------ scoring --

def page_of(chunk_id: str, page: int | None = None) -> str:
    """`us_p17_2024:98:0` -> `us_p17_2024|98`, the `gold_pages` key format."""
    doc_id, _, rest = chunk_id.partition(":")
    if page is None:
        page = int(rest.split(":")[0])
    return f"{doc_id}|{page}"


def observed_behavior(trace: list[dict]) -> str:
    """`answer` | `clarify` | `refuse` | `error`, from the trace's terminal node."""
    steps = {row.get("step") for row in trace}
    if "clarify" in steps:
        return "clarify"
    if "refuse" in steps:
        return "refuse"
    if "ground_check" in steps:
        return "answer"
    return "error"


def behaviour_detail(gold: dict, state: dict) -> tuple[str | None, bool | None]:
    """Stricter behavioural check: was it the right refusal reason or clarifying topic?

    Compares the refusal reason with `refusal_because`, and the clarifying question
    with `clarification_about`. Reported separately from the behavioural metric.
    """
    trace = state.get("trace") or []
    if gold["expected_behavior"] == "refuse":
        want = gold.get("refusal_because")
        row = next((r for r in trace if r.get("step") == "refuse"), None)
        got = row.get("reason") if row else None
        return got, (None if not want else got == want)

    if gold["expected_behavior"] == "clarify":
        row = next((r for r in trace if r.get("step") == "clarify"), None)
        asked = (row.get("question") or "") if row else ""
        topic = gold.get("clarification_about")
        # One keyword per topic used in the gold set.
        keywords = {"tax_year": ("year",), "scotland_or_rest_of_uk": ("scotland",)}
        wanted = keywords.get(topic or "", ())
        ok = any(w in asked.lower() for w in wanted) if wanted else None
        return asked or None, ok
    return None, None


@dataclass
class SubQueryScore:
    label: str
    question: str
    attempts: int
    sufficient: bool
    queries_tried: list[str]
    hit_attempt1: bool | None
    hit_final: bool
    pages_final: list[str]


@dataclass
class QuestionScore:
    id: str
    type: str
    question: str
    expected_behavior: str
    observed_behavior: str
    behaviour_ok: bool
    behaviour_reason: str | None = None
    behaviour_reason_ok: bool | None = None
    expected_filters: list[list] = field(default_factory=list)
    dispatched_filters: list[list] = field(default_factory=list)
    filters_ok: bool | None = None
    sub_queries: list[dict] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    missing_contains: list[str] = field(default_factory=list)
    contains_ok: bool | None = None
    grounding_checked: int = 0
    grounding_grounded: int = 0
    violations: list[dict] = field(default_factory=list)
    total_ms: float = 0.0
    tokens_by_model: dict[str, int] = field(default_factory=dict)
    paced_s: float = 0.0
    final_answer: str = ""
    error: str | None = None
    # Follow-up rows only: turn 1's gold id, its outcome, and what it carried.
    # Everything above scores turn 2.
    previous_turn: str | None = None
    turn1_behavior: str | None = None
    carried: dict | None = None


def tokens_by_model(trace: list[dict]) -> dict[str, int]:
    """Tokens used per model in one turn (each model has its own rate-limit bucket)."""
    out: dict[str, int] = {}
    for row in trace:
        llm = row.get("llm")
        if isinstance(llm, dict):
            out[llm.get("model", "?")] = out.get(llm.get("model", "?"), 0) + llm.get("tokens", 0)
        elif "tokens" in row:
            model = row.get("model") or "?"
            out[model] = out.get(model, 0) + row["tokens"]
    return out


def score_question(gold: dict, state: dict) -> QuestionScore:
    trace = list(state.get("trace") or [])
    plan = state.get("plan")
    observed = observed_behavior(trace)
    reason, reason_ok = behaviour_detail(gold, state)

    score = QuestionScore(
        id=gold["id"],
        type=gold["type"],
        question=gold["question"],
        expected_behavior=gold["expected_behavior"],
        observed_behavior=observed,
        behaviour_ok=observed == gold["expected_behavior"],
        behaviour_reason=reason,
        behaviour_reason_ok=reason_ok,
        total_ms=float((state.get("timings") or {}).get("total_ms", 0.0)),
        tokens_by_model=tokens_by_model(trace),
        final_answer=state.get("final_answer") or "",
    )

    expected_pairs = [[f["country"], f["tax_year_key"]] for f in gold["expected_filters"]]
    dispatched = [[sq.country, sq.tax_year_key] for sq in (plan.sub_queries if plan else [])]
    score.expected_filters = expected_pairs
    score.dispatched_filters = dispatched

    if gold["expected_behavior"] != "answer":
        # Clarify/refuse questions are scored on behaviour only, not filters.
        return score

    score.filters_ok = sorted(map(tuple, dispatched)) == sorted(map(tuple, expected_pairs))

    # --- recall@8, per dispatched sub-query, per attempt --------------------
    gold_pages = set(gold["gold_pages"])
    attempt1: dict[str, list[str]] = {}
    for row in trace:
        if row.get("step") == "search.attempt" and row.get("attempt") == 1:
            attempt1[row["sub_query"]] = [page_of(cid) for cid in row.get("chunk_ids", [])]

    rows: list[SubQueryScore] = []
    for pack in state.get("verified_evidence") or state.get("evidence") or []:
        label = f"{pack.sub_query.country}/{pack.sub_query.tax_year_key}"
        pages = [page_of(p.chunk_id, p.page) for p in pack.passages]
        first = attempt1.get(label)
        rows.append(SubQueryScore(
            label=label,
            question=pack.sub_query.question,
            attempts=pack.attempts,
            sufficient=pack.sufficient,
            queries_tried=list(pack.queries_tried),
            hit_attempt1=(any(p in gold_pages for p in first) if first is not None else None),
            hit_final=any(p in gold_pages for p in pages),
            pages_final=pages,
        ))
    score.sub_queries = [asdict(r) for r in rows]

    # --- answer contains ----------------------------------------------------
    want = list(gold["expected_answer_contains"])
    answer = score.final_answer
    score.expected_contains = want
    score.missing_contains = [w for w in want if w not in answer]
    score.contains_ok = not score.missing_contains

    # --- grounding ----------------------------------------------------------
    report = state.get("grounding_report") or {}
    score.grounding_checked = report.get("checked", 0)
    score.grounding_grounded = report.get("grounded", 0)
    score.violations = report.get("violations", [])
    return score


# ---------------------------------------------------------------- the run --

def load_followups() -> list[dict]:
    if not FOLLOWUPS_PATH.exists():
        return []
    return [json.loads(line) for line in FOLLOWUPS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


async def run_pass(gold: list[dict], *, paced: bool, label: str,
                   previous: dict[str, dict] | None = None) -> list[QuestionScore]:
    """Run every question once and score it.

    With `previous` (gold rows by id), each row is a follow-up: its
    `previous_turn` runs first, `carry()` passes context on, and only turn 2 is
    scored and timed.
    """
    from agents.run import select_runner

    run, _ = select_runner()   # resolved per pass so tests can patch it
    pacer = TokenPacer()
    # Seed both buckets so the gap before question 2 is paced too.
    pacer.expected = {os.getenv("MODEL_NAME", "openai/gpt-oss-120b"): TokenPacer.SEED,
                      os.getenv("FAST_MODEL_NAME", "openai/gpt-oss-20b"): TokenPacer.SEED}

    scores: list[QuestionScore] = []
    for i, question in enumerate(gold, start=1):
        waited = await pacer.pace() if paced else 0.0
        started = time.perf_counter()
        carried: dict | None = None
        turn1: str | None = None
        try:
            if previous is not None:
                first = await run(previous[question["previous_turn"]]["question"],
                                  today=EVAL_TODAY)
                for model, tokens in tokens_by_model(first.get("trace") or []).items():
                    pacer.record(model, tokens)
                turn1 = observed_behavior(first.get("trace") or [])
                carried = carry(first)
                waited += await pacer.pace() if paced else 0.0
                started = time.perf_counter()
            state = await run(question["question"], today=EVAL_TODAY, carried_filters=carried)
            score = score_question(question, state)
        except Exception as exc:  # noqa: BLE001 - a failed turn is a result, not a crashed harness
            score = QuestionScore(
                id=question["id"], type=question["type"], question=question["question"],
                expected_behavior=question["expected_behavior"], observed_behavior="error",
                behaviour_ok=False, total_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        score.paced_s = round(waited, 1)
        if previous is not None:
            score.previous_turn = question["previous_turn"]
            score.turn1_behavior, score.carried = turn1, carried
        for model, tokens in score.tokens_by_model.items():
            pacer.record(model, tokens)
        scores.append(score)

        mark = "ok " if score.behaviour_ok else "BAD"
        print(f"  [{label}] {i:>2}/{len(gold)} {mark} {score.id} "
              f"{score.observed_behavior:<8} {score.total_ms / 1000:6.1f}s "
              f"(waited {waited:.1f}s)  {sum(score.tokens_by_model.values())} tok",
              flush=True)
        # Print the error so it can be diagnosed.
        if score.error:
            print(f"        {score.error}", flush=True)
        # Stop on the daily quota (it will not recover); keep results so far.
        if score.error and is_daily_quota_error(score.error):
            print(f"  [{label}] STOPPING: the 200,000-token DAILY free-tier quota is "
                  f"gone. It refills at ~2.3 tok/s, so it cannot recover inside a "
                  f"run. {len(scores)}/{len(gold)} attempted; the rest are skipped.",
                  flush=True)
            break
    print(f"  [{label}] paced out {pacer.slept:.0f}s in total", flush=True)
    return scores


def aggregate(scores: list[QuestionScore]) -> dict:
    answerable = [s for s in scores if s.expected_behavior == "answer"]
    behavioural = [s for s in scores if s.expected_behavior != "answer"]
    latencies = [s.total_ms / 1000 for s in scores if not s.error]

    # Questions that errored are excluded from the ratios and listed in
    # `not_measured`, so infrastructure failures are not counted as wrong answers.
    attempted = [s for s in scores if not s.error]
    answerable = [s for s in answerable if not s.error]
    behavioural = [s for s in behavioural if not s.error]
    sub_rows = [r for s in answerable for r in s.sub_queries]
    first = [r for r in sub_rows if r["hit_attempt1"] is not None]
    violations = [v for s in answerable for v in s.violations]
    by_kind = {}
    for v in violations:
        key = v.get("subkind") or v.get("kind", "?")
        by_kind[key] = by_kind.get(key, 0) + 1

    return {
        "n_questions": len(scores),
        "n_attempted": len(attempted),
        "not_measured": [s.id for s in scores if s.error],
        "n_answerable": len(answerable),
        "n_sub_queries": len(sub_rows),
        "filter_precision": _ratio([bool(s.filters_ok) for s in answerable]),
        "recall@8": _ratio([r["hit_final"] for r in sub_rows]),
        "recall@8_attempt1": _ratio([bool(r["hit_attempt1"]) for r in first]),
        "answer_contains": _ratio([bool(s.contains_ok) for s in answerable]),
        "grounding_checked": sum(s.grounding_checked for s in answerable),
        "grounding_grounded": sum(s.grounding_grounded for s in answerable),
        "grounding_violations": len(violations),
        "violations_by_kind": by_kind,
        "behavioural_correct": sum(1 for s in behavioural if s.behaviour_ok),
        "behavioural_n": len(behavioural),
        "behavioural_in_gold": sum(1 for s in scores if s.expected_behavior != "answer"),
        "behavioural_reason_correct": sum(1 for s in behavioural if s.behaviour_reason_ok),
        "retries_fired": sum(1 for r in sub_rows if r["attempts"] > 1),
        "retries_recovered": sum(1 for r in sub_rows
                                 if r["attempts"] > 1 and not r["hit_attempt1"] and r["hit_final"]),
        "p50_latency_s": round(statistics.median(latencies), 2) if latencies else 0.0,
        "max_latency_s": round(max(latencies), 2) if latencies else 0.0,
        "tokens_total": sum(sum(s.tokens_by_model.values()) for s in scores),
        "errors": [s.id for s in scores if s.error],
    }


def _ratio(flags: list[bool]) -> float:
    return round(sum(1 for f in flags if f) / len(flags), 3) if flags else 0.0


# ----------------------------------------------------------------- reporting --

def save(label: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def git_commit() -> str:
    """The current commit (with `-dirty` if there are local edits), stored in results."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        head = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def measured_at(label: str, payload: dict) -> str:
    return (payload.get("config", {}).get("commit")
            or MEASURED_AT_BEFORE_COMMITS_WERE_RECORDED.get(label, "not recorded"))


def _load(path: Path) -> dict:
    """Load a saved run and recompute its aggregate from the per-question rows.

    This way a change to a metric definition applies to every saved run.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions") or []
    if rows:
        payload["aggregate"] = aggregate([QuestionScore(**row) for row in rows])
    return payload


def load_results() -> dict[str, dict]:
    """Saved full gold-set runs by label.

    Excludes the follow-up run and `--only` spot checks, which cover different
    question sets and would skew the comparison.
    """
    if not RESULTS_DIR.exists():
        return {}
    runs = {path.stem: _load(path) for path in sorted(RESULTS_DIR.glob("*.json"))
            if path.stem != FOLLOWUPS_LABEL}
    return {label: run for label, run in runs.items()
            if not (run.get("config") or {}).get("only")}


def load_followup_results() -> dict | None:
    path = RESULTS_DIR / f"{FOLLOWUPS_LABEL}.json"
    return _load(path) if path.exists() else None


def _pass(ok: bool) -> str:
    return "**PASS**" if ok else "**FAIL**"


def metric_table(agg: dict, latency: dict, behavioural_in_gold: int = 4) -> str:
    """The metrics table. Each row is PASS, FAIL or *not measured* (no questions ran)."""
    rows = [
        ("Filter precision", "dispatched `(country, tax_year_key)` == `expected_filters`",
         f"{agg['filter_precision']:.3f}", ">= 0.9",
         agg["filter_precision"] >= TARGETS["filter_precision"], agg["n_answerable"] == 0),
        ("Recall@8", ">=1 gold page retrieved, per dispatched sub-query",
         f"{agg['recall@8']:.3f}", ">= 0.8", agg["recall@8"] >= TARGETS["recall@8"],
         agg["n_sub_queries"] == 0),
        ("Answer contains", "all expected figures present in the rendered answer",
         f"{agg['answer_contains']:.3f}", ">= 0.8",
         agg["answer_contains"] >= TARGETS["answer_contains"], agg["n_answerable"] == 0),
        ("Grounding violations", "numbers in the answer absent from the passage cited for them",
         str(agg["grounding_violations"]), "0",
         agg["grounding_violations"] == TARGETS["grounding_violations"],
         agg["grounding_checked"] == 0),
        ("Behavioural accuracy", "correct clarify/refuse on the 4 behavioural questions",
         f"{agg['behavioural_correct']}/{agg['behavioural_n']}", "4/4",
         agg["behavioural_correct"] == TARGETS["behavioural"],
         agg["behavioural_n"] < behavioural_in_gold),
        ("p50 latency (cold)", "wall-clock per question, cache cold",
         f"{latency['cold_p50']:.1f} s" if latency["cold_p50"] else "-", "< 15 s",
         0 < latency["cold_p50"] < TARGETS["p50_latency_s"], not latency["cold_p50"]),
    ]
    out = ["| Metric | Definition | Result | Target | |",
           "|---|---|---|---|---|"]
    for name, definition, value, target, ok, incomplete in rows:
        verdict = "*not measured*" if incomplete else _pass(ok)
        out.append(f"| {name} | {definition} | `{value}` | {target} | {verdict} |")
    return "\n".join(out)


def question_table(scores: list[dict]) -> str:
    out = ["| id | type | behaviour | filters | recall@8 | figures | grounding | cold |",
           "|---|---|---|---|---|---|---|---|"]
    for s in scores:
        behaviour = f"{s['observed_behavior']}"
        if s["expected_behavior"] != s["observed_behavior"]:
            behaviour += f" (want {s['expected_behavior']})"
        mark = "ok" if s["behaviour_ok"] else "**X**"
        subs = s["sub_queries"]
        recall = (f"{sum(1 for r in subs if r['hit_final'])}/{len(subs)}" if subs else "-")
        filters = "-" if s["filters_ok"] is None else ("ok" if s["filters_ok"] else "**X**")
        figures = ("-" if s["contains_ok"] is None
                   else ("ok" if s["contains_ok"] else f"missing {s['missing_contains']}"))
        ground = (f"{s['grounding_grounded']}/{s['grounding_checked']}"
                  if s["grounding_checked"] else ("0/0" if s["expected_behavior"] == "answer"
                                                  else "-"))
        if s["violations"]:
            ground += f" **{len(s['violations'])} violation(s)**"
        out.append(f"| {s['id']} | {s['type']} | {mark} {behaviour} | {filters} | {recall} "
                   f"| {figures} | {ground} | {s['total_ms'] / 1000:.1f}s |")
    return "\n".join(out)


def render_report(results: dict[str, dict], followups: dict | None = None) -> str:
    if not results:
        raise SystemExit("no runs in eval/results - run `python -m eval.run_eval` first")

    primary = results.get("baseline") or next(iter(results.values()))
    agg = primary["aggregate"]
    latency = primary["latency"]

    gold_total = primary["config"].get("gold_total", agg["n_questions"])
    ran = len(primary["questions"])

    doc: list[str] = []
    doc.append(f"# Eval report - {ran} of {gold_total} gold questions, end to end\n")
    doc.append(
        f"Generated {primary['generated']} from `eval/gold_set.jsonl` "
        f"through "
        f"`agents.{primary['config']['orchestrator']}.run`, "
        f"`ORCHESTRATOR={primary['config']['orchestrator']}`, "
        f"`ALLOW_FAST_MODEL={primary['config']['allow_fast_model']}`.\n"
    )
    doc.append(
        "Reproduce:\n\n```\nuv run python -m ingest.build_index\n"
        "uv run python -m eval.run_eval --label baseline\n"
        "uv run python -m eval.run_eval --label diversify --diversify\n"
        "uv run python -m eval.run_eval --followups\n```\n"
    )

    doc.append("### Which code these numbers describe\n")
    doc.append("| run | measured at | generated |\n|---|---|---|")
    for label, payload in sorted(results.items()):
        doc.append(f"| `{label}` | `{measured_at(label, payload)}` | {payload['generated']} |")
    if followups:
        doc.append(f"| `{FOLLOWUPS_LABEL}` | `{measured_at(FOLLOWUPS_LABEL, followups)}` "
                   f"| {followups['generated']} |")
    stale = sorted(label for label in results
                   if label in MEASURED_AT_BEFORE_COMMITS_WERE_RECORDED
                   and not results[label].get("config", {}).get("commit"))
    if stale:
        doc.append(
            "\n" + ", ".join(f"`{label}.json`" for label in stale)
            + " predate commit hashes in results files. **They were measured before "
            "two planner changes** - the region re-plan (q05) and the resolver-backed "
            "`year_not_in_corpus` refusal (q12) - so a difference on q05 or q12 "
            "against a later run is due to the code, not run-to-run variance.\n"
        )
    else:
        doc.append("")

    doc.append("## Metrics\n")
    doc.append(metric_table(
        agg, latency,
        behavioural_in_gold=sum(1 for q in load_gold()
                                if q["expected_behavior"] != "answer")) + "\n")

    skipped = [q["id"] for q in load_gold()
               if q["id"] not in {row["id"] for row in primary["questions"]}]
    if skipped:
        doc.append(
            f"> **{len(skipped)} of the {gold_total} gold questions were not run at all: "
            f"{', '.join('`' + q + '`' for q in skipped)}.** The 200,000-token "
            f"*daily* free-tier quota was exhausted and refills at ~2.3 tokens/"
            f"second, so these were hours away rather than minutes. They are "
            f"excluded from every ratio above rather than counted as failures. "
            f"`uv run python -m eval.run_eval --label baseline` runs the full "
            f"twelve once the quota resets.\n"
        )

    missing = agg.get("not_measured") or []
    if missing:
        doc.append(
            f"> **{len(missing)} of {agg['n_questions']} questions did not run: "
            f"{', '.join('`' + q + '`' for q in missing)}.** Every ratio above is "
            f"over the {agg['n_attempted']} that did. A question that never reached "
            f"the model is not a question the system got wrong, and scoring the two "
            f"together would report an infrastructure problem as a regression. The "
            f"per-question table below names the failure for each.\n"
        )

    doc.append("### No LLM judge, and why\n")
    doc.append(
        "Every metric above is exact-match or set-membership: filter precision "
        "compares resolved `(country, tax_year_key)` pairs, recall checks "
        "`doc_id|page` against the gold pages, *answer contains* is a substring "
        "test, grounding is the deterministic numeric check, and behaviour is read "
        "off which node ended the trace. At n=12 an unvalidated judge would add "
        "variance without adding information, and it would also use the same "
        "per-minute token budget as the system under test. Judge-based "
        "groundedness would be a next step alongside a larger gold set.\n"
    )

    doc.append("## Latency: cold and cached\n")
    if not primary["config"].get("cold", True):
        doc.append(
            "> **This run had no cold pass.** The 200,000-token *daily* free-tier "
            "quota was exhausted, and it refills at ~2.3 tokens/second, so a cold "
            "pass was hours away rather than minutes. Every **correctness** metric "
            "above is unaffected: the cached completions are exactly the ones the "
            "cold calls produced, so the same plan, the same evidence packs and the "
            "same answer are scored. What is missing is the latency row - the "
            "cached figure below is a cache measurement, not a system measurement. "
            "Run `uv run python -m eval.run_eval --label baseline` once the daily "
            "quota resets to fill it in.\n"
        )
    cold_row = (
        f"| cold (`LLM_CACHE=0`) | **{latency['cold_p50']:.1f} s** | "
        f"{latency['cold_max']:.1f} s | every call a miss |\n"
        if latency["cold_p50"] else
        "| cold (`LLM_CACHE=0`) | - | - | **not run** - daily quota exhausted |\n"
    )
    doc.append(
        "| pass | p50 | max | note |\n|---|---|---|---|\n"
        + cold_row
        + f"| cached | **{latency['cached_p50']:.2f} s** | "
        + f"{latency['cached_max']:.2f} s | the same {ran} questions, replayed "
        + "from `data/.llm_cache` |\n"
    )
    doc.append(
        f"**The budget is met per turn, not per minute.** Groq's free tier meters "
        f"{TPM_LIMIT} tokens/minute *per model*, and one three-branch turn costs "
        f"~14k split across two buckets. Two cold turns do not fit in one minute: "
        f"the same query measured 6.9 s on a rested bucket and 22.5 s right after "
        f"another cold turn, the difference being backoff. So this harness "
        f"**paces itself** - `TokenPacer` tracks each model's bucket and waits "
        f"between turns until the next one fits. "
        + (f"It slept **{latency['paced_s']:.0f} s** across the cold pass, and that "
           f"time is outside every measured turn. "
           if primary["config"].get("cold", True)
           else "It did not run on this pass, because a cached replay spends no tokens. ")
        + "\n"
    )

    doc.append("## Per question\n")
    doc.append(question_table(primary["questions"]) + "\n")

    doc.append("## Filter precision is the planner's score\n")
    doc.append(
        f"`eval/retrieval_eval.py` scores retrieval **in isolation**, using the "
        f"sub-query text each gold `expected_filters` entry carries - what a "
        f"competent planner ought to emit. It measures recall@8 **0.833** "
        f"(lookups 4/4). This harness re-scores the same gold pages using the "
        f"sub-queries the planner **actually emitted**, and the two harnesses stay "
        f"separate because the gap between them is the planner's contribution.\n\n"
        f"| harness | sub-query text | recall@8 |\n|---|---|---|\n"
        f"| `retrieval_eval.py` | gold (what a good planner should write) | 0.833 |\n"
        f"| `run_eval.py`, attempt 1 | the planner's own, first try | "
        f"{agg['recall@8_attempt1']:.3f} |\n"
        f"| `run_eval.py`, final pack | the planner's own, after <=2 attempts | "
        f"**{agg['recall@8']:.3f}** |\n\n"
        f"Row 2 minus row 1 is the planner. Row 3 minus row 2 is the retry loop: "
        f"**{agg['retries_fired']}** of {agg['n_sub_queries']} sub-queries went to "
        f"a second attempt and **{agg['retries_recovered']}** of those recovered a "
        f"gold page they had missed. Filters are compared as resolved "
        f"`(country, tax_year_key)` pairs and never as question text - Groq at "
        f"temperature 0 is not reproducible, so the phrasing varies run to run "
        f"while the routed filter does not.\n"
    )

    doc.append("## Grounding violations, by kind\n")
    kinds = agg["violations_by_kind"]
    doc.append(
        f"{agg['grounding_grounded']}/{agg['grounding_checked']} checkable figures "
        f"verified against the passage cited **for that claim**, across the "
        # What was answered, not what the gold set says should be: a question the
        # planner clarified or refused has no figures to check.
        f"{sum(q['observed_behavior'] == 'answer' for q in primary['questions'])} "
        f"answered questions. "
        f"{agg['grounding_violations']} violation(s).\n"
    )
    doc.append(
        "| kind | count | what it means | where the fix goes |\n|---|---|---|---|\n"
        f"| `derived_arithmetic` | {kinds.get('derived_arithmetic', 0)} | the figure "
        "is a sum or difference of figures that *are* cited - the model did "
        "arithmetic | the synthesizer prompt |\n"
        f"| `invented_figure` | {kinds.get('invented_figure', 0)} | nothing in the "
        "cited passages produces it | a bug report |\n"
        f"| `unknown_citation` | {kinds.get('unknown_citation', 0)} | the claim "
        "cites a `chunk_id` that is not in the evidence pack | the synthesizer "
        "prompt |\n"
    )
    doc.append(
        "The split is deterministic, not judged: `guardrails/numeric_grounding.py` "
        "pairs up the figures in the cited passages and reports which two produce "
        "the flagged number. For example, an early answer said the deduction "
        "\"increased by $750\" - correct arithmetic across two tables, but stated in "
        "no passage - and the fix was an instruction in `synthesize_system.md`.\n"
    )
    if any(v["violations"] for v in primary["questions"]):
        doc.append("Violations in full:\n")
        for q in primary["questions"]:
            for v in q["violations"]:
                doc.append(f"- `{q['id']}` {v.get('subkind') or v['kind']}: {v['detail']}")
        doc.append("")

    doc.append("## Behaviour: the four questions that must not be answered\n")
    doc.append("| id | expected | observed | reason / question asked | reason correct |\n"
               "|---|---|---|---|---|")
    for q in primary["questions"]:
        if q["expected_behavior"] == "answer":
            continue
        reason = (q["behaviour_reason"] or "-").replace("\n", " ")
        reason = reason if len(reason) < 90 else reason[:87] + "..."
        ok = {True: "yes", False: "**no**", None: "-"}[q["behaviour_reason_ok"]]
        doc.append(f"| {q['id']} | {q['expected_behavior']} | "
                   f"{'ok' if q['behaviour_ok'] else '**X**'} {q['observed_behavior']} "
                   f"| {reason} | {ok} |")
    doc.append(
        f"\nThe behavioural target is 4/4 on clarify-vs-refuse: "
        f"**{agg['behavioural_correct']}/{agg['behavioural_n']}**. The last column "
        f"is a stricter check reported separately (refusing q12 for the wrong "
        f"reason would still pass the behavioural target). It scores "
        f"{agg['behavioural_reason_correct']}/{agg['behavioural_n']}.\n"
    )

    if len(results) > 1:
        doc.append("## Page diversification, measured end to end\n")
        doc.append(_comparison(results))
    else:
        doc.append("## Page diversification, measured end to end\n")
        doc.append(
            "**Not run.** `diversify_by_page` collapses the top-8 to one chunk per "
            "(doc, page). In the retrieval-only eval it raised recall@8 from 0.833 "
            "to 0.917, but it is off by default because the duplicate it drops is "
            "the **prose** copy of a US table page, which carries qualifiers the "
            "table does not (\"but not more than the regular standard deduction "
            "amount\"). Measuring it end to end needs a second full cold run "
            "(about 70,000 tokens):\n\n"
            "```\nuv run python -m eval.run_eval --label diversify --diversify\n```\n\n"
            "This section is filled in once a second run exists.\n"
        )

    doc.append("## Follow-ups: two turns, the second one scored\n")
    doc.append(_followup_section(followups))

    # Only shown for a cold pass; a cached replay uses no tokens.
    if agg["tokens_total"]:
        doc.append("## Token spend\n")
        doc.append(
            f"{agg['tokens_total']} tokens across the cold pass, metered in two "
            f"buckets of {TPM_LIMIT}/minute. Per turn:\n"
        )
        models = sorted({m for q in primary["questions"] for m in q["tokens_by_model"]})
        doc.append("| id | " + " | ".join(f"`{m}`" for m in models) + " | total |")
        doc.append("|---|" + "---|" * (len(models) + 1))
        for q in primary["questions"]:
            cells = " | ".join(str(q["tokens_by_model"].get(m, 0)) for m in models)
            doc.append(f"| {q['id']} | {cells} | {sum(q['tokens_by_model'].values())} |")
        doc.append("")
    return "\n".join(doc) + "\n"


def _followup_section(followups: dict | None) -> str:
    """Follow-up results: whether the planner uses the context `carry()` passes on."""
    intro = (
        f"`eval/followups.jsonl` ({len(load_followups())} row(s), separate from the "
        f"twelve). Turn 1 is a gold question; turn 2 is a fragment such as *\"and "
        f"what about 2023?\"* run with whatever `agents.carry.carry` took from turn 1 "
        f"- the same function the app uses. Only turn 2 is scored, with the same "
        f"scorer as the gold set. It was added after an early follow-up was "
        f"answered with *\"which country?\"*.\n\n"
    )
    if not followups:
        return intro + (
            "**Not run.** Measuring it needs cold planner calls. Once the daily quota "
            "allows:\n\n```\nuv run python -m eval.run_eval --followups\n```\n"
        )
    agg = followups["aggregate"]
    out = [intro,
           "| id | turn 1 | carried | turn 2 | filters | figures | grounding |",
           "|---|---|---|---|---|---|---|"]
    for q in followups["questions"]:
        carried = q.get("carried") or {}
        carried_text = (", ".join(f"{r['country']} {r['tax_year_label']}"
                                  for r in carried.get("resolved", [])) or "nothing")
        behaviour = q["observed_behavior"]
        if q["expected_behavior"] != behaviour:
            behaviour = f"**{behaviour}** (want {q['expected_behavior']})"
        filters = "-" if q["filters_ok"] is None else ("ok" if q["filters_ok"] else "**X**")
        figures = ("-" if q["contains_ok"] is None
                   else ("ok" if q["contains_ok"] else f"missing {q['missing_contains']}"))
        out.append(f"| {q['id']} | {q.get('previous_turn')} {q.get('turn1_behavior') or '-'} "
                   f"| {carried_text} | {behaviour} | {filters} | {figures} "
                   f"| {q['grounding_grounded']}/{q['grounding_checked']} |")
    out.append(
        f"\nFilter precision {agg['filter_precision']:.3f}, answer contains "
        f"{agg['answer_contains']:.3f}, {agg['grounding_violations']} grounding "
        f"violation(s), over {agg['n_attempted']} of {agg['n_questions']} attempted. "
        f"Measured at `{measured_at(FOLLOWUPS_LABEL, followups)}`, "
        f"{'cold' if followups['config'].get('cold', True) else 'cached replay only'}.\n"
    )
    return "\n".join(out)


def _comparison(results: dict[str, dict]) -> str:
    """Side-by-side comparison of all saved runs."""
    keys = ["filter_precision", "recall@8_attempt1", "recall@8", "answer_contains",
            "grounding_violations", "behavioural_correct", "retries_fired"]
    labels = sorted(results)
    out = [
        ("`diversify_by_page` collapses the top-8 to one chunk per (doc, page). In "
         "the retrieval-only eval it raised recall@8 from 0.833 to 0.917, but it is "
         "off by default because the duplicate it drops is the **prose** copy of a "
         "US table page, which carries qualifiers the table does not (\"but not "
         "more than the regular standard deduction amount\"). The table below "
         "compares the saved runs end to end.\n"),
        "| metric | " + " | ".join(f"`{k}`" for k in labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for key in keys:
        cells = []
        for label in labels:
            value = results[label]["aggregate"].get(key, 0)
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        out.append(f"| {key} | " + " | ".join(cells) + " |")
    for label in labels:
        lat = results[label]["latency"]
        out.append("")
        out.append(f"- `{label}` cold p50 {lat['cold_p50']:.1f} s, "
                   f"cached p50 {lat['cached_p50']:.2f} s, "
                   f"violations by kind: {results[label]['aggregate']['violations_by_kind']}")
    out.append(_like_for_like(results, labels))
    return "\n".join(out) + "\n"


def _like_for_like(results: dict[str, dict], labels: list[str]) -> str:
    """Compare runs over the questions answered in every run.

    `diversify_by_page` only affects retrieval, so questions whose planner
    behaviour differed between runs are listed separately rather than averaged in.
    """
    rows = {label: {q["id"]: q for q in results[label]["questions"]} for label in labels}
    ids = sorted(set.intersection(*(set(r) for r in rows.values())))
    common = [i for i in ids
              if all(rows[label][i]["observed_behavior"] == "answer" for label in labels)]
    differs = [i for i in ids
               if len({rows[label][i]["observed_behavior"] for label in labels}) > 1
               # Compare the reason check, not the free-text question.
               or len({rows[label][i].get("behaviour_reason_ok") for label in labels}) > 1]

    def seen(label: str, i: str) -> str:
        q = rows[label][i]
        reason = q.get("behaviour_reason")
        return q["observed_behavior"] + (f", `{reason}`" if reason else "")

    # Runs from different commits may differ because the planner code changed.
    commits = {measured_at(label, results[label]) for label in labels}
    cause = ("run-to-run variance or the planner code between the commits above"
             if len(commits) > 1 else "run-to-run variance")

    out = ["", "### Like for like: only what the flag can move\n"]
    if differs:
        out.append(
            "Behaviour differed between the runs on "
            + ", ".join(
                f"`{i}` (" + " / ".join(f"{label}: {seen(label, i)}" for label in labels) + ")"
                for i in differs
            )
            + ". That is decided by the planner, before retrieval runs, so it is "
            + cause + " and not the flag, and it is where the "
              "difference in filter precision and answer-contains above comes from.\n"
        )
    out.append(f"Over the {len(common)} questions answered in every run "
               f"({', '.join(f'`{i}`' for i in common)}):\n")
    out.append("| | " + " | ".join(f"`{label}`" for label in labels) + " |")
    out.append("|---|" + "---|" * len(labels))

    def cells(fn) -> str:
        return " | ".join(fn(label) for label in labels)

    def subs(label: str) -> list[dict]:
        return [s for i in common for s in rows[label][i]["sub_queries"]]

    out.append("| recall@8, attempt 1 | " + cells(
        lambda l: f"{sum(s['hit_attempt1'] for s in subs(l))}/{len(subs(l))}") + " |")
    out.append("| recall@8, final | " + cells(
        lambda l: f"{sum(s['hit_final'] for s in subs(l))}/{len(subs(l))}") + " |")
    out.append("| answer contains | " + cells(
        lambda l: f"{sum(bool(rows[l][i]['contains_ok']) for i in common)}/{len(common)}") + " |")
    out.append("| figures grounded | " + cells(
        lambda l: f"{sum(rows[l][i]['grounding_grounded'] for i in common)}/"
                  f"{sum(rows[l][i]['grounding_checked'] for i in common)}") + " |")
    out.append("| retries fired | " + cells(
        lambda l: str(sum(s["attempts"] > 1 for s in subs(l)))) + " |")
    out.append("| duplicate-page slots in final packs | " + cells(
        lambda l: f"{sum(len(s['pages_final']) - len(set(s['pages_final'])) for s in subs(l))}"
                  f"/{sum(len(s['pages_final']) for s in subs(l))}") + " |")
    out.append("")
    out.append(
        "**Decision: `DIVERSIFY_BY_PAGE` stays off.** The bar for flipping it was "
        "recall or answer-contains improving, with 0 grounding violations and no "
        "answer losing a qualifier it had. Like for like, nothing improved: the "
        "same sub-queries hit on attempt 1 and after retry, and every answer that "
        "contained its figures still does. In the original pair of runs, q01's "
        "final answer was identical and q07's differed only in the year-assumption "
        "note, so no qualifier was lost - but none of the 12 gold questions depends "
        "on a prose-only qualifier, so the eval cannot measure that risk either. "
        "The flag frees duplicate slots for other pages without moving any metric "
        "here, so the default stays unchanged."
    )
    return "\n".join(out)


# --------------------------------------------------------------------- main --

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="End-to-end eval over the gold set. No LLM judge.")
    ap.add_argument("--label", default="baseline", help="Name this run in eval/results/.")
    ap.add_argument("--diversify", action="store_true",
                    help="Collapse the top-8 to one chunk per page.")
    ap.add_argument("--no-pace", action="store_true",
                    help="Skip the token pacer. The p50 then measures the rate limit.")
    ap.add_argument("--only", help="Comma-separated gold ids, for iterating on one question.")
    ap.add_argument("--no-cold", action="store_true",
                    help="Skip the cold pass and score the cached one. Use when the "
                         "DAILY quota is gone: correctness is still measured, latency "
                         "is cache-replay only and the report says so.")
    ap.add_argument("--report-only", action="store_true",
                    help="Re-render eval/report.md from eval/results/ without calling the LLM.")
    ap.add_argument("--followups", action="store_true",
                    help="Run eval/followups.jsonl (two turns each, turn 2 scored) instead "
                         f"of the gold set. Saved as eval/results/{FOLLOWUPS_LABEL}.json.")
    args = ap.parse_args(argv)

    # Prevent `--only` from overwriting the default `baseline` results.
    # `--followups` always uses its own label.
    if (args.only and args.label == ap.get_default("label")
            and not (args.report_only or args.followups)):
        ap.error("--only runs are spot checks, not a full run: name them with --label "
                 "(e.g. --label q07_fix) so they don't overwrite "
                 f"eval/results/{args.label}.json.")

    if args.report_only:
        REPORT_PATH.write_text(render_report(load_results(), load_followup_results()),
                               encoding="utf-8")
        print(f"wrote {REPORT_PATH}")
        return 0

    # `core.retrieval` reads DIVERSIFY_BY_PAGE at import time (already done), so
    # set the module attribute as well as the env var.
    if args.diversify:
        from core import retrieval

        os.environ["DIVERSIFY_BY_PAGE"] = "1"
        retrieval.DIVERSIFY_BY_PAGE = True

    from agents.run import warm_up
    warmed = warm_up()
    print(f"[warm-up {warmed:.1f}s: embedding model + indexes, off the clock]")

    # Follow-up rows reference a gold question by id for turn 1.
    previous: dict[str, dict] | None = None
    if args.followups:
        previous = {q["id"]: q for q in load_gold()}
        gold = load_followups()
        args.label = FOLLOWUPS_LABEL
    else:
        gold = load_gold()
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        gold = [q for q in gold if q["id"] in wanted]

    from core.llm import reset_client

    cold: list[QuestionScore] = []
    paced_s = 0.0
    if not args.no_cold:
        # Pass 1: cold. Reads disabled, writes on, so pass 2 replays what this filled.
        os.environ["LLM_CACHE"] = "0"
        reset_client()
        print(f"\ncold pass ({len(gold)} questions, "
              f"{'paced' if not args.no_pace else 'UNPACED'}):")
        cold = asyncio.run(run_pass(gold, paced=not args.no_pace, label="cold",
                                    previous=previous))
        paced_s = sum(s.paced_s for s in cold)

    # Pass 2: cached. Same questions, no tokens spent, so no pacing needed.
    os.environ["LLM_CACHE"] = "1"
    reset_client()
    print("\ncached pass (same questions, replayed):")
    cached = asyncio.run(run_pass(gold, paced=False, label="cached", previous=previous))

    # With --no-cold, the cached pass is scored. Correctness metrics are the same;
    # only latency differs, and the report notes it.
    scored = cold if cold else cached
    cold_times = [s.total_ms / 1000 for s in cold if not s.error]
    cached_times = [s.total_ms / 1000 for s in cached if not s.error]
    payload = {
        "label": args.label,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "config": {
            "orchestrator": orchestrator(),
            "model": os.getenv("MODEL_NAME", "openai/gpt-oss-120b"),
            "fast_model": os.getenv("FAST_MODEL_NAME", "openai/gpt-oss-20b"),
            "allow_fast_model": os.getenv("ALLOW_FAST_MODEL", "1"),
            "diversify_by_page": bool(args.diversify),
            "paced": not args.no_pace,
            "cold": bool(cold),
            "only": args.only or "",
            "gold_total": len(load_followups() if args.followups else load_gold()),
            "today": EVAL_TODAY.isoformat(),
            "commit": git_commit(),
            "followups": bool(args.followups),
        },
        "aggregate": aggregate(scored),
        "latency": {
            "cold_p50": round(statistics.median(cold_times), 2) if cold_times else 0.0,
            "cold_max": round(max(cold_times), 2) if cold_times else 0.0,
            "cached_p50": round(statistics.median(cached_times), 2) if cached_times else 0.0,
            "cached_max": round(max(cached_times), 2) if cached_times else 0.0,
            "paced_s": round(paced_s, 1),
        },
        "questions": [asdict(s) for s in scored],
    }
    path = save(args.label, payload)
    print(f"\nwrote {path}")
    if args.only:
        print(f"  (an --only run: kept in {path.name}, left out of the report's comparison)")

    REPORT_PATH.write_text(render_report(load_results(), load_followup_results()),
                           encoding="utf-8")
    print(f"wrote {REPORT_PATH}")

    agg = payload["aggregate"]
    print(f"\nfilter precision {agg['filter_precision']:.3f} | "
          f"recall@8 {agg['recall@8']:.3f} (attempt1 {agg['recall@8_attempt1']:.3f}) | "
          f"contains {agg['answer_contains']:.3f} | "
          f"violations {agg['grounding_violations']} {agg['violations_by_kind']} | "
          f"behaviour {agg['behavioural_correct']}/{agg['behavioural_n']} | "
          f"p50 cold {payload['latency']['cold_p50']}s cached "
          f"{payload['latency']['cached_p50']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
