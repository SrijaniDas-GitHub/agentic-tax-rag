"""Planner support: prompt context, and deterministic resolution of the LLM's draft plan.

`plan_node` makes one LLM call (two if `region_stated_note` applies) that returns
a `DraftPlan` with `country` and the raw `year_text`. Tax years are then resolved
here by `core.year_resolver`, not by the LLM:

    DraftPlan --> resolve_plan(draft, today=..., known=..., question=...) --> Plan
"""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache

from agents.contracts import DraftPlan, Plan, SubQuery

# Reuse the resolver's country aliases and year patterns so they cannot drift.
from core.year_resolver import _ALIASES as _COUNTRY_ALIASES
from core.year_resolver import _SINGLE4 as _YEAR4
from core.year_resolver import _SPAN as _YEAR_SPAN
from core.year_resolver import AmbiguousYearError, ResolvedYear, YearResolutionError, resolve

# Phrases that mark an out_of_scope refusal as a personal-advice refusal. The
# planner's reason is free text, so it is mapped to the enum here.
_ADVICE_WORDS = (
    "advice", "advise", "recommend", "should i", "personalis", "personaliz",
    "best for", "better for", "minimis", "minimiz", "optimis", "optimiz",
    "which is better", "which option",
)


# ------------------------------------------------------------------ coverage --

@lru_cache(maxsize=1)
def coverage() -> dict[str, dict[int, str]]:
    """{country: {tax_year_key: tax_year_label}}, read from the manifest."""
    from ingest.manifest import load_manifest

    out: dict[str, dict[int, str]] = {}
    for doc in load_manifest():
        out.setdefault(doc.country, {})[doc.tax_year_key] = doc.tax_year_label
    return {c: dict(sorted(years.items())) for c, years in sorted(out.items())}


def coverage_prompt() -> str:
    """Corpus coverage for the planner prompt, one line per country."""
    lines = []
    for country, years in coverage().items():
        labels = ", ".join(f"{label} (key {key})" for key, label in years.items())
        lines.append(f"- **{country}** - tax years: {labels}")
    return "\n".join(lines)


def coverage_sentence() -> str:
    """Corpus coverage as one sentence, for refusal messages."""
    parts = [
        f"{country} {', '.join(years.values())}"
        for country, years in coverage().items()
    ]
    return "; ".join(parts)


# ------------------------------------------------------------ carried context --

NO_PREVIOUS_TURN = "(No previous turn.)"


def carried_context(carried: dict | None) -> str:
    """Describe the previous turn for the planner's user message.

    Follow-up instructions live here rather than in the system prompt so that
    first-turn prompts stay unchanged. They explicitly override the
    "no country -> clarify" rule for inherited context.
    """
    if not carried:
        return NO_PREVIOUS_TURN
    if carried.get("clarifying_question"):
        return _awaiting_answer(carried)
    resolved = ", ".join(
        f"{r['country']} {r['tax_year_label']}" for r in carried.get("resolved", [])
    ) or "nothing"
    previous = carried.get("previous_question") or "(not recorded)"
    return (
        "## Previous turn - context only, do not plan it again\n\n"
        f"Previous question: {previous}\n"
        f"It resolved to: {resolved}\n\n"
        "If the new question is a follow-up fragment (\"and what about 2023?\", "
        "\"and for the UK?\"), it inherits the previous question's subject and "
        "everything the fragment does not restate: a new year keeps the country, a "
        "new country keeps the year. Write each sub-query as a complete, "
        "self-contained question on that subject. For an inherited year, use the "
        "tax year label above as `year_text`. A subject, country or year carried "
        "this way counts as stated: do not ask for it. If the new question stands "
        "on its own, ignore this block."
    )


def _awaiting_answer(carried: dict) -> str:
    """The previous turn asked the user a clarifying question; this may be the reply."""
    previous = carried.get("previous_question") or "(not recorded)"
    return (
        "## Previous turn - you asked the user a clarifying question\n\n"
        f"Previous question: {previous}\n"
        f"You asked: {carried['clarifying_question']}\n\n"
        "If the new message answers that (\"2023\", \"Scotland\", \"the US\"), plan the "
        "previous question with the answer filled in: write complete, self-contained "
        "sub-queries for the previous question, and take `year_text` and country from "
        "the previous question and the answer together. What the previous question "
        "and the answer state counts as stated: do not ask for it again. If the new "
        "message is a new question, ignore this block."
    )


# -------------------------------------------------------------- region stated --
#
# The planner sometimes asks which UK region the user means even when the question
# already names it. When that happens, the planner is called again with a note
# saying the region was stated. The note goes in the user message, so calls that
# do not need it are unchanged.

_REGIONS = ("Northern Ireland", "rest of the UK", "Scotland", "England", "Wales")
_REGION_RE = re.compile(r"\b(" + "|".join(re.escape(r) for r in _REGIONS) + r")\b",
                        re.IGNORECASE)
# A clarifying question about the region names one, or asks about "the region" /
# "which part of the UK" without naming any.
_ASKS_REGION_RE = re.compile(_REGION_RE.pattern + r"|\bregion\b|\bpart of the UK\b",
                             re.IGNORECASE)

REGION_STATED_NOTE = (
    "Note: the question already names the UK region ({region}). That region counts "
    "as stated: do not ask about it. Plan the question as asked."
)


def named_region(text: str | None) -> str | None:
    """The first UK region the text names, in its canonical spelling, or None."""
    match = _REGION_RE.search(text or "")
    if not match:
        return None
    return next(r for r in _REGIONS if r.lower() == match.group(1).lower())


def region_stated_note(draft: DraftPlan, question: str) -> str | None:
    """The note to re-plan with, or None.

    Applies only when the draft asks for clarification, the question names a UK
    region, and the clarifying question is about the region.
    """
    if draft.intent != "needs_clarification":
        return None
    region = named_region(question)
    if region is None or not _ASKS_REGION_RE.search(draft.clarifying_question or ""):
        return None
    return REGION_STATED_NOTE.format(region=region)


# ----------------------------------------------------------------- resolution --

def _clarify(question: str, note: str) -> Plan:
    return Plan(
        intent="needs_clarification",
        clarifying_question=question,
        assumption_notes=[note] if note else [],
    )


def _refuse(reason: str, detail: str | None) -> Plan:
    return Plan(intent="out_of_scope", refusal_reason=reason, refusal_detail=detail)


def _refusal_reason_for(text: str | None) -> str:
    lowered = (text or "").lower()
    return "personalised_advice" if any(w in lowered for w in _ADVICE_WORDS) else "other"


def _factual_questions(draft: DraftPlan) -> str | None:
    """The advice refusal's detail: the factual sub-queries the planner drafted, as
    questions the user can ask instead, or None if it drafted none.

    Replaces the planner's own reason, which only restates the refusal
    ("I cannot provide personalized tax advice.").
    """
    questions = list(dict.fromkeys(sq.question.strip() for sq in draft.sub_queries
                                   if sq.question.strip()))
    if not questions:
        return None
    return ("For example, you could ask one of these, with a tax year from the list "
            "below:\n" + "\n".join(f"- {q}" for q in questions))


def _not_held(resolved: ResolvedYear) -> str:
    return (f"I do not hold {resolved.country} {resolved.tax_year_label}. "
            f"I have {coverage_sentence()}.")


def _names_country(question: str, country: str) -> bool:
    for alias, target in _COUNTRY_ALIASES.items():
        if target != country:
            continue
        # Short codes are case-sensitive ("tell us" is not the US, "in" is not India).
        flags = 0 if len(alias.rstrip(".")) <= 3 else re.IGNORECASE
        if re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", question, flags):
            return True
    return False


def _year_gap_in(question: str, *, today: date,
                 known: dict[str, dict[int, str]]) -> str | None:
    """`year_not_in_corpus` detail for a planner refusal with no sub-queries, or None.

    Resolves the year from the user's question for each country it names (or all
    held countries if none). Returns a detail only if the year is missing for all.
    """
    named = [c for c in known if _names_country(question, c)]
    gaps: list[ResolvedYear] = []
    for country in named or list(known):
        try:
            resolved = resolve(country, question, today)
        except YearResolutionError:
            # No year, or an ambiguous one (AmbiguousYearError is a subclass).
            return None
        if resolved.tax_year_key in known.get(resolved.country, {}):
            return None
        gaps.append(resolved)
    return _not_held(gaps[0]) if gaps else None


def _years_named(question: str) -> set[int]:
    """The distinct four-digit years in the question, e.g. {2024} for "2024-25"."""
    return {int(y) for y in _YEAR4.findall(question) if 1900 < int(y) < 2100}


def _the_one_year_in(question: str | None) -> str | None:
    """The question's year as written, if it names exactly one; else None.

    Used for legs the planner left without a year, e.g. "compare the US standard
    deduction with the UK Personal Allowance for 2024". Distinct four-digit years
    are counted directly, because the resolver would read "from 2023 to 2024" as
    a single span.
    """
    if not question:
        return None
    years = _years_named(question)
    if len(years) != 1:
        return None
    (year,) = years
    # Return a span like "2024-25" as written so each country reads it in its
    # own convention.
    for span in _YEAR_SPAN.finditer(question):
        if span.group(1) == str(year):
            return span.group(0)
    return str(year)


def _no_year_detail(country: str, question: str | None) -> str:
    """The refusal detail for a leg with no year. Never quotes the empty string."""
    if question and len(_years_named(question)) > 1:
        return (f"No tax year was given for the {country} part of the question, and "
                f"the question names more than one, so I can't tell which it means.")
    return f"No tax year was given for the {country} part of the question."


def resolve_plan(draft: DraftPlan, *, today: date,
                 known: dict[str, dict[int, str]] | None = None,
                 question: str | None = None) -> Plan:
    """Resolve a `DraftPlan` into a `Plan`. No LLM calls.

    `question` (the user's text) is used in two cases: to detect a missing year
    when the planner refused with no sub-queries, and to fill in a leg the
    planner left without a year.

    Outcomes:

    * `AmbiguousYearError` -> clarify (e.g. "FY24").
    * `YearResolutionError` -> refuse; no year can be read.
    * country/year not in the corpus -> refuse, listing what is held.
    * otherwise -> the plan, with each sub-query's assumption note attached.
    """
    known = known if known is not None else coverage()

    if draft.intent == "needs_clarification":
        return Plan(
            intent="needs_clarification",
            clarifying_question=draft.clarifying_question
            or "Could you tell me which country and tax year you mean?",
        )

    if draft.intent == "out_of_scope":
        reason = _refusal_reason_for(draft.out_of_scope_reason)
        if reason == "personalised_advice":
            return _refuse(reason, _factual_questions(draft))
        if reason == "other" and not draft.sub_queries and question:
            gap = _year_gap_in(question, today=today, known=known)
            if gap:
                return _refuse("year_not_in_corpus", gap)
        return _refuse(
            reason,
            draft.out_of_scope_reason or "That question is outside what this corpus can answer.",
        )

    sub_queries: list[SubQuery] = []
    notes: list[str] = []
    seen: set[tuple[str, int]] = set()

    for planned in draft.sub_queries:
        # A leg without a year takes the question's year if it names exactly one.
        blank = not (planned.year_text or "").strip()
        lent = _the_one_year_in(question) if blank else None
        year_text = lent or planned.year_text
        try:
            resolved: ResolvedYear = resolve(planned.country, year_text, today)
        except AmbiguousYearError as exc:
            # Several possible years: ask rather than pick one.
            return _clarify(
                f"Which tax year do you mean by {year_text!r}? "
                f"I hold {coverage_sentence()}.",
                str(exc),
            )
        except YearResolutionError as exc:
            # No year can be read from the text.
            if blank:
                return _refuse("unresolvable_year", _no_year_detail(planned.country, question))
            return _refuse("unresolvable_year",
                           f"I could not read a tax year from {year_text!r}: {exc}")

        if resolved.country not in known:
            return _refuse(
                "country_not_in_corpus",
                f"I do not hold any {resolved.country} documents. I have {coverage_sentence()}.",
            )
        if resolved.tax_year_key not in known[resolved.country]:
            return _refuse("year_not_in_corpus", _not_held(resolved))

        key = (resolved.country, resolved.tax_year_key)
        if key in seen:
            # Drop duplicate legs for the same (country, year).
            continue
        seen.add(key)

        # If the year came from the question, the code writes the assumption note.
        said = (f"No year was given for the {resolved.country} part, so I used the one "
                f"in your question, {resolved.tax_year_label}." if lent
                else planned.year_assumption)
        note = " ".join(n for n in (said, resolved.assumption_note) if n) or None
        if note:
            notes.append(f"{resolved.country} {resolved.tax_year_label}: {note}")

        sub_queries.append(
            SubQuery(
                question=planned.question,
                country=resolved.country,
                tax_year_key=resolved.tax_year_key,
                tax_year_label=resolved.tax_year_label,
                year_text=planned.year_text,
                assumption_note=note,
            )
        )

    return Plan(intent=draft.intent, sub_queries=sub_queries, assumption_notes=notes)
