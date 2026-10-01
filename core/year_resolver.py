"""Deterministic tax-year resolution from free text, for four conventions.

`tax_year_key` is the calendar year the tax year starts in: US TY2024 -> 2024,
UK 2024-25 -> 2024, AU 2023-24 -> 2023, India FY2024-25 (= AY2025-26) -> 2024.
It is the value retrieval filters on.

US and UK are in the corpus; AU and IN are implemented and tested so adding a
country only needs manifest entries. This is rule-based rather than LLM-based
because models get the UK/AU off-by-one wrong.

`today` is a parameter so relative years are testable.

Grammar accepted (case-insensitive, optional TY/FY/AY/"tax year" prefix):
    2024 | 2024-25 | 2024/25 | 2024 to 2025 | FY24-25 | AY 2025-26
    this year | current tax year | last year | previous year | next year
Bare two-digit years ("FY24") are rejected - see `AmbiguousYearError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

__all__ = [
    "COUNTRIES",
    "AmbiguousYearError",
    "ResolvedYear",
    "YearResolutionError",
    "current_tax_year_start",
    "label_for",
    "normalise_country",
    "resolve",
]


class YearResolutionError(ValueError):
    """The text does not name a tax year this resolver can pin down."""


class AmbiguousYearError(YearResolutionError):
    """The text means different years in different conventions.

    The planner routes this to `clarify` rather than `refuse`.
    """


@dataclass(frozen=True)
class ResolvedYear:
    country: str            # normalised code: US | UK | AU | IN
    tax_year_key: int       # the year the tax year starts in; the filter value
    tax_year_label: str     # how that country writes it: "2024" | "2024-25"
    assumption_note: str | None = None
    source_text: str = ""   # the user's original text, for the trace


# Per-country conventions.
#   start:       (month, day) on which the tax year begins
#   spans:       does the label name two calendar years?
#   bare_year:   how a lone "2024" is read -
#                "start"  the tax year starting in 2024   (US, UK, IN)
#                "end"    the tax year ending   in 2024   (AU)
#   always_note: always attach an assumption_note (the mapping is never 1:1)
COUNTRIES: dict[str, dict] = {
    "US": {"start": (1, 1), "spans": False, "bare_year": "start", "always_note": False,
           "period": "1 January to 31 December"},
    "UK": {"start": (4, 6), "spans": True, "bare_year": "start", "always_note": False,
           "period": "6 April to 5 April"},
    "AU": {"start": (7, 1), "spans": True, "bare_year": "end", "always_note": True,
           "period": "1 July to 30 June"},
    "IN": {"start": (4, 1), "spans": True, "bare_year": "start", "always_note": False,
           "period": "1 April to 31 March"},
}

_ALIASES = {
    "US": "US", "USA": "US", "U.S.": "US", "U.S.A.": "US", "UNITED STATES": "US", "AMERICA": "US",
    "UK": "UK", "GB": "UK", "GBR": "UK", "UNITED KINGDOM": "UK", "BRITAIN": "UK", "ENGLAND": "UK",
    "SCOTLAND": "UK", "WALES": "UK", "NORTHERN IRELAND": "UK",
    "AU": "AU", "AUS": "AU", "AUSTRALIA": "AU",
    "IN": "IN", "IND": "IN", "INDIA": "IN",
}


def normalise_country(country: str) -> str:
    key = _ALIASES.get(str(country).strip().upper())
    if key is None:
        raise YearResolutionError(
            f"unknown country {country!r}; known: {sorted(set(_ALIASES.values()))}"
        )
    return key


# -------------------------------------------------------------------- labels --

def label_for(country: str, start_year: int) -> str:
    """Display label for the tax year starting in `start_year`.

    US "2024"; others "2024-25" (zero-padded, so "1999-00").
    """
    c = normalise_country(country)
    if not COUNTRIES[c]["spans"]:
        return str(start_year)
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def current_tax_year_start(country: str, today: date) -> int:
    """The calendar year in which the tax year containing `today` started.

    E.g. 5 April 2026 is still UK 2025-26; 6 April 2026 is 2026-27.
    """
    c = normalise_country(country)
    month, day = COUNTRIES[c]["start"]
    return today.year if (today.month, today.day) >= (month, day) else today.year - 1


# --------------------------------------------------------------------- parse --

# "2024 to 2025", "2024-25", "2024/2025", "24-25". Guarded against adjacent digits
# so "125,140" is not read as a year span.
_SPAN = re.compile(r"(?<!\d)(\d{4}|\d{2})\s*(?:-|/|–|—|\bto\b)\s*(\d{4}|\d{2})(?!\d)")
_SINGLE4 = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_SINGLE2 = re.compile(r"(?<!\d)(\d{2})(?!\d)")

_PREFIX = re.compile(
    r"^(?:the\s+)?"
    r"(?P<kind>assessment\s+year|financial\s+year|fiscal\s+year|income\s+year|tax\s+year"
    r"|ay|fy|ty|year)?"
    r"\s*[:\-]?\s*",
    re.IGNORECASE,
)

# Offset from the current tax year (in tax years, not calendar years).
_RELATIVE = {
    "this": 0, "current": 0, "present": 0, "now": 0, "today": 0,
    "latest": 0, "most recent": 0,
    "last": -1, "previous": -1, "prior": -1, "preceding": -1, "before": -1,
    "next": +1, "following": +1, "coming": +1, "upcoming": +1,
}

# Words with no year information, so "the current tax year" reduces to "current".
_FILLER = re.compile(r"\b(?:the|tax|financial|fiscal|income|assessment|years?|one)\b")

_ASSESSMENT_KINDS = {"ay", "assessmentyear"}


def _expand(token: str, anchor: int) -> int:
    """'25' -> 2025, taking the century from `anchor`. '2025' passes through."""
    if len(token) == 4:
        return int(token)
    return (anchor // 100) * 100 + int(token)


def _relative_offset(rest: str) -> int | None:
    """Map 'last year' / 'the current tax year' / 'next' to a tax-year offset, or None."""
    probe = rest.strip()
    if probe in _RELATIVE:
        return _RELATIVE[probe]
    stripped = re.sub(r"\s+", " ", _FILLER.sub(" ", probe)).strip()
    return _RELATIVE.get(stripped) if stripped else None


# ------------------------------------------------------------------- resolve --

def resolve(country: str, user_year_text: str, today: date) -> ResolvedYear:
    """'2024' | 'FY24-25' | 'AY 2025-26' | '2024/25' | 'last year' -> ResolvedYear.

    Raises `AmbiguousYearError` when the text means different years in different
    conventions, and `YearResolutionError` when it names no year. The planner
    asks a clarifying question in both cases.
    """
    c = normalise_country(country)
    conv = COUNTRIES[c]
    raw = (user_year_text or "").strip()
    if not raw:
        raise YearResolutionError("no tax year given")

    text = re.sub(r"\s+", " ", raw.lower())
    match = _PREFIX.match(text)
    kind = (match.group("kind") or "").replace(" ", "") if match else ""
    rest = text[match.end():].strip() if match else text
    assessment = kind in _ASSESSMENT_KINDS

    notes: list[str] = []

    # 1. Relative - "last year", "this tax year".
    offset = _relative_offset(rest)
    if offset is not None:
        start = current_tax_year_start(c, today) + offset
        notes.append(
            f"Read {raw!r}, as of {today.isoformat()}, as the {c} tax year "
            f"{label_for(c, start)}."
        )
        return _finish(c, start, conv, notes, raw, assessment=False)

    # 2. A span - "2024-25", "2024 to 2025", "FY24-25", "AY 2025-26".
    span = _SPAN.search(rest)
    if span:
        lo = _expand(span.group(1), today.year)
        hi = _expand(span.group(2), lo)
        if hi != lo + 1:
            raise YearResolutionError(
                f"{raw!r} spans {hi - lo} calendar years; a tax year spans exactly one"
            )
        if not conv["spans"] and not assessment:
            notes.append(
                f"{c} tax years are calendar years ({conv['period']}); "
                f"read {raw!r} as tax year {lo}."
            )
        return _finish(c, lo, conv, notes, raw, assessment=assessment)

    # 3. A single four-digit year.
    single = _SINGLE4.search(rest)
    if single:
        named = int(single.group(1))
        if not 1900 < named < 2100:
            raise YearResolutionError(f"{raw!r}: {named} is not a plausible tax year")
        if conv["bare_year"] == "end":
            # AU: "2024" usually means the income year ending 30 June 2024.
            start = named - 1
        else:
            start = named
            if conv["spans"] and not assessment and c == "IN":
                # India: a bare year is read as the FY starting then, not the AY.
                # Both readings are common, so the note says which was used.
                notes.append(
                    f"Read {raw!r} as the Indian financial year beginning in {named} "
                    f"({label_for(c, start)}, assessment year {label_for(c, start + 1)})."
                )
        return _finish(c, start, conv, notes, raw, assessment=assessment)

    # 4. A bare two-digit year is ambiguous: "FY24" is a different tax year in
    #    AU, IN and the US, so ask instead of guessing.
    if _SINGLE2.search(rest):
        raise AmbiguousYearError(
            f"{raw!r} is a two-digit year, and that means a different tax year in "
            f"each convention (the year ending, in AU and IN; the year starting, in "
            f"the US). Write it in full: '2024' or '2024-25'."
        )

    raise YearResolutionError(f"no tax year found in {raw!r}")


def _finish(
    country: str,
    start_year: int,
    conv: dict,
    notes: list[str],
    raw: str,
    *,
    assessment: bool,
) -> ResolvedYear:
    """Apply the assessment-year shift, then the always-note rule, then build."""
    if assessment:
        if country == "IN":
            # AY 2025-26 is the year in which FY 2024-25 is assessed. The key is the FY.
            start_year -= 1
            notes.append(
                f"India: assessment year {label_for('IN', start_year + 1)} assesses "
                f"financial year {label_for('IN', start_year)}, so tax_year_key is "
                f"the financial year, {start_year}."
            )
        else:
            # "Assessment year" is not a US/UK/AU concept; read it as the tax year
            # and say so in the note.
            notes.append(
                f"'Assessment year' is an Indian convention; read {raw!r} as the "
                f"{country} tax year {label_for(country, start_year)}."
            )

    if conv["always_note"]:
        note = (
            f"{country} income years run {conv['period']}, so "
            f"{label_for(country, start_year)} is the year ending 30 June "
            f"{start_year + 1}."
        )
        if note not in notes:
            notes.append(note)

    return ResolvedYear(
        country=country,
        tax_year_key=start_year,
        tax_year_label=label_for(country, start_year),
        assumption_note=" ".join(notes) if notes else None,
        source_text=raw,
    )
