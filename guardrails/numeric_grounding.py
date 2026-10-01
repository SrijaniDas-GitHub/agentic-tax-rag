"""Numeric grounding: every number in an answer must appear in a passage its claim cites.

Numbers are checked only against the passages each claim cites, not the whole
evidence pack, since the same figure (e.g. `29,200`) can mean different things
in different passages.

* `$ £ € ,` are stripped and values compared as `Decimal`, so `$14,600`,
  `14,600` and `14600` match.
* Percentages only match percentages.
* Four-digit years, tax-year labels (`2024-25`, `2024 to 2025`) and page/section
  references (`p.96`, `§5`) are whitelisted.
* The report lists every number, whether it matched, and which passage matched it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation

from agents.contracts import AnswerClaim, Passage

__all__ = [
    "GroundingReport",
    "NumberToken",
    "check_answer",
    "extract_numbers",
    "normalise",
    "passage_values",
]

# Optional currency symbol, digits with optional thousands separators and decimals,
# and an optional percent suffix.
_FIGURE = re.compile(r"(?P<sym>[$£€])?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
                     r"(?P<pct>\s?%|\s?percent\b)?", re.IGNORECASE)

# Spans that are never a tax figure. Matched first, so "2024-25" is not split
# into 2024 and 25.
_WHITELIST_SPANS = [
    # tax-year labels: 2024-25, 2024/25, 2024 to 2025, 2024-2025
    re.compile(r"(?<!\d)(19|20)\d{2}\s*(?:-|/|–|—|\bto\b)\s*\d{2,4}(?!\d)", re.IGNORECASE),
    # page and section references, in both countries' forms
    re.compile(r"(?:p\.|pp\.|page|§|section)\s?\d+", re.IGNORECASE),
    # citation markers the renderer inserts
    re.compile(r"\[\d+\]"),
    # chapter / table / schedule numbering ("Table 10-1", "chapter 2")
    re.compile(r"(?:table|chapter|schedule|publication|pub\.?)\s?\d+(?:-\d+)?", re.IGNORECASE),
]

# A bare four-digit year is whitelisted only with no currency symbol, separator
# or percent. "$2,024" and "2024%" are figures.
_YEAR_RANGE = range(1900, 2101)


@dataclass(frozen=True)
class NumberToken:
    """A number found in an answer."""

    text: str                 # exactly as written in the answer
    value: Decimal | None  # normalised; None when unparseable
    is_percent: bool
    whitelisted: bool
    reason: str = ""          # why it was whitelisted, for the report


@dataclass
class GroundingReport:
    ok: bool
    checked: int
    grounded: int
    violations: list[dict]
    numbers: list[dict]

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------- normalisation --

def normalise(raw: str) -> Decimal | None:
    """`$14,600` / `14,600` / `14600` / `14600.00` -> the same `Decimal`."""
    cleaned = re.sub(r"[$£€,\s]", "", raw)
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _whitelisted_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for pattern in _WHITELIST_SPANS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), "year-label/reference"))
    return spans


def extract_numbers(text: str) -> list[NumberToken]:
    """Every number in `text`, marked as whitelisted or checkable."""
    spans = _whitelisted_spans(text)
    tokens: list[NumberToken] = []

    for match in _FIGURE.finditer(text):
        start, end = match.start(), match.end()
        covering = next((r for r in spans if r[0] <= start and end <= r[1]), None)
        if covering is None:
            # Partial overlap (e.g. the "2024" of "2024-25") also counts.
            covering = next((r for r in spans if start < r[1] and end > r[0]), None)

        raw_num = match.group("num")
        is_percent = bool(match.group("pct"))
        value = normalise(raw_num)
        written = text[start:end].strip()

        if covering is not None:
            tokens.append(NumberToken(written, value, is_percent, True, covering[2]))
            continue

        bare_year = (
            not is_percent
            and match.group("sym") is None
            and "," not in raw_num
            and "." not in raw_num
            and value is not None
            and int(value) in _YEAR_RANGE
            and len(raw_num) == 4
        )
        if bare_year:
            tokens.append(NumberToken(written, value, False, True, "four-digit year"))
            continue

        tokens.append(NumberToken(written, value, is_percent, False))

    return tokens


def passage_values(text: str) -> tuple[set[Decimal], set[Decimal]]:
    """Every figure in a passage, as (plain values, percent values)."""
    plain: set[Decimal] = set()
    percent: set[Decimal] = set()
    for match in _FIGURE.finditer(text):
        value = normalise(match.group("num"))
        if value is None:
            continue
        (percent if match.group("pct") else plain).add(value)
    return plain, percent


# ----------------------------------------------------- why it is ungrounded --
#
# An ungrounded number is classified as `derived_arithmetic` if it is the sum or
# difference of two figures in the cited passages (e.g. "increased by $750" from
# 14,600 and 13,850), otherwise as `invented_figure`.

# Cap on how many figures are paired (the pairing is O(n^2)).
_ARITHMETIC_POOL = 60


def _derives(value: Decimal, pool: set[Decimal]) -> str | None:
    """Return "b - a" or "a + b" if `value` derives from two figures in `pool`.

    Sorted so the result does not depend on set iteration order.
    """
    values = sorted(pool)[:_ARITHMETIC_POOL]
    for i, a in enumerate(values):
        for b in values[i:]:
            if b - a == value:
                return f"{b} - {a}"
            if a + b == value:
                return f"{a} + {b}"
    return None


# -------------------------------------------------------------------- check --

def _index(passages: Iterable[Passage]) -> dict[str, tuple[set[Decimal], set[Decimal]]]:
    return {p.chunk_id: passage_values(p.text) for p in passages}


def check_answer(claims: list[AnswerClaim], passages: list[Passage]) -> GroundingReport:
    """Check every claim's numbers against the passages that claim cites.

    Any number in a claim with no citations is a violation. Claims without
    numbers are not checked.
    """
    index = _index(passages)
    known = {p.chunk_id for p in passages}

    numbers: list[dict] = []
    violations: list[dict] = []
    checked = grounded = 0

    for i, claim in enumerate(claims):
        cited = [cid for cid in claim.chunk_ids if cid in known]
        hallucinated_cites = [cid for cid in claim.chunk_ids if cid not in known]

        for cid in hallucinated_cites:
            violations.append({
                "claim": i,
                "kind": "unknown_citation",
                "chunk_id": cid,
                "detail": f"claim {i} cites {cid!r}, which is not in the evidence pack",
            })

        for token in extract_numbers(claim.text):
            row = {
                "claim": i,
                "text": token.text,
                "value": str(token.value) if token.value is not None else None,
                "is_percent": token.is_percent,
                "whitelisted": token.whitelisted,
                "reason": token.reason,
                "matched_chunk_id": None,
                "grounded": None,
            }
            if token.whitelisted:
                numbers.append(row)
                continue

            checked += 1
            match_id = None
            for cid in cited:
                plain, percent = index[cid]
                pool = percent if token.is_percent else plain
                if token.value is not None and token.value in pool:
                    match_id = cid
                    break

            row["matched_chunk_id"] = match_id
            row["grounded"] = match_id is not None
            numbers.append(row)

            if match_id is None:
                pool: set[Decimal] = set()
                for cid in cited:
                    plain, percent = index[cid]
                    pool |= percent if token.is_percent else plain
                derivation = (_derives(token.value, pool)
                              if token.value is not None else None)
                violations.append({
                    "claim": i,
                    "kind": "ungrounded_number",
                    # See the note above `_derives`.
                    "subkind": "derived_arithmetic" if derivation else "invented_figure",
                    "derivation": derivation,
                    "number": token.text,
                    "cited": claim.chunk_ids,
                    "detail": (
                        f"{token.text!r} does not appear in "
                        + (f"the cited passage(s) {cited}" if cited
                           else "any passage - the claim cites none")
                        + (f"; it is {derivation} from figures that do - the model "
                           "did arithmetic" if derivation else "")
                    ),
                })
            else:
                grounded += 1

    return GroundingReport(
        ok=not violations,
        checked=checked,
        grounded=grounded,
        violations=violations,
        numbers=numbers,
    )
