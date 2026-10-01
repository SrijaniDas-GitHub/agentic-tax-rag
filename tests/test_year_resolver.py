"""Year resolver tests: tax_year_key is the year the tax year starts in.

Covers four conventions (US, UK, AU, IN). Only US and UK are in the corpus; AU
and IN show the resolver extends to other tax-year conventions.

`today` is injected, so relative-year cases are deterministic.

    uv run python -m pytest tests/test_year_resolver.py -q
"""

from __future__ import annotations

from datetime import date

import pytest

from core.year_resolver import (
    AmbiguousYearError,
    YearResolutionError,
    current_tax_year_start,
    label_for,
    resolve,
)

TODAY = date(2026, 9, 20)      # a fixed "now" for every relative case


# ------------------------------------------------- the rule, all four countries --
#
# (country, user text, expected key, expected display label)
CASES = [
    # US - calendar year. "2024" is TY2024 and nothing else.
    ("US", "2024", 2024, "2024"),
    ("US", "TY2024", 2024, "2024"),
    ("US", "tax year 2024", 2024, "2024"),
    ("US", "2023", 2023, "2023"),
    # A span given for a calendar-year country is read as its start year.
    ("US", "2024-25", 2024, "2024"),

    # UK - 6 April to 5 April. The key is the start year.
    ("UK", "2024-25", 2024, "2024-25"),
    ("UK", "2024/25", 2024, "2024-25"),
    ("UK", "2024 to 2025", 2024, "2024-25"),
    ("UK", "FY24-25", 2024, "2024-25"),
    ("UK", "2024", 2024, "2024-25"),
    ("UK", "2023-24", 2023, "2023-24"),

    # AU - 1 July to 30 June. A bare year names the year it ends in: "2024" is 2023-24.
    ("AU", "2024", 2023, "2023-24"),
    ("AU", "2023-24", 2023, "2023-24"),
    ("AU", "2024-25", 2024, "2024-25"),
    ("AU", "income year 2024", 2023, "2023-24"),

    # IN - FY April to March, AY = FY + 1. AY2025-26 assesses FY2024-25 -> key 2024.
    ("IN", "AY 2025-26", 2024, "2024-25"),
    ("IN", "assessment year 2025-26", 2024, "2024-25"),
    ("IN", "FY 2024-25", 2024, "2024-25"),
    ("IN", "FY2024-25", 2024, "2024-25"),
    ("IN", "2024", 2024, "2024-25"),
    ("IN", "AY 2024-25", 2023, "2023-24"),
]


@pytest.mark.parametrize(("country", "text", "key", "label"), CASES,
                         ids=[f"{c}:{t}" for c, t, _, _ in CASES])
def test_key_is_the_year_the_tax_year_starts_in(country, text, key, label):
    got = resolve(country, text, TODAY)
    assert (got.tax_year_key, got.tax_year_label) == (key, label)
    assert got.country == country
    assert got.source_text == text


# ------------------------------------------------------------------ the corpus --

@pytest.mark.parametrize(("country", "text", "key"), [
    ("US", "2024", 2024), ("US", "2023", 2023),
    ("UK", "2024-25", 2024), ("UK", "2023-24", 2023),
])
def test_the_four_manifest_labels_round_trip(country, text, key):
    """The four labels used in the manifest."""
    assert resolve(country, text, TODAY).tax_year_key == key


# ------------------------------------------------------------------- relative --

@pytest.mark.parametrize(("country", "text", "key"), [
    # today = 20 Sep 2026: past 6 Apr, past 1 Jul, past 1 Apr - every country is in
    # the tax year that started in 2026.
    ("US", "this year", 2026), ("US", "last year", 2025), ("US", "next year", 2027),
    ("UK", "this tax year", 2026), ("UK", "last year", 2025),
    ("UK", "the current tax year", 2026), ("UK", "previous year", 2025),
    ("AU", "this year", 2026), ("AU", "last financial year", 2025),
    ("IN", "current", 2026), ("IN", "prior year", 2025),
])
def test_relative_years_resolve_against_the_injected_today(country, text, key):
    got = resolve(country, text, TODAY)
    assert got.tax_year_key == key
    # A year the user did not state is an assumption and must be surfaced.
    assert got.assumption_note


@pytest.mark.parametrize(("country", "day", "key"), [
    # One day either side of each start date.
    ("US", date(2026, 1, 1), 2026), ("US", date(2025, 12, 31), 2025),
    ("UK", date(2026, 4, 5), 2025), ("UK", date(2026, 4, 6), 2026),
    ("AU", date(2026, 6, 30), 2025), ("AU", date(2026, 7, 1), 2026),
    ("IN", date(2026, 3, 31), 2025), ("IN", date(2026, 4, 1), 2026),
])
def test_current_tax_year_boundaries(country, day, key):
    assert current_tax_year_start(country, day) == key
    assert resolve(country, "this year", day).tax_year_key == key


# ------------------------------------------------------- assumption_note rules --

def test_au_always_carries_a_note():
    """AU years never map 1:1 to a bare year, so a note is always attached."""
    for text in ("2024", "2023-24", "this year", "FY24-25"):
        note = resolve("AU", text, TODAY).assumption_note
        assert note and "30 June" in note


def test_unambiguous_cases_carry_no_note():
    """Notes are shown to the user, so unambiguous years get none."""
    assert resolve("US", "2024", TODAY).assumption_note is None
    assert resolve("UK", "2024-25", TODAY).assumption_note is None


@pytest.mark.parametrize(("country", "text", "needle"), [
    ("US", "2024-25", "calendar year"),      # a span given to a country with none
    ("IN", "2024", "financial year"),        # bare year read as the FY, not the AY
    ("IN", "AY 2025-26", "assessment year"), # the AY -> FY shift, stated
    ("UK", "AY 2025-26", "Indian convention"),  # AY asked of a country that has none
])
def test_non_one_to_one_mappings_explain_themselves(country, text, needle):
    note = resolve(country, text, TODAY).assumption_note
    assert note and needle in note


# ------------------------------------------------------ refusals and clarifies --

@pytest.mark.parametrize("text", ["FY24", "24", "fy 25"])
def test_bare_two_digit_years_are_ambiguous_not_guessed(text):
    """'FY24' means different years in different countries, so it is not guessed."""
    with pytest.raises(AmbiguousYearError):
        resolve("AU", text, TODAY)


@pytest.mark.parametrize("text", ["", "   ", "sometime", "the good year", "my return"])
def test_text_naming_no_year_raises(text):
    with pytest.raises(YearResolutionError):
        resolve("UK", text, TODAY)


@pytest.mark.parametrize("text", ["2024-26", "2020 to 2025", "1850"])
def test_impossible_years_raise(text):
    with pytest.raises(YearResolutionError):
        resolve("UK", text, TODAY)


def test_unknown_country_raises():
    with pytest.raises(YearResolutionError):
        resolve("FR", "2024", TODAY)


@pytest.mark.parametrize(("alias", "canonical"), [
    ("usa", "US"), ("United States", "US"), ("gb", "UK"), ("Scotland", "UK"),
    ("australia", "AU"), ("india", "IN"),
])
def test_country_aliases_normalise(alias, canonical):
    assert resolve(alias, "2024-25", TODAY).country == canonical


# ------------------------------------------------------------- label rendering --

@pytest.mark.parametrize(("country", "start", "label"), [
    ("US", 2024, "2024"),
    ("UK", 2024, "2024-25"),
    ("UK", 1999, "1999-00"),   # zero-padded, not "1999-0"
    ("AU", 2023, "2023-24"),
    ("IN", 2024, "2024-25"),
])
def test_label_for(country, start, label):
    assert label_for(country, start) == label


def test_numbers_in_tax_text_are_not_mistaken_for_year_spans():
    """'125,140' is the UK additional-rate threshold, not a tax year."""
    with pytest.raises(YearResolutionError):
        resolve("UK", "125,140", TODAY)
