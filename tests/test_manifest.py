"""The year resolver and the manifest must agree on every document's `tax_year_key`.

Retrieval filters on that key, so a mismatch returns nothing or the wrong year
without raising. UK documents are labelled by the year the tax year ends, which
makes "2024-25 -> 2024" the case most likely to break.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.year_resolver import label_for, resolve
from ingest.manifest import load_manifest

DOCS = load_manifest()
TODAY = date(2026, 9, 20)   # irrelevant here: all labels are explicit


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_resolver_agrees_with_the_manifest_key(doc):
    got = resolve(doc.country, doc.tax_year_label, TODAY)
    assert got.tax_year_key == doc.tax_year_key, (
        f"{doc.doc_id}: manifest says tax_year_key={doc.tax_year_key} for label "
        f"{doc.tax_year_label!r}, resolver says {got.tax_year_key}. One of them is "
        f"wrong, and the filter joins on this number."
    )


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_resolver_regenerates_the_manifest_label(doc):
    """key -> label must reproduce the manifest string exactly."""
    assert label_for(doc.country, doc.tax_year_key) == doc.tax_year_label


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_a_bare_year_reaches_the_same_document(doc):
    """Users type "2024", not "2024-25"."""
    got = resolve(doc.country, str(doc.tax_year_key), TODAY)
    assert got.tax_year_key == doc.tax_year_key


def test_the_uk_off_by_one_is_the_case_that_matters():
    assert resolve("UK", "2024-25", TODAY).tax_year_key == 2024
    assert resolve("UK", "2023-24", TODAY).tax_year_key == 2023
    assert resolve("UK", "2024-25", TODAY).tax_year_key != 2025


def test_every_country_in_the_manifest_is_known_to_the_resolver():
    for doc in DOCS:
        assert resolve(doc.country, doc.tax_year_label, TODAY).country == doc.country
