"""Corpus and chunk checks.

Guards two silent failures: chunks over the embedding model's token limit are
truncated without error, and a tax_year_key that does not match its label
returns the wrong year.

    uv run python -m pytest tests/test_corpus.py -q
"""

from __future__ import annotations

import pytest

from ingest import chunk as chunk_mod
from ingest.manifest import CHUNKS_PATH, load_manifest

DOCS = load_manifest()


@pytest.fixture(scope="module")
def chunks():
    if not CHUNKS_PATH.exists():
        pytest.skip("data/chunks.jsonl not built - run `uv run python -m ingest.chunk`")
    return chunk_mod.load()


# ------------------------------------------------------------------ manifest --

def test_manifest_has_four_docs_two_countries_two_years():
    assert len(DOCS) == 4
    assert {d.country for d in DOCS} == {"US", "UK"}
    assert {(d.country, d.tax_year_key) for d in DOCS} == {
        ("US", 2023), ("US", 2024), ("UK", 2023), ("UK", 2024),
    }


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_tax_year_key_is_the_year_the_year_starts_in(doc):
    """UK 2024-25 -> 2024, not 2025."""
    assert doc.tax_year_label.startswith(str(doc.tax_year_key))


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_page_offset_only_where_there_are_printed_pages(doc):
    if doc.media == "pdf":
        assert doc.page_offset is not None
        assert doc.printed_page(doc.pages()[0]) == doc.pages()[0] - doc.page_offset
    else:
        assert doc.page_offset is None
        assert doc.printed_page(1) is None


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_trimmed_not_whole_document(doc):
    """Only the relevant pages of each source are ingested."""
    assert len(doc.pages()) <= 60


# -------------------------------------------------------------------- chunks --

def test_every_chunk_fits_the_embedding_model(chunks):
    limit = chunk_mod.max_seq_length()
    over = [(c["chunk_id"], c["n_tokens"]) for c in chunks if c["n_tokens"] > limit]
    assert not over, f"chunks exceed {limit} tokens and would be silently truncated: {over[:5]}"


def test_recorded_token_counts_are_real(chunks):
    """Re-measure a sample of the stored n_tokens values."""
    for c in chunks[::40]:
        assert chunk_mod.assert_fits(c["chunk_id"], c["text"]) == c["n_tokens"]


def test_every_doc_produced_chunks(chunks):
    got = {c["doc_id"] for c in chunks}
    assert got == {d.doc_id for d in DOCS}


def test_contextual_prefix_on_every_chunk(chunks):
    """Every chunk starts with a prefix like 'US · TY2024 · Pub 17 · p.98'."""
    by_id = {d.doc_id: d for d in DOCS}
    for c in chunks:
        doc = by_id[c["doc_id"]]
        assert c["text"].startswith(doc.prefix(c["page"]) + "\n\n")


@pytest.mark.parametrize(
    "needle, expected_docs",
    [
        ("14,600", {"us_p17_2024"}),   # US 2024 standard deduction, single
        ("13,850", {"us_p17_2023"}),   # US 2023 standard deduction, single
        ("12,570", {"uk_rates_2023", "uk_rates_2024"}),  # UK personal allowance, frozen
        ("47%", {"uk_rates_2023"}),    # Scottish top rate, 2023-24 only
        ("48%", {"uk_rates_2024"}),    # Scottish top rate, 2024-25 only
    ],
)
def test_canary_figures_land_in_exactly_the_right_documents(chunks, needle, expected_docs):
    """Year-specific figures must appear only in their own year's documents."""
    got = {c["doc_id"] for c in chunks if needle in c["text"]}
    assert got == expected_docs


@pytest.mark.parametrize("country", ["US", "UK"])
def test_one_rate_table_per_country_survived_extraction(chunks, country):
    """Table extraction keeps row labels attached to their figures."""
    tables = [c for c in chunks if c["country"] == country and c["has_table"]]
    assert tables, f"no table chunks for {country}"
    marker = "Schedule X" if country == "US" else "England, Northern Ireland and Wales"
    rate = [c for c in tables if marker in c["text"]]
    assert rate, f"no {country} rate table chunk containing {marker!r}"
    text = rate[0]["text"]
    assert text.count("\n|") >= 4, "table collapsed to fewer than 4 rows"
    if country == "UK":
        assert "Higher rate | 40% | £37,701 to £125,140" in text


def test_uk_table_chunks_say_which_country_they_are_for(chunks):
    """England and Scotland tables are structurally identical; only the heading differs."""
    uk_tables = [c for c in chunks if c["country"] == "UK" and c["has_table"]]
    rate_tables = [c for c in uk_tables if "| Basic rate | 20% |" in c["text"]]
    assert len(rate_tables) == 4, "expected an England and a Scotland band table per year"
    for c in rate_tables:
        assert ("England, Northern Ireland and Wales" in c["text"]) or ("Scotland" in c["text"])


def test_uk_chunks_contain_only_their_own_tax_year_column(chunks):
    """parse.py keeps only the document's own tax-year column."""
    for c in chunks:
        if c["country"] != "UK" or not c["has_table"]:
            continue
        other = "2023 to 2024" if c["tax_year_key"] == 2024 else "2024 to 2025"
        assert other not in c["text"], f"{c['chunk_id']} leaked the {other} column"
