"""Retrieval tests.

`rrf` and `bm25_tokens` are tested without an index. Tests that need
`data/chroma/` (gitignored) are skipped until the index is built:

    uv run python -m ingest.build_index
    uv run python -m pytest tests/test_smoke.py -q
"""

from __future__ import annotations

import pytest

from core.lexical import bm25_tokens
from core.paths import BM25_PATH, CHROMA_DIR
from core.retrieval import Filters, rrf, search
from ingest.chunk import load as load_chunks

needs_index = pytest.mark.skipif(
    not (CHROMA_DIR.exists() and BM25_PATH.exists()),
    reason="no index - run `uv run python -m ingest.build_index`",
)


# ------------------------------------------------------------ pure: RRF fusion --

def test_rrf_rewards_agreement_over_enthusiasm():
    """A chunk both retrievers rank second beats one ranked first by only one."""
    dense = ["loved", "agreed", "c"]
    lexical = ["x", "agreed", "y"]
    ranked = [cid for cid, _ in rrf(dense, lexical)]
    assert ranked[0] == "agreed"
    assert ranked.index("agreed") < ranked.index("loved")


def test_rrf_is_deterministic_on_ties():
    """Ties break by chunk_id."""
    assert rrf(["b", "a"], ["a", "b"]) == rrf(["b", "a"], ["a", "b"])


def test_rrf_of_one_list_preserves_its_order():
    assert [cid for cid, _ in rrf(["a", "b", "c"])] == ["a", "b", "c"]


# ------------------------------------------------- pure: the BM25 tokenizer --

@pytest.mark.parametrize(("text", "expected"), [
    ("$14,600", {"14,600", "14600"}),
    ("45%", {"45%", "45"}),
    ("33.75%", {"33.75%", "33.75"}),
])
def test_figures_are_indexed_both_ways(text, expected):
    """"14600" and "$14,600" must both match."""
    assert expected <= set(bm25_tokens(text))


def test_year_labels_split_into_both_years():
    """"2024-25" must be findable as 2024."""
    assert {"2024", "25"} <= set(bm25_tokens("the 2024-25 tax year"))


def test_tokenizer_is_shared_by_build_and_query():
    """Build and query use the same tokenizer."""
    from core import lexical

    assert lexical.bm25_tokens is bm25_tokens


# --------------------------------------------------------- the metadata filter --

def test_filters_build_the_where_clause_chroma_expects():
    assert Filters().as_where() is None
    assert Filters(country="US").as_where() == {"country": "US"}
    assert Filters(country="US", tax_year_key=2024).as_where() == {
        "$and": [{"country": "US"}, {"tax_year_key": 2024}]
    }


def test_the_python_predicate_agrees_with_the_where_clause():
    """BM25 filters in Python; it must agree with Chroma's `where=` clause."""
    f = Filters(country="UK", tax_year_key=2023)
    assert f.matches({"country": "UK", "tax_year_key": 2023, "doc_id": "uk_rates_2023"})
    assert not f.matches({"country": "UK", "tax_year_key": 2024})
    assert not f.matches({"country": "US", "tax_year_key": 2023})


# ------------------------------------------------------------ filtered search --

@needs_index
@pytest.mark.parametrize("mode", ["hybrid", "dense", "lexical"])
def test_the_prefilter_admits_nothing_from_another_year_or_country(mode):
    """A 2024 US query must never return passages from another year or country."""
    passages = search("standard deduction", country="US", tax_year_key=2024, mode=mode)
    assert passages
    assert {(p.country, p.tax_year_label) for p in passages} == {("US", "2024")}


@needs_index
def test_the_known_canary_retrieves_its_table():
    """The 2024 standard deduction table is Table 10-1 on PDF page 98 (printed 96)."""
    passages = search(
        "What is the 2024 standard deduction for a single filer?",
        country="US", tax_year_key=2024,
    )
    assert 98 in [p.page for p in passages]
    table = next(p for p in passages if p.page == 98)
    assert table.printed_page == 96          # page_offset 2, for display only
    assert "14,600" in table.text


@needs_index
def test_uk_passages_have_no_printed_page():
    """UK `page` is a section index, so there is no printed page."""
    passages = search("Personal Allowance", country="UK", tax_year_key=2024)
    assert passages
    assert all(p.printed_page is None for p in passages)
    assert all(1 <= p.page <= 7 for p in passages)


@needs_index
def test_search_returns_top_k_and_scores_descend():
    passages = search("income tax rates", country="UK", tax_year_key=2024, k=5)
    assert len(passages) == 5
    assert [p.score for p in passages] == sorted((p.score for p in passages), reverse=True)


@needs_index
def test_every_indexed_chunk_is_reachable_by_its_own_id():
    """Every chunk in Chroma must exist in the chunk table, and vice versa."""
    from core.retrieval import chunks_by_id, collection

    ids = set(collection().get(include=[])["ids"])
    assert ids == set(chunks_by_id())
    assert len(ids) == len(load_chunks())


@needs_index
def test_page_diversification_is_off_by_default_and_works_when_asked():
    """Default packs may repeat a page; diversified packs may not."""
    q = "What is the 2024 standard deduction for a single filer?"
    plain = search(q, country="US", tax_year_key=2024)
    spread = search(q, country="US", tax_year_key=2024, diversify_by_page=True)

    assert len({p.page for p in spread}) == len(spread)
    assert len({p.page for p in plain}) < len(plain)


def test_the_llm_client_has_one_retry_policy_and_a_bounded_request():
    """SDK retries are disabled so `_raw_call`'s retry loop is the only one."""
    from core.llm import REQUEST_TIMEOUT_S, SDK_RETRIES

    assert SDK_RETRIES == 0
    # Long enough for a slow call, short enough that a stall cannot hang a turn.
    assert 10 < REQUEST_TIMEOUT_S <= 60
