"""Hybrid retrieval: metadata prefilter -> dense top-20 + BM25 top-20 -> RRF (k=60) -> top-8.

The country/tax-year filter is applied before ranking, so both retrievers only
see chunks from the requested year. RRF fuses by rank because cosine and BM25
scores are on different scales. No reranker: after filtering the candidate pool
is small. `eval/retrieval_eval.py` compares dense, lexical and hybrid.

    from core.retrieval import search
    search("2024 standard deduction married filing jointly", country="US", tax_year_key=2024)
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from agents.contracts import Passage
from core import lexical
from core.paths import CHROMA_DIR
from ingest.build_index import COLLECTION
from ingest.chunk import load as load_chunks

RRF_K = 60          # standard RRF constant
DENSE_K = 20
LEXICAL_K = 20
TOP_K = 8

Mode = Literal["hybrid", "dense", "lexical"]

# Process-wide default for `one_per_page`. `eval/run_eval.py --diversify` sets
# it for a run; `eval/retrieval_eval.py` passes the flag explicitly.
DIVERSIFY_BY_PAGE = os.getenv("DIVERSIFY_BY_PAGE", "0").strip().lower() in {
    "1", "true", "yes"
}


@dataclass(frozen=True)
class Filters:
    """Metadata filter, rendered as a Chroma `where=` clause or applied in Python."""

    country: str | None = None
    tax_year_key: int | None = None
    doc_id: str | None = None
    has_table: bool | None = None

    def as_where(self) -> dict | None:
        clauses = [
            {field: value}
            for field, value in (
                ("country", self.country),
                ("tax_year_key", self.tax_year_key),
                ("doc_id", self.doc_id),
                ("has_table", self.has_table),
            )
            if value is not None
        ]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def matches(self, meta: dict) -> bool:
        """The same predicate in Python, for BM25."""
        where = self.as_where()
        if where is None:
            return True
        for clause in where.get("$and", [where]):
            (field, value), = clause.items()
            if meta.get(field) != value:
                return False
        return True


# ------------------------------------------------------------------- loading --
#
# Both indexes and the chunk table are loaded once per process and are read-only.

_collection = None
_bm25: dict[str, Any] | None = None
_chunks: dict[str, dict] | None = None


def collection():
    global _collection
    if _collection is None:
        import chromadb

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"{CHROMA_DIR} not found - run `uv run python -m ingest.build_index`"
            )
        _collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    return _collection


def bm25_index() -> dict[str, Any]:
    global _bm25
    if _bm25 is None:
        _bm25 = lexical.load()
    return _bm25


def chunks_by_id() -> dict[str, dict]:
    global _chunks
    if _chunks is None:
        _chunks = {c["chunk_id"]: c for c in load_chunks()}
    return _chunks


def reset_caches() -> None:
    """Clear the loaded indexes (tests and rebuilds only)."""
    global _collection, _bm25, _chunks
    _collection, _bm25, _chunks = None, None, None


# ---------------------------------------------------------------- retrievers --

def dense(query: str, filters: Filters, k: int = DENSE_K) -> list[str]:
    """Chunk ids, best first, from the vector index with `where=` applied natively."""
    from core.embedding import embed_query

    result = collection().query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        where=filters.as_where(),
        include=[],          # ids come back regardless; documents are joined locally
    )
    return list(result["ids"][0])


def bm25(query: str, filters: Filters, k: int = LEXICAL_K) -> list[str]:
    """Chunk ids, best first, from BM25 with the filter applied in Python.

    `rank_bm25` has no metadata filter, so the whole corpus is scored and then
    filtered. Fine at this corpus size.
    """
    index = bm25_index()
    scores = index["bm25"].get_scores(lexical.bm25_tokens(query))
    table = chunks_by_id()

    scored = [
        (cid, float(score))
        for cid, score in zip(index["chunk_ids"], scores)
        if score > 0 and filters.matches(table[cid])
    ]
    scored.sort(key=lambda row: (-row[1], row[0]))
    return [cid for cid, _ in scored[:k]]


# ----------------------------------------------------------------------- RRF --

def rrf(*rankings: Iterable[str], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d)).

    Ties break by chunk_id.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda row: (-row[1], row[0]))


# -------------------------------------------------------------------- search --

def to_passage(chunk: dict, score: float) -> Passage:
    return Passage(
        chunk_id=chunk["chunk_id"],
        text=chunk["text"],
        country=chunk["country"],
        tax_year_label=chunk["tax_year_label"],
        doc_id=chunk["doc_id"],
        doc_title=chunk["doc_title"],
        page=chunk["page"],
        printed_page=chunk.get("printed_page"),
        url=chunk.get("url", ""),
        score=score,
    )


def one_per_page(fused: list[tuple[str, float]], table: dict[str, dict], k: int
                 ) -> list[tuple[str, float]]:
    """Keep the best-ranked chunk per (doc_id, page), then take k.

    US table pages appear twice in the index (markdown table and page text), so
    a top-8 often holds several copies of the same page. This raises recall@8 on
    the gold set (0.833 -> 0.917) but is off by default: the prose copy carries
    qualifiers that the table omits.
    """
    kept: list[tuple[str, float]] = []
    seen: set[tuple[str, int]] = set()
    for cid, score in fused:
        chunk = table[cid]
        key = (chunk["doc_id"], chunk["page"])
        if key in seen:
            continue
        seen.add(key)
        kept.append((cid, score))
        if len(kept) == k:
            break
    return kept


def search(
    query: str,
    *,
    country: str | None = None,
    tax_year_key: int | None = None,
    doc_id: str | None = None,
    k: int = TOP_K,
    mode: Mode = "hybrid",
    dense_k: int = DENSE_K,
    lexical_k: int = LEXICAL_K,
    diversify_by_page: bool | None = None,
) -> list[Passage]:
    """Retrieve the top-k passages. `mode` selects hybrid, dense-only or lexical-only.

    `diversify_by_page=None` uses the process default (`DIVERSIFY_BY_PAGE`).
    """
    filters = Filters(country=country, tax_year_key=tax_year_key, doc_id=doc_id)

    if mode == "dense":
        rankings = [dense(query, filters, dense_k)]
    elif mode == "lexical":
        rankings = [bm25(query, filters, lexical_k)]
    else:
        rankings = [dense(query, filters, dense_k), bm25(query, filters, lexical_k)]

    table = chunks_by_id()
    fused = rrf(*rankings)
    spread = DIVERSIFY_BY_PAGE if diversify_by_page is None else diversify_by_page
    top = one_per_page(fused, table, k) if spread else fused[:k]
    return [to_passage(table[cid], score) for cid, score in top]
