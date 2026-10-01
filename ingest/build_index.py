"""Build the Chroma collection and BM25 index from data/chunks.jsonl.

Both indexes are built from the same chunk list in one pass, since hybrid
retrieval fuses their rankings by chunk_id. Chunk metadata is stored for
`where=` filtering by country and tax year.

    uv run python -m ingest.build_index
    uv run python -m ingest.build_index --stats
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

from core.paths import use_local_hf_cache

use_local_hf_cache()   # before anything imports torch (see core/paths.py)

from core import lexical
from core.embedding import EMBED_MODEL, dimension, embed_passages
from core.paths import BM25_PATH, CHROMA_DIR
from ingest.chunk import load as load_chunks

COLLECTION = "tax_chunks"

# Everything in chunks.jsonl except `text` (the document) and `chunk_id` (the id).
META_FIELDS = (
    "country", "tax_year_key", "tax_year_label", "doc_id", "doc_title",
    "page", "printed_page", "url", "has_table", "n_tokens",
)


def chroma_client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def metadata_of(chunk: dict) -> dict:
    """Chunk metadata for Chroma. None values are dropped (Chroma rejects them)."""
    return {k: chunk[k] for k in META_FIELDS if chunk.get(k) is not None}


def build(rebuild: bool = True) -> dict:
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("data/chunks.jsonl is empty - run `uv run python -m ingest.chunk`")

    print(f"  {len(chunks)} chunks from data/chunks.jsonl")
    print(f"  embedding with {EMBED_MODEL}")

    if rebuild and CHROMA_DIR.exists():
        # Start from an empty directory so stale chunk ids cannot survive a rebuild.
        shutil.rmtree(CHROMA_DIR)

    t0 = time.perf_counter()
    vectors = embed_passages([c["text"] for c in chunks])
    t_embed = time.perf_counter() - t0
    print(f"  embedded {len(vectors)} x {dimension()}d in {t_embed:.1f}s "
          f"({len(vectors) / max(t_embed, 1e-9):.0f} chunks/s)")

    client = chroma_client()
    collection = client.get_or_create_collection(
        COLLECTION,
        # Vectors are normalized, so cosine distance = 1 - similarity.
        metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
    )
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[metadata_of(c) for c in chunks],
    )
    print(f"  chroma  -> {CHROMA_DIR.name}/ ({collection.count()} docs)")

    t0 = time.perf_counter()
    lexical.save(lexical.build(chunks))
    print(f"  bm25    -> {BM25_PATH.name} "
          f"({BM25_PATH.stat().st_size / 1e6:.2f} MB, {time.perf_counter() - t0:.1f}s)")

    return {"chunks": len(chunks), "dim": dimension(), "embed_seconds": t_embed}


def stats() -> None:
    """Print chunk counts per document and check both indexes agree."""
    collection = chroma_client().get_collection(COLLECTION)
    got = collection.get(include=["metadatas"])
    rows: dict[tuple, int] = {}
    for m in got["metadatas"]:
        key = (m["country"], m["tax_year_label"], m["doc_id"])
        rows[key] = rows.get(key, 0) + 1
    print(f"  collection {COLLECTION!r}: {collection.count()} chunks")
    for (country, label, doc_id), n in sorted(rows.items()):
        print(f"    {country}  TY{label:8s}  {doc_id:16s} {n:4d}")
    bm = lexical.load()
    print(f"  bm25 pickle: {bm['n_chunks']} chunks")
    if bm["n_chunks"] != collection.count():
        raise SystemExit("MISMATCH: the two indexes disagree on how many chunks exist")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Build the Chroma + BM25 indexes")
    ap.add_argument("--stats", action="store_true", help="describe the existing index and exit")
    args = ap.parse_args(argv)

    if args.stats:
        stats()
        return 0

    build()
    print()
    stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
