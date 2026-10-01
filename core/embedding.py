"""Embedding model (BAAI/bge-small-en-v1.5), loaded once.

`EMBED_MODEL` comes from `ingest.chunk` so the chunker's token limit always
matches the model used here. bge is asymmetric: queries get an instruction
prefix, passages do not. Vectors are L2-normalized, so inner product = cosine.
"""

from __future__ import annotations

from core.paths import use_local_hf_cache

use_local_hf_cache()   # must precede the sentence_transformers import (see core/paths.py)

from ingest.chunk import EMBED_MODEL

# Retrieval instruction from the model card. Queries only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def dimension() -> int:
    m = model()
    # sentence-transformers 6 renamed this method.
    fn = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
    return int(fn())


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Index side: no instruction prefix."""
    vecs = model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    """Query side: with the instruction prefix."""
    return embed_queries([text])[0]


def embed_queries(texts: list[str]) -> list[list[float]]:
    vecs = model().encode(
        [QUERY_INSTRUCTION + t for t in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]
