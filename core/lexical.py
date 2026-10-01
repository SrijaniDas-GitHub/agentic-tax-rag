"""BM25 index over the same chunks stored in Chroma.

Dense embeddings barely distinguish "$14,600" from "$13,850", so exact figures
are matched lexically. `bm25_tokens` emits every number both with and without
separators ("14,600" and "14600"). Index build and query must share this tokenizer.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from core.paths import BM25_PATH

# Words, and numbers with their separators. "-" is excluded so "2024-25" splits
# into "2024" and "25".
_TOKEN = re.compile(r"[a-z]+|\d[\d,./]*%?")
_STRIPPABLE = ",.%/"


def bm25_tokens(text: str) -> list[str]:
    """Lowercase tokens, plus a punctuation-free copy of every figure."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if any(ch in tok for ch in _STRIPPABLE):
            bare = tok.replace(",", "").replace("/", "").rstrip("%").rstrip(".")
            if bare and bare != tok:
                out.append(bare)
    return out


def build(chunks: list[dict]) -> dict[str, Any]:
    """Return the picklable index: the BM25 model plus its chunk order."""
    from rank_bm25 import BM25Okapi

    corpus = [bm25_tokens(c["text"]) for c in chunks]
    return {
        "bm25": BM25Okapi(corpus),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "n_chunks": len(chunks),
    }


def save(index: dict[str, Any], path: Path = BM25_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path = BM25_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `uv run python -m ingest.build_index`"
        )
    with path.open("rb") as fh:
        return pickle.load(fh)
