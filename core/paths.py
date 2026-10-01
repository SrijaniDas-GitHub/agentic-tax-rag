"""Repo paths and Hugging Face cache location.

`use_local_hf_cache()` must run before `transformers` or `sentence_transformers`
is imported, because both read HF_HOME at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"          # gitignored; rebuilt by ingest.build_index
BM25_PATH = DATA_DIR / "bm25.pkl"         # gitignored (*.pkl)
HF_CACHE_DIR = REPO_ROOT / ".hf_cache"    # gitignored


def use_local_hf_cache() -> Path:
    """Use ./.hf_cache for model downloads unless HF_HOME is already set."""
    existing = os.environ.get("HF_HOME")
    if existing:
        return Path(existing)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    return HF_CACHE_DIR
