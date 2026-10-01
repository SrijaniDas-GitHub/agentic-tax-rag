"""Corpus definition, loaded from data/manifest.yaml.

Two rules, enforced in `_validate`:

1. `tax_year_key` is the calendar year the tax year starts in (US TY2024 -> 2024,
   UK 2024-25 -> 2024). It is the key the retrieval filter joins on.
2. `page` is the 1-based PDF index, not the printed page number.
   `printed_page = page - page_offset` is for display only.

Adding a country means adding manifest entries; no code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"

# `pdf`   - a PDF; pages are real pages, PyMuPDF for text + pdfplumber for tables.
# `govuk` - a gov.uk Content API document; "pages" are heading sections (see parse.py).
MEDIA = ("pdf", "govuk")


@dataclass(frozen=True)
class Doc:
    doc_id: str
    country: str
    tax_year_key: int
    tax_year_label: str
    doc_title: str
    short_title: str          # used in the chunk prefix, so keep it short
    url: str                  # the human-facing page shown in citations
    media: str
    filename: str             # inside data/raw/
    page_ranges: list[tuple[int, int]]   # inclusive, 1-based
    page_offset: int | None = None       # printed_page = page - page_offset; None if no printed pages
    fetch_url: str | None = None         # machine-readable source, when it differs from `url`
    licence: str = ""
    sha256: str | None = None
    notes: str = ""

    @property
    def path(self) -> Path:
        return RAW_DIR / self.filename

    @property
    def download_url(self) -> str:
        return self.fetch_url or self.url

    @property
    def year_column(self) -> str:
        """The gov.uk column header for this tax year: 2024 -> '2024 to 2025'."""
        return f"{self.tax_year_key} to {self.tax_year_key + 1}"

    def pages(self) -> list[int]:
        """Every 1-based page index this document contributes, sorted and deduplicated."""
        seen: list[int] = []
        for lo, hi in self.page_ranges:
            seen.extend(range(lo, hi + 1))
        return sorted(set(seen))

    def printed_page(self, page: int) -> int | None:
        return None if self.page_offset is None else page - self.page_offset

    def prefix(self, page: int) -> str:
        """Chunk prefix, e.g. 'US · TY2024 · Pub 17 · p.98'. Web sections use '§N'."""
        marker = f"p.{page}" if self.media == "pdf" else f"§{page}"
        return f"{self.country} · TY{self.tax_year_label} · {self.short_title} · {marker}"


def _validate(docs: list[Doc]) -> None:
    ids = [d.doc_id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate doc_id in manifest: {sorted(dupes)}")

    for d in docs:
        where = f"{d.doc_id}:"
        if d.media not in MEDIA:
            raise ValueError(f"{where} media {d.media!r} not one of {MEDIA}")
        if not 1900 < d.tax_year_key < 2100:
            raise ValueError(f"{where} implausible tax_year_key {d.tax_year_key}")
        # The key must be the year the label starts in (catches UK off-by-one errors).
        if not d.tax_year_label.startswith(str(d.tax_year_key)):
            raise ValueError(
                f"{where} tax_year_label {d.tax_year_label!r} must start with "
                f"tax_year_key {d.tax_year_key} - the key is the year the tax year STARTS in"
            )
        if not d.page_ranges:
            raise ValueError(f"{where} empty page_ranges - list the pages to ingest")
        for lo, hi in d.page_ranges:
            if lo < 1 or hi < lo:
                raise ValueError(f"{where} bad page range [{lo}, {hi}] (1-based, inclusive)")


def load_manifest(path: Path = MANIFEST_PATH) -> list[Doc]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    docs = [
        Doc(**{**entry, "page_ranges": [tuple(r) for r in entry["page_ranges"]]})
        for entry in raw
    ]
    _validate(docs)
    return docs


def save_manifest(docs: list[Doc], path: Path = MANIFEST_PATH) -> None:
    """Write sha256 values into the manifest, leaving the rest of the file untouched."""
    text = path.read_text(encoding="utf-8")
    for d in docs:
        if d.sha256:
            text = _set_sha(text, d.doc_id, d.sha256)
    path.write_text(text, encoding="utf-8")


def _set_sha(text: str, doc_id: str, sha: str) -> str:
    lines = text.splitlines()
    inside = False
    for i, line in enumerate(lines):
        if line.strip().startswith("- doc_id:"):
            inside = line.split("doc_id:")[1].strip() == doc_id
        elif inside and line.strip().startswith("sha256:"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}sha256: {sha}"
            inside = False
    return "\n".join(lines) + "\n"
