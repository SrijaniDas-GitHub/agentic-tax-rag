"""Formatting a `Passage` as a citation.

US documents are PDFs and cite the printed page number (`p.96`), which differs
from the PDF index. UK documents are web pages split into numbered sections and
cite the section (`§5`). `printed_page is None` decides which applies.
"""

from __future__ import annotations

from agents.contracts import Passage


def locator(passage: Passage) -> str:
    """`p.96` for a PDF with printed page numbers, `§5` for a web document."""
    if passage.printed_page is not None:
        return f"p.{passage.printed_page}"
    return f"§{passage.page}"


def format_citation(passage: Passage) -> str:
    """Human-readable citation, e.g. `IRS Publication 17 (2024), p.96`."""
    return f"{passage.doc_title}, {locator(passage)}"


def citation_dict(passage: Passage) -> dict:
    """The citation as structured data, for the trace panel and UI."""
    return {
        "chunk_id": passage.chunk_id,
        "label": format_citation(passage),
        "doc_id": passage.doc_id,
        "doc_title": passage.doc_title,
        "country": passage.country,
        "tax_year_label": passage.tax_year_label,
        "page": passage.page,
        "printed_page": passage.printed_page,
        "locator": locator(passage),
        "url": passage.url,
    }


def numbered(passages: list[Passage]) -> tuple[list[dict], dict[str, int]]:
    """Number the cited passages [1], [2], ... in order of first use.

    Two chunks of the same page or section share one number: a reader cannot tell
    `[1] ..., §1` from `[2] ..., §1` apart, so listing both is noise.

    Returns the citation list and a `chunk_id -> marker number` map.
    """
    order: dict[str, int] = {}
    by_label: dict[str, int] = {}
    out: list[dict] = []
    for passage in passages:
        if passage.chunk_id in order:
            continue
        label = format_citation(passage)
        if label in by_label:
            order[passage.chunk_id] = by_label[label]
            continue
        order[passage.chunk_id] = by_label[label] = len(out) + 1
        entry = citation_dict(passage)
        entry["n"] = order[passage.chunk_id]
        out.append(entry)
    return out, order
