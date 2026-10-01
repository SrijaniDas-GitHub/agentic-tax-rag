"""Parse manifest documents into pages of text and markdown tables.

Tables are extracted separately and rendered as markdown so row labels stay
attached to their figures; a plain text layer flattens them into one line.

`pdf`   - PyMuPDF for text, pdfplumber for tables. `page` is the 1-based PDF index.

`govuk` - a gov.uk Content API document. `page` is a 1-based heading-section index.
          Tables list every tax year as columns, so each one is reduced to the
          document's own year (`project_year`). Otherwise a 2024 chunk would also
          contain 2023-24 figures.

    uv run python -m ingest.parse
    uv run python -m ingest.parse --show us_p17_2024:98
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser

import pdfplumber
import pymupdf

from ingest.manifest import Doc, load_manifest

# Not \b-anchored: gov.uk headers run words into the year ("allowances2024 to 2025").
YEAR_COL = re.compile(r"(?<!\d)(\d{4})\s+to\s+(\d{4})(?!\d)")
EMPTY_CELL = {"", "-", "—", "–", "n/a", "N/A"}


@dataclass
class Page:
    """One unit of the corpus that carries a citable `number`."""

    number: int
    text: str = ""
    tables: list[str] = field(default_factory=list)
    heading: str = ""

    def is_empty(self) -> bool:
        return not self.text.strip() and not self.tables


# --------------------------------------------------------------------- tables --

def to_markdown(rows: list[list[str | None]]) -> str:
    """Render a table as markdown: header row, separator, body."""
    clean = [[re.sub(r"\s+", " ", (c or "")).strip() for c in row] for row in rows]
    clean = [r for r in clean if any(r)]
    if not clean:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    out = ["| " + " | ".join(clean[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(out)


def c_norm(cell: str) -> str:
    """gov.uk headers run words together: 'Income after allowances2024 to 2025'."""
    return re.sub(r"\s+", " ", cell).strip()


def project_year(rows: list[list[str]], year_column: str) -> list[list[str]] | None:
    """Reduce a multi-year gov.uk table to its label columns plus one year's column.

    Returns None if the table has no column for this year (e.g. historical tables).
    Rows whose value is a dash (a band that did not exist that year) are dropped.
    """
    if not rows:
        return None
    header = rows[0]
    year_idx = [i for i, c in enumerate(header) if YEAR_COL.search(c)]
    keep_year = next((i for i in year_idx if year_column in c_norm(header[i])), None)
    if keep_year is None:
        return None
    labels = [i for i in range(len(header)) if i not in year_idx]
    keep = sorted(labels + [keep_year])

    out: list[list[str]] = []
    for r_no, row in enumerate(rows):
        padded = row + [""] * (len(header) - len(row))
        picked = [padded[i] for i in keep]
        if r_no and picked[-1].strip() in EMPTY_CELL:
            continue
        out.append(picked)
    return out if len(out) > 1 else None


# ------------------------------------------------------------------------ pdf --

def parse_pdf(doc: Doc) -> list[Page]:
    wanted = doc.pages()
    pages: dict[int, Page] = {}

    with pymupdf.open(doc.path) as pdf:
        if wanted and wanted[-1] > pdf.page_count:
            raise ValueError(
                f"{doc.doc_id}: manifest asks for page {wanted[-1]} but the PDF has "
                f"{pdf.page_count}. Re-pick page_ranges against this edition."
            )
        for n in wanted:
            pages[n] = Page(number=n, text=pdf[n - 1].get_text().strip())

    # pdfplumber is far slower than PyMuPDF, so only touch the trimmed pages.
    with pdfplumber.open(doc.path) as pdf:
        for n in wanted:
            for table in pdf.pages[n - 1].extract_tables() or []:
                md = to_markdown(table)
                # A one-row "table" is usually a layout artefact, not data.
                if md and md.count("\n") >= 2:
                    pages[n].tables.append(md)

    return [pages[n] for n in wanted]


# ----------------------------------------------------------------------- html --

class _Blocks(HTMLParser):
    """Flatten govspeak HTML into headings, paragraphs and tables, in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, object]] = []
        self._buf: list[str] = []
        self._tag: str | None = None
        self._rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._rows = []
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag in ("h2", "h3", "p", "li"):
            self._tag, self._buf = tag, []

    def handle_endtag(self, tag):
        if tag == "table":
            if self._rows:
                self.blocks.append(("table", self._rows))
            self._rows = []
        elif tag == "tr" and self._row is not None:
            if any(c.strip() for c in self._row):
                self._rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            if self._row is not None:
                self._row.append(html.unescape("".join(self._cell)).strip())
            self._cell = None
        elif tag == self._tag:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text:
                kind = "heading" if tag in ("h2", "h3") else "para"
                self.blocks.append((kind, text))
            self._tag, self._buf = None, []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)
        elif self._tag:
            self._buf.append(data)


def parse_govuk(doc: Doc) -> list[Page]:
    payload = json.loads(doc.path.read_text(encoding="utf-8"))
    body = payload.get("details", {}).get("body", "")
    if not body:
        raise ValueError(f"{doc.doc_id}: gov.uk API payload has no details.body")

    parser = _Blocks()
    parser.feed(body)

    pages: list[Page] = []
    current: Page | None = None
    for kind, value in parser.blocks:
        if kind == "heading":
            current = Page(number=len(pages) + 1, heading=str(value))
            pages.append(current)
            continue
        if current is None:          # licence boilerplate ahead of the first heading
            continue
        if kind == "para":
            current.text = f"{current.text}\n{value}".strip()
        elif kind == "table":
            projected = project_year(value, doc.year_column)   # type: ignore[arg-type]
            if projected:
                current.tables.append(to_markdown(projected))

    # Keep source section numbers (no renumbering after filtering) so citations
    # and the manifest's page_ranges refer to the same sections in every year.
    wanted = set(doc.pages())
    return [p for p in pages if p.number in wanted and not p.is_empty()]


# ---------------------------------------------------------------------- entry --

def parse(doc: Doc) -> list[Page]:
    if doc.media == "pdf":
        return parse_pdf(doc)
    if doc.media == "govuk":
        return parse_govuk(doc)
    raise ValueError(f"{doc.doc_id}: unknown media {doc.media!r}")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Parse manifest documents into pages")
    ap.add_argument("--show", metavar="DOC_ID:PAGE", help="print one page in full")
    args = ap.parse_args(argv)

    docs = load_manifest()
    if args.show:
        doc_id, _, page_no = args.show.partition(":")
        doc = next(d for d in docs if d.doc_id == doc_id)
        page = next(p for p in parse(doc) if p.number == int(page_no))
        print(f"=== {doc.prefix(page.number)} ===")
        if page.heading:
            print(f"# {page.heading}")
        for md in page.tables:
            print(f"\n{md}\n")
        print(page.text)
        return 0

    for doc in docs:
        pages = parse(doc)
        tables = sum(len(p.tables) for p in pages)
        chars = sum(len(p.text) + sum(len(t) for t in p.tables) for p in pages)
        print(f"  {doc.doc_id:16s} {len(pages):3d} pages  {tables:3d} tables  {chars:7,d} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
