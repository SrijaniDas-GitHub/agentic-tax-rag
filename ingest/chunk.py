"""Chunk parsed pages into data/chunks.jsonl.

Chunks are 450 tokens with 60 overlap. bge-small-en-v1.5 has a 512-token limit
and truncates longer input silently, so `assert_fits` fails the build if any
chunk (including its contextual prefix) is over the limit.

Chunks are assembled from the original text, not decoded from tokens, because the
tokenizer is uncased and would alter figures like "$14,600".

    uv run python -m ingest.chunk
    uv run python -m ingest.chunk --grep 14,600
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from ingest.manifest import CHUNKS_PATH, Doc, load_manifest
from ingest.parse import Page, parse

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

TARGET_TOKENS = 450      # leaves headroom under the model's 512-token limit
OVERLAP_TOKENS = 60
SPECIAL_TOKENS = 2       # [CLS] ... [SEP], charged to every sequence

_tok = None


def tokenizer():
    """The embedding model's tokenizer, so token counts match what gets embedded."""
    global _tok
    if _tok is None:
        import transformers
        from transformers import AutoTokenizer

        # Measuring oversize units before splitting triggers a length warning;
        # assert_fits is the real check, so silence it.
        transformers.logging.set_verbosity_error()
        _tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    return _tok


def n_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


def max_seq_length() -> int:
    return int(tokenizer().model_max_length)


class ChunkTooLong(RuntimeError):
    """A chunk exceeds the embedding model's token limit."""


def assert_fits(chunk_id: str, text: str) -> int:
    """Hard gate. Returns the true token count including special tokens."""
    total = len(tokenizer()(text, add_special_tokens=True)["input_ids"])
    limit = max_seq_length()
    if total > limit:
        raise ChunkTooLong(
            f"{chunk_id}: {total} tokens > {EMBED_MODEL} limit of {limit}.\n"
            f"  This would be truncated SILENTLY at embed time - degraded recall with "
            f"no error anywhere.\n"
            f"  Lower TARGET_TOKENS, or split the offending table/paragraph.\n"
            f"  First 200 chars: {text[:200]!r}"
        )
    return total


# ------------------------------------------------------------------- packing --

def _split_oversize(unit: str, budget: int) -> list[str]:
    """Break a single unit that cannot fit, on word boundaries."""
    words = unit.split(" ")
    out, cur, cur_n = [], [], 0
    for w in words:
        n = n_tokens(w + " ")
        if cur and cur_n + n > budget:
            out.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(w)
        cur_n += n
    if cur:
        out.append(" ".join(cur))
    return out


def pack(units: list[str], budget: int, overlap: int) -> list[str]:
    """Greedily fill chunks up to `budget` tokens, carrying `overlap` tokens back."""
    sized: list[tuple[str, int]] = []
    for u in units:
        u = u.strip()
        if not u:
            continue
        n = n_tokens(u)
        if n > budget:
            sized.extend((part, n_tokens(part)) for part in _split_oversize(u, budget))
        else:
            sized.append((u, n))

    chunks: list[str] = []
    cur: list[tuple[str, int]] = []
    cur_n = 0
    for text, n in sized:
        if cur and cur_n + n > budget:
            chunks.append("\n".join(t for t, _ in cur))
            tail: list[tuple[str, int]] = []
            tail_n = 0
            for t, tn in reversed(cur):
                if tail_n + tn > overlap:
                    break
                tail.insert(0, (t, tn))
                tail_n += tn
            cur, cur_n = tail, tail_n
        cur.append((text, n))
        cur_n += n
    if cur:
        chunks.append("\n".join(t for t, _ in cur))
    return chunks


def split_table(md: str, budget: int) -> list[str]:
    """Split an oversized markdown table by rows, repeating the header in each part."""
    lines = md.splitlines()
    if len(lines) < 3:
        return [md]
    head, sep, body = lines[0], lines[1], lines[2:]
    head_n = n_tokens(f"{head}\n{sep}")

    parts, cur, cur_n = [], [], 0
    for row in body:
        n = n_tokens(row)
        if cur and head_n + cur_n + n > budget:
            parts.append("\n".join([head, sep, *cur]))
            cur, cur_n = [], 0
        cur.append(row)
        cur_n += n
    if cur:
        parts.append("\n".join([head, sep, *cur]))
    return parts


# -------------------------------------------------------------------- chunks --

def chunk_page(doc: Doc, page: Page) -> list[dict]:
    prefix = doc.prefix(page.number)
    budget = TARGET_TOKENS - n_tokens(prefix) - SPECIAL_TOKENS

    bodies: list[tuple[str, bool]] = []

    # Each table gets its own chunk, prefixed with the section heading. The England
    # and Scotland rate tables have identical structure; only the heading tells
    # them apart.
    caption = f"{page.heading}\n" if page.heading else ""
    cap_n = n_tokens(caption)
    for md in page.tables:
        parts = split_table(md, budget - cap_n) if n_tokens(md) + cap_n > budget else [md]
        bodies.extend((f"{caption}{part}", True) for part in parts)

    prose = page.text
    if page.heading:
        prose = f"{page.heading}\n{prose}".strip()
    if prose:
        # Split on blank lines if present, otherwise on single lines (PDF text).
        units = re.split(r"\n\s*\n", prose) if "\n\n" in prose else prose.splitlines()
        bodies.extend((body, False) for body in pack(units, budget, OVERLAP_TOKENS))

    out = []
    for i, (body, has_table) in enumerate(bodies):
        chunk_id = f"{doc.doc_id}:{page.number}:{i}"
        text = f"{prefix}\n\n{body}"
        total = assert_fits(chunk_id, text)
        out.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                # Metadata used by the Chroma `where=` filter.
                "country": doc.country,
                "tax_year_key": doc.tax_year_key,
                "tax_year_label": doc.tax_year_label,
                "doc_id": doc.doc_id,
                "doc_title": doc.doc_title,
                "page": page.number,
                "printed_page": doc.printed_page(page.number),
                "url": doc.url,
                "has_table": has_table,
                "n_tokens": total,
            }
        )
    return out


def build(docs: list[Doc] | None = None) -> list[dict]:
    docs = docs or load_manifest()
    all_chunks: list[dict] = []
    print(f"  embedding model {EMBED_MODEL}, max_seq_length {max_seq_length()}")
    print(f"  target {TARGET_TOKENS} tokens, overlap {OVERLAP_TOKENS}\n")

    for doc in docs:
        pages = parse(doc)
        chunks = [c for p in pages for c in chunk_page(doc, p)]
        all_chunks.extend(chunks)
        tabled = sum(c["has_table"] for c in chunks)
        longest = max((c["n_tokens"] for c in chunks), default=0)
        print(
            f"  {doc.doc_id:16s} {len(chunks):4d} chunks "
            f"({tabled:3d} table, {len(chunks) - tabled:3d} prose)  "
            f"{len(pages):3d} pages  longest {longest:3d} tok"
        )

    print(f"\n  {'TOTAL':16s} {len(all_chunks):4d} chunks")
    return all_chunks


def write(chunks: list[dict]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    size = CHUNKS_PATH.stat().st_size
    print(f"  wrote {CHUNKS_PATH.relative_to(CHUNKS_PATH.parent.parent)} ({size / 1e6:.2f} MB)")


def load() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Build data/chunks.jsonl from the manifest")
    ap.add_argument("--grep", metavar="NEEDLE", help="search an existing chunks.jsonl instead of rebuilding")
    args = ap.parse_args(argv)

    if args.grep:
        hits = [c for c in load() if args.grep in c["text"]]
        print(f"  {len(hits)} chunk(s) contain {args.grep!r}")
        for c in hits[:10]:
            print(f"    {c['chunk_id']:24s} {c['country']} TY{c['tax_year_label']:8s} "
                  f"page {c['page']:3d}  table={c['has_table']}")
        return 0 if hits else 1

    try:
        chunks = build()
    except ChunkTooLong as exc:
        print(f"\nBUILD FAILED\n{exc}", file=sys.stderr)
        return 1
    write(chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
