"""How much evidence each LLM call is shown.

Groq's free tier caps a single request at 8000 tokens (`core.llm.TPM_LIMIT`).
`EvidencePack` keeps every retrieved passage for the trace, citations and
grounding check; prompts get a trimmed subset sized from the limits below.

Sizes are in characters (~4 per token) so trimming is deterministic; actual
token counts are recorded in the trace.
"""

from __future__ import annotations

import re

from agents.contracts import EvidencePack, Passage

_HAS_DIGIT = re.compile(r"\d")

CHARS_PER_TOKEN = 4          # rule of thumb; the trace records measured counts

# The sufficiency check sees figure-bearing lines (see `figure_excerpt`) rather
# than a plain prefix, since table rows often sit past a short cut-off. Fewer
# passages or shorter excerpts caused it to miss figures that were in the pack.
SUFFICIENCY_PASSAGES = 4
SUFFICIENCY_CHARS = 700

# The synthesizer gets fewer, longer passages, with the self-check's chosen
# answer passage first (see `for_synthesis`).
SYNTHESIS_PASSAGES = 2
SYNTHESIS_CHARS = 850

# Completion cap for the sufficiency check (four short fields).
SUFFICIENCY_MAX_COMPLETION = 450


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def figure_excerpt(text: str, chars: int) -> str:
    """Keep the chunk's prefix line plus the lines that contain digits.

    The prefix identifies the document and page. A passage with no digits falls
    back to its opening text.
    """
    lines = text.splitlines()
    if not lines:
        return text[:chars]

    head, rest = lines[0], lines[1:]
    kept = [line for line in rest if _HAS_DIGIT.search(line)]
    if not kept:
        return text[:chars]

    out, used, elided = [head], len(head), False
    for line in kept:
        if used + len(line) + 1 > chars:
            elided = True
            break
        out.append(line)
        used += len(line) + 1
    if elided or len(kept) < len(rest):
        out.append("[...lines without figures omitted]")
    return "\n".join(out)


def for_synthesis(pack: EvidencePack,
                  limit: int = SYNTHESIS_PASSAGES) -> list[Passage]:
    """Passages for the synthesizer: the self-check's answer passage first, then by rank.

    The top-ranked passages are often duplicates of the same page, so the passage
    the self-check identified is a better first choice than raw rank.
    """
    if not pack.answer_chunk_id:
        return pack.passages[:limit]
    named = [p for p in pack.passages if p.chunk_id == pack.answer_chunk_id]
    rest = [p for p in pack.passages if p.chunk_id != pack.answer_chunk_id]
    return (named + rest)[:limit]
