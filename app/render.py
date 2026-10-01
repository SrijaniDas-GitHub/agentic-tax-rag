"""Text helpers for the UI, separate from `app/main.py` so tests can import them
without starting Streamlit."""

from __future__ import annotations


def no_math(text: str) -> str:
    """Escape `$` so Streamlit renders dollar amounts instead of LaTeX.

    `st.markdown` treats text between two `$` signs as math, so any answer with
    two dollar figures would otherwise render incorrectly.
    """
    return text.replace("$", "\\$")


def without_sources(text: str) -> str:
    """Drop the `Sources:` paragraph from a rendered answer.

    The UI lists the same sources as linked chips under the answer, so showing
    the paragraph too prints every source twice. The paragraph stays in
    `final_answer` for the CLI and the eval.
    """
    return "\n\n".join(p for p in text.split("\n\n") if not p.startswith("Sources:\n"))
