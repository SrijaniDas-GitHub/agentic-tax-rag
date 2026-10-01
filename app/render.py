"""Text helpers for the UI, separate from `app/main.py` so tests can import them
without starting Streamlit."""

from __future__ import annotations


def no_math(text: str) -> str:
    """Escape `$` so Streamlit renders dollar amounts instead of LaTeX.

    `st.markdown` treats text between two `$` signs as math, so any answer with
    two dollar figures would otherwise render incorrectly.
    """
    return text.replace("$", "\\$")
