"""Prompt loading and rendering.

Prompts are stored as `.md` files in `string.Template` syntax (`$name`) and
rendered through LangChain's `ChatPromptTemplate`. Rendered output must stay
byte-identical to `Template.substitute`, because it is part of the LLM cache key
(see `tests/test_prompts.py`).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from string import Template

from langchain_core.prompts import ChatPromptTemplate

PROMPT_DIR = Path(__file__).resolve().parent


@cache
def load(name: str) -> str:
    """Read `<name>.md` from this directory."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"no prompt {name!r} in {PROMPT_DIR}")
    return path.read_text(encoding="utf-8").strip()


def _as_fstring(text: str) -> str:
    """Convert a `$name` template to the f-string syntax `ChatPromptTemplate` expects.

    `$$` becomes `$`, a `$` not followed by an identifier (e.g. `$14,600`) stays
    literal, and literal braces are doubled.
    """
    out, pos = [], 0
    for match in Template.pattern.finditer(text):
        out.append(text[pos:match.start()].replace("{", "{{").replace("}", "}}"))
        name = match.group("named") or match.group("braced")
        out.append("{" + name + "}" if name else "$")
        pos = match.end()
    out.append(text[pos:].replace("{", "{{").replace("}", "}}"))
    return "".join(out)


@cache
def template(name: str) -> ChatPromptTemplate:
    """`<name>.md` as a single-message `ChatPromptTemplate`; role is taken from the name."""
    role = "system" if name.endswith("_system") else "human"
    return ChatPromptTemplate.from_messages([(role, _as_fstring(load(name)))])


def render(name: str, **values: object) -> str:
    """Load and fill a prompt. Raises `KeyError` if a placeholder is missing."""
    return template(name).format_messages(**values)[0].content
