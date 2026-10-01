"""Prompt files render through ChatPromptTemplate exactly as `string.Template` would.

Rendered prompts are part of the LLM cache key, so any byte difference would
invalidate every cached completion. The test values deliberately include JSON
braces, dollar amounts, and placeholder-like text.
"""

from __future__ import annotations

from string import Template

import pytest
from langchain_core.prompts import ChatPromptTemplate

from agents.prompts import PROMPT_DIR, load, render, template

PROMPTS = sorted(p.stem for p in PROMPT_DIR.glob("*.md"))

HOSTILE = ('{"amount": "$14,600", "nested": {"k": [1, 2]}} {{x}} }{ '
           "$$ $question ${carried} {question} $14,600 £12,570 \\n $")


def placeholders(text: str) -> set[str]:
    return {m.group("named") or m.group("braced")
            for m in Template.pattern.finditer(text)
            if m.group("named") or m.group("braced")}


def test_there_are_six_prompts():
    assert PROMPTS == [
        "planner_system", "planner_user",
        "sufficiency_system", "sufficiency_user",
        "synthesize_system", "synthesize_user",
    ]


@pytest.mark.parametrize("name", PROMPTS)
def test_render_is_byte_identical_to_string_template(name):
    text = load(name)
    values = {key: f"<{key}> {HOSTILE}" for key in placeholders(text)}

    rendered = render(name, **values)

    # `safe_substitute` differs from `substitute` only on a bare `$` (e.g. "$750"
    # in synthesize_system), where `substitute` raises and ChatPromptTemplate keeps it.
    assert rendered == Template(text).safe_substitute(**values)
    try:
        substituted = Template(text).substitute(**values)
    except ValueError:
        # System prompts with bare `$` have no placeholders and are used as-is.
        assert not values and rendered == text
    else:
        assert rendered == substituted


@pytest.mark.parametrize("name", PROMPTS)
def test_each_prompt_is_a_chat_prompt_template_with_its_role(name):
    tmpl = template(name)
    assert isinstance(tmpl, ChatPromptTemplate)
    assert set(tmpl.input_variables) == placeholders(load(name))
    role = tmpl.format_messages(**{k: "" for k in tmpl.input_variables})[0].type
    assert role == ("system" if name.endswith("_system") else "human")


def test_a_missing_placeholder_still_fails_loudly():
    with pytest.raises(KeyError):
        render("planner_user", question="What is the 2024 standard deduction?")
