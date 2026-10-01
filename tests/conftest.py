"""Keep the suite offline whatever is in `.env`.

`.env` may set `ORCHESTRATOR=graph`, which routes the eval harness past tests that
patch `agents.runner.run` and into the real graph, and a real `GROQ_API_KEY`
would then spend tokens. Tests that need the graph set `ORCHESTRATOR` themselves.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR", "runner")
    monkeypatch.setenv("GROQ_API_KEY", "test-no-network")
