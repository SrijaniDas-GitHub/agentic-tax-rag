# Every target is one `uv run` line, so none of this needs make: the README lists
# the plain equivalents. Windows does not ship make; nothing here depends on it.

.PHONY: sync index app test eval followups retrieval report parity

sync:
	uv sync

index:
	uv run python -m ingest.build_index

app:
	uv run streamlit run app/main.py

test:
	uv run python -m pytest -q

# COLD: ~66k Groq tokens, a third of a free key's 200k/day. Not part of `test`.
eval:
	uv run python -m eval.run_eval --label baseline

# Two turns (q01, then "and what about 2023?"): ~12k tokens cold.
followups:
	uv run python -m eval.run_eval --followups

# Free: retrieval only, gold sub-queries, no LLM.
retrieval:
	uv run python -m eval.retrieval_eval

# Free: re-render eval/report.md from eval/results/*.json.
report:
	uv run python -m eval.run_eval --report-only

# Free: both runners (ORCHESTRATOR=runner|graph) over the gold set from the LLM cache,
# whole final state diffed. A dummy key is swapped in, so a cache miss errors, never spends.
parity:
	uv run python -m eval.parity
