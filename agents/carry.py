"""Context carried from one turn to the next, shared by the app and the eval."""

from __future__ import annotations


def carry(state: dict) -> dict | None:
    """Return the previous turn's resolved filters and question for the next planner call.

    Lets follow-ups like "and what about 2023?" inherit country and subject.
    Clarifications and refusals carry nothing.
    """
    plan = state.get("plan")
    if not plan or not plan.sub_queries:
        return None
    if plan.intent in {"needs_clarification", "out_of_scope"}:
        return None
    return {
        "previous_question": state.get("user_query"),
        "countries": sorted({sq.country for sq in plan.sub_queries}),
        "tax_years": sorted({sq.tax_year_label for sq in plan.sub_queries}),
        "resolved": [
            {"country": sq.country, "tax_year_key": sq.tax_year_key,
             "tax_year_label": sq.tax_year_label}
            for sq in plan.sub_queries
        ],
    }
