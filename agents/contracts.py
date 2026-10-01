"""Pydantic models passed between nodes.

The plan is split in two so the LLM never produces a tax_year_key:

    DraftPlan  - what the LLM returns: `country` + the raw `year_text`.
        |
        |  agents.planner.resolve_plan()  - deterministic, calls
        v                                   core.year_resolver.resolve()
    Plan       - what the graph routes on, with resolved year keys.

Docstrings and field descriptions on the LLM-facing models (PlannedSubQuery,
DraftPlan, Sufficiency, AnswerClaim) are part of the prompt via the JSON schema.
"""

from typing import Literal

from pydantic import BaseModel, Field

Country = Literal["US", "UK"]  # widen when the corpus widens

Intent = Literal[
    "lookup",
    "compare_years",
    "compare_countries",
    "out_of_scope",
    "needs_clarification",
]

# Why a refusal happened. `refuse_node` renders a message for each, and the eval
# scores against these values.
RefusalReason = Literal[
    "personalised_advice",
    "year_not_in_corpus",
    "country_not_in_corpus",
    "unresolvable_year",
    "insufficient_evidence",
    "other",
]


# ----------------------------------------------------------- the LLM's output --

class PlannedSubQuery(BaseModel):
    """One leg of the plan, as the PLANNER writes it. No tax_year_key - see above."""

    question: str = Field(description="A self-contained retrieval question for ONE country "
                                      "and ONE tax year.")
    country: Country
    year_text: str = Field(description="The tax year exactly as the user expressed it "
                                       "('2024', '2024-25', 'last year'). Never a number you "
                                       "computed yourself.")
    year_assumption: str | None = Field(
        default=None,
        description="One short SENTENCE explaining where this year came from, set only "
                    "if the user did not state it for this leg - e.g. 'You did not give "
                    "a year for the UK figure, so I used the most recent year in the "
                    "question.' Never just the year itself; this sentence is printed to "
                    "the user verbatim.",
    )


class DraftPlan(BaseModel):
    """The structured-output schema. This is the object `.with_structured_output()` fills."""

    intent: Intent
    sub_queries: list[PlannedSubQuery] = Field(default_factory=list)
    clarifying_question: str | None = Field(
        default=None,
        description="Exactly one question, set only when intent is needs_clarification.",
    )
    out_of_scope_reason: str | None = Field(
        default=None, description="One sentence, set only when intent is out_of_scope."
    )


# ---------------------------------------------------- the resolved, routed plan --

class SubQuery(BaseModel):
    """One country and one resolved tax year, ready for retrieval."""

    question: str
    country: Country
    tax_year_key: int          # from core.year_resolver.resolve(), never from the LLM
    tax_year_label: str        # display form, e.g. "2024-25"
    year_text: str = ""        # the user's original wording, for the trace
    assumption_note: str | None = None


class Plan(BaseModel):
    intent: Intent
    sub_queries: list[SubQuery] = Field(default_factory=list)
    clarifying_question: str | None = None
    assumption_notes: list[str] = Field(default_factory=list)
    refusal_reason: RefusalReason | None = None
    refusal_detail: str | None = None


# ------------------------------------------------------------------- evidence --

class Passage(BaseModel):
    chunk_id: str
    text: str
    country: Country
    tax_year_label: str
    doc_id: str
    doc_title: str
    page: int                        # 1-based PDF index, not the printed page number
    printed_page: int | None = None  # None for web documents (UK)
    url: str = ""
    score: float


class Sufficiency(BaseModel):
    """The search agent's self-check. Strict by construction: it has to NAME the passage.

    `answer_chunk_id` is what stops this degenerating into a rubber stamp. A check
    that only returns a boolean will say "yes, this looks relevant" about eight
    chunks of filing-status prose; a check that must point at the chunk containing
    the figure cannot.

    `suggested_query` folds the rewrite into the same call. It began as a second
    LLM call that re-read the same passages to propose a new query, and that was
    both wasteful and worse: on a free tier metered at 8000 tokens/minute a
    three-way fan-out cannot afford two calls per branch, and the model that has
    just read the pack is better placed to say what to search for than one handed
    the pack again a round-trip later. The loop is unchanged - retrieve, judge,
    re-retrieve - and it is now one round-trip shorter per retry.
    """

    sufficient: bool
    answer_chunk_id: str | None = Field(
        default=None,
        description="The chunk_id of the passage that actually contains the answer. "
                    "Required when sufficient is true. If you cannot name one, "
                    "sufficient is false.",
    )
    reason: str = Field(description="One sentence. When insufficient, say what is missing.")
    missing: str | None = Field(
        default=None, description="The specific fact or table that was absent."
    )
    suggested_query: str | None = Field(
        default=None,
        description="Required when sufficient is false: the search query to try "
                    "instead, in the source document's own vocabulary. Null otherwise.",
    )


class EvidencePack(BaseModel):
    sub_query: SubQuery
    passages: list[Passage]
    sufficient: bool
    reason: str
    attempts: int
    queries_tried: list[str] = Field(default_factory=list)
    answer_chunk_id: str | None = None


# ------------------------------------------------------------------- answering --

class AnswerClaim(BaseModel):
    """One sentence of the answer, bound to the passages it came from.

    The binding is the whole point.
    '29,200' is the 2024 married-filing-jointly standard deduction AND the 2023
    filing threshold for a 65-or-older spouse, and both are in the corpus - so a
    grounding check that asks "does this number appear anywhere in the evidence
    pack?" passes a wrong answer holding a real citation. Numbers are checked
    against THIS claim's `chunk_ids`, nothing wider.
    """

    text: str = Field(description="One sentence of the answer. Every figure in it must come "
                                  "from the passages you list in chunk_ids.")
    chunk_ids: list[str] = Field(default_factory=list,
                                 description="chunk_ids of the passages supporting THIS sentence.")


class DraftAnswer(BaseModel):
    claims: list[AnswerClaim]
    caveats: list[str] = Field(default_factory=list)
