# Eval report - 12 of 12 gold questions, end to end

Generated 2026-09-22 06:29 UTC from `eval/gold_set.jsonl` through `agents.runner.run`, `ORCHESTRATOR=runner`, `ALLOW_FAST_MODEL=1`.

Reproduce:

```
uv run python -m ingest.build_index
uv run python -m eval.run_eval --label baseline
uv run python -m eval.run_eval --label diversify --diversify
uv run python -m eval.run_eval --followups
```

### Which code these numbers describe

| run | measured at | generated |
|---|---|---|
| `baseline` | `b812c33` | 2026-09-22 06:29 UTC |
| `diversify` | `850ae31` | 2026-09-21 06:05 UTC |
| `graph_cold` | `601e878` | 2026-09-24 05:01 UTC |
| `langchain_cold` | `a1b0625` | 2026-09-24 06:09 UTC |
| `followups` | `b812c33-dirty` | 2026-09-22 06:29 UTC |

`diversify.json` predate commit hashes in results files. **They were measured before two planner changes** - the region re-plan (q05) and the resolver-backed `year_not_in_corpus` refusal (q12) - so a difference on q05 or q12 against a later run is due to the code, not run-to-run variance.

## Metrics

| Metric | Definition | Result | Target | |
|---|---|---|---|---|
| Filter precision | dispatched `(country, tax_year_key)` == `expected_filters` | `0.875` | >= 0.9 | **FAIL** |
| Recall@8 | >=1 gold page retrieved, per dispatched sub-query | `1.000` | >= 0.8 | **PASS** |
| Answer contains | all expected figures present in the rendered answer | `0.875` | >= 0.8 | **PASS** |
| Grounding violations | numbers in the answer absent from the passage cited for them | `0` | 0 | **PASS** |
| Behavioural accuracy | correct clarify/refuse on the 4 behavioural questions | `4/4` | 4/4 | **PASS** |
| p50 latency (cold) | wall-clock per question, cache cold | `2.5 s` | < 15 s | **PASS** |

### No LLM judge, and why

Every metric above is exact-match or set-membership: filter precision compares resolved `(country, tax_year_key)` pairs, recall checks `doc_id|page` against the gold pages, *answer contains* is a substring test, grounding is the deterministic numeric check, and behaviour is read off which node ended the trace. At n=12 an unvalidated judge would add variance without adding information, and it would also use the same per-minute token budget as the system under test. Judge-based groundedness would be a next step alongside a larger gold set.

## Latency: cold and cached

| pass | p50 | max | note |
|---|---|---|---|
| cold (`LLM_CACHE=0`) | **2.5 s** | 16.9 s | every call a miss |
| cached | **0.08 s** | 0.23 s | the same 12 questions, replayed from `data/.llm_cache` |

**The budget is met per turn, not per minute.** Groq's free tier meters 8000 tokens/minute *per model*, and one three-branch turn costs ~14k split across two buckets. Two cold turns do not fit in one minute: the same query measured 6.9 s on a rested bucket and 22.5 s right after another cold turn, the difference being backoff. So this harness **paces itself** - `TokenPacer` tracks each model's bucket and waits between turns until the next one fits. It slept **263 s** across the cold pass, and that time is outside every measured turn. 

## Per question

| id | type | behaviour | filters | recall@8 | figures | grounding | cold |
|---|---|---|---|---|---|---|---|
| q01 | lookup | ok answer | ok | 1/1 | ok | 1/1 | 3.2s |
| q02 | lookup | ok answer | ok | 1/1 | ok | 1/1 | 2.2s |
| q03 | lookup | ok answer | ok | 1/1 | ok | 9/9 | 3.1s |
| q04 | lookup | ok answer | ok | 1/1 | ok | 38/38 | 3.5s |
| q05 | compare_years | ok answer | ok | 2/2 | ok | 4/4 | 5.9s |
| q06 | compare_years | ok answer | ok | 2/2 | ok | 2/2 | 16.9s |
| q07 | compare_countries | **X** refuse (want answer) | **X** | - | missing ['14,600', '12,570'] | 0/0 | 1.5s |
| q08 | compare_countries | ok answer | ok | 2/2 | ok | 3/3 | 2.6s |
| q09 | needs_clarification | ok clarify | - | - | - | - | 1.3s |
| q10 | needs_clarification | ok clarify | - | - | - | - | 1.0s |
| q11 | out_of_scope | ok refuse | - | - | - | - | 2.4s |
| q12 | out_of_scope | ok refuse | - | - | - | - | 0.9s |

## Filter precision is the planner's score

`eval/retrieval_eval.py` scores retrieval **in isolation**, using the sub-query text each gold `expected_filters` entry carries - what a competent planner ought to emit. It measures recall@8 **0.833** (lookups 4/4). This harness re-scores the same gold pages using the sub-queries the planner **actually emitted**, and the two harnesses stay separate because the gap between them is the planner's contribution.

| harness | sub-query text | recall@8 |
|---|---|---|
| `retrieval_eval.py` | gold (what a good planner should write) | 0.833 |
| `run_eval.py`, attempt 1 | the planner's own, first try | 0.800 |
| `run_eval.py`, final pack | the planner's own, after <=2 attempts | **1.000** |

Row 2 minus row 1 is the planner. Row 3 minus row 2 is the retry loop: **3** of 10 sub-queries went to a second attempt and **2** of those recovered a gold page they had missed. Filters are compared as resolved `(country, tax_year_key)` pairs and never as question text - Groq at temperature 0 is not reproducible, so the phrasing varies run to run while the routed filter does not.

## Grounding violations, by kind

58/58 checkable figures verified against the passage cited **for that claim**, across the 7 answered questions. 0 violation(s).

| kind | count | what it means | where the fix goes |
|---|---|---|---|
| `derived_arithmetic` | 0 | the figure is a sum or difference of figures that *are* cited - the model did arithmetic | the synthesizer prompt |
| `invented_figure` | 0 | nothing in the cited passages produces it | a bug report |
| `unknown_citation` | 0 | the claim cites a `chunk_id` that is not in the evidence pack | the synthesizer prompt |

The split is deterministic, not judged: `guardrails/numeric_grounding.py` pairs up the figures in the cited passages and reports which two produce the flagged number. For example, an early answer said the deduction "increased by $750" - correct arithmetic across two tables, but stated in no passage - and the fix was an instruction in `synthesize_system.md`.

## Behaviour: the four questions that must not be answered

| id | expected | observed | reason / question asked | reason correct |
|---|---|---|---|---|
| q09 | clarify | ok clarify | Which tax year would you like the standard deduction information for? | yes |
| q10 | clarify | ok clarify | Are you asking about the tax band for England/Wales/Northern Ireland or for Scotland? | yes |
| q11 | refuse | ok refuse | personalised_advice | yes |
| q12 | refuse | ok refuse | year_not_in_corpus | yes |

The behavioural target is 4/4 on clarify-vs-refuse: **4/4**. The last column is a stricter check reported separately (refusing q12 for the wrong reason would still pass the behavioural target). It scores 4/4.

## Page diversification, measured end to end

`diversify_by_page` collapses the top-8 to one chunk per (doc, page). In the retrieval-only eval it raised recall@8 from 0.833 to 0.917, but it is off by default because the duplicate it drops is the **prose** copy of a US table page, which carries qualifiers the table does not ("but not more than the regular standard deduction amount"). The table below compares the saved runs end to end.

| metric | `baseline` | `diversify` | `graph_cold` | `langchain_cold` |
|---|---|---|---|---|
| filter_precision | 0.875 | 1.000 | 1.000 | 1.000 |
| recall@8_attempt1 | 0.800 | 0.833 | 0.833 | 0.833 |
| recall@8 | 1.000 | 1.000 | 1.000 | 1.000 |
| answer_contains | 0.875 | 1.000 | 1.000 | 1.000 |
| grounding_violations | 0 | 0 | 0 | 0 |
| behavioural_correct | 4 | 4 | 4 | 4 |
| retries_fired | 3 | 3 | 3 | 3 |

- `baseline` cold p50 2.5 s, cached p50 0.08 s, violations by kind: {}

- `diversify` cold p50 3.0 s, cached p50 0.07 s, violations by kind: {}

- `graph_cold` cold p50 2.8 s, cached p50 0.08 s, violations by kind: {}

- `langchain_cold` cold p50 2.9 s, cached p50 0.09 s, violations by kind: {}

### Like for like: only what the flag can move

Behaviour differed between the runs on `q07` (baseline: refuse / diversify: answer / graph_cold: answer / langchain_cold: answer), `q12` (baseline: refuse, `year_not_in_corpus` / diversify: refuse, `other` / graph_cold: refuse, `year_not_in_corpus` / langchain_cold: refuse, `year_not_in_corpus`). That is decided by the planner, before retrieval runs, so it is run-to-run variance or the planner code between the commits above and not the flag, and it is where the difference in filter precision and answer-contains above comes from.

Over the 7 questions answered in every run (`q01`, `q02`, `q03`, `q04`, `q05`, `q06`, `q08`):

| | `baseline` | `diversify` | `graph_cold` | `langchain_cold` |
|---|---|---|---|---|
| recall@8, attempt 1 | 8/10 | 8/10 | 8/10 | 8/10 |
| recall@8, final | 10/10 | 10/10 | 10/10 | 10/10 |
| answer contains | 7/7 | 7/7 | 7/7 | 7/7 |
| figures grounded | 58/58 | 57/57 | 53/53 | 50/50 |
| retries fired | 3 | 3 | 3 | 3 |
| duplicate-page slots in final packs | 26/80 | 0/75 | 26/80 | 26/80 |

**Decision: `DIVERSIFY_BY_PAGE` stays off.** The bar for flipping it was recall or answer-contains improving, with 0 grounding violations and no answer losing a qualifier it had. Like for like, nothing improved: the same sub-queries hit on attempt 1 and after retry, and every answer that contained its figures still does. In the original pair of runs, q01's final answer was identical and q07's differed only in the year-assumption note, so no qualifier was lost - but none of the 12 gold questions depends on a prose-only qualifier, so the eval cannot measure that risk either. The flag frees duplicate slots for other pages without moving any metric here, so the default stays unchanged.

## Follow-ups: two turns, the second one scored

`eval/followups.jsonl` (1 row(s), separate from the twelve). Turn 1 is a gold question; turn 2 is a fragment such as *"and what about 2023?"* run with whatever `agents.carry.carry` took from turn 1 - the same function the app uses. Only turn 2 is scored, with the same scorer as the gold set. It was added after an early follow-up was answered with *"which country?"*.


| id | turn 1 | carried | turn 2 | filters | figures | grounding |
|---|---|---|---|---|---|---|
| f01 | q01 answer | US 2024 | answer | ok | ok | 1/1 |

Filter precision 1.000, answer contains 1.000, 0 grounding violation(s), over 1 of 1 attempted. Measured at `b812c33-dirty`, cold.

## Token spend

65470 tokens across the cold pass, metered in two buckets of 8000/minute. Per turn:

| id | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` | total |
|---|---|---|---|
| q01 | 3720 | 2169 | 5889 |
| q02 | 3385 | 1739 | 5124 |
| q03 | 3718 | 1806 | 5524 |
| q04 | 4306 | 2266 | 6572 |
| q05 | 4209 | 5417 | 9626 |
| q06 | 4989 | 8965 | 13954 |
| q07 | 2244 | 0 | 2244 |
| q08 | 4284 | 4047 | 8331 |
| q09 | 1982 | 0 | 1982 |
| q10 | 1971 | 0 | 1971 |
| q11 | 2260 | 0 | 2260 |
| q12 | 1993 | 0 | 1993 |

