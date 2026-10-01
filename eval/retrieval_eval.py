"""Retrieval-only evaluation over the gold set: dense vs lexical vs hybrid.

No LLM calls. `eval/run_eval.py` covers the end-to-end evaluation.

- Scoring is per sub-query: a cross-year question counts once per year.
- `gold_pages` are strict: only the page stating the asked-about quantity counts.
- Reports recall@1/4/8, MRR, and the off-target rate (share of passages from the
  wrong country or year), with and without the metadata filter.

    uv run python -m eval.retrieval_eval
    uv run python -m eval.retrieval_eval --show-misses
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.paths import REPO_ROOT
from core.retrieval import TOP_K, search

GOLD_PATH = REPO_ROOT / "eval" / "gold_set.jsonl"

MODES = ("hybrid", "dense", "lexical")
KS = (1, 4, 8)
ANSWERABLE = "answer"


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sub_queries(gold: list[dict]) -> list[tuple[dict, dict]]:
    """(question, expected filter) pairs - the unit recall is scored on."""
    return [
        (q, filters)
        for q in gold
        if q["expected_behavior"] == ANSWERABLE
        for filters in q["expected_filters"]
    ]


def run_one(question: dict, filters: dict, mode: str, use_filters: bool, k: int,
            diversify: bool = False) -> dict:
    """Run one sub-query. Returns the rank of the first gold page and purity stats."""
    passages = search(
        filters.get("question", question["question"]),
        country=filters["country"] if use_filters else None,
        tax_year_key=filters["tax_year_key"] if use_filters else None,
        k=k,
        mode=mode,
        diversify_by_page=diversify,
    )
    gold = set(question["gold_pages"])
    pages = [f"{p.chunk_id.split(':')[0]}|{p.page}" for p in passages]
    rank = next((i + 1 for i, page in enumerate(pages) if page in gold), None)

    # Passages from a country or tax year the sub-query did not ask for. Always
    # zero with the filter on; reported to show the effect of removing it.
    want = (filters["country"], filters["tax_year_key"])
    off = sum(1 for p in passages if (p.country, _year_key(p)) != want)
    # Slots taken by a page already in the pack (table and prose copies of one page).
    repeats = len(pages) - len(set(pages))
    return {"rank": rank, "pages": pages, "off": off, "n": len(passages), "repeats": repeats}


def _year_key(passage) -> int:
    """tax_year_key from the label: "2024-25" -> 2024, "2024" -> 2024."""
    return int(passage.tax_year_label[:4])


def score(gold: list[dict], mode: str, use_filters: bool = True, k: int = TOP_K,
          diversify: bool = False) -> dict:
    pairs = sub_queries(gold)
    results = [(q, run_one(q, f, mode, use_filters, k, diversify)) for q, f in pairs]

    ranks = [r["rank"] for _, r in results]
    lookup_ranks = [r["rank"] for q, r in results if q["type"] == "lookup"]
    retrieved = sum(r["n"] for _, r in results)

    row = {
        "mode": mode + ("+1/page" if diversify else ""),
        "filters": use_filters,
        "n": len(pairs),
        "repeat_rate": sum(r["repeats"] for _, r in results) / retrieved if retrieved else 0.0,
        "mrr": sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0,
        "off_target": sum(r["off"] for _, r in results) / retrieved if retrieved else 0.0,
        "lookup_hits": sum(1 for r in lookup_ranks if r),
        "lookup_recall": sum(1 for r in lookup_ranks if r) / len(lookup_ranks),
        "lookup_n": len(lookup_ranks),
        "misses": [(q["id"], f, r["pages"]) for (q, f), (_, r) in zip(pairs, results) if not r["rank"]],
    }
    for cutoff in KS:
        row[f"recall@{cutoff}"] = sum(1 for r in ranks if r and r <= cutoff) / len(ranks)
    return row


def behavioural_counts(gold: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for q in gold:
        counts[q["expected_behavior"]] = counts.get(q["expected_behavior"], 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Recall@k and MRR: dense-only vs hybrid")
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("-k", type=int, default=TOP_K)
    args = ap.parse_args(argv)

    gold = load_gold()
    pairs = sub_queries(gold)
    designed = sum(1 for q in gold if q.get("first_attempt_should_fail")
                   for _ in q["expected_filters"])
    print(f"  {len(gold)} gold questions  {behavioural_counts(gold)}")
    print(f"  {len(pairs)} answerable sub-queries, k={args.k}")
    print(f"  {designed} of them are designed to fail attempt 1 (the retry case)\n")

    rows = [score(gold, mode, True, args.k) for mode in MODES]
    rows.append(score(gold, "hybrid", False, args.k))
    rows.append(score(gold, "hybrid", True, args.k, diversify=True))

    head = f"  {'mode':14s} {'filter':7s} " + " ".join(f"{'r@' + str(c):>7s}" for c in KS)
    print(head + f" {'MRR':>7s} {'off-tgt':>8s} {'repeats':>8s} {'lookups':>9s}")
    print("  " + "-" * (len(head) + 32))
    for r in rows:
        cells = " ".join(f"{r['recall@' + str(c)]:7.3f}" for c in KS)
        print(
            f"  {r['mode']:14s} {str(r['filters']).lower():7s} {cells} "
            f"{r['mrr']:7.3f} {r['off_target']:8.1%} {r['repeat_rate']:8.1%} "
            f"{r['lookup_recall']:5.3f}({r['lookup_hits']}/{r['lookup_n']})"
        )

    if args.show_misses:
        print()
        for r in rows:
            for qid, filters, pages in r["misses"]:
                print(f"  MISS {r['mode']:8s} filters={str(r['filters']).lower():5s} "
                      f"{qid} {filters['country']}/{filters['tax_year_key']} -> {pages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
