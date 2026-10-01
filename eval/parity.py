"""Parity check: run both orchestrators over the gold set from the LLM cache and diff.

`uv run python -m eval.parity`

Replaying cached completions removes model randomness, so any difference in the
final state comes from the orchestrator. The whole final state is compared, minus
timings and the runner-only `fan_out` trace row.

`GROQ_API_KEY` is replaced with a dummy value, so a cache miss fails instead of
making a paid call. `--live-key` keeps the real key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from pydantic import BaseModel

REPLAY_KEY = "parity-replay-no-network"


def normalise(value: Any) -> Any:
    """Plain JSON-ish data, with every `*_ms` key and `api_ms` dropped."""
    if isinstance(value, BaseModel):
        return normalise(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()
                if not (isinstance(k, str) and k.endswith("_ms"))}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def comparable(state: dict) -> dict:
    state = dict(state)
    state.pop("timings", None)
    state["trace"] = [row for row in state.get("trace") or [] if row.get("step") != "fan_out"]
    return normalise(state)


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    """Path to the first difference between two normalized states, or None."""
    if type(a) is not type(b):
        return f"{path or '<root>'}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}: only in {'runner' if key in a else 'graph'}"
            found = first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} items vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


async def one_turn(run, question: str, previous: str | None, today) -> dict:
    from agents.carry import carry

    carried = None
    if previous is not None:
        carried = carry(await run(previous, today=today))
    return await run(question, today=today, carried_filters=carried)


async def replay(rows: list[dict], gold_by_id: dict[str, dict]) -> list[dict]:
    from agents.graph import run as graph_run
    from agents.runner import run as runner_run
    from eval.run_eval import EVAL_TODAY, score_question

    out = []
    for row in rows:
        previous = row.get("previous_turn")
        previous_q = gold_by_id[previous]["question"] if previous else None
        states, errors = {}, {}
        for name, run in (("runner", runner_run), ("graph", graph_run)):
            try:
                states[name] = await one_turn(run, row["question"], previous_q, EVAL_TODAY)
            except Exception as exc:  # noqa: BLE001 - a miss under the dummy key lands here
                errors[name] = f"{type(exc).__name__}: {exc}"
        result = {"id": row["id"], "errors": errors}
        if not errors:
            a, b = comparable(states["runner"]), comparable(states["graph"])
            result["difference"] = first_difference(a, b)
            result["scores"] = {name: score_question(row, s) for name, s in states.items()}
            result["fan_out_rows"] = {
                name: sum(1 for r in s.get("trace") or [] if r.get("step") == "fan_out")
                for name, s in states.items()
            }
        out.append(result)
        verdict = ("ERROR " + "; ".join(f"{k}: {v[:120]}" for k, v in errors.items()) if errors
                   else "identical" if result["difference"] is None
                   else f"DIFFERS at {result['difference']}")
        behaviour = (result["scores"]["runner"].observed_behavior if not errors else "-")
        print(f"  {row['id']:<4} {behaviour:<8} {verdict}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live-key", action="store_true",
                    help="Keep the real GROQ_API_KEY, so a cache miss is filled (and paid).")
    ap.add_argument("--only", help="Comma-separated gold/follow-up ids.")
    args = ap.parse_args(argv)

    from agents.run import use_utf8_stdout, warm_up

    use_utf8_stdout()
    if not args.live_key:
        os.environ["GROQ_API_KEY"] = REPLAY_KEY
    os.environ["LLM_CACHE"] = "1"

    from core.llm import reset_client
    from eval.retrieval_eval import load_gold
    from eval.run_eval import aggregate, load_followups

    reset_client()
    warm_up()

    gold = load_gold()
    followups = load_followups()
    rows = gold + followups
    if args.only:
        wanted = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in wanted]
    gold_by_id = {g["id"]: g for g in gold}

    print(f"replaying {len(rows)} turns through both runners "
          f"({'real key' if args.live_key else 'dummy key: a cache miss errors, never spends'}):")
    results = asyncio.run(replay(rows, gold_by_id))

    clean = [r for r in results if not r["errors"]]
    same = [r for r in clean if r["difference"] is None]
    print(f"\n{len(same)}/{len(results)} turns identical, "
          f"{len(results) - len(clean)} errored")

    gold_ids = {g["id"] for g in gold}
    for name in ("runner", "graph"):
        scores = [r["scores"][name] for r in clean if r["id"] in gold_ids]
        if not scores:
            continue
        agg = aggregate(scores)
        print(f"  {name:<6} filter precision {agg['filter_precision']:.3f} | "
              f"recall@8 {agg['recall@8']:.3f} | contains {agg['answer_contains']:.3f} | "
              f"violations {agg['grounding_violations']} | "
              f"behaviour {agg['behavioural_correct']}/{agg['behavioural_n']} "
              f"(n={len(scores)})")
    rows_by_runner = {name: sum(r["fan_out_rows"][name] for r in clean)
                      for name in ("runner", "graph")}
    print(f"  fan_out trace rows (dropped before the diff): {json.dumps(rows_by_runner)}")

    return 0 if len(same) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
