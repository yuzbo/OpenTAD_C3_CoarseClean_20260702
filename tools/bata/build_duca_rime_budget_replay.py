from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


SCHEMA = "duca_rime_budget_replay_v1"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("budget replay input is empty")
    return rows


def _write_immutable(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different replay: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {
        "path": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "record_count": len(rows),
    }


def _key(row: Mapping[str, Any]) -> tuple[str, int]:
    video = str(row.get("video_id") or row.get("video_name") or "")
    start = int(row.get("window_start_frame", -1))
    if not video or start < 0:
        raise ValueError("budget record requires video_id and non-negative window_start_frame")
    return video, start


def _validate_candidate_budgets(candidate_budgets: Sequence[int]) -> tuple[int, ...]:
    budgets = tuple(int(value) for value in candidate_budgets)
    if (
        len(budgets) < 2
        or tuple(sorted(set(budgets))) != budgets
        or budgets[0] <= 0
    ):
        raise ValueError("candidate budgets must be positive, unique, and increasing")
    return budgets


def paired_same_k(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_budgets: Sequence[int],
) -> list[dict[str, Any]]:
    budgets = _validate_candidate_budgets(candidate_budgets)
    output = []
    seen = set()
    for row in rows:
        key = _key(row)
        if key in seen:
            raise ValueError("duplicate allocation key")
        seen.add(key)
        requested = int(row["requested_k"])
        if requested not in budgets:
            raise ValueError("paired replay requested_k is outside candidate_budgets")
        output.append(
            {
                "schema_version": SCHEMA,
                "video_id": key[0],
                "window_start_frame": key[1],
                "requested_k": requested,
                "provenance": {
                    "role": "paired_same_realized_cost_control",
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_prediction_cache": False,
                    "source_allocation_sha256": str(row.get("allocation_sha256", "")),
                },
            }
        )
    return output


def histogram_shuffle(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    candidate_budgets: Sequence[int],
) -> list[dict[str, Any]]:
    candidates = _validate_candidate_budgets(candidate_budgets)
    ordered = sorted(rows, key=_key)
    keys = [_key(row) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate allocation key")
    budgets = [int(row["requested_k"]) for row in ordered]
    if any(value not in candidates for value in budgets):
        raise ValueError("shuffle replay requested_k is outside candidate_budgets")
    shuffled = list(budgets)
    random.Random(int(seed)).shuffle(shuffled)
    if sorted(shuffled) != sorted(budgets):
        raise RuntimeError("histogram shuffle changed the K multiset")
    return [
        {
            "schema_version": SCHEMA,
            "video_id": key[0],
            "window_start_frame": key[1],
            "requested_k": budget,
            "provenance": {
                "role": "histogram_shuffled_budget_control",
                "seed": int(seed),
                "uses_gt": False,
                "uses_teacher": False,
                "uses_prediction_cache": False,
                "preserves_exact_k_histogram": True,
            },
        }
        for key, budget in zip(keys, shuffled)
    ]


def adaptok_test_batch_ilp(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_budgets: Sequence[int],
    candidate_costs: Sequence[int],
    target_mean_cost: float,
) -> list[dict[str, Any]]:
    budgets = _validate_candidate_budgets(candidate_budgets)
    costs = tuple(int(value) for value in candidate_costs)
    if (
        len(costs) != len(budgets)
        or any(value <= 0 for value in costs)
        or any(right <= left for left, right in zip(costs[:-1], costs[1:]))
        or not float(costs[0]) <= float(target_mean_cost) <= float(costs[-1])
    ):
        raise ValueError("AdapTok candidate budgets/costs are invalid")
    ordered = sorted(rows, key=_key)
    keys = [_key(row) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate AdapTok curve key")
    curves = []
    for row in ordered:
        provenance = row.get("provenance")
        if (
            row.get("schema_version") != "duca_adaptok_total_loss_curve_v1"
            or not isinstance(provenance, Mapping)
            or provenance.get("uses_gt") is not False
            or provenance.get("uses_teacher") is not False
            or provenance.get("test_batch_curve") is not True
        ):
            raise ValueError("AdapTok direct transfer requires clean test-batch loss curves")
        values = tuple(float(value) for value in row["predicted_total_loss"])
        if len(values) != len(budgets) or not all(math.isfinite(value) for value in values):
            raise ValueError("AdapTok total-loss curve must align with candidate budgets")
        curves.append(values)
    maximum = int(math.floor(float(target_mean_cost) * len(curves) + 1.0e-9))
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for curve in curves:
        next_states = {}
        for used, (loss, path) in states.items():
            for index, cost in enumerate(costs):
                total = used + cost
                if total > maximum:
                    continue
                candidate = (loss + curve[index], path + (index,))
                previous = next_states.get(total)
                if previous is None or candidate < previous:
                    next_states[total] = candidate
        if not next_states:
            raise RuntimeError("AdapTok ILP has no feasible batch allocation")
        states = next_states
    used, (_loss, path) = min(
        states.items(),
        key=lambda item: (item[1][0], -item[0], item[1][1]),
    )
    output = []
    for key, index in zip(keys, path):
        output.append(
            {
                "schema_version": SCHEMA,
                "video_id": key[0],
                "window_start_frame": key[1],
                "requested_k": budgets[index],
                "provenance": {
                    "role": "adaptok_total_loss_curve_test_batch_ilp",
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_prediction_cache": False,
                    "uses_test_batch_composition": True,
                    "deployment_candidate": False,
                    "direct_transfer_baseline": True,
                },
            }
        )
    realized = mean(costs[index] for index in path)
    if realized > float(target_mean_cost) + 1.0e-9 or used > maximum:
        raise RuntimeError("AdapTok ILP exceeded its mean-cost contract")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build immutable RIME K replay controls")
    parser.add_argument("--mode", choices=("paired", "shuffle", "adaptok"), required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--candidate-budgets", nargs="+", type=int)
    parser.add_argument("--candidate-costs", nargs="+", type=int)
    parser.add_argument("--target-mean-cost", type=float)
    args = parser.parse_args(argv)
    rows = _read_jsonl(args.input_jsonl)
    if not args.candidate_budgets:
        parser.error("all replay modes require --candidate-budgets")
    if args.mode == "paired":
        output = paired_same_k(
            rows,
            candidate_budgets=args.candidate_budgets,
        )
    elif args.mode == "shuffle":
        output = histogram_shuffle(
            rows,
            seed=args.seed,
            candidate_budgets=args.candidate_budgets,
        )
    else:
        if (
            not args.candidate_budgets
            or not args.candidate_costs
            or args.target_mean_cost is None
        ):
            parser.error("adaptok mode requires budgets, costs, and target mean cost")
        output = adaptok_test_batch_ilp(
            rows,
            candidate_budgets=args.candidate_budgets,
            candidate_costs=args.candidate_costs,
            target_mean_cost=args.target_mean_cost,
        )
    result = _write_immutable(args.output_jsonl, output)
    result["mode"] = args.mode
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
