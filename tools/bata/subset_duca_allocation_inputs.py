from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from tools.bata.diagnose_duca_allocation_family_ceiling import read_input_records
from tools.bata.export_duca_allocation_ceiling_inputs import sha256, write_json_exclusive


def create_subset(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    first_n: int,
    strategy: str = "first_n",
    seed: str = "duca-allocation-ceiling-v1",
) -> dict[str, Any]:
    input_path = Path(input_jsonl).resolve()
    output_path = Path(output_jsonl).resolve()
    summary_path = Path(summary_json).resolve()
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("allocation input subset never overwrites artifacts")
    if first_n < 1:
        raise ValueError("first_n must be positive")
    if strategy not in {"first_n", "hash_video_round_robin"}:
        raise ValueError("unknown allocation input subset strategy")
    if not str(seed):
        raise ValueError("subset seed must be non-empty")
    records = read_input_records(input_path)
    if strategy == "first_n":
        selected = records[:first_n]
    else:
        selected = _hash_video_round_robin(records, first_n=first_n, seed=str(seed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False)
                + "\n"
            )
    summary = {
        "schema_version": "duca_allocation_input_subset_v1",
        "selection_rule": strategy,
        "selection_seed": str(seed),
        "requested_first_n": int(first_n),
        "source_sample_count": len(records),
        "selected_sample_count": len(selected),
        "selected_sample_ids": [row["sample_id"] for row in selected],
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256(input_path),
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": sha256(output_path),
        "contract": {
            "source_order_preserved": strategy == "first_n",
            "records_unchanged": True,
            "outcome_dependent_selection": False,
            "cross_video_round_robin": strategy == "hash_video_round_robin",
        },
    }
    write_json_exclusive(summary_path, summary)
    return summary


def _hash_video_round_robin(
    records: Sequence[dict[str, Any]],
    *,
    first_n: int,
    seed: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["video_id"])].append(row)
    for video_id, rows in groups.items():
        rows.sort(key=lambda row: _rank(seed, video_id, str(row["sample_id"])))
    video_ids = sorted(groups, key=lambda video_id: _rank(seed, video_id))
    selected: list[dict[str, Any]] = []
    depth = 0
    target = min(int(first_n), len(records))
    while len(selected) < target:
        added = False
        for video_id in video_ids:
            rows = groups[video_id]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != target:
        raise RuntimeError("deterministic cross-video subset did not reach its target size")
    return selected


def _rank(seed: str, *parts: str) -> str:
    payload = "\0".join((seed, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic first-N subset of DUCA allocation inputs."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--first-n", type=int, required=True)
    parser.add_argument(
        "--strategy",
        choices=["first_n", "hash_video_round_robin"],
        default="first_n",
    )
    parser.add_argument("--seed", default="duca-allocation-ceiling-v1")
    args = parser.parse_args(argv)
    result = create_subset(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        first_n=args.first_n,
        strategy=args.strategy,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
