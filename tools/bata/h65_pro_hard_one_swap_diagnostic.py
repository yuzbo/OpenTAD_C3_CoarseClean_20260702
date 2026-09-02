from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "h65_pro_hard_one_swap_diagnostic_v1"


def _as_float(record: Mapping[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return float(value)
    if "baseline_score" in record and "candidate_score" in record:
        return float(record["candidate_score"]) - float(record["baseline_score"])
    raise ValueError(f"record lacks any of: {', '.join(keys)}")


def _video_key(record: Mapping[str, Any], index: int) -> str:
    for key in ("video_id", "video_name", "sample_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row-{index}"


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_no} is not a JSON object")
            records.append(payload)
    return records


def summarize_one_swap(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("hard one-swap diagnostic requires at least one record")
    rows = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        predicted_delta = _as_float(
            record,
            (
                "predicted_delta",
                "selector_delta",
                "utility_delta",
                "predicted_utility_delta",
            ),
        )
        observed_delta = _as_float(
            record,
            (
                "observed_delta",
                "detector_delta",
                "map_delta",
                "loss_reduction",
            ),
        )
        row = {
            "index": index,
            "video_id": _video_key(record, index),
            "predicted_delta": predicted_delta,
            "observed_delta": observed_delta,
            "predicted_positive": predicted_delta > 0.0,
            "observed_positive": observed_delta > 0.0,
        }
        row["sign_match"] = row["predicted_positive"] == row["observed_positive"]
        rows.append(row)
        groups[row["video_id"]].append(row)

    positive_predictions = [row for row in rows if row["predicted_positive"]]
    sign_matches = sum(1 for row in rows if row["sign_match"])
    top_rows = [max(group, key=lambda item: item["predicted_delta"]) for group in groups.values()]
    top_improvements = sum(1 for row in top_rows if row["observed_positive"])
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "training_path": False,
            "model_training": False,
            "gradient_graph": False,
            "input": "offline_jsonl_one_swap_records",
        },
        "record_count": len(rows),
        "video_count": len(groups),
        "positive_prediction_count": len(positive_predictions),
        "sign_match_rate": sign_matches / len(rows),
        "top_predicted_swap_observed_improvement_rate": top_improvements / len(top_rows),
        "top_predicted_swaps": top_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize offline H65-Pro hard one-swap diagnostics.")
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    records = load_records(args.input_jsonl)
    summary = summarize_one_swap(records)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "H65-Pro hard one-swap diagnostic: "
        f"{summary['record_count']} records, "
        f"sign_match_rate={summary['sign_match_rate']:.4f}, "
        "top_predicted_swap_observed_improvement_rate="
        f"{summary['top_predicted_swap_observed_improvement_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
