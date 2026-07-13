from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.duca_ceiling_utils import mean, read_jsonl, sha256, validate_selection, write_csv, write_json


def evaluate_record(row: Mapping[str, Any]) -> dict[str, Any]:
    valid_len, budget, max_hole = int(row["valid_len"]), int(row["budget"]), int(row["max_hole"])
    candidates = row.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        raise ValueError(f"{row.get('sample_id')}: candidates are required")
    valid: list[tuple[float, Mapping[str, Any]]] = []
    rejected = {"budget": 0, "range": 0, "max_hole": 0}
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or "detector_loss" not in candidate:
            raise ValueError("each candidate requires train/eval-only detector_loss")
        ok, reason = validate_selection(valid_len, budget, max_hole, candidate.get("selected_positions", []))
        if ok:
            valid.append((float(candidate["detector_loss"]), candidate))
        else:
            rejected[reason] += 1
    if not valid:
        raise ValueError(f"{row.get('sample_id')}: no feasible candidate")
    valid.sort(key=lambda pair: (pair[0], str(pair[1].get("candidate_id", ""))))
    best_loss, best = valid[0]
    reference = next((pair for pair in valid if pair[1].get("is_reference")), None)
    if reference is None:
        raise ValueError(f"{row.get('sample_id')}: a feasible is_reference candidate is required")
    return {
        "sample_id": row.get("sample_id"),
        "video_id": row.get("video_id"),
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(valid),
        "best_candidate_id": best.get("candidate_id"),
        "best_detector_loss": best_loss,
        "reference_candidate_id": reference[1].get("candidate_id"),
        "reference_detector_loss": reference[0],
        "counterfactual_gain": reference[0] - best_loss,
        **{f"rejected_{key}": value for key, value in rejected.items()},
    }


def run(records_jsonl: str | Path, output_dir: str | Path) -> dict[str, Any]:
    rows = [evaluate_record(row) for row in read_jsonl(records_jsonl)]
    summary = {
        "schema_version": "duca_fixed_constraint_counterfactual_ceiling_v1",
        "diagnostic_role": "evaluation_only_best_of_supplied_feasible_candidates",
        "sample_count": len(rows),
        "mean_reference_detector_loss": mean(row["reference_detector_loss"] for row in rows),
        "mean_best_detector_loss": mean(row["best_detector_loss"] for row in rows),
        "mean_counterfactual_gain": mean(row["counterfactual_gain"] for row in rows),
        "positive_gain_fraction": mean(float(row["counterfactual_gain"] > 0) for row in rows),
        "input_sha256": sha256(records_jsonl),
        "contract": {"fixed_k_checked": True, "max_hole_checked": True, "detector_loss_required": True, "paper_deployable": False},
    }
    out = Path(output_dir)
    write_json(out / "feasible_set_counterfactual_ceiling.json", summary)
    write_csv(out / "feasible_set_counterfactual_per_sample.csv", rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Best-of-candidate ceiling under exact K and max-hole constraints.")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.records_jsonl, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
