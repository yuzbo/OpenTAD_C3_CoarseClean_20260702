from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.duca_ceiling_utils import (
    mean, read_jsonl, require_finite, sha256, validate_provenance,
    validate_selection, write_csv, write_json,
)


def evaluate_record(row: Mapping[str, Any]) -> dict[str, Any]:
    provenance = validate_provenance(row.get("provenance"), context=str(row.get("sample_id")))
    valid_len, budget, max_hole = int(row["valid_len"]), int(row["budget"]), int(row["max_hole"])
    candidates = row.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        raise ValueError(f"{row.get('sample_id')}: candidates are required")
    valid: list[tuple[float, Mapping[str, Any]]] = []
    candidate_ids: set[str] = set()
    selections: set[tuple[int, ...]] = set()
    rejected = {"budget": 0, "range": 0, "max_hole": 0}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("each candidate must be an object")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_ids:
            raise ValueError("candidate_id must be present and unique")
        candidate_ids.add(candidate_id)
        positions = tuple(sorted(int(value) for value in candidate.get("selected_positions", [])))
        if positions in selections:
            raise ValueError("duplicate candidate selection")
        selections.add(positions)
        components = candidate.get("detector_loss_components")
        if not isinstance(components, Mapping):
            raise ValueError("each candidate requires detector_loss_components")
        cls_loss = require_finite(components.get("cls_loss"), "cls_loss")
        reg_loss = require_finite(components.get("reg_loss"), "reg_loss")
        detector_loss = require_finite(candidate.get("detector_loss"), "detector_loss")
        if abs(detector_loss - (cls_loss + reg_loss)) > 1e-6 * max(1.0, abs(detector_loss)):
            raise ValueError("detector_loss must equal cls_loss + reg_loss")
        ok, reason = validate_selection(valid_len, budget, max_hole, positions)
        if ok:
            valid.append((detector_loss, candidate))
        else:
            rejected[reason] += 1
    if not valid:
        raise ValueError(f"{row.get('sample_id')}: no feasible candidate")
    valid.sort(key=lambda pair: (pair[0], str(pair[1].get("candidate_id", ""))))
    best_loss, best = valid[0]
    references = [pair for pair in valid if pair[1].get("is_reference") is True]
    if len(references) != 1:
        raise ValueError(f"{row.get('sample_id')}: exactly one feasible is_reference candidate is required")
    reference = references[0]
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
        "provenance_key": json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        **{f"rejected_{key}": value for key, value in rejected.items()},
    }


def run(records_jsonl: str | Path, output_dir: str | Path) -> dict[str, Any]:
    rows = [evaluate_record(row) for row in read_jsonl(records_jsonl)]
    provenance_keys = {row.pop("provenance_key") for row in rows}
    if len(provenance_keys) != 1:
        raise ValueError("all records must share one provenance binding")
    provenance = json.loads(next(iter(provenance_keys)))
    summary = {
        "schema_version": "duca_supplied_candidate_counterfactual_v2",
        "diagnostic_role": "evaluation_only_best_of_supplied_candidates_not_upper_bound",
        "sample_count": len(rows),
        "mean_reference_detector_loss": mean(row["reference_detector_loss"] for row in rows),
        "mean_best_detector_loss": mean(row["best_detector_loss"] for row in rows),
        "mean_counterfactual_gain": mean(row["counterfactual_gain"] for row in rows),
        "positive_gain_fraction": mean(float(row["counterfactual_gain"] > 0) for row in rows),
        "input_sha256": sha256(records_jsonl),
        "provenance": provenance,
        "contract": {"fixed_k_checked": True, "max_hole_checked": True, "finite_cls_reg_detector_loss_required": True, "candidate_search_exhaustive": False, "paper_deployable": False},
    }
    out = Path(output_dir)
    write_json(out / "supplied_candidate_counterfactual.json", summary)
    write_csv(out / "supplied_candidate_counterfactual_per_sample.csv", rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Best of supplied detector-evaluated candidates; not an upper bound.")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.records_jsonl, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
