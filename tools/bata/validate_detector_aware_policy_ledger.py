from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import detector_aware_acquisition_policy as detector_policy
from tools.bata import detector_deploy_leakage
from tools.bata import validate_paction_learned_policy_ledger as base_validator


SUMMARY_SCHEMA_VERSION = "c3_detector_aware_policy_ledger_validation_v1"
READY = "C3_DETECTOR_AWARE_POLICY_LEDGER_VALIDATION_PASS"


_read_jsonl = base_validator._read_jsonl
_write_json = base_validator._write_json
_sha256_file = base_validator._sha256_file
_sample_map = base_validator._sample_map
_positions = base_validator._positions


def _teacher_utility(row: Mapping[str, Any]) -> list[float]:
    split = str(row.get("split") or row.get("subset") or row.get("subset_name") or "").strip().lower()
    if split not in {"train", "training"}:
        return []
    provenance = row.get("teacher_utility_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("split_scope") != "train_only":
        return []
    teacher = row.get("teacher_utility")
    if isinstance(teacher, Mapping) and isinstance(teacher.get("frame_utility"), list):
        return [float(item) for item in teacher["frame_utility"]]
    if isinstance(row.get("frame_utility"), list):
        return [float(item) for item in row["frame_utility"]]
    return []


def _check_detector_metadata(
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    require_policy_source: str | None,
    require_checkpoint_path: str | Path | None,
    require_checkpoint_sha256: str | None,
    require_paction_provenance: bool,
    require_deployable: bool,
) -> None:
    sample_by_id = _sample_map(sample_rows)
    for line_no, row in enumerate(ledger_rows, start=1):
        sample_id = str(row.get("sample_id"))
        sample = sample_by_id[sample_id]
        if require_deployable:
            detector_deploy_leakage.reject_detector_deploy_forbidden_payloads(
                sample,
                source_name=f"{sample_id}: deploy sample_jsonl",
            )
            detector_deploy_leakage.reject_detector_deploy_forbidden_payloads(
                row,
                source_name=f"{sample_id}: deploy ledger_jsonl",
            )
        policy = sample.get("detector_aware_policy")
        if not isinstance(policy, Mapping):
            raise ValueError(f"{sample_id}: detector_aware_policy metadata is required")
        if policy.get("stage_label") is not None and policy.get("stage_label") != detector_policy.STAGE_LABEL:
            raise ValueError(f"{sample_id}: detector-aware policy stage label mismatch")
        if policy.get("end_to_end") is not None and policy.get("end_to_end") is not False:
            raise ValueError(f"{sample_id}: detector-aware policy must declare end_to_end=false")
        if policy.get("uses_uniform_fill") is not False or policy.get("uses_uniform_scaffold") is not False:
            raise ValueError(f"{sample_id}: detector-aware policy must disable uniform fill/scaffold")
        if policy.get("acquisition_unit") != "temporal_observation_center":
            raise ValueError(f"{sample_id}: detector-aware policy acquisition_unit must be temporal_observation_center")
        if policy.get("selected_positions_coordinate_system") != "local_dense_snippet_index":
            raise ValueError(f"{sample_id}: detector-aware policy coordinate system must be local_dense_snippet_index")
        if policy.get("dense_grid_unit") != "snippet_center":
            raise ValueError(f"{sample_id}: detector-aware policy dense_grid_unit must be snippet_center")
        if policy.get("raw_frame_claim_allowed") is not False:
            raise ValueError(f"{sample_id}: detector-aware policy must not allow raw-frame claims")
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), Mapping) else {}
        ledger_source = row.get("policy_source", diagnostics.get("policy_source"))
        ledger_checkpoint_path = row.get("policy_checkpoint_path", diagnostics.get("policy_checkpoint_path"))
        ledger_checkpoint_sha256 = row.get("policy_checkpoint_sha256", diagnostics.get("policy_checkpoint_sha256"))
        if require_policy_source is not None:
            if policy.get("source") != str(require_policy_source) or ledger_source != str(require_policy_source):
                raise ValueError(f"{sample_id}: policy_source must be {require_policy_source}")
        if require_checkpoint_path is not None:
            if str(policy.get("checkpoint_path")) != str(require_checkpoint_path) or str(ledger_checkpoint_path) != str(require_checkpoint_path):
                raise ValueError(f"{sample_id}: policy checkpoint_path mismatch")
        metadata_sha = policy.get("checkpoint_sha256") or policy.get("policy_checkpoint_sha256")
        if require_checkpoint_sha256 is not None:
            if metadata_sha != str(require_checkpoint_sha256) or ledger_checkpoint_sha256 != str(require_checkpoint_sha256):
                raise ValueError(f"{sample_id}: policy checkpoint_sha256 mismatch")
        if require_paction_provenance:
            base_validator._validate_paction_positive_provenance(policy, source_name=f"{sample_id}:detector_aware_policy")


def _utility_metrics_for_ledgers(
    *,
    metric_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metric_by_id = _sample_map(metric_rows)
    coverages: list[float] = []
    ndcgs: list[float] = []
    selected_sums: list[float] = []
    total_sums: list[float] = []
    rows_with_utility = 0
    for line_no, row in enumerate(ledger_rows, start=1):
        sample_id = str(row.get("sample_id"))
        metric_row = metric_by_id[sample_id]
        utility = _teacher_utility(metric_row)
        if not utility:
            continue
        selected = _positions(row.get("selected_positions"), name=f"ledger:{line_no}: selected_positions")
        metrics = detector_policy.detector_utility_metrics(
            selected,
            utility,
            valid_len=int(row.get("valid_len") or len(utility)),
        )
        rows_with_utility += 1
        if metrics["detector_utility_coverage"] is not None:
            coverages.append(float(metrics["detector_utility_coverage"]))
        if metrics["detector_utility_ndcg"] is not None:
            ndcgs.append(float(metrics["detector_utility_ndcg"]))
        selected_sums.append(float(metrics["detector_utility_selected_sum"] or 0.0))
        total_sums.append(float(metrics["detector_utility_total_sum"] or 0.0))
    mean = lambda values: None if not values else sum(values) / float(len(values))
    return {
        "detector_utility_rows": int(rows_with_utility),
        "detector_utility_coverage": mean(coverages),
        "detector_utility_ndcg": mean(ndcgs),
        "detector_utility_selected_sum": sum(selected_sums),
        "detector_utility_total_sum": sum(total_sums),
        "detector_utility_metric_source": "metric_sample_jsonl_train_only_teacher_utility_when_available",
        "detector_utility_metric_availability": (
            "available_train_only_teacher_utility"
            if rows_with_utility > 0
            else "not_available_no_train_only_teacher_utility"
        ),
    }


def validate_ledger(
    *,
    sample_jsonl: str | Path,
    ledger_jsonl: str | Path,
    strategy: str,
    metric_sample_jsonl: str | Path | None = None,
    expected_target_len: int | None = None,
    require_selected_count: int | None = None,
    allow_short_valid_ratio_count: bool = False,
    require_nonconstant_selected_count: bool = False,
    require_deployable: bool = True,
    boundary_radius: int = 1,
    boundary_radii: Sequence[int] | None = None,
    max_unselected_hole: int | None = None,
    max_p95_unselected_hole: float | None = None,
    max_uniform_similarity: float | None = None,
    require_policy_source: str | None = detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
    require_checkpoint_path: str | Path | None = None,
    require_checkpoint_sha256: str | None = None,
    require_paction_provenance: bool = False,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    sample_rows = _read_jsonl(sample_jsonl)
    metric_rows = _read_jsonl(metric_sample_jsonl) if metric_sample_jsonl is not None else sample_rows
    ledger_rows = _read_jsonl(ledger_jsonl)
    _check_detector_metadata(
        sample_rows=sample_rows,
        ledger_rows=ledger_rows,
        require_policy_source=require_policy_source,
        require_checkpoint_path=require_checkpoint_path,
        require_checkpoint_sha256=require_checkpoint_sha256,
        require_paction_provenance=bool(require_paction_provenance),
        require_deployable=bool(require_deployable),
    )
    summary = base_validator.validate_ledger(
        sample_jsonl=sample_jsonl,
        metric_sample_jsonl=metric_sample_jsonl,
        ledger_jsonl=ledger_jsonl,
        strategy=strategy,
        expected_target_len=expected_target_len,
        require_selected_count=require_selected_count,
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
        require_nonconstant_selected_count=bool(require_nonconstant_selected_count),
        require_deployable=bool(require_deployable),
        boundary_radius=int(boundary_radius),
        boundary_radii=boundary_radii,
        max_unselected_hole=max_unselected_hole,
        max_p95_unselected_hole=max_p95_unselected_hole,
        max_uniform_similarity=max_uniform_similarity,
        require_policy_source=None,
        require_checkpoint_path=None,
        require_checkpoint_sha256=None,
        require_paction_provenance=False,
    )
    summary.update(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "decision": READY,
            "stage_label": detector_policy.STAGE_LABEL,
            "dynamic_gain_calibration": dict(detector_policy.DEFAULT_DYNAMIC_GAIN_CALIBRATION),
            "required_policy_source": require_policy_source,
            "required_checkpoint_path": None if require_checkpoint_path is None else str(require_checkpoint_path),
            "required_checkpoint_sha256": require_checkpoint_sha256,
            "require_paction_provenance": bool(require_paction_provenance),
            "adatad_map": None,
            "adatad_map_source": "locked_until_full_AdaTAD_train_eval",
            "map_claim_allowed": False,
            "end_to_end": False,
        }
    )
    summary.update(_utility_metrics_for_ledgers(metric_rows=metric_rows, ledger_rows=ledger_rows))
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate strict detector-aware Stage-2 selector ledgers.")
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--ledger-jsonl", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--metric-sample-jsonl")
    parser.add_argument("--expected-target-len", type=int)
    parser.add_argument("--require-selected-count", type=int)
    parser.add_argument("--allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--require-nonconstant-selected-count", action="store_true")
    parser.add_argument("--require-deployable", action="store_true")
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--boundary-radii", type=int, nargs="+")
    parser.add_argument("--max-unselected-hole", type=int)
    parser.add_argument("--max-p95-unselected-hole", type=float)
    parser.add_argument("--max-uniform-similarity", type=float)
    parser.add_argument("--require-policy-source", default=detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE)
    parser.add_argument("--require-checkpoint-path")
    parser.add_argument("--require-checkpoint-sha256")
    parser.add_argument("--require-paction-provenance", action="store_true")
    parser.add_argument("--summary-json")
    args = parser.parse_args(argv)
    summary = validate_ledger(
        sample_jsonl=args.sample_jsonl,
        ledger_jsonl=args.ledger_jsonl,
        strategy=args.strategy,
        metric_sample_jsonl=args.metric_sample_jsonl,
        expected_target_len=args.expected_target_len,
        require_selected_count=args.require_selected_count,
        allow_short_valid_ratio_count=bool(args.allow_short_valid_ratio_count),
        require_nonconstant_selected_count=bool(args.require_nonconstant_selected_count),
        require_deployable=bool(args.require_deployable),
        boundary_radius=int(args.boundary_radius),
        boundary_radii=args.boundary_radii,
        max_unselected_hole=args.max_unselected_hole,
        max_p95_unselected_hole=args.max_p95_unselected_hole,
        max_uniform_similarity=args.max_uniform_similarity,
        require_policy_source=args.require_policy_source,
        require_checkpoint_path=args.require_checkpoint_path,
        require_checkpoint_sha256=args.require_checkpoint_sha256,
        require_paction_provenance=bool(args.require_paction_provenance),
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
