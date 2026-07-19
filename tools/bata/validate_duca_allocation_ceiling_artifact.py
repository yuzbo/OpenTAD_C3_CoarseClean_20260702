from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from tools.bata.diagnose_duca_allocation_family_ceiling import (
    OUTPUT_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    allocation_metrics,
    axis_from_record,
    coarse_signal_metrics,
    read_input_records,
    summarize_outputs,
)
from tools.bata.duca_allocation_families import (
    effective_budget,
    physical_gap_report,
    resolve_physical_cap,
    select_family_a,
    select_family_b,
    select_family_c,
    validate_physical_selection,
)
from tools.bata.duca_exact_physical_solver import (
    GroundTruthObjectiveSpec,
    select_family_d,
    solve_ground_truth_lexicographic,
)
from tools.bata.export_duca_allocation_ceiling_inputs import (
    canonical_sha256,
    sha256,
    write_json_exclusive,
)


_OUTPUT_KEYS = {
    "schema_version",
    "sample_id",
    "video_id",
    "split",
    "valid_len",
    "requested_budget",
    "score_key",
    "cap",
    "coarse_signal_metrics",
    "families",
    "input_record_sha256",
    "contract",
    "record_sha256",
}
_FAMILY_KEYS = {
    "family",
    "family_key",
    "positions",
    "budget",
    "score_sum",
    "exact",
    "deployable",
    "privileged",
    "solver_status",
    "physical_cap_compliant",
    "gap_report",
    "scaffold_positions",
    "residual_positions",
    "allocation_metrics",
    "additive_solver",
    "gt_solver",
}


def validate_artifact(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_jsonl).resolve()
    output_path = Path(output_jsonl).resolve()
    summary_path = Path(summary_json).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be an object")
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported summary schema")
    if Path(str(summary.get("input_jsonl"))).resolve() != input_path:
        raise ValueError("summary input path mismatch")
    if Path(str(summary.get("output_jsonl"))).resolve() != output_path:
        raise ValueError("summary output path mismatch")
    if summary.get("input_jsonl_sha256") != sha256(input_path):
        raise ValueError("summary input SHA-256 mismatch")
    if summary.get("output_jsonl_sha256") != sha256(output_path):
        raise ValueError("summary output SHA-256 mismatch")
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("summary contract is required")
    if contract.get("deploy_score_uses_gt") is not False:
        raise ValueError("summary permits GT leakage into deploy scores")
    if contract.get("gt_families_deployable") is not False:
        raise ValueError("summary marks privileged GT families as deployable")
    if contract.get("exact_status_required") != "OPTIMAL":
        raise ValueError("summary weakens exact solver status")
    if contract.get("detector_mAP_evaluated") is not False:
        raise ValueError("geometry artifact cannot claim detector mAP")

    score_key = str(summary.get("score_key"))
    cap_policy = str(summary.get("cap_policy"))
    cap_value = summary.get("cap_value")
    gt_families = str(summary.get("gt_families"))
    quantization_scale = int(summary.get("quantization_scale", 0))
    raw_solver_options = summary.get("gt_solver_options")
    if not isinstance(raw_solver_options, Mapping):
        raise ValueError("summary GT solver options are required")
    expected_solver_option_keys = {
        "backend",
        "presolve",
        "mip_rel_gap",
        "time_limit_seconds",
        "compute_upper_envelopes",
    }
    if set(raw_solver_options) != expected_solver_option_keys:
        raise ValueError("summary GT solver option fields mismatch")
    if raw_solver_options.get("backend") != "scipy.optimize.milp_highs":
        raise ValueError("summary GT solver backend mismatch")
    if raw_solver_options.get("presolve") is not True:
        raise ValueError("summary GT solver must enable presolve")
    raw_mip_rel_gap = raw_solver_options.get("mip_rel_gap")
    if (
        isinstance(raw_mip_rel_gap, bool)
        or not isinstance(raw_mip_rel_gap, (int, float))
        or not math.isfinite(float(raw_mip_rel_gap))
        or float(raw_mip_rel_gap) != 0.0
    ):
        raise ValueError("summary GT solver must require zero MIP gap")
    raw_time_limit = raw_solver_options.get("time_limit_seconds")
    gt_time_limit_seconds = (
        None if raw_time_limit is None else float(raw_time_limit)
    )
    if gt_time_limit_seconds is not None and (
        not math.isfinite(gt_time_limit_seconds) or gt_time_limit_seconds <= 0
    ):
        raise ValueError("summary GT time limit must be finite and positive")
    compute_upper_envelopes = raw_solver_options.get("compute_upper_envelopes")
    if not isinstance(compute_upper_envelopes, bool):
        raise ValueError("summary upper-envelope option must be boolean")
    raw_objective_spec = summary.get("objective_spec")
    if not isinstance(raw_objective_spec, Mapping):
        raise ValueError("summary objective specification is required")
    objective_spec = GroundTruthObjectiveSpec(
        boundary_radii=tuple(int(value) for value in raw_objective_spec.get("boundary_radii", [])),
        short_action_max_length=float(raw_objective_spec.get("short_action_max_length", -1)),
        distance_scale=int(raw_objective_spec.get("distance_scale", 0)),
        lex_block_size=int(raw_objective_spec.get("lex_block_size", 0)),
    )
    if cap_policy not in {"uniform_reference", "explicit_frames", "explicit_seconds"}:
        raise ValueError("summary physical cap policy is invalid")
    if gt_families not in {"none", "d", "e", "both"}:
        raise ValueError("summary GT-family mode is invalid")
    if quantization_scale < 1:
        raise ValueError("summary score quantization scale must be positive")

    input_rows = read_input_records(input_path)
    inputs = {str(row["sample_id"]): row for row in input_rows}
    outputs = _read_outputs(output_path)
    if int(summary.get("sample_count", -1)) != len(outputs):
        raise ValueError("summary sample count mismatch")
    if set(outputs) != set(inputs):
        raise ValueError("input/output sample sets differ")

    family_counts: dict[str, int] = {}
    for sample_id, row in outputs.items():
        input_row = inputs[sample_id]
        _validate_output_record(
            row,
            input_row,
            score_key=score_key,
            cap_policy=cap_policy,
            cap_value=cap_value,
            quantization_scale=quantization_scale,
            objective_spec=objective_spec,
            gt_families=gt_families,
            gt_time_limit_seconds=gt_time_limit_seconds,
            compute_upper_envelopes=compute_upper_envelopes,
        )
        for family in row["families"]:
            key = str(family["family_key"])
            family_counts[key] = family_counts.get(key, 0) + 1

    expected_summary = summarize_outputs(
        [outputs[str(row["sample_id"])] for row in input_rows],
        input_path=input_path,
        output_path=output_path,
        score_key=score_key,
        cap_policy=cap_policy,
        cap_value=cap_value,
        gt_families=gt_families,
        objective_spec=objective_spec,
        quantization_scale=quantization_scale,
        gt_time_limit_seconds=gt_time_limit_seconds,
        compute_upper_envelopes=compute_upper_envelopes,
    )
    _assert_numeric_tree_close(expected_summary, summary, context="summary")
    return {
        "schema_version": "duca_allocation_ceiling_validation_v1",
        "validation_passed": True,
        "gt_solver_replayed": gt_families != "none",
        "gt_families": gt_families,
        "gt_solver_options": dict(raw_solver_options),
        "sample_count": len(outputs),
        "family_counts": family_counts,
        "input_jsonl_sha256": sha256(input_path),
        "output_jsonl_sha256": sha256(output_path),
        "summary_json_sha256": sha256(summary_path),
    }


def validate_artifact_receipt(
    *,
    validation_json: str | Path,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    require_gt_solver_replay: bool,
) -> dict[str, Any]:
    validation_path = Path(validation_json).resolve()
    input_path = Path(input_jsonl).resolve()
    output_path = Path(output_jsonl).resolve()
    summary_path = Path(summary_json).resolve()
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ceiling validation receipt must be an object")
    expected_keys = {
        "schema_version",
        "validation_passed",
        "gt_solver_replayed",
        "gt_families",
        "gt_solver_options",
        "sample_count",
        "family_counts",
        "input_jsonl_sha256",
        "output_jsonl_sha256",
        "summary_json_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("ceiling validation receipt fields mismatch")
    if payload.get("schema_version") != "duca_allocation_ceiling_validation_v1":
        raise ValueError("ceiling validation receipt schema mismatch")
    if payload.get("validation_passed") is not True:
        raise ValueError("ceiling validation receipt did not pass")
    if payload.get("input_jsonl_sha256") != sha256(input_path):
        raise ValueError("ceiling validation receipt input hash mismatch")
    if payload.get("output_jsonl_sha256") != sha256(output_path):
        raise ValueError("ceiling validation receipt output hash mismatch")
    if payload.get("summary_json_sha256") != sha256(summary_path):
        raise ValueError("ceiling validation receipt summary hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("ceiling summary must be an object")
    if int(payload.get("sample_count", -1)) != int(summary.get("sample_count", -2)):
        raise ValueError("ceiling validation receipt sample count mismatch")
    if payload.get("gt_families") != summary.get("gt_families"):
        raise ValueError("ceiling validation receipt GT-family mode mismatch")
    if payload.get("gt_solver_options") != summary.get("gt_solver_options"):
        raise ValueError("ceiling validation receipt solver options mismatch")
    has_gt = summary.get("gt_families") != "none"
    if require_gt_solver_replay and (
        not has_gt or payload.get("gt_solver_replayed") is not True
    ):
        raise ValueError("ceiling validation receipt lacks required GT solver replay")
    if not has_gt and payload.get("gt_solver_replayed") is not False:
        raise ValueError("non-GT ceiling receipt incorrectly claims GT solver replay")
    return dict(payload)


def _read_outputs(path: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            sample_id = str(row.get("sample_id"))
            if sample_id in outputs:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id")
            outputs[sample_id] = row
    if not outputs:
        raise ValueError("output artifact has no records")
    return outputs


def _validate_output_record(
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    *,
    score_key: str,
    cap_policy: str,
    cap_value: float | int | None,
    quantization_scale: int,
    objective_spec: GroundTruthObjectiveSpec,
    gt_families: str,
    gt_time_limit_seconds: float | None,
    compute_upper_envelopes: bool,
) -> None:
    unknown = set(row) - _OUTPUT_KEYS
    missing = _OUTPUT_KEYS - set(row)
    if unknown or missing:
        raise ValueError(f"strict output fields mismatch: unknown={unknown}, missing={missing}")
    if row.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise ValueError("unsupported output record schema")
    recorded_hash = row.get("record_sha256")
    unhashed = dict(row)
    unhashed.pop("record_sha256", None)
    if canonical_sha256(unhashed) != recorded_hash:
        raise ValueError("output record SHA-256 mismatch")
    if row.get("input_record_sha256") != input_row.get("record_sha256"):
        raise ValueError("output is not bound to its exact input record")
    if row.get("video_id") != input_row.get("video_id") or row.get("split") != input_row.get("split"):
        raise ValueError("output identity differs from input")
    axis = axis_from_record(input_row)
    if int(row.get("valid_len", -1)) != axis.valid_len:
        raise ValueError("output valid_len mismatch")
    requested_budget = int(row.get("requested_budget", -1))
    if requested_budget != int(input_row.get("requested_budget", -2)):
        raise ValueError("output requested budget mismatch")
    if row.get("score_key") != score_key:
        raise ValueError("output score key differs from summary")
    exact_budget = effective_budget(axis.valid_len, requested_budget)
    raw_scores = input_row.get("scores")
    if not isinstance(raw_scores, Mapping) or score_key not in raw_scores:
        raise ValueError("input is missing the summary-bound deploy score")
    scores = tuple(float(value) for value in raw_scores[score_key])
    if len(scores) != axis.valid_len or any(not math.isfinite(value) for value in scores):
        raise ValueError("summary-bound deploy score is invalid")
    cap = resolve_physical_cap(
        axis,
        requested_budget=requested_budget,
        policy=cap_policy,
        value=cap_value,
    )
    _assert_numeric_tree_close(cap.to_dict(), row.get("cap"), context="cap")
    expected_a = select_family_a(
        axis,
        requested_budget=requested_budget,
        cap=cap,
    )
    expected_b = select_family_b(
        axis,
        scores,
        requested_budget=requested_budget,
        cap=cap,
    )
    expected_c = select_family_c(
        axis,
        scores,
        requested_budget=requested_budget,
        cap=cap,
    )
    expected_d, expected_d_solver = select_family_d(
        axis,
        scores,
        requested_budget=requested_budget,
        cap=cap,
        quantization_scale=quantization_scale,
    )
    expected_deploy = {
        "A_exact_uniform": expected_a,
        "B_one_per_uniform_cell": expected_b,
        "C_uniform_scaffold_residual": expected_c,
        "D_deploy_score": expected_d,
    }
    expected_gt: dict[str, Any] = {}
    if gt_families in {"d", "both"}:
        expected_gt["D_privileged_gt_ceiling"] = solve_ground_truth_lexicographic(
            axis,
            input_row.get("gt_segments", []),
            requested_budget=requested_budget,
            cap=cap,
            objective_spec=objective_spec,
            compute_upper_envelopes=compute_upper_envelopes,
            time_limit_seconds=gt_time_limit_seconds,
        )
    if gt_families in {"e", "both"}:
        expected_gt["E_privileged_unrestricted_gt"] = solve_ground_truth_lexicographic(
            axis,
            input_row.get("gt_segments", []),
            requested_budget=requested_budget,
            cap=None,
            objective_spec=objective_spec,
            compute_upper_envelopes=compute_upper_envelopes,
            time_limit_seconds=gt_time_limit_seconds,
        )
    contract = row.get("contract")
    expected_contract = {
        "offline_full_window": True,
        "deploy_score_uses_gt": False,
        "gt_families_privileged_only": True,
        "detector_mAP_oracle_claim": False,
        "exact_language_requires_optimal": True,
    }
    if not isinstance(contract, Mapping) or dict(contract) != expected_contract:
        raise ValueError("strict output contract mismatch")
    families = row.get("families")
    if not isinstance(families, Sequence) or not families:
        raise ValueError("output families are required")
    family_keys: set[str] = set()
    selections: dict[str, tuple[int, ...]] = {}
    for family in families:
        if not isinstance(family, Mapping):
            raise ValueError("family row must be an object")
        unknown_family = set(family) - _FAMILY_KEYS
        required_family = _FAMILY_KEYS - {"additive_solver", "gt_solver"}
        missing_family = required_family - set(family)
        if unknown_family or missing_family:
            raise ValueError(
                f"strict family fields mismatch: unknown={unknown_family}, missing={missing_family}"
            )
        family_key = str(family["family_key"])
        if family_key in family_keys:
            raise ValueError("family_key must be unique per sample")
        family_keys.add(family_key)
        positions = tuple(int(value) for value in family["positions"])
        if positions != tuple(sorted(set(positions))):
            raise ValueError(f"{family_key}: positions must be unique and ordered")
        if len(positions) != exact_budget or int(family["budget"]) != exact_budget:
            raise ValueError(f"{family_key}: exact-K violation")
        if positions[0] < 0 or positions[-1] >= axis.valid_len:
            raise ValueError(f"{family_key}: position outside valid prefix")
        if family.get("exact") is not True or family.get("solver_status") != "OPTIMAL":
            raise ValueError(f"{family_key}: non-optimal result cannot be exact evidence")
        if family.get("deployable") is True and family.get("privileged") is True:
            raise ValueError(f"{family_key}: privileged result cannot be deployable")
        if family_key.startswith(("D_privileged", "E_privileged")):
            if "additive_solver" in family:
                raise ValueError(f"{family_key}: GT family cannot carry an additive solver")
            if family.get("privileged") is not True or family.get("deployable") is not False:
                raise ValueError(f"{family_key}: GT family privilege flags are invalid")
            if family.get("family") != family_key:
                raise ValueError(f"{family_key}: GT family identity mismatch")
            if family.get("score_sum") is not None:
                raise ValueError(f"{family_key}: GT family must not claim an additive score")
            if tuple(family.get("scaffold_positions", ())) or tuple(
                family.get("residual_positions", ())
            ):
                raise ValueError(f"{family_key}: GT family must not claim scaffold decomposition")
            solver = family.get("gt_solver")
            if not isinstance(solver, Mapping) or tuple(solver.get("positions", [])) != positions:
                raise ValueError(f"{family_key}: GT solver payload mismatch")
            expected_solver = expected_gt.get(family_key)
            if expected_solver is None:
                raise ValueError(f"{family_key}: GT family was not authorized by summary")
            _assert_numeric_tree_close(
                expected_solver.to_dict(),
                solver,
                context=f"{family_key}.gt_solver.exact_replay",
            )
            if tuple(expected_solver.positions) != positions:
                raise ValueError(f"{family_key}: positions are not the replayed GT optimum")
            if family_key == "D_privileged_gt_ceiling":
                validate_physical_selection(
                    axis,
                    positions,
                    requested_budget=requested_budget,
                    cap=cap,
                )
        if family_key == "D_deploy_score":
            if "gt_solver" in family:
                raise ValueError("D deploy-score family cannot carry a GT solver")
            solver = family.get("additive_solver")
            if not isinstance(solver, Mapping) or tuple(solver.get("positions", [])) != positions:
                raise ValueError("D deploy-score solver payload mismatch")
            _assert_numeric_tree_close(
                expected_d_solver.to_dict(),
                solver,
                context="D_deploy_score.additive_solver",
            )
        elif not family_key.startswith(("D_privileged", "E_privileged")) and (
            "additive_solver" in family or "gt_solver" in family
        ):
            raise ValueError(f"{family_key}: family carries an unexpected solver payload")
        report = physical_gap_report(axis, positions).to_dict()
        _assert_numeric_tree_close(report, family.get("gap_report"), context=f"{family_key}.gap_report")
        if family_key in {
            "A_exact_uniform",
            "C_uniform_scaffold_residual",
            "D_deploy_score",
            "D_privileged_gt_ceiling",
        } and family.get("physical_cap_compliant") is not True:
            raise ValueError(f"{family_key}: registered cap-feasible family violates cap")
        if family_key == "E_privileged_unrestricted_gt":
            expected_compliance = True
            if cap.max_source_frame_interval is not None:
                expected_compliance = (
                    report["source_frame_max_interval"]
                    <= cap.max_source_frame_interval + 1.0e-9
                )
            if cap.max_seconds_interval is not None:
                expected_compliance = expected_compliance and (
                    report["seconds_max_interval"]
                    <= cap.max_seconds_interval + 1.0e-9
                )
            if family.get("physical_cap_compliant") is not expected_compliance:
                raise ValueError("E privileged family cap-compliance flag is incorrect")
        if family_key in expected_deploy:
            _validate_recomputed_deploy_family(
                family,
                expected_deploy[family_key],
                family_key=family_key,
            )
        expected_metrics = allocation_metrics(
            positions,
            input_row.get("gt_segments", []),
            valid_len=axis.valid_len,
            radii=objective_spec.boundary_radii,
            short_action_max_length=objective_spec.short_action_max_length,
        )
        expected_metrics["uniform_overlap"] = len(
            set(positions) & set(expected_a.positions)
        ) / len(expected_a.positions)
        _assert_numeric_tree_close(
            expected_metrics,
            family.get("allocation_metrics"),
            context=f"{family_key}.allocation_metrics",
        )
        if family_key.startswith(("D_privileged", "E_privileged")):
            _validate_gt_solver_payload(
                family,
                input_row=input_row,
                objective_spec=objective_spec,
                uniform_positions=expected_a.positions,
            )
        selections[family_key] = positions
    required = {
        "A_exact_uniform",
        "B_one_per_uniform_cell",
        "C_uniform_scaffold_residual",
        "D_deploy_score",
    }
    expected_family_keys = required | set(expected_gt)
    if family_keys != expected_family_keys:
        raise ValueError(
            "family set differs from summary-bound GT mode: "
            f"expected={expected_family_keys}, actual={family_keys}"
        )
    expected_coarse = coarse_signal_metrics(
        input_row["scores"],
        input_row.get("gt_segments", []),
        valid_len=axis.valid_len,
        transition_radius=max(1, min(objective_spec.boundary_radii[-1], 4)),
    )
    _assert_numeric_tree_close(
        expected_coarse,
        row.get("coarse_signal_metrics"),
        context="coarse_signal_metrics",
    )


def _validate_recomputed_deploy_family(
    actual: Mapping[str, Any],
    expected: Any,
    *,
    family_key: str,
) -> None:
    expected_payload = expected.to_dict()
    if family_key == "D_deploy_score":
        expected_payload["family"] = "D_global_exact_k_physical_gap"
    for key in (
        "family",
        "positions",
        "budget",
        "score_sum",
        "exact",
        "deployable",
        "privileged",
        "solver_status",
        "physical_cap_compliant",
        "gap_report",
        "scaffold_positions",
        "residual_positions",
    ):
        _assert_numeric_tree_close(
            expected_payload[key],
            actual.get(key),
            context=f"{family_key}.{key}",
        )


def _validate_gt_solver_payload(
    family: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    objective_spec: GroundTruthObjectiveSpec,
    uniform_positions: Sequence[int],
) -> None:
    solver = family.get("gt_solver")
    if not isinstance(solver, Mapping):
        raise ValueError("privileged family requires a GT solver payload")
    if solver.get("solver_status") != "OPTIMAL" or solver.get("exact") is not True:
        raise ValueError("privileged GT solver is not exact OPTIMAL")
    if solver.get("privileged") is not True or solver.get("deployable") is not False:
        raise ValueError("privileged GT solver flags are invalid")
    identity = solver.get("solver_identity")
    if not isinstance(identity, str) or not identity.startswith("scipy_highs_milp_"):
        raise ValueError("privileged GT solver identity is invalid")
    mip_gap = solver.get("mip_gap")
    if (
        isinstance(mip_gap, bool)
        or not isinstance(mip_gap, (int, float))
        or not math.isfinite(float(mip_gap))
        or float(mip_gap) != 0.0
    ):
        raise ValueError("privileged GT solver did not certify zero MIP gap")
    positions = tuple(int(value) for value in family["positions"])
    expected_objectives = _ground_truth_objectives_for_positions(
        positions,
        input_row.get("gt_segments", []),
        valid_len=int(input_row["valid_len"]),
        objective_spec=objective_spec,
        uniform_positions=uniform_positions,
    )
    _assert_numeric_tree_close(
        expected_objectives,
        solver.get("objective_vector"),
        context=f"{family['family_key']}.gt_solver.objective_vector",
    )
    envelopes = solver.get("metric_upper_envelopes")
    if not isinstance(envelopes, Mapping):
        raise ValueError("GT metric envelope payload must be an object")
    canonical_names = [
        name
        for name in expected_objectives
        if not name.startswith("lex_block_")
    ]
    if envelopes and set(envelopes) != set(canonical_names):
        raise ValueError("GT metric envelope keys differ from the canonical objectives")
    minimize_names = {"total_endpoint_distance_q", "selected_background"}
    for name, envelope in envelopes.items():
        if not isinstance(envelope, (int, float)) or not math.isfinite(float(envelope)):
            raise ValueError(f"GT metric envelope {name} must be finite")
        canonical = expected_objectives[name]
        if name in minimize_names and canonical < int(round(float(envelope))):
            raise ValueError(f"GT canonical objective beats its claimed minimum envelope: {name}")
        if name not in minimize_names and canonical > int(round(float(envelope))):
            raise ValueError(f"GT canonical objective beats its claimed maximum envelope: {name}")


def _ground_truth_objectives_for_positions(
    positions: Sequence[int],
    gt_segments: Sequence[Sequence[float | int]],
    *,
    valid_len: int,
    objective_spec: GroundTruthObjectiveSpec,
    uniform_positions: Sequence[int],
) -> dict[str, int]:
    selected = tuple(int(value) for value in positions)
    upper = float(valid_len - 1)
    segments_list: list[tuple[float, float]] = []
    for index, segment in enumerate(gt_segments):
        start, end = float(segment[0]), float(segment[1])
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0.0
            or end > upper
            or end < start
        ):
            raise ValueError(
                f"GT segment {index} is outside dense valid prefix [0,{upper}]"
            )
        segments_list.append((start, end))
    segments = tuple(segments_list)
    endpoints = tuple(value for segment in segments for value in segment)
    result: dict[str, int] = {}
    for radius in objective_spec.boundary_radii:
        endpoint_hits = [
            any(abs(float(position) - endpoint) <= radius + 1.0e-9 for position in selected)
            for endpoint in endpoints
        ]
        result[f"both_endpoints_r{radius}"] = sum(
            endpoint_hits[2 * index] and endpoint_hits[2 * index + 1]
            for index in range(len(segments))
        )
    for radius in objective_spec.boundary_radii:
        result[f"distinct_endpoint_hits_r{radius}"] = sum(
            any(abs(float(position) - endpoint) <= radius + 1.0e-9 for position in selected)
            for endpoint in endpoints
        )
    result["total_endpoint_distance_q"] = sum(
        min(
            int(round(abs(float(position) - endpoint) * objective_spec.distance_scale))
            for position in selected
        )
        for endpoint in endpoints
    )
    result["short_action_support"] = sum(
        any(start - 1.0e-9 <= position <= end + 1.0e-9 for position in selected)
        for start, end in segments
        if end - start <= objective_spec.short_action_max_length + 1.0e-9
    )
    result["selected_background"] = sum(
        not any(start - 1.0e-9 <= position <= end + 1.0e-9 for start, end in segments)
        for position in selected
    )
    result["exact_uniform_overlap"] = len(set(selected) & set(uniform_positions))
    for block_start in range(0, valid_len, objective_spec.lex_block_size):
        block_end = min(valid_len, block_start + objective_spec.lex_block_size)
        result[f"lex_block_{block_start:04d}_{block_end:04d}"] = sum(
            1 << (block_end - position - 1)
            for position in selected
            if block_start <= position < block_end
        )
    return result


def _assert_numeric_tree_close(expected: Any, actual: Any, *, context: str) -> None:
    if isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(f"{context}: boolean mismatch")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ValueError(f"{context}: object keys mismatch")
        for key in expected:
            _assert_numeric_tree_close(expected[key], actual[key], context=f"{context}.{key}")
        return
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(expected) != len(actual):
            raise ValueError(f"{context}: sequence mismatch")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_numeric_tree_close(left, right, context=f"{context}[{index}]")
        return
    if isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{context}: numeric value required")
        if not math.isclose(float(expected), float(actual), rel_tol=1.0e-9, abs_tol=1.0e-9):
            raise ValueError(f"{context}: numeric mismatch")
        return
    if expected != actual:
        raise ValueError(f"{context}: value mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a DUCA allocation-ceiling artifact fail closed.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--validation-json")
    args = parser.parse_args(argv)
    result = validate_artifact(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
    )
    if args.validation_json:
        write_json_exclusive(args.validation_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
