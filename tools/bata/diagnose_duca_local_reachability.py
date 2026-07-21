from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from tools.bata.analyze_duca_selection_quality import (
    RECORD_SCHEMA_VERSION,
    _boundaries,
    _selection_metrics,
    _validated_segments,
    analyze_record,
)
from tools.bata.duca_allocation_families import (
    PhysicalAxis,
    exact_uniform_positions,
    physical_gap_report,
    resolve_physical_cap,
    uniform_cell_bounds,
    validate_physical_selection,
)
from tools.bata.duca_exact_physical_solver import (
    GroundTruthObjectiveSpec,
    solve_additive_one_per_cell_physical,
    solve_ground_truth_lexicographic,
)


OUTPUT_SCHEMA_VERSION = "duca_local_reachability_record_v1"
SUMMARY_SCHEMA_VERSION = "duca_local_reachability_summary_v1"
FAMILY_KEYS = (
    "U_exact_uniform",
    "D_pure_delta_one_per_cell",
    "C_current_checkpoint",
    "L_privileged_local_gt_oracle",
    "G_privileged_global_gt_oracle",
)
SCORE_KEYS = (
    "p_action",
    "actionness_logits",
    "transition_policy_scores",
    "raw_transition_scores",
    "abs_delta_p_action",
    "uncertainty",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_mean(values: Sequence[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else mean(finite)


def _read_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"{path}:{line_number}: missing or duplicate sample_id")
            sample_ids.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def _load_holdout_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("holdout manifest must be an object")
    required = {
        "schema": "duca_frontend_train_holdout_split_v1",
        "source_subset": "training",
        "test_subset_consumed": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"holdout manifest contract mismatch: {key}")
    holdout = payload.get("holdout_videos")
    train = payload.get("train_videos")
    if not isinstance(holdout, list) or not holdout or not isinstance(train, list):
        raise ValueError("holdout manifest video partitions are invalid")
    if set(map(str, holdout)) & set(map(str, train)):
        raise ValueError("holdout and train video partitions overlap")
    return payload


def _validate_quality_record(row: Mapping[str, Any], holdout_videos: set[str]) -> None:
    sample_id = str(row.get("sample_id", ""))
    if row.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError(f"{sample_id}: unsupported quality-record schema")
    video_id = str(row.get("video_id", ""))
    if video_id not in holdout_videos:
        raise ValueError(f"{sample_id}: record is outside the training holdout")
    valid_len = int(row.get("valid_len", 0))
    budget = int(row.get("budget", 0))
    if valid_len < 1 or budget < 1 or budget > valid_len:
        raise ValueError(f"{sample_id}: invalid valid_len/budget")
    for key in SCORE_KEYS:
        values = row.get(key)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != valid_len
            or any(not math.isfinite(float(value)) for value in values)
        ):
            raise ValueError(f"{sample_id}: invalid score vector {key}")
    selected = tuple(int(value) for value in row.get("selected_positions", ()))
    if len(selected) != budget or tuple(sorted(set(selected))) != selected:
        raise ValueError(f"{sample_id}: current selection is not ordered exact-K")
    if selected[0] < 0 or selected[-1] >= valid_len:
        raise ValueError(f"{sample_id}: current selection is outside valid prefix")
    source = row.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{sample_id}: source provenance is missing")
    if source.get("detector_backbone_executed") is not False:
        raise ValueError(f"{sample_id}: detector executed during selector export")
    if source.get("uses_gt_for_selection") is not False:
        raise ValueError(f"{sample_id}: deploy-visible selection used GT")
    if row.get("gt_role") != "evaluation_only_not_selector_input":
        raise ValueError(f"{sample_id}: GT role is not evaluation-only")


def _solver_segments(
    segments: Sequence[tuple[float, float]],
    valid_len: int,
) -> tuple[tuple[float, float], ...]:
    upper = float(valid_len - 1)
    return tuple(
        (
            min(upper, max(0.0, float(start))),
            min(upper, max(0.0, float(end))),
        )
        for start, end in segments
    )


def _cap_compliant(axis: PhysicalAxis, positions: Sequence[int], cap) -> bool:
    report = physical_gap_report(axis, positions)
    frame_ok = (
        cap.max_source_frame_interval is None
        or report.source_frame_max_interval <= cap.max_source_frame_interval + 1.0e-9
    )
    seconds_ok = (
        cap.max_seconds_interval is None
        or report.seconds_max_interval <= cap.max_seconds_interval + 1.0e-9
    )
    return bool(frame_ok and seconds_ok)


def _family_payload(
    *,
    key: str,
    positions: Sequence[int],
    valid_len: int,
    budget: int,
    segments: Sequence[tuple[float, float]],
    boundaries: Sequence[float],
    uniform_positions: Sequence[int],
    cap,
    deployable: bool,
    privileged: bool,
    solver: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = tuple(int(value) for value in positions)
    if len(selected) != budget or tuple(sorted(set(selected))) != selected:
        raise ValueError(f"{key}: selection is not ordered exact-K")
    axis = PhysicalAxis.from_source_frames(
        range(valid_len), decoder_fps=1.0, annotation_fps=1.0
    )
    metrics = _selection_metrics(
        valid_len=valid_len,
        positions=selected,
        segments=segments,
        boundaries=boundaries,
    )
    metrics["uniform_overlap"] = len(set(selected) & set(uniform_positions)) / float(budget)
    return {
        "family_key": key,
        "positions": selected,
        "budget": budget,
        "deployable": bool(deployable),
        "privileged": bool(privileged),
        "physical_cap_compliant": _cap_compliant(axis, selected, cap),
        "gap_report": physical_gap_report(axis, selected).to_dict(),
        "selection_metrics": metrics,
        "solver": None if solver is None else dict(solver),
    }


def _primary_objective_key(
    vector: Mapping[str, int],
    spec: GroundTruthObjectiveSpec,
) -> tuple[int, ...]:
    key: list[int] = []
    key.extend(-int(vector[f"both_endpoints_r{radius}"]) for radius in spec.boundary_radii)
    key.extend(-int(vector[f"distinct_endpoint_hits_r{radius}"]) for radius in spec.boundary_radii)
    key.append(int(vector["total_endpoint_distance_q"]))
    key.append(-int(vector["short_action_support"]))
    key.append(int(vector["selected_background"]))
    key.append(-int(vector["exact_uniform_overlap"]))
    return tuple(key)


def diagnose_record(
    row: Mapping[str, Any],
    *,
    holdout_videos: set[str],
    objective_spec: GroundTruthObjectiveSpec,
    gt_time_limit_seconds: float | None,
) -> dict[str, Any]:
    _validate_quality_record(row, holdout_videos)
    valid_len = int(row["valid_len"])
    budget = int(row["budget"])
    axis = PhysicalAxis.from_source_frames(
        range(valid_len), decoder_fps=1.0, annotation_fps=1.0
    )
    cap = resolve_physical_cap(axis, requested_budget=budget)
    uniform = exact_uniform_positions(valid_len, budget)
    validate_physical_selection(axis, uniform, requested_budget=budget, cap=cap)
    _anchors, starts, ends = uniform_cell_bounds(valid_len, budget)
    cell_bounds = tuple(zip(starts, ends))
    local_delta = solve_additive_one_per_cell_physical(
        axis,
        row["abs_delta_p_action"],
        requested_budget=budget,
        cap=cap,
        one_per_cell_bounds=cell_bounds,
    )
    validate_physical_selection(
        axis,
        local_delta.positions,
        requested_budget=budget,
        cap=cap,
    )
    current = tuple(int(value) for value in row["selected_positions"])
    segments = _validated_segments(row, valid_len)
    boundaries = _boundaries(valid_len, segments)
    solver_segments = _solver_segments(segments, valid_len)
    local_oracle = solve_ground_truth_lexicographic(
        axis,
        solver_segments,
        requested_budget=budget,
        cap=cap,
        objective_spec=objective_spec,
        one_per_cell_bounds=cell_bounds,
        time_limit_seconds=gt_time_limit_seconds,
    )
    global_oracle = solve_ground_truth_lexicographic(
        axis,
        solver_segments,
        requested_budget=budget,
        cap=cap,
        objective_spec=objective_spec,
        time_limit_seconds=gt_time_limit_seconds,
    )
    if _primary_objective_key(global_oracle.objective_vector, objective_spec) > _primary_objective_key(
        local_oracle.objective_vector, objective_spec
    ):
        raise RuntimeError("global GT oracle is worse than its local feasible subset")
    families = {
        "U_exact_uniform": _family_payload(
            key="U_exact_uniform",
            positions=uniform,
            valid_len=valid_len,
            budget=budget,
            segments=segments,
            boundaries=boundaries,
            uniform_positions=uniform,
            cap=cap,
            deployable=True,
            privileged=False,
        ),
        "D_pure_delta_one_per_cell": _family_payload(
            key="D_pure_delta_one_per_cell",
            positions=local_delta.positions,
            valid_len=valid_len,
            budget=budget,
            segments=segments,
            boundaries=boundaries,
            uniform_positions=uniform,
            cap=cap,
            deployable=True,
            privileged=False,
        ),
        "C_current_checkpoint": _family_payload(
            key="C_current_checkpoint",
            positions=current,
            valid_len=valid_len,
            budget=budget,
            segments=segments,
            boundaries=boundaries,
            uniform_positions=uniform,
            cap=cap,
            deployable=True,
            privileged=False,
        ),
        "L_privileged_local_gt_oracle": _family_payload(
            key="L_privileged_local_gt_oracle",
            positions=local_oracle.positions,
            valid_len=valid_len,
            budget=budget,
            segments=segments,
            boundaries=boundaries,
            uniform_positions=uniform,
            cap=cap,
            deployable=False,
            privileged=True,
            solver=local_oracle.to_dict(),
        ),
        "G_privileged_global_gt_oracle": _family_payload(
            key="G_privileged_global_gt_oracle",
            positions=global_oracle.positions,
            valid_len=valid_len,
            budget=budget,
            segments=segments,
            boundaries=boundaries,
            uniform_positions=uniform,
            cap=cap,
            deployable=False,
            privileged=True,
            solver=global_oracle.to_dict(),
        ),
    }
    analyzed = analyze_record(row)
    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "valid_len": valid_len,
        "budget": budget,
        "gt_segment_count": len(segments),
        "cap": cap.to_dict(),
        "coarse_actionness": analyzed["coarse"],
        "transition_quality": analyzed["transition"],
        "families": families,
        "contract": {
            "task": "offline_tad",
            "axis": "selected_dense_ordinal",
            "exact_k_matched": True,
            "uniform_reference_cap_matched": True,
            "deployable_families_use_gt": False,
            "local_oracle_uses_gt": True,
            "global_oracle_uses_gt": True,
            "oracles_are_evaluation_only": True,
            "detector_map_oracle_claim": False,
        },
    }
    output["record_sha256"] = _canonical_sha256(output)
    return output


def _summary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family_key in FAMILY_KEYS:
        families = [row["families"][family_key] for row in rows]
        summary[family_key] = {
            "sample_count": len(families),
            "cap_compliance_fraction": _finite_mean(
                [float(family["physical_cap_compliant"]) for family in families]
            ),
            "mean_max_unselected_hole": _finite_mean(
                [family["selection_metrics"]["max_unselected_hole"] for family in families]
            ),
            "mean_endpoint_distance": _finite_mean(
                [family["selection_metrics"]["mean_endpoint_distance"] for family in families]
            ),
            "p90_endpoint_distance_mean": _finite_mean(
                [family["selection_metrics"]["p90_endpoint_distance"] for family in families]
            ),
            "boundary_recall": {
                f"r{radius}": _finite_mean(
                    [family["selection_metrics"]["boundary_recall"][f"r{radius}"] for family in families]
                )
                for radius in (0, 1, 2, 4, 8)
            },
            "both_endpoint_coverage": {
                f"r{radius}": _finite_mean(
                    [
                        family["selection_metrics"]["both_endpoint_coverage"][f"r{radius}"]
                        for family in families
                    ]
                )
                for radius in (0, 1, 2, 4, 8)
            },
            "uniform_overlap": _finite_mean(
                [family["selection_metrics"]["uniform_overlap"] for family in families]
            ),
        }
    return summary


def run_diagnostic(
    *,
    input_jsonl: str | Path,
    split_manifest: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    objective_spec: GroundTruthObjectiveSpec,
    gt_time_limit_seconds: float | None = None,
    limit_records: int = 0,
) -> dict[str, Any]:
    input_path = Path(input_jsonl).resolve()
    manifest_path = Path(split_manifest).resolve()
    output_path = Path(output_jsonl).resolve()
    summary_path = Path(summary_json).resolve()
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("local reachability diagnostic never overwrites artifacts")
    manifest = _load_holdout_manifest(manifest_path)
    holdout_videos = set(map(str, manifest["holdout_videos"]))
    records = _read_records(input_path)
    if int(limit_records) > 0:
        records = records[: int(limit_records)]
    outputs = []
    for index, row in enumerate(records):
        sample_id = str(row.get("sample_id", f"index_{index}"))
        try:
            outputs.append(
                diagnose_record(
                    row,
                    holdout_videos=holdout_videos,
                    objective_spec=objective_spec,
                    gt_time_limit_seconds=gt_time_limit_seconds,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"local reachability failed at record {index} sample_id={sample_id}"
            ) from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial output exists: {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    temporary.replace(output_path)
    video_counts: dict[str, int] = defaultdict(int)
    for row in outputs:
        video_counts[str(row["video_id"])] += 1
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "diagnostic_role": "local_geometry_reachability_not_detector_map_oracle",
        "sample_count": len(outputs),
        "video_count": len(video_counts),
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": _sha256(input_path),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": _sha256(manifest_path),
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": _sha256(output_path),
        "objective_spec": {
            "boundary_radii": objective_spec.boundary_radii,
            "short_action_max_length": objective_spec.short_action_max_length,
            "distance_scale": objective_spec.distance_scale,
            "lex_block_size": objective_spec.lex_block_size,
            "position_tie_break": objective_spec.position_tie_break,
        },
        "families": _summary_metrics(outputs),
        "coarse_actionness": {
            "auroc": _finite_mean([row["coarse_actionness"]["auroc"] for row in outputs]),
            "auprc": _finite_mean([row["coarse_actionness"]["auprc"] for row in outputs]),
            "brier": _finite_mean([row["coarse_actionness"]["brier"] for row in outputs]),
        },
        "contract": {
            "training_holdout_only": True,
            "test_subset_consumed": False,
            "exact_k_matched": True,
            "uniform_reference_cap_matched": True,
            "privileged_oracles_deployable": False,
            "detector_map_evaluated": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure DUCA local-cell reachability against matched exact GT ceilings."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--boundary-radii", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--short-action-max-length", type=float, default=16.0)
    parser.add_argument("--distance-scale", type=int, default=1000)
    # Keep binary tie-break coefficients small enough for HiGHS to preserve
    # exact bounds after many sequentially pinned objectives.
    parser.add_argument("--lex-block-size", type=int, default=8)
    parser.add_argument(
        "--position-tie-break",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resolve semantic-optimal ties by position. Disabled by default because "
            "reachability depends only on the pinned semantic objective vector."
        ),
    )
    parser.add_argument("--gt-time-limit-seconds", type=float)
    parser.add_argument("--limit-records", type=int, default=0)
    args = parser.parse_args(argv)
    summary = run_diagnostic(
        input_jsonl=args.input_jsonl,
        split_manifest=args.split_manifest,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        objective_spec=GroundTruthObjectiveSpec(
            boundary_radii=tuple(args.boundary_radii),
            short_action_max_length=args.short_action_max_length,
            distance_scale=args.distance_scale,
            lex_block_size=args.lex_block_size,
            position_tie_break=args.position_tie_break,
        ),
        gt_time_limit_seconds=args.gt_time_limit_seconds,
        limit_records=args.limit_records,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
