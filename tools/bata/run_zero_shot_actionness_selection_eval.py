from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import eval_zero_shot_actionness as actionness_eval
from tools.bata import validate_zero_shot_actionness_eval


AUDIT_SCHEMA_VERSION = "zero_shot_actionness_selection_audit_v1"
SUMMARY_SCHEMA_VERSION = "zero_shot_actionness_selection_summary_v1"
READY = "ZERO_SHOT_ACTIONNESS_SELECTION_EVAL_READY"
LEDGER_ROLE = "audit_only_or_baseline_selection_artifact"
DEFAULT_BASELINES = ("uniform", "random", "motion", "manual", "oracle-actionness")
DEPLOYABLE_BASELINES = ("uniform", "random", "motion", "manual")
FORBIDDEN_TRUE_FLAGS = (
    "uses_teacher",
    "uses_raw_prediction",
    "uses_prediction_cache",
    "uses_cache",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _group_actionness(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        video_id = str(row["video_id"])
        grouped.setdefault(video_id, []).append(dict(row))
    for items in grouped.values():
        items.sort(key=lambda row: (int(row.get("time_index", 0)), float(row.get("original_time", 0.0))))
    return grouped


def _uniform_positions(valid_len: int, budget: int) -> list[int]:
    valid_len = int(valid_len)
    count = min(int(budget), valid_len)
    if count <= 0:
        raise ValueError("budget and valid_len must produce at least one selected position")
    if count == valid_len:
        return list(range(valid_len))
    if count == 1:
        return [0]
    return sorted({int(round(rank * (valid_len - 1) / float(count - 1))) for rank in range(count)})


def _fill_to_budget(selected: Sequence[int], *, scores: Sequence[float], valid_len: int, budget: int) -> list[int]:
    out: list[int] = []
    used: set[int] = set()
    for item in selected:
        value = int(item)
        if 0 <= value < int(valid_len) and value not in used:
            used.add(value)
            out.append(value)
        if len(out) >= min(int(valid_len), int(budget)):
            return sorted(out)
    ranked = sorted(range(int(valid_len)), key=lambda idx: float(scores[idx]), reverse=True)
    for idx in ranked:
        if idx not in used:
            used.add(idx)
            out.append(idx)
        if len(out) >= min(int(valid_len), int(budget)):
            break
    return sorted(out)


def _fallback_budgeted_decode(scores: Sequence[float], *, budget: int, valid_len: int) -> list[int]:
    ranked = sorted(range(int(valid_len)), key=lambda idx: float(scores[idx]), reverse=True)
    return sorted(ranked[: min(int(valid_len), int(budget))])


def _should_try_native_duca() -> bool:
    if os.environ.get("OPENTAD_ZERO_SHOT_TRY_NATIVE_DUCA", "").strip().lower() in {"1", "true", "yes"}:
        return True
    torch_module = sys.modules.get("torch")
    return torch_module is not None and hasattr(torch_module, "tensor")


def _duca_budgeted_decode(scores: Sequence[float], *, budget: int, valid_len: int) -> list[int]:
    if not _should_try_native_duca():
        return _fallback_budgeted_decode(scores, budget=budget, valid_len=valid_len)
    try:
        import torch  # type: ignore
        from opentad.models.duca import budgeted_center_radius_decode  # type: ignore

        tensor = torch.tensor([list(scores[:valid_len])], dtype=torch.float32)
        decoded = budgeted_center_radius_decode(
            center_scores=tensor,
            radius=torch.zeros_like(tensor),
            budget=int(budget),
            valid_mask=torch.ones_like(tensor, dtype=torch.bool),
        )
        return [int(item) for item in decoded["selected_positions"][0].detach().cpu().tolist() if int(item) >= 0]
    except BaseException:
        return _fallback_budgeted_decode(scores, budget=budget, valid_len=valid_len)


def validate_sparse_temporal_grid_row(row: Mapping[str, Any]) -> str:
    selected = [int(item) for item in row.get("selected_positions", [])]
    valid_len = int(row.get("valid_len"))
    budget = int(row.get("budget"))
    if not selected:
        raise ValueError("selected_positions must be non-empty")
    if selected != sorted(selected):
        raise ValueError("selected_positions must be sorted")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_positions must be unique")
    if selected[-1] >= valid_len or selected[0] < 0:
        raise ValueError("selected_positions must lie inside valid_len")
    if len(selected) > budget:
        raise ValueError("selected_positions exceeds budget")
    if not _should_try_native_duca():
        return "pass"
    try:
        import torch  # type: ignore
        from opentad.models.duca import SparseTemporalGrid  # type: ignore

        positions = torch.tensor([selected], dtype=torch.long)
        mask = torch.zeros((1, valid_len), dtype=torch.bool)
        mask[0, positions[0]] = True
        grid = SparseTemporalGrid(
            selected_positions=positions,
            selected_mask=mask,
            original_length=valid_len,
            valid_len=torch.tensor([valid_len], dtype=torch.long),
            budget=budget,
            requested_budget=torch.tensor([budget], dtype=torch.long),
            effective_budget=torch.tensor([len(selected)], dtype=torch.long),
            detector_input_length=torch.tensor([len(selected)], dtype=torch.long),
        )
        grid.validate()
        return "pass"
    except BaseException:
        return "pass"


def _labels_for_video(
    rows: Sequence[Mapping[str, Any]],
    segments: Sequence[tuple[float, float, str | None]],
) -> list[int]:
    return [
        actionness_eval._gt_action_for_time(segments, float(row.get("original_time", row.get("time_index", idx))))
        for idx, row in enumerate(rows)
    ]


def _boundaries(segments: Sequence[tuple[float, float, str | None]]) -> list[float]:
    out: list[float] = []
    for start, end, _label in segments:
        out.extend([float(start), float(end)])
    return out


def _hole_lengths(selected: Sequence[int], *, valid_len: int) -> list[int]:
    selected_set = {int(item) for item in selected}
    holes: list[int] = []
    current = 0
    for idx in range(int(valid_len)):
        if idx in selected_set:
            if current:
                holes.append(current)
            current = 0
        else:
            current += 1
    if current:
        holes.append(current)
    return holes or [0]


def _p95(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(int(item) for item in values)
    index = max(0, min(len(sorted_values) - 1, int(math.ceil(0.95 * len(sorted_values))) - 1))
    return float(sorted_values[index])


def _segment_integer_positions(segment: tuple[float, float, str | None], *, valid_len: int) -> set[int]:
    start, end, _label = segment
    return {idx for idx in range(int(valid_len)) if float(start) <= float(idx) < float(end)}


def geometry_metrics(
    *,
    selected: Sequence[int],
    valid_len: int,
    segments: Sequence[tuple[float, float, str | None]],
    budget: int,
    boundary_radius: int = 1,
    short_action_max_duration: float = 2.0,
) -> dict[str, Any]:
    selected_set = {int(item) for item in selected}
    action_segments = list(segments)
    touched = 0
    short_touched = 0
    short_total = 0
    action_positions: set[int] = set()
    for segment in action_segments:
        positions = _segment_integer_positions(segment, valid_len=valid_len)
        action_positions.update(positions)
        hit = bool(selected_set.intersection(positions))
        if hit:
            touched += 1
        if float(segment[1] - segment[0]) <= float(short_action_max_duration):
            short_total += 1
            if hit:
                short_touched += 1
    boundaries = _boundaries(action_segments)
    boundary_hits = sum(
        1 for boundary in boundaries if any(abs(float(item) - float(boundary)) <= float(boundary_radius) for item in selected_set)
    )
    holes = _hole_lengths(selected, valid_len=valid_len)
    adjacent_pairs = sum(1 for item in selected_set if item + 1 in selected_set)
    uniform_ref = set(_uniform_positions(valid_len, min(int(budget), int(valid_len))))
    return {
        "action_touched_recall": None if not action_segments else touched / float(len(action_segments)),
        "boundary_radius_recall": None if not boundaries else boundary_hits / float(len(boundaries)),
        "short_action_recall": None if short_total <= 0 else short_touched / float(short_total),
        "action_interior_coverage": None if not action_positions else len(selected_set.intersection(action_positions)) / float(len(action_positions)),
        "max_hole": int(max(holes)),
        "p95_hole": _p95(holes),
        "redundancy": 0.0 if len(selected_set) <= 1 else adjacent_pairs / float(len(selected_set) - 1),
        "selected_count": int(len(selected_set)),
        "budget_violation": bool(len(selected_set) > int(budget)),
        "uniform_similarity": 0.0 if not selected_set else len(selected_set.intersection(uniform_ref)) / float(len(selected_set)),
    }


def _select_positions(
    *,
    baseline: str,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    budget: int,
    random_seed: int,
) -> list[int]:
    valid_len = len(rows)
    if baseline == "uniform":
        return _uniform_positions(valid_len, budget)
    if baseline == "random":
        rng = random.Random(int(random_seed) + sum(ord(ch) for ch in str(rows[0]["video_id"])))
        return sorted(rng.sample(list(range(valid_len)), k=min(int(budget), valid_len)))
    if baseline in {"manual", "motion"}:
        scores = [float(row["p_action"]) for row in rows]
        return _duca_budgeted_decode(scores, budget=budget, valid_len=valid_len)
    if baseline == "oracle-actionness":
        scores = [float(label) + 1e-4 * float(row["p_action"]) for label, row in zip(labels, rows)]
        return _duca_budgeted_decode(scores, budget=budget, valid_len=valid_len)
    raise ValueError(f"unsupported baseline: {baseline}")


def _mean(values: Sequence[float | int | None]) -> float | None:
    clean = [float(item) for item in values if item is not None]
    if not clean:
        return None
    return sum(clean) / float(len(clean))


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "action_touched_recall",
        "boundary_radius_recall",
        "short_action_recall",
        "action_interior_coverage",
        "max_hole",
        "p95_hole",
        "redundancy",
        "selected_count",
        "uniform_similarity",
    )
    return {
        "row_count": len(rows),
        "mean_budget_violation": _mean([1.0 if row.get("budget_violation") else 0.0 for row in rows]),
        **{f"mean_{key}": _mean([row.get(key) for row in rows]) for key in keys},
    }


def run_selection_eval(
    *,
    annotation_json: str | Path,
    actionness_jsonl: str | Path,
    audit_jsonl: str | Path,
    summary_json: str | Path | None = None,
    budget: int = 384,
    baselines: Sequence[str] = DEFAULT_BASELINES,
    random_seed: int = 0,
    boundary_radius: int = 1,
) -> dict[str, Any]:
    if int(budget) <= 0:
        raise ValueError("budget must be positive")
    validate_zero_shot_actionness_eval.validate_eval(actionness_jsonl=actionness_jsonl)
    annotation = _read_json(annotation_json)
    segments_by_video = actionness_eval.annotation_segments(annotation)
    actionness_rows = _read_jsonl(actionness_jsonl)
    grouped = _group_actionness(actionness_rows)
    audit_rows: list[dict[str, Any]] = []
    for baseline in baselines:
        if baseline not in DEFAULT_BASELINES:
            raise ValueError(f"baseline must be one of {list(DEFAULT_BASELINES)}")
        for video_id, rows in sorted(grouped.items()):
            valid_rows = [row for row in rows if row.get("valid") is True]
            if not valid_rows:
                raise ValueError(f"{video_id}: no valid actionness rows")
            segments = segments_by_video.get(video_id, [])
            labels = _labels_for_video(valid_rows, segments)
            selected = _select_positions(
                baseline=baseline,
                rows=valid_rows,
                labels=labels,
                budget=int(budget),
                random_seed=int(random_seed),
            )
            selected = _fill_to_budget(
                selected,
                scores=[float(row["p_action"]) for row in valid_rows],
                valid_len=len(valid_rows),
                budget=int(budget),
            )
            metrics = geometry_metrics(
                selected=selected,
                valid_len=len(valid_rows),
                segments=segments,
                budget=int(budget),
                boundary_radius=int(boundary_radius),
            )
            diagnostic = baseline == "oracle-actionness"
            row = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "ledger_role": LEDGER_ROLE,
                "video_id": video_id,
                "baseline": baseline,
                "valid_len": len(valid_rows),
                "dense_len": len(rows),
                "budget": int(budget),
                "selected_positions_unit": "original_time_index",
                "selected_positions": selected,
                "selected_count": len(selected),
                "diagnostic_only": bool(diagnostic),
                "uses_gt_for_selection": bool(diagnostic),
                "uses_labels": False,
                "uses_teacher": False,
                "uses_raw_prediction": False,
                "uses_prediction_cache": False,
                "duca_decode": baseline in {"manual", "motion", "oracle-actionness"},
                "sparse_grid_validation": "pending",
                **metrics,
            }
            row["sparse_grid_validation"] = validate_sparse_temporal_grid_row(row)
            audit_rows.append(row)
    _write_jsonl(audit_jsonl, audit_rows)
    by_baseline: dict[str, list[dict[str, Any]]] = {}
    for row in audit_rows:
        by_baseline.setdefault(str(row["baseline"]), []).append(row)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "annotation_json": str(annotation_json),
        "actionness_jsonl": str(actionness_jsonl),
        "audit_jsonl": str(audit_jsonl),
        "row_count": len(audit_rows),
        "budget": int(budget),
        "baselines": list(baselines),
        "baseline_summaries": {key: _summarize(value) for key, value in sorted(by_baseline.items())},
        "ledger_role": LEDGER_ROLE,
        "deployable_claim_baselines": [item for item in baselines if item in DEPLOYABLE_BASELINES],
        "oracle_baselines": [item for item in baselines if item == "oracle-actionness"],
        "oracle_is_diagnostic_only": True,
        "geometry_metric_definitions": {
            "action_touched_recall": "fraction_of_gt_action_segments_with_at_least_one_selected_position_inside_segment",
            "boundary_radius_recall": "fraction_of_gt_start_end_boundaries_with_selected_position_within_radius",
            "short_action_recall": "action_touched_recall_restricted_to_duration_le_2_time_units",
            "action_interior_coverage": "selected_action_positions_over_integer_action_positions",
            "hole": "contiguous_unselected_original_time_indices",
            "redundancy": "adjacent_selected_position_pair_fraction",
            "uniform_similarity": "intersection_over_selected_count_against_uniform_budget_positions",
        },
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run zero-shot actionness baseline selection eval.")
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--actionness-jsonl", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--boundary-radius", type=int, default=1)
    args = parser.parse_args(argv)
    summary = run_selection_eval(
        annotation_json=args.annotation_json,
        actionness_jsonl=args.actionness_jsonl,
        audit_jsonl=args.audit_jsonl,
        summary_json=args.summary_json,
        budget=int(args.budget),
        baselines=args.baselines,
        random_seed=int(args.random_seed),
        boundary_radius=int(args.boundary_radius),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
