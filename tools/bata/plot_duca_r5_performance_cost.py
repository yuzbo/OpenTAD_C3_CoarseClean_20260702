from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence
from uuid import uuid4

from tools.bata.duca_full_stack_cost import validate_and_rebuild_profile_summary
from tools.bata.duca_p0_evaluation import official_evaluator_identity


AGGREGATE_SCHEMA = "duca_r5_paper_matrix_results_v1"
COMPLETE_STATUS = "r5_raw_evidence_complete_pending_claim_adjudication"
OUTPUT_SCHEMA = "duca_r5_performance_cost_v1"
IOU_KEYS = tuple(f"mAP@{threshold:.1f}" for threshold in (0.3, 0.4, 0.5, 0.6, 0.7))
UNAVAILABLE = "unavailable"
RAW_COLUMNS = (
    "row_kind",
    "id",
    "backend",
    "arm",
    "K",
    "seed",
    "Avg-mAP",
    *IOU_KEYS,
    "frontend_latency_ms_p50",
    "coarse_latency_ms_p50",
    "selector_latency_ms_p50",
    "backbone_latency_ms_p50",
    "detector_latency_ms_p50",
    "total_latency_ms_p50",
    "FLOPs",
    "peak_memory_mb_p50",
    "selected_count_p50",
    "speedup_vs_same_backend_dense",
    "cost_profile_path",
    "cost_profile_sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"DUCA R5 performance/cost export failed: {message}")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: str | Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DUCA R5 performance/cost export failed: {label} is invalid JSON") from exc
    _require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload, resolved


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _optional_p50(payload: Mapping[str, Any], *path: str) -> float | str:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return UNAVAILABLE
        value = value[key]
    if value is None:
        return UNAVAILABLE
    return _finite(value, ".".join(path))


def _validate_aggregate(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    unsigned = dict(payload)
    observed_hash = unsigned.pop("result_sha256", None)
    _require(observed_hash == _canonical_sha256(unsigned), "aggregate result self-hash drift")
    _require(payload.get("schema") == AGGREGATE_SCHEMA, "aggregate schema is not final R5")
    _require(payload.get("ok") is True, "aggregate is incomplete")
    _require(payload.get("status") == COMPLETE_STATUS, "aggregate status is incomplete")
    _require(payload.get("task") == "offline_temporal_action_detection", "aggregate task drift")
    rows = payload.get("rows")
    axes = payload.get("matrix_axes")
    if isinstance(axes, Mapping):
        backends = tuple(str(value) for value in axes.get("backends", ()))
        arms = tuple(str(value) for value in axes.get("arms", ()))
        budgets = tuple(int(value) for value in axes.get("budgets", ()))
        seeds = tuple(int(value) for value in axes.get("seeds", ()))
    else:
        backends = ("actionformer", "temporalmaxer")
        arms = ("uniform", "learned")
        budgets = (384, 256)
        seeds = (3407, 5801, 8123)
    expected_rows = len(backends) * len(arms) * len(budgets) * len(seeds)
    _require(
        backends and arms and budgets and seeds
        and isinstance(rows, list) and len(rows) == expected_rows,
        "aggregate lacks its complete declared matrix",
    )
    evaluator = official_evaluator_identity()
    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        _require(isinstance(row, Mapping), "aggregate contains a non-object row")
        backend = str(row.get("backend", ""))
        arm = str(row.get("arm", ""))
        budget = row.get("budget")
        seed = row.get("seed")
        _require(backend and arm in {"uniform", "learned"}, "aggregate row identity is invalid")
        _require(isinstance(budget, int) and budget in set(budgets), "aggregate row K is invalid")
        _require(isinstance(seed, int) and seed in set(seeds), "aggregate row seed is invalid")
        _require(row.get("evaluator") == evaluator, f"{backend}/{arm}/K{budget}/s{seed} is not official mAP")
        config = row.get("evaluation_config")
        _require(
            isinstance(config, Mapping)
            and config.get("subset") == "validation"
            and config.get("blocked_videos") is None,
            f"{backend}/{arm}/K{budget}/s{seed} is not a full official validation evaluation",
        )
        iou_map = row.get("iou_mAP")
        _require(isinstance(iou_map, Mapping), f"{backend}/{arm}/K{budget}/s{seed} lacks IoU-wise mAP")
        normalized_row = {
            "id": str(row.get("id", "")),
            "backend": backend,
            "arm": arm,
            "K": budget,
            "seed": seed,
            "Avg-mAP": _finite(row.get("average_mAP"), "average_mAP"),
            **{key: _finite(iou_map.get(key), key) for key in IOU_KEYS},
        }
        _require(normalized_row["id"], "aggregate row id is missing")
        normalized.append(normalized_row)
        grouped[(backend, arm, budget)].append(seed)
    _require(
        len(grouped) == len(backends) * len(arms) * len(budgets),
        "aggregate does not contain all backend/arm/K groups",
    )
    _require(
        all(len(group_seeds) == len(seeds) and set(group_seeds) == set(seeds) for group_seeds in grouped.values()),
        "aggregate lacks the declared independent seeds per backend/arm/K",
    )
    return normalized


def _profile_identity(payload: Mapping[str, Any], label: str) -> tuple[str, str, int]:
    session = str(payload.get("profile_session_id", "")).strip()
    pair = str(payload.get("profile_pair_id", "")).strip()
    order = payload.get("profile_order_position")
    _require(session and pair and order in {1, 2}, f"{label} lacks paired profile identity")
    return session, pair, int(order)


def _validate_raw_costs(
    aggregate: Mapping[str, Any], supplied_paths: Sequence[str | Path]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    costs = aggregate.get("costs")
    dense_costs = aggregate.get("paired_dense_costs")
    _require(isinstance(costs, list) and isinstance(dense_costs, list), "aggregate lacks paired raw cost references")
    _require(
        len(costs) == len(dense_costs) and len(costs) > 0,
        "aggregate lacks matched candidate/dense cost profiles",
    )
    expected: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for row in costs:
        _require(isinstance(row, Mapping), "aggregate candidate cost reference is invalid")
        path = str(Path(str(row.get("summary_path", ""))).resolve())
        expected[path] = ("candidate", row)
    for row in dense_costs:
        _require(isinstance(row, Mapping), "aggregate dense cost reference is invalid")
        path = str(Path(str(row.get("summary_path", ""))).resolve())
        _require(path not in expected, "aggregate repeats a raw cost profile")
        expected[path] = ("dense", row)
    supplied = {str(Path(path).expanduser().resolve()) for path in supplied_paths}
    _require(supplied == set(expected), "supplied raw cost summaries must exactly match the aggregate paired summaries")

    candidates: dict[str, dict[str, Any]] = {}
    dense_rows: list[dict[str, Any]] = []
    for path_text in sorted(supplied):
        kind, reference = expected[path_text]
        profile, path = _load_json(path_text, "raw cost summary")
        _require(_sha256(path) == reference.get("summary_sha256"), f"raw cost summary hash drift: {path}")
        try:
            validate_and_rebuild_profile_summary(profile)
        except ValueError as exc:
            raise RuntimeError(f"DUCA R5 performance/cost export failed: raw cost summary is not reproducible: {path}") from exc
        _require(profile.get("random_init") is False and profile.get("uses_ema") is True, f"raw cost summary is not a terminal EMA measurement: {path}")
        session, pair, order = _profile_identity(profile, str(path))
        record = {
            "profile": profile,
            "path": str(path),
            "sha256": _sha256(path),
            "session": session,
            "pair": pair,
            "order": order,
            "reference": reference,
        }
        if kind == "candidate":
            cell_id = str(reference.get("source_cell", ""))
            _require(profile.get("method") == cell_id, f"candidate cost source-cell drift: {path}")
            binding = profile.get("r5_cost_binding")
            _require(isinstance(binding, Mapping) and binding.get("method") == cell_id, f"candidate cost lacks terminal R5 binding: {path}")
            _require(cell_id not in candidates, f"duplicate candidate cost for {cell_id}")
            candidates[cell_id] = record
        else:
            _require(profile.get("method") == "dense-adatad", f"dense cost method drift: {path}")
            dense_rows.append(record)
    _require(
        len(candidates) == len(dense_rows) == len(costs),
        "raw candidate/dense profile count drift",
    )
    return candidates, dense_rows


def _cost_fields(profile: Mapping[str, Any]) -> dict[str, float | str]:
    return {
        "frontend_latency_ms_p50": _optional_p50(profile, "stages", "frame_selector_total_ms", "p50"),
        "coarse_latency_ms_p50": _optional_p50(profile, "stages", "coarse_probe_ms", "p50"),
        "selector_latency_ms_p50": _optional_p50(profile, "stages", "selector_policy_ms", "p50"),
        "backbone_latency_ms_p50": _optional_p50(profile, "stages", "heavy_backbone_ms", "p50"),
        "detector_latency_ms_p50": _optional_p50(profile, "stages", "model_forward_ms", "p50"),
        "total_latency_ms_p50": _optional_p50(profile, "stages", "end_to_end_serial_ms", "p50"),
        "FLOPs": UNAVAILABLE,
        "peak_memory_mb_p50": _optional_p50(profile, "resources", "peak_gpu_memory_mb", "p50"),
        "selected_count_p50": _optional_p50(profile, "selected_count", "p50"),
    }


def _pair_dense_costs(
    aggregate: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    dense_backend = str(aggregate.get("dense_baseline_receipt", {}).get("backend", ""))
    _require(dense_backend, "aggregate dense backend is missing")
    by_identity = {(row["session"], row["pair"]): row for row in dense_rows}
    _require(len(by_identity) == len(dense_rows), "dense cost pair identities are duplicated")
    paired: dict[str, Mapping[str, Any]] = {}
    for cell_id, candidate in candidates.items():
        dense = by_identity.get((candidate["session"], candidate["pair"]))
        _require(dense is not None, f"{cell_id} has no same-session dense profile")
        _require({candidate["order"], dense["order"]} == {1, 2}, f"{cell_id} pair order is invalid")
        reference = candidate["reference"]
        _require(reference.get("backend") == dense_backend, f"{cell_id} cross-backend dense ratio is forbidden")
        dense_backend_reference = dense["reference"].get("backend")
        _require(dense_backend_reference == dense_backend, f"{cell_id} dense backend identity drift")
        candidate_latency = _cost_fields(candidate["profile"])["total_latency_ms_p50"]
        dense_latency = _cost_fields(dense["profile"])["total_latency_ms_p50"]
        _require(isinstance(candidate_latency, float) and candidate_latency > 0.0, f"{cell_id} lacks positive total latency")
        _require(isinstance(dense_latency, float) and dense_latency > 0.0, f"{cell_id} dense profile lacks positive total latency")
        paired[cell_id] = dense
    return paired


def build_performance_cost_report(
    *, aggregate_json: str | Path, raw_cost_summaries: Sequence[str | Path]
) -> dict[str, Any]:
    aggregate, aggregate_path = _load_json(aggregate_json, "final aggregate JSON")
    performance_rows = _validate_aggregate(aggregate)
    candidates, dense_rows = _validate_raw_costs(aggregate, raw_cost_summaries)
    paired_dense = _pair_dense_costs(aggregate, candidates, dense_rows)
    raw_rows: list[dict[str, Any]] = []
    for row in performance_rows:
        record = {"row_kind": "r5_cell", **row}
        record.update({column: UNAVAILABLE for column in RAW_COLUMNS if column not in record})
        candidate = candidates.get(row["id"])
        if candidate is not None:
            record.update(_cost_fields(candidate["profile"]))
            dense = paired_dense[row["id"]]
            dense_latency = _cost_fields(dense["profile"])["total_latency_ms_p50"]
            record["speedup_vs_same_backend_dense"] = dense_latency / record["total_latency_ms_p50"]
            record["cost_profile_path"] = candidate["path"]
            record["cost_profile_sha256"] = candidate["sha256"]
        raw_rows.append(record)
    for dense in dense_rows:
        reference = dense["reference"]
        record = {
            "row_kind": "dense_profile",
            "backend": reference["backend"],
            "arm": "dense",
            "K": 768,
            "seed": 3407,
            "Avg-mAP": UNAVAILABLE,
            **{key: UNAVAILABLE for key in IOU_KEYS},
            **_cost_fields(dense["profile"]),
            "speedup_vs_same_backend_dense": UNAVAILABLE,
            "cost_profile_path": dense["path"],
            "cost_profile_sha256": dense["sha256"],
        }
        raw_rows.append(record)

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in performance_rows:
        groups[(row["backend"], row["arm"], row["K"])].append(row)
    aggregates = []
    for (backend, arm, budget), rows in sorted(groups.items()):
        _require(len(rows) >= 2, f"{backend}/{arm}/K{budget} lacks repeated seeds")
        summary = {
            "backend": backend,
            "arm": arm,
            "K": budget,
            "seed_count": len(rows),
            "seeds": sorted(row["seed"] for row in rows),
        }
        for metric in ("Avg-mAP", *IOU_KEYS):
            values = [float(row[metric]) for row in rows]
            summary[metric] = {"mean": mean(values), "std": stdev(values)}
        aggregates.append(summary)
    report = {
        "schema": OUTPUT_SCHEMA,
        "aggregate_json_path": str(aggregate_path),
        "aggregate_json_sha256": _sha256(aggregate_path),
        "performance_rows": raw_rows,
        "performance_aggregates": aggregates,
        "unavailable_sentinel": UNAVAILABLE,
        "latency_semantics": {
            "frontend_latency_ms_p50": "frame_selector_total_ms; includes coarse and selector policy",
            "coarse_latency_ms_p50": "coarse_probe_ms; nested in frontend",
            "selector_latency_ms_p50": "selector_policy_ms; frontend minus coarse",
            "backbone_latency_ms_p50": "heavy_backbone_ms; nested in detector forward",
            "detector_latency_ms_p50": "model_forward_ms; raw detector forward",
            "total_latency_ms_p50": "end_to_end_serial_ms; input, H2D, forward, postprocess",
            "FLOPs": "unavailable because paired R5 runtime summaries do not measure FLOPs",
        },
        "plot_status": "not_requested",
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]], delimiter: str) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, delimiter=delimiter, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pareto_indices(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    selected = []
    for index, row in enumerate(rows):
        latency = float(row["total_latency_ms_p50"])
        score = float(row["Avg-mAP"])
        dominated = any(
            other_index != index
            and float(other["total_latency_ms_p50"]) <= latency
            and float(other["Avg-mAP"]) >= score
            and (float(other["total_latency_ms_p50"]) < latency or float(other["Avg-mAP"]) > score)
            for other_index, other in enumerate(rows)
        )
        if not dominated:
            selected.append(index)
    return selected


def _plot(report: Mapping[str, Any], output_dir: Path) -> tuple[str, list[str]]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped_missing_matplotlib", []
    rows = [
        row for row in report["performance_rows"]
        if row["row_kind"] == "r5_cell"
        and isinstance(row["Avg-mAP"], float)
        and isinstance(row["total_latency_ms_p50"], float)
        and isinstance(row["selected_count_p50"], float)
    ]
    _require(rows, "no rows have both official mAP and raw paired cost")
    colors = {"uniform": "#0072B2", "learned": "#D55E00"}
    files: list[str] = []
    for filename, x_key, xlabel in (
        ("duca_r5_map_vs_latency", "total_latency_ms_p50", "Total latency (ms, p50)"),
        ("duca_r5_map_vs_selected_count", "selected_count_p50", "Selected frames (p50)"),
    ):
        figure, axis = plt.subplots(figsize=(4.8, 3.5))
        for arm in ("uniform", "learned"):
            subset = [row for row in rows if row["arm"] == arm]
            if subset:
                axis.scatter([row[x_key] for row in subset], [row["Avg-mAP"] for row in subset], label=arm.capitalize(), color=colors[arm], s=42)
        if x_key == "total_latency_ms_p50":
            frontier = [rows[index] for index in _pareto_indices(rows)]
            frontier.sort(key=lambda row: float(row[x_key]))
            axis.plot([row[x_key] for row in frontier], [row["Avg-mAP"] for row in frontier], color="#000000", linewidth=1.0, linestyle="--", label="Pareto")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Avg. mAP")
        axis.legend(frameon=False)
        figure.tight_layout()
        for suffix, format_name in (("png", "png"), ("pdf", "pdf")):
            path = output_dir / f"{filename}.{suffix}"
            figure.savefig(path, dpi=300 if suffix == "png" else None, format=format_name)
            files.append(str(path))
        plt.close(figure)
    return "generated", files


def export_performance_cost(
    *, aggregate_json: str | Path, raw_cost_summaries: Sequence[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    _require(not destination.exists(), f"refusing to overwrite output directory {destination}")
    report = build_performance_cost_report(
        aggregate_json=aggregate_json, raw_cost_summaries=raw_cost_summaries
    )
    destination.mkdir(parents=True)
    try:
        _write_table(destination / "duca_r5_performance_cost_raw.csv", report["performance_rows"], ",")
        _write_table(destination / "duca_r5_performance_cost_raw.tsv", report["performance_rows"], "\t")
        status, figures = _plot(report, destination)
        report["plot_status"] = status
        report["figure_files"] = figures
        report["report_sha256"] = _canonical_sha256({key: value for key, value in report.items() if key != "report_sha256"})
        _atomic_text(destination / "duca_r5_performance_cost.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
        snippet = "\\begin{figure}[t]\n  \\centering\n  \\includegraphics[width=0.49\\linewidth]{duca_r5_map_vs_latency.pdf}\n  \\includegraphics[width=0.49\\linewidth]{duca_r5_map_vs_selected_count.pdf}\n  \\caption{DUCA R5 official performance against measured paired runtime cost.}\n  \\label{fig:duca-r5-performance-cost}\n\\end{figure}\n"
        _atomic_text(destination / "duca_r5_performance_cost_figures.tex", snippet)
    except Exception:
        # The directory is intentionally left as evidence of a failed export; partial artifacts are never reused.
        raise
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DUCA R5 performance/cost evidence and paper figures")
    parser.add_argument("--aggregate-json", required=True, help="final aggregate_duca_r5_paper_matrix JSON")
    parser.add_argument("--cost-summary", action="append", required=True, help="raw paired cost summary JSON; pass all eight profiles")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_performance_cost(
        aggregate_json=args.aggregate_json,
        raw_cost_summaries=args.cost_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps({"plot_status": result["plot_status"], "rows": len(result["performance_rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
