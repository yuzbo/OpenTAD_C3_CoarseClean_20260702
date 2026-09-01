from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = "c3_sparse_selector_evidence_summary_v1"
READY = "C3_SPARSE_SELECTOR_EVIDENCE_SUMMARY_READY"


_EPOCH_RE = re.compile(r"\b(?:epoch|Epoch)\s*(?:\[|=|:|\s)\s*(\d+)", re.IGNORECASE)
_AVG_MAP_RE = re.compile(r"\b(?:Average[-_\s]?mAP|Avg[-_\s]?mAP|average_mAP)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_TIOU_MAP_RE = re.compile(
    r"(?:tIoU|tiou|mAP@)\s*(?:=|:|@)?\s*(0\.[0-9]+)\D{0,40}?(?:mAP)?\s*(?:=|:|is)?\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_percent(value: str) -> float:
    return float(value)


def parse_adatad_map_curve(log_path: str | Path) -> list[dict[str, Any]]:
    """Extract coarse AdaTAD mAP eval records from a train log.

    The parser is intentionally format-tolerant because OpenTAD/AdaTAD logs can
    be emitted through different logger wrappers. It groups tIoU rows and the
    following average mAP under the most recently observed epoch.
    """

    path = Path(log_path).expanduser()
    if not path.is_file():
        raise ValueError(f"log file does not exist: {path}")
    current_epoch: int | None = None
    current_tiou: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = _EPOCH_RE.search(line)
        if epoch_match is not None:
            current_epoch = int(epoch_match.group(1))
        for tiou, value in _TIOU_MAP_RE.findall(line):
            # Avoid treating the average mAP number as a tIoU pair on loose log lines.
            tiou_key = f"{float(tiou):.2f}"
            current_tiou[tiou_key] = _coerce_percent(value)
        avg_match = _AVG_MAP_RE.search(line)
        if avg_match is not None:
            record = {
                "epoch": current_epoch,
                "average_mAP": _coerce_percent(avg_match.group(1)),
                "tIoU_mAP": dict(sorted(current_tiou.items())),
            }
            records.append(record)
            current_tiou = {}
    return records


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _slope(records: Sequence[Mapping[str, Any]], *, key: str = "average_mAP") -> float | None:
    points: list[tuple[float, float]] = []
    for idx, record in enumerate(records):
        epoch = record.get("epoch")
        x_value = float(idx if epoch is None else epoch)
        y_value = _finite_float(record.get(key))
        if y_value is not None:
            points.append((x_value, y_value))
    if len(points) < 2:
        return None
    mean_x = sum(item[0] for item in points) / len(points)
    mean_y = sum(item[1] for item in points) / len(points)
    denom = sum((x_value - mean_x) ** 2 for x_value, _ in points)
    if denom <= 0.0:
        return None
    numer = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in points)
    return numer / denom


def _best_record(records: Sequence[Mapping[str, Any]], *, key: str) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    best_value: float | None = None
    for record in records:
        value = _finite_float(record.get(key))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best = record
            best_value = value
    return best


def curve_diagnostics(records: Sequence[Mapping[str, Any]], *, plateau_epsilon: float = 1.0) -> dict[str, Any]:
    sorted_records = sorted(records, key=lambda item: (10**9 if item.get("epoch") is None else int(item["epoch"])))
    if not sorted_records:
        return {
            "eval_count": 0,
            "status": "missing_map_curve",
        }
    first = sorted_records[0]
    last = sorted_records[-1]
    best = _best_record(sorted_records, key="average_mAP") or last
    first_avg = _finite_float(first.get("average_mAP"))
    last_avg = _finite_float(last.get("average_mAP"))
    best_avg = _finite_float(best.get("average_mAP"))
    late_records = sorted_records[1:] if len(sorted_records) > 1 else []
    high_iou_key = "0.70"
    first_high = _finite_float((first.get("tIoU_mAP") or {}).get(high_iou_key))
    last_high = _finite_float((last.get("tIoU_mAP") or {}).get(high_iou_key))
    best_high_record = _best_record(
        [
            {"epoch": record.get("epoch"), "average_mAP": (record.get("tIoU_mAP") or {}).get(high_iou_key)}
            for record in sorted_records
        ],
        key="average_mAP",
    )
    return {
        "eval_count": len(sorted_records),
        "first_eval_epoch": first.get("epoch"),
        "first_eval_average_mAP": _round(first_avg),
        "last_eval_epoch": last.get("epoch"),
        "last_eval_average_mAP": _round(last_avg),
        "best_eval_epoch": best.get("epoch"),
        "best_average_mAP": _round(best_avg),
        "first_to_last_average_mAP_delta": _round(None if first_avg is None or last_avg is None else last_avg - first_avg),
        "first_to_best_average_mAP_delta": _round(None if first_avg is None or best_avg is None else best_avg - first_avg),
        "best_to_last_average_mAP_drop": _round(None if best_avg is None or last_avg is None else best_avg - last_avg),
        "overall_slope_mAP_per_epoch": _round(_slope(sorted_records)),
        "late_slope_mAP_per_epoch_after_first_eval": _round(_slope(late_records)),
        "plateau_after_first_eval": (
            None
            if first_avg is None or last_avg is None or len(sorted_records) < 2
            else abs(last_avg - first_avg) <= float(plateau_epsilon)
        ),
        "high_iou_key": high_iou_key,
        "first_high_iou_mAP": _round(first_high),
        "last_high_iou_mAP": _round(last_high),
        "first_to_last_high_iou_delta": _round(None if first_high is None or last_high is None else last_high - first_high),
        "best_high_iou_epoch": None if best_high_record is None else best_high_record.get("epoch"),
        "best_high_iou_mAP": None if best_high_record is None else _round(_finite_float(best_high_record.get("average_mAP"))),
        "records_have_high_iou": all(high_iou_key in (record.get("tIoU_mAP") or {}) for record in sorted_records),
    }


def _records_by_epoch(records: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for record in records:
        epoch = record.get("epoch")
        if epoch is None:
            continue
        out[int(epoch)] = record
    return out


def compare_map_curves(curves: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(curves)
    comparisons: dict[str, Any] = {}
    for left_idx, left_name in enumerate(names):
        left_records = _records_by_epoch(curves[left_name].get("records", []))
        for right_name in names[left_idx + 1 :]:
            right_records = _records_by_epoch(curves[right_name].get("records", []))
            matched_epochs = sorted(set(left_records) & set(right_records))
            delta_by_epoch: dict[str, Any] = {}
            for epoch in matched_epochs:
                left = left_records[epoch]
                right = right_records[epoch]
                left_avg = _finite_float(left.get("average_mAP"))
                right_avg = _finite_float(right.get("average_mAP"))
                tiou_deltas = {}
                for tiou in sorted(set((left.get("tIoU_mAP") or {})) & set((right.get("tIoU_mAP") or {}))):
                    left_tiou = _finite_float((left.get("tIoU_mAP") or {}).get(tiou))
                    right_tiou = _finite_float((right.get("tIoU_mAP") or {}).get(tiou))
                    tiou_deltas[tiou] = _round(None if left_tiou is None or right_tiou is None else left_tiou - right_tiou)
                delta_by_epoch[str(epoch)] = {
                    f"{left_name}_minus_{right_name}_average_mAP": _round(
                        None if left_avg is None or right_avg is None else left_avg - right_avg
                    ),
                    "tIoU_mAP_delta": tiou_deltas,
                }
            last_common = None if not matched_epochs else str(matched_epochs[-1])
            comparisons[f"{left_name}__minus__{right_name}"] = {
                "matched_epochs": matched_epochs,
                "last_common_epoch": None if last_common is None else int(last_common),
                "last_common_delta": None if last_common is None else delta_by_epoch[last_common],
                "delta_by_epoch": delta_by_epoch,
            }
    return comparisons


def _metric_subset(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "strategy",
        "row_count",
        "min_selected_count",
        "max_selected_count",
        "mean_selected_count",
        "selected_count_histogram",
        "max_gap",
        "p95_gap",
        "max_unselected_hole",
        "p95_unselected_hole",
        "mean_uniform_similarity",
        "max_uniform_similarity",
        "boundary_support_r1",
        "boundary_support@r1",
        "boundary_bracket_support_r1",
        "boundary_bracket_support@r1",
        "action_positive_coverage",
        "action_interior_bin_coverage",
        "p_action_rank_spearman",
        "p_action_topk_jaccard",
        "p_action_topk_overlap_ratio",
        "dynamic_budget_entropy",
        "dynamic_budget_iqr",
        "uses_uniform_fill",
        "uses_uniform_scaffold",
    ]
    return {key: payload.get(key) for key in keys if key in payload}


def compare_ledger_metrics(ledgers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(ledgers)
    comparisons: dict[str, Any] = {}
    for left_idx, left_name in enumerate(names):
        left = ledgers[left_name].get("metrics", {})
        for right_name in names[left_idx + 1 :]:
            right = ledgers[right_name].get("metrics", {})
            shared = sorted(set(left) & set(right))
            numeric_delta = {}
            for key in shared:
                left_value = _finite_float(left.get(key))
                right_value = _finite_float(right.get(key))
                if left_value is not None and right_value is not None:
                    numeric_delta[key] = _round(left_value - right_value)
            comparisons[f"{left_name}__minus__{right_name}"] = {
                "shared_keys": shared,
                "numeric_delta": numeric_delta,
            }
    return comparisons


def evidence_gaps(curves: Mapping[str, Mapping[str, Any]], ledgers: Mapping[str, Mapping[str, Any]]) -> list[str]:
    gaps: list[str] = []
    if len(curves) < 2:
        gaps.append("need_at_least_two_map_curves_for_matched_comparison")
    if not ledgers:
        gaps.append("need_ledger_validation_summaries_for_selector_distribution_claims")
    for name, curve in curves.items():
        records = curve.get("records", [])
        if not records:
            gaps.append(f"{name}:missing_map_records")
            continue
        if not all("0.70" in (record.get("tIoU_mAP") or {}) for record in records):
            gaps.append(f"{name}:missing_high_iou_0.70_breakdown")
    for name, ledger in ledgers.items():
        metrics = ledger.get("metrics", {})
        for key in ("max_unselected_hole", "p95_unselected_hole", "boundary_support_r1", "action_positive_coverage"):
            if key not in metrics and not (key == "boundary_support_r1" and "boundary_support@r1" in metrics):
                gaps.append(f"{name}:missing_{key}")
        if "p_action_topk_jaccard" not in metrics and "p_action_topk_overlap_ratio" not in metrics:
            gaps.append(f"{name}:missing_p_action_topk_overlap_metrics")
    comparisons = compare_map_curves(curves)
    if curves and not any(item.get("matched_epochs") for item in comparisons.values()):
        gaps.append("map_curves_have_no_common_eval_epoch")
    return gaps


def build_evidence_summary(
    *,
    map_logs: Sequence[tuple[str, str | Path]] = (),
    ledger_summaries: Sequence[tuple[str, str | Path]] = (),
    plateau_epsilon: float = 1.0,
) -> dict[str, Any]:
    curves = {
        str(name): {
            "log_path": str(path),
            "records": parse_adatad_map_curve(path),
        }
        for name, path in map_logs
    }
    ledgers = {
        str(name): {
            "summary_path": str(path),
            "metrics": _metric_subset(_read_json(path)),
        }
        for name, path in ledger_summaries
    }
    for curve in curves.values():
        curve["diagnostics"] = curve_diagnostics(curve["records"], plateau_epsilon=plateau_epsilon)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "map_curves": curves,
        "ledger_summaries": ledgers,
        "map_curve_comparisons": compare_map_curves(curves),
        "ledger_metric_comparisons": compare_ledger_metrics(ledgers),
        "evidence_gaps": evidence_gaps(curves, ledgers),
        "claim_status": "diagnostic_only_no_causal_claim",
        "required_next_step": (
            "run matched same-commit ablations: GAS-VT current fixed_384, "
            "PAction fixed_384 without hard repair, PAction fixed_384 with max-hole repair, "
            "and raw p_action top-k fixed_384 under identical AdaTAD settings"
        ),
    }


def _parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"expected non-empty NAME=PATH, got: {value}")
    return name.strip(), path.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize sparse selector mAP curves and ledger validation evidence.")
    parser.add_argument("--map-log", action="append", default=[], help="Named train log as NAME=PATH")
    parser.add_argument("--ledger-summary", action="append", default=[], help="Named validator summary JSON as NAME=PATH")
    parser.add_argument("--plateau-epsilon", type=float, default=1.0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    summary = build_evidence_summary(
        map_logs=[_parse_named_path(item) for item in args.map_log],
        ledger_summaries=[_parse_named_path(item) for item in args.ledger_summary],
        plateau_epsilon=args.plateau_epsilon,
    )
    _write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
