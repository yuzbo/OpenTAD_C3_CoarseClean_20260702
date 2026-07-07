from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA_VERSION = "c3_sparse_selector_evidence_summary_v1"
READY = "C3_SPARSE_SELECTOR_EVIDENCE_SUMMARY_READY"


_EPOCH_RE = re.compile(r"\b(?:epoch|Epoch)\s*(?:\[|=|:|\s)\s*(\d+)", re.IGNORECASE)
_AVG_MAP_RE = re.compile(r"\b(?:Average[-_\s]?mAP|Avg[-_\s]?mAP|average_mAP)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_TIOU_MAP_RE = re.compile(
    r"(?:tIoU|tiou|mAP@)\s*(?:=|:|@)?\s*(0\.[0-9]+)\D{0,40}?(?:mAP)?\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
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


def build_evidence_summary(
    *,
    map_logs: Sequence[tuple[str, str | Path]] = (),
    ledger_summaries: Sequence[tuple[str, str | Path]] = (),
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
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "map_curves": curves,
        "ledger_summaries": ledgers,
        "claim_status": "diagnostic_only_no_causal_claim",
        "required_next_step": "matched_same_commit_ablation_matrix",
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
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    summary = build_evidence_summary(
        map_logs=[_parse_named_path(item) for item in args.map_log],
        ledger_summaries=[_parse_named_path(item) for item in args.ledger_summary],
    )
    _write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
