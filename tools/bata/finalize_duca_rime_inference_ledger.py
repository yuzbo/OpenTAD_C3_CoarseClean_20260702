from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import math
from typing import Any, Mapping, Sequence


SCHEMA = "duca_rime_inference_ledger_v1"
SUMMARY_SCHEMA = "duca_rime_inference_ledger_summary_v1"


def exact_uniform_positions(temporal_len: int, k: int) -> list[int]:
    """Mirror the runtime round-half-to-even exact-uniform anchors without torch."""

    temporal_len = int(temporal_len)
    k = int(k)
    if temporal_len < 0 or k < 0 or k > temporal_len:
        raise ValueError("exact-uniform requires 0 <= k <= temporal_len")
    if k == 0:
        return []
    if k == 1:
        return [0]
    denominator = k - 1
    anchors = []
    for index in range(k):
        quotient, remainder = divmod(index * (temporal_len - 1), denominator)
        if 2 * remainder > denominator or (
            2 * remainder == denominator and quotient % 2 == 1
        ):
            quotient += 1
        anchors.append(quotient)
    return anchors


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def finalize_ledger(
    *,
    shards: Sequence[str | Path],
    output_jsonl: str | Path,
    expected_arm: str,
    expected_protocol_sha256: str | None = None,
    require_explicit_budget_truth: bool = False,
) -> dict[str, Any]:
    if not shards:
        raise ValueError("at least one RIME inference-ledger shard is required")
    if require_explicit_budget_truth and (
        expected_protocol_sha256 is None
        or len(str(expected_protocol_sha256)) != 64
        or any(
            value not in "0123456789abcdef"
            for value in str(expected_protocol_sha256).lower()
        )
    ):
        raise ValueError(
            "explicit budget truth requires an exact expected protocol SHA-256"
        )
    source_artifacts = []
    rows = {}
    for shard in shards:
        path = Path(shard).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source_artifacts.append({"path": str(path), "sha256": _sha256_file(path)})
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            prefix = f"{path}:{line_number}"
            provenance = row.get("provenance")
            if (
                row.get("schema_version") != SCHEMA
                or row.get("arm") != str(expected_arm)
                or not isinstance(provenance, Mapping)
                or any(
                    bool(provenance.get(key, False))
                    for key in (
                        "uses_gt",
                        "uses_teacher",
                        "uses_prediction_cache",
                        "uses_test_batch_composition",
                        "raw_predictions_stored",
                    )
                )
            ):
                raise ValueError(f"{prefix}: contaminated or mismatched inference ledger")
            protocol_hash = row.get("budget_protocol_sha256")
            if expected_protocol_sha256 is not None and protocol_hash != str(
                expected_protocol_sha256
            ):
                raise ValueError(f"{prefix}: frozen budget protocol drift")
            requested = int(row["requested_k"])
            effective = int(row["effective_k"])
            unique = int(row["unique_k"])
            backbone = int(row["backbone_input_k"])
            padded = int(row["padded_k"])
            positions = [int(value) for value in row["selected_dense_indices"]]
            dense_valid_len = int(row.get("dense_valid_len", -1))
            gap_cap = float(row["max_gap_seconds_cap"])
            observed_gap = float(row["observed_max_gap_seconds"])
            budget_truth_fields = (
                "raw_budget",
                "reachable_budget",
                "realized_budget",
                "projection_unused_budget",
                "solver_unused_budget",
                "budget_scope",
                "claim_scope",
            )
            has_budget_truth = any(name in row for name in budget_truth_fields)
            if require_explicit_budget_truth and not has_budget_truth:
                raise ValueError(f"{prefix}: explicit budget truth is required")
            if has_budget_truth:
                if not all(name in row for name in budget_truth_fields):
                    raise ValueError(f"{prefix}: incomplete explicit budget truth")
                raw_budget = int(row["raw_budget"])
                reachable_budget = int(row["reachable_budget"])
                realized_budget = int(row["realized_budget"])
                projection_unused = int(row["projection_unused_budget"])
                solver_unused = int(row["solver_unused_budget"])
                if (
                    raw_budget != requested
                    or reachable_budget != effective
                    or realized_budget != effective
                    or projection_unused != raw_budget - reachable_budget
                    or solver_unused != reachable_budget - realized_budget
                    or solver_unused != 0
                    or row["budget_scope"] != "window_fixed_request"
                    or row["claim_scope"]
                    != "stage0_engineering_window_execution"
                ):
                    raise ValueError(f"{prefix}: explicit budget truth is inconsistent")
            if (
                requested < effective
                or not effective == unique == backbone == padded > 0
                or dense_valid_len < effective
                or positions != sorted(set(positions))
                or len(positions) != effective
                or any(
                    position < 0 or position >= dense_valid_len
                    for position in positions
                )
                or not math.isfinite(gap_cap)
                or not math.isfinite(observed_gap)
                or gap_cap < 0.0
                or observed_gap < 0.0
                or observed_gap > gap_cap + 1.0e-8
            ):
                raise ValueError(f"{prefix}: exact-K/no-padding cost ledger violation")
            video = str(row.get("video_id") or "")
            start = int(row.get("window_start_frame", -1))
            if not video or start < 0:
                raise ValueError(f"{prefix}: invalid window identity")
            key = (video, start)
            if key in rows:
                raise ValueError(f"{prefix}: duplicate inference window {key}")
            rows[key] = row
    if not rows:
        raise ValueError("RIME inference ledger contains no records")
    target = Path(output_jsonl).expanduser().resolve()
    text = "".join(
        json.dumps(rows[key], sort_keys=True, separators=(",", ":")) + "\n"
        for key in sorted(rows)
    )
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different RIME ledger: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    requested_values = [int(row["requested_k"]) for row in rows.values()]
    effective_values = [int(row["effective_k"]) for row in rows.values()]
    observed_gaps = [
        float(row["observed_max_gap_seconds"]) for row in rows.values()
    ]
    gap_caps = [float(row["max_gap_seconds_cap"]) for row in rows.values()]
    has_explicit_budget_truth = all("raw_budget" in row for row in rows.values())
    if any("raw_budget" in row for row in rows.values()) and not has_explicit_budget_truth:
        raise ValueError("inference ledger mixes explicit and legacy budget truth")
    if require_explicit_budget_truth and not has_explicit_budget_truth:
        raise ValueError("sealed inference ledger requires explicit budget truth")
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "sealed",
        "arm": str(expected_arm),
        "path": str(target),
        "sha256": _sha256_file(target),
        "record_count": len(rows),
        "video_count": len({key[0] for key in rows}),
        "requested_mean_k": mean(requested_values),
        "effective_mean_k": mean(effective_values),
        "requested_k_histogram": {
            str(value): requested_values.count(value)
            for value in sorted(set(requested_values))
        },
        "max_observed_gap_seconds": max(observed_gaps),
        "max_gap_seconds_cap": max(gap_caps),
        "all_observed_gaps_within_cap": True,
        "no_padding_ledger": True,
        "source_shards": source_artifacts,
        "official_final_labels_used_for_decision": False,
        "claim_scope": "inference_allocation_and_cost_ledger_only",
    }
    if has_explicit_budget_truth:
        raw_total = sum(int(row["raw_budget"]) for row in rows.values())
        reachable_total = sum(
            int(row["reachable_budget"]) for row in rows.values()
        )
        realized_total = sum(int(row["realized_budget"]) for row in rows.values())
        projection_unused_total = sum(
            int(row["projection_unused_budget"]) for row in rows.values()
        )
        solver_unused_total = sum(
            int(row["solver_unused_budget"]) for row in rows.values()
        )
        if (
            raw_total - reachable_total != projection_unused_total
            or reachable_total - realized_total != solver_unused_total
            or solver_unused_total != 0
        ):
            raise ValueError("sealed explicit budget totals are inconsistent")
        summary.update(
            {
                "explicit_budget_truth": True,
                "budget_scope": "window_fixed_request",
                "raw_budget_total": raw_total,
                "reachable_budget_total": reachable_total,
                "realized_budget_total": realized_total,
                "projection_unused_budget_total": projection_unused_total,
                "solver_unused_budget_total": solver_unused_total,
                "budget_protocol_sha256": str(expected_protocol_sha256),
            }
        )
    else:
        summary["explicit_budget_truth"] = False
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal RIME inference-ledger shards")
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--expected-arm", required=True)
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--require-explicit-budget-truth", action="store_true")
    parser.add_argument("--summary-json")
    args = parser.parse_args(argv)
    result = finalize_ledger(
        shards=args.shard,
        output_jsonl=args.output_jsonl,
        expected_arm=args.expected_arm,
        expected_protocol_sha256=args.expected_protocol_sha256,
        require_explicit_budget_truth=args.require_explicit_budget_truth,
    )
    if args.summary_json:
        path = Path(args.summary_json).expanduser().resolve()
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different summary: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
