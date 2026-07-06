from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import convert_lowres_probe_samples_to_value_transport_ledger as base_convert
from tools.bata import detector_aware_acquisition_policy as detector_policy
from tools.bata import paction_budget_contract
from tools.bata import paction_source_samples


OUTPUT_SCHEMA_VERSION = base_convert.OUTPUT_SCHEMA_VERSION
SUMMARY_SCHEMA_VERSION = "c3_detector_aware_value_transport_ledger_summary_v1"
READY = "C3_DETECTOR_AWARE_LEDGER_READY"
DEPLOY_CHECKPOINT_POLICY_SOURCES = {detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE}
FORBIDDEN_TRUE_FLAGS = base_convert.FORBIDDEN_TRUE_FLAGS


_read_jsonl = base_convert._read_jsonl
_write_jsonl = base_convert._write_jsonl
_is_true = base_convert._is_true
_optional_int = base_convert._optional_int
_int_positions = base_convert._int_positions
_validate_sample_id = base_convert._validate_sample_id
write_json = base_convert.write_json
validate_value_transport_selection_row = base_convert.validate_value_transport_selection_row


def _selected_positions_from_sample(row: Mapping[str, Any], *, line_no: int, strategy: str) -> list[int]:
    strategy_rows = row.get("strategy_selected_positions")
    if isinstance(strategy_rows, Mapping) and strategy in strategy_rows:
        return _int_positions(strategy_rows[strategy], name=f"line {line_no}: strategy_selected_positions.{strategy}")
    raise ValueError(f"line {line_no}: strategy '{strategy}' is missing from strategy_selected_positions")


def sample_row_to_value_transport_row(
    row: Mapping[str, Any],
    *,
    line_no: int,
    strategy: str,
    target_len: int,
    require_selected_count: int | None = None,
    allow_short_valid_ratio_count: bool = False,
    deploy_selection_ledger: bool = False,
    route_variant: str = "c3_detector_aware_stage2_offline_selector",
) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"line {line_no}: sample_id must be a non-empty string")
    _validate_sample_id(sample_id, line_no=line_no, require_window_sample_id=True)
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"line {line_no}: forbidden source flag {key}=true")
    if deploy_selection_ledger:
        for key in ("teacher_utility", "teacher_utility_provenance", "frame_utility"):
            if key in row:
                raise ValueError(f"line {line_no}: deploy detector-aware sample must not include {key}")
    dense_len = _optional_int(row, "dense_len")
    valid_len = _optional_int(row, "valid_len")
    if valid_len is None:
        valid_len = dense_len
    if valid_len is None or valid_len <= 0:
        raise ValueError(f"line {line_no}: valid_len or dense_len must be positive")
    if dense_len is not None and valid_len > dense_len:
        raise ValueError(f"line {line_no}: valid_len cannot exceed dense_len")
    selected = _selected_positions_from_sample(row, line_no=line_no, strategy=strategy)
    if any(position >= valid_len for position in selected):
        raise ValueError(f"line {line_no}: selected_positions exceed valid_len={valid_len}")
    expected_required_count = paction_budget_contract.expected_selected_count(
        require_selected_count,
        valid_len=int(valid_len),
        dense_len=int(dense_len or 0),
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
    )
    if expected_required_count is not None and len(selected) != int(expected_required_count):
        raise ValueError(
            f"line {line_no}: selected_count={len(selected)} does not match required count {int(expected_required_count)}"
        )
    policy_metadata = row.get("detector_aware_policy")
    if deploy_selection_ledger and not isinstance(policy_metadata, Mapping):
        raise ValueError(f"line {line_no}: detector_aware_policy metadata is required for deploy ledger")
    diagnostics = {
        "source_strategy": str(strategy),
        "source_selected_count": len(selected),
        "uniform_visible_fill_count": 0,
        "required_selected_count": expected_required_count,
        "allow_short_valid_ratio_count": bool(allow_short_valid_ratio_count),
    }
    if isinstance(policy_metadata, Mapping):
        diagnostics.update(
            {
                "policy_family": policy_metadata.get("policy_family"),
                "policy_source": policy_metadata.get("source"),
                "policy_checkpoint_path": policy_metadata.get("checkpoint_path"),
                "policy_checkpoint_sha256": policy_metadata.get("checkpoint_sha256") or policy_metadata.get("policy_checkpoint_sha256"),
                "policy_fixed_budget": policy_metadata.get("fixed_budgets"),
                "policy_dynamic_budget": policy_metadata.get("dynamic_budget"),
                "policy_uses_uniform_scaffold": policy_metadata.get("uses_uniform_scaffold"),
                "policy_uses_uniform_fill": policy_metadata.get("uses_uniform_fill"),
                "p_action_provenance": policy_metadata.get("p_action_provenance"),
                "stage_label": policy_metadata.get("stage_label"),
                "end_to_end": policy_metadata.get("end_to_end"),
            }
        )
    if deploy_selection_ledger and isinstance(policy_metadata, Mapping):
        if policy_metadata.get("source") not in DEPLOY_CHECKPOINT_POLICY_SOURCES:
            raise ValueError(
                f"line {line_no}: deploy ledger requires checkpoint policy source "
                f"{sorted(DEPLOY_CHECKPOINT_POLICY_SOURCES)}, got {policy_metadata.get('source')}"
            )
        if not policy_metadata.get("checkpoint_path"):
            raise ValueError(f"line {line_no}: deploy ledger requires policy checkpoint_path")
        if not (policy_metadata.get("checkpoint_sha256") or policy_metadata.get("policy_checkpoint_sha256")):
            raise ValueError(f"line {line_no}: deploy ledger requires policy checkpoint_sha256")
        paction_source_samples.validate_paction_positive_provenance(
            policy_metadata.get("p_action_provenance"),
            source_name=f"line {line_no}: detector_aware_policy",
        )
    ledger_row = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "selected_positions_unit": "local_dense_index",
        "selected_positions": selected,
        "target_len": int(target_len),
        "selected_count": len(selected),
        "valid_len": int(valid_len),
        "dense_len": dense_len,
        "route": "C3_DETECTOR_AWARE_STAGE2",
        "route_variant": str(route_variant),
        "policy": f"c3_detector_aware_{strategy}",
        "policy_source": diagnostics.get("policy_source"),
        "policy_checkpoint_path": diagnostics.get("policy_checkpoint_path"),
        "policy_checkpoint_sha256": diagnostics.get("policy_checkpoint_sha256"),
        "source_schema_version": "c3_lowres_probe_samples_jsonl",
        "diagnostics": diagnostics,
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "diagnostic_only": not bool(deploy_selection_ledger),
        "training_only": False,
        "diagnostic_uses_train_utility_for_audit": False,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "prediction_uses_gt": False,
    }
    validate_value_transport_selection_row(
        ledger_row,
        line_no=line_no,
        require_deployable=bool(deploy_selection_ledger),
    )
    return ledger_row


def run_conversion(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    strategy: str,
    target_len: int,
    summary_json: str | Path | None = None,
    require_selected_count: int | None = None,
    allow_short_valid_ratio_count: bool = False,
    deploy_selection_ledger: bool = False,
    deduplicate_sample_id: bool = False,
    route_variant: str = "c3_detector_aware_stage2_offline_selector",
) -> dict[str, Any]:
    source_rows = _read_jsonl(input_jsonl)
    out_rows = [
        sample_row_to_value_transport_row(
            row,
            line_no=line_no,
            strategy=strategy,
            target_len=target_len,
            require_selected_count=require_selected_count,
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
            deploy_selection_ledger=bool(deploy_selection_ledger),
            route_variant=route_variant,
        )
        for line_no, row in enumerate(source_rows, start=1)
    ]
    duplicate_sample_id_count = 0
    if deduplicate_sample_id:
        deduped_rows: list[dict[str, Any]] = []
        by_sample_id: dict[str, dict[str, Any]] = {}
        for row in out_rows:
            sample_id = str(row["sample_id"])
            previous = by_sample_id.get(sample_id)
            if previous is None:
                by_sample_id[sample_id] = row
                deduped_rows.append(row)
                continue
            duplicate_sample_id_count += 1
            for key in ("selected_positions", "selected_count", "target_len", "valid_len", "dense_len", "policy_source"):
                if previous.get(key) != row.get(key):
                    raise ValueError(f"duplicate sample_id {sample_id} has conflicting {key}")
        out_rows = deduped_rows
    sample_ids = [str(row["sample_id"]) for row in out_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("converted detector-aware ledger has duplicate sample_id")
    _write_jsonl(output_jsonl, out_rows)
    counts = [int(row["selected_count"]) for row in out_rows]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "row_count": len(out_rows),
        "strategy": str(strategy),
        "target_len": int(target_len),
        "require_selected_count": require_selected_count,
        "allow_short_valid_ratio_count": bool(allow_short_valid_ratio_count),
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "deduplicate_sample_id": bool(deduplicate_sample_id),
        "duplicate_sample_id_count": int(duplicate_sample_id_count),
        "route_variant": str(route_variant),
        "stage_label": detector_policy.STAGE_LABEL,
        "gap_control": "detector_aware_source_strategy_only_no_uniform_fill_for_deploy",
        "min_selected_count": min(counts),
        "max_selected_count": max(counts),
        "total_uniform_visible_fill_count": 0,
    }
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert detector-aware selector samples to strict value-transport ledgers.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--target-len", type=int, required=True)
    parser.add_argument("--require-selected-count", type=int)
    parser.add_argument("--allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--deploy-selection-ledger", action="store_true")
    parser.add_argument("--deduplicate-sample-id", action="store_true")
    parser.add_argument("--route-variant", default="c3_detector_aware_stage2_offline_selector")
    args = parser.parse_args(argv)
    summary = run_conversion(
        args.input_jsonl,
        args.output_jsonl,
        strategy=args.strategy,
        target_len=int(args.target_len),
        summary_json=args.summary_json,
        require_selected_count=args.require_selected_count,
        allow_short_valid_ratio_count=bool(args.allow_short_valid_ratio_count),
        deploy_selection_ledger=bool(args.deploy_selection_ledger),
        deduplicate_sample_id=bool(args.deduplicate_sample_id),
        route_variant=args.route_variant,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
