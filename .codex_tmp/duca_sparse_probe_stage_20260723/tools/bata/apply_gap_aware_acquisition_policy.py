from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import apply_paction_acquisition_policy as base_apply
from tools.bata import gas_vt_paction_policy as gas_vt
from tools.bata import paction_acquisition_policy as base_policy
from tools.bata import paction_source_samples


SUMMARY_SCHEMA_VERSION = "c3_gas_vt_policy_application_v1"
READY = "C3_GAS_VT_POLICY_APPLICATION_READY"
BOOTSTRAP_POLICY_SOURCE = gas_vt.GAS_VT_BOOTSTRAP_POLICY_SOURCE
CHECKPOINT_POLICY_SOURCE = gas_vt.GAS_VT_CHECKPOINT_POLICY_SOURCE


_read_jsonl = base_apply._read_jsonl
_write_jsonl = base_apply._write_jsonl
_write_json = base_apply._write_json
_sha256_file = base_apply._sha256_file
_strip_deploy_invisible_payload = base_apply._strip_deploy_invisible_payload
_extract_paction = base_apply._extract_paction
_reject_forbidden_source_flags = base_apply._reject_forbidden_source_flags
_paction_positive_provenance = base_apply._paction_positive_provenance


def bootstrap_policy_scores(
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    dynamic_budget_buckets: Sequence[int] = gas_vt.DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS,
) -> tuple[list[float], list[float]]:
    features = gas_vt.build_gap_aware_feature_matrix(
        p_action,
        valid=valid,
        target_budget=max(int(item) for item in dynamic_budget_buckets) if dynamic_budget_buckets else len(p_action),
    )
    p_idx = gas_vt.feature_index("p_action")
    density_idx = gas_vt.feature_index("local_density")
    change_idx = gas_vt.feature_index("local_change")
    urgency_idx = gas_vt.feature_index("gap_urgency")
    valid_idx = gas_vt.feature_index("valid")
    frame_values = [
        (
            0.75 * float(row[p_idx])
            + 0.45 * float(row[density_idx])
            + 0.30 * float(row[change_idx])
            + 0.20 * float(row[urgency_idx])
        )
        * float(row[valid_idx])
        for row in features
    ]
    valid_values = [value for value, row in zip(frame_values, features) if float(row[valid_idx]) > 0.0]
    mean_value = sum(valid_values) / float(len(valid_values)) if valid_values else 0.0
    target_idx = int(round(max(0.0, min(1.0, mean_value)) * float(max(0, len(dynamic_budget_buckets) - 1))))
    budget_scores = [-abs(idx - target_idx) for idx in range(len(dynamic_budget_buckets))]
    return [float(item) for item in frame_values], [float(item) for item in budget_scores]


def load_policy_checkpoint(checkpoint_path: str | Path, *, device: str = "cuda") -> tuple[gas_vt.GapAwareSequentialAcquisitionPolicy, dict[str, Any]]:
    import torch

    from tools.bata import train_gap_aware_acquisition_policy as train_policy

    payload = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("GAS-VT policy checkpoint must be a mapping")
    if payload.get("schema_version") != train_policy.CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("GAS-VT policy checkpoint schema_version mismatch")
    if payload.get("decision") != train_policy.READY:
        raise ValueError("GAS-VT policy checkpoint decision is not ready")
    state_dict = payload.get("policy_state_dict")
    if state_dict is None:
        raise ValueError("GAS-VT policy checkpoint is missing policy_state_dict")
    model_kwargs = dict(payload.get("model_kwargs") or {})
    if not model_kwargs:
        model_kwargs = {
            "input_dim": len(gas_vt.GAS_VT_FEATURE_NAMES),
            "budget_buckets": payload.get("dynamic_budget_buckets", gas_vt.DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS),
        }
    model = gas_vt.GapAwareSequentialAcquisitionPolicy(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, dict(payload)


def checkpoint_policy_scores(
    model: gas_vt.GapAwareSequentialAcquisitionPolicy,
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    target_budget: int | None = None,
    device: str = "cuda",
) -> tuple[list[float], list[float]]:
    import torch

    features = gas_vt.build_gap_aware_feature_matrix(p_action, valid=valid, target_budget=target_budget)
    with torch.no_grad():
        feature_tensor = torch.tensor([features], dtype=torch.float32, device=device)
        valid_tensor = None if valid is None else torch.tensor([list(valid)], dtype=torch.bool, device=device)
        budget_tensor = None if target_budget is None else torch.tensor([int(target_budget)], dtype=torch.long, device=device)
        outputs = model(feature_tensor, valid_tensor, target_budget=budget_tensor)
    frame_values = [float(item) for item in outputs["frame_value"][0].detach().cpu().tolist()]
    budget_scores = [float(item) for item in outputs["budget_logits"][0].detach().cpu().tolist()]
    return frame_values, budget_scores


def _effective_strategy_budget(row: Mapping[str, Any], requested_budget: int, *, frame_len: int) -> int:
    valid_len = int(row.get("valid_len") or row.get("dense_len") or frame_len)
    dense_len = int(row.get("dense_len") or frame_len)
    return base_policy.short_valid_ratio_budget(int(requested_budget), valid_len=valid_len, dense_len=dense_len)


def _checkpoint_budget_conditioned_scores(
    model: gas_vt.GapAwareSequentialAcquisitionPolicy,
    row: Mapping[str, Any],
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any],
    fixed_budgets: Sequence[int],
    dynamic_budget_buckets: Sequence[int],
    device: str,
) -> tuple[list[float], list[float], dict[str, list[float]], dict[str, int], int]:
    valid_len = int(row.get("valid_len") or row.get("dense_len") or len(p_action))
    initial_budget = max([int(item) for item in dynamic_budget_buckets] or [valid_len])
    _initial_frame_values, budget_scores = checkpoint_policy_scores(
        model,
        p_action,
        valid=valid,
        target_budget=initial_budget,
        device=device,
    )
    dynamic_budget = base_policy.decode_budget_from_scores(
        budget_scores,
        dynamic_budget_buckets,
        valid_len=valid_len,
    )
    frame_values_by_strategy: dict[str, list[float]] = {}
    target_budgets: dict[str, int] = {}
    fixed_strategy_names = (gas_vt.GAS_VT_FIXED_384_STRATEGY, gas_vt.GAS_VT_FIXED_768_STRATEGY)
    fixed_values = [int(item) for item in fixed_budgets]
    for strategy_idx, strategy_name in enumerate(fixed_strategy_names):
        requested = fixed_values[strategy_idx] if strategy_idx < len(fixed_values) else fixed_values[-1]
        strategy_budget = _effective_strategy_budget(row, requested, frame_len=len(p_action))
        frame_values, _unused_budget_scores = checkpoint_policy_scores(
            model,
            p_action,
            valid=valid,
            target_budget=strategy_budget,
            device=device,
        )
        frame_values_by_strategy[strategy_name] = frame_values
        target_budgets[strategy_name] = int(strategy_budget)
    dynamic_frame_values, _unused_budget_scores = checkpoint_policy_scores(
        model,
        p_action,
        valid=valid,
        target_budget=dynamic_budget,
        device=device,
    )
    frame_values_by_strategy[gas_vt.GAS_VT_DYNAMIC_STRATEGY] = dynamic_frame_values
    target_budgets[gas_vt.GAS_VT_DYNAMIC_STRATEGY] = int(dynamic_budget)
    return dynamic_frame_values, budget_scores, frame_values_by_strategy, target_budgets, int(initial_budget)


def run_policy_application(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    summary_json: str | Path | None = None,
    fixed_budgets: Sequence[int] = (384, 768),
    dynamic_budget_buckets: Sequence[int] = gas_vt.DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
    strip_deploy_invisible_payload: bool = False,
    strict_deploy_source: bool = False,
    allow_bootstrap_for_tests: bool = False,
    max_unselected_hole: int | None = None,
    source_jsonl_for_hash: str | Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    if checkpoint_path is None and not allow_bootstrap_for_tests:
        raise ValueError("checkpoint_path is required for GAS-VT policy application")
    if checkpoint_path is not None and allow_bootstrap_for_tests:
        raise ValueError("allow_bootstrap_for_tests cannot be combined with checkpoint_path")
    source_jsonl_sha256 = _sha256_file(source_jsonl_for_hash or input_jsonl)
    checkpoint_sha256: str | None = None
    checkpoint_model: gas_vt.GapAwareSequentialAcquisitionPolicy | None = None
    checkpoint_payload: dict[str, Any] | None = None
    source = BOOTSTRAP_POLICY_SOURCE
    if checkpoint_path is not None:
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        source = CHECKPOINT_POLICY_SOURCE
        if not allow_bootstrap_for_tests:
            checkpoint_model, checkpoint_payload = load_policy_checkpoint(checkpoint_path, device=device)
            dynamic_budget_buckets = checkpoint_payload.get("dynamic_budget_buckets", dynamic_budget_buckets)
    enriched_rows: list[dict[str, Any]] = []
    dynamic_budgets: list[int] = []
    for line_no, row in enumerate(rows, start=1):
        if strict_deploy_source:
            p_action_provenance = paction_source_samples.reject_strict_deploy_source_row(
                row,
                source_name=f"{input_jsonl}:{line_no}",
                reject_payload=True,
            )
        else:
            _reject_forbidden_source_flags(row, line_no=line_no)
            p_action_provenance = _paction_positive_provenance(row)
        p_action = _extract_paction(row, line_no=line_no)
        valid_len = int(row.get("valid_len") or row.get("dense_len") or len(p_action))
        valid = [idx < valid_len for idx in range(len(p_action))]
        if checkpoint_model is None:
            frame_values, budget_scores = bootstrap_policy_scores(
                p_action,
                valid=valid,
                dynamic_budget_buckets=dynamic_budget_buckets,
            )
            frame_values_by_strategy = None
            apply_time_target_budgets: dict[str, int] = {}
            budget_score_target_budget = None
        else:
            (
                frame_values,
                budget_scores,
                frame_values_by_strategy,
                apply_time_target_budgets,
                budget_score_target_budget,
            ) = _checkpoint_budget_conditioned_scores(
                checkpoint_model,
                row,
                p_action,
                valid=valid,
                fixed_budgets=fixed_budgets,
                dynamic_budget_buckets=dynamic_budget_buckets,
                device=device,
            )
        enriched = gas_vt.add_gas_vt_decision_to_sample_row(
            row,
            frame_values=frame_values,
            frame_values_by_strategy=frame_values_by_strategy,
            fixed_budgets=fixed_budgets,
            dynamic_budget_scores=budget_scores,
            dynamic_budget_buckets=dynamic_budget_buckets,
            max_unselected_hole=max_unselected_hole,
            source=source,
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            source_jsonl_sha256=source_jsonl_sha256,
        )
        enriched["gas_vt_policy"]["p_action_provenance"] = p_action_provenance
        if checkpoint_model is not None:
            enriched["gas_vt_policy"]["apply_time_target_budgets"] = apply_time_target_budgets
            enriched["gas_vt_policy"]["budget_score_target_budget"] = budget_score_target_budget
            enriched["gas_vt_policy"]["budget_conditioning_rule"] = "checkpoint_two_pass_strategy_specific_target_budget"
        enriched["gas_vt_policy"]["frame_value_summary"] = {
            "min": min(frame_values) if frame_values else None,
            "max": max(frame_values) if frame_values else None,
            "mean": None if not frame_values else sum(frame_values) / float(len(frame_values)),
        }
        dynamic_budgets.append(int(enriched["gas_vt_policy"]["dynamic_budget"]))
        if strip_deploy_invisible_payload:
            enriched = _strip_deploy_invisible_payload(enriched)
        enriched_rows.append(enriched)
    _write_jsonl(output_jsonl, enriched_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "policy_family": "GAS-VT",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "row_count": len(enriched_rows),
        "fixed_budgets": [int(item) for item in fixed_budgets],
        "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
        "min_dynamic_budget": min(dynamic_budgets) if dynamic_budgets else None,
        "max_dynamic_budget": max(dynamic_budgets) if dynamic_budgets else None,
        "mean_dynamic_budget": None if not dynamic_budgets else sum(dynamic_budgets) / float(len(dynamic_budgets)),
        "source": source,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "policy_checkpoint_sha256": checkpoint_sha256,
        "source_jsonl_sha256": source_jsonl_sha256,
        "strip_deploy_invisible_payload": bool(strip_deploy_invisible_payload),
        "strict_deploy_source": bool(strict_deploy_source),
        "allow_bootstrap_for_tests": bool(allow_bootstrap_for_tests),
        "decode_mode": "hard_gap_aware_topk",
        "uses_uniform_scaffold": False,
        "uses_uniform_fill": False,
        "budget_conditioning_rule": (
            "bootstrap_max_dynamic_budget"
            if checkpoint_path is None
            else "checkpoint_two_pass_strategy_specific_target_budget"
        ),
    }
    if summary_json is not None:
        _write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a GAS-VT gap-aware acquisition policy to p_action rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--fixed-budgets", type=int, nargs="+", default=[384, 768])
    parser.add_argument("--dynamic-budget-buckets", type=int, nargs="+", default=list(gas_vt.DEFAULT_GAS_VT_DYNAMIC_BUDGET_BUCKETS))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strip-deploy-invisible-payload", action="store_true")
    parser.add_argument("--strict-deploy-source", action="store_true")
    parser.add_argument("--max-unselected-hole", type=int)
    args = parser.parse_args(argv)
    summary = run_policy_application(
        args.input_jsonl,
        args.output_jsonl,
        summary_json=args.summary_json,
        fixed_budgets=args.fixed_budgets,
        dynamic_budget_buckets=args.dynamic_budget_buckets,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        strip_deploy_invisible_payload=bool(args.strip_deploy_invisible_payload),
        strict_deploy_source=bool(args.strict_deploy_source),
        allow_bootstrap_for_tests=False,
        max_unselected_hole=args.max_unselected_hole,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
