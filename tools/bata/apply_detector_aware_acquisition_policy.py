from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata import apply_gap_aware_acquisition_policy as gas_apply
from tools.bata import detector_aware_acquisition_policy as detector_policy
from tools.bata import detector_deploy_leakage
from tools.bata import paction_acquisition_policy as base_policy
from tools.bata import paction_source_samples


SUMMARY_SCHEMA_VERSION = "c3_detector_aware_policy_application_v1"
READY = "C3_DETECTOR_AWARE_POLICY_APPLICATION_READY"
BOOTSTRAP_POLICY_SOURCE = detector_policy.DETECTOR_AWARE_BOOTSTRAP_POLICY_SOURCE
CHECKPOINT_POLICY_SOURCE = detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE
DETECTOR_DEPLOY_FORBIDDEN_PAYLOAD_KEYS = detector_deploy_leakage.DETECTOR_DEPLOY_FORBIDDEN_PAYLOAD_KEYS


_read_jsonl = gas_apply._read_jsonl
_write_jsonl = gas_apply._write_jsonl
_write_json = gas_apply._write_json
_sha256_file = gas_apply._sha256_file
_strip_deploy_invisible_payload = gas_apply._strip_deploy_invisible_payload
_extract_paction = gas_apply._extract_paction
_reject_forbidden_source_flags = gas_apply._reject_forbidden_source_flags
_paction_positive_provenance = gas_apply._paction_positive_provenance


def _strip_detector_invisible_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _strip_deploy_invisible_payload(row)
    out = detector_deploy_leakage.strip_detector_deploy_forbidden_payloads(out)
    out["detector_deploy_forbidden_payload_stripped"] = True
    return out


def _reject_detector_payload(row: Mapping[str, Any], *, source_name: str) -> None:
    detector_deploy_leakage.reject_detector_deploy_forbidden_payloads(row, source_name=source_name)


def bootstrap_policy_scores(
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    dynamic_budget_buckets: Sequence[int] = detector_policy.DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS,
) -> tuple[list[float], list[float]]:
    features = detector_policy.build_detector_aware_feature_matrix(
        p_action,
        valid=valid,
        target_budget=max(int(item) for item in dynamic_budget_buckets) if dynamic_budget_buckets else len(p_action),
    )
    p_idx = detector_policy.feature_index("p_action")
    density_idx = detector_policy.feature_index("local_density")
    change_idx = detector_policy.feature_index("local_change")
    entropy_idx = detector_policy.feature_index("entropy")
    urgency_idx = detector_policy.feature_index("gap_urgency")
    valid_idx = detector_policy.feature_index("valid")
    frame_values = [
        (
            0.55 * float(row[p_idx])
            + 0.50 * float(row[density_idx])
            + 0.40 * float(row[change_idx])
            + 0.25 * float(row[entropy_idx])
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


def _validate_checkpoint_utility_contract(payload: Mapping[str, Any], *, require_point_responsibility_utility: bool) -> None:
    if not require_point_responsibility_utility:
        return
    if payload.get("utility_semantics") != "signed_point_responsibility_utility_v1":
        raise ValueError("detector-aware main checkpoint must use signed_point_responsibility_utility_v1")
    if payload.get("utility_source_type") != "point_loss_gradient_responsibility_v1":
        raise ValueError("detector-aware main checkpoint must use point_loss_gradient_responsibility_v1")
    if payload.get("point_responsibility_utility") is not True:
        raise ValueError("detector-aware main checkpoint requires point_responsibility_utility=true")
    if payload.get("proposal_score_surrogate_utility") is not False:
        raise ValueError("detector-aware main checkpoint rejects proposal_score_surrogate_utility")
    if payload.get("paper_main_target_allowed") is not True:
        raise ValueError("detector-aware main checkpoint must declare paper_main_target_allowed=true")


def load_policy_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cuda",
    require_point_responsibility_utility: bool = False,
) -> tuple[detector_policy.DetectorAwareSequentialAcquisitionPolicy, dict[str, Any]]:
    import torch

    from tools.bata import train_detector_aware_acquisition_policy as train_policy

    payload = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("detector-aware policy checkpoint must be a mapping")
    if payload.get("schema_version") != train_policy.CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("detector-aware policy checkpoint schema_version mismatch")
    if payload.get("decision") != train_policy.READY:
        raise ValueError("detector-aware policy checkpoint decision is not ready")
    if payload.get("policy_family") != "detector_aware_offline_selector":
        raise ValueError("detector-aware policy checkpoint policy_family mismatch")
    if payload.get("teacher_target_scope") != "train_only":
        raise ValueError("detector-aware policy checkpoint teacher_target_scope must be train_only")
    if payload.get("end_to_end") is not False:
        raise ValueError("detector-aware policy checkpoint must declare end_to_end=false")
    if payload.get("uses_uniform_scaffold") is not False or payload.get("uses_uniform_fill") is not False:
        raise ValueError("detector-aware policy checkpoint must disable uniform scaffold/fill")
    if not isinstance(payload.get("train_jsonl"), str) or not payload.get("train_jsonl"):
        raise ValueError("detector-aware policy checkpoint is missing train_jsonl")
    if not isinstance(payload.get("train_jsonl_sha256"), str) or len(payload.get("train_jsonl_sha256", "")) != 64:
        raise ValueError("detector-aware policy checkpoint is missing train_jsonl_sha256")
    _validate_checkpoint_utility_contract(
        payload,
        require_point_responsibility_utility=bool(require_point_responsibility_utility),
    )
    state_dict = payload.get("policy_state_dict")
    if state_dict is None:
        raise ValueError("detector-aware policy checkpoint is missing policy_state_dict")
    model_kwargs = dict(payload.get("model_kwargs") or {})
    if not model_kwargs:
        model_kwargs = {
            "input_dim": len(detector_policy.DETECTOR_AWARE_FEATURE_NAMES),
            "budget_buckets": payload.get("dynamic_budget_buckets", detector_policy.DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS),
        }
    model = detector_policy.DetectorAwareSequentialAcquisitionPolicy(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, dict(payload)


def checkpoint_policy_scores(
    model: detector_policy.DetectorAwareSequentialAcquisitionPolicy,
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any] | None = None,
    target_budget: int | None = None,
    device: str = "cuda",
    return_context_radius: bool = False,
) -> tuple[list[float], list[float]] | tuple[list[float], list[float], list[float]]:
    import torch

    features = detector_policy.build_detector_aware_feature_matrix(p_action, valid=valid, target_budget=target_budget)
    with torch.no_grad():
        feature_tensor = torch.tensor([features], dtype=torch.float32, device=device)
        valid_tensor = None if valid is None else torch.tensor([list(valid)], dtype=torch.bool, device=device)
        budget_tensor = None if target_budget is None else torch.tensor([int(target_budget)], dtype=torch.long, device=device)
        outputs = model(feature_tensor, valid_tensor, target_budget=budget_tensor)
    frame_values = [float(item) for item in outputs["frame_value"][0].detach().cpu().tolist()]
    budget_scores = [float(item) for item in outputs["budget_logits"][0].detach().cpu().tolist()]
    if return_context_radius:
        context_radius = [float(item) for item in outputs["context_radius"][0].detach().cpu().tolist()]
        return frame_values, budget_scores, context_radius
    return frame_values, budget_scores


def _effective_strategy_budget(row: Mapping[str, Any], requested_budget: int, *, frame_len: int) -> int:
    valid_len = int(row.get("valid_len") or row.get("dense_len") or frame_len)
    return detector_policy.fixed_deploy_budget(int(requested_budget), valid_len=valid_len)


def _checkpoint_policy_scores_with_context(
    model: detector_policy.DetectorAwareSequentialAcquisitionPolicy,
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any],
    target_budget: int,
    device: str,
    allow_legacy_no_context_radius: bool = False,
    legacy_context_radius: float | None = None,
) -> tuple[list[float], list[float], list[float]]:
    def legacy_radius(frame_values: Sequence[Any]) -> list[float]:
        if not bool(allow_legacy_no_context_radius):
            raise ValueError(
                "detector-aware checkpoint did not return context_radius; "
                "retrain with context_radius_head or pass explicit legacy context radius opt-in"
            )
        if legacy_context_radius is None:
            raise ValueError("legacy_context_radius is required when allow_legacy_no_context_radius=true")
        radius = detector_policy._clamp_context_radius(float(legacy_context_radius))
        return [float(radius)] * len(frame_values)

    try:
        result = checkpoint_policy_scores(
            model,
            p_action,
            valid=valid,
            target_budget=target_budget,
            device=device,
            return_context_radius=True,
        )
    except TypeError as exc:
        if "return_context_radius" not in str(exc):
            raise
        frame_values, budget_scores = checkpoint_policy_scores(
            model,
            p_action,
            valid=valid,
            target_budget=target_budget,
            device=device,
        )
        context_radius = legacy_radius(frame_values)
        return list(frame_values), list(budget_scores), context_radius
    if len(result) == 2:
        frame_values, budget_scores = result
        context_radius = legacy_radius(frame_values)
        return list(frame_values), list(budget_scores), context_radius
    frame_values, budget_scores, context_radius = result
    return list(frame_values), list(budget_scores), list(context_radius)


def _checkpoint_budget_conditioned_scores(
    model: detector_policy.DetectorAwareSequentialAcquisitionPolicy,
    row: Mapping[str, Any],
    p_action: Sequence[Any],
    *,
    valid: Sequence[Any],
    fixed_budgets: Sequence[int],
    dynamic_budget_buckets: Sequence[int],
    device: str,
    allow_legacy_no_context_radius: bool = False,
    legacy_context_radius: float | None = None,
) -> tuple[list[float], list[float], dict[str, list[float]], dict[str, list[float]], dict[str, int], int, str]:
    valid_len = int(row.get("valid_len") or row.get("dense_len") or len(p_action))
    initial_budget = max([int(item) for item in dynamic_budget_buckets] or [valid_len])
    _initial_frame_values, budget_scores, _initial_context_radius = _checkpoint_policy_scores_with_context(
        model,
        p_action,
        valid=valid,
        target_budget=initial_budget,
        device=device,
        allow_legacy_no_context_radius=bool(allow_legacy_no_context_radius),
        legacy_context_radius=legacy_context_radius,
    )
    dynamic_budget = base_policy.decode_budget_from_scores(
        budget_scores,
        dynamic_budget_buckets,
        valid_len=valid_len,
    )
    frame_values_by_strategy: dict[str, list[float]] = {}
    context_radii_by_strategy: dict[str, list[float]] = {}
    target_budgets: dict[str, int] = {}
    fixed_strategy_names = (
        detector_policy.DETECTOR_AWARE_FIXED_384_STRATEGY,
        detector_policy.DETECTOR_AWARE_FIXED_768_STRATEGY,
    )
    fixed_values = [int(item) for item in fixed_budgets]
    for strategy_idx, strategy_name in enumerate(fixed_strategy_names):
        requested = fixed_values[strategy_idx] if strategy_idx < len(fixed_values) else fixed_values[-1]
        strategy_budget = _effective_strategy_budget(row, requested, frame_len=len(p_action))
        frame_values, _unused_budget_scores, context_radius = _checkpoint_policy_scores_with_context(
            model,
            p_action,
            valid=valid,
            target_budget=strategy_budget,
            device=device,
            allow_legacy_no_context_radius=bool(allow_legacy_no_context_radius),
            legacy_context_radius=legacy_context_radius,
        )
        frame_values_by_strategy[strategy_name] = frame_values
        context_radii_by_strategy[strategy_name] = context_radius
        target_budgets[strategy_name] = int(strategy_budget)
    dynamic_frame_values, _unused_budget_scores, dynamic_context_radius = _checkpoint_policy_scores_with_context(
        model,
        p_action,
        valid=valid,
        target_budget=dynamic_budget,
        device=device,
        allow_legacy_no_context_radius=bool(allow_legacy_no_context_radius),
        legacy_context_radius=legacy_context_radius,
    )
    frame_values_by_strategy[detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY] = dynamic_frame_values
    context_radii_by_strategy[detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY] = dynamic_context_radius
    target_budgets[detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY] = int(dynamic_budget)
    context_radius_source = "legacy_fixed_explicit" if bool(allow_legacy_no_context_radius) else "checkpoint_context_radius_head"
    return (
        dynamic_frame_values,
        budget_scores,
        frame_values_by_strategy,
        context_radii_by_strategy,
        target_budgets,
        int(initial_budget),
        context_radius_source,
    )


def run_policy_application(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    summary_json: str | Path | None = None,
    fixed_budgets: Sequence[int] = (384, 768),
    dynamic_budget_buckets: Sequence[int] = detector_policy.DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
    strip_deploy_invisible_payload: bool = False,
    strict_deploy_source: bool = False,
    allow_bootstrap_for_tests: bool = False,
    max_unselected_hole: int | None = None,
    max_uniform_similarity: float | None = None,
    source_jsonl_for_hash: str | Path | None = None,
    require_point_responsibility_utility: bool = False,
    allow_legacy_no_context_radius: bool = False,
    legacy_context_radius: float | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    if checkpoint_path is None and not allow_bootstrap_for_tests:
        raise ValueError("checkpoint_path is required for detector-aware policy application")
    if checkpoint_path is not None and allow_bootstrap_for_tests:
        raise ValueError("allow_bootstrap_for_tests cannot be combined with checkpoint_path")
    source_jsonl_sha256 = _sha256_file(source_jsonl_for_hash or input_jsonl)
    checkpoint_sha256: str | None = None
    checkpoint_model: detector_policy.DetectorAwareSequentialAcquisitionPolicy | None = None
    checkpoint_payload: dict[str, Any] | None = None
    source = BOOTSTRAP_POLICY_SOURCE
    if checkpoint_path is not None:
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        source = CHECKPOINT_POLICY_SOURCE
        if not allow_bootstrap_for_tests:
            checkpoint_model, checkpoint_payload = load_policy_checkpoint(
                checkpoint_path,
                device=device,
                require_point_responsibility_utility=bool(require_point_responsibility_utility),
            )
            dynamic_budget_buckets = checkpoint_payload.get("dynamic_budget_buckets", dynamic_budget_buckets)
    dynamic_gain_calibration = dict(detector_policy.DEFAULT_DYNAMIC_GAIN_CALIBRATION)
    if checkpoint_payload is not None and isinstance(checkpoint_payload.get("dynamic_gain_calibration"), Mapping):
        dynamic_gain_calibration.update(dict(checkpoint_payload["dynamic_gain_calibration"]))
    enriched_rows: list[dict[str, Any]] = []
    dynamic_budgets: list[int] = []
    for line_no, row in enumerate(rows, start=1):
        source_name = f"{input_jsonl}:{line_no}"
        if strict_deploy_source:
            _reject_detector_payload(row, source_name=source_name)
            p_action_provenance = paction_source_samples.reject_strict_deploy_source_row(
                row,
                source_name=source_name,
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
            context_radii_by_strategy = None
            context_radius_source = "bootstrap_no_context_radius"
            apply_time_target_budgets: dict[str, int] = {}
            budget_score_target_budget = None
        else:
            (
                frame_values,
                budget_scores,
                frame_values_by_strategy,
                context_radii_by_strategy,
                apply_time_target_budgets,
                budget_score_target_budget,
                context_radius_source,
            ) = _checkpoint_budget_conditioned_scores(
                checkpoint_model,
                row,
                p_action,
                valid=valid,
                fixed_budgets=fixed_budgets,
                dynamic_budget_buckets=dynamic_budget_buckets,
                device=device,
                allow_legacy_no_context_radius=bool(allow_legacy_no_context_radius),
                legacy_context_radius=legacy_context_radius,
            )
        enriched = detector_policy.add_detector_aware_decision_to_sample_row(
            row,
            frame_values=frame_values,
            frame_values_by_strategy=frame_values_by_strategy,
            context_radii_by_strategy=context_radii_by_strategy,
            fixed_budgets=fixed_budgets,
            dynamic_budget_scores=budget_scores,
            dynamic_budget_buckets=dynamic_budget_buckets,
            dynamic_gain_calibration=dynamic_gain_calibration,
            max_unselected_hole=max_unselected_hole,
            max_uniform_similarity=max_uniform_similarity,
            source=source,
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            source_jsonl_sha256=source_jsonl_sha256,
        )
        enriched["detector_aware_policy"]["p_action_provenance"] = p_action_provenance
        if checkpoint_model is not None:
            enriched["detector_aware_policy"]["apply_time_target_budgets"] = apply_time_target_budgets
            enriched["detector_aware_policy"]["budget_score_target_budget"] = budget_score_target_budget
            enriched["detector_aware_policy"]["budget_conditioning_rule"] = "checkpoint_two_pass_strategy_specific_target_budget"
            enriched["detector_aware_policy"]["context_radius_source"] = context_radius_source
        enriched["detector_aware_policy"]["frame_value_summary"] = {
            "min": min(frame_values) if frame_values else None,
            "max": max(frame_values) if frame_values else None,
            "mean": None if not frame_values else sum(frame_values) / float(len(frame_values)),
        }
        dynamic_budgets.append(int(enriched["detector_aware_policy"]["dynamic_budget"]))
        if strip_deploy_invisible_payload:
            enriched = _strip_detector_invisible_payload(enriched)
        enriched_rows.append(enriched)
    _write_jsonl(output_jsonl, enriched_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "stage_label": detector_policy.STAGE_LABEL,
        "policy_family": "detector_aware_offline_selector",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "row_count": len(enriched_rows),
        "fixed_budgets": [int(item) for item in fixed_budgets],
        "dynamic_budget_buckets": [int(item) for item in dynamic_budget_buckets],
        "dynamic_gain_calibration": dynamic_gain_calibration,
        "min_dynamic_budget": min(dynamic_budgets) if dynamic_budgets else None,
        "max_dynamic_budget": max(dynamic_budgets) if dynamic_budgets else None,
        "mean_dynamic_budget": None if not dynamic_budgets else sum(dynamic_budgets) / float(len(dynamic_budgets)),
        "source": source,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "policy_checkpoint_sha256": checkpoint_sha256,
        "require_point_responsibility_utility": bool(require_point_responsibility_utility),
        "allow_legacy_no_context_radius": bool(allow_legacy_no_context_radius),
        "legacy_context_radius": None if legacy_context_radius is None else float(detector_policy._clamp_context_radius(legacy_context_radius)),
        "source_jsonl_sha256": source_jsonl_sha256,
        "decoder_max_uniform_similarity": None if max_uniform_similarity is None else float(max_uniform_similarity),
        "strip_deploy_invisible_payload": bool(strip_deploy_invisible_payload),
        "strict_deploy_source": bool(strict_deploy_source),
        "allow_bootstrap_for_tests": bool(allow_bootstrap_for_tests),
        "decode_mode": "hard_gap_aware_topk",
        "teacher_payload_visible_to_deploy": False,
        "end_to_end": False,
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
    parser = argparse.ArgumentParser(description="Apply a Stage-2 detector-aware offline selector to p_action rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--fixed-budgets", type=int, nargs="+", default=[384, 768])
    parser.add_argument("--dynamic-budget-buckets", type=int, nargs="+", default=list(detector_policy.DEFAULT_DETECTOR_AWARE_DYNAMIC_BUDGET_BUCKETS))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strip-deploy-invisible-payload", action="store_true")
    parser.add_argument("--strict-deploy-source", action="store_true")
    parser.add_argument("--max-unselected-hole", type=int)
    parser.add_argument("--max-uniform-similarity", type=float)
    parser.add_argument("--require-point-responsibility-utility", action="store_true")
    parser.add_argument("--allow-legacy-no-context-radius", action="store_true")
    parser.add_argument("--legacy-context-radius", type=float)
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
        max_uniform_similarity=args.max_uniform_similarity,
        require_point_responsibility_utility=bool(args.require_point_responsibility_utility),
        allow_legacy_no_context_radius=bool(args.allow_legacy_no_context_radius),
        legacy_context_radius=args.legacy_context_radius,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
