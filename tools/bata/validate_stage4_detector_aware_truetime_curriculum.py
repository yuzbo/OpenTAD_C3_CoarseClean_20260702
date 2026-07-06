from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.bata import detector_aware_acquisition_policy as detector_policy
from tools.bata import detector_teacher_utility


SCHEMA_VERSION = "c3_stage4_detector_aware_truetime_curriculum_evidence_v1"
READY = "C3_STAGE4_CURRICULUM_EVIDENCE_PASS"
ROUTE = "DIVERGENT_INNOVATION_DETECTOR_AWARE_TRUETIME_CURRICULUM_DO_NOT_MERGE_WITH_C3"
STAGE2_ROUTE = "DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3"
STAGE3_ROUTE = "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3"
REQUIRED_PHASES = [
    "dense_teacher_utility_export",
    "detector_aware_selector_pretrain",
    "offline_sparse_detector_warmup",
    "truetime_st_joint_smoke",
    "bilevel_fulltrain_candidate",
]
REQUIRED_QUESTIONS = [
    "Does AdaTAD teacher utility train a better acquisition policy than p_action-only?",
    "Does sparse detector mAP survive fixed_384/fixed_768/dynamic ledgers?",
    "Does detector loss produce non-zero selector gradients through TrueTime ST selection?",
    "Can curriculum/bilevel training avoid selector collapse and high-IoU mAP degradation?",
]
REQUIRED_LEDGER_VARIANTS = [
    detector_policy.DETECTOR_AWARE_FIXED_384_STRATEGY,
    detector_policy.DETECTOR_AWARE_FIXED_768_STRATEGY,
    detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY,
]
REQUIRED_LEDGER_SPLITS = ["train", "val", "test"]
CLAIM_LOCK_FALSE_KEYS = [
    "end_to_end_claim_allowed",
    "paper_claim_allowed",
    "runtime_flops_claim_allowed",
    "deploy_claim_allowed",
    "map_claim_allowed",
]
FORBIDDEN_TRUE_FLAGS = [
    "uses_gt_for_selection",
    "uses_val_or_test_gt_for_selection",
    "uses_val_gt",
    "uses_test_gt",
    "uses_oracle",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "load_from_raw_predictions",
    "uses_uniform_fill",
    "uses_uniform_scaffold",
]


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _require_false_flags(payload: Mapping[str, Any], *, context: str, keys: Sequence[str] = FORBIDDEN_TRUE_FLAGS) -> None:
    for key in keys:
        if _is_true(payload.get(key, False)):
            raise ValueError(f"{context}: forbidden flag {key}=true")


def _require_claim_locks(payload: Mapping[str, Any], *, context: str) -> None:
    for key in CLAIM_LOCK_FALSE_KEYS:
        _require(payload.get(key) is False, f"{context}: {key} must be false")


def _require_nonempty_string(payload: Mapping[str, Any], key: str, *, context: str) -> None:
    _require(isinstance(payload.get(key), str) and bool(str(payload[key]).strip()), f"{context}: missing {key}")


def _validate_stage2_teacher_evidence(payload: Mapping[str, Any]) -> None:
    _require(payload.get("decision") == "C3_DETECTOR_TEACHER_UTILITY_EVIDENCE_PASS", "Stage2 teacher evidence must pass")
    _require(payload.get("stage_label") == detector_teacher_utility.STAGE_LABEL, "Stage2 teacher stage_label mismatch")
    _require(payload.get("route_label") == STAGE2_ROUTE, "Stage2 teacher route_label mismatch")
    _require(payload.get("teacher_signal_source") == "adatad_dense_teacher", "Stage2 teacher source must be AdaTAD dense teacher")
    _require(payload.get("split_scope") == "train_only", "Stage2 teacher split_scope must be train_only")
    for key in (
        "teacher_checkpoint_path",
        "teacher_checkpoint_sha256",
        "teacher_config_path",
        "teacher_config_sha256",
        "validated_output_jsonl_sha256",
    ):
        _require_nonempty_string(payload, key, context="Stage2 teacher evidence")
    _require_false_flags(payload, context="Stage2 teacher evidence")
    _require(payload.get("end_to_end") is False, "Stage2 teacher evidence must not claim end-to-end")


def _validate_stage2_policy_evidence(payload: Mapping[str, Any]) -> None:
    _require(payload.get("decision") == "C3_DETECTOR_AWARE_POLICY_TRAIN_READY", "Stage2 policy must be trained and ready")
    _require(payload.get("policy_family") == "detector_aware_offline_selector", "Stage2 policy family mismatch")
    _require(payload.get("stage_label") == detector_policy.STAGE_LABEL, "Stage2 policy stage_label mismatch")
    _require(payload.get("teacher_target_scope") == "train_only", "Stage2 policy teacher_target_scope must be train_only")
    _require_nonempty_string(payload, "checkpoint_sha256", context="Stage2 policy evidence")
    _require_nonempty_string(payload, "train_jsonl_sha256", context="Stage2 policy evidence")
    source = payload.get("policy_source")
    if source is not None:
        _require(
            source == detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
            "Stage2 policy_source must be learned_detector_aware_policy_checkpoint",
        )
    _require(payload.get("end_to_end") is False, "Stage2 policy must not claim end-to-end")
    _require_false_flags(payload, context="Stage2 policy evidence")


def _split_name(payload: Mapping[str, Any]) -> str | None:
    for key in ("split", "subset", "subset_name", "dataset_split"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            raw = value.strip().lower()
            if raw == "training":
                return "train"
            if raw in {"valid", "validation"}:
                return "val"
            return raw
    return None


def _require_numeric_key(payload: Mapping[str, Any], key: str, *, context: str) -> None:
    _require(key in payload, f"{context}: missing {key}")
    value = payload.get(key)
    if value is None:
        return
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{context}: {key} must be numeric or null")


def _validate_stage2_ledger_summary(payload: Mapping[str, Any]) -> str:
    _require(payload.get("decision") == "C3_DETECTOR_AWARE_POLICY_LEDGER_VALIDATION_PASS", "Stage2 ledger must pass")
    _require(payload.get("schema_version") == "c3_detector_aware_policy_ledger_validation_v1", "Stage2 ledger schema mismatch")
    _require(payload.get("stage_label") == detector_policy.STAGE_LABEL, "Stage2 ledger stage_label mismatch")
    strategy = str(payload.get("strategy") or "")
    _require(strategy in REQUIRED_LEDGER_VARIANTS, f"Stage2 ledger strategy must be one of {REQUIRED_LEDGER_VARIANTS}")
    _require(
        payload.get("required_policy_source") == detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
        "Stage2 ledger must require learned_detector_aware_policy_checkpoint",
    )
    _require(payload.get("map_claim_allowed") is False, "Stage2 ledger mAP claim must be locked")
    _require(payload.get("end_to_end") is False, "Stage2 ledger must not claim end-to-end")
    _require(payload.get("adatad_map") is None, "Stage2 ledger validation must not contain detector mAP")
    _require(payload.get("uses_uniform_fill") is False, "Stage2 ledger must disable uniform fill")
    _require(payload.get("uses_uniform_scaffold") is False, "Stage2 ledger must disable uniform scaffold")
    _require(int(payload.get("total_uniform_visible_fill_count", 0) or 0) == 0, "Stage2 ledger visible fill count must be 0")
    for key in (
        "min_selected_count",
        "max_selected_count",
        "mean_selected_count",
        "max_gap",
        "p95_gap",
        "max_unselected_hole",
        "p95_unselected_hole",
        "max_uniform_similarity",
        "action_positive_coverage",
    ):
        _require_numeric_key(payload, key, context=f"Stage2 ledger {strategy}")
    _require(
        "boundary_support_r1" in payload or "boundary_support@r1" in payload,
        f"Stage2 ledger {strategy}: boundary_support r1 key is required",
    )
    if strategy == detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY:
        min_count = int(payload.get("min_selected_count", 0) or 0)
        max_count = int(payload.get("max_selected_count", 0) or 0)
        iqr = float(payload.get("dynamic_budget_iqr", 0.0) or 0.0)
        entropy = float(payload.get("dynamic_budget_entropy", 0.0) or 0.0)
        _require(max_count > min_count or iqr > 0.0 or entropy > 0.0, "Stage2 dynamic ledger selected_count collapsed")
    return strategy


def _validate_stage2_ledger_summaries(payloads: Sequence[Mapping[str, Any]]) -> None:
    _require(payloads, "Stage4 requires Stage2 detector-aware ledger validation summaries")
    seen_variants: set[str] = set()
    seen_with_split: set[tuple[str, str]] = set()
    split_annotated = 0
    for idx, payload in enumerate(payloads, start=1):
        strategy = _validate_stage2_ledger_summary(payload)
        seen_variants.add(strategy)
        split = _split_name(payload)
        if split is not None:
            split_annotated += 1
            seen_with_split.add((strategy, split))
    missing_variants = sorted(set(REQUIRED_LEDGER_VARIANTS) - seen_variants)
    _require(not missing_variants, f"Stage2 ledger summaries missing variants: {missing_variants}")
    if split_annotated:
        missing = [
            (variant, split)
            for variant in REQUIRED_LEDGER_VARIANTS
            for split in REQUIRED_LEDGER_SPLITS
            if (variant, split) not in seen_with_split
        ]
        _require(not missing, f"Stage2 ledger summaries missing split coverage: {missing}")


def _validate_stage3_proof(payload: Mapping[str, Any]) -> None:
    _require(payload.get("route_variant") == STAGE3_ROUTE, "Stage3 proof route mismatch")
    _require(payload.get("geometry_roundtrip_passed") is True, "Stage3 geometry roundtrip proof missing")
    _require(payload.get("prediction_inverse_map_passed") is True, "Stage3 prediction inverse-map proof missing")
    _require(payload.get("selected_input_st_gradient_passed") is True, "Stage3 selected-input ST proof missing")
    _require(float(payload.get("selected_input_selector_grad_norm", 0.0)) > 0.0, "Stage3 selected-input grad norm must be > 0")
    _require(payload.get("detector_loss_selector_grad_passed") is True, "Stage3 detector-loss selector proof missing")
    _require(float(payload.get("detector_loss_selector_grad_norm", 0.0)) > 0.0, "Stage3 detector-loss grad norm must be > 0")
    _require(payload.get("actionformer_detector_loss_selector_grad_passed") is True, "ActionFormer detector-loss proof missing")
    _require(
        float(payload.get("actionformer_detector_loss_selector_grad_norm", 0.0)) > 0.0,
        "ActionFormer detector-loss grad norm must be > 0",
    )
    _require(payload.get("selector_grad_nonzero") is True, "Stage3 selector_grad_nonzero proof missing")
    _require(payload.get("proof_source") == "registered_detector_forward_train_cost_backward", "Stage3 proof source mismatch")
    _require(
        payload.get("actionformer_proof_source") == "opentad_actionformer_forward_train_cost_backward",
        "Stage3 ActionFormer proof source mismatch",
    )
    _require("loss_cls" in list(payload.get("loss_keys", [])), "Stage3 proof missing loss_cls")
    _require("loss_reg" in list(payload.get("loss_keys", [])), "Stage3 proof missing loss_reg")
    _require("cls_loss" in list(payload.get("actionformer_loss_keys", [])), "Stage3 ActionFormer proof missing cls_loss")
    _require("reg_loss" in list(payload.get("actionformer_loss_keys", [])), "Stage3 ActionFormer proof missing reg_loss")
    _require(payload.get("actionformer_selected_axis_smoke") is True, "Stage3 selected-axis smoke flag missing")
    _require_false_flags(payload, context="Stage3 proof")


def build_evidence(
    *,
    stage2_teacher_evidence: Mapping[str, Any],
    stage2_policy_evidence: Mapping[str, Any],
    stage2_ledger_validation_summaries: Sequence[Mapping[str, Any]],
    stage3_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fail-closed Stage4 evidence bundle from Stage2 and Stage3 artifacts."""
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "route_label": ROUTE,
        "stage_label": "Stage-4 detector-aware TrueTime curriculum evidence gate",
        "required_phases": list(REQUIRED_PHASES),
        "questions_answered_only_after_fulltrain": list(REQUIRED_QUESTIONS),
        "stage2_teacher_utility_evidence": dict(stage2_teacher_evidence),
        "stage2_policy_evidence": dict(stage2_policy_evidence),
        "stage2_ledger_validation_summaries": [dict(item) for item in stage2_ledger_validation_summaries],
        "stage3_truetime_grad_proof": dict(stage3_proof),
        "curriculum_contract": {
            "dense_teacher_to_selector_pretrain": True,
            "selector_pretrain_to_sparse_detector_warmup": True,
            "sparse_detector_warmup_to_st_joint_finetune": True,
            "bilevel_fulltrain_requires_sparse_detector_map": True,
            "stage3_smoke_stays_separate_from_stage2_offline_selector": True,
            "true_time_adapter_required": True,
            "hard_topk_inference_required": True,
        },
        "claim_locks": {key: False for key in CLAIM_LOCK_FALSE_KEYS},
        "leakage_locks": {key: False for key in FORBIDDEN_TRUE_FLAGS},
        "stage_status": {
            "stage2_detector_aware_selector": "implemented_evidence_required",
            "stage3_st_joint_smoke": "implemented_gradient_smoke_required",
            "stage4_curriculum_bilevel": "planned_gate_only_no_map_claim",
        },
        "end_to_end_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "map_claim_allowed": False,
        "uses_gt_for_selection": False,
        "uses_val_or_test_gt_for_selection": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "load_from_raw_predictions": False,
        "uses_uniform_fill": False,
        "uses_uniform_scaffold": False,
    }


def validate_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "Stage4 schema_version mismatch")
    _require(payload.get("decision") == READY, "Stage4 decision is not pass")
    _require(payload.get("route_label") == ROUTE, "Stage4 route_label mismatch")
    _require(list(payload.get("required_phases", [])) == REQUIRED_PHASES, "Stage4 required phases changed")
    _require_claim_locks(payload, context="Stage4 top-level claim locks")
    _require_false_flags(payload, context="Stage4 top-level leakage locks")
    claim_locks = payload.get("claim_locks")
    _require(isinstance(claim_locks, Mapping), "Stage4 claim_locks are required")
    _require_claim_locks(claim_locks, context="Stage4 claim_locks")
    leakage_locks = payload.get("leakage_locks")
    _require(isinstance(leakage_locks, Mapping), "Stage4 leakage_locks are required")
    _require_false_flags(leakage_locks, context="Stage4 leakage_locks")

    curriculum = payload.get("curriculum_contract")
    _require(isinstance(curriculum, Mapping), "Stage4 curriculum_contract is required")
    for key in (
        "dense_teacher_to_selector_pretrain",
        "selector_pretrain_to_sparse_detector_warmup",
        "sparse_detector_warmup_to_st_joint_finetune",
        "bilevel_fulltrain_requires_sparse_detector_map",
        "stage3_smoke_stays_separate_from_stage2_offline_selector",
        "true_time_adapter_required",
        "hard_topk_inference_required",
    ):
        _require(curriculum.get(key) is True, f"Stage4 curriculum contract requires {key}=true")

    stage2_teacher = payload.get("stage2_teacher_utility_evidence")
    _require(isinstance(stage2_teacher, Mapping), "Stage4 requires Stage2 teacher utility evidence")
    _validate_stage2_teacher_evidence(stage2_teacher)

    stage2_policy = payload.get("stage2_policy_evidence")
    _require(isinstance(stage2_policy, Mapping), "Stage4 requires Stage2 policy evidence")
    _validate_stage2_policy_evidence(stage2_policy)

    stage2_ledgers = payload.get("stage2_ledger_validation_summaries")
    _require(isinstance(stage2_ledgers, list), "Stage4 requires Stage2 ledger validation summaries")
    _validate_stage2_ledger_summaries([item for item in stage2_ledgers if isinstance(item, Mapping)])

    stage3_proof = payload.get("stage3_truetime_grad_proof")
    _require(isinstance(stage3_proof, Mapping), "Stage4 requires Stage3 TrueTime grad proof")
    _validate_stage3_proof(stage3_proof)

    out = dict(payload)
    out["decision"] = READY
    return out


def _template() -> dict[str, Any]:
    ledger_template = [
        {
            "schema_version": "c3_detector_aware_policy_ledger_validation_v1",
            "decision": "C3_DETECTOR_AWARE_POLICY_LEDGER_VALIDATION_PASS",
            "stage_label": detector_policy.STAGE_LABEL,
            "strategy": strategy,
            "required_policy_source": detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
            "min_selected_count": 384 if "384" in strategy else 512,
            "max_selected_count": 384 if "384" in strategy else 768,
            "mean_selected_count": 384.0 if "384" in strategy else 640.0,
            "max_gap": 8,
            "p95_gap": 8.0,
            "max_unselected_hole": 8,
            "p95_unselected_hole": 8.0,
            "max_uniform_similarity": 0.50,
            "boundary_support_r1": 0.0,
            "action_positive_coverage": 0.0,
            "dynamic_budget_iqr": 1.0 if strategy == detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY else 0.0,
            "dynamic_budget_entropy": 1.0 if strategy == detector_policy.DETECTOR_AWARE_DYNAMIC_STRATEGY else 0.0,
            "total_uniform_visible_fill_count": 0,
            "uses_uniform_fill": False,
            "uses_uniform_scaffold": False,
            "map_claim_allowed": False,
            "adatad_map": None,
            "end_to_end": False,
        }
        for strategy in REQUIRED_LEDGER_VARIANTS
    ]
    return build_evidence(
        stage2_teacher_evidence={
            "decision": "C3_DETECTOR_TEACHER_UTILITY_EVIDENCE_PASS",
            "stage_label": detector_teacher_utility.STAGE_LABEL,
            "route_label": STAGE2_ROUTE,
            "teacher_signal_source": "adatad_dense_teacher",
            "split_scope": "train_only",
            "teacher_checkpoint_path": "REPLACE_WITH_DENSE_TEACHER_CHECKPOINT",
            "teacher_checkpoint_sha256": "REPLACE_WITH_SHA256",
            "teacher_config_path": "REPLACE_WITH_DENSE_TEACHER_CONFIG",
            "teacher_config_sha256": "REPLACE_WITH_SHA256",
            "validated_output_jsonl_sha256": "REPLACE_WITH_SHA256",
            "uses_gt_for_selection": False,
            "uses_val_or_test_gt_for_selection": False,
            "uses_prediction_cache": False,
            "uses_raw_prediction": False,
            "load_from_raw_predictions": False,
            "uses_uniform_fill": False,
            "uses_uniform_scaffold": False,
            "end_to_end": False,
        },
        stage2_policy_evidence={
            "decision": "C3_DETECTOR_AWARE_POLICY_TRAIN_READY",
            "policy_family": "detector_aware_offline_selector",
            "stage_label": detector_policy.STAGE_LABEL,
            "policy_source": detector_policy.DETECTOR_AWARE_CHECKPOINT_POLICY_SOURCE,
            "teacher_target_scope": "train_only",
            "checkpoint_sha256": "REPLACE_WITH_SHA256",
            "train_jsonl_sha256": "REPLACE_WITH_SHA256",
            "uses_gt_for_selection": False,
            "uses_val_or_test_gt_for_selection": False,
            "uses_prediction_cache": False,
            "uses_raw_prediction": False,
            "load_from_raw_predictions": False,
            "uses_uniform_fill": False,
            "uses_uniform_scaffold": False,
            "end_to_end": False,
        },
        stage2_ledger_validation_summaries=ledger_template,
        stage3_proof={
            "route_variant": STAGE3_ROUTE,
            "geometry_roundtrip_passed": True,
            "prediction_inverse_map_passed": True,
            "selected_input_st_gradient_passed": True,
            "selected_input_selector_grad_norm": 0.1,
            "detector_loss_selector_grad_passed": True,
            "detector_loss_selector_grad_norm": 0.1,
            "selector_grad_nonzero": True,
            "loss_keys": ["loss_cls", "loss_reg"],
            "proof_source": "registered_detector_forward_train_cost_backward",
            "actionformer_proof_source": "opentad_actionformer_forward_train_cost_backward",
            "actionformer_detector_loss_selector_grad_passed": True,
            "actionformer_detector_loss_selector_grad_norm": 0.1,
            "actionformer_loss_keys": ["cls_loss", "reg_loss"],
            "actionformer_selected_axis_smoke": True,
            "uses_gt_for_selection": False,
            "uses_val_or_test_gt_for_selection": False,
            "uses_prediction_cache": False,
            "uses_raw_prediction": False,
            "load_from_raw_predictions": False,
            "uses_uniform_fill": False,
            "uses_uniform_scaffold": False,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Stage4 detector-aware TrueTime curriculum evidence.")
    parser.add_argument("--evidence-json")
    parser.add_argument("--write-evidence-json")
    parser.add_argument("--write-template-json")
    parser.add_argument("--stage2-teacher-summary-json")
    parser.add_argument("--stage2-teacher-output-jsonl")
    parser.add_argument("--stage2-policy-summary-json")
    parser.add_argument("--stage2-ledger-summary-json", nargs="*", default=[])
    parser.add_argument("--stage3-proof-json")
    args = parser.parse_args(argv)

    if args.write_template_json:
        _write_json(args.write_template_json, _template())
        print(str(args.write_template_json), flush=True)
        return 0

    if args.evidence_json:
        payload = _read_json(args.evidence_json)
    else:
        for name in ("stage2_teacher_summary_json", "stage2_policy_summary_json", "stage3_proof_json"):
            if getattr(args, name) is None:
                raise ValueError(f"--{name.replace('_', '-')} is required when --evidence-json is not provided")
        if not args.stage2_ledger_summary_json:
            raise ValueError("--stage2-ledger-summary-json is required at least once")
        stage2_teacher = detector_teacher_utility.validate_teacher_utility_export_evidence(
            args.stage2_teacher_summary_json,
            output_jsonl=args.stage2_teacher_output_jsonl,
            require_paction=True,
        )
        payload = build_evidence(
            stage2_teacher_evidence=stage2_teacher,
            stage2_policy_evidence=_read_json(args.stage2_policy_summary_json),
            stage2_ledger_validation_summaries=[_read_json(path) for path in args.stage2_ledger_summary_json],
            stage3_proof=_read_json(args.stage3_proof_json),
        )

    validated = validate_evidence(payload)
    if args.write_evidence_json:
        _write_json(args.write_evidence_json, validated)
    print(READY, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
