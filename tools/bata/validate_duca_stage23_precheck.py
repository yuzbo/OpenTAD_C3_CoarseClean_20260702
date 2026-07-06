from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bata.detector_teacher_utility import validate_teacher_utility_export_evidence
from tools.bata.validate_c3_detector_aware_adatad_full_train import validate_config as validate_stage2_config
from tools.bata.validate_truetime_joint_selector_precheck import validate_config as validate_stage3_config


STAGE2_READY = "DUCA_STAGE2_PRECHECK_PASS"
STAGE3_READY = "DUCA_STAGE3_PRECHECK_PASS"
STAGE23_READY = "DUCA_STAGE23_PRECHECK_PASS"
STAGE2_POLICY_READY = "C3_DETECTOR_AWARE_POLICY_TRAIN_READY"
STAGE2_VARIANTS = (
    "detector_aware_fixed_384",
    "detector_aware_fixed_768",
    "detector_aware_dynamic",
)
FORBIDDEN_LEAKAGE_FLAGS = (
    "uses_gt",
    "uses_gt_for_selection",
    "uses_val_gt",
    "uses_test_gt",
    "uses_oracle",
    "uses_teacher",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "prediction_uses_gt",
    "uses_evaluator_outputs",
    "load_from_raw_predictions",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    _require(isinstance(payload, dict), f"JSON file must contain an object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            _require(isinstance(row, dict), f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    _require(rows, f"JSONL has no rows: {path}")
    return rows


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _split(row: Mapping[str, Any]) -> str:
    for key in ("split", "subset", "subset_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _validate_signed_list(values: Any, *, expected_len: int, source: str) -> list[float]:
    _require(isinstance(values, list), f"{source}: expected signed utility list")
    _require(len(values) == int(expected_len), f"{source}: signed utility length mismatch")
    out: list[float] = []
    for idx, value in enumerate(values):
        number = float(value)
        _require(math.isfinite(number), f"{source}[{idx}]: signed utility is not finite")
        _require(-1.000001 <= number <= 1.000001, f"{source}[{idx}]: signed utility outside [-1, 1]")
        out.append(number)
    return out


def _validate_file_sha(path: str | Path, *, expected_sha256: str | None, label: str) -> str:
    target = Path(path).expanduser()
    _require(target.is_file(), f"{label} missing: {target}")
    actual = _sha256_file(target)
    if expected_sha256:
        _require(actual == expected_sha256, f"{label} sha256 mismatch")
    return actual


def _validate_teacher_rows(output_jsonl: str | Path) -> dict[str, Any]:
    signed_abs_max = 0.0
    rows = _read_jsonl(output_jsonl)
    for line_no, row in enumerate(rows, start=1):
        _require(_split(row) in {"train", "training"}, f"{output_jsonl}:{line_no}: teacher utility must be train-only")
        teacher_payload = row.get("teacher_utility")
        _require(isinstance(teacher_payload, Mapping), f"{output_jsonl}:{line_no}: missing teacher_utility")
        _require(
            teacher_payload.get("utility_semantics") == "signed_detector_utility_v1",
            f"{output_jsonl}:{line_no}: wrong utility semantics",
        )
        signed = teacher_payload.get("signed_frame_utility", row.get("signed_frame_utility"))
        signed_values = _validate_signed_list(
            signed,
            expected_len=int(row.get("dense_len", 0)),
            source=f"{output_jsonl}:{line_no}:signed_frame_utility",
        )
        gain = teacher_payload.get("positive_observation_gain", row.get("positive_observation_gain"))
        risk = teacher_payload.get("negative_observation_risk", row.get("negative_observation_risk"))
        _require(isinstance(gain, list) and len(gain) == len(signed_values), f"{output_jsonl}:{line_no}: missing gain")
        _require(isinstance(risk, list) and len(risk) == len(signed_values), f"{output_jsonl}:{line_no}: missing risk")
        for idx, signed_value in enumerate(signed_values):
            _require(float(gain[idx]) == max(0.0, signed_value), f"{output_jsonl}:{line_no}: gain mismatch at {idx}")
            _require(float(risk[idx]) == max(0.0, -signed_value), f"{output_jsonl}:{line_no}: risk mismatch at {idx}")
        _require("marginal_gain_frame_utility" not in teacher_payload, f"{output_jsonl}:{line_no}: legacy utility field present")
        provenance = row.get("teacher_utility_provenance")
        _require(isinstance(provenance, Mapping), f"{output_jsonl}:{line_no}: missing teacher provenance")
        _require(provenance.get("split_scope") == "train_only", f"{output_jsonl}:{line_no}: split_scope is not train_only")
        _require(row.get("training_only") is True, f"{output_jsonl}:{line_no}: teacher row must be training_only")
        for flag in FORBIDDEN_LEAKAGE_FLAGS:
            if flag == "uses_teacher":
                continue
            _require(not _is_true(row.get(flag, False)), f"{output_jsonl}:{line_no}: forbidden flag {flag}=true")
        signed_abs_max = max(signed_abs_max, max((abs(item) for item in signed_values), default=0.0))
    _require(signed_abs_max > 0.0, "signed teacher utility is present but all values are zero")
    return {"row_count": len(rows), "signed_abs_max": signed_abs_max}


def _validate_policy_summary(summary_json: str | Path, *, checkpoint_path: str | Path | None) -> dict[str, Any]:
    summary = _read_json(summary_json)
    _require(summary.get("decision") == STAGE2_POLICY_READY, "policy summary decision is not ready")
    _require(summary.get("policy_family") == "detector_aware_offline_selector", "wrong policy family")
    _require(summary.get("utility_semantics") == "signed_detector_utility_v1", "policy must use signed utility")
    _require(summary.get("signed_utility_supported") is True, "policy must declare signed utility support")
    _require(summary.get("teacher_target_scope") == "train_only", "policy target scope must be train_only")
    _require(summary.get("end_to_end") is False, "Stage2 policy must not claim end_to_end")
    calibration = summary.get("dynamic_gain_calibration")
    _require(isinstance(calibration, Mapping), "missing dynamic gain calibration")
    _require(calibration.get("schema_version") == "c3_detector_aware_dynamic_gain_calibration_v1", "wrong calibration schema")
    _require(calibration.get("calibration_fitted") is True, "dynamic calibration must be fitted")
    _require(calibration.get("fit_split") == "training", "dynamic calibration must be fit on training only")
    _require(
        calibration.get("budget_target_rule") == "count_positive_gain_at_global_threshold_then_nearest_bucket",
        "dynamic budget must use train-global threshold rule",
    )
    train_jsonl = Path(str(summary.get("train_jsonl", ""))).expanduser()
    _require(train_jsonl.is_file(), f"policy train_jsonl missing: {train_jsonl}")
    train_sha = _sha256_file(train_jsonl)
    _require(summary.get("train_jsonl_sha256") == train_sha, "policy train_jsonl_sha256 mismatch")
    checkpoint = Path(checkpoint_path or summary.get("checkpoint_path", "")).expanduser()
    _require(checkpoint.is_file(), f"policy checkpoint missing: {checkpoint}")
    checkpoint_sha = _sha256_file(checkpoint)
    _require(summary.get("checkpoint_sha256") == checkpoint_sha, "policy checkpoint sha256 mismatch")
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "train_jsonl": str(train_jsonl),
        "train_jsonl_sha256": train_sha,
        "dynamic_gain_calibration": dict(calibration),
    }


def _validate_no_teacher_leakage_jsonl(path: str | Path, *, split_name: str) -> dict[str, Any]:
    rows = _read_jsonl(path)
    for line_no, row in enumerate(rows, start=1):
        _require(_split(row) not in {"train", "training"}, f"{path}:{line_no}: {split_name} row is training split")
        for key in ("teacher_utility", "frame_utility", "signed_frame_utility", "teacher_utility_provenance"):
            _require(key not in row, f"{path}:{line_no}: val/test row contains {key}")
        for flag in FORBIDDEN_LEAKAGE_FLAGS:
            _require(not _is_true(row.get(flag, False)), f"{path}:{line_no}: forbidden leakage flag {flag}=true")
    return {"path": str(path), "row_count": len(rows)}


def validate_stage2(args: argparse.Namespace) -> dict[str, Any]:
    teacher_evidence = None
    teacher_rows = None
    if args.stage2_teacher_summary_json:
        teacher_evidence = validate_teacher_utility_export_evidence(
            args.stage2_teacher_summary_json,
            output_jsonl=args.stage2_teacher_output_jsonl,
            require_paction=True,
            require_generator_manifest=bool(args.require_stage2_generator_manifest),
        )
        teacher_rows = _validate_teacher_rows(teacher_evidence["validated_output_jsonl"])
        _validate_file_sha(
            args.stage2_teacher_checkpoint_path or teacher_evidence.get("teacher_checkpoint_path"),
            expected_sha256=teacher_evidence.get("teacher_checkpoint_sha256"),
            label="dense AdaTAD teacher checkpoint",
        )
        _validate_file_sha(
            args.stage2_teacher_config_path or teacher_evidence.get("teacher_config_path"),
            expected_sha256=teacher_evidence.get("teacher_config_sha256"),
            label="dense AdaTAD teacher config",
        )
    elif args.require_stage2_teacher_evidence:
        raise AssertionError("Stage2 teacher evidence summary is required")

    policy_evidence = None
    if args.stage2_policy_summary_json:
        policy_evidence = _validate_policy_summary(
            args.stage2_policy_summary_json,
            checkpoint_path=args.stage2_policy_checkpoint_path,
        )
    elif args.require_stage2_policy_evidence:
        raise AssertionError("Stage2 policy training summary is required")

    no_leakage = []
    if args.stage2_val_source_jsonl:
        no_leakage.append(_validate_no_teacher_leakage_jsonl(args.stage2_val_source_jsonl, split_name="val"))
    if args.stage2_test_source_jsonl:
        no_leakage.append(_validate_no_teacher_leakage_jsonl(args.stage2_test_source_jsonl, split_name="test"))

    old_variant = os.environ.get("C3_DETECTOR_AWARE_LEDGER_VARIANT")
    old_root = os.environ.get("C3_DETECTOR_AWARE_LEDGER_ROOT")
    try:
        if args.stage2_ledger_root:
            os.environ["C3_DETECTOR_AWARE_LEDGER_ROOT"] = str(args.stage2_ledger_root)
        for variant in STAGE2_VARIANTS:
            os.environ["C3_DETECTOR_AWARE_LEDGER_VARIANT"] = variant
            validate_stage2_config(args.stage2_config, require_ledger_files=bool(args.require_stage2_ledgers), allow_launch_unlocked=False)
            validate_stage2_config(args.stage2_exec_config, require_ledger_files=bool(args.require_stage2_ledgers), allow_launch_unlocked=True)
    finally:
        if old_variant is None:
            os.environ.pop("C3_DETECTOR_AWARE_LEDGER_VARIANT", None)
        else:
            os.environ["C3_DETECTOR_AWARE_LEDGER_VARIANT"] = old_variant
        if old_root is None:
            os.environ.pop("C3_DETECTOR_AWARE_LEDGER_ROOT", None)
        else:
            os.environ["C3_DETECTOR_AWARE_LEDGER_ROOT"] = old_root

    return {
        "decision": STAGE2_READY,
        "stage": "Stage2 dense AdaTAD teacher utility -> detector-aware strict ledger",
        "teacher_evidence": teacher_evidence,
        "teacher_rows": teacher_rows,
        "policy_evidence": policy_evidence,
        "no_test_leakage": no_leakage,
        "validated_variants": list(STAGE2_VARIANTS),
        "full_run_gate": "PASS artifact required before PRECHECK_ONLY=0 runner",
    }


def _validate_stage3_proof(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    _require(payload.get("geometry_roundtrip_passed") is True, "TrueTime roundtrip proof missing")
    _require(payload.get("prediction_inverse_map_passed") is True, "prediction inverse-map proof missing")
    _require(payload.get("selected_input_st_gradient_passed") is True, "selector ST gradient proof missing")
    _require(float(payload.get("selected_input_selector_grad_norm", 0.0)) > 0.0, "selected input selector grad must be > 0")
    _require(payload.get("detector_loss_selector_grad_passed") is True, "detector loss must backprop to selector")
    _require(float(payload.get("detector_loss_selector_grad_norm", 0.0)) > 0.0, "detector selector grad must be > 0")
    _require(
        payload.get("actionformer_detector_loss_selector_grad_passed") is True,
        "ActionFormer detector loss must backprop to selector",
    )
    _require(
        float(payload.get("actionformer_detector_loss_selector_grad_norm", 0.0)) > 0.0,
        "ActionFormer detector selector grad must be > 0",
    )
    if payload.get("stage") == "stage3_true_time_e2e_adatad_selector_precheck":
        _require(
            payload.get("real_detector_proof_source") == "opentad_actionformer_forward_train_cost_backward",
            "wrong real detector proof source",
        )
        _require(payload.get("real_detector_loss_selector_grad_passed") is True, "real detector proof missing")
        _require(float(payload.get("real_detector_loss_selector_grad_norm", 0.0)) > 0.0, "real detector grad must be > 0")
        _require({"cls_loss", "reg_loss"}.issubset(set(payload.get("real_detector_loss_keys", []))), "real detector losses missing")
        _require(payload.get("actionformer_selected_axis_smoke") is False, "precheck proof must not be smoke-only")
        _require(payload.get("actionformer_physical_grid_precheck") is True, "physical-grid precheck flag missing")
    else:
        _require(payload.get("selector_grad_path") == "st_sparse_gather", "selector gradient path must be st_sparse_gather")
        _require({"loss_cls", "loss_reg"}.issubset(set(payload.get("loss_keys", []))), "registered detector losses missing")
    _require({"cls_loss", "reg_loss"}.issubset(set(payload.get("actionformer_loss_keys", []))), "ActionFormer losses missing")
    return {
        "proof_json": str(path),
        "selector_grad_norm": float(payload.get("selector_grad_norm", 0.0)),
        "detector_loss_selector_grad_norm": float(payload.get("detector_loss_selector_grad_norm", 0.0)),
        "actionformer_detector_loss_selector_grad_norm": float(payload.get("actionformer_detector_loss_selector_grad_norm", 0.0)),
    }


def validate_stage3(args: argparse.Namespace) -> dict[str, Any]:
    validate_stage3_config(args.stage3_config, require_grad_proof=False, allow_launch_unlocked=False)
    validate_stage3_config(
        args.stage3_exec_config,
        require_grad_proof=bool(args.require_stage3_grad_proof),
        allow_launch_unlocked=True,
        proof_json=args.stage3_grad_proof_json,
    )
    proof_report = None
    if args.stage3_grad_proof_json:
        proof_report = _validate_stage3_proof(args.stage3_grad_proof_json)
    elif args.require_stage3_grad_proof:
        raise AssertionError("Stage3 gradient/remap proof JSON is required")
    return {
        "decision": STAGE3_READY,
        "stage": "Stage3 TrueTime ST hard selector + real ActionFormer detector loss gradient precheck",
        "proof": proof_report,
        "truetime_remap_metadata": "validated in train/val/test Collect meta_keys",
        "full_run_gate": "PASS artifact required before PRECHECK_ONLY=0 runner",
    }


def _write_summary(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed DUCA Stage2/Stage3 precheck validator.")
    parser.add_argument("--stage", choices=("stage2", "stage3", "all"), default="all")
    parser.add_argument("--summary-json")
    parser.add_argument("--stage2-config", default="configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py")
    parser.add_argument("--stage2-exec-config", default="configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train_exec.py")
    parser.add_argument("--stage2-ledger-root")
    parser.add_argument("--require-stage2-ledgers", action="store_true")
    parser.add_argument("--require-stage2-teacher-evidence", action="store_true")
    parser.add_argument("--require-stage2-policy-evidence", action="store_true")
    parser.add_argument("--require-stage2-generator-manifest", action="store_true")
    parser.add_argument("--stage2-teacher-summary-json")
    parser.add_argument("--stage2-teacher-output-jsonl")
    parser.add_argument("--stage2-teacher-checkpoint-path")
    parser.add_argument("--stage2-teacher-config-path")
    parser.add_argument("--stage2-policy-summary-json")
    parser.add_argument("--stage2-policy-checkpoint-path")
    parser.add_argument("--stage2-val-source-jsonl")
    parser.add_argument("--stage2-test-source-jsonl")
    parser.add_argument("--stage3-config", default="configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck.py")
    parser.add_argument("--stage3-exec-config", default="configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck_exec.py")
    parser.add_argument("--require-stage3-grad-proof", action="store_true")
    parser.add_argument("--stage3-grad-proof-json")
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {
        "schema_version": "duca_stage23_precheck_v1",
        "decision": STAGE23_READY,
    }
    if args.stage in {"stage2", "all"}:
        payload["stage2"] = validate_stage2(args)
    if args.stage in {"stage3", "all"}:
        payload["stage3"] = validate_stage3(args)
    if args.stage == "stage2":
        payload["decision"] = STAGE2_READY
    elif args.stage == "stage3":
        payload["decision"] = STAGE3_READY
    if args.summary_json:
        _write_summary(args.summary_json, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
