from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from tools.bata.diagnose_duca_allocation_family_ceiling import read_input_records
from tools.bata.export_duca_allocation_ceiling_inputs import (
    data_directory_provenance,
    sha256,
    write_json_exclusive,
)
from tools.bata.validate_duca_allocation_ceiling_artifact import (
    validate_artifact_receipt,
)
from tools.bata.validate_duca_allocation_candidate_loss_artifact import (
    validate_candidate_artifact,
)
from tools.bata.validate_duca_allocation_solver_cost_artifact import (
    validate_solver_cost_artifact,
)


def finalize_gate(
    *,
    expected_commit: str,
    expected_checkpoint_epoch: int,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    pretrain: str | Path,
    expected_pretrain_sha256: str,
    suite_manifest_json: str | Path,
    suite_manifest_sha256: str,
    ceiling_validation_json: str | Path,
    gt_runtime_json: str | Path,
    max_projected_gt32_seconds: float,
    execution_cluster: str,
    input_jsonl: str | Path,
    ceiling_jsonl: str | Path,
    ceiling_summary_json: str | Path,
    candidate_jsonl: str | Path,
    candidate_summary_json: str | Path,
    solver_cost_samples_jsonl: str | Path,
    solver_cost_summary_json: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("expected_commit must be a full lowercase Git commit")
    if execution_cluster != "n16r4":
        raise ValueError("allocation real gate must execute on n16r4")
    checkpoint_path = Path(checkpoint).resolve()
    pretrain_path = Path(pretrain).resolve()
    manifest_path = Path(suite_manifest_json).resolve()
    validation_path = Path(ceiling_validation_json).resolve()
    runtime_path = Path(gt_runtime_json).resolve()
    input_path = Path(input_jsonl).resolve()
    ceiling_path = Path(ceiling_jsonl).resolve()
    ceiling_summary_path = Path(ceiling_summary_json).resolve()
    candidate_jsonl_path = Path(candidate_jsonl).resolve()
    candidate_path = Path(candidate_summary_json).resolve()
    cost_samples_path = Path(solver_cost_samples_jsonl).resolve()
    cost_path = Path(solver_cost_summary_json).resolve()
    output_path = Path(output_json).resolve()
    if output_path.exists():
        raise FileExistsError("allocation gate never overwrites artifacts")
    if sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("checkpoint bytes differ from the submitted binding")
    if sha256(pretrain_path) != expected_pretrain_sha256:
        raise ValueError("pretrain bytes differ from the submitted binding")
    manifest = _validate_suite_manifest(
        manifest_path,
        expected_sha256=suite_manifest_sha256,
        expected_commit=expected_commit,
        expected_checkpoint=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_epoch=expected_checkpoint_epoch,
        expected_pretrain=pretrain_path,
        expected_pretrain_sha256=expected_pretrain_sha256,
    )
    validation = validate_artifact_receipt(
        validation_json=validation_path,
        input_jsonl=input_path,
        output_jsonl=ceiling_path,
        summary_json=ceiling_summary_path,
        require_gt_solver_replay=True,
    )
    runtime = _validate_gt_runtime(
        runtime_path,
        max_projected_gt32_seconds=max_projected_gt32_seconds,
    )
    candidate = _load_mapping(candidate_path)
    candidate_validation = validate_candidate_artifact(
        ceiling_jsonl=ceiling_path,
        candidate_jsonl=candidate_jsonl_path,
        summary_json=candidate_path,
    )
    cost_validation = validate_solver_cost_artifact(
        input_jsonl=input_path,
        samples_jsonl=cost_samples_path,
        summary_json=cost_path,
    )
    cost = _load_mapping(cost_path)
    if candidate.get("schema_version") != "duca_allocation_candidate_detector_loss_summary_v1":
        raise ValueError("candidate detector summary schema mismatch")
    if int(candidate.get("sample_count", 0)) < 1:
        raise ValueError("candidate detector gate evaluated no samples")
    candidate_contract = candidate.get("contract")
    if not isinstance(candidate_contract, Mapping):
        raise ValueError("candidate detector contract is missing")
    required_candidate = {
        "model_training": False,
        "dense_axis_gt": True,
        "selected_axis_gt_remap": False,
        "physical_grid_actionformer": True,
        "mAP_evaluated": False,
        "paper_claim_allowed": False,
    }
    for key, value in required_candidate.items():
        if candidate_contract.get(key) is not value:
            raise ValueError(f"candidate detector contract mismatch: {key}")
    candidate_source = candidate.get("source")
    if not isinstance(candidate_source, Mapping):
        raise ValueError("candidate detector source is missing")
    if candidate_source.get("git_commit") != expected_commit:
        raise ValueError("candidate detector source commit mismatch")
    if candidate_source.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("candidate detector checkpoint binding mismatch")
    if candidate_source.get("checkpoint_state_key") != "state_dict_ema":
        raise ValueError("candidate detector did not use state_dict_ema")
    if int(candidate_source.get("checkpoint_epoch", -1)) != int(expected_checkpoint_epoch):
        raise ValueError("candidate detector checkpoint epoch mismatch")
    if candidate_source.get("backbone_pretrain_sha256") != expected_pretrain_sha256:
        raise ValueError("candidate detector pretrain binding mismatch")
    if (
        Path(str(candidate_source.get("ceiling_validation_json", ""))).resolve()
        != validation_path
        or candidate_source.get("ceiling_validation_json_sha256")
        != sha256(validation_path)
        or candidate_source.get("ceiling_validation") != validation
    ):
        raise ValueError("candidate detector validation-receipt binding mismatch")
    input_records = read_input_records(input_path)
    checkpoint_sha = sha256(checkpoint_path)
    reference_source = input_records[0]["source"]
    _revalidate_dataset_files(reference_source)
    manifest_source_bindings = {
        "annotation_path": manifest["annotation"],
        "annotation_sha256": manifest["annotation_sha256"],
        "class_map_path": manifest["class_map"],
        "class_map_sha256": manifest["class_map_sha256"],
        "data_path": manifest["train_data_path"],
        "config": manifest["training_config"],
        "config_sha256": manifest["training_config_sha256"],
    }
    for key, value in manifest_source_bindings.items():
        if reference_source.get(key) != value:
            raise ValueError(f"allocation source differs from suite manifest: {key}")
    for row in input_records:
        source = row["source"]
        if source.get("git_commit") != expected_commit:
            raise ValueError("allocation input source commit mismatch")
        if source.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("allocation input checkpoint binding mismatch")
        if source.get("checkpoint_state_key") != "state_dict_ema":
            raise ValueError("allocation input did not use state_dict_ema")
        if int(source.get("checkpoint_epoch", -1)) != int(expected_checkpoint_epoch):
            raise ValueError("allocation input checkpoint epoch mismatch")
        if source.get("split") != "train":
            raise ValueError("real allocation gate must consume training-side data only")
        for key in (
            "annotation_sha256",
            "class_map_sha256",
            "data_path",
            "config",
            "config_sha256",
            "dataset_data_manifest_sha256",
            "dataset_data_file_count",
            "dataset_data_total_bytes",
            "dataset_data_hash_algorithm",
            "dataset_subset_name",
            "dataset_test_mode",
            "dataset_filter_gt",
            "dataset_ioa_thresh",
            "dataset_feature_stride",
            "dataset_sample_stride",
            "dataset_window_size",
            "dataset_window_overlap_ratio",
            "dataset_offset_frames",
            "dataset_config_sha256",
            "dataset_window_manifest_sha256",
            "dataset_window_count",
            "dataset_window_deduplication",
            "dataset_duplicate_window_count_removed",
        ):
            if source.get(key) != reference_source.get(key):
                raise ValueError(f"allocation input dataset provenance drift: {key}")
    for key in (
        "annotation_sha256",
        "class_map_sha256",
        "data_path",
        "config",
        "config_sha256",
        "dataset_data_manifest_sha256",
        "dataset_data_file_count",
        "dataset_data_total_bytes",
        "dataset_data_hash_algorithm",
        "dataset_subset_name",
        "dataset_test_mode",
        "dataset_filter_gt",
        "dataset_ioa_thresh",
        "dataset_feature_stride",
        "dataset_sample_stride",
        "dataset_window_size",
        "dataset_window_overlap_ratio",
        "dataset_offset_frames",
        "dataset_config_sha256",
        "dataset_window_manifest_sha256",
        "dataset_window_count",
        "dataset_window_deduplication",
        "dataset_duplicate_window_count_removed",
    ):
        if candidate_source.get(key) != reference_source.get(key):
            raise ValueError(f"candidate dataset provenance mismatch: {key}")
    if cost.get("schema_version") != "duca_allocation_solver_cost_summary_v1":
        raise ValueError("solver cost summary schema mismatch")
    cost_contract = cost.get("contract")
    if not isinstance(cost_contract, Mapping):
        raise ValueError("solver cost contract is missing")
    if cost_contract.get("exact_decoder_only") is not True:
        raise ValueError("solver cost did not profile the exact decoder")
    if cost_contract.get("full_stack_claim") is not False:
        raise ValueError("solver-only cost cannot claim full-stack savings")
    result = {
        "schema_version": "duca_allocation_ceiling_real_gate_v1",
        "gate_passed": True,
        "git_commit": expected_commit,
        "execution_cluster": execution_cluster,
        "checkpoint_epoch": int(expected_checkpoint_epoch),
        "checkpoint_state_key": "state_dict_ema",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "pretrain": str(pretrain_path),
        "pretrain_sha256": sha256(pretrain_path),
        "suite_manifest_json": str(manifest_path),
        "suite_manifest_json_sha256": sha256(manifest_path),
        "suite_manifest": manifest,
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256(input_path),
        "ceiling_jsonl": str(ceiling_path),
        "ceiling_jsonl_sha256": sha256(ceiling_path),
        "ceiling_summary_json": str(ceiling_summary_path),
        "ceiling_summary_json_sha256": sha256(ceiling_summary_path),
        "ceiling_validation_json": str(validation_path),
        "ceiling_validation_json_sha256": sha256(validation_path),
        "gt_runtime_json": str(runtime_path),
        "gt_runtime_json_sha256": sha256(runtime_path),
        "gt_runtime": runtime,
        "candidate_summary_json": str(candidate_path),
        "candidate_summary_json_sha256": sha256(candidate_path),
        "candidate_jsonl": str(candidate_jsonl_path),
        "candidate_jsonl_sha256": sha256(candidate_jsonl_path),
        "candidate_validation": candidate_validation,
        "solver_cost_samples_jsonl": str(cost_samples_path),
        "solver_cost_samples_jsonl_sha256": sha256(cost_samples_path),
        "solver_cost_summary_json": str(cost_path),
        "solver_cost_summary_json_sha256": sha256(cost_path),
        "solver_cost_validation": cost_validation,
        "artifact_validation": validation,
        "contract": {
            "offline_full_window": True,
            "model_training": False,
            "actual_decoded_coordinates": True,
            "exact_solver_optimal": True,
            "physical_grid_dense_axis_gt": True,
            "selected_axis_gt_remap": False,
            "paper_claim_allowed": False,
            "authorizes_training_side_ceiling_suite": True,
            "authorizes_selector_training": False,
        },
    }
    write_json_exclusive(output_path, result)
    return result


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected an object")
    return payload


def _validate_suite_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_commit: str,
    expected_checkpoint: Path,
    expected_checkpoint_sha256: str,
    expected_checkpoint_epoch: int,
    expected_pretrain: Path,
    expected_pretrain_sha256: str,
) -> dict[str, Any]:
    if sha256(path) != expected_sha256:
        raise ValueError("suite manifest SHA-256 mismatch")
    payload = _load_mapping(path)
    required = {
        "schema_version",
        "git_commit",
        "task",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "checkpoint_state_key",
        "pretrain",
        "pretrain_sha256",
        "annotation",
        "annotation_sha256",
        "class_map",
        "class_map_sha256",
        "train_data_path",
        "training_config",
        "training_config_sha256",
        "target_cluster",
        "validation_subset_consumed",
        "selector_training_authorized",
    }
    if set(payload) != required:
        raise ValueError("strict suite manifest fields mismatch")
    base = expected_pretrain.parents[1]
    annotation = (base / "thumos14/annotations/thumos_14_anno.json").resolve()
    class_map = (base / "thumos14/annotations/category_idx.txt").resolve()
    train_data = (base / "thumos14/train").resolve()
    repo_root = Path(__file__).resolve().parents[2]
    training_config = (
        repo_root
        / "configs/adatad/thumos/duca_allocation_ceiling_training_windows.py"
    ).resolve()
    if not annotation.is_file() or not class_map.is_file() or not train_data.is_dir():
        raise ValueError("canonical THUMOS14 data binding is missing")
    if not training_config.is_file():
        raise ValueError("canonical allocation training config is missing")
    expected = {
        "schema_version": "duca_allocation_training_suite_manifest_v1",
        "git_commit": expected_commit,
        "task": "offline_temporal_action_detection",
        "checkpoint": str(expected_checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_epoch": int(expected_checkpoint_epoch),
        "checkpoint_state_key": "state_dict_ema",
        "pretrain": str(expected_pretrain),
        "pretrain_sha256": expected_pretrain_sha256,
        "annotation": str(annotation),
        "annotation_sha256": sha256(annotation),
        "class_map": str(class_map),
        "class_map_sha256": sha256(class_map),
        "train_data_path": str(train_data),
        "training_config": str(training_config),
        "training_config_sha256": sha256(training_config),
        "target_cluster": "n16r4",
        "validation_subset_consumed": False,
        "selector_training_authorized": False,
    }
    if dict(payload) != expected:
        raise ValueError("suite manifest binding mismatch")
    return dict(payload)


def _validate_gt_runtime(
    path: Path,
    *,
    max_projected_gt32_seconds: float,
) -> dict[str, Any]:
    payload = _load_mapping(path)
    required = {
        "schema_version",
        "gt_generation_seconds",
        "gt_validation_seconds",
        "projected_gt32_seconds",
        "max_projected_gt32_seconds",
    }
    if set(payload) != required:
        raise ValueError("strict GT runtime fields mismatch")
    if payload.get("schema_version") != "duca_allocation_gt_runtime_projection_v1":
        raise ValueError("GT runtime schema mismatch")
    generation = float(payload["gt_generation_seconds"])
    validation = float(payload["gt_validation_seconds"])
    projected = float(payload["projected_gt32_seconds"])
    maximum = float(payload["max_projected_gt32_seconds"])
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (generation, validation, projected, maximum)
    ):
        raise ValueError("GT runtime values must be finite and positive")
    expected_projected = (generation + validation) * 32.0
    if not math.isclose(projected, expected_projected, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("GT runtime projection mismatch")
    if not math.isclose(
        maximum,
        float(max_projected_gt32_seconds),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError("GT runtime threshold differs from the submitted contract")
    if projected > maximum:
        raise ValueError(
            "projected 32-sample GT solver cost exceeds the fail-closed threshold"
        )
    return dict(payload)


def _revalidate_dataset_files(source: Mapping[str, Any]) -> None:
    for path_key, hash_key in (
        ("annotation_path", "annotation_sha256"),
        ("class_map_path", "class_map_sha256"),
        ("config", "config_sha256"),
    ):
        path = Path(str(source.get(path_key, ""))).resolve()
        if not path.is_file() or sha256(path) != source.get(hash_key):
            raise ValueError(f"dataset file provenance mismatch: {path_key}")
    data_path = Path(str(source.get("data_path", ""))).resolve()
    current = data_directory_provenance(data_path)
    for key, value in current.items():
        if source.get(key) != value:
            raise ValueError(f"dataset data provenance mismatch: {key}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize the real DUCA allocation-ceiling gate.")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-checkpoint-epoch", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--expected-pretrain-sha256", required=True)
    parser.add_argument("--suite-manifest-json", required=True)
    parser.add_argument("--suite-manifest-sha256", required=True)
    parser.add_argument("--ceiling-validation-json", required=True)
    parser.add_argument("--gt-runtime-json", required=True)
    parser.add_argument(
        "--max-projected-gt32-seconds",
        type=float,
        required=True,
    )
    parser.add_argument("--execution-cluster", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--ceiling-jsonl", required=True)
    parser.add_argument("--ceiling-summary-json", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--candidate-summary-json", required=True)
    parser.add_argument("--solver-cost-samples-jsonl", required=True)
    parser.add_argument("--solver-cost-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    result = finalize_gate(
        expected_commit=args.expected_commit,
        expected_checkpoint_epoch=args.expected_checkpoint_epoch,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        pretrain=args.pretrain,
        expected_pretrain_sha256=args.expected_pretrain_sha256,
        suite_manifest_json=args.suite_manifest_json,
        suite_manifest_sha256=args.suite_manifest_sha256,
        ceiling_validation_json=args.ceiling_validation_json,
        gt_runtime_json=args.gt_runtime_json,
        max_projected_gt32_seconds=args.max_projected_gt32_seconds,
        execution_cluster=args.execution_cluster,
        input_jsonl=args.input_jsonl,
        ceiling_jsonl=args.ceiling_jsonl,
        ceiling_summary_json=args.ceiling_summary_json,
        candidate_jsonl=args.candidate_jsonl,
        candidate_summary_json=args.candidate_summary_json,
        solver_cost_samples_jsonl=args.solver_cost_samples_jsonl,
        solver_cost_summary_json=args.solver_cost_summary_json,
        output_json=args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
