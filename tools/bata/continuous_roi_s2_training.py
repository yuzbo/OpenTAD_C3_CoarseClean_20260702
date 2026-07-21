from __future__ import annotations

import ast
import copy
import importlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_contract import (  # noqa: E402
    canonical_sha256,
    load_protocol,
    validate_protocol,
)
from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    sha256_file,
    validate_s1_manifest,
)

S2_TRAINING_BINDING_SCHEMA = "continuous_roi_s2_training_binding_v1"
S2_CHECKPOINT_METADATA_SCHEMA = "continuous_roi_s2_checkpoint_metadata_v1"
S2_CHECKPOINT_SIDECAR_SCHEMA = "continuous_roi_s2_checkpoint_sidecar_v1"
S2_TRAINING_COMPLETION_SCHEMA = "continuous_roi_s2_training_completion_v1"
S2_EXPERIMENT_NAMESPACE_SCHEMA = "continuous_roi_s2_experiment_namespace_v1"
S2_FULL_MODEL_GATE_SCHEMA = "continuous_roi_s2_full_model_one_step_cuda_gate_v3"
S2_TRAINING_RUNTIME_PRECHECK_SCHEMA = "continuous_roi_s2_training_runtime_precheck_v2"
S2_FAMILIES = ("D160", "G96", "U128")
S2_TRAINING_SEEDS = (3407, 3408, 3409)
S2_SUCCESSFUL_UPDATES = 4800
S2_UPDATES_PER_EPOCH = 80
S2_EPOCHS = 60
S2_CANONICAL_EXPERIMENTS_ROOT = (
    "/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/"
    "continuous_roi_s2_canonical"
)
S2_SOURCE_CONFIGS = {
    "D160": Path(
        "configs/adatad/thumos/" "continuous_roi_s2_d160_videomae_s_768x1_adapter.py"
    ),
    "G96": Path(
        "configs/adatad/thumos/" "continuous_roi_s2_g96_videomae_s_768x1_adapter.py"
    ),
    "U128": Path(
        "configs/adatad/thumos/" "continuous_roi_s2_u128_videomae_s_768x1_adapter.py"
    ),
}


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _git_output(repository_root: str | Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=Path(repository_root).resolve(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def current_git_commit(repository_root: str | Path = ROOT) -> str:
    commit = _git_output(repository_root, "rev-parse", "HEAD").lower()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("formal Continuous-RoI S2 requires a full Git commit")
    return commit


def require_clean_git_checkout(
    *, expected_commit: str, repository_root: str | Path = ROOT
) -> None:
    repository_root = Path(repository_root).resolve()
    if current_git_commit(repository_root) != str(expected_commit).lower():
        raise RuntimeError(
            "formal Continuous-RoI S2 checkout differs from the bound commit"
        )
    if _git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ):
        raise RuntimeError(
            "formal Continuous-RoI S2 execution requires a clean Git checkout"
        )


def _decode_pure_config_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_decode_pure_config_value(value) for value in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.Dict):
        decoded = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise ValueError(
                    "Continuous-RoI S2 config forbids dictionary unpacking"
                )
            decoded[_decode_pure_config_value(key)] = _decode_pure_config_value(value)
        return decoded
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _decode_pure_config_value(node.operand)
        if not isinstance(operand, (int, float, complex)):
            raise ValueError("Continuous-RoI S2 unary config value is not numeric")
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
        and all(keyword.arg is not None for keyword in node.keywords)
    ):
        return {
            keyword.arg: _decode_pure_config_value(keyword.value)
            for keyword in node.keywords
        }
    raise ValueError("Continuous-RoI S2 bound config is not a pure data assignment")


def load_pure_data_config(path: str | Path) -> Config:
    """Decode a materialized config without executing it as Python."""

    path = Path(path).resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path), mode="exec")
    config_data = {}
    for statement in tree.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            raise ValueError(
                "Continuous-RoI S2 bound config contains executable syntax"
            )
        target = statement.targets[0].id
        if (
            target.startswith("_")
            or target == "custom_imports"
            or target in config_data
        ):
            raise ValueError(
                "Continuous-RoI S2 bound config contains a reserved or duplicate key"
            )
        config_data[target] = _decode_pure_config_value(statement.value)
    return Config(config_data, filename=str(path))


def validate_full_model_gate(
    gate_path: str | Path, *, expected_commit: str
) -> dict[str, Any]:
    gate_path = Path(gate_path).resolve()
    gate = _load_json(gate_path)
    gate_hash = gate.pop("gate_sha256", None)
    if not gate_hash or canonical_sha256(gate) != gate_hash:
        raise ValueError("Continuous-RoI S2 full-model Gate self-hash mismatch")
    gate["gate_sha256"] = gate_hash
    if gate.get("schema_version") != S2_FULL_MODEL_GATE_SCHEMA:
        raise ValueError("unsupported Continuous-RoI S2 full-model Gate schema")
    if gate.get("status") != "PASS":
        raise ValueError("Continuous-RoI S2 full-model Gate did not pass")
    expected_commit = str(expected_commit).lower()
    if (
        gate.get("expected_commit") != expected_commit
        or gate.get("code_provenance", {}).get("git_commit") != expected_commit
        or gate.get("code_provenance", {}).get("complete_worktree_clean") is not True
    ):
        raise ValueError("Continuous-RoI S2 Gate commit provenance mismatch")
    protocol_audit = gate.get("protocol_audit", {})
    implementation_audit = gate.get("implementation_audit", {})
    if (
        protocol_audit.get("static_protocol_valid") is not True
        or protocol_audit.get("official_test_open_allowed") is not False
        or implementation_audit.get("status") != "PASS"
        or implementation_audit.get("official_test_materialized") is not False
        or gate.get("optimizer_step_completed") is not True
        or gate.get("training_external_geometry_rejected") is not True
        or gate.get("official_test_annotation_records_loaded") != 0
        or gate.get("official_test_video_files_opened") != 0
        or gate.get("training_runtime_gate_authorized_by_this_gate") is not True
        or gate.get("formal_training_authorized_by_this_gate") is not False
    ):
        raise ValueError("Continuous-RoI S2 Gate lacks a required PASS invariant")
    if gate.get("projection_input_shape") != [1, 384, 768]:
        raise ValueError("Continuous-RoI S2 Gate detector feature contract changed")
    if (
        gate.get("checkpoint_sha256")
        != load_protocol()["data"]["videomae_s_checkpoint_sha256"]
    ):
        raise ValueError("Continuous-RoI S2 Gate checkpoint identity changed")
    for audit_name in ("detector_only_gradient_audit", "total_gradient_audit"):
        audit = gate.get(audit_name, {})
        if (
            audit.get("missing_target_gradients") != []
            or audit.get("all_present_gradients_finite") is not True
        ):
            raise ValueError(f"Continuous-RoI S2 {audit_name} is incomplete")
    from tools.bata.run_continuous_roi_s2_one_step_gate import (
        AUDITED_SOURCE_PATHS,
    )

    source_hashes = gate.get("code_provenance", {}).get("audited_source_sha256")
    optimizer_audit = gate.get("optimizer_audit", {})
    cuda_audit = gate.get("cuda_device_identity", {})
    backbone_audit = gate.get("backbone_audit", {})
    sampler_audit = gate.get("sampler_geometry_gradient_audit", {})
    protocol_copy = dict(protocol_audit)
    protocol_hash = protocol_copy.pop("audit_sha256", None)
    implementation_copy = dict(implementation_audit)
    implementation_hash = implementation_copy.pop("implementation_audit_sha256", None)
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(AUDITED_SOURCE_PATHS)
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in source_hashes.values()
        )
        or any(
            sha256_file(ROOT / relative_path) != source_hashes[relative_path]
            for relative_path in AUDITED_SOURCE_PATHS
        )
        or not protocol_hash
        or canonical_sha256(protocol_copy) != protocol_hash
        or protocol_audit.get("protocol_sha256")
        != load_protocol()["declared_protocol_sha256"]
        or protocol_audit.get("check_count", 0) <= 0
        or protocol_audit.get("state_assignments_checked") != 128
        or protocol_audit.get("training_authorized") is not False
        or not implementation_hash
        or canonical_sha256(implementation_copy) != implementation_hash
        or implementation_audit.get("protocol_sha256")
        != load_protocol()["declared_protocol_sha256"]
        or set(implementation_audit.get("config_hashes", {})) != set(S2_FAMILIES)
        or set(implementation_audit.get("pipeline_audits", {})) != set(S2_FAMILIES)
        or implementation_audit.get("detector_model_surface_matches_reference")
        is not True
        or implementation_audit.get("post_processing_matches_reference") is not True
        or implementation_audit.get("u128_selector_parameters") != 0
        or implementation_audit.get("u128_new_parameters") != 609449
        or implementation_audit.get("training_authorized") is not False
        or implementation_audit.get("full_model_cuda_gate_required") is not True
        or optimizer_audit.get("every_requires_grad_parameter_exactly_once") is not True
        or optimizer_audit.get("frozen_parameters_excluded") is not True
        or int(optimizer_audit.get("requires_grad_parameter_tensors", 0)) <= 0
        or optimizer_audit.get("requires_grad_parameter_tensors")
        != optimizer_audit.get("optimizer_parameter_tensors")
        or cuda_audit.get("logical_device") != "cuda:0"
        or not cuda_audit.get("slurm_job_id")
        or not cuda_audit.get("slurm_step_id")
        or not cuda_audit.get("cuda_visible_device_uuid")
        or backbone_audit.get("shared_backbone_instances") != 1
        or backbone_audit.get("videomae_evaluations") != 2
        or backbone_audit.get("contains_selector") is not False
        or sampler_audit.get("runtime_bitwise_parity") is not True
    ):
        raise ValueError("Continuous-RoI S2 full-model Gate evidence is incomplete")
    return gate


def validate_training_runtime_precheck(
    precheck_path: str | Path,
    *,
    expected_commit: str,
    expected_full_model_gate_sha256: str,
) -> dict[str, Any]:
    precheck = _load_json(precheck_path)
    precheck_hash = precheck.pop("precheck_sha256", None)
    if not precheck_hash or canonical_sha256(precheck) != precheck_hash:
        raise ValueError("Continuous-RoI S2 training precheck self-hash mismatch")
    precheck["precheck_sha256"] = precheck_hash
    bindings = precheck.get("bindings")
    family_runtime = precheck.get("family_runtime")
    expected_cells = {
        (family, seed) for family in S2_FAMILIES for seed in S2_TRAINING_SEEDS
    }
    actual_cells = (
        {
            (str(cell.get("family", "")).upper(), int(cell.get("seed", -1)))
            for cell in bindings
            if isinstance(cell, Mapping)
        }
        if isinstance(bindings, list)
        else set()
    )
    visible = [
        value.strip()
        for value in str(precheck.get("cuda_visible_devices") or "").split(",")
        if value.strip()
    ]
    binding_invariants = isinstance(bindings, list) and all(
        isinstance(cell, Mapping)
        and cell.get("config_dump_reload_valid") is True
        and int(cell.get("successful_updates", -1)) == S2_SUCCESSFUL_UPDATES
        and int(cell.get("updates_per_epoch", -1)) == S2_UPDATES_PER_EPOCH
        and cell.get("checkpoint_selection") == "final_ema_only"
        and isinstance(cell.get("bound_config_sha256"), str)
        and len(cell["bound_config_sha256"]) == 64
        and bool(cell.get("work_dir"))
        for cell in bindings
    )
    expected_sample_policies = {
        "D160": ("full_frame_letterbox", [2, 1, 3, 768, 160, 160]),
        "G96": ("full_frame_letterbox", [2, 1, 3, 768, 96, 96]),
        "U128": ("none_pre_policy_source", None),
    }
    family_invariants = isinstance(family_runtime, Mapping)
    if family_invariants:
        for family, (policy, dense_shape) in expected_sample_policies.items():
            audit = family_runtime.get(family, {})
            sample = audit.get("sample_audit", {})
            real_batch = audit.get("real_training_batch_audit", {})
            if (
                audit.get("fit_video_count") != 160
                or audit.get("fit_sample_count") != 160
                or audit.get("development_gate_video_count") != 40
                or audit.get("development_gate_window_count") != 129
                or audit.get("train_batches_per_epoch") != S2_UPDATES_PER_EPOCH
                or not isinstance(audit.get("gate_window_identity_sha256"), str)
                or len(audit["gate_window_identity_sha256"]) != 64
                or sample.get("geometry_policy") != policy
                or sample.get("uses_gt") is not False
                or sample.get("uses_teacher") is not False
                or sample.get("uses_oracle") is not False
                or sample.get("uses_test_evidence") is not False
                or real_batch.get("batch_size") != 2
                or real_batch.get("mask_shape") != [2, 768]
                or real_batch.get("uses_gt_for_geometry") is not False
                or real_batch.get("uses_teacher") is not False
                or real_batch.get("uses_oracle") is not False
                or real_batch.get("uses_test_evidence") is not False
            ):
                family_invariants = False
                break
            if (
                dense_shape is not None
                and real_batch.get("input_shapes", {}).get("dense") != dense_shape
            ):
                family_invariants = False
                break
            if family == "U128" and real_batch.get("input_shapes") != {
                "global": [2, 1, 3, 768, 96, 96],
                "source": [2, 1, 3, 768, 180, 320],
                "sample_key": [2],
                "window_start": [2],
            }:
                family_invariants = False
                break
    inventory = precheck.get("development_video_inventory", {})
    census = precheck.get("development_video_census", {})
    if (
        precheck.get("schema_version") != S2_TRAINING_RUNTIME_PRECHECK_SCHEMA
        or precheck.get("status") != "PASS"
        or precheck.get("code_commit") != str(expected_commit).lower()
        or precheck.get("full_model_gate_sha256")
        != str(expected_full_model_gate_sha256)
        or precheck.get("all_nine_bindings_valid") is not True
        or precheck.get("all_nine_config_dump_reload_valid") is not True
        or precheck.get("train_batches_per_epoch") != S2_UPDATES_PER_EPOCH
        or precheck.get("development_gate_window_count") != 129
        or not isinstance(precheck.get("development_gate_window_identity_sha256"), str)
        or len(precheck["development_gate_window_identity_sha256"]) != 64
        or precheck.get("official_test_annotation_records_loaded") != 0
        or precheck.get("official_test_video_files_opened") != 0
        or precheck.get("official_test_open_allowed") is not False
        or precheck.get("learned_roi_policy_present") is not False
        or precheck.get("paper_claim_allowed") is not False
        or not isinstance(precheck.get("slurm_job_id"), str)
        or not precheck["slurm_job_id"]
        or not isinstance(precheck.get("slurm_step_id"), str)
        or not precheck["slurm_step_id"]
        or not isinstance(precheck.get("slurm_step_gpu_identity"), str)
        or not precheck["slurm_step_gpu_identity"]
        or precheck.get("slurm_cpus_per_task") != 5
        or int(precheck.get("effective_memory_limit_mb", 0)) < 90000
        or len(visible) != 1
        or actual_cells != expected_cells
        or len(bindings) != len(expected_cells)
        or not binding_invariants
        or not isinstance(family_runtime, Mapping)
        or set(family_runtime) != set(S2_FAMILIES)
        or not family_invariants
        or inventory.get("video_count") != 200
        or inventory.get("symlinks_allowed") is not False
        or inventory.get("path_escape_allowed") is not False
        or not isinstance(inventory.get("inventory_sha256"), str)
        or len(inventory["inventory_sha256"]) != 64
        or census.get("video_count") != 200
        or census.get("all_videos_ffprobe_decodable") is not True
        or census.get("sealed_test_files_probed") != 0
        or census.get("annotation_or_gt_read") is not False
        or not isinstance(census.get("records_sha256"), str)
        or len(census["records_sha256"]) != 64
    ):
        raise ValueError(
            "Continuous-RoI S2 training runtime precheck lacks a PASS invariant"
        )
    return precheck


def audit_development_video_inventory(
    video_root: str | Path, video_ids: list[str]
) -> dict[str, Any]:
    raw_root = Path(video_root)
    if raw_root.is_symlink():
        raise ValueError("Continuous-RoI S2 development root cannot be a symlink")
    video_root = raw_root.resolve()
    root_text = str(video_root).replace("\\", "/").lower()
    if any(
        token in root_text
        for token in ("/test data/", "/th14_test_set_mp4", "/sealed_test")
    ):
        raise ValueError(
            "Continuous-RoI S2 refuses a sealed-test development video root"
        )
    if not video_root.is_dir():
        raise FileNotFoundError(video_root)
    inventory = []
    for video_id in sorted(map(str, video_ids)):
        raw_path = video_root / f"{video_id}.mp4"
        if raw_path.is_symlink():
            raise ValueError(
                f"Continuous-RoI S2 development video cannot be a symlink: {raw_path}"
            )
        path = raw_path.resolve()
        try:
            relative_path = path.relative_to(video_root)
        except ValueError as exc:
            raise ValueError(
                f"development video escapes its bound root: {path}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        size_bytes = int(path.stat().st_size)
        if size_bytes <= 0:
            raise ValueError(f"empty development video: {path}")
        inventory.append(
            {
                "video_id": video_id,
                "relative_path": relative_path.as_posix(),
                "size_bytes": size_bytes,
            }
        )
    return {
        "video_root": str(video_root),
        "video_count": len(inventory),
        "inventory_sha256": canonical_sha256(inventory),
        "symlinks_allowed": False,
        "path_escape_allowed": False,
    }


def audit_development_video_census(
    video_root: str | Path, video_ids: list[str]
) -> dict[str, Any]:
    from tools.bata.native_crop_s1_geometry_census import probe_video_geometry

    inventory = audit_development_video_inventory(video_root, video_ids)
    root = Path(inventory["video_root"])
    records = []
    for video_id in sorted(map(str, video_ids)):
        path = root / f"{video_id}.mp4"
        records.append(
            {
                "video_id": video_id,
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                **probe_video_geometry(path),
            }
        )
    return {
        "video_root": str(root),
        "video_count": len(records),
        "records_sha256": canonical_sha256(records),
        "all_videos_ffprobe_decodable": True,
        "sealed_test_files_probed": 0,
        "annotation_or_gt_read": False,
    }


def validate_external_training_inputs(
    *,
    manifest_path: str | Path,
    development_annotation_path: str | Path,
    class_map_path: str | Path,
    pretrained_checkpoint_path: str | Path,
) -> dict[str, Any]:
    protocol = load_protocol()
    validate_protocol(protocol)
    expected = protocol["data"]
    manifest_path = Path(manifest_path).resolve()
    development_annotation_path = Path(development_annotation_path).resolve()
    class_map_path = Path(class_map_path).resolve()
    pretrained_checkpoint_path = Path(pretrained_checkpoint_path).resolve()
    file_hashes = {
        "base_manifest_file_sha256": sha256_file(manifest_path),
        "development_annotation_sha256": sha256_file(development_annotation_path),
        "class_map_sha256": sha256_file(class_map_path),
        "videomae_s_checkpoint_sha256": sha256_file(pretrained_checkpoint_path),
    }
    for key, actual in file_hashes.items():
        if actual != expected[key]:
            raise ValueError(f"Continuous-RoI S2 external input hash mismatch: {key}")

    manifest = validate_s1_manifest(_load_json(manifest_path))
    if (
        manifest["manifest_sha256"] != expected["base_manifest_semantic_sha256"]
        or manifest["split_hashes"]["fit"] != expected["fit_split_sha256"]
        or manifest["split_hashes"]["gate"] != expected["gate_split_sha256"]
        or manifest["split_hashes"]["test"] != expected["sealed_test_split_sha256"]
        or len(manifest["splits"]["fit"]) != int(expected["fit_count"])
        or len(manifest["splits"]["gate"]) != int(expected["gate_count"])
    ):
        raise ValueError("Continuous-RoI S2 base manifest differs from the protocol")

    development_annotation = _load_json(development_annotation_path)
    database = development_annotation.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("development annotation has no database mapping")
    development_ids = set(map(str, database))
    expected_development = set(manifest["splits"]["fit"]) | set(
        manifest["splits"]["gate"]
    )
    if development_ids != expected_development:
        raise ValueError(
            "development-only annotation does not match the frozen fit/gate union"
        )
    if any(
        str(video.get("subset")) != "training"
        for video in database.values()
        if isinstance(video, Mapping)
    ):
        raise ValueError("development-only annotation contains a non-training video")
    if set(manifest["splits"]["test"]) & development_ids:
        raise ValueError("sealed test IDs leaked into the development annotation")
    return {
        "protocol": protocol,
        "manifest": manifest,
        "file_hashes": file_hashes,
        "development_video_count": len(development_ids),
        "official_test_annotation_records_loaded": 0,
        "official_test_video_files_opened": 0,
    }


def build_experiment_identity(
    *,
    code_commit: str,
    gate_path: str | Path,
    gate: Mapping[str, Any],
    external_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = external_inputs["protocol"]
    manifest = external_inputs["manifest"]
    payload = {
        "schema_version": S2_EXPERIMENT_NAMESPACE_SCHEMA,
        "code_commit": str(code_commit).lower(),
        "protocol_sha256": protocol["declared_protocol_sha256"],
        "base_manifest_file_sha256": external_inputs["file_hashes"][
            "base_manifest_file_sha256"
        ],
        "base_manifest_semantic_sha256": manifest["manifest_sha256"],
        "fit_split_sha256": manifest["split_hashes"]["fit"],
        "gate_split_sha256": manifest["split_hashes"]["gate"],
        "development_annotation_sha256": external_inputs["file_hashes"][
            "development_annotation_sha256"
        ],
        "class_map_sha256": external_inputs["file_hashes"]["class_map_sha256"],
        "pretrained_checkpoint_sha256": external_inputs["file_hashes"][
            "videomae_s_checkpoint_sha256"
        ],
        "full_model_gate_file_sha256": sha256_file(gate_path),
        "full_model_gate_sha256": gate["gate_sha256"],
    }
    namespace = canonical_sha256(payload)
    return {
        **payload,
        "experiment_namespace": namespace,
        "canonical_experiment_root": f"{S2_CANONICAL_EXPERIMENTS_ROOT}/{namespace}",
    }


def _source_config_path(family: str, *, repository_root: str | Path = ROOT) -> Path:
    family = str(family).upper()
    try:
        relative = S2_SOURCE_CONFIGS[family]
    except KeyError as exc:
        raise ValueError(f"unsupported Continuous-RoI S2 family {family}") from exc
    return (Path(repository_root).resolve() / relative).resolve()


def _bind_deterministic_temporal_upsampling(
    cfg: Config, *, family: str
) -> dict[str, Any]:
    if family == "U128":
        return {
            "implementation": "continuous_roi_wrapper_explicit_linear_2x",
            "input_length": 384,
            "output_length": 768,
            "align_corners": False,
        }
    pipeline = cfg.model.backbone.custom.post_processing_pipeline
    transforms = [
        transform for transform in pipeline if transform["type"] == "Interpolate"
    ]
    if (
        len(transforms) != 1
        or int(transforms[0]["size"]) != 768
        or str(transforms[0].get("mode", "linear")) != "linear"
    ):
        raise ValueError(
            "Continuous-RoI S2 dense comparator temporal interpolation changed"
        )
    transform = transforms[0]
    transform["mode"] = "linear"
    transform["deterministic"] = True
    transform["expected_input_size"] = 384
    return {
        "implementation": "explicit_linear_2x_no_cuda_atomics",
        "input_length": 384,
        "output_length": 768,
        "align_corners": False,
    }


def bind_training_config(
    *,
    source_config_path: str | Path,
    family: str,
    seed: int,
    work_dir: str | Path,
    manifest_path: str | Path,
    development_annotation_path: str | Path,
    class_map_path: str | Path,
    development_video_root: str | Path,
    pretrained_checkpoint_path: str | Path,
    full_model_gate_path: str | Path,
    training_runtime_precheck_path: str | Path | None = None,
    runtime_authorization_path: str | Path | None = None,
    repository_root: str | Path = ROOT,
    code_commit: str | None = None,
    require_runtime_precheck: bool = True,
    require_runtime_authorization: bool = True,
) -> Config:
    repository_root = Path(repository_root).resolve()
    family = str(family).upper()
    seed = int(seed)
    if family not in S2_FAMILIES or seed not in S2_TRAINING_SEEDS:
        raise ValueError("Continuous-RoI S2 family or seed is outside the protocol")
    source_config_path = Path(source_config_path).resolve()
    if source_config_path != _source_config_path(
        family, repository_root=repository_root
    ):
        raise ValueError("Continuous-RoI S2 must bind one canonical source config")
    code_commit = (
        current_git_commit(repository_root)
        if code_commit is None
        else str(code_commit).lower()
    )
    require_clean_git_checkout(
        expected_commit=code_commit, repository_root=repository_root
    )
    gate = validate_full_model_gate(full_model_gate_path, expected_commit=code_commit)
    external = validate_external_training_inputs(
        manifest_path=manifest_path,
        development_annotation_path=development_annotation_path,
        class_map_path=class_map_path,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
    )
    identity = build_experiment_identity(
        code_commit=code_commit,
        gate_path=full_model_gate_path,
        gate=gate,
        external_inputs=external,
    )
    runtime_precheck = None
    if training_runtime_precheck_path is not None:
        runtime_precheck = validate_training_runtime_precheck(
            training_runtime_precheck_path,
            expected_commit=code_commit,
            expected_full_model_gate_sha256=gate["gate_sha256"],
        )
    elif require_runtime_precheck:
        raise ValueError("formal Continuous-RoI S2 binding requires a runtime precheck")
    runtime_authorization = None
    if runtime_authorization_path is not None:
        if runtime_precheck is None:
            raise ValueError(
                "runtime authorization requires a validated runtime precheck"
            )
        from tools.bata.continuous_roi_s2_runtime_gate import (
            validate_runtime_authorization,
        )

        runtime_authorization = validate_runtime_authorization(
            runtime_authorization_path,
            expected_commit=code_commit,
            expected_full_model_gate_sha256=gate["gate_sha256"],
            expected_precheck_sha256=runtime_precheck["precheck_sha256"],
        )
        if (
            runtime_authorization.get("base_experiment_namespace")
            != identity["experiment_namespace"]
        ):
            raise ValueError("runtime authorization belongs to another base experiment")
    elif require_runtime_authorization:
        raise ValueError(
            "formal Continuous-RoI S2 binding requires a runtime authorization"
        )
    work_dir = Path(work_dir).resolve()
    canonical_experiment_root = (
        runtime_authorization["canonical_experiment_root"]
        if runtime_authorization is not None
        else identity["canonical_experiment_root"]
    )
    experiment_namespace = (
        runtime_authorization["campaign_namespace"]
        if runtime_authorization is not None
        else identity["experiment_namespace"]
    )
    expected_work_dir = (
        Path(canonical_experiment_root) / family.lower() / f"seed{seed}"
    ).resolve()
    if work_dir != expected_work_dir:
        raise ValueError(
            "formal Continuous-RoI S2 work_dir must equal " f"{expected_work_dir}"
        )
    development_video_root = Path(development_video_root).resolve()
    development_inventory = audit_development_video_inventory(
        development_video_root,
        list(external["manifest"]["splits"]["fit"])
        + list(external["manifest"]["splits"]["gate"]),
    )
    if development_inventory["video_count"] != 200:
        raise RuntimeError("Continuous-RoI S2 development inventory is not 200 videos")
    if runtime_precheck is not None:
        expected_binding_cells = {
            (
                str(cell["family"]).upper(),
                int(cell["seed"]),
                str(Path(cell["work_dir"]).resolve()),
            )
            for cell in runtime_precheck["bindings"]
        }
        canonical_binding_cells = {
            (
                bound_family,
                bound_seed,
                str(
                    (
                        Path(identity["canonical_experiment_root"])
                        / bound_family.lower()
                        / f"seed{bound_seed}"
                    ).resolve()
                ),
            )
            for bound_family in S2_FAMILIES
            for bound_seed in S2_TRAINING_SEEDS
        }
        if (
            runtime_precheck.get("full_model_gate_file_sha256")
            != identity["full_model_gate_file_sha256"]
            or runtime_precheck.get("protocol_sha256")
            != external["protocol"]["declared_protocol_sha256"]
            or runtime_precheck.get("experiment_namespace")
            != identity["experiment_namespace"]
            or runtime_precheck.get("canonical_experiment_root")
            != identity["canonical_experiment_root"]
            or runtime_precheck.get("development_video_inventory")
            != development_inventory
            or expected_binding_cells != canonical_binding_cells
        ):
            raise ValueError(
                "Continuous-RoI S2 runtime precheck is not bound to these inputs"
            )

    source = Config.fromfile(str(source_config_path))
    if "continuous_roi_s2_runtime_binding" in source:
        raise ValueError("Continuous-RoI S2 source config is already bound")
    cfg = copy.deepcopy(source)
    manifest = external["manifest"]
    for split_name in ("train", "val"):
        split = cfg.dataset[split_name]
        split.ann_file = str(Path(development_annotation_path).resolve())
        split.class_map = str(Path(class_map_path).resolve())
        split.data_path = str(development_video_root)
        split.subset_name = "training"
        split.block_list = list(
            manifest["splits"]["gate"]
            if split_name == "train"
            else manifest["splits"]["fit"]
        )
    cfg.dataset.test = copy.deepcopy(cfg.dataset.val)
    cfg.evaluation.subset = "training"
    cfg.post_processing.save_dict = False
    cfg.model.backbone.custom.pretrain = str(Path(pretrained_checkpoint_path).resolve())
    temporal_upsampling = _bind_deterministic_temporal_upsampling(cfg, family=family)
    if family == "U128":
        cfg.model.backbone.custom.continuous_roi_training_seed = seed
    cfg.work_dir = str(work_dir)
    cfg.workflow.checkpoint_interval = S2_EPOCHS
    cfg.workflow.val_loss_interval = -1
    cfg.workflow.val_eval_interval = -1
    cfg.workflow.val_start_epoch = S2_EPOCHS
    cfg.workflow.end_epoch = S2_EPOCHS
    cfg.workflow.max_train_iters = None
    cfg.workflow.disable_checkpoint = False
    cfg.workflow.schedule_and_ema_on_success_only = True
    cfg.workflow.max_amp_retries_per_batch = 8
    cfg.workflow.fail_on_skipped_update = True
    cfg.continuous_roi_s2_gate.update(
        stage="formal-development-training",
        precheck_only=False,
        requires_launch_gate=True,
        launch_gate_passed=True,
        allow_detector_training=True,
        allow_tools_train=True,
        allow_tools_test=False,
        allow_detector_map=False,
        allowed_entrypoints=["tools/train.py"],
        official_test_open_allowed=False,
        learned_crop_policy_allowed=False,
        selector_parameters=0,
        paper_claim_allowed=False,
    )
    binding = {
        "schema_version": S2_TRAINING_BINDING_SCHEMA,
        "runtime_bound": True,
        "family": family,
        "seed": seed,
        "source_config_path": str(source_config_path),
        "source_config_sha256": canonical_sha256(source.to_dict()),
        "code_commit": code_commit,
        "protocol_sha256": external["protocol"]["declared_protocol_sha256"],
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_file_sha256": external["file_hashes"]["base_manifest_file_sha256"],
        "manifest_semantic_sha256": manifest["manifest_sha256"],
        "development_annotation_path": str(Path(development_annotation_path).resolve()),
        "development_annotation_sha256": external["file_hashes"][
            "development_annotation_sha256"
        ],
        "class_map_path": str(Path(class_map_path).resolve()),
        "class_map_sha256": external["file_hashes"]["class_map_sha256"],
        "development_video_root": str(development_video_root),
        "development_video_inventory_sha256": development_inventory["inventory_sha256"],
        "fit_split_sha256": manifest["split_hashes"]["fit"],
        "gate_split_sha256": manifest["split_hashes"]["gate"],
        "fit_video_ids": list(manifest["splits"]["fit"]),
        "gate_video_ids": list(manifest["splits"]["gate"]),
        "pretrained_checkpoint_path": str(Path(pretrained_checkpoint_path).resolve()),
        "pretrained_checkpoint_sha256": external["file_hashes"][
            "videomae_s_checkpoint_sha256"
        ],
        "full_model_gate_path": str(Path(full_model_gate_path).resolve()),
        "full_model_gate_file_sha256": identity["full_model_gate_file_sha256"],
        "full_model_gate_sha256": gate["gate_sha256"],
        "training_runtime_precheck_path": (
            None
            if training_runtime_precheck_path is None
            else str(Path(training_runtime_precheck_path).resolve())
        ),
        "training_runtime_precheck_file_sha256": (
            None
            if training_runtime_precheck_path is None
            else sha256_file(training_runtime_precheck_path)
        ),
        "training_runtime_precheck_sha256": (
            None if runtime_precheck is None else runtime_precheck["precheck_sha256"]
        ),
        "runtime_authorization_path": (
            None
            if runtime_authorization_path is None
            else str(Path(runtime_authorization_path).resolve())
        ),
        "runtime_authorization_file_sha256": (
            None
            if runtime_authorization_path is None
            else sha256_file(runtime_authorization_path)
        ),
        "runtime_authorization_sha256": (
            None
            if runtime_authorization is None
            else runtime_authorization["authorization_sha256"]
        ),
        "base_experiment_namespace": identity["experiment_namespace"],
        "experiment_namespace": experiment_namespace,
        "canonical_experiment_root": canonical_experiment_root,
        "work_dir": str(work_dir),
        "successful_updates": S2_SUCCESSFUL_UPDATES,
        "updates_per_epoch": S2_UPDATES_PER_EPOCH,
        "epochs": S2_EPOCHS,
        "checkpoint_selection": "final_ema_only",
        "checkpoint_consumer_state_key": "state_dict_ema",
        "temporal_upsampling": temporal_upsampling,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    cfg.continuous_roi_s2_runtime_binding = binding
    return cfg


def validate_bound_training_config(cfg: Config, *, seed: int) -> dict[str, Any]:
    binding = cfg.get("continuous_roi_s2_runtime_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("Continuous-RoI S2 source configs are not directly trainable")
    if binding.get("schema_version") != S2_TRAINING_BINDING_SCHEMA:
        raise ValueError("unsupported Continuous-RoI S2 training binding")
    family = str(binding.get("family", "")).upper()
    seed = int(seed)
    if seed != int(binding.get("seed", -1)):
        raise ValueError("Continuous-RoI S2 runtime seed differs from its binding")
    source_config_path = Path(str(binding["source_config_path"])).resolve()
    repository_root = source_config_path
    for _ in S2_SOURCE_CONFIGS[family].parts:
        repository_root = repository_root.parent
    require_clean_git_checkout(
        expected_commit=str(binding["code_commit"]),
        repository_root=repository_root,
    )
    expected = bind_training_config(
        source_config_path=source_config_path,
        family=family,
        seed=seed,
        work_dir=binding["work_dir"],
        manifest_path=binding["manifest_path"],
        development_annotation_path=binding["development_annotation_path"],
        class_map_path=binding["class_map_path"],
        development_video_root=binding["development_video_root"],
        pretrained_checkpoint_path=binding["pretrained_checkpoint_path"],
        full_model_gate_path=binding["full_model_gate_path"],
        training_runtime_precheck_path=binding["training_runtime_precheck_path"],
        runtime_authorization_path=binding["runtime_authorization_path"],
        repository_root=repository_root,
        code_commit=binding["code_commit"],
        require_runtime_precheck=True,
        require_runtime_authorization=True,
    )
    if canonical_sha256(cfg.to_dict()) != canonical_sha256(expected.to_dict()):
        raise ValueError(
            "Continuous-RoI S2 bound config was modified after materialization"
        )
    return dict(binding)


def should_save_final_checkpoint(epoch: int, binding: Mapping[str, Any]) -> bool:
    return int(epoch) == int(binding["epochs"]) - 1


def build_checkpoint_metadata(
    cfg: Config,
    *,
    seed: int,
    epoch: int,
    successful_updates: int,
    train_batches_per_epoch: int,
    amp_skipped_attempts: int,
    max_amp_retries_observed: int,
) -> dict[str, Any]:
    binding = validate_bound_training_config(cfg, seed=seed)
    if (
        int(epoch) != S2_EPOCHS - 1
        or int(train_batches_per_epoch) != S2_UPDATES_PER_EPOCH
        or int(successful_updates) != S2_SUCCESSFUL_UPDATES
    ):
        raise RuntimeError(
            "Continuous-RoI S2 final checkpoint has an incomplete update ledger"
        )
    if (
        int(amp_skipped_attempts) < 0
        or int(max_amp_retries_observed) < 0
        or int(max_amp_retries_observed) > 8
    ):
        raise ValueError("Continuous-RoI S2 AMP retry audit is inconsistent")
    metadata = {
        "schema_version": S2_CHECKPOINT_METADATA_SCHEMA,
        "family": binding["family"],
        "seed": int(seed),
        "epoch": int(epoch),
        "successful_updates": int(successful_updates),
        "updates_per_epoch": int(train_batches_per_epoch),
        "optimizer_attempts": int(successful_updates) + int(amp_skipped_attempts),
        "amp_skipped_attempts": int(amp_skipped_attempts),
        "max_amp_retries_per_batch": 8,
        "max_amp_retries_observed": int(max_amp_retries_observed),
        "checkpoint_selection": "final_ema_only",
        "checkpoint_consumer_state_key": "state_dict_ema",
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "source_config_sha256": binding["source_config_sha256"],
        "code_commit": binding["code_commit"],
        "protocol_sha256": binding["protocol_sha256"],
        "manifest_file_sha256": binding["manifest_file_sha256"],
        "manifest_semantic_sha256": binding["manifest_semantic_sha256"],
        "development_annotation_sha256": binding["development_annotation_sha256"],
        "development_video_inventory_sha256": binding[
            "development_video_inventory_sha256"
        ],
        "class_map_sha256": binding["class_map_sha256"],
        "fit_split_sha256": binding["fit_split_sha256"],
        "gate_split_sha256": binding["gate_split_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "full_model_gate_file_sha256": binding["full_model_gate_file_sha256"],
        "full_model_gate_sha256": binding["full_model_gate_sha256"],
        "training_runtime_precheck_file_sha256": binding[
            "training_runtime_precheck_file_sha256"
        ],
        "training_runtime_precheck_sha256": binding["training_runtime_precheck_sha256"],
        "runtime_authorization_file_sha256": binding[
            "runtime_authorization_file_sha256"
        ],
        "runtime_authorization_sha256": binding["runtime_authorization_sha256"],
        "base_experiment_namespace": binding["base_experiment_namespace"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return metadata


def checkpoint_sidecar_path(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".metadata.json")


def validate_checkpoint_sidecar(
    checkpoint_path: str | Path,
    *,
    expected_sidecar_schema: str = S2_CHECKPOINT_SIDECAR_SCHEMA,
    expected_metadata_schema: str = S2_CHECKPOINT_METADATA_SCHEMA,
    expected_successful_updates: int = S2_SUCCESSFUL_UPDATES,
    expected_checkpoint_selection: str = "final_ema_only",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    sidecar_path = checkpoint_sidecar_path(checkpoint_path)
    if not checkpoint_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            checkpoint_path if not checkpoint_path.is_file() else sidecar_path
        )
    sidecar = _load_json(sidecar_path)
    sidecar_hash = sidecar.pop("sidecar_sha256", None)
    if not sidecar_hash or canonical_sha256(sidecar) != sidecar_hash:
        raise ValueError("Continuous-RoI S2 checkpoint sidecar hash mismatch")
    sidecar["sidecar_sha256"] = sidecar_hash
    if sidecar.get("schema_version") != expected_sidecar_schema:
        raise ValueError("unsupported Continuous-RoI S2 checkpoint sidecar")
    if Path(str(sidecar.get("checkpoint_path", ""))).resolve() != checkpoint_path:
        raise ValueError("Continuous-RoI S2 sidecar checkpoint path mismatch")
    if sidecar.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Continuous-RoI S2 checkpoint file hash mismatch")
    metadata = sidecar.get("experiment_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Continuous-RoI S2 sidecar has no experiment metadata")
    metadata_copy = dict(metadata)
    metadata_hash = metadata_copy.pop("metadata_sha256", None)
    if (
        metadata.get("schema_version") != expected_metadata_schema
        or not metadata_hash
        or canonical_sha256(metadata_copy) != metadata_hash
        or metadata.get("successful_updates") != expected_successful_updates
        or metadata.get("checkpoint_selection") != expected_checkpoint_selection
        or metadata.get("checkpoint_consumer_state_key") != "state_dict_ema"
        or metadata.get("official_test_opened") is not False
    ):
        raise ValueError("Continuous-RoI S2 checkpoint metadata is invalid")
    return sidecar


def _count_nonfinite_values(value: Any, torch_module: Any) -> int:
    if torch_module.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            return int((~torch_module.isfinite(value)).sum().item())
        return 0
    if isinstance(value, float):
        return 0 if math.isfinite(value) else 1
    if isinstance(value, Mapping):
        return sum(
            _count_nonfinite_values(item, torch_module) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_count_nonfinite_values(item, torch_module) for item in value)
    return 0


class _CheckpointAuditLogger:
    def info(self, *_args, **_kwargs) -> None:
        return None


def build_checkpoint_validation_runtime(cfg: Config) -> tuple[Any, Any]:
    """Build the real detector and optimizer without loading pretrained weights."""

    importlib.import_module("opentad.datasets")
    importlib.import_module("opentad.models")
    importlib.import_module("opentad.models.backbones")
    from opentad.cores import build_optimizer
    from opentad.models import build_detector

    model_cfg = copy.deepcopy(cfg.model)
    custom = model_cfg.backbone.custom
    if not hasattr(custom, "pretrain"):
        raise ValueError("Continuous-RoI S2 model has no bound pretrain field")
    custom.pretrain = None
    model = build_detector(model_cfg).cpu().eval()
    optimizer = build_optimizer(
        copy.deepcopy(cfg.optimizer),
        SimpleNamespace(module=model),
        _CheckpointAuditLogger(),
    )
    return model, optimizer


def audit_checkpoint_against_model(
    checkpoint: Mapping[str, Any], *, model: Any, optimizer: Any
) -> dict[str, Any]:
    """Strict-load both model states and the optimizer into the real runtime."""

    raw_state = checkpoint["state_dict"]
    ema_state = checkpoint["state_dict_ema"]
    raw_result = model.load_state_dict(raw_state, strict=True)
    ema_result = model.load_state_dict(ema_state, strict=True)
    if (
        raw_result.missing_keys
        or raw_result.unexpected_keys
        or ema_result.missing_keys
        or ema_result.unexpected_keys
    ):
        raise ValueError("Continuous-RoI S2 strict model loading was incomplete")
    serialized_optimizer = checkpoint["optimizer"]
    serialized_groups = serialized_optimizer.get("param_groups")
    if not isinstance(serialized_groups, list):
        raise ValueError("Continuous-RoI S2 optimizer has no parameter groups")
    serialized_param_ids = [
        param_id for group in serialized_groups for param_id in group.get("params", [])
    ]
    if len(serialized_param_ids) != len(set(serialized_param_ids)):
        raise ValueError("Continuous-RoI S2 optimizer repeats a parameter ID")
    serialized_state_ids = set(serialized_optimizer.get("state", {}))
    if not serialized_state_ids.issubset(set(serialized_param_ids)):
        raise ValueError("Continuous-RoI S2 optimizer contains orphan state")
    serialized_group_sizes = [
        len(group.get("params", [])) for group in serialized_groups
    ]
    runtime_group_sizes = [len(group["params"]) for group in optimizer.param_groups]
    runtime_params = [
        param for group in optimizer.param_groups for param in group["params"]
    ]
    runtime_param_ids = {id(param) for param in runtime_params}
    if len(runtime_param_ids) != len(runtime_params):
        raise ValueError("Continuous-RoI S2 runtime optimizer repeats a parameter")
    if serialized_group_sizes != runtime_group_sizes:
        raise ValueError("Continuous-RoI S2 optimizer parameter groups differ")
    optimizer.load_state_dict(serialized_optimizer)
    if len(optimizer.state) != len(checkpoint["optimizer"]["state"]):
        raise ValueError("Continuous-RoI S2 strict optimizer loading was incomplete")
    if any(id(param) not in runtime_param_ids for param in optimizer.state):
        raise ValueError("Continuous-RoI S2 loaded optimizer contains orphan state")
    return {
        "raw_model_strict_load_valid": True,
        "ema_model_strict_load_valid": True,
        "optimizer_strict_load_valid": True,
        "runtime_model_state_key_count": len(model.state_dict()),
        "runtime_optimizer_state_count": len(optimizer.state),
        "runtime_optimizer_param_group_count": len(optimizer.param_groups),
        "runtime_optimizer_param_group_sizes": runtime_group_sizes,
    }


def audit_final_checkpoint_state(
    checkpoint: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    expected_scheduler_max_epoch: int | None = None,
    expected_scheduler_warmup_epoch: int | None = None,
) -> dict[str, Any]:
    import torch

    raw_state = checkpoint.get("state_dict")
    ema_state = checkpoint.get("state_dict_ema")
    optimizer = checkpoint.get("optimizer")
    scheduler = checkpoint.get("scheduler")
    amp_skipped_attempts = int(metadata.get("amp_skipped_attempts", -1))
    max_amp_retries_observed = int(metadata.get("max_amp_retries_observed", -1))
    max_amp_retries_per_batch = int(metadata.get("max_amp_retries_per_batch", -1))
    if (
        not isinstance(raw_state, Mapping)
        or not isinstance(ema_state, Mapping)
        or not raw_state
        or not ema_state
        or set(raw_state) != set(ema_state)
        or checkpoint.get("experiment_metadata") != metadata
        or int(checkpoint.get("epoch", -1)) != S2_EPOCHS - 1
        or int(metadata.get("epoch", -1)) != S2_EPOCHS - 1
        or int(metadata.get("updates_per_epoch", -1)) != S2_UPDATES_PER_EPOCH
        or int(metadata.get("successful_updates", -1)) != S2_SUCCESSFUL_UPDATES
        or int(metadata.get("optimizer_attempts", -1))
        != int(metadata.get("successful_updates", -1)) + amp_skipped_attempts
        or amp_skipped_attempts < 0
        or max_amp_retries_per_batch != 8
        or max_amp_retries_observed < 0
        or max_amp_retries_observed > max_amp_retries_per_batch
        or (amp_skipped_attempts == 0) != (max_amp_retries_observed == 0)
        or not isinstance(optimizer, Mapping)
        or not isinstance(optimizer.get("state"), Mapping)
        or not optimizer["state"]
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer["param_groups"]
        or not isinstance(scheduler, Mapping)
        or int(scheduler.get("last_epoch", -1)) != S2_SUCCESSFUL_UPDATES
        or int(scheduler.get("_step_count", -1)) != S2_SUCCESSFUL_UPDATES + 1
        or not isinstance(scheduler.get("_last_lr"), list)
        or not scheduler["_last_lr"]
        or len(scheduler["_last_lr"]) != len(optimizer["param_groups"])
        or (
            expected_scheduler_max_epoch is not None
            and int(scheduler.get("max_epoch", -1)) != int(expected_scheduler_max_epoch)
        )
        or (
            expected_scheduler_warmup_epoch is not None
            and int(scheduler.get("warmup_epoch", -1))
            != int(expected_scheduler_warmup_epoch)
        )
    ):
        raise ValueError(
            "Continuous-RoI S2 final checkpoint lacks complete training state"
        )

    optimizer_steps = []
    for state in optimizer["state"].values():
        if not isinstance(state, Mapping) or "step" not in state:
            raise ValueError(
                "Continuous-RoI S2 optimizer state has no successful-update step"
            )
        step = state["step"]
        if torch.is_tensor(step):
            if step.numel() != 1:
                raise ValueError("Continuous-RoI S2 optimizer step is not scalar")
            step = step.item()
        optimizer_steps.append(int(step))
    if set(optimizer_steps) != {S2_SUCCESSFUL_UPDATES}:
        raise ValueError(
            "Continuous-RoI S2 optimizer states do not close at 4,800 updates"
        )

    changed_tensors = 0
    for key in raw_state:
        raw_value = raw_state[key]
        ema_value = ema_state[key]
        if torch.is_tensor(raw_value) != torch.is_tensor(ema_value):
            raise ValueError("Continuous-RoI S2 raw/EMA state types differ")
        if torch.is_tensor(raw_value):
            if raw_value.shape != ema_value.shape or raw_value.dtype != ema_value.dtype:
                raise ValueError("Continuous-RoI S2 raw/EMA tensor metadata differ")
            changed_tensors += int(not torch.equal(raw_value, ema_value))
        elif raw_value != ema_value:
            changed_tensors += 1
    nonfinite_count = sum(
        _count_nonfinite_values(value, torch)
        for value in (raw_state, ema_state, optimizer, scheduler)
    )
    if nonfinite_count or changed_tensors <= 0:
        raise ValueError(
            "Continuous-RoI S2 final checkpoint has non-finite or stale EMA state"
        )
    return {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "raw_state_key_count": len(raw_state),
        "ema_state_key_count": len(ema_state),
        "ema_changed_value_count": changed_tensors,
        "optimizer_state_count": len(optimizer["state"]),
        "optimizer_param_group_count": len(optimizer["param_groups"]),
        "optimizer_state_step_min": min(optimizer_steps),
        "optimizer_state_step_max": max(optimizer_steps),
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_step_count": int(scheduler["_step_count"]),
        "scheduler_max_epoch": int(scheduler.get("max_epoch", -1)),
        "scheduler_warmup_epoch": int(scheduler.get("warmup_epoch", -1)),
        "amp_skipped_attempts": amp_skipped_attempts,
        "max_amp_retries_per_batch": max_amp_retries_per_batch,
        "max_amp_retries_observed": max_amp_retries_observed,
        "nonfinite_value_count": nonfinite_count,
    }


def _build_training_completion_and_audit(
    *,
    cfg: Config,
    seed: int,
    checkpoint_path: str | Path,
    strict_model: Any | None = None,
    strict_optimizer: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = validate_bound_training_config(cfg, seed=seed)
    checkpoint_path = Path(checkpoint_path).resolve()
    sidecar = validate_checkpoint_sidecar(checkpoint_path)
    metadata = sidecar["experiment_metadata"]
    metadata_binding_fields = (
        "source_config_sha256",
        "code_commit",
        "protocol_sha256",
        "manifest_file_sha256",
        "manifest_semantic_sha256",
        "development_annotation_sha256",
        "development_video_inventory_sha256",
        "class_map_sha256",
        "fit_split_sha256",
        "gate_split_sha256",
        "pretrained_checkpoint_sha256",
        "full_model_gate_file_sha256",
        "full_model_gate_sha256",
        "training_runtime_precheck_file_sha256",
        "training_runtime_precheck_sha256",
        "runtime_authorization_file_sha256",
        "runtime_authorization_sha256",
        "base_experiment_namespace",
        "experiment_namespace",
        "canonical_experiment_root",
    )
    if (
        metadata["family"] != binding["family"]
        or int(metadata["seed"]) != int(seed)
        or metadata["bound_config_sha256"] != canonical_sha256(cfg.to_dict())
        or any(
            metadata.get(field) != binding.get(field)
            for field in metadata_binding_fields
        )
    ):
        raise ValueError("Continuous-RoI S2 checkpoint is bound to another run")
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Continuous-RoI S2 final checkpoint is not a mapping")
    scheduler_max_epoch = int(cfg.scheduler["max_epoch"]) * S2_UPDATES_PER_EPOCH
    scheduler_warmup_epoch = (
        int(cfg.scheduler.get("warmup_epoch", 0)) * S2_UPDATES_PER_EPOCH
    )
    checkpoint_audit = audit_final_checkpoint_state(
        checkpoint,
        metadata=metadata,
        expected_scheduler_max_epoch=scheduler_max_epoch,
        expected_scheduler_warmup_epoch=scheduler_warmup_epoch,
    )
    checkpoint_audit["runtime_binding"] = {
        field: binding[field] for field in metadata_binding_fields
    }
    if (strict_model is None) != (strict_optimizer is None):
        raise ValueError("strict model and optimizer must be provided together")
    if strict_model is not None:
        checkpoint_audit.update(
            audit_checkpoint_against_model(
                checkpoint,
                model=strict_model,
                optimizer=strict_optimizer,
            )
        )
    del checkpoint
    report = {
        "schema_version": S2_TRAINING_COMPLETION_SCHEMA,
        "status": "PASS",
        "family": binding["family"],
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
        "checkpoint_sidecar_path": str(checkpoint_sidecar_path(checkpoint_path)),
        "checkpoint_sidecar_sha256": sha256_file(
            checkpoint_sidecar_path(checkpoint_path)
        ),
        "checkpoint_metadata_sha256": metadata["metadata_sha256"],
        "successful_updates": metadata["successful_updates"],
        "optimizer_attempts": metadata["optimizer_attempts"],
        "amp_skipped_attempts": metadata["amp_skipped_attempts"],
        "max_amp_retries_observed": metadata["max_amp_retries_observed"],
        "ema_state_nonempty": True,
        "ema_keys_match_model": True,
        "checkpoint_consumer_state_key": "state_dict_ema",
        "experiment_namespace": binding["experiment_namespace"],
        "code_commit": binding["code_commit"],
        "protocol_sha256": binding["protocol_sha256"],
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    report["completion_sha256"] = canonical_sha256(report)
    return report, checkpoint_audit


def build_training_completion(
    *,
    cfg: Config,
    seed: int,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    report, _ = _build_training_completion_and_audit(
        cfg=cfg,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )
    return report


def validate_training_completion(
    completion: Mapping[str, Any],
    *,
    cfg: Config,
    seed: int,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Rebuild a completion receipt from its live artifacts and compare exactly."""

    checked = dict(completion)
    completion_hash = checked.pop("completion_sha256", None)
    if not completion_hash or canonical_sha256(checked) != completion_hash:
        raise ValueError("Continuous-RoI S2 training completion hash mismatch")
    checked["completion_sha256"] = completion_hash
    rebuilt, _ = _build_training_completion_and_audit(
        cfg=cfg,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )
    if checked != rebuilt:
        raise ValueError(
            "Continuous-RoI S2 training completion no longer matches its artifacts"
        )
    return rebuilt


def validate_training_completion_with_audit(
    completion: Mapping[str, Any],
    *,
    cfg: Config,
    seed: int,
    checkpoint_path: str | Path,
    strict_model: Any | None = None,
    strict_optimizer: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checked = dict(completion)
    completion_hash = checked.pop("completion_sha256", None)
    if not completion_hash or canonical_sha256(checked) != completion_hash:
        raise ValueError("Continuous-RoI S2 training completion hash mismatch")
    checked["completion_sha256"] = completion_hash
    rebuilt, checkpoint_audit = _build_training_completion_and_audit(
        cfg=cfg,
        seed=seed,
        checkpoint_path=checkpoint_path,
        strict_model=strict_model,
        strict_optimizer=strict_optimizer,
    )
    if checked != rebuilt:
        raise ValueError(
            "Continuous-RoI S2 training completion no longer matches its artifacts"
        )
    return rebuilt, checkpoint_audit


__all__ = [
    "S2_CHECKPOINT_SIDECAR_SCHEMA",
    "S2_EPOCHS",
    "S2_FAMILIES",
    "S2_SOURCE_CONFIGS",
    "S2_SUCCESSFUL_UPDATES",
    "S2_TRAINING_SEEDS",
    "S2_UPDATES_PER_EPOCH",
    "audit_development_video_inventory",
    "audit_development_video_census",
    "audit_checkpoint_against_model",
    "audit_final_checkpoint_state",
    "bind_training_config",
    "build_checkpoint_metadata",
    "build_checkpoint_validation_runtime",
    "build_training_completion",
    "checkpoint_sidecar_path",
    "current_git_commit",
    "load_pure_data_config",
    "require_clean_git_checkout",
    "should_save_final_checkpoint",
    "validate_bound_training_config",
    "validate_checkpoint_sidecar",
    "validate_training_completion",
    "validate_training_completion_with_audit",
    "validate_external_training_inputs",
    "validate_full_model_gate",
    "validate_training_runtime_precheck",
]
