from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.native_crop_s1_contract import (  # noqa: E402
    NATIVE_CROP_CHECKPOINT_CORE_NUMEL,
    NATIVE_CROP_CHECKPOINT_CORE_TENSORS,
    NATIVE_CROP_CHECKPOINT_STATE_TENSORS,
    NATIVE_CROP_CLASS_MAP_SHA256,
    NATIVE_CROP_DEVELOPMENT_ANNOTATION_SHA256,
    NATIVE_CROP_DEVELOPMENT_VIDEO_COUNT,
    NATIVE_CROP_DEVELOPMENT_WINDOW_COUNT,
    NATIVE_CROP_GATE_WINDOW_COUNT,
    NATIVE_CROP_GEOMETRY_SCHEMA,
    NATIVE_CROP_MANIFEST_FILE_SHA256,
    NATIVE_CROP_PRECHECK_SCHEMA,
    NATIVE_CROP_PRETRAINED_FILENAME,
    NATIVE_CROP_PRETRAINED_SHA256,
    build_cost_schema,
    finalize_self_hash,
    validate_development_only_manifest,
)
from tools.bata.native_crop_s1_geometry_census import (  # noqa: E402
    probe_video_geometry,
    summarize_records,
)
from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)


EXPECTED_UNUSED_TRAINABLE = (
    "backbone.model.backbone.fc_norm.bias",
    "backbone.model.backbone.fc_norm.weight",
)

AUDITED_SOURCE_PATHS = (
    "configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",
    "configs/_base_/models/actionformer.py",
    "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py",
    "configs/adatad/thumos/native_crop_s1_center_videomae_s_768x1_adapter.py",
    "opentad/datasets/transforms/__init__.py",
    "opentad/datasets/transforms/formatting.py",
    "opentad/datasets/transforms/native_crop.py",
    "opentad/models/backbones/__init__.py",
    "opentad/models/backbones/native_crop_wrapper.py",
    "opentad/models/builder.py",
    "tools/bata/build_native_crop_s1_development_annotation.py",
    "tools/bata/native_crop_s1_contract.py",
    "tools/bata/native_crop_s1_geometry_census.py",
    "tools/bata/run_native_crop_s1_precheck.py",
    "scripts/run_native_crop_s1_gate_slurm.sh",
    "tests/test_native_crop_s1_vertical_slice.py",
)
CANONICAL_CONFIG_PATH = (
    ROOT
    / "configs/adatad/thumos/native_crop_s1_center_videomae_s_768x1_adapter.py"
)
REFERENCE_CONFIG_PATH = (
    ROOT / "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
)


def audit_code_provenance(*, expected_commit: str) -> dict:
    def run_git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def read_git_blob(commit: str, relative_path: str) -> bytes:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return completed.stdout

    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("Native-Crop expected_commit must be one full Git commit")
    commit = run_git("rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError(
            f"Native-Crop checkout {commit} differs from expected {expected_commit}"
        )
    worktree_status = run_git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if worktree_status:
        raise ValueError(
            "Native-Crop formal gate requires a completely clean worktree: "
            f"{worktree_status.splitlines()[:8]}"
        )
    source_hashes = {}
    head_blob_hashes = {}
    for relative_path in AUDITED_SOURCE_PATHS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        run_git("ls-files", "--error-unmatch", "--", relative_path)
        file_bytes = path.read_bytes()
        head_bytes = read_git_blob(commit, relative_path)
        if file_bytes != head_bytes:
            raise ValueError(
                "Native-Crop audited source differs from the expected HEAD blob: "
                f"{relative_path}"
            )
        source_hashes[relative_path] = hashlib.sha256(file_bytes).hexdigest()
        head_blob_hashes[relative_path] = hashlib.sha256(head_bytes).hexdigest()
    return {
        "expected_commit": expected_commit,
        "git_commit": commit,
        "complete_worktree_clean": True,
        "all_audited_sources_tracked": True,
        "all_audited_sources_equal_head": True,
        "audited_source_sha256": source_hashes,
        "head_blob_sha256": head_blob_hashes,
    }


def audit_census_record_source(
    record: dict,
    *,
    expected_path: Path,
    video_root: Path,
) -> dict:
    root = video_root.resolve()
    resolved_path = expected_path.resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Native-Crop census source escapes video_root: {resolved_path}"
        ) from exc
    if Path(record.get("path", "")).resolve() != resolved_path:
        raise ValueError("Native-Crop geometry census record path changed")
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    current_size = int(resolved_path.stat().st_size)
    if int(record.get("file_size_bytes", -1)) != current_size:
        raise ValueError(
            f"Native-Crop geometry census source size changed: {resolved_path}"
        )
    actual_geometry = probe_video_geometry(resolved_path)
    for key in (
        "width",
        "height",
        "rotation_degrees",
        "nb_frames",
        "avg_frame_rate",
    ):
        if record.get(key) != actual_geometry.get(key):
            raise ValueError(
                "Native-Crop geometry census differs from the current source: "
                f"path={resolved_path} field={key} "
                f"recorded={record.get(key)!r} actual={actual_geometry.get(key)!r}"
            )
    return {
        "path": str(resolved_path),
        "file_size_bytes": current_size,
        **actual_geometry,
    }


def audit_geometry_census(
    census_path: Path,
    *,
    manifest_path: Path,
    video_root: Path,
) -> dict:
    report = json.loads(census_path.read_text(encoding="utf-8"))
    payload = copy.deepcopy(report)
    census_sha256 = payload.pop("census_sha256", None)
    if (
        not isinstance(census_sha256, str)
        or len(census_sha256) != 64
        or canonical_sha256(payload) != census_sha256
    ):
        raise ValueError("Native-Crop geometry census self-hash is invalid")
    if (
        payload.get("schema_version") != NATIVE_CROP_GEOMETRY_SCHEMA
        or payload.get("sealed_test_files_probed") != 0
        or payload.get("annotation_or_gt_read") is not False
        or payload.get("development_splits_probed") != ["fit", "gate"]
        or payload.get("crop_sizes") != [96, 112, 128]
    ):
        raise ValueError("Native-Crop geometry census opened sealed-test files")
    if Path(payload["manifest_path"]).resolve() != manifest_path.resolve():
        raise ValueError("Native-Crop geometry census changed the manifest path")
    if Path(payload["video_root"]).resolve() != video_root.resolve():
        raise ValueError("Native-Crop geometry census changed the video root")
    if (
        payload.get("manifest_file_sha256") != NATIVE_CROP_MANIFEST_FILE_SHA256
        or payload.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("Native-Crop geometry census manifest file identity changed")
    manifest = validate_development_only_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if (
        payload.get("manifest_sha256") != manifest["manifest_sha256"]
        or payload.get("sealed_test_identity_count") != len(manifest["sealed_test"])
    ):
        raise ValueError("Native-Crop geometry census manifest contents changed")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != (
        NATIVE_CROP_DEVELOPMENT_VIDEO_COUNT
    ):
        raise ValueError("Native-Crop geometry census must contain 200 records")
    expected_ids = {
        "fit": set(manifest["fit"]),
        "gate": set(manifest["gate"]),
    }
    observed_ids = {"fit": set(), "gate": set()}
    for record in records:
        split_name = record.get("split")
        video_id = str(record.get("video_id", ""))
        if split_name not in observed_ids or video_id in observed_ids[split_name]:
            raise ValueError("Native-Crop geometry census has duplicate/invalid records")
        expected_path = (video_root / f"{video_id}.mp4").resolve()
        audit_census_record_source(
            record,
            expected_path=expected_path,
            video_root=video_root,
        )
        observed_ids[split_name].add(video_id)
    if observed_ids != expected_ids:
        raise ValueError("Native-Crop geometry census population changed")
    expected_summary = {
        "combined": summarize_records(records, [96, 112, 128]),
        "fit": summarize_records(
            [record for record in records if record["split"] == "fit"],
            [96, 112, 128],
        ),
        "gate": summarize_records(
            [record for record in records if record["split"] == "gate"],
            [96, 112, 128],
        ),
    }
    if payload.get("summary") != expected_summary:
        raise ValueError("Native-Crop geometry census summary was not record-derived")
    combined = expected_summary["combined"]
    crop_128 = combined.get("crop_sizes", {}).get("128", {})
    if crop_128.get("no_padding_count") != NATIVE_CROP_DEVELOPMENT_VIDEO_COUNT:
        raise ValueError("Native-Crop local128 is not padding-free on development")
    return {
        "census_sha256": census_sha256,
        "census_file_sha256": sha256_file(census_path),
        "development_video_count": NATIVE_CROP_DEVELOPMENT_VIDEO_COUNT,
        "local128_no_padding_count": NATIVE_CROP_DEVELOPMENT_VIDEO_COUNT,
        "sealed_test_files_probed": 0,
    }


def audit_loaded_pretrained_state(model, checkpoint_path: Path) -> dict:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Native-Crop pretrained checkpoint is not a mapping")
    checkpoint_state = payload.get("state_dict", payload)
    if not isinstance(checkpoint_state, dict):
        raise ValueError("Native-Crop checkpoint has no valid state dictionary")
    checkpoint_state = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in checkpoint_state.items()
    }
    checkpoint_core_names = sorted(
        name for name in checkpoint_state if name.startswith("backbone.")
    )
    checkpoint_core_numel = sum(
        int(checkpoint_state[name].numel()) for name in checkpoint_core_names
    )
    if (
        len(checkpoint_state) != NATIVE_CROP_CHECKPOINT_STATE_TENSORS
        or len(checkpoint_core_names) != NATIVE_CROP_CHECKPOINT_CORE_TENSORS
        or checkpoint_core_numel != NATIVE_CROP_CHECKPOINT_CORE_NUMEL
    ):
        raise ValueError("Native-Crop frozen checkpoint tensor contract changed")
    recognizer_state = model.backbone.model.state_dict()
    core_names = sorted(
        name
        for name, value in recognizer_state.items()
        if name.startswith("backbone.")
        and ".adapter." not in name
        and not name.startswith("backbone.chronotransport.")
        and value.numel() > 0
    )
    if not core_names:
        raise ValueError("Native-Crop precheck found no VideoMAE core parameters")
    if core_names != checkpoint_core_names:
        raise ValueError(
            "Native-Crop model core key set differs from the complete frozen "
            "checkpoint core"
        )
    missing = [name for name in checkpoint_core_names if name not in recognizer_state]
    shape_mismatch = [
        name
        for name in checkpoint_core_names
        if name in checkpoint_state
        and tuple(checkpoint_state[name].shape) != tuple(recognizer_state[name].shape)
    ]
    value_mismatch = [
        name
        for name in checkpoint_core_names
        if name in checkpoint_state
        and name not in shape_mismatch
        and not torch.equal(
            recognizer_state[name].detach().cpu(),
            checkpoint_state[name].detach().cpu(),
        )
    ]
    if missing or shape_mismatch or value_mismatch:
        raise ValueError(
            "Native-Crop VideoMAE core did not exactly load the frozen checkpoint: "
            f"missing={missing[:8]} shape_mismatch={shape_mismatch[:8]} "
            f"value_mismatch={value_mismatch[:8]}"
        )
    return {
        "verified": True,
        "checkpoint_state_tensors": len(checkpoint_state),
        "core_parameter_tensors": len(checkpoint_core_names),
        "core_parameter_numel": checkpoint_core_numel,
        "core_key_set_matches_complete_checkpoint": True,
        "missing": [],
        "shape_mismatch": [],
        "value_mismatch": [],
    }


def _pipeline_types(pipeline) -> list[str]:
    return [str(step["type"]) for step in pipeline]


def validate_native_crop_config(
    cfg: Config,
    *,
    config_path: Path | None = None,
) -> dict:
    if (
        config_path is not None
        and config_path.resolve() != CANONICAL_CONFIG_PATH.resolve()
    ):
        raise ValueError("Native-Crop gate refuses a non-canonical config path")
    gate = cfg.native_crop_s1_gate
    if (
        gate.route != "spatial-zoom-native-crop-s1"
        or gate.precheck_only is not True
        or gate.allow_detector_training is not False
        or gate.allow_tools_test is not False
        or gate.official_test_open_allowed is not False
        or gate.teacher_allowed is not False
        or gate.oracle_allowed is not False
        or gate.learned_crop_policy_allowed is not False
        or gate.paper_claim_allowed is not False
    ):
        raise ValueError("Native-Crop vertical-slice gate is not fail-closed")
    if cfg.dataset.test is not None:
        raise ValueError("Native-Crop vertical slice must not materialize dataset.test")
    expected_prefix = [
        "PrepareVideoInfo",
        "mmaction.DecordInit",
        "LoadFrames",
        "mmaction.DecordDecode",
        "NativeCropSourceViews",
    ]
    pipeline_audit = {}
    for split_name in ("train", "val"):
        split = cfg.dataset[split_name]
        if split.subset_name != "training":
            raise ValueError(
                f"Native-Crop {split_name} must remain on the development subset"
            )
        pipeline = split.pipeline
        types = _pipeline_types(pipeline)
        if types[:5] != expected_prefix:
            raise ValueError(
                f"Native-Crop {split_name} must crop immediately after DecordDecode"
            )
        if any("Resize" in item or "Crop" in item for item in types[4:] if item != "NativeCropSourceViews"):
            raise ValueError(
                f"Native-Crop {split_name} contains an unreviewed spatial transform"
            )
        transform = pipeline[4]
        if (
            int(transform.global_size) != 96
            or int(transform.local_size) != 128
            or transform.allow_local_padding is not False
        ):
            raise ValueError("Native-Crop vertical-slice view contract changed")
        pipeline_audit[split_name] = types
    if float(cfg.dataset.val.window_overlap_ratio) != 0.5:
        raise ValueError(
            "Native-Crop development validation must retain the audited 0.5 "
            "overlap that covers the final short-action window"
        )
    custom = cfg.model.backbone.custom
    if (
        custom.wrapper_type != "native_crop_shared_videomae"
        or custom.native_crop_fusion_mode != "fixed_mean"
        or int(custom.native_crop_intermediate_length) != 384
        or int(custom.native_crop_output_length) != 768
    ):
        raise ValueError("Native-Crop shared-backbone contract changed")
    backbone = cfg.model.backbone.backbone
    expected_architecture = {
        "type": "VisionTransformerAdapter",
        "img_size": 224,
        "patch_size": 16,
        "embed_dims": 384,
        "depth": 12,
        "num_heads": 6,
        "num_frames": 16,
        "return_feat_map": True,
        "with_cp": True,
        "total_frames": 768,
        "adapter_index": list(range(12)),
    }
    observed_architecture = {
        key: copy.deepcopy(backbone[key]) for key in expected_architecture
    }
    if observed_architecture != expected_architecture:
        raise ValueError("Native-Crop VideoMAE-S architecture contract changed")

    reference = Config.fromfile(str(REFERENCE_CONFIG_PATH))
    candidate_model = copy.deepcopy(cfg.model.to_dict())
    reference_model = copy.deepcopy(reference.model.to_dict())
    candidate_custom = candidate_model["backbone"]["custom"]
    for key in list(candidate_custom):
        if key == "wrapper_type" or key.startswith("native_crop_"):
            candidate_custom.pop(key)
    if candidate_model != reference_model:
        raise ValueError(
            "Native-Crop changed the reference detector/model surface outside "
            "the audited wrapper fields"
        )
    if cfg.post_processing.to_dict() != reference.post_processing.to_dict():
        raise ValueError("Native-Crop changed the reference detector NMS contract")
    resolved_config_sha256 = canonical_sha256(cfg.to_dict())
    reference_model_sha256 = canonical_sha256(reference_model)
    return {
        "development_only": True,
        "official_test_dataset_materialized": False,
        "pipelines": pipeline_audit,
        "crop_after_decode_before_any_resize": True,
        "local_interpolation_allowed": False,
        "canonical_config_path": str(CANONICAL_CONFIG_PATH.resolve()),
        "resolved_config_sha256": resolved_config_sha256,
        "reference_model_sha256": reference_model_sha256,
        "detector_model_surface_matches_reference": True,
        "detector_nms_surface_matches_reference": True,
        "videomae_architecture": observed_architecture,
    }


def audit_development_annotation(
    *,
    annotation_path: Path,
    class_map_path: Path,
    manifest: dict,
) -> dict:
    annotation_sha256 = sha256_file(annotation_path)
    if annotation_sha256 != NATIVE_CROP_DEVELOPMENT_ANNOTATION_SHA256:
        raise ValueError(
            "Native-Crop gate requires the frozen development-only annotation"
        )
    class_map_sha256 = sha256_file(class_map_path)
    if class_map_sha256 != NATIVE_CROP_CLASS_MAP_SHA256:
        raise ValueError("Native-Crop class-map SHA-256 mismatch")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    database = payload.get("database")
    if not isinstance(database, dict):
        raise ValueError("Native-Crop development annotation has no database")
    expected_ids = set(manifest["fit"]) | set(manifest["gate"])
    if set(database) != expected_ids:
        raise ValueError(
            "Native-Crop development annotation population differs from manifest"
        )
    retained_subsets = sorted(
        {str(video.get("subset")) for video in database.values()}
    )
    if retained_subsets != ["training"]:
        raise ValueError(
            "Native-Crop development annotation contains an official-test subset"
        )
    return {
        "annotation_file_sha256": annotation_sha256,
        "class_map_file_sha256": class_map_sha256,
        "annotation_video_count": len(database),
        "annotation_video_ids_sha256": canonical_sha256(sorted(database)),
        "retained_subsets": retained_subsets,
        "official_test_annotation_records_loaded": 0,
    }


def audit_development_dataset(
    cfg: Config,
    *,
    manifest_path: Path,
    annotation_path: Path,
    class_map_path: Path,
    video_root: Path,
) -> dict:
    from opentad.datasets import build_dataset
    from opentad.datasets.builder import collate

    root_text = str(video_root.resolve()).replace("\\", "/").lower()
    if any(
        token in root_text
        for token in ("/test data/", "/th14_test_set_mp4", "/sealed_test")
    ):
        raise ValueError("Native-Crop dataset precheck refuses a sealed-test root")
    manifest = validate_development_only_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if sha256_file(manifest_path) != NATIVE_CROP_MANIFEST_FILE_SHA256:
        raise ValueError("Native-Crop manifest file SHA-256 mismatch")
    annotation_audit = audit_development_annotation(
        annotation_path=annotation_path,
        class_map_path=class_map_path,
        manifest=manifest,
    )
    train_cfg = copy.deepcopy(cfg.dataset.train)
    train_cfg.ann_file = str(annotation_path)
    train_cfg.class_map = str(class_map_path)
    train_cfg.data_path = str(video_root)
    train_cfg.block_list = list(manifest["gate"])
    fit_dataset = build_dataset(train_cfg)
    fit_ids = sorted({str(row[0]) for row in fit_dataset.data_list})
    if fit_ids != manifest["fit"] or len(fit_dataset) != 160:
        raise RuntimeError(
            "Native-Crop fit dataset population differs from the frozen manifest"
        )

    val_cfg = copy.deepcopy(cfg.dataset.val)
    val_cfg.ann_file = str(annotation_path)
    val_cfg.class_map = str(class_map_path)
    val_cfg.data_path = str(video_root)
    val_cfg.block_list = list(manifest["fit"])
    gate_dataset = build_dataset(val_cfg)
    gate_ids = sorted({str(row[0]) for row in gate_dataset.data_list})
    if (
        gate_ids != manifest["gate"]
        or len(gate_dataset) != NATIVE_CROP_GATE_WINDOW_COUNT
    ):
        raise RuntimeError(
            "Native-Crop gate dataset population differs from the frozen manifest"
        )

    development_cfg = copy.deepcopy(cfg.dataset.val)
    development_cfg.ann_file = str(annotation_path)
    development_cfg.class_map = str(class_map_path)
    development_cfg.data_path = str(video_root)
    development_cfg.block_list = None
    development_dataset = build_dataset(development_cfg)
    development_ids = sorted(
        {str(row[0]) for row in development_dataset.data_list}
    )
    if (
        development_ids != sorted(manifest["fit"] + manifest["gate"])
        or len(development_ids) != NATIVE_CROP_DEVELOPMENT_VIDEO_COUNT
        or len(development_dataset) != NATIVE_CROP_DEVELOPMENT_WINDOW_COUNT
    ):
        raise RuntimeError(
            "Native-Crop full development sliding population is not 200/664"
        )
    sample = gate_dataset[0]
    sample_inputs = sample.get("inputs")
    if not isinstance(sample_inputs, dict) or set(sample_inputs) != {
        "global",
        "local",
    }:
        raise RuntimeError("Native-Crop real dataset sample has invalid structured inputs")
    expected_shapes = {
        "global": (1, 3, 768, 96, 96),
        "local": (1, 3, 768, 128, 128),
    }
    for name, expected_shape in expected_shapes.items():
        value = sample_inputs[name]
        if tuple(value.shape) != expected_shape or str(value.dtype) != "uint8":
            raise RuntimeError(
                f"Native-Crop real {name} sample violates uint8 view contract"
            )
    collated = collate([sample])
    collated_inputs = collated.get("inputs")
    expected_collated_shapes = {
        "global": (1, 1, 3, 768, 96, 96),
        "local": (1, 1, 3, 768, 128, 128),
    }
    if not isinstance(collated_inputs, dict):
        raise RuntimeError("Native-Crop DataLoader did not preserve structured inputs")
    for name, expected_shape in expected_collated_shapes.items():
        value = collated_inputs.get(name)
        if (
            value is None
            or tuple(value.shape) != expected_shape
            or str(value.dtype) != "torch.uint8"
        ):
            raise RuntimeError(
                f"Native-Crop collated {name} violates the backbone input contract"
            )
    if tuple(collated["masks"].shape) != (1, 768):
        raise RuntimeError("Native-Crop collated mask violates the dense time contract")
    geometry = sample.get("metas", {}).get("native_crop_geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("local_interpolation") is not False
        or geometry.get("local_padding_ltrb") != [0, 0, 0, 0]
        or geometry.get("uses_gt") is not False
        or geometry.get("uses_teacher") is not False
        or geometry.get("uses_oracle") is not False
        or geometry.get("uses_test_evidence") is not False
    ):
        raise RuntimeError("Native-Crop real dataset geometry audit failed")
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "annotation_audit": annotation_audit,
        "fit_video_count": len(fit_ids),
        "gate_video_count": len(gate_ids),
        "gate_window_count": len(gate_dataset),
        "development_video_count": len(development_ids),
        "development_window_count": len(development_dataset),
        "sample_video_name": sample["metas"]["video_name"],
        "sample_input_shapes": {
            name: list(sample_inputs[name].shape) for name in sorted(sample_inputs)
        },
        "sample_collated_input_shapes": {
            name: list(collated_inputs[name].shape)
            for name in sorted(collated_inputs)
        },
        "sample_collated_mask_shape": list(collated["masks"].shape),
        "sample_input_dtype": "uint8",
        "sample_source_hw": geometry["source_hw"],
        "sample_local_source_box_xyxy": geometry["local_source_box_xyxy"],
        "sample_local_padding_ltrb": geometry["local_padding_ltrb"],
        "crop_decision_uses_gt": False,
        "opened_input_files": [
            {
                "role": "frozen_manifest",
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            {
                "role": "development_only_annotation",
                "path": str(annotation_path.resolve()),
                "sha256": annotation_audit["annotation_file_sha256"],
            },
            {
                "role": "class_map",
                "path": str(class_map_path.resolve()),
                "sha256": annotation_audit["class_map_file_sha256"],
            },
            {
                "role": "development_video_sample",
                "path": str(
                    (
                        video_root
                        / f"{sample['metas']['video_name']}.mp4"
                    ).resolve()
                ),
                "video_name": sample["metas"]["video_name"],
            },
        ],
        "official_test_annotation_records_loaded": 0,
        "official_test_video_files_opened": 0,
    }


def _gradient_audit(model) -> dict:
    import torch

    missing = []
    nonfinite = []
    nonzero_by_component = {}
    trainable_by_component = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = name.split(".", 1)[0]
        trainable_by_component[component] = trainable_by_component.get(component, 0) + 1
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            nonfinite.append(name)
        if bool(torch.count_nonzero(parameter.grad).item()):
            nonzero_by_component[component] = nonzero_by_component.get(component, 0) + 1
    if sorted(missing) != sorted(EXPECTED_UNUSED_TRAINABLE):
        raise RuntimeError(
            "Native-Crop gradient graph has unexpected missing trainable parameters: "
            f"{missing}"
        )
    if nonfinite:
        raise FloatingPointError(
            f"Native-Crop precheck has non-finite gradients: {nonfinite}"
        )
    for component in ("backbone", "projection", "rpn_head"):
        if trainable_by_component.get(component, 0) <= 0:
            raise RuntimeError(f"Native-Crop precheck found no trainable {component}")
        if nonzero_by_component.get(component, 0) <= 0:
            raise RuntimeError(
                f"Native-Crop detector loss produced no nonzero {component} gradient"
            )
    return {
        "trainable_parameter_tensors_by_component": trainable_by_component,
        "nonzero_gradient_tensors_by_component": nonzero_by_component,
        "expected_unused_trainable_parameters": sorted(missing),
        "all_present_gradients_finite": True,
    }


def run_full_precheck(
    *,
    config_path: Path,
    expected_commit: str,
    device_text: str,
    amp: bool,
    manifest_path: Path,
    geometry_census_path: Path,
    annotation_path: Path,
    class_map_path: Path,
    video_root: Path,
) -> dict:
    import torch
    import torch.nn.functional as functional

    importlib.import_module("opentad.datasets")
    importlib.import_module("opentad.models.backbones")
    from opentad.models import build_detector

    cfg = Config.fromfile(str(config_path))
    code_provenance = audit_code_provenance(expected_commit=expected_commit)
    config_audit = validate_native_crop_config(cfg, config_path=config_path)
    geometry_census_audit = audit_geometry_census(
        geometry_census_path,
        manifest_path=manifest_path,
        video_root=video_root,
    )
    dataset_audit = audit_development_dataset(
        cfg,
        manifest_path=manifest_path,
        annotation_path=annotation_path,
        class_map_path=class_map_path,
        video_root=video_root,
    )
    model_cfg = copy.deepcopy(cfg.model)
    checkpoint_path = ROOT / str(model_cfg.backbone.custom.pretrain)
    if checkpoint_path.name != NATIVE_CROP_PRETRAINED_FILENAME:
        raise ValueError("Native-Crop config changed the pretrained checkpoint identity")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != NATIVE_CROP_PRETRAINED_SHA256:
        raise ValueError("Native-Crop pretrained checkpoint SHA-256 mismatch")

    model = build_detector(model_cfg)
    pretrained_load_audit = audit_loaded_pretrained_state(
        model, checkpoint_path
    )
    device = torch.device(device_text)
    model = model.to(device).train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(3407)
    inputs = {
        "global": torch.randint(
            0,
            256,
            (1, 1, 3, 768, 96, 96),
            dtype=torch.uint8,
            generator=generator,
        ).to(device),
        "local": torch.randint(
            0,
            256,
            (1, 1, 3, 768, 128, 128),
            dtype=torch.uint8,
            generator=generator,
        ).to(device),
    }
    if any(value.device != device for value in inputs.values()):
        raise RuntimeError("Native-Crop synthetic views did not move to model device")
    masks = torch.ones((1, 768), device=device, dtype=torch.bool)
    metas = [
        {
            "video_name": "native_crop_s1_development_precheck",
            "fps": 30.0,
            "duration": 25.6,
            "snippet_stride": 1,
            "window_start_frame": 0,
            "window_size": 768,
            "offset_frames": 0,
            "native_crop_geometry": {
                "policy": "fixed_center",
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_test_evidence": False,
            },
        }
    ]
    gt_segments = [
        torch.tensor([[128.0, 320.0]], device=device, dtype=torch.float32)
    ]
    gt_labels = [torch.tensor([0], device=device, dtype=torch.long)]

    captured = {}

    def capture_projection(_module, args):
        captured["projection_input_shape"] = list(args[0].shape)

    projection_hook = model.projection.register_forward_pre_hook(capture_projection)
    branch_features = {}

    def capture_fusion_inputs(_module, args):
        if len(args) != 2:
            raise RuntimeError("Native-Crop fusion did not receive two branches")
        branch_features["global"] = args[0]
        branch_features["local"] = args[1]
        args[0].retain_grad()
        args[1].retain_grad()

    fusion_hook = model.backbone.fusion.register_forward_pre_hook(
        capture_fusion_inputs
    )
    interpolation_calls = []
    original_interpolate = functional.interpolate

    def observed_interpolate(value, *args, **kwargs):
        size = kwargs.get("size", args[0] if args else None)
        if value.ndim == 4 and tuple(value.shape[-2:]) == (14, 14):
            interpolation_calls.append(list(size))
        return original_interpolate(value, *args, **kwargs)

    started = time.perf_counter()
    functional.interpolate = observed_interpolate
    try:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            losses = model.forward_train(
                inputs,
                masks,
                metas,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
            )
            cost = losses["cost"]
        if not bool(torch.isfinite(cost).item()):
            raise FloatingPointError("Native-Crop full-model cost is non-finite")
        cost.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        functional.interpolate = original_interpolate
        projection_hook.remove()
        fusion_hook.remove()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if captured.get("projection_input_shape") != [1, 384, 768]:
        raise RuntimeError(
            "Native-Crop did not preserve the AdaTAD detector feature contract"
        )
    if interpolation_calls != [[6, 6], [8, 8]]:
        raise RuntimeError(
            "Native-Crop runtime position interpolation was not exactly global96/local128: "
            f"{interpolation_calls}"
        )
    wrapper_audit = model.backbone.latest_native_crop_audit
    if (
        wrapper_audit is None
        or wrapper_audit["shared_backbone_instances"] != 1
        or wrapper_audit["intermediate_shape"] != [1, 384, 384]
        or wrapper_audit["output_shape"] != [1, 384, 768]
    ):
        raise RuntimeError("Native-Crop shared-backbone audit is incomplete")
    branch_gradient_audit = {}
    for branch_name in ("global", "local"):
        feature = branch_features.get(branch_name)
        gradient = None if feature is None else feature.grad
        if (
            gradient is None
            or not bool(torch.isfinite(gradient).all().item())
            or not bool(torch.count_nonzero(gradient).item())
        ):
            raise RuntimeError(
                f"Native-Crop detector loss did not reach {branch_name} features"
            )
        branch_gradient_audit[branch_name] = {
            "shape": list(gradient.shape),
            "finite": True,
            "nonzero_elements": int(torch.count_nonzero(gradient).item()),
        }
    gradient_audit = _gradient_audit(model)
    result = {
        "schema_version": NATIVE_CROP_PRECHECK_SCHEMA,
        "status": "PASS",
        "expected_commit": expected_commit,
        "config_path": str(config_path.resolve()),
        "code_provenance": code_provenance,
        "config_audit": config_audit,
        "geometry_census_audit": geometry_census_audit,
        "dataset_audit": dataset_audit,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "pretrained_checkpoint_loaded": True,
        "pretrained_load_audit": pretrained_load_audit,
        "device": device_text,
        "amp": bool(amp),
        "train_cost": float(cost.detach().float().cpu().item()),
        "diagnostic_elapsed_ms": elapsed_ms,
        "paper_latency_claim_allowed": False,
        "projection_input_shape": captured["projection_input_shape"],
        "position_interpolation_calls": interpolation_calls,
        "backbone_audit": wrapper_audit,
        "branch_gradient_audit": branch_gradient_audit,
        "gradient_audit": gradient_audit,
        "cost_schema": build_cost_schema(global_size=96, local_size=128),
        "official_test_annotation_records_loaded": 0,
        "official_test_video_files_opened": 0,
        "teacher_used": False,
        "oracle_used": False,
        "paper_claim_allowed": False,
    }
    if device.type == "cuda":
        result["peak_gpu_memory_bytes_diagnostic"] = int(
            torch.cuda.max_memory_allocated(device)
        )
    return finalize_self_hash(result, "precheck_sha256")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="No-training full-model gate for Native-Crop S1"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/adatad/thumos/native_crop_s1_center_videomae_s_768x1_adapter.py",
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry-census", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_full_precheck(
        config_path=args.config,
        expected_commit=args.expected_commit,
        device_text=args.device,
        amp=args.amp,
        manifest_path=args.manifest,
        geometry_census_path=args.geometry_census,
        annotation_path=args.annotation,
        class_map_path=args.class_map,
        video_root=args.video_root,
    )
    _atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                "precheck_sha256": report["precheck_sha256"],
                "projection_input_shape": report["projection_input_shape"],
                "official_test_annotation_records_loaded": 0,
                "official_test_video_files_opened": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
