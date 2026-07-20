import json
import importlib.util
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.continuous_roi_s2_contract import canonical_sha256, load_protocol
from tools.bata.continuous_roi_s2_training import (
    S2_FULL_MODEL_GATE_SCHEMA,
    S2_TRAINING_RUNTIME_PRECHECK_SCHEMA,
    _bind_deterministic_temporal_upsampling,
    should_save_final_checkpoint,
    validate_bound_training_config,
    validate_full_model_gate,
    validate_training_runtime_precheck,
)
from tools.bata.run_continuous_roi_s2_one_step_gate import (
    AUDITED_SOURCE_PATHS,
)
from tools.bata.spatial_zoom_s1_contract import sha256_file

_GUARD_SPEC = importlib.util.spec_from_file_location(
    "continuous_roi_s2_training_guard_under_test",
    Path("opentad/utils/training_guard.py").resolve(),
)
assert _GUARD_SPEC is not None and _GUARD_SPEC.loader is not None
_GUARD_MODULE = importlib.util.module_from_spec(_GUARD_SPEC)
_GUARD_SPEC.loader.exec_module(_GUARD_MODULE)
assert_safe_cfg_options_for_gated_config = (
    _GUARD_MODULE.assert_safe_cfg_options_for_gated_config
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_gate(commit: str) -> dict:
    protocol_sha256 = load_protocol()["declared_protocol_sha256"]
    protocol_audit = {
        "static_protocol_valid": True,
        "official_test_open_allowed": False,
        "protocol_sha256": protocol_sha256,
        "check_count": 1,
        "state_assignments_checked": 128,
        "training_authorized": False,
    }
    protocol_audit["audit_sha256"] = canonical_sha256(protocol_audit)
    implementation_audit = {
        "status": "PASS",
        "official_test_materialized": False,
        "protocol_sha256": protocol_sha256,
        "config_hashes": {
            family: "a" * 64 for family in ("D160", "G96", "U128")
        },
        "pipeline_audits": {
            family: {} for family in ("D160", "G96", "U128")
        },
        "detector_model_surface_matches_reference": True,
        "post_processing_matches_reference": True,
        "u128_selector_parameters": 0,
        "u128_new_parameters": 609449,
        "training_authorized": False,
        "full_model_cuda_gate_required": True,
    }
    implementation_audit["implementation_audit_sha256"] = canonical_sha256(
        implementation_audit
    )
    payload = {
        "schema_version": S2_FULL_MODEL_GATE_SCHEMA,
        "status": "PASS",
        "expected_commit": commit,
        "code_provenance": {
            "git_commit": commit,
            "complete_worktree_clean": True,
            "audited_source_sha256": {
                path: sha256_file(path) for path in AUDITED_SOURCE_PATHS
            },
        },
        "protocol_audit": protocol_audit,
        "implementation_audit": implementation_audit,
        "optimizer_step_completed": True,
        "training_external_geometry_rejected": True,
        "official_test_annotation_records_loaded": 0,
        "official_test_video_files_opened": 0,
        "projection_input_shape": [1, 384, 768],
        "checkpoint_sha256": load_protocol()["data"][
            "videomae_s_checkpoint_sha256"
        ],
        "detector_only_gradient_audit": {
            "missing_target_gradients": [],
            "all_present_gradients_finite": True,
        },
        "total_gradient_audit": {
            "missing_target_gradients": [],
            "all_present_gradients_finite": True,
        },
        "optimizer_audit": {
            "requires_grad_parameter_tensors": 3,
            "optimizer_parameter_tensors": 3,
            "every_requires_grad_parameter_exactly_once": True,
            "frozen_parameters_excluded": True,
        },
        "cuda_device_identity": {
            "logical_device": "cuda:0",
            "slurm_job_id": "123",
            "slurm_step_id": "123.0",
            "cuda_visible_device_uuid": "GPU-test",
        },
        "backbone_audit": {
            "shared_backbone_instances": 1,
            "videomae_evaluations": 2,
            "contains_selector": False,
        },
        "sampler_geometry_gradient_audit": {
            "runtime_bitwise_parity": True,
        },
        "training_runtime_gate_authorized_by_this_gate": True,
        "formal_training_authorized_by_this_gate": False,
    }
    payload["gate_sha256"] = canonical_sha256(payload)
    return payload


def test_source_config_is_not_directly_trainable():
    cfg = Config.fromfile(
        "configs/adatad/thumos/"
        "continuous_roi_s2_u128_videomae_s_768x1_adapter.py"
    )
    with pytest.raises(ValueError, match="not directly trainable"):
        validate_bound_training_config(cfg, seed=3407)


def test_s2_source_config_rejects_every_cfg_override_before_merge():
    cfg = Config.fromfile(
        "configs/adatad/thumos/"
        "continuous_roi_s2_u128_videomae_s_768x1_adapter.py"
    )
    attacks = (
        {"continuous_roi_s2_gate.allow_tools_train": True},
        {"continuous_roi_s2_gate.allow_detector_training": True},
        {"dataset.test.data_path": "/sealed/test"},
        {"workflow.end_epoch": 999},
    )
    for options in attacks:
        with pytest.raises(RuntimeError, match="rejected all --cfg-options"):
            assert_safe_cfg_options_for_gated_config(cfg, options)


@pytest.mark.parametrize("family", ("d160", "g96"))
def test_dense_comparators_use_deterministic_exact_2x_temporal_upsampling(
    family,
):
    cfg = Config.fromfile(
        "configs/adatad/thumos/"
        f"continuous_roi_s2_{family}_videomae_s_768x1_adapter.py"
    )
    audit = _bind_deterministic_temporal_upsampling(
        cfg, family=family.upper()
    )
    transforms = cfg.model.backbone.custom.post_processing_pipeline
    interpolate = [
        transform
        for transform in transforms
        if transform["type"] == "Interpolate"
    ]
    assert interpolate == [
        {
            "type": "Interpolate",
            "keys": ["feats"],
            "size": 768,
            "mode": "linear",
            "deterministic": True,
            "expected_input_size": 384,
        }
    ]
    assert audit == {
        "implementation": "explicit_linear_2x_no_cuda_atomics",
        "input_length": 384,
        "output_length": 768,
        "align_corners": False,
    }


def test_full_model_gate_validator_is_self_hash_and_commit_bound(tmp_path):
    commit = "a" * 40
    gate_path = _write_json(tmp_path / "gate.json", _valid_gate(commit))
    assert validate_full_model_gate(
        gate_path, expected_commit=commit
    )["status"] == "PASS"
    with pytest.raises(ValueError, match="commit provenance"):
        validate_full_model_gate(gate_path, expected_commit="b" * 40)
    incomplete = _valid_gate(commit)
    incomplete["optimizer_audit"] = {}
    incomplete["gate_sha256"] = canonical_sha256(
        {key: value for key, value in incomplete.items() if key != "gate_sha256"}
    )
    incomplete_path = _write_json(tmp_path / "incomplete_gate.json", incomplete)
    with pytest.raises(ValueError, match="evidence is incomplete"):
        validate_full_model_gate(incomplete_path, expected_commit=commit)


def test_runtime_precheck_validator_is_gate_bound(tmp_path):
    commit = "a" * 40
    payload = {
        "schema_version": S2_TRAINING_RUNTIME_PRECHECK_SCHEMA,
        "status": "PASS",
        "code_commit": commit,
        "full_model_gate_sha256": "c" * 64,
        "all_nine_bindings_valid": True,
        "all_nine_config_dump_reload_valid": True,
        "train_batches_per_epoch": 80,
        "development_gate_window_count": 129,
        "development_gate_window_identity_sha256": "e" * 64,
        "official_test_annotation_records_loaded": 0,
        "official_test_video_files_opened": 0,
        "official_test_open_allowed": False,
        "learned_roi_policy_present": False,
        "paper_claim_allowed": False,
        "slurm_job_id": "123",
        "slurm_step_id": "123.0",
        "slurm_step_gpu_identity": "4",
        "slurm_cpus_per_task": 5,
        "effective_memory_limit_mb": 96000,
        "cuda_visible_devices": "0",
        "bindings": [
            {
                "family": family,
                "seed": seed,
                "work_dir": f"/tmp/{family}/{seed}",
                "bound_config_sha256": "f" * 64,
                "successful_updates": 4800,
                "updates_per_epoch": 80,
                "checkpoint_selection": "final_ema_only",
                "config_dump_reload_valid": True,
            }
            for family in ("D160", "G96", "U128")
            for seed in (3407, 3408, 3409)
        ],
        "family_runtime": {
            "D160": {
                "fit_video_count": 160,
                "fit_sample_count": 160,
                "development_gate_video_count": 40,
                "development_gate_window_count": 129,
                "train_batches_per_epoch": 80,
                "gate_window_identity_sha256": "1" * 64,
                "sample_audit": {
                    "geometry_policy": "full_frame_letterbox",
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_oracle": False,
                    "uses_test_evidence": False,
                },
                "real_training_batch_audit": {
                    "batch_size": 2,
                    "input_shapes": {
                        "dense": [2, 1, 3, 768, 160, 160]
                    },
                    "mask_shape": [2, 768],
                    "uses_gt_for_geometry": False,
                    "uses_teacher": False,
                    "uses_oracle": False,
                    "uses_test_evidence": False,
                },
            },
            "G96": {
                "fit_video_count": 160,
                "fit_sample_count": 160,
                "development_gate_video_count": 40,
                "development_gate_window_count": 129,
                "train_batches_per_epoch": 80,
                "gate_window_identity_sha256": "1" * 64,
                "sample_audit": {
                    "geometry_policy": "full_frame_letterbox",
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_oracle": False,
                    "uses_test_evidence": False,
                },
                "real_training_batch_audit": {
                    "batch_size": 2,
                    "input_shapes": {
                        "dense": [2, 1, 3, 768, 96, 96]
                    },
                    "mask_shape": [2, 768],
                    "uses_gt_for_geometry": False,
                    "uses_teacher": False,
                    "uses_oracle": False,
                    "uses_test_evidence": False,
                },
            },
            "U128": {
                "fit_video_count": 160,
                "fit_sample_count": 160,
                "development_gate_video_count": 40,
                "development_gate_window_count": 129,
                "train_batches_per_epoch": 80,
                "gate_window_identity_sha256": "1" * 64,
                "sample_audit": {
                    "geometry_policy": "none_pre_policy_source",
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_oracle": False,
                    "uses_test_evidence": False,
                },
                "real_training_batch_audit": {
                    "batch_size": 2,
                    "input_shapes": {
                        "global": [2, 1, 3, 768, 96, 96],
                        "source": [2, 1, 3, 768, 180, 320],
                        "sample_key": [2],
                        "window_start": [2],
                    },
                    "mask_shape": [2, 768],
                    "uses_gt_for_geometry": False,
                    "uses_teacher": False,
                    "uses_oracle": False,
                    "uses_test_evidence": False,
                },
            },
        },
        "development_video_inventory": {
            "video_count": 200,
            "inventory_sha256": "2" * 64,
            "symlinks_allowed": False,
            "path_escape_allowed": False,
        },
        "development_video_census": {
            "video_count": 200,
            "records_sha256": "3" * 64,
            "all_videos_ffprobe_decodable": True,
            "sealed_test_files_probed": 0,
            "annotation_or_gt_read": False,
        },
    }
    payload["precheck_sha256"] = canonical_sha256(payload)
    path = _write_json(tmp_path / "precheck.json", payload)
    assert validate_training_runtime_precheck(
        path,
        expected_commit=commit,
        expected_full_model_gate_sha256="c" * 64,
    )["status"] == "PASS"
    with pytest.raises(ValueError, match="lacks a PASS invariant"):
        validate_training_runtime_precheck(
            path,
            expected_commit=commit,
            expected_full_model_gate_sha256="d" * 64,
        )
    incomplete = dict(payload)
    incomplete["family_runtime"] = {
        "D160": {},
        "G96": {},
        "U128": {},
    }
    incomplete["precheck_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in incomplete.items()
            if key != "precheck_sha256"
        }
    )
    incomplete_path = _write_json(
        tmp_path / "incomplete_precheck.json", incomplete
    )
    with pytest.raises(ValueError, match="lacks a PASS invariant"):
        validate_training_runtime_precheck(
            incomplete_path,
            expected_commit=commit,
            expected_full_model_gate_sha256="c" * 64,
        )


def test_s2_saves_only_the_final_ema_checkpoint():
    binding = {"epochs": 60}
    assert not should_save_final_checkpoint(0, binding)
    assert not should_save_final_checkpoint(58, binding)
    assert should_save_final_checkpoint(59, binding)


def test_training_launcher_requires_both_gate_certificates():
    launcher = Path(
        "scripts/run_continuous_roi_s2_train_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "CONTINUOUS_ROI_S2_FULL_MODEL_GATE" in launcher
    assert "CONTINUOUS_ROI_S2_TRAINING_RUNTIME_PRECHECK" in launcher
    assert "CONTINUOUS_ROI_S2_RUNTIME_AUTHORIZATION" in launcher
    assert "--training-runtime-precheck" in launcher
    assert "srun --exact" in launcher
    assert "--gpus=1" in launcher
    assert "--cpus-per-task=5" in launcher
    assert "--mem=96000M" in launcher
    assert "export CUDA_VISIBLE_DEVICES=" not in launcher
    assert "tools/train.py" in launcher


def test_cuda_gate_runs_real_development_runtime_precheck():
    launcher = Path(
        "scripts/run_continuous_roi_s2_cuda_gate_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "precheck_continuous_roi_s2_training_runtime.py" in launcher
    assert "training_runtime_precheck.json" in launcher
    assert "training_runtime_authorization.json" in launcher
    assert "build_continuous_roi_s2_runtime_gate_config.py" in launcher
    assert launcher.count("torchrun --nnodes=1 --nproc_per_node=1") >= 1
    assert "CONTINUOUS_ROI_S2_DEVELOPMENT_ANNOTATION" in launcher
    assert "CONTINUOUS_ROI_S2_DEVELOPMENT_VIDEO_ROOT" in launcher


def test_runtime_gate_uses_mmengine_supported_config_removal():
    source = Path(
        "tools/bata/continuous_roi_s2_runtime_gate.py"
    ).read_text(encoding="utf-8")
    assert 'cfg.pop("continuous_roi_s2_runtime_binding")' in source
    assert 'del cfg["continuous_roi_s2_runtime_binding"]' not in source


def test_train_entrypoint_enforces_s2_runtime_binding_and_80_batches():
    source = Path("tools/train.py").read_text(encoding="utf-8")
    assert '"continuous_roi_s2_runtime_binding" in cfg' in source
    assert "validate_bound_training_config" in source
    assert "exactly 80 batches per epoch" in source
    assert "build_s2_checkpoint_metadata" in source


def test_deployment_uses_site_memory_allocation_and_exact_inner_step():
    deployer = Path(
        "tools/bata/deploy_continuous_roi_s2_training_matrix.py"
    ).read_text(encoding="utf-8")
    assert '"--gpus=2"' in deployer
    assert '"--cpus-per-task=8"' in deployer
    assert '"gpus": 1' in deployer
    assert '"cpus": 5' in deployer
    assert '"memory_mib": 96000' in deployer
    assert '"overrides_cuda_visible_devices": False' in deployer
    assert "CONTINUOUS_ROI_S2_TRAINING_RUNTIME_PRECHECK" in deployer
    assert "CONTINUOUS_ROI_S2_RUNTIME_AUTHORIZATION" in deployer
    assert "--comment=crs2:" in deployer
    assert "cell intent exists but its Slurm job is not visible" in deployer
    assert "official_test_open_allowed" in deployer
