from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.bata import finalize_georoute_estimator_pilot as pilot_finalizer
from tools.bata import finalize_georoute_estimator_pilot_p0 as pilot_p0_finalizer
from tools.bata import georoute_estimator_pilot_stage_runner as pilot_stage
from tools.bata.georoute_estimator_pilot_contract import (
    PILOT_ARM_ORDER,
    PILOT_ARMS,
    PILOT_CONTRACT_SCHEMA,
    PILOT_CONTRASTS,
    PILOT_EPOCHS,
    PILOT_K,
    PILOT_P0_SUITE_SCHEMA,
    PILOT_P0_FAILURE_SCHEMA,
    PILOT_SEED,
    PILOT_STAGE_RESULT_SCHEMA,
    PILOT_STUDY_ID,
    REPRESENTATION_KEYS,
    bind_pilot_config,
    pilot_arm_spec,
    pilot_cell_relative_path,
    validate_pilot_job_receipt,
)
from tools.bata.georoute_estimator_pilot_stage_runner import (
    summarize_pilot_telemetry,
    validate_pilot_stage_result,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file
from tools.bata.georoute_stage_runner import build_torchrun_prefix
from tools.bata.run_georoute_p0_gate import (
    GEOROUTE_P0_GATE_SCHEMA,
    build_p0_gate_report,
    validate_p0_gate_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _p0_payload(arm: str) -> dict:
    spec = pilot_arm_spec(arm)
    claim = {
        "none": "no_policy_gradient",
        "straight_through": "biased_straight_through",
        "score_function": "score_function_candidate",
    }[spec["policy_estimator"]]
    required = {
        "rpn_head",
        "projection",
        "sparse_adapter",
        "videomae_adapter",
    }
    if spec["learned_geometry_enabled"]:
        required.add("scout_geometry")
    if spec["learned_residual_enabled"]:
        required.add("scout_residual")
    representation = {
        "absolute_position_enabled": True,
        "geometry_side_channel": spec["geometry_side_channel"],
        "learned_geometry_enabled": spec["learned_geometry_enabled"],
        "learned_residual_enabled": spec["learned_residual_enabled"],
        **{key: spec[key] for key in REPRESENTATION_KEYS},
    }
    return {
        "schema_version": GEOROUTE_P0_GATE_SCHEMA,
        "status": "PASS",
        "official_test_opened": False,
        "heavy_backbone_forward_count": 1,
        "shared_backbone_instances": 1,
        "uses_grid_sample": False,
        "uses_resized_local_crop": False,
        "exact_k": {
            "target_k": PILOT_K,
            "observed_min": PILOT_K,
            "observed_max": PILOT_K,
            "duplicates": 0,
        },
        "estimator": {
            "name": spec["policy_estimator"],
            "claim": claim,
        },
        "score_function_amp_horizon": (
            {
                "status": "PASS_AMP_PRODUCTION_HORIZON",
                "passed": True,
                "source_dtype": "torch.float16",
                "likelihood_dtype": "torch.float32",
                "policy_loss_dtype": "torch.float32",
                "tubelets": 384,
                "patch_capacity": 220,
                "target_k": PILOT_K,
                "loss_scale": 256.0,
                "all_likelihoods_finite": True,
                "policy_loss_finite": True,
                "all_scaled_gradients_finite": True,
                "policy_loss_abs": 100000.0,
                "fp16_max": 65504.0,
            }
            if spec["policy_estimator"] == "score_function"
            else {
                "status": "NOT_APPLICABLE_NON_SCORE_FUNCTION",
                "executed": False,
            }
        ),
        "score_function_full_graph_amp": (
            {
                "status": "PASS_FULL_GRAPH_AMP_OPTIMIZER_UPDATE",
                "executed": True,
                "autocast_dtype": "torch.float16",
                "loss_scale_before": 256.0,
                "loss_scale_after": 256.0,
                "optimizer": "sgd_lr_zero_overflow_probe",
                "optimizer_update_succeeded": True,
                "all_required_gradients_finite": True,
                "scout_autocast_enabled": False,
                "scout_compute_dtype": "torch.float32",
                "model_backward_scope": "detector_plus_score_function",
            }
            if spec["policy_estimator"] == "score_function"
            else {
                "status": "NOT_APPLICABLE_NON_SCORE_FUNCTION",
                "executed": False,
            }
        ),
        "pilot_arm": arm,
        "route_mode": spec["route_mode"],
        "route_parameters": {
            "context_tokens": spec["context_tokens"],
            "roi_fraction": spec["roi_fraction"],
            "policy_temperature": spec["policy_temperature"],
            "score_function_weight": spec["score_function_weight"],
            "score_function_baseline_momentum": spec["score_function_baseline_momentum"],
        },
        "representation": representation,
        "memory": {
            "peak_allocated_bytes": 4096,
            "peak_reserved_bytes": 8192,
        },
        "losses": {"cls_loss": 1.0, "reg_loss": 1.0},
        "gradient": {
            "all_required_gradients_finite": True,
            "nonzero_components": sorted(required),
            "required_components": sorted(required),
            "missing_required_components": [],
        },
        "detector": {
            "training_forward": True,
            "backward_completed": True,
            "output_length": 768,
            "detector_loss_keys": ["cls_loss", "reg_loss"],
        },
        "source_grid": {
            "height": 180,
            "width": 320,
            "patch_size": 16,
            "grid_height": 11,
            "grid_width": 20,
            "patch_capacity": 220,
        },
        "native_route": {
            "selected_native_tubelet_shape": [
                1,
                384,
                PILOT_K,
                3,
                2,
                16,
                16,
            ],
            "output_shape": [1, 384, 768],
            "selected_unique_count_min": PILOT_K,
            "selected_unique_count_max": PILOT_K,
            "native_packed_invocation_counter_before": 4,
            "native_packed_invocation_counter_after": 5,
        },
        "dense_native_reference": None,
        "score_function_detector_binding": ({"detector_loss_keys": ["cls_loss", "reg_loss"]} if spec["policy_estimator"] == "score_function" else None),
        "component_trace": {
            "packed_attention_forward_count": 12,
            "packed_mlp_forward_count": 12,
            "packed_adapter_forward_count": 12,
            "dense_adapter_forward_count": 0,
            "adapter_execution": "coordinate_lineage_packed",
        },
        "checkpoint_receipt": {
            "checkpoint_count": 0,
            "policy": "p0_no_checkpoint",
        },
        "storage_receipt": {
            "status": "PASS_STORAGE_PREFLIGHT",
            "atomic_publish_peak_included": True,
        },
        "runtime_commit": "a" * 40,
        "slurm_job_id": "1204001",
        "rendezvous_isolation": {
            "path": f"/data/run01/sczc063/yuzibo/p0/{arm}.rendezvous.json",
            "file_sha256": "b" * 64,
            "gate_sha256": "c" * 64,
            "slurm_job_id": "1204001",
            "status": "PASS_CONCURRENT_RENDEZVOUS_ISOLATION",
        },
        "checkpoint_storage_measurement": {
            "checkpoint_policy": "final_only",
            "checkpoint_upper_bound_bytes": 4096,
            "peak_checkpoint_copies_per_cell": 1,
            "auxiliary_upper_bound_bytes_per_cell": 2048,
            "stage_fixed_overhead_bytes": 1024,
            "safety_fraction": 0.25,
            "safety_bytes": 1024,
            "measurement_method": "unit_test",
        },
        "p0_scope": {
            "synthetic_inputs_only": True,
            "full_training": False,
            "official_evaluation": False,
        },
    }


def _rehash_stage_result(result: dict) -> dict:
    result["stage_result_sha256"] = canonical_sha256({key: value for key, value in result.items() if key != "stage_result_sha256"})
    return result


def _rendezvous(arm: str, *, phase: str, job_id: str) -> dict:
    _, receipt = build_torchrun_prefix(
        phase=phase,
        slurm_job_id=job_id,
        stage="estimator_pilot",
        variant=arm,
        seed=PILOT_SEED,
    )
    return receipt


def _binding(arm: str, *, work_dir: str) -> dict:
    spec = pilot_arm_spec(arm)
    binding = {
        "schema_version": PILOT_CONTRACT_SCHEMA,
        "study_id": PILOT_STUDY_ID,
        "arm": arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": PILOT_SEED,
        "epochs": PILOT_EPOCHS,
        "source_config": "/repo/source.py",
        "source_config_sha256": "5" * 64,
        "manifest_path": "/data/manifest.json",
        "manifest_file_sha256": "6" * 64,
        "fit_video_ids": ["fit"],
        "gate_video_ids": ["gate"],
        "development_annotation": {
            "path": "/data/annotation.json",
            "sha256": "7" * 64,
        },
        "class_map_path": "/data/class_map.json",
        "class_map_sha256": "8" * 64,
        "development_video_root": "/data/run01/sczc063/yuzibo/videos",
        "pretrained_checkpoint_path": "/data/pretrained.pth",
        "pretrained_checkpoint_sha256": "9" * 64,
        "work_dir": work_dir,
        "single_seed_exploratory": True,
        "old_selector_reused": False,
        "selector_emitted": False,
        "p2_p3_opened": False,
        "official_test_opened": False,
        "manual_roi_used": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "raw_prediction_cache_used": False,
        "checkpoint_policy": "final_only_atomic",
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def _stage_result(
    arm: str,
    *,
    job_index: int = 0,
    work_dir: str | None = None,
) -> dict:
    spec = pilot_arm_spec(arm)
    job_id = str(1205000 + job_index)
    binding = _binding(
        arm,
        work_dir=work_dir or f"/data/run01/sczc063/yuzibo/{arm}",
    )
    audit = {
        "route_mode": spec["route_mode"],
        "policy_estimator": spec["policy_estimator"],
        "policy_temperature": spec["policy_temperature"],
        "score_function_weight": spec["score_function_weight"],
        "score_function_baseline_momentum": spec["score_function_baseline_momentum"],
        "geometry_smoothness_weight": spec["geometry_smoothness_weight"],
        "area_prior_weight": spec["area_prior_weight"],
        "pooling_mode": spec["pooling_mode"],
        "adapter_mode": spec["adapter_mode"],
        "target_k": PILOT_K,
        "selected_unique_count_min": PILOT_K,
        "selected_unique_count_max": PILOT_K,
        "selected_duplicate_count": 0,
        "heavy_backbone_forward_count": 1,
        "uses_grid_sample": False,
        "uses_resized_local_crop": False,
        "absolute_position_enabled": True,
        "geometry_side_channel": spec["geometry_side_channel"],
        "learned_geometry_enabled": spec["learned_geometry_enabled"],
        "learned_residual_enabled": spec["learned_residual_enabled"],
        **{key: spec[key] for key in REPRESENTATION_KEYS},
    }
    metrics = {
        "average_mAP": 10.0 + job_index,
        "mAP@0.3": 14.0 + job_index,
        "mAP@0.4": 13.0 + job_index,
        "mAP@0.5": 12.0 + job_index,
        "mAP@0.6": 11.0 + job_index,
        "mAP@0.7": 9.0 + job_index,
        "high_iou_composite": 10.0 + job_index,
    }
    result = {
        "schema_version": PILOT_STAGE_RESULT_SCHEMA,
        "status": "PASS_EXPLORATORY_DEVELOPMENT_ONLY",
        "study_id": PILOT_STUDY_ID,
        "experiment_schema_version": "pilot",
        "arm": arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": PILOT_SEED,
        "epochs": PILOT_EPOCHS,
        "token_budget": PILOT_K,
        "metrics": metrics,
        "profile": {
            "sample_count": 136,
            "steady_sample_count": 131,
            "loader_wait_p50_ms": 10.0,
            "loader_wait_p95_ms": 20.0,
            "model_and_postprocess_p50_ms": 30.0 + job_index,
            "model_and_postprocess_p95_ms": 40.0 + job_index,
            "window_wall_p50_ms": 50.0,
            "window_wall_p95_ms": 60.0,
            "peak_allocated_mb": 1000.0,
            "paper_grade_end_to_end_claim_allowed": False,
            "profile_file_sha256": "a" * 64,
            "scope": {
                "development_only": True,
                "evaluator_excluded": True,
                "paper_grade_end_to_end_claim_allowed": False,
            },
        },
        "telemetry_summary": {
            "dataset_count": 136,
            "record_count": 136,
            "population_sha256": "1" * 64,
            "population_descriptor_sha256": "2" * 64,
            "telemetry_file_sha256": "3" * 64,
        },
        "routing_audit": audit,
        "binding": binding,
        "binding_sha256": binding["binding_sha256"],
        "config_path": f"/data/run01/sczc063/yuzibo/{arm}/bound.py",
        "config_sha256": "d" * 64,
        "input_receipts": {
            "source_config_sha256": "5" * 64,
            "manifest_file_sha256": "6" * 64,
            "development_annotation_sha256": "7" * 64,
            "class_map_sha256": "8" * 64,
            "pretrained_checkpoint_sha256": "9" * 64,
            "development_video_root": "/data/run01/sczc063/yuzibo/videos",
        },
        "checkpoint_receipt": {
            "path": f"/data/run01/sczc063/yuzibo/{arm}/epoch_19.pth",
            "sha256": "a" * 64,
            "size_bytes": 1024,
            "policy": "final_only_atomic",
        },
        "storage_receipt": {"status": "PASS_STORAGE_PREFLIGHT"},
        "prediction_path": (f"/data/run01/sczc063/yuzibo/{arm}/result_detection.json"),
        "prediction_sha256": "e" * 64,
        "profile_path": (f"/data/run01/sczc063/yuzibo/{arm}/georoute_development_profile.json"),
        "telemetry_path": (f"/data/run01/sczc063/yuzibo/{arm}/georoute_diagnostic_telemetry.json"),
        "test_log_path": f"/data/run01/sczc063/yuzibo/{arm}/test.out",
        "test_log_sha256": "f" * 64,
        "runtime_commit": "a" * 40,
        "rendezvous": {
            "isolation_policy": ("job_scoped_loopback_kernel_assigned_endpoint_and_unique_cell_phase_id"),
            "train": _rendezvous(arm, phase="train", job_id=job_id),
            "test": _rendezvous(arm, phase="test", job_id=job_id),
        },
        "parent_p0_suite": {
            "path": "/data/run01/sczc063/yuzibo/pilot_p0_suite.json",
            "file_sha256": "b" * 64,
            "suite_sha256": "c" * 64,
        },
        "single_seed_exploratory": True,
        "old_selector_reused": False,
        "selector_emitted": False,
        "p2_p3_opened": False,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "raw_prediction_cache_used": False,
        "manual_roi_used": False,
        "paper_grade_result_record_emitted": False,
        "paper_claim_allowed": False,
    }
    result["telemetry_summary"]["summary_sha256"] = canonical_sha256(result["telemetry_summary"])
    return _rehash_stage_result(result)


def _telemetry_payload(arm: str, *, population_shift: float = 0.0) -> dict:
    records = []
    population_rows = []
    role_counts = {
        "context": 0,
        "roi": PILOT_K if pilot_arm_spec(arm)["route_mode"] == "roi" else 0,
        "residual": 0,
        "free": PILOT_K if pilot_arm_spec(arm)["route_mode"] == "free" else 0,
        "dense": 0,
        "uniform": (PILOT_K if pilot_arm_spec(arm)["route_mode"] == "uniform" else 0),
        "random": 0,
    }
    for index in range(2):
        descriptor = {
            "dataset_index": index,
            "video_id": f"video_{index}",
            "window_center_count": 2,
            "window_center_first": float(index * 10),
            "window_center_last": float(index * 10 + 5) + (population_shift if index == 1 else 0.0),
        }
        descriptor_sha256 = hashlib.sha256(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        population_row = {
            **descriptor,
            "window_descriptor_sha256": descriptor_sha256,
        }
        population_rows.append(population_row)
        records.append(
            {
                **population_row,
                "route": {
                    "selected_index_sha256": hashlib.sha256(f"{arm}-{index}".encode("utf-8")).hexdigest(),
                    "role_counts": role_counts,
                },
            }
        )
    population_sha256 = hashlib.sha256(
        json.dumps(
            population_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "georoute_diagnostic_telemetry_v1",
        "development_only": True,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "oracle_used": False,
        "raw_prediction_cache_used": False,
        "dataset_count": len(records),
        "record_count": len(records),
        "population_sha256": population_sha256,
        "records": records,
    }


def _materialize_stage_result(
    *,
    run_root: Path,
    parent_path: Path,
    parent_payload: dict,
    arm: str,
    job_index: int,
    population_shift: float = 0.0,
) -> dict:
    cell = run_root / pilot_cell_relative_path(
        arm=arm,
        seed=PILOT_SEED,
    )
    cell.mkdir(parents=True)
    bound_config = run_root / "control" / "bound_configs" / f"{arm}_seed{PILOT_SEED}.py"
    bound_config.parent.mkdir(parents=True, exist_ok=True)
    bound_config.write_text(f"arm = {arm!r}\n", encoding="utf-8")
    prediction = cell / "result_detection.json"
    prediction.write_text("{}\n", encoding="utf-8")
    profile_path = cell / "georoute_development_profile.json"
    profile_payload = {
        "scope": {
            "development_only": True,
            "evaluator_excluded": True,
            "paper_grade_end_to_end_claim_allowed": False,
        },
        "sample_count": 2,
        "steady_sample_count": 2,
        "loader_wait_p50_ms": 10.0,
        "loader_wait_p95_ms": 20.0,
        "model_and_postprocess_p50_ms": 30.0 + job_index,
        "model_and_postprocess_p95_ms": 40.0 + job_index,
        "window_wall_p50_ms": 50.0,
        "window_wall_p95_ms": 60.0,
        "peak_allocated_mb": 1000.0,
    }
    profile_path.write_text(
        json.dumps(profile_payload),
        encoding="utf-8",
    )
    telemetry_path = cell / "georoute_diagnostic_telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            _telemetry_payload(arm, population_shift=population_shift),
        ),
        encoding="utf-8",
    )
    test_log = cell / "test.out"
    test_log.write_text("development-only fixture\n", encoding="utf-8")
    checkpoint = cell / "gpu1_id0" / "checkpoint" / "epoch_19.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint-{arm}".encode("utf-8"))

    result = _stage_result(
        arm,
        job_index=job_index,
        work_dir=str(cell.resolve()),
    )
    result["config_path"] = str(bound_config.resolve())
    result["config_sha256"] = sha256_file(bound_config)
    result["prediction_path"] = str(prediction.resolve())
    result["prediction_sha256"] = sha256_file(prediction)
    result["profile_path"] = str(profile_path.resolve())
    result["profile"] = pilot_stage._pilot_profile(profile_path)
    result["telemetry_path"] = str(telemetry_path.resolve())
    result["telemetry_summary"] = summarize_pilot_telemetry(telemetry_path)
    result["test_log_path"] = str(test_log.resolve())
    result["test_log_sha256"] = sha256_file(test_log)
    result["checkpoint_receipt"] = {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "policy": "final_only_atomic",
    }
    result["parent_p0_suite"] = {
        "path": str(parent_path.resolve()),
        "file_sha256": sha256_file(parent_path),
        "suite_sha256": parent_payload["suite_sha256"],
    }
    _rehash_stage_result(result)
    validate_pilot_stage_result(
        result,
        expected_arm=arm,
        expected_commit="a" * 40,
    )
    (cell / "stage_result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
    return result


def _parent_p0_payload() -> dict:
    payload = {
        "schema_version": PILOT_P0_SUITE_SCHEMA,
        "status": "PASS_MECHANICAL_ONLY",
        "runtime_commit": "a" * 40,
        "study_id": PILOT_STUDY_ID,
        "arms": list(PILOT_ARM_ORDER),
        "official_test_opened": False,
        "training_completed": False,
        "paper_claim_allowed": False,
    }
    payload["suite_sha256"] = canonical_sha256(payload)
    return payload


def test_six_arm_contract_is_single_intervention_and_single_seed():
    assert PILOT_ARM_ORDER == (
        "residual_st_rep_off",
        "residual_pl_rep_off",
        "fixed_rep_off",
        "fixed_rep_on",
        "roi_pl_rep_off",
        "roi_pl_rep_on",
    )
    assert set(PILOT_CONTRASTS) == {
        "estimator_pl_minus_st_rep_off",
        "fixed_representation_on_minus_off",
        "roi_representation_on_minus_off",
        "roi_support_minus_residual_support_pl_rep_off",
    }
    for arm in PILOT_ARM_ORDER:
        spec = pilot_arm_spec(arm)
        assert spec["tokens_per_tubelet"] == PILOT_K
        assert spec["context_tokens"] == 0
        assert spec["absolute_position_enabled"] is True
        assert len({spec[key] for key in REPRESENTATION_KEYS}) == 1
        assert (
            pilot_cell_relative_path(
                arm=arm,
                seed=PILOT_SEED,
            ).parts[-1]
            == f"seed{PILOT_SEED}"
        )
    with pytest.raises(ValueError, match="frozen exploratory seed"):
        pilot_cell_relative_path(arm=PILOT_ARM_ORDER[0], seed=3408)


def test_bound_configs_materialize_all_representation_switches(tmp_path):
    manifest = tmp_path / "manifest.json"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.json"
    pretrained = tmp_path / "pretrained.pth"
    videos = tmp_path / "videos"
    videos.mkdir()
    manifest.write_text(
        json.dumps({"splits": {"fit": ["fit"], "gate": ["gate"]}}),
        encoding="utf-8",
    )
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "fit": {"subset": "training"},
                    "gate": {"subset": "training"},
                }
            }
        ),
        encoding="utf-8",
    )
    class_map.write_text("{}", encoding="utf-8")
    pretrained.write_bytes(b"test")
    source_config = ROOT / "configs" / "adatad" / "thumos" / "georoute_adatad_development_base.py"

    for arm in PILOT_ARM_ORDER:
        cfg = bind_pilot_config(
            source_config_path=source_config,
            arm=arm,
            seed=PILOT_SEED,
            work_dir=tmp_path / arm,
            manifest_path=manifest,
            development_annotation_path=annotation,
            class_map_path=class_map,
            development_video_root=videos,
            pretrained_checkpoint_path=pretrained,
        )
        spec = pilot_arm_spec(arm)
        custom = cfg.model.backbone.custom
        assert custom.georoute_route_mode == spec["route_mode"]
        assert custom.georoute_policy_estimator == spec["policy_estimator"]
        assert custom.georoute_absolute_position_enabled is True
        for key in REPRESENTATION_KEYS:
            assert getattr(custom, f"georoute_{key}") is spec[key]
        assert custom.georoute_geometry_side_channel is spec["geometry_side_channel"]
        assert cfg.scheduler.max_epoch == PILOT_EPOCHS
        assert cfg.georoute_estimator_pilot_binding.paper_claim_allowed is False


@pytest.mark.parametrize("arm", PILOT_ARM_ORDER)
def test_p0_validator_accepts_each_exact_pilot_arm_and_rejects_leakage(arm):
    report = build_p0_gate_report(_p0_payload(arm))
    validate_p0_gate_report(report)

    report["representation"]["absolute_coordinates_enabled"] = not report["representation"]["absolute_coordinates_enabled"]
    report = build_p0_gate_report(report)
    with pytest.raises(ValueError, match="representation"):
        validate_p0_gate_report(report)


@pytest.mark.parametrize("arm", PILOT_ARM_ORDER)
def test_stage_result_validator_binds_exact_arm_k_and_claim_boundary(arm):
    result = _stage_result(arm)
    validate_pilot_stage_result(
        result,
        expected_arm=arm,
        expected_commit="a" * 40,
    )

    result["token_budget"] = 32
    result["stage_result_sha256"] = canonical_sha256({key: value for key, value in result.items() if key != "stage_result_sha256"})
    with pytest.raises(ValueError, match="contract"):
        validate_pilot_stage_result(result, expected_arm=arm)


def test_stage_result_validator_rejects_binding_cost_and_test_job_tampering():
    arm = "residual_pl_rep_off"
    result = _stage_result(arm)
    result["profile"]["model_and_postprocess_p95_ms"] = float("inf")
    _rehash_stage_result(result)
    with pytest.raises(ValueError, match="cost profile"):
        validate_pilot_stage_result(result, expected_arm=arm)

    result = _stage_result(arm)
    result["binding"]["epochs"] = PILOT_EPOCHS + 1
    result["binding"]["binding_sha256"] = canonical_sha256({key: value for key, value in result["binding"].items() if key != "binding_sha256"})
    result["binding_sha256"] = result["binding"]["binding_sha256"]
    _rehash_stage_result(result)
    with pytest.raises(ValueError, match="immutable binding"):
        validate_pilot_stage_result(result, expected_arm=arm)

    result = _stage_result(arm)
    wrong_job = "1205999"
    result["rendezvous"]["test"] = _rendezvous(
        arm,
        phase="test",
        job_id=wrong_job,
    )
    _rehash_stage_result(result)
    with pytest.raises(ValueError, match="share the bound Slurm leaf"):
        validate_pilot_stage_result(result, expected_arm=arm)

    result = _stage_result(arm)
    result["routing_audit"]["policy_temperature"] = 0.8
    _rehash_stage_result(result)
    with pytest.raises(ValueError, match="routing audit"):
        validate_pilot_stage_result(result, expected_arm=arm)


def test_telemetry_summary_recomputes_window_and_population_hashes(tmp_path):
    path = tmp_path / "telemetry.json"
    payload = _telemetry_payload("fixed_rep_off")
    path.write_text(json.dumps(payload), encoding="utf-8")
    summary = summarize_pilot_telemetry(path)
    assert summary["population_sha256"] == payload["population_sha256"]
    assert summary["summary_sha256"] == canonical_sha256({key: value for key, value in summary.items() if key != "summary_sha256"})

    payload["population_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="population hash changed"):
        summarize_pilot_telemetry(path)


def test_exploratory_finalizer_emits_contrasts_but_never_a_winner(
    tmp_path,
):
    run_root = tmp_path / "run"
    parent_path = run_root / "control" / "pilot_p0_suite.json"
    parent_path.parent.mkdir(parents=True)
    parent_payload = _parent_p0_payload()
    parent_path.write_text(json.dumps(parent_payload), encoding="utf-8")

    for index, arm in enumerate(PILOT_ARM_ORDER):
        _materialize_stage_result(
            run_root=run_root,
            parent_path=parent_path,
            parent_payload=parent_payload,
            arm=arm,
            job_index=index,
        )

    finalization = pilot_finalizer.finalize_pilot_results(
        run_root=run_root,
        expected_commit="a" * 40,
        expected_stage_jobs={arm: str(1205000 + index) for index, arm in enumerate(PILOT_ARM_ORDER)},
    )
    assert finalization["decision"] == "PILOT_COMPLETE_NO_PROMOTION"
    assert set(finalization["descriptive_contrasts"]) == set(PILOT_CONTRASTS)
    assert finalization["selector_emitted"] is False
    assert finalization["p2_p3_opened"] is False
    assert finalization["official_test_opened"] is False
    assert finalization["paper_claim_allowed"] is False


def test_exploratory_finalizer_rejects_cross_arm_population_change(tmp_path):
    run_root = tmp_path / "run"
    parent_path = run_root / "control" / "pilot_p0_suite.json"
    parent_path.parent.mkdir(parents=True)
    parent_payload = _parent_p0_payload()
    parent_path.write_text(json.dumps(parent_payload), encoding="utf-8")
    for index, arm in enumerate(PILOT_ARM_ORDER):
        _materialize_stage_result(
            run_root=run_root,
            parent_path=parent_path,
            parent_payload=parent_payload,
            arm=arm,
            job_index=index,
            population_shift=1.0 if index == len(PILOT_ARM_ORDER) - 1 else 0.0,
        )
    finalization = pilot_finalizer.finalize_pilot_results(
        run_root=run_root,
        expected_commit="a" * 40,
        expected_stage_jobs={arm: str(1205000 + index) for index, arm in enumerate(PILOT_ARM_ORDER)},
    )
    assert finalization["decision"] == ("PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE")
    assert finalization["all_six_arms_passed"] is False
    assert finalization["descriptive_contrasts"] == {}


def test_exploratory_finalizer_rejects_noncanonical_p0_parent(tmp_path):
    run_root = tmp_path / "run"
    parent_path = run_root / "control" / "pilot_p0_suite.json"
    parent_path.parent.mkdir(parents=True)
    parent_payload = _parent_p0_payload()
    parent_path.write_text(json.dumps(parent_payload), encoding="utf-8")
    for index, arm in enumerate(PILOT_ARM_ORDER):
        _materialize_stage_result(
            run_root=run_root,
            parent_path=parent_path,
            parent_payload=parent_payload,
            arm=arm,
            job_index=index,
        )
    alternate_parent = run_root / "control" / "alternate_p0_suite.json"
    alternate_parent.write_text(json.dumps(parent_payload), encoding="utf-8")
    arm = PILOT_ARM_ORDER[0]
    result_path = run_root / pilot_cell_relative_path(arm=arm, seed=PILOT_SEED) / "stage_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["parent_p0_suite"] = {
        "path": str(alternate_parent.resolve()),
        "file_sha256": sha256_file(alternate_parent),
        "suite_sha256": parent_payload["suite_sha256"],
    }
    _rehash_stage_result(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")

    finalization = pilot_finalizer.finalize_pilot_results(
        run_root=run_root,
        expected_commit="a" * 40,
        expected_stage_jobs={name: str(1205000 + index) for index, name in enumerate(PILOT_ARM_ORDER)},
    )
    assert finalization["decision"] == ("PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE")
    assert finalization["failures"][arm]["status"] == "INVALID_STAGE_RESULT"
    assert "P0 parent file changed" in finalization["failures"][arm]["exception_message"]


def test_exploratory_finalizer_seals_missing_leaves_without_performance_inference(
    tmp_path,
):
    finalization = pilot_finalizer.finalize_pilot_results(
        run_root=tmp_path / "run",
        expected_commit="a" * 40,
        expected_stage_jobs={arm: str(1205000 + index) for index, arm in enumerate(PILOT_ARM_ORDER)},
    )
    assert finalization["status"] == "INCOMPLETE_EXPLORATORY_PILOT"
    assert finalization["decision"] == "PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE"
    assert finalization["arms"] == {}
    assert set(finalization["failures"]) == set(PILOT_ARM_ORDER)
    assert finalization["descriptive_contrasts"] == {}
    assert finalization["paper_claim_allowed"] is False


def test_p0_finalizer_failsafe_receipt_is_hashed_and_blocks_training(
    tmp_path,
    monkeypatch,
):
    run_root = tmp_path / "run"
    args = argparse.Namespace(
        run_root=run_root,
        expected_commit="a" * 40,
    )
    monkeypatch.setattr(pilot_p0_finalizer, "_inside", lambda *_: True)
    monkeypatch.setattr(
        pilot_p0_finalizer,
        "_git_output",
        lambda *_: "a" * 40,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "1206010")
    try:
        raise RuntimeError("synthetic prevalidation failure")
    except RuntimeError as error:
        pilot_p0_finalizer._write_failsafe_failure(args=args, error=error)
    path = run_root / "control" / "pilot_p0_failure.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("failure_sha256")
    assert payload["schema_version"] == PILOT_P0_FAILURE_SCHEMA
    assert payload["status"] == "FAIL_P0_FINALIZER_MECHANICAL_ONLY"
    assert payload["training_authorized"] is False
    assert payload["performance_inference_allowed"] is False
    assert digest == canonical_sha256(payload)


def test_finalizer_failsafe_seals_incomplete_without_metrics(
    tmp_path,
    monkeypatch,
):
    run_root = tmp_path / "run"
    args = argparse.Namespace(
        run_root=run_root,
        expected_commit="a" * 40,
    )
    monkeypatch.setattr(pilot_finalizer, "_inside", lambda *_: True)
    monkeypatch.setattr(pilot_finalizer, "_git_output", lambda *_: "a" * 40)
    monkeypatch.setenv("SLURM_JOB_ID", "1206099")
    try:
        raise RuntimeError("synthetic finalizer validation failure")
    except RuntimeError as error:
        pilot_finalizer._write_failsafe_finalization(args=args, error=error)
    path = run_root / "control" / "pilot_finalization.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("finalization_sha256")
    assert payload["status"] == "INCOMPLETE_EXPLORATORY_PILOT"
    assert payload["decision"] == "PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE"
    assert payload["arms"] == {}
    assert payload["descriptive_contrasts"] == {}
    assert payload["receipt_validation_passed"] is False
    assert payload["paper_claim_allowed"] is False
    assert digest == canonical_sha256(payload)


def test_stage_wrapper_checks_pass_p0_before_cell_creation_or_training(
    tmp_path,
    monkeypatch,
):
    run_root = tmp_path / "run"
    cell_root = run_root / pilot_cell_relative_path(
        arm=PILOT_ARM_ORDER[0],
        seed=PILOT_SEED,
    )
    args = argparse.Namespace(
        arm=PILOT_ARM_ORDER[0],
        run_root=run_root,
        expected_commit="a" * 40,
    )
    monkeypatch.setattr(pilot_stage, "_current_commit", lambda: "a" * 40)
    monkeypatch.setattr(pilot_stage, "_inside", lambda *_: True)
    monkeypatch.setattr(
        pilot_stage.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    launches: list[list[str]] = []
    monkeypatch.setattr(
        pilot_stage,
        "_run_logged",
        lambda command, **_: launches.append(command),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "1206020")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(
        FileNotFoundError,
        match="requires its sealed P0 suite",
    ):
        pilot_stage._execute(args, cell_root=cell_root)
    assert not cell_root.exists()
    assert launches == []


def test_p0_script_mode_import_bootstraps_repository_root(tmp_path):
    script = ROOT / "tools" / "bata" / "run_georoute_p0_gate.py"
    code = (
        "import runpy, sys\n"
        f"namespace = runpy.run_path({str(script)!r}, run_name='p0_import_probe')\n"
        "assert str(namespace['ROOT']) in sys.path\n"
        "import tools.bata.georoute_estimator_pilot_contract\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_pilot_job_receipt_survives_sorted_json_key_order():
    jobs = {
        "p0": {arm: str(1206000 + index) for index, arm in enumerate(PILOT_ARM_ORDER)},
        "p0_finalizer": "1206010",
        "stage": {arm: str(1206020 + index) for index, arm in enumerate(PILOT_ARM_ORDER)},
    }
    sorted_round_trip = json.loads(json.dumps(jobs, sort_keys=True))
    validated = validate_pilot_job_receipt(
        sorted_round_trip,
        expected_p0_finalizer="1206010",
    )
    assert tuple(validated["p0"]) == PILOT_ARM_ORDER
    assert tuple(validated["stage"]) == PILOT_ARM_ORDER

    duplicated = json.loads(json.dumps(jobs))
    duplicated["stage"][PILOT_ARM_ORDER[0]] = duplicated["p0"][PILOT_ARM_ORDER[0]]
    with pytest.raises(ValueError, match="reuses a Slurm job ID"):
        validate_pilot_job_receipt(duplicated)


def test_pilot_deployer_and_launchers_do_not_reuse_old_selector_or_open_test():
    deployer = (ROOT / "tools" / "bata" / "deploy_georoute_estimator_pilot.py").read_text(encoding="utf-8")
    finalizer = (ROOT / "tools" / "bata" / "finalize_georoute_estimator_pilot.py").read_text(encoding="utf-8")
    stage_launcher = (ROOT / "scripts" / "run_georoute_estimator_pilot_stage_slurm.sh").read_text(encoding="utf-8")

    assert "select_p1_roi_candidate" not in deployer + finalizer
    assert "select_p2_roi_candidate" not in deployer + finalizer
    assert "PILOT_COMPLETE_NO_PROMOTION" in finalizer
    assert '"GEOROUTE_P0_HEIGHT": "180"' in deployer
    assert '"GEOROUTE_P0_WIDTH": "320"' in deployer
    assert '"p2_p3_opened": False' in deployer
    assert '"official_test_opened": False' in deployer
    assert '"p0_finalizer_afterany_all_p0": True' in deployer
    assert '"training_wrappers_afterany_p0_finalizer": True' in deployer
    assert "p0_finalizer_afterok_all_p0" not in deployer
    assert "training_six_parallel_afterok_p0_suite" not in deployer
    assert deployer.count('dependency_type="afterany"') >= 3
    assert "--not_eval" not in stage_launcher
    assert "tools.bata.georoute_estimator_pilot_stage_runner" in stage_launcher
