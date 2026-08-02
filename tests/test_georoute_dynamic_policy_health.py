from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.bata.georoute_dynamic_policy_health import (
    DYNAMIC_POLICY_HEALTH_BINDING_SCHEMA,
    DYNAMIC_POLICY_HEALTH_PASS,
    DYNAMIC_POLICY_HEALTH_SCHEMA,
    DYNAMIC_POLICY_HEALTH_STUDY_ID,
    HEALTH_SEED,
    INITIAL_LOSS_SCALE,
    MAX_AMP_RETRIES_PER_BATCH,
    MAX_TOTAL_SKIPPED_ATTEMPTS,
    MINIMUM_LOSS_SCALE,
    REQUIRED_GRADIENT_COMPONENTS,
    REQUIRED_LOSS_KEYS,
    ROLE_NAMES,
    TARGET_SUCCESSFUL_UPDATES,
    WINDOW_TOKEN_BUDGET,
    bind_dynamic_policy_health_config,
    canonical_sha256,
    validate_dynamic_policy_health_config,
    validate_dynamic_policy_health_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _binding() -> dict:
    binding = {
        "schema_version": DYNAMIC_POLICY_HEALTH_BINDING_SCHEMA,
        "study_id": DYNAMIC_POLICY_HEALTH_STUDY_ID,
        "runtime_commit": "a" * 40,
        "world_size": 1,
        "seed": HEALTH_SEED,
        "target_successful_updates": TARGET_SUCCESSFUL_UPDATES,
        "max_amp_retries_per_batch": MAX_AMP_RETRIES_PER_BATCH,
        "max_total_skipped_attempts": MAX_TOTAL_SKIPPED_ATTEMPTS,
        "initial_loss_scale": INITIAL_LOSS_SCALE,
        "minimum_loss_scale": MINIMUM_LOSS_SCALE,
        "source_config": "/data/source.py",
        "source_config_sha256": "b" * 64,
        "manifest_path": "/data/manifest.json",
        "manifest_file_sha256": "c" * 64,
        "fit_video_ids": ["fit-a"],
        "gate_video_ids": ["gate-a"],
        "training_video_ids": ["fit-a"],
        "training_block_list_video_ids": ["gate-a"],
        "development_annotation": {
            "path": "/data/annotation.json",
            "sha256": "d" * 64,
            "video_ids": ["fit-a", "gate-a"],
            "official_test_records_loaded": 0,
        },
        "class_map_path": "/data/class_map.txt",
        "class_map_sha256": "e" * 64,
        "development_video_root": "/data/videos",
        "pretrained_checkpoint_path": "/data/pretrained.pth",
        "pretrained_checkpoint_sha256": "f" * 64,
        "work_dir": "/data/run01/sczc063/yuzibo/health",
        "report_path": "/data/run01/sczc063/yuzibo/health/policy_health_report.json",
        "dataset_split_built": "train_only",
        "development_fit_annotations_used": True,
        "gt_used_for_detector_and_auxiliary_fit_only": True,
        "gt_used_for_route": False,
        "validation_loader_built": False,
        "test_loader_built": False,
        "metric_computed": False,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def _losses() -> dict[str, float]:
    losses = {name: 0.25 for name in REQUIRED_LOSS_KEYS}
    losses["georoute_geometry_regularization_loss"] = 0.0
    return losses


def _gradient() -> dict:
    return {
        "components": {},
        "nonzero_components": sorted(REQUIRED_GRADIENT_COMPONENTS),
        "all_gradients_finite": True,
        "nonfinite_gradient_tensors": [],
    }


def _route(update_index: int) -> dict:
    return {
        "successful_update": update_index,
        "window_token_budget": WINDOW_TOKEN_BUDGET,
        "k_t_min": 63,
        "k_t_max": 65,
        "k_t_zero_count": 0,
        "k_t_histogram": {"63": 192, "65": 192},
        "role_counts": {"context": 4000, "roi": 14500, "residual": 6076},
        "soft_budget_sums": [float(WINDOW_TOKEN_BUDGET)],
        "soft_budget_max_abs_residual": 0.0,
        "attention_pairs_per_window": [WINDOW_TOKEN_BUDGET**2],
        "statistics": {},
        "geometry_min_extent_wh": [0.05, 1.0 / 11.0],
        "geometry_extent_floor_cells": 1,
        "source_grid_hw": [11, 20],
        "dynamic_auxiliary_raw": 0.5,
        "dynamic_proxy_raw": 0.5,
        "dynamic_proxy_weight": 0.5,
    }


def _valid_report() -> dict:
    updates = []
    attempts = []
    for index in range(TARGET_SUCCESSFUL_UPDATES):
        attempt = {
            "attempt_index": index,
            "iter_idx": index,
            "retry_count": 0,
            "scale_before": INITIAL_LOSS_SCALE,
            "scale_after": INITIAL_LOSS_SCALE,
            "update_succeeded": True,
            "losses": _losses(),
            "gradient": _gradient(),
        }
        attempts.append(copy.deepcopy(attempt))
        updates.append(
            {
                "update_index": index,
                **copy.deepcopy(attempt),
                "route": _route(index),
            }
        )
    role_counts = {name: 0 for name in ROLE_NAMES}
    for update in updates:
        for name, count in update["route"]["role_counts"].items():
            role_counts[name] += count
    report = {
        "schema_version": DYNAMIC_POLICY_HEALTH_SCHEMA,
        "status": DYNAMIC_POLICY_HEALTH_PASS,
        "study_id": DYNAMIC_POLICY_HEALTH_STUDY_ID,
        "binding": _binding(),
        "source": {
            "commit": "a" * 40,
            "expected_commit": "a" * 40,
            "branch": "codex/dynamic",
            "origin_ref": "a" * 40,
            "head_matches_expected": True,
            "origin_ref_matches_expected": True,
            "tree_clean": True,
        },
        "slurm": {
            "job_id": "1234",
            "logical_device": "cuda:0",
            "visible_device_count": 1,
            "cuda_visible_devices_sha256": "0" * 64,
        },
        "successful_updates": TARGET_SUCCESSFUL_UPDATES,
        "attempts": attempts,
        "updates": updates,
        "summary": {
            "target_successful_updates": TARGET_SUCCESSFUL_UPDATES,
            "successful_update_count": TARGET_SUCCESSFUL_UPDATES,
            "consumed_batch_count": TARGET_SUCCESSFUL_UPDATES,
            "optimizer_attempt_count": TARGET_SUCCESSFUL_UPDATES,
            "amp_skipped_attempt_count": 0,
            "minimum_observed_scale": INITIAL_LOSS_SCALE,
            "final_scale": INITIAL_LOSS_SCALE,
            "all_losses_finite": True,
            "all_gradients_finite": True,
            "component_nonzero_update_counts": {
                name: TARGET_SUCCESSFUL_UPDATES for name in REQUIRED_GRADIENT_COMPONENTS
            },
            "minimum_component_healthy_updates": 60,
            "weak_gradient_components": [],
            "aggregate_role_counts": role_counts,
            "missing_roles": [],
            "dominant_role_fraction": role_counts["roi"] / sum(role_counts.values()),
            "max_dominant_role_fraction": 0.995,
            "aggregate_k_t_histogram": {
                "63": 192 * TARGET_SUCCESSFUL_UPDATES,
                "65": 192 * TARGET_SUCCESSFUL_UPDATES,
            },
            "k_t_min": 63,
            "k_t_max": 65,
            "k_t_zero_count": 0,
            "k_t_zero_is_capability_not_pass_requirement": True,
            "peak_cuda_allocated_bytes": 6_000_000_000,
            "hold_reasons": [],
            "policy_health_gate_passed": True,
        },
        "update_audit": {
            "optimizer_attempts": TARGET_SUCCESSFUL_UPDATES,
            "amp_skipped_attempts": 0,
            "max_amp_retries_observed": 0,
            "consumed_batches": TARGET_SUCCESSFUL_UPDATES,
            "replay_attempts": 0,
            "scheduler_advances": TARGET_SUCCESSFUL_UPDATES,
            "ema_updates": TARGET_SUCCESSFUL_UPDATES,
        },
        "artifact_audit": {
            "checkpoint_payload_count": 0,
            "temporary_payload_count": 0,
            "prediction_payload_count": 0,
            "evaluator_output_count": 0,
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        },
        "scope": {
            "real_development_fit_data_loaded": True,
            "fit_labels_used_for_detector_and_auxiliary_only": True,
            "gt_used_for_route": False,
            "validation_loader_built": False,
            "test_loader_built": False,
            "metric_computed": False,
            "prediction_written": False,
            "checkpoint_written": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "performance_inference_allowed": False,
            "paper_claim_allowed": False,
        },
        "execution_error": None,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _resign(report: dict) -> dict:
    report.pop("report_sha256", None)
    report["report_sha256"] = canonical_sha256(report)
    return report


def test_policy_health_report_accepts_exact_64_update_no_performance_receipt():
    validate_dynamic_policy_health_report(_valid_report())


def test_policy_health_report_rejects_tampered_exact_budget():
    report = _valid_report()
    report["updates"][0]["route"]["k_t_histogram"] = {"64": 383}
    with pytest.raises(ValueError, match="exact B"):
        validate_dynamic_policy_health_report(_resign(report))


def test_policy_health_report_rejects_role_collapse_claimed_as_pass():
    report = _valid_report()
    for update in report["updates"]:
        update["route"]["role_counts"] = {
            "context": 0,
            "roi": WINDOW_TOKEN_BUDGET,
            "residual": 0,
        }
    report["summary"]["aggregate_role_counts"] = {
        "context": 0,
        "roi": WINDOW_TOKEN_BUDGET * TARGET_SUCCESSFUL_UPDATES,
        "residual": 0,
    }
    report["summary"]["dominant_role_fraction"] = 1.0
    with pytest.raises(ValueError, match="violates its gate"):
        validate_dynamic_policy_health_report(_resign(report))


def test_policy_health_report_rejects_performance_artifact_scope():
    report = _valid_report()
    report["scope"]["metric_computed"] = True
    with pytest.raises(ValueError, match="no-performance scope"):
        validate_dynamic_policy_health_report(_resign(report))


def test_policy_health_binder_builds_only_fit_train_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    source = (
        ROOT / "configs" / "adatad" / "thumos" / "georoute_dynamic_scnr_stage1_base.py"
    )
    annotation = tmp_path / "development.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "fit-a": {"subset": "training", "annotations": []},
                    "gate-a": {"subset": "training", "annotations": []},
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"fit": ["fit-a"], "gate": ["gate-a"]}}),
        encoding="utf-8",
    )
    class_map = tmp_path / "class_map.txt"
    class_map.write_text("action\n", encoding="utf-8")
    pretrained = tmp_path / "pretrained.pth"
    pretrained.write_bytes(b"not-loaded-by-contract-test")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    cfg = bind_dynamic_policy_health_config(
        source_config_path=source,
        work_dir=tmp_path / "run",
        manifest_path=manifest,
        development_annotation_path=annotation,
        class_map_path=class_map,
        development_video_root=video_root,
        pretrained_checkpoint_path=pretrained,
        runtime_commit="a" * 40,
        seed=HEALTH_SEED,
    )
    binding = validate_dynamic_policy_health_config(cfg, seed=HEALTH_SEED)
    assert set(cfg.dataset) == {"train"}
    assert binding["training_video_ids"] == ["fit-a"]
    assert cfg.dataset.train.block_list == ["gate-a"]
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.solver.fp16_compress is False


def test_policy_health_launcher_preserves_slurm_gpu_and_no_resume_contract():
    text = (ROOT / "scripts" / "run_georoute_dynamic_policy_health_slurm.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:1" in text
    assert "torchrun --standalone --nnodes=1 --nproc_per_node=1" in text
    assert "CUDA_VISIBLE_DEVICES=" not in text
    assert "--seed 4423" in text
    assert "PRECHECK_ONLY" in text
