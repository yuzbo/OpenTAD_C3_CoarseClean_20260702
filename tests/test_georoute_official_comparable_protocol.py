from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata.finalize_georoute_official_comparable_preflight import (
    _classify,
)
from tools.bata.georoute_amp_diagnostic import (
    AMP_DIAGNOSTIC_ARMS,
    AMP_FORMAL_PREFLIGHT_PROFILE,
    amp_protocol_spec,
)
from tools.bata.georoute_experiment_contract import canonical_sha256
from tools.bata.georoute_official_comparable_contract import (
    FORMAL_CONFIG_BATCH_SIZE,
    FORMAL_DEVELOPMENT_ARM_ORDER,
    FORMAL_DEVELOPMENT_CHECKPOINT_SIDECAR_SCHEMA,
    FORMAL_DEVELOPMENT_SEEDS,
    FORMAL_GLOBAL_BATCH_SIZE,
    FORMAL_PER_RANK_BATCH_SIZE,
    FORMAL_WORLD_SIZE,
    OFFICIAL_CONFIG_SHA256,
    OFFICIAL_COMPARABLE_PREFLIGHT_SCHEMA,
    OFFICIAL_DDP_WORLD2_KAT_PASS,
    OFFICIAL_DDP_WORLD2_KAT_SCHEMA,
    OFFICIAL_UPSTREAM_RELEASE_COMMIT,
    bind_formal_development_config,
    build_protocol_manifest,
    sha256_file,
    validate_formal_checkpoint_sidecar,
    validate_formal_development_config,
    validate_protocol_manifest,
    validate_world2_kat_receipt,
)
from tools.bata.finalize_georoute_official_development import (
    _selector_eligibility,
    _strict_pareto_dominates,
)


ROOT = Path(__file__).resolve().parents[1]


def _repair_parent() -> dict:
    payload = {
        "schema_version": "georoute_ddp_fp16_cast_repair_finalization_v1",
        "status": "COMPLETE_DDP_FP16_CAST_REPAIR_GATE_ONLY",
        "decision": (
            "DDP_FP16_CAST_REPAIR_GATE_PASS_"
            "MATCHED_FORMAL_PROTOCOL_FREEZE_AUTHORIZED"
        ),
        "runtime_commit": "a" * 40,
        "all_arms_passed": True,
        "repair_gate_passed": True,
        "matched_formal_protocol_freeze_authorized": True,
        "official_protocol_freeze_authorized": False,
        "performance_metrics": {},
        "performance_inference_allowed": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    payload["finalization_sha256"] = canonical_sha256(payload)
    return payload


def test_protocol_manifest_freezes_upstream_bridge_matrix_and_test_boundary(
    tmp_path: Path,
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"fit": ["v1"], "gate": ["v2"]}}),
        encoding="utf-8",
    )
    annotation = tmp_path / "development.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "v1": {"subset": "training", "annotations": []},
                    "v2": {"subset": "training", "annotations": []},
                }
            }
        ),
        encoding="utf-8",
    )
    class_map = tmp_path / "class_map.txt"
    class_map.write_text("0 action\n", encoding="utf-8")
    pretrained = tmp_path / "pretrained.pth"
    pretrained.write_bytes(b"pretrained")
    source = tmp_path / "georoute.py"
    source.write_text("source = True\n", encoding="utf-8")
    videos = tmp_path / "training_videos"
    videos.mkdir()
    official = (
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "e2e_thumos_videomae_s_768x1_160_adapter.py"
    )

    protocol = build_protocol_manifest(
        runtime_commit="b" * 40,
        runtime_origin_ref="refs/remotes/origin/formal",
        current_official_config_path=official,
        georoute_source_config_path=source,
        manifest_path=manifest,
        development_annotation_path=annotation,
        class_map_path=class_map,
        development_video_root=videos,
        pretrained_checkpoint_path=pretrained,
        repair_parent=_repair_parent(),
    )
    validated = validate_protocol_manifest(protocol)
    assert validated["upstream_anchor"]["release_commit"] == (
        OFFICIAL_UPSTREAM_RELEASE_COMMIT
    )
    assert validated["upstream_anchor"]["config_sha256"] == OFFICIAL_CONFIG_SHA256
    assert tuple(validated["formal_development_matrix"]["arm_order"]) == (
        FORMAL_DEVELOPMENT_ARM_ORDER
    )
    assert tuple(validated["formal_development_matrix"]["seeds"]) == (
        FORMAL_DEVELOPMENT_SEEDS
    )
    assert (
        validated["formal_development_matrix"]["global_batch_size"]
        == FORMAL_GLOBAL_BATCH_SIZE
    )
    assert validated["sealed_test_policy"]["allowed_during_development_matrix"] is False
    assert validated["paper_claim_allowed"] is False


def test_formal_resource_profile_matches_official_scheduler_and_batch_contract():
    spec = amp_protocol_spec(AMP_FORMAL_PREFLIGHT_PROFILE)
    assert spec["single_rank_stress_batch_size"] == 2
    assert spec["formal_target_config_batch_size"] == 2
    assert spec["formal_target_per_rank_batch_size"] == 1
    assert spec["formal_target_world_size"] == 2
    assert spec["official_scheduler_hyperparameters_matched"] is True
    assert spec["ddp_fp16_compress_enabled"] is False
    assert spec["seed"] not in {42, *FORMAL_DEVELOPMENT_SEEDS}
    stage_runner = (
        ROOT / "tools" / "bata" / "georoute_amp_diagnostic_stage_runner.py"
    ).read_text(encoding="utf-8")
    assert "AMP_FORMAL_PREFLIGHT_PROFILE" in stage_runner
    assert FORMAL_CONFIG_BATCH_SIZE == FORMAL_GLOBAL_BATCH_SIZE == 2
    assert FORMAL_PER_RANK_BATCH_SIZE == 1
    assert FORMAL_WORLD_SIZE == 2


def _stats(dtype: str, *, finite: bool, max_abs: float, nonfinite_count: int = 0):
    return {
        "dtype": dtype,
        "shape": [2],
        "finite": finite,
        "finite_count": 2 - nonfinite_count,
        "nonfinite_count": nonfinite_count,
        "max_abs": max_abs,
    }


def test_world2_kat_requires_fp32_reduction_and_overflowing_fp16_shadow():
    payload = {
        "schema_version": OFFICIAL_DDP_WORLD2_KAT_SCHEMA,
        "status": OFFICIAL_DDP_WORLD2_KAT_PASS,
        "runtime_commit": "c" * 40,
        "slurm_job_id": "1209001",
        "backend": "nccl",
        "world_size": 2,
        "rank_local_scaled_gradient_targets": [70000.0, 90000.0],
        "comm_hook_registration_invoked": False,
        "default_fp32_ddp_reduction_completed": True,
        "reduced_scaled_gradient": _stats(
            "torch.float32", finite=True, max_abs=80000.0
        ),
        "unscaled_gradient": _stats(
            "torch.float32", finite=True, max_abs=80000.0 / 65536.0
        ),
        "detached_fp16_cast_shadow": _stats(
            "torch.float16",
            finite=False,
            max_abs=0.0,
            nonfinite_count=2,
        ),
        "optimizer_update_completed_on_all_ranks": True,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    payload["kat_sha256"] = canonical_sha256(payload)
    validate_world2_kat_receipt(
        payload,
        expected_commit="c" * 40,
        expected_slurm_job_id="1209001",
    )
    broken = dict(payload)
    broken["detached_fp16_cast_shadow"] = _stats(
        "torch.float16", finite=True, max_abs=65504.0
    )
    broken.pop("kat_sha256")
    broken["kat_sha256"] = canonical_sha256(broken)
    with pytest.raises(ValueError, match="world2 default FP32 DDP"):
        validate_world2_kat_receipt(broken)


def _formal_stage_result(*, failed_attempts: int = 1, final_scale: float = 16384.0):
    spec = amp_protocol_spec(AMP_FORMAL_PREFLIGHT_PROFILE)
    summary = {
        "batch_count": spec["max_batches"],
        "failed_attempt_count": failed_attempts,
        "max_consecutive_skipped_attempts": 1,
        "minimum_observed_scale": final_scale,
        "stable_tail_all_success": True,
        "stable_tail_success_count": spec["stable_tail_batches"],
        "retry_attempt_count": 0,
        "replay_attempt_count": 0,
        "scheduler_advance_count": spec["max_batches"],
        "ema_update_count": spec["max_batches"],
        "all_forward_losses_finite": True,
        "data_fingerprint_sha256_by_batch": [
            f"{index:064x}" for index in range(spec["max_batches"])
        ],
        "final_scale": final_scale,
    }
    return {
        "status": spec["stage_pass_status"],
        "diagnostic_receipt": {
            "status": spec["receipt_pass_status"],
            "summary": summary,
            "successful_updates": spec["max_batches"] - failed_attempts,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        },
    }


def test_preflight_classifier_requires_both_real_batch_arms_and_world2_kat():
    stages = {arm: _formal_stage_result() for arm in AMP_DIAGNOSTIC_ARMS}
    passed = _classify(stage_results=stages, kat_passed=True)
    assert passed["passed"] is True
    assert passed["decision"] == "FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED"
    held = _classify(stage_results=stages, kat_passed=False)
    assert held["passed"] is False
    assert held["decision"] == "OFFICIAL_COMPARABLE_PREFLIGHT_HOLD"


def test_preflight_classifier_treats_missing_final_scale_as_hold():
    stages = {arm: _formal_stage_result() for arm in AMP_DIAGNOSTIC_ARMS}
    stages[AMP_DIAGNOSTIC_ARMS[0]]["diagnostic_receipt"]["summary"][
        "final_scale"
    ] = None
    held = _classify(stage_results=stages, kat_passed=True)
    assert held["passed"] is False
    assert held["decision"] == "OFFICIAL_COMPARABLE_PREFLIGHT_HOLD"
    missing_scale = held["final_scales"][AMP_DIAGNOSTIC_ARMS[0]]
    assert missing_scale != missing_scale


def test_p1_do_binder_initializes_protocol_for_exact_official_config(tmp_path: Path):
    runtime_commit = "c" * 40
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"fit": ["v1"], "gate": ["v2"]}}),
        encoding="utf-8",
    )
    annotation = tmp_path / "development.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "v1": {"subset": "training", "annotations": []},
                    "v2": {"subset": "training", "annotations": []},
                }
            }
        ),
        encoding="utf-8",
    )
    class_map = tmp_path / "class_map.txt"
    class_map.write_text("0 action\n", encoding="utf-8")
    pretrained = tmp_path / "pretrained.pth"
    pretrained.write_bytes(b"pretrained")
    videos = tmp_path / "training_videos"
    videos.mkdir()
    preflight = {
        "schema_version": OFFICIAL_COMPARABLE_PREFLIGHT_SCHEMA,
        "status": "PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY",
        "decision": "FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED",
        "runtime_commit": runtime_commit,
        "formal_development_matrix_authorized": True,
        "official_protocol_freeze_authorized": False,
        "performance_metrics": {},
        "performance_inference_allowed": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    preflight["finalization_sha256"] = canonical_sha256(preflight)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    official = (
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "e2e_thumos_videomae_s_768x1_160_adapter.py"
    )

    cfg = bind_formal_development_config(
        source_config_path=official,
        arm="dense_native",
        seed=3407,
        work_dir=tmp_path / "do",
        manifest_path=manifest,
        development_annotation_path=annotation,
        class_map_path=class_map,
        development_video_root=videos,
        pretrained_checkpoint_path=pretrained,
        runtime_commit=runtime_commit,
        preflight_finalization_path=preflight_path,
        expected_preflight_file_sha256=sha256_file(preflight_path),
    )
    assert cfg.georoute_protocol.status == (
        "official_comparable_three_seed_development_only"
    )
    assert cfg.georoute_protocol.official_test_open_allowed is False
    assert cfg.workflow.require_successful_update_hook is False
    validated = validate_formal_development_config(cfg, seed=3407)
    assert validated["arm"] == "dense_native"


def test_slurm_world2_gate_uses_two_logical_gpus_without_physical_indices():
    launcher = (
        ROOT / "scripts" / "run_georoute_official_world2_ddp_kat_slurm.sh"
    ).read_text(encoding="utf-8")
    deployer = (
        ROOT
        / "tools"
        / "bata"
        / "deploy_georoute_official_comparable_preflight.py"
    ).read_text(encoding="utf-8")
    assert "srun --exact --ntasks=1 --gpus=2 --cpus-per-task=10" in launcher
    assert "--mem=" not in launcher
    assert "--nproc_per_node=2" in launcher
    assert "CUDA_VISIBLE_DEVICES=" not in launcher
    assert 'command.extend(["--gpus", "2", "--cpus-per-task", "12"])' in deployer
    assert '"--mem"' not in deployer
    assert "additional_jobs=4" in deployer
    assert "OFFICIAL_UPSTREAM_TRACKING_REF" in deployer
    assert "no_artifact_storage_capacity_receipt" in deployer


def test_formal_development_dag_reserves_all_fifteen_cells_plus_finalizer():
    deployer = (
        ROOT
        / "tools"
        / "bata"
        / "deploy_georoute_official_development.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT
        / "scripts"
        / "run_georoute_official_development_stage_slurm.sh"
    ).read_text(encoding="utf-8")
    assert "_require_submit_capacity(additional_jobs=16)" in deployer
    assert '"all_fifteen_cells_parallel": True' in deployer
    assert '"--gpus",\n                "2"' in deployer
    assert "CUDA_VISIBLE_DEVICES=" not in launcher
    assert "--nproc_per_node=2" not in launcher
    runner = (
        ROOT
        / "tools"
        / "bata"
        / "georoute_official_development_stage_runner.py"
    ).read_text(encoding="utf-8")
    assert "nproc_per_node=FORMAL_WORLD_SIZE" in runner
    test_entrypoint = (ROOT / "tools" / "test.py").read_text(encoding="utf-8")
    assert "deterministic_warn_only=(s1_binding is None)" in test_entrypoint


def test_formal_checkpoint_sidecar_closes_update_and_binding_commit_marker(
    tmp_path: Path,
):
    checkpoint = tmp_path / "epoch_59.pth"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {
        "schema_version": FORMAL_DEVELOPMENT_CHECKPOINT_SIDECAR_SCHEMA,
        "runtime_commit": "d" * 40,
        "binding_sha256": "e" * 64,
        "arm": "residual_st_rep_off",
        "seed": 3407,
        "epoch": 59,
        "successful_updates": 598,
        "train_batches_per_epoch": 10,
        "amp_skipped_attempts": 2,
        "max_amp_retries_observed": 0,
        "world_size": 2,
        "global_batch_size": 2,
        "checkpoint_policy": "final_epoch_ema_only_atomic",
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    sidecar = {
        "schema_version": FORMAL_DEVELOPMENT_CHECKPOINT_SIDECAR_SCHEMA,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "experiment_metadata": metadata,
    }
    sidecar["sidecar_sha256"] = canonical_sha256(sidecar)
    sidecar_path = Path(str(checkpoint) + ".metadata.json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validated = validate_formal_checkpoint_sidecar(checkpoint)
    assert validated["experiment_metadata"]["successful_updates"] == 598
    broken = json.loads(sidecar_path.read_text(encoding="utf-8"))
    broken["experiment_metadata"]["amp_skipped_attempts"] = 1
    broken["experiment_metadata"].pop("metadata_sha256")
    broken["experiment_metadata"]["metadata_sha256"] = canonical_sha256(
        broken["experiment_metadata"]
    )
    broken.pop("sidecar_sha256")
    broken["sidecar_sha256"] = canonical_sha256(broken)
    sidecar_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="update accounting"):
        validate_formal_checkpoint_sidecar(checkpoint)


def _selection_matrix():
    values = {}
    for arm in FORMAL_DEVELOPMENT_ARM_ORDER:
        values[arm] = {}
        for index, seed in enumerate(FORMAL_DEVELOPMENT_SEEDS):
            accuracy = {
                "dense_native": 60.0,
                "fixed_lattice": 50.0,
                "random": 49.0,
                "residual_st_rep_off": 53.0,
                "residual_pl_rep_off": 52.0,
            }[arm] + index * 0.1
            cost = {
                "dense_native": 20.0,
                "fixed_lattice": 12.0,
                "random": 12.5,
                "residual_st_rep_off": 10.0,
                "residual_pl_rep_off": 11.0,
            }[arm]
            values[arm][seed] = {
                "metrics": {"high_iou_composite": accuracy},
                "profile": {"model_and_postprocess_p50_ms": cost},
            }
    return values


def test_frozen_selector_requires_every_seed_controls_and_strict_pareto():
    matrix = _selection_matrix()
    eligibility = _selector_eligibility(
        matrix,
        arm="residual_st_rep_off",
    )
    assert eligibility["all_three_seeds_passed"] is True
    dominance = _strict_pareto_dominates(
        matrix,
        treatment="residual_st_rep_off",
        control="residual_pl_rep_off",
    )
    assert dominance["strict_pareto_dominates"] is True
    matrix["residual_st_rep_off"][3408]["metrics"][
        "high_iou_composite"
    ] = 48.0
    held = _selector_eligibility(matrix, arm="residual_st_rep_off")
    assert held["all_three_seeds_passed"] is False
