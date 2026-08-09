from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file
from tools.bata.georoute_residual_centering_cost_contract import (
    RESIDUAL_CENTERING_COST_BINDING_SCHEMA,
    RESIDUAL_CENTERING_COST_DEPLOYMENT_SCHEMA,
    RESIDUAL_CENTERING_COST_ORDER,
    RESIDUAL_CENTERING_COST_PAIRS,
    RESIDUAL_CENTERING_COST_SENSITIVE_RUNTIME_PATHS,
    analyze_residual_centering_paired_cost,
    build_residual_centering_cost_config,
    finalize_residual_centering_cost,
    validate_frozen_residual_centering_cost_contract,
    validate_residual_centering_cost_deployment,
)
from tools.bata.georoute_residual_centering_training_contract import (
    RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER,
    bind_residual_centering_training_config,
    configure_residual_centering_accuracy,
)
from tools.bata.spatial_zoom_s1_cost import make_profile_exposure_id


ROOT = Path(__file__).resolve().parents[1]


def _inputs(tmp_path: Path) -> dict[str, Path]:
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
    pretrained.write_bytes(b"contract-only")
    videos = tmp_path / "videos"
    videos.mkdir()
    return {
        "source_config_path": ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "georoute_dynamic_scnr_stage1_base.py",
        "manifest_path": manifest,
        "development_annotation_path": annotation,
        "class_map_path": class_map,
        "development_video_root": videos,
        "pretrained_checkpoint_path": pretrained,
    }


def test_residual_centering_cost_freezes_abba_baab_and_one_job_policy():
    validate_frozen_residual_centering_cost_contract()
    assert RESIDUAL_CENTERING_COST_ORDER[:4] == (
        "none_control",
        "residual_window_center",
        "residual_window_center",
        "none_control",
    )
    assert RESIDUAL_CENTERING_COST_ORDER[4:] == (
        "residual_window_center",
        "none_control",
        "none_control",
        "residual_window_center",
    )
    assert RESIDUAL_CENTERING_COST_PAIRS == ((0, 1), (3, 2), (5, 4), (6, 7))


@pytest.mark.parametrize(
    ("variant", "mode"),
    [
        ("none_control", "none"),
        ("residual_window_center", "residual_window_center"),
    ],
)
def test_residual_centering_cost_config_consumes_checkpoint_without_training(
    tmp_path: Path, monkeypatch, variant: str, mode: str
):
    train = bind_residual_centering_training_config(
        **_inputs(tmp_path),
        variant=variant,
        seed=3407,
        work_dir=tmp_path / "train",
        runtime_commit="c" * 40,
    )
    accuracy = configure_residual_centering_accuracy(
        train,
        variant=variant,
        replay="accuracy_a",
        work_dir=tmp_path / "accuracy",
        training_config_sha256=canonical_sha256(train.to_dict()),
    )
    train_path = tmp_path / "train.py"
    accuracy_path = tmp_path / "accuracy.py"
    train.dump(str(train_path))
    accuracy.dump(str(accuracy_path))
    stage = {
        "variant": variant,
        "runtime_commit": "c" * 40,
        "stage_result_sha256": "d" * 64,
        "binding_sha256": train.georoute_residual_centering_training_binding[
            "binding_sha256"
        ],
        "binding": dict(train.georoute_residual_centering_training_binding),
        "config_receipts": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "accuracy_a": {
                "path": str(accuracy_path),
                "sha256": sha256_file(accuracy_path),
            },
        },
    }
    monkeypatch.setattr(
        "tools.bata.georoute_residual_centering_cost_contract.validate_residual_centering_training_stage_result",
        lambda payload, **_kwargs: payload,
    )
    cost = build_residual_centering_cost_config(stage, variant=variant)
    binding = cost.georoute_residual_centering_cost_binding
    assert binding.schema_version == RESIDUAL_CENTERING_COST_BINDING_SCHEMA
    assert binding.training_or_resume_allowed is False
    assert cost.model.backbone.custom.georoute_branch_calibration_mode == mode
    assert cost.model.backbone.custom.georoute_diagnostic_telemetry_enabled is False
    assert cost.georoute_diagnostic_telemetry.enabled is False
    assert cost.solver.test.batch_size == 1
    assert cost.solver.test.num_workers == 0


def _synthetic_rows(*, center_ratio: float) -> list[dict]:
    rows = []
    for pass_index, variant in enumerate(RESIDUAL_CENTERING_COST_ORDER):
        ratio = center_ratio if variant == "residual_window_center" else 1.0
        for ordinal, (video, base) in enumerate(
            (("video-a", 10.0), ("video-a", 12.0), ("video-b", 14.0))
        ):
            physical_window_id = f"{video}:{ordinal}"
            rows.append(
                {
                    "pass_index": pass_index,
                    "arm": variant,
                    "sample_ordinal": ordinal,
                    "loader_ordinal": ordinal,
                    "video_id": video,
                    "physical_window_id": physical_window_id,
                    "window_id": make_profile_exposure_id(
                        physical_window_id, ordinal
                    ),
                    "end_to_end_serial_ms": base * ratio,
                    "gpu_energy_j": (base / 10.0) * ratio,
                }
            )
    return rows


def test_residual_centering_paired_bootstrap_opens_only_cost_noninferiority():
    better = analyze_residual_centering_paired_cost(
        _synthetic_rows(center_ratio=0.90), bootstrap_iterations=200
    )
    assert better["cost_noninferior"] is True
    assert better["strict_pareto_observed"] is True
    assert better["primary_metrics"]["end_to_end_p50"][
        "center_over_none_ratio"
    ] == pytest.approx(0.90)

    worse = analyze_residual_centering_paired_cost(
        _synthetic_rows(center_ratio=1.08), bootstrap_iterations=200
    )
    assert worse["cost_noninferior"] is False
    assert worse["strict_pareto_observed"] is False


def test_residual_centering_paired_bootstrap_rejects_pseudo_pairing():
    rows = _synthetic_rows(center_ratio=1.0)
    rows[0]["physical_window_id"] = "video-a:wrong"
    with pytest.raises(ValueError, match="exposure identity is invalid"):
        analyze_residual_centering_paired_cost(rows, bootstrap_iterations=10)

    rows = _synthetic_rows(center_ratio=1.0)
    rows[1]["window_id"] = rows[0]["window_id"]
    rows[1]["physical_window_id"] = rows[0]["physical_window_id"]
    rows[1]["loader_ordinal"] = rows[0]["loader_ordinal"]
    rows[1]["sample_ordinal"] = rows[0]["sample_ordinal"]
    with pytest.raises(ValueError, match="identities are not unique"):
        analyze_residual_centering_paired_cost(rows, bootstrap_iterations=10)


def test_residual_centering_cost_finalizer_authorizes_seeds_not_paper(
    tmp_path: Path, monkeypatch
):
    raw_path = tmp_path / "rows.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    profile = {
        "training_run_root": str(tmp_path),
        "artifact_receipts": {"raw_samples": {"path": str(raw_path)}},
    }
    base = {
        "average_mAP": 10.0,
        "mAP@0.3": 13.0,
        "mAP@0.4": 12.0,
        "mAP@0.5": 11.0,
        "mAP@0.6": 9.0,
        "mAP@0.7": 7.0,
        "high_iou_composite": 8.0,
    }
    centered = {key: value + 1.0 for key, value in base.items()}
    source = {
        "stages": {
            "none_control": {
                "accuracy_replays": {"accuracy_a": {"metrics": base}}
            },
            "residual_window_center": {
                "accuracy_replays": {"accuracy_a": {"metrics": centered}}
            },
        }
    }
    monkeypatch.setattr(
        "tools.bata.georoute_residual_centering_cost_contract.validate_residual_centering_cost_profile",
        lambda payload, **_kwargs: payload,
    )
    monkeypatch.setattr(
        "tools.bata.georoute_residual_centering_cost_contract.validate_residual_centering_cost_source",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        "tools.bata.georoute_residual_centering_cost_contract._load_jsonl_objects",
        lambda *_args, **_kwargs: [{}],
    )
    monkeypatch.setattr(
        "tools.bata.georoute_residual_centering_cost_contract.analyze_residual_centering_paired_cost",
        lambda _rows: {
            "cost_noninferior": True,
            "strict_pareto_observed": False,
        },
    )
    result = finalize_residual_centering_cost(
        profile,
        expected_model_runtime_commit="c" * 40,
        expected_execution_commit="d" * 40,
    )
    assert result["seeds_3408_3409_opened"] is True
    assert result["paper_claim_allowed"] is False
    assert result["single_job_cost_is_paper_efficiency_claim"] is False


def test_residual_centering_cost_deployment_binds_one_job_and_training_receipts(
    tmp_path: Path, monkeypatch
):
    run_root = tmp_path / "cost"
    training_root = tmp_path / "training"
    run_root.mkdir()
    training_root.mkdir()
    source = {
        "stage_result_receipts": {
            variant: {"path": f"/{variant}", "sha256": variant}
            for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
        },
        "training_finalization_receipt": {
            "path": "/finalization",
            "sha256": "f" * 64,
        },
    }
    monkeypatch.setattr(
        "tools.bata.georoute_residual_centering_cost_contract.validate_residual_centering_cost_source",
        lambda *_args, **_kwargs: source,
    )
    changed_files = ["tools/bata/georoute_residual_centering_cost_contract.py"]
    execution_delta = {
        "model_runtime_commit": "c" * 40,
        "execution_commit": "d" * 40,
        "model_runtime_is_ancestor": True,
        "source_model_or_config_changed": False,
        "sensitive_runtime_paths": list(
            RESIDUAL_CENTERING_COST_SENSITIVE_RUNTIME_PATHS
        ),
        "changed_files": changed_files,
        "changed_files_sha256": canonical_sha256(changed_files),
    }
    deployment = {
        "schema_version": RESIDUAL_CENTERING_COST_DEPLOYMENT_SCHEMA,
        "status": "DEPLOYED_RESIDUAL_CENTERING_SINGLE_JOB_PAIRED_COST",
        "study_id": "scnr_residual_centering_paired_cost_v1",
        "model_runtime_commit": "c" * 40,
        "execution_commit": "d" * 40,
        "run_root": str(run_root.resolve()),
        "training_run_root": str(training_root.resolve()),
        "cost_order": list(RESIDUAL_CENTERING_COST_ORDER),
        "paired_pass_indices": [list(pair) for pair in RESIDUAL_CENTERING_COST_PAIRS],
        "jobs": {"paired_cost": "123"},
        "single_slurm_job": True,
        "single_visible_gpu": True,
        "continuous_power_sidecar": True,
        "training_or_resume_allowed": False,
        "source_model_or_config_changed": False,
        "execution_delta": execution_delta,
        "stage_result_receipts": source["stage_result_receipts"],
        "training_finalization_receipt": source["training_finalization_receipt"],
        "all_jobs_held_until_immutable_receipt": True,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    deployment["deployment_sha256"] = canonical_sha256(deployment)
    validate_residual_centering_cost_deployment(
        deployment,
        run_root=run_root,
        training_run_root=training_root,
        expected_model_runtime_commit="c" * 40,
        expected_execution_commit="d" * 40,
        expected_job_id="123",
    )
    deployment["jobs"]["paired_cost"] = "124"
    with pytest.raises(ValueError, match="deployment receipt is invalid"):
        validate_residual_centering_cost_deployment(
            deployment,
            run_root=run_root,
            training_run_root=training_root,
            expected_model_runtime_commit="c" * 40,
            expected_execution_commit="d" * 40,
            expected_job_id="123",
        )


def test_residual_centering_cost_launcher_is_single_gpu_no_retrain_same_job():
    launcher = (
        ROOT / "scripts" / "run_georoute_residual_centering_cost_slurm.sh"
    ).read_text(encoding="utf-8")
    deployer = (
        ROOT / "tools" / "bata" / "deploy_georoute_residual_centering_cost.py"
    ).read_text(encoding="utf-8")
    profiler = (
        ROOT / "tools" / "bata" / "profile_georoute_residual_centering_cost.py"
    ).read_text(encoding="utf-8")
    assert launcher.count("#SBATCH --gres=gpu:1") == 1
    assert "CUDA_VISIBLE_DEVICES=" not in launcher
    assert "source /etc/profile\nset -euo pipefail" in launcher
    assert "taskset -c" in launcher
    assert "--standalone" not in launcher
    assert "--rdzv-endpoint=127.0.0.1:0" in launcher
    assert "tools/bata/finalize_georoute_residual_centering_cost.py" in launcher
    assert "tools/train.py" not in launcher
    assert 'additional_jobs=1' in deployer
    assert 'hold=True' in deployer
    assert 'release_jobs(ROOT, submitted)' in deployer
    assert "for pass_index, variant in enumerate(RESIDUAL_CENTERING_COST_ORDER)" in profiler
    assert "sampler.start()" in profiler and "sampler.stop()" in profiler
    assert "PROFILE_STATUS=$?" in launcher
    assert "FINALIZER_STATUS=$?" in launcher
    assert "failure finalizer status=" in launcher
