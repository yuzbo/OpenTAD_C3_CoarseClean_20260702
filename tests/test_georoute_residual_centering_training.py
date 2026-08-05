from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.bata.georoute_dynamic_floor_m2_contract import (
    DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_TRAINING_ACCURACY_SCHEMA,
    DYNAMIC_FLOOR_M2_SEED,
    resolve_dynamic_floor_m2_accuracy_execution_commit,
)
from tools.bata.georoute_experiment_contract import canonical_sha256
from tools.bata.finalize_georoute_residual_centering_training import (
    _validate_deployment,
)
from tools.bata.georoute_residual_centering_training_contract import (
    RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES,
    RESIDUAL_CENTERING_TRAINING_DEPLOYMENT_SCHEMA,
    RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER,
    bind_residual_centering_training_config,
    configure_residual_centering_accuracy,
    finalize_residual_centering_training,
    residual_centering_training_cell_relative_path,
    summarize_residual_centering_training_branch,
    validate_frozen_residual_centering_training_contract,
    validate_residual_centering_training_config,
)


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


def _bound(tmp_path: Path, *, variant: str):
    return bind_residual_centering_training_config(
        **_inputs(tmp_path),
        variant=variant,
        seed=DYNAMIC_FLOOR_M2_SEED,
        work_dir=tmp_path / "work" / variant,
        runtime_commit="c" * 40,
    )


def test_residual_centering_contract_freezes_fresh_g1_one_variable_design():
    validate_frozen_residual_centering_training_contract()
    assert RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER == (
        "none_control",
        "residual_window_center",
    )
    assert RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES == 9600
    assert residual_centering_training_cell_relative_path(
        variant="none_control"
    ) == Path("development/none_control/seed3407")


def test_residual_centering_binder_keeps_shared_protocol_and_changes_only_mode(
    tmp_path: Path,
):
    inputs = _inputs(tmp_path)
    configs = {
        variant: bind_residual_centering_training_config(
            **inputs,
            variant=variant,
            seed=DYNAMIC_FLOOR_M2_SEED,
            work_dir=tmp_path / "work" / variant,
            runtime_commit="c" * 40,
        )
        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
    }
    none = configs["none_control"]
    center = configs["residual_window_center"]
    assert none.model.backbone.custom.georoute_branch_calibration_mode == "none"
    assert (
        center.model.backbone.custom.georoute_branch_calibration_mode
        == "residual_window_center"
    )
    assert none.model.backbone.custom.georoute_roi_extent_floor_cells == 1
    assert center.model.backbone.custom.georoute_roi_extent_floor_cells == 1
    assert none.solver.fp16_compress is False
    assert center.solver.fp16_compress is False
    assert (
        none.georoute_residual_centering_training_binding.shared_protocol_sha256
        == center.georoute_residual_centering_training_binding.shared_protocol_sha256
    )
    assert (
        none.georoute_residual_centering_training_binding.shared_protocol
        == center.georoute_residual_centering_training_binding.shared_protocol
    )
    for variant, cfg in configs.items():
        validate_residual_centering_training_config(
            cfg, variant=variant, phase="train"
        )
    configs["residual_window_center"].optimizer.lr *= 2.0
    with pytest.raises(ValueError, match="matched-training binding is invalid"):
        validate_residual_centering_training_config(
            configs["residual_window_center"],
            variant="residual_window_center",
            phase="train",
        )


@pytest.mark.parametrize(
    ("variant", "mode"),
    [
        ("none_control", "none"),
        ("residual_window_center", "residual_window_center"),
    ],
)
def test_residual_centering_accuracy_binding_is_strict_metric_replay(
    tmp_path: Path, variant: str, mode: str
):
    from mmengine.config import Config

    cfg = _bound(tmp_path, variant=variant)
    accuracy = configure_residual_centering_accuracy(
        cfg,
        variant=variant,
        replay="accuracy_a",
        work_dir=tmp_path / "accuracy" / variant,
        training_config_sha256=canonical_sha256(cfg.to_dict()),
    )
    validate_residual_centering_training_config(
        accuracy, variant=variant, phase="accuracy"
    )
    phase = accuracy.georoute_phase_m_binding
    assert (
        phase.schema_version
        == DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_TRAINING_ACCURACY_SCHEMA
    )
    assert phase.strict_deterministic_algorithms is True
    assert phase.sdp_backend == "math"
    assert phase.tf32_enabled is False
    assert phase.metric_evaluation_enabled is True
    assert accuracy.model.backbone.custom.georoute_branch_calibration_mode == mode
    train_path = tmp_path / f"{variant}_train.py"
    accuracy_path = tmp_path / f"{variant}_accuracy.py"
    cfg.dump(str(train_path))
    accuracy.dump(str(accuracy_path))
    loaded_train = Config.fromfile(str(train_path))
    loaded_accuracy = Config.fromfile(str(accuracy_path))
    assert loaded_accuracy.georoute_phase_m_binding.training_config_sha256 == (
        canonical_sha256(loaded_train.to_dict())
    )
    validate_residual_centering_training_config(
        loaded_accuracy, variant=variant, phase="accuracy"
    )
    assert (
        resolve_dynamic_floor_m2_accuracy_execution_commit(
            accuracy,
            binding=accuracy.georoute_dynamic_floor_m2_binding,
        )
        == "c" * 40
    )
    accuracy.georoute_phase_m_binding.tf32_enabled = True
    with pytest.raises(ValueError, match="matched-accuracy binding is invalid"):
        resolve_dynamic_floor_m2_accuracy_execution_commit(
            accuracy,
            binding=accuracy.georoute_dynamic_floor_m2_binding,
        )


def _branch_payload(*, mode: str, before: float, after: float) -> dict:
    return {
        "records": [
            {
                "dataset_index": 0,
                "video_id": "gate-a",
                "route": {
                    "tubelet_count": 2,
                    "item_count": 3,
                    "branch_calibration": {
                        "schema_version": (
                            "scnr_dynamic_branch_calibration_window_v1"
                        ),
                        "mode": mode,
                        "target": "delta_residual",
                        "scope": (
                            "complete_window_all_valid_candidates"
                            if mode == "residual_window_center"
                            else "disabled"
                        ),
                        "valid_candidate_count": 6,
                        "residual_valid_mean_before": before,
                        "residual_valid_mean_after": after,
                        "changes_q_base": False,
                        "changes_delta_roi": False,
                        "changes_context_zero_modifier": False,
                        "changes_budget_or_role_quota": False,
                        "mean_detached": False,
                    },
                },
            }
        ]
    }


def test_residual_centering_branch_gate_distinguishes_identity_and_centering():
    none = summarize_residual_centering_training_branch(
        _branch_payload(mode="none", before=2.5, after=2.5),
        expected_mode="none",
    )
    center = summarize_residual_centering_training_branch(
        _branch_payload(
            mode="residual_window_center", before=2.5, after=1e-7
        ),
        expected_mode="residual_window_center",
    )
    assert none["identity_required"] is True
    assert center["residual_valid_mean_after_max_abs"] == pytest.approx(1e-7)
    with pytest.raises(ValueError, match="branch receipt is invalid"):
        summarize_residual_centering_training_branch(
            _branch_payload(mode="none", before=2.5, after=2.4),
            expected_mode="none",
        )
    with pytest.raises(ValueError, match="branch receipt is invalid"):
        summarize_residual_centering_training_branch(
            _branch_payload(
                mode="residual_window_center", before=2.5, after=1e-2
            ),
            expected_mode="residual_window_center",
        )


def _stage(*, variant: str, job_id: str, metrics: dict[str, float]) -> dict:
    return {
        "variant": variant,
        "slurm_job_id": job_id,
        "binding": {"shared_protocol_sha256": "a" * 64},
        "accuracy_replays": {"accuracy_a": {"metrics": metrics}},
    }


def test_residual_centering_finalizer_opens_only_paired_cost_after_accuracy_pass(
    monkeypatch,
):
    base = {
        "average_mAP": 50.0,
        "mAP@0.3": 60.0,
        "mAP@0.4": 55.0,
        "mAP@0.5": 50.0,
        "mAP@0.6": 45.0,
        "mAP@0.7": 40.0,
        "high_iou_composite": 42.5,
    }
    centered = {key: value + 1.0 for key, value in base.items()}
    stages = {
        "none_control": _stage(
            variant="none_control", job_id="11", metrics=base
        ),
        "residual_window_center": _stage(
            variant="residual_window_center", job_id="12", metrics=centered
        ),
    }
    monkeypatch.setattr(
        "tools.bata.georoute_residual_centering_training_contract.validate_residual_centering_training_stage_result",
        lambda result, **_kwargs: result,
    )
    result = finalize_residual_centering_training(
        stages,
        expected_commit="c" * 40,
        expected_job_ids={
            "none_control": "11",
            "residual_window_center": "12",
        },
    )
    assert result["status"] == "PASS_ACCURACY_SCREEN_PAIRED_COST_AUTHORIZED"
    assert result["paired_cost_authorized"] is True
    assert result["seeds_3408_3409_opened"] is False
    assert result["center_minus_none_metrics_pp"]["mAP@0.7"] == pytest.approx(
        1.0
    )

    crossing = copy.deepcopy(stages)
    crossing["residual_window_center"]["accuracy_replays"]["accuracy_a"][
        "metrics"
    ]["mAP@0.7"] = 39.0
    held = finalize_residual_centering_training(
        crossing,
        expected_commit="c" * 40,
        expected_job_ids={
            "none_control": "11",
            "residual_window_center": "12",
        },
    )
    assert held["status"] == "HOLD_COMPLETE_ACCURACY_SCREEN_NO_COST"
    assert held["paired_cost_authorized"] is False


def test_residual_centering_incomplete_finalization_has_empty_contrast(monkeypatch):
    monkeypatch.setattr(
        "tools.bata.georoute_residual_centering_training_contract.validate_residual_centering_training_stage_result",
        lambda result, **_kwargs: result,
    )
    result = finalize_residual_centering_training(
        {}, expected_commit="c" * 40
    )
    assert result["status"] == "FAIL_INCOMPLETE_EMPTY_CONTRAST_NO_COST"
    assert result["center_minus_none_metrics_pp"] == {}
    assert result["paired_cost_authorized"] is False


def test_residual_centering_finalizer_converts_malformed_stage_to_fail_closed(
    monkeypatch,
):
    monkeypatch.setattr(
        "tools.bata.georoute_residual_centering_training_contract.validate_residual_centering_training_stage_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AttributeError("malformed nested receipt")
        ),
    )
    result = finalize_residual_centering_training(
        {variant: {} for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER},
        expected_commit="c" * 40,
    )
    assert result["status"] == "FAIL_INCOMPLETE_EMPTY_CONTRAST_NO_COST"
    assert set(result["errors"]) == set(RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER)


def test_residual_centering_deployment_binds_afterany_and_finalizer_job(
    tmp_path: Path,
):
    jobs = {
        "none_control": "11",
        "residual_window_center": "12",
        "finalizer": "13",
    }
    deployment = {
        "schema_version": RESIDUAL_CENTERING_TRAINING_DEPLOYMENT_SCHEMA,
        "status": "DEPLOYED_RESIDUAL_CENTERING_MATCHED_TRAINING_DAG",
        "runtime_commit": "c" * 40,
        "run_root": str(tmp_path.resolve()),
        "variants": list(RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER),
        "jobs": jobs,
        "dependencies": {
            "finalizer": {
                "type": "afterany",
                "predecessors": ["11", "12"],
            }
        },
        "all_jobs_held_until_immutable_receipt": True,
        "partial_survivor_inference_allowed": False,
        "paired_cost_submitted": False,
        "additional_seeds_opened": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    deployment["deployment_sha256"] = canonical_sha256(deployment)
    validated = _validate_deployment(
        deployment,
        run_root=tmp_path.resolve(),
        expected_commit="c" * 40,
        expected_jobs={
            "none_control": "11",
            "residual_window_center": "12",
        },
        finalizer_job_id="13",
    )
    assert validated["deployment_sha256"] == deployment["deployment_sha256"]
    deployment["dependencies"]["finalizer"]["type"] = "afterok"
    with pytest.raises(ValueError, match="deployment receipt is invalid"):
        _validate_deployment(
            deployment,
            run_root=tmp_path.resolve(),
            expected_commit="c" * 40,
            expected_jobs={
                "none_control": "11",
                "residual_window_center": "12",
            },
            finalizer_job_id="13",
        )


def test_residual_centering_slurm_sources_freeze_gpu_and_no_cuda_override():
    stage = (
        ROOT
        / "scripts"
        / "run_georoute_residual_centering_training_stage_slurm.sh"
    ).read_text(encoding="utf-8")
    finalizer = (
        ROOT
        / "scripts"
        / "run_georoute_residual_centering_training_finalizer_slurm.sh"
    ).read_text(encoding="utf-8")
    deployer = (
        ROOT
        / "tools"
        / "bata"
        / "deploy_georoute_residual_centering_training.py"
    ).read_text(encoding="utf-8")
    sources = stage + "\n" + finalizer
    assert sources.count("#SBATCH --gres=gpu:1") == 2
    assert "CUDA_VISIBLE_DEVICES=" not in sources
    assert "source /etc/profile\nset -euo pipefail" in stage
    assert "source /etc/profile\nset -euo pipefail" in finalizer
    assert "taskset -c" in stage
    assert "ABBA" not in stage
    assert 'dependency_type="afterany"' in deployer
    assert "hold=True" in deployer
    assert "release_jobs(ROOT, submitted)" in deployer
    assert '"paired_cost_submitted": False' in deployer
