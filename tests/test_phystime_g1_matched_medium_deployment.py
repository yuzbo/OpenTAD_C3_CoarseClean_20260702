from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"


def _load(name):
    return Config.fromfile(str(CONFIG_ROOT / name), lazy_import=False)


def _pipeline_step(cfg, split, step_type):
    for step in cfg.dataset[split].pipeline:
        if step["type"] == step_type:
            return dict(step)
    raise AssertionError(f"{split} pipeline has no {step_type}")


def test_matched_medium_submitter_freezes_three_arms_and_one_gate():
    submit = (ROOT / "scripts" / "submit_phystime_g1_matched_medium.sh").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "scripts" / "run_phystime_g1_matched_medium_slurm.sh").read_text(
        encoding="utf-8"
    )

    assert "PHYSTIME_MEDIUM_EPOCHS:-20" in submit
    assert '[[ "${EPOCHS}" == "20" ]]' in submit
    assert '[[ "${SEED}" == "42" ]]' in submit
    assert submit.count('bash scripts/run_phystime_g1a_gate_slurm.sh') == 1
    assert submit.count('bash scripts/run_phystime_g1b_sdpq_gate_slurm.sh') == 1
    for variant in ("selected_axis", "physical_metric", "g1b_sdpq"):
        assert f'"{variant}|' in submit
        assert f'"{variant}": "${{jobs[{variant}]}}"' in submit
    assert 'submit --dependency="afterok:${gate_job}"' in submit

    assert "validate_phystime_g1_matched_medium_artifacts.py" in runner
    assert "workflow.checkpoint_save_mode=lightweight" in runner
    assert "workflow.checkpoint_include_ema=True" in runner
    assert '"state_dict_ema"' in (
        ROOT / "tools" / "bata" / "validate_phystime_g1_matched_medium_artifacts.py"
    ).read_text(encoding="utf-8")
    assert "MEDIUM_COMPLETE.json" in runner
    assert "epoch_5.pth" not in runner


def test_matched_medium_configs_share_sampling_backbone_and_schedule():
    selected = _load("phystime_g1a_selected_axis_native_j192.py")
    physical = _load("phystime_g1a_physical_metric_native_j192.py")
    sdpq = _load("phystime_g1b_sdpq_pool_native_j192.py")

    for split in ("train", "val", "test"):
        selected_load = _pipeline_step(selected, split, "LoadFrames")
        physical_load = _pipeline_step(physical, split, "LoadFrames")
        sdpq_load = _pipeline_step(sdpq, split, "LoadFrames")
        assert selected_load == physical_load == sdpq_load

        selected_raw = _pipeline_step(selected, split, "BuildPhysTimeRawFrameGeometry")
        physical_raw = _pipeline_step(physical, split, "BuildPhysTimeRawFrameGeometry")
        sdpq_raw = _pipeline_step(sdpq, split, "BuildPhysTimeRawFrameGeometry")
        assert selected_raw == physical_raw == sdpq_raw

    assert selected.optimizer == physical.optimizer == sdpq.optimizer
    assert selected.scheduler == physical.scheduler == sdpq.scheduler
    assert selected.workflow == physical.workflow == sdpq.workflow
    assert (
        selected.model.backbone.backbone.total_frames
        == physical.model.backbone.backbone.total_frames
        == sdpq.model.backbone.backbone.total_frames
        == 384
    )
    for cfg in (selected, physical, sdpq):
        post_types = [
            step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline
        ]
        assert "Interpolate" not in post_types
        assert cfg.model.native_temporal_geometry.expected_token_count == 192


def test_matched_medium_changes_only_declared_geometry_or_head_family():
    selected = _load("phystime_g1a_selected_axis_native_j192.py")
    physical = _load("phystime_g1a_physical_metric_native_j192.py")
    sdpq = _load("phystime_g1b_sdpq_pool_native_j192.py")

    for split in ("train", "val", "test"):
        selected_native = _pipeline_step(
            selected, split, "BuildPhysTimeNativeTubeletGeometry"
        )
        physical_native = _pipeline_step(
            physical, split, "BuildPhysTimeNativeTubeletGeometry"
        )
        sdpq_native = _pipeline_step(
            sdpq, split, "BuildPhysTimeNativeTubeletGeometry"
        )
        assert selected_native.pop("coordinate_mode") == "uniform_rank_seconds"
        assert physical_native.pop("coordinate_mode") == "physical_time_seconds"
        assert sdpq_native.pop("coordinate_mode") == "physical_time_seconds"
        assert selected_native == physical_native == sdpq_native

    assert selected.model.type == physical.model.type == "ActionFormer"
    assert sdpq.model.type == "PhysTimeTAD"
    assert physical.model.rpn_head.type == selected.model.rpn_head.type
    assert sdpq.model.rpn_head.type == "SupportDecoupledPhysicalQueryHead"
