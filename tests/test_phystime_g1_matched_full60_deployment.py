import ast
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


def test_full60_submitter_freezes_two_arms_and_one_g1a_gate():
    submit = (ROOT / "scripts" / "submit_phystime_g1_matched_full60.sh").read_text(
        encoding="utf-8"
    )
    runner = (
        ROOT / "scripts" / "run_phystime_g1_matched_full60_slurm.sh"
    ).read_text(encoding="utf-8")
    validator = (
        ROOT / "tools" / "bata" / "validate_phystime_g1_matched_full60_artifacts.py"
    ).read_text(encoding="utf-8")

    assert "PHYSTIME_FULL_EPOCHS:-60" in submit
    assert '[[ "${EPOCHS}" == "60" ]]' in submit
    assert '[[ "${SEED}" == "42" ]]' in submit
    assert submit.count("bash scripts/run_phystime_g1a_gate_slurm.sh") == 1
    assert "run_phystime_g1b_sdpq_gate_slurm.sh" not in submit
    for variant in ("selected_axis", "physical_metric"):
        assert f'"{variant}|' in submit
        assert f'"{variant}": "${{jobs[{variant}]}}"' in submit
    assert "g1b_sdpq" not in submit
    assert 'submit --dependency="afterok:${gate_job}"' in submit

    assert "workflow.end_epoch=${EPOCHS}" in runner
    assert "scheduler.max_epoch=${EPOCHS}" in runner
    assert "workflow.val_start_epoch=40" in runner
    assert "workflow.val_eval_interval=2" in runner
    assert "workflow.checkpoint_interval=${EPOCHS}" in runner
    assert "workflow.checkpoint_save_mode=lightweight" in runner
    assert "workflow.checkpoint_include_ema=True" in runner
    assert "FULL_COMPLETE.json" in runner
    assert "EXPECTED_EPOCHS = 60" in validator
    assert "EXPECTED_FINAL_EPOCH = 59" in validator
    assert '"state_dict_ema"' not in validator
    assert "_validate_lightweight_checkpoint" in validator
    assert '"effective_config_sha256"' in validator
    assert '"scheduler.max_epoch": EXPECTED_EPOCHS' in validator
    assert "PHYSTIME_MIN_FREE_KB:-8388608" in submit


def test_full60_validator_freezes_final_epoch_seed_and_variants():
    validator_path = (
        ROOT / "tools" / "bata" / "validate_phystime_g1_matched_full60_artifacts.py"
    )
    module = ast.parse(validator_path.read_text(encoding="utf-8"))
    assignments = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass

    assert assignments["EXPECTED_EPOCHS"] == 60
    assert assignments["EXPECTED_FINAL_EPOCH"] == 59
    assert assignments["EXPECTED_SEED"] == 42
    assert assignments["VARIANT_CONFIGS"] == {
        "selected_axis": "phystime_g1a_selected_axis_native_j192.py",
        "physical_metric": "phystime_g1a_physical_metric_native_j192.py",
    }


def test_full60_arms_share_everything_except_declared_coordinate_mode():
    selected = _load("phystime_g1a_selected_axis_native_j192.py")
    physical = _load("phystime_g1a_physical_metric_native_j192.py")

    for split in ("train", "val", "test"):
        assert _pipeline_step(selected, split, "LoadFrames") == _pipeline_step(
            physical, split, "LoadFrames"
        )
        assert _pipeline_step(
            selected, split, "BuildPhysTimeRawFrameGeometry"
        ) == _pipeline_step(physical, split, "BuildPhysTimeRawFrameGeometry")
        selected_native = _pipeline_step(
            selected, split, "BuildPhysTimeNativeTubeletGeometry"
        )
        physical_native = _pipeline_step(
            physical, split, "BuildPhysTimeNativeTubeletGeometry"
        )
        assert selected_native.pop("coordinate_mode") == "uniform_rank_seconds"
        assert physical_native.pop("coordinate_mode") == "physical_time_seconds"
        assert selected_native == physical_native

    assert selected.optimizer == physical.optimizer
    assert selected.scheduler == physical.scheduler
    assert selected.model.type == physical.model.type == "ActionFormer"
    assert selected.model.rpn_head.type == physical.model.rpn_head.type
    assert (
        selected.model.backbone.backbone.total_frames
        == physical.model.backbone.backbone.total_frames
        == 384
    )
    for cfg in (selected, physical):
        post_types = [
            step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline
        ]
        assert "Interpolate" not in post_types
        assert cfg.model.native_temporal_geometry.expected_token_count == 192
