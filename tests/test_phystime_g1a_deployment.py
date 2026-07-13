from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(name):
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_g1a_launchers_are_slurm_fail_closed_and_gate_pilots():
    gate = _text("run_phystime_g1a_gate_slurm.sh")
    pilot = _text("run_phystime_g1a_pilot_slurm.sh")
    submit = _text("submit_phystime_g1a_pilot.sh")
    artifact_validator = (
        ROOT / "tools" / "bata" / "validate_phystime_g1a_pilot_artifacts.py"
    ).read_text(encoding="utf-8")

    assert '[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]' in gate
    assert '[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]' in pilot
    assert "--gpu-bind=map_gpu:1" not in submit
    assert "audit_phystime_g0_native_geometry.py" in gate
    assert "validate_phystime_g1a_track.py" in gate
    assert "run_phystime_g1a_real_gate.py" in gate
    assert "g1a_contract.json" in submit
    assert "PHYSTIME_G1A_CONTRACT_JSON" in gate
    assert "PHYSTIME_G1A_CONTRACT_JSON" in pilot
    assert "afterok:${gate_job}" in submit
    assert "phystime_g1a_selected_axis_native_j192.py" in submit
    assert "phystime_g1a_physical_metric_native_j192.py" in submit
    assert "Interpolate" in pilot
    assert "G1a pilot forbids J192-to-K384 interpolation" in pilot
    assert 'PHYSTIME_G1A_PILOT_EPOCHS:-6' in pilot
    assert "cfg.workflow.checkpoint_interval = int(epochs)" in pilot
    assert "workflow.checkpoint_interval=${PILOT_EPOCHS}" in pilot
    assert "workflow.checkpoint_interval=1" not in pilot
    assert "post_processing.save_dict=True" in pilot
    assert "evaluation_metrics.json" in artifact_validator
    assert "result_detection.json" in artifact_validator
    assert '[[ "${PILOT_EPOCHS}" == "6" ]]' in pilot
    assert "gpu1_id0" in artifact_validator
    assert "validate_phystime_g1a_pilot_artifacts.py" in pilot
    assert "PHYSTIME_EXPECTED_COMMIT" in gate
    assert "PHYSTIME_EXPECTED_TREE" in gate
    assert "PHYSTIME_EXPECTED_COMMIT" in pilot
    assert "PHYSTIME_EXPECTED_TREE" in pilot
    assert '[[ ! -e "${RUN_DIR}" ]]' in pilot
    assert '[[ ! -e "${RUN_ROOT}" ]]' in submit
    assert "g1b_g2_status" in submit


def test_g1a_real_gate_evaluates_tail_windows_with_the_test_dataset():
    gate = (
        ROOT / "tools" / "bata" / "run_phystime_g1a_real_gate.py"
    ).read_text(encoding="utf-8")

    assert "build_dataset(cfg.dataset.test)" in gate
    assert "tail_datasets = {}" in gate
    assert "tail_datasets[name] = tail_dataset" in gate
    assert "tail_datasets[name],\n            tail_video_names[name]" in gate
    assert "tail_datasets[name].class_map" in gate
    assert '"test_videos": _directory_inventory(cfg.dataset.test.data_path)' in gate
