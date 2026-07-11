from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "run_phystime_adatad_full_train_gpu1.sh"
SUBMIT = ROOT / "scripts" / "submit_phystime_adatad_head_comparison.sh"


def test_training_launcher_is_raw_video_gate_locked_and_auditable():
    text = TRAIN.read_text(encoding="utf-8")
    assert '[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]]' in text
    assert "SLURM_JOB_ID" in text
    assert "command -v module" in text
    assert "PHYSTIME_REAL_GATE_JSON" in text
    assert 'payload.get("gate_pass") is True' in text
    assert 'payload.get("git_commit") == commit' in text
    assert 'payload.get("checkpoint_sha256") == checkpoint_sha256' in text
    assert "OPENTAD_THUMOS14_ANNOTATION" in text
    assert "OPENTAD_THUMOS14_CLASS_MAP" in text
    assert "OPENTAD_THUMOS14_TRAIN_VIDEOS" in text
    assert "OPENTAD_THUMOS14_TEST_VIDEOS" in text
    assert "PHYSTIME_VIDEOMAE_CHECKPOINT" in text
    assert "tools/train.py" in text
    assert "model.backbone.custom.pretrain" in text
    assert "TRAINING_COMPLETE" in text
    assert "peak_gpu_memory_mb" in text
    assert "wall_time_sec" in text
    assert "FEATURE" in text and "PATH" in text
    assert "LoadFeats" not in text
    assert "--not_eval" not in text


def test_submission_is_gate_dependent_and_has_exactly_three_heads():
    text = SUBMIT.read_text(encoding="utf-8")
    configs = (
        "selected_axis_adatad_sparse_k384.py",
        "physical_grid_adatad_sparse_k384.py",
        "phystime_adatad_sparse_k384.py",
    )
    for config in configs:
        assert text.count(config) == 1
    assert 'variants=(' in text
    assert 'gate_job="$(submit' in text
    assert 'afterok:${gate_job}' in text
    assert 'submit --dependency="afterok:${gate_job}"' in text
    assert "run_phystime_adatad_gate_gpu1.sh" in text
    assert "run_phystime_adatad_full_train_gpu1.sh" in text
    assert '"formal_job_count": 3' in text
    assert '"logical_window": 768' in text
    assert '"decoded_frame_budget": 384' in text
    assert '"phase2_status": "held"' in text
    assert "PHYSTIME_SEED='42'" in text
    assert "prepare_phystime_thumos_i3d" not in text
    assert "PHYSTIME_FEATURE_PATH" not in text
    assert "LoadFeats" not in text
