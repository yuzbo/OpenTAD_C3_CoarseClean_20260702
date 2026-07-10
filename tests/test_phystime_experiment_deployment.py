import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pilot_manifest_has_gate_dependencies_and_held_phase2():
    manifest = json.loads(
        (ROOT / "docs" / "evaluation" / "phystime-tad-pilot-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["base_commit"] == "5a46ea6"
    assert len(manifest["pilots"]) == 7
    assert manifest["phases"][1]["depends_on"] == ["data"]
    assert manifest["phases"][2]["depends_on"] == ["real_gate"]
    assert manifest["phases"][3]["auto_submit"] is False


def test_submission_script_binds_all_pilots_to_real_gate_and_forbids_old_routes():
    script = (ROOT / "scripts" / "submit_phystime_tad_experiment_track.sh").read_text(encoding="utf-8")

    for experiment_id in (
        "phys_support_k384_s42",
        "phys_point_k384_s42",
        "phys_nodisc_k384_s42",
        "selected_k384_s42",
        "timestamp_k384_s42",
        "phys_support_k192_s42",
        "phys_support_k768_s42",
    ):
        assert experiment_id in script
    assert 'afterok:${data_job}' in script
    assert 'afterok:${gate_job}' in script
    assert 'write_job "${data_script}" phystime_data 1' in script
    for forbidden in ("DUCA", "X3D", "ledger", "actionness"):
        assert forbidden not in script


def test_data_and_training_launchers_are_fail_closed():
    data_script = (ROOT / "scripts" / "prepare_phystime_thumos_i3d_n16r4.sh").read_text(encoding="utf-8")
    train_script = (ROOT / "scripts" / "run_phystime_feature_full_train_gpu1.sh").read_text(encoding="utf-8")
    gate_script = (ROOT / "scripts" / "run_phystime_tad_gate0b_gpu1.sh").read_text(encoding="utf-8")

    assert "PHYSTIME_MIN_FEATURE_FILES" in data_script
    assert "data_ready.json" in data_script
    assert "original_feature_ownership_cells" in data_script
    assert "gdown --continue" in data_script
    assert "--fuzzy" not in data_script
    assert "PHYSTIME_DOWNLOAD_PROXY" in data_script
    assert "PHYSTIME_DOWNLOAD_ATTEMPTS" in data_script
    assert "download attempt ${attempt}/${DOWNLOAD_ATTEMPTS} failed" in data_script
    assert "data_ready.json" in train_script
    assert "torchrun" in train_script.lower()
    assert "TRAINING_COMPLETE" in train_script
    assert "BASH_SOURCE[0]" in train_script
    assert "SLURM_SUBMIT_DIR" not in train_script
    assert "BASH_SOURCE[0]" in gate_script
    assert "SLURM_SUBMIT_DIR" not in gate_script


def test_submission_uses_parseable_job_ids_and_bounded_retry():
    script = (ROOT / "scripts" / "submit_phystime_tad_experiment_track.sh").read_text(
        encoding="utf-8"
    )

    assert "sbatch --parsable" in script
    assert "PHYSTIME_SUBMIT_RETRIES" in script
    assert "PHYSTIME_DOWNLOAD_PROXY" in script
    assert "export HOME=" in script
    assert "XDG_CACHE_HOME" in script
    assert "failed after ${retries} attempts" in script
