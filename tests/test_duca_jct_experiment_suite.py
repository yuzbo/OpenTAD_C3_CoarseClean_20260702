from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scripts" / "submit_duca_jct_experiment_suite.sh"


def test_duca_jct_suite_submits_all_required_main_and_trainfree_jobs() -> None:
    text = SUITE.read_text(encoding="utf-8")

    assert "run_duca_online_official_adatad_backend_gpu1.sh" in text
    assert "run_duca_must_dynamic_official_adatad_backend_gpu1.sh" in text
    assert "run_duca_trainfree_x3d_interval_grid_gpu0.sh" in text
    assert "run_duca_x3d_official_adatad_backend_gpu1.sh" in text
    assert "run_duca_must_dynamic_x3d_official_adatad_backend_gpu1.sh" in text
    assert "tests/test_duca_joint_training_contract.py" in text
    assert "tests/test_trainfree_x3d_actionness_materialize.py" in text


def test_duca_jct_suite_preserves_x3d_dependency_and_formal_jsonl_contract() -> None:
    text = SUITE.read_text(encoding="utf-8")

    assert "best_x3d_actionness.jsonl" in text
    assert "materialization.json" in text
    assert "DUCA_X3D_ACTIONNESS_JSONL" in text
    assert "--dependency=afterok:${x3d_grid_job}" in text
    assert "DUCA_X3D_FORMAL_PROVIDER" in text
    assert "DUCA_X3D_FORMAL_FRAME_INTERVAL" in text
    assert "best_by_metric" not in text


def test_duca_jct_suite_is_submit_limit_tolerant_and_records_summary() -> None:
    text = SUITE.read_text(encoding="utf-8")

    assert "AssocMaxSubmitJobLimit" in text
    assert "submit_with_retry" in text
    assert "deployment_summary.json" in text
    assert "git pull --ff-only" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=" not in text
