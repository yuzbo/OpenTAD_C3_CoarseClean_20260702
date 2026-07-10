from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scripts" / "submit_duca_jct_experiment_suite.sh"


def test_duca_jct_suite_is_fail_closed_as_a_legacy_diagnostic_matrix() -> None:
    text = SUITE.read_text(encoding="utf-8")

    assert "ALLOW_LEGACY_DIAGNOSTIC_SUITE" in text
    assert "legacy suite mixes padded-cap MUST and X3D appendix jobs" in text
    assert "run_duca_online_official_adatad_backend_gpu1.sh" in text
    assert "run_duca_online_official_adatad_budget_curve_gpu1.sh" in text
    assert "run_duca_must_dynamic_official_adatad_backend_gpu1.sh" in text
    assert "run_duca_must_dynamic_budget_target_curve_gpu1.sh" in text
    assert "run_duca_trainfree_x3d_interval_grid_gpu0.sh" in text
    assert "run_duca_x3d_official_adatad_backend_gpu1.sh" in text
    assert "run_duca_must_dynamic_x3d_official_adatad_backend_gpu1.sh" in text
    assert "monitor_duca_jct_experiment_suite.py" in text
    assert "collect_duca_jct_paper_evidence.py" in text
    assert "run_duca_official_adatad_one_step_grad_proof.py" in text
    assert "duca_jct_one_step_grad_proof.json" in text
    assert "tests/test_duca_joint_training_contract.py" in text
    assert "tests/test_duca_jct_one_step_grad_proof.py" in text
    assert "tests/test_trainfree_x3d_actionness_materialize.py" in text
    assert "tests/test_duca_jct_suite_monitor.py" in text
    assert "tests/test_duca_jct_paper_evidence.py" in text


def test_duca_jct_suite_registers_fixed_and_dynamic_budget_studies() -> None:
    text = SUITE.read_text(encoding="utf-8")

    assert "duca_budget_curve_job" in text
    assert "duca_must_target_curve_job" in text
    assert "DUCA_ONLINE_BUDGETS" in text
    assert "128 192 256 320 384" in text
    assert "DUCA_MUST_TARGETS" in text
    assert "128 192 256 320" in text
    assert "fixed_budget_curve" in text
    assert "dynamic_target_curve" in text
    assert "SLURM_JOB_ID" in text
    assert r"MASTER_PORT=\$((20000" in text
    assert r"MASTER_PORT_BASE=\$((25000" in text


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
    assert "deployment_summary.pending.json" in text
    assert text.index("write_deployment_summary pending") < text.index('tests_job="$(submit_with_retry')
    assert "git pull --ff-only" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=" not in text
