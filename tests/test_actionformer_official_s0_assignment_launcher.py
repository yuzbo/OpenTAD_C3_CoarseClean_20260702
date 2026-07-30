from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "run_actionformer_official_s0_assignment_audit_n16r4.sbatch"
)


def test_launcher_is_slurm_only_fail_closed_and_diagnostic():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in source
    assert 'test -n "${SLURM_JOB_ID:-}"' in source
    assert 'test -n "${CUDA_VISIBLE_DEVICES:-}"' in source
    assert "git -C \"$CANDIDATE_ROOT\" status --porcelain" in source
    assert "git -C \"$AUDIT_ROOT\" status --porcelain" in source
    assert "validate_actionformer_s0_screening.py" in source
    assert "audit_actionformer_s0_assignment_support.py" in source
    assert "paper_main_table_eligible" in source
    assert "primary_result_allowed" in source
    assert "ENGINEERING_FAILURE.json" in source
    assert "train.py" not in source
    assert "eval.py" not in source
    assert "optimizer" not in source
