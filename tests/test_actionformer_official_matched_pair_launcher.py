import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_actionformer_official_matched_pair_n16r4.sbatch"


def test_official_matched_pair_launcher_is_syntax_valid_and_fail_closed():
    subprocess.run(
        ["bash", "-n", LAUNCHER.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = LAUNCHER.read_text(encoding="utf-8")
    assert source.index("source /etc/profile") < source.index("set -u")
    assert "CUDA_VISIBLE_DEVICES=" not in source
    assert 'test "$SCREENING_SEED" = "1234567891"' in source
    assert 'test "$EXPECTED_TERMINAL_EPOCH" = "35"' in source
    assert source.count("run_arm ") == 2
    assert 'run_arm dense "configs/thumos_i3d.yaml"' in source
    assert (
        'run_arm sparse "configs/thumos_i3d_sparsehead_k384_uniform.yaml"'
        in source
    )
    assert source.count("-epoch 35") == 2
    assert "--saveonly" in source
    assert "evaluate_actionformer_raw_predictions.py" in source
    assert "validate_attestation_snapshot" in source
    assert "validate_attestation_live(payload)" not in source
    assert '"paper_main_table_eligible": False' in source
    assert '"end_to_end_cost_claim_allowed": False' in source
    assert "requires_preregistered_multiseed" in source


def test_launcher_never_resumes_or_changes_the_official_seed():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "--resume" not in source
    assert "1234567891" in source
    assert "epoch_035.pth.tar" in source
    assert "state_dict_ema" in source
    assert "optimizer_epochs" in source
    assert "warmup_epochs" in source
