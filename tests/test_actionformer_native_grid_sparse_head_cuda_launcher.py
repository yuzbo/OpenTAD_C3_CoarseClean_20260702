import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "scripts"
    / "run_actionformer_native_grid_k384_cuda_gate_n16r4.sbatch"
)


def test_cuda_gate_launcher_is_syntax_valid_and_fail_closed():
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
    assert ": \"${EXPECTED_CANDIDATE_COMMIT:?" in source
    assert ": \"${EXPECTED_SOURCE_DIFF_SHA256:?" in source
    assert "--budget 384" in source
    assert '"paper_metric_claim_allowed": False' in source
    assert '"end_to_end_wall_clock_claim_allowed": False' in source
