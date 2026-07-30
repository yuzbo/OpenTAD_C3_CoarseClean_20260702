from pathlib import Path

from libs.core import load_config
from tools.aggregate_dcsr_internal_pairs import EXPECTED_DEV_SEEDS


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_dcsr_g0_g1_internal_n16r4.sbatch"


def test_dcsr_launcher_uses_three_disjoint_development_seeds():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2" in text
    assert "#SBATCH --mem=" not in text
    assert EXPECTED_DEV_SEEDS == (
        2026073001,
        2026073002,
        2026073003,
    )
    for seed in EXPECTED_DEV_SEEDS:
        assert str(seed) in text
    final_seeds = {
        1234567891,
        2234567891,
        3234567891,
        4234567891,
        5234567891,
    }
    assert set(EXPECTED_DEV_SEEDS).isdisjoint(final_seeds)


def test_dcsr_launcher_preserves_slurm_cuda_and_validation_only_holdout():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash\n")
    assert text.index("source /etc/profile") < text.index("set -u")
    assert text.index("source /etc/profile") < text.index("module load cuda/11.8")
    assert 'test -n "${CUDA_VISIBLE_DEVICES:-}"' in text
    assert "export CUDA_VISIBLE_DEVICES=" not in text
    assert "thumos_i3d_dcsr_dev_dense.yaml" in text
    assert "thumos_i3d_dcsr_dev_g1_uniform.yaml" in text
    assert "configs/thumos_i3d.yaml" not in text
    assert "configs/thumos_i3d_dcsr_g1_uniform.yaml" not in text

    for relative in (
        "configs/thumos_i3d_dcsr_dev_dense.yaml",
        "configs/thumos_i3d_dcsr_dev_g1_uniform.yaml",
    ):
        cfg = load_config(str(ROOT / relative))
        assert cfg["train_split"] == ["validation"]
        assert cfg["val_split"] == ["validation"]
        assert "test" not in cfg["train_split"]
        assert "test" not in cfg["val_split"]


def test_dcsr_launcher_has_fail_closed_direct_g0_only_mode():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'DCSR_RUN_MODE="${DCSR_RUN_MODE:-g1_pair}"' in text
    assert "g0_only|g1_pair" in text
    assert 'if [ "$DCSR_RUN_MODE" = g0_only ]; then' in text
    assert "python tools/" not in text
    assert "python -m tools.validate_dcsr_g0_equivalence" in text
    assert "python -m tools.evaluate_dcsr_internal_predictions" in text
    assert "python -m tools.finalize_dcsr_internal_pair" in text
    assert text.index("tools.validate_dcsr_g0_equivalence") < text.index(
        'if [ "$DCSR_RUN_MODE" = g0_only ]; then'
    )
    assert text.index('if [ "$DCSR_RUN_MODE" = g0_only ]; then') < text.index(
        "run_arm dense"
    )
    assert "ACTIONFORMER_DCSR_G0_ONLY_COMPLETE" in text
    for required in (
        "CANDIDATE_ROOT",
        "RUN_ROOT",
        "EXPECTED_CANDIDATE_COMMIT",
        "EXPECTED_CANDIDATE_TREE",
        "ACTIONFORMER_PYTHON_ENV",
        "ACTIONFORMER_NMS_EXTENSION",
    ):
        assert f'${{{required}:?missing {required}}}' in text


def test_dcsr_launcher_treats_negative_metrics_as_model_results():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    aggregator = (
        ROOT / "tools" / "aggregate_dcsr_internal_pairs.py"
    ).read_text(encoding="utf-8")
    assert "tools.finalize_dcsr_internal_pair" in launcher
    assert '"g1_gate_pass": g1_gate_pass' in aggregator
    assert "raise SystemExit(3)" not in aggregator
    assert '"next_step_if_fail": "terminate SparseHead route"' in aggregator
