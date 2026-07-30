from pathlib import Path

from libs.core import load_config
from tools.aggregate_dcsr_internal_pairs import EXPECTED_DEV_SEEDS


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_dcsr_g0_g1_internal_n16r4.sbatch"


def test_dcsr_launcher_uses_three_disjoint_development_seeds():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2" in text
    assert "#SBATCH --mem=55G" in text
    assert "#SBATCH --mem=64G" not in text
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
    assert text.index("source /etc/profile") < text.index("set -u")
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


def test_dcsr_launcher_treats_negative_metrics_as_model_results():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    aggregator = (
        ROOT / "tools" / "aggregate_dcsr_internal_pairs.py"
    ).read_text(encoding="utf-8")
    assert "finalize_dcsr_internal_pair.py" in launcher
    assert '"g1_gate_pass": g1_gate_pass' in aggregator
    assert "raise SystemExit(3)" not in aggregator
    assert '"next_step_if_fail": "terminate SparseHead route"' in aggregator
