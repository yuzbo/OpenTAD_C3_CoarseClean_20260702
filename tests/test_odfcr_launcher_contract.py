from pathlib import Path

from libs.core import load_config
from tools.aggregate_odfcr_internal_matrices import EXPECTED_DEV_SEEDS


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_odfcr_internal_factorial_n16r4.sbatch"
G2_LAUNCHER = (
    ROOT / "scripts" / "aggregate_odfcr_internal_factorial_n16r4.sbatch"
)
K384_LAUNCHER = ROOT / "scripts" / "run_odfcr_k384_replay_n16r4.sbatch"
G3_LAUNCHER = ROOT / "scripts" / "aggregate_odfcr_k384_n16r4.sbatch"
ARMS = ("d1_off", "d1_all", "d3_off", "d3_all")


def test_odfcr_launcher_uses_frozen_seeds_and_four_serial_arms():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2" in text
    assert "#SBATCH --mem=" not in text
    assert EXPECTED_DEV_SEEDS == (
        2026073101,
        2026073102,
        2026073103,
    )
    for seed in EXPECTED_DEV_SEEDS:
        assert str(seed) in text
    arm_calls = [
        'run_arm d1_off "configs/thumos_i3d_odfcr_dev_d1_off.yaml"',
        'run_arm d1_all "configs/thumos_i3d_odfcr_dev_d1_all.yaml"',
        'run_arm d3_off "configs/thumos_i3d_odfcr_dev_d3_off.yaml"',
        'run_arm d3_all "configs/thumos_i3d_odfcr_dev_d3_all.yaml"',
    ]
    positions = [text.index(call) for call in arm_calls]
    assert positions == sorted(positions)


def test_odfcr_launcher_preserves_slurm_cuda_and_validation_only_data():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash\n")
    assert text.index("set -o pipefail") < text.index("source /etc/profile")
    assert text.index("source /etc/profile") < text.index("set -eu")
    assert text.index("source /etc/profile") < text.index(
        "module load cuda/11.8"
    )
    assert "set -eo pipefail" not in text
    assert 'test -n "${CUDA_VISIBLE_DEVICES:-}"' in text
    assert "export CUDA_VISIBLE_DEVICES=" not in text
    assert "ODFCR_PREVIOUS_HOLDOUT_MANIFEST" in text
    assert "ODFCR_INTERNAL_HOLDOUT_MANIFEST" in text
    for arm in ARMS:
        relative = "configs/thumos_i3d_odfcr_dev_{:s}.yaml".format(arm)
        assert relative in text
        cfg = load_config(str(ROOT / relative))
        assert cfg["train_split"] == ["validation"]
        assert cfg["val_split"] == ["validation"]
        assert "test" not in cfg["train_split"]
        assert "test" not in cfg["val_split"]


def test_odfcr_launcher_g0_and_evidence_chain_are_fail_closed():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'ODFCR_RUN_MODE="${ODFCR_RUN_MODE:-factorial}"' in text
    assert "g0_only|factorial" in text
    assert 'if [ "$ODFCR_RUN_MODE" = g0_only ]; then' in text
    assert "python tools/" not in text
    required_modules = (
        "tools.validate_odfcr_g0_equivalence",
        "tools.evaluate_odfcr_internal_predictions",
        "tools.finalize_odfcr_internal_matrix",
    )
    for module in required_modules:
        assert "python -m " + module in text
    assert text.index("tools.validate_odfcr_g0_equivalence") < text.index(
        "run_arm d1_off"
    )
    assert text.index("run_arm d3_all") < text.index(
        "tools.finalize_odfcr_internal_matrix"
    )
    assert "actionformer_odfcr_engineering_failure_v1" in text
    assert "ACTIONFORMER_ODFCR_INTERNAL_MATRIX_COMPLETE" in text
    assert "tests/test_odfcr_k384_contract.py" in text


def test_odfcr_launcher_does_not_train_or_tune_k384():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "run_odfcr_k384_replay" not in text
    assert "--budget" not in text
    assert "2026073100" not in text


def test_odfcr_yaml_support_tokens_remain_strings_after_parsing():
    for arm in ARMS:
        cfg = load_config(
            str(
                ROOT
                / "configs"
                / "thumos_i3d_odfcr_dev_{:s}.yaml".format(arm)
            )
        )
        support = cfg["model"]["odfcr_head"]["residual_execution_support"]
        assert type(support) is str
        assert support == ("all_valid" if arm.endswith("_all") else "off")


def test_odfcr_followup_launchers_separate_g2_from_conditional_g3():
    g2_text = G2_LAUNCHER.read_text(encoding="utf-8")
    replay_text = K384_LAUNCHER.read_text(encoding="utf-8")
    g3_text = G3_LAUNCHER.read_text(encoding="utf-8")
    for text in (g2_text, replay_text, g3_text):
        assert text.index("set -o pipefail") < text.index(
            "source /etc/profile"
        )
        assert text.index("source /etc/profile") < text.index("set -eu")
        assert "set -eo pipefail" not in text
    assert "tools.aggregate_odfcr_internal_matrices" in g2_text
    assert "ODFCR_G2_AGGREGATE_COMPLETE.json" in g2_text
    assert "#SBATCH --array=0-2" in replay_text
    assert "tools.run_odfcr_k384_replay" in replay_text
    assert 'assert record["residual_utility_gate_pass"] is True' in replay_text
    assert "--g2-aggregate" in replay_text
    assert "tools.aggregate_odfcr_k384_replays" in g3_text
    assert "ODFCR_G3_AGGREGATE_COMPLETE.json" in g3_text


def test_odfcr_identity_reference_is_not_a_fifth_training_arm():
    text = LAUNCHER.read_text(encoding="utf-8")
    reference = "configs/thumos_i3d_odfcr_dev_dense_reference.yaml"
    assert reference not in text
    reference_cfg = load_config(str(ROOT / reference))
    d3_off = load_config(
        str(ROOT / "configs/thumos_i3d_odfcr_dev_d3_off.yaml")
    )
    intervention = d3_off["model"].pop("odfcr_head")
    assert d3_off == reference_cfg
    assert intervention["scaffold_num_layers"] == 3
    assert intervention["residual_enabled"] is False
