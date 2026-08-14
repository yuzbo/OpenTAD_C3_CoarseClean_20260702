from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/adatad/thumos/duca_dynamic_b_n16r4_protocol.py"
LAUNCHER = ROOT / "scripts/run_duca_dynamic_b_n16r4_future_pilot_pre_run.sh"


def test_static_packet_declares_protocol_and_all_arms():
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    text = CONFIG.read_text(encoding="utf-8")
    for arm in ("dense", "uniform_k384", "dynamic_A", "dynamic_B", "k_shuffle", "no_risk"):
        assert arm in text
    assert "dynamic_outer_k_required=True" in text
    assert "matched_realized_mean_full_stack_cost" in text
    assert tree.body


def test_future_launcher_is_fail_closed_and_non_executing():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "DUCA_EVALUATOR_PRE_RUN_ADMISSION_ARTIFACT:?" in text
    assert "sbatch" not in text and "srun" not in text
    assert "THUMOS14_TRAIN_DATA_PATH" not in text
    assert "tools/train.py" in text
    assert "PRE_RUN-only" in text


def test_dynamic_b_metadata_does_not_claim_unexecuted_f2():
    source = (ROOT / "opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py").read_text(encoding="utf-8")
    assert '"f2": {"enabled": False' in source
    helper = (ROOT / "opentad/models/selectors/duca_dynamic_physical.py").read_text(encoding="utf-8")
    assert 'selected = candidates[: int(k)]' not in helper
    assert 'local_radius == 0' not in helper
