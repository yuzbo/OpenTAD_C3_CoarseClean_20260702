from pathlib import Path

import pytest

from tools.bata.bafdr_training_contract import (
    validate_optimizer_steps, validate_terminal_checkpoint,
)


def terminal():
    return {
        "epoch": 59, "state_dict_ema": {"weight": 1},
        "total_successful_updates": 6000,
        "optimizer": {"state": {0: {"step": 6000}, 1: {"step": 6000}}},
        "scheduler": {"last_epoch": 6000},
        "update_audit": dict(successful_optimizer_updates=6000,
                             scheduler_updates=6000, ema_updates=6000),
    }


def test_real_terminal_updates_pass():
    validate_terminal_checkpoint(terminal())


@pytest.mark.parametrize("actual", [5994, 5995, 5996, 5997])
def test_old_bafdr_counts_cannot_be_hidden_by_receipt(actual):
    checkpoint = terminal()
    for state in checkpoint["optimizer"]["state"].values():
        state["step"] = actual
    with pytest.raises(RuntimeError, match="actual optimizer"):
        validate_terminal_checkpoint(checkpoint)


@pytest.mark.parametrize("key", ["successful_optimizer_updates", "scheduler_updates", "ema_updates"])
def test_counter_disagreement_fails(key):
    checkpoint = terminal()
    checkpoint["update_audit"][key] -= 1
    with pytest.raises(RuntimeError, match=key):
        validate_terminal_checkpoint(checkpoint)


def test_missing_actual_state_and_partial_parameter_updates_fail():
    with pytest.raises(RuntimeError, match="actual optimizer"):
        validate_optimizer_steps({}, 6000)
    checkpoint = terminal()
    checkpoint["optimizer"]["state"][1]["step"] = 5999
    with pytest.raises(RuntimeError, match="actual optimizer"):
        validate_terminal_checkpoint(checkpoint)
    checkpoint = terminal()
    checkpoint["scheduler"]["last_epoch"] = 5999
    with pytest.raises(RuntimeError, match="actual scheduler"):
        validate_terminal_checkpoint(checkpoint)


def test_teacher_and_evaluator_use_actual_terminal_check():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools/bata/bafdr_k16_fullmatrix_train.py").read_text(encoding="utf-8")
    assert "validate_terminal_checkpoint(checkpoint)" in source
    assert "validate_terminal_checkpoint(loaded_ckpt)" in source
    assert "if args.checkpoint and not args.eval_only:" in source
    assert "len(train_loader) != EXPECTED_UPDATES_PER_EPOCH" in source
    assert "updates_per_epoch=3" in source
    launcher = (root / "scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh").read_text(encoding="utf-8")
    assert 'WORK_DIR_ROOT="${RUN_ROOT}/precheck_work_dirs"' in launcher
