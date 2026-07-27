from pathlib import Path

import pytest

from tools.bata.duca_rime_training import (
    PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE,
    PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS,
    after_checkpoint_saved,
    validate_phase2_baseline_checkpoint_compatibility,
)


def test_rime_retains_only_the_latest_resumable_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    old = checkpoint_dir / "epoch_4.pth"
    current = checkpoint_dir / "epoch_9.pth"
    old.write_bytes(b"old")
    current.write_bytes(b"current")
    removed = after_checkpoint_saved(
        checkpoint_path=current,
        work_dir=tmp_path,
        epoch=9,
        contract={"checkpoint_retention": 1},
    )
    assert removed == [str(old.resolve())]
    assert not old.exists()
    assert current.read_bytes() == b"current"


def test_rime_checkpoint_retention_rejects_a_future_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    current = checkpoint_dir / "epoch_9.pth"
    future = checkpoint_dir / "epoch_14.pth"
    current.write_bytes(b"current")
    future.write_bytes(b"future")
    with pytest.raises(RuntimeError, match="future checkpoint"):
        after_checkpoint_saved(
            checkpoint_path=current,
            work_dir=tmp_path,
            epoch=9,
            contract={"checkpoint_retention": 1},
        )


def test_phase2_baseline_accepts_only_the_historical_unused_uniform_score_net():
    result = validate_phase2_baseline_checkpoint_compatibility(
        missing_keys=[],
        unexpected_keys=reversed(PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS),
    )
    assert result["mode"] == PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE
    assert result["missing_keys"] == []
    assert result["ignored_unexpected_keys"] == sorted(
        PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS
    )


@pytest.mark.parametrize(
    ("missing_keys", "unexpected_keys"),
    [
        (["module.projection.input.weight"], PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS),
        ([], [*PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS, "module.extra.weight"]),
        ([], PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS[:-1]),
    ],
)
def test_phase2_baseline_rejects_any_checkpoint_drift(
    missing_keys,
    unexpected_keys,
):
    with pytest.raises(RuntimeError, match="Phase-2 baseline checkpoint"):
        validate_phase2_baseline_checkpoint_compatibility(
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
