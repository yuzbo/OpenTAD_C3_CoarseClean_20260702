from pathlib import Path

import pytest

from tools.bata.duca_rime_training import after_checkpoint_saved


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
