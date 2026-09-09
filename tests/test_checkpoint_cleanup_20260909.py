import json
from pathlib import Path

import pytest

from tools.bata import prune_owned_checkpoints_20260909 as cleanup


def snapshot(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test checkpoint")
    return path


@pytest.fixture
def fake_validation(monkeypatch):
    def record(path):
        stat = path.stat()
        return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    inode=stat.st_ino, device=stat.st_dev, allocated_bytes=stat.st_size)

    monkeypatch.setattr(cleanup, "file_record", record)
    monkeypatch.setattr(cleanup, "validate_weights", lambda path, epoch: {"epoch": epoch})
    monkeypatch.setattr(cleanup, "assert_not_running", lambda roots: [])


def test_keep_latest_per_run_and_remove_earlier_best(tmp_path, fake_validation):
    old = snapshot(tmp_path, "arm1/checkpoint/epoch_9.pth")
    last = snapshot(tmp_path, "arm1/checkpoint/epoch_59.pth")
    best = snapshot(tmp_path, "arm1/test_guided/best_test.pth")
    (best.parent / "best_test_metrics.json").write_text(json.dumps({"epoch": 49}))
    other = snapshot(tmp_path, "arm2/checkpoint/epoch_14.pth")
    metadata = Path(str(last) + ".metadata.json")
    metadata.write_text("{}")
    unrelated = snapshot(tmp_path, "pretrained/model.pth")
    plan = cleanup.make_plan({}, [tmp_path])
    assert len(plan["runs"]) == 2
    assert {run["kept"]["path"] for run in plan["runs"]} == {str(last), str(other)}
    result = cleanup.apply_plan(plan, [tmp_path])
    assert result["deleted_files"] == 2
    assert not old.exists() and not best.exists()
    assert last.exists() and other.exists() and metadata.exists() and unrelated.exists()


def test_corrupt_latest_falls_back_without_losing_last_usable(tmp_path, fake_validation, monkeypatch):
    earlier = snapshot(tmp_path, "arm/checkpoint/epoch_54.pth")
    snapshot(tmp_path, "arm/checkpoint/epoch_59.pth")

    def validate(path, epoch):
        if epoch == 59:
            raise RuntimeError("non-finite state_dict")
        return {"epoch": epoch}

    monkeypatch.setattr(cleanup, "validate_weights", validate)
    run = cleanup.make_plan({}, [tmp_path])["runs"][0]
    assert run["kept"]["path"] == str(earlier)
    assert len(run["invalid_candidates"]) == 1


def test_no_usable_checkpoint_means_no_deletions(tmp_path, fake_validation, monkeypatch):
    snapshot(tmp_path, "arm/checkpoint/epoch_59.pth")
    monkeypatch.setattr(cleanup, "validate_weights", lambda *args: (_ for _ in ()).throw(RuntimeError("corrupt")))
    run = cleanup.make_plan({}, [tmp_path])["runs"][0]
    assert run["status"] == "BLOCKED_NO_USABLE_CHECKPOINT"
    assert run["kept"] is None and run["remove"] == []


def test_changed_retained_checkpoint_aborts_before_any_delete(tmp_path, fake_validation):
    old = snapshot(tmp_path, "arm/checkpoint/epoch_9.pth")
    last = snapshot(tmp_path, "arm/checkpoint/epoch_59.pth")
    plan = cleanup.make_plan({}, [tmp_path])
    last.write_bytes(b"changed after CPU validation")
    with pytest.raises(RuntimeError, match="changed since validation"):
        cleanup.apply_plan(plan, [tmp_path])
    assert old.exists()
