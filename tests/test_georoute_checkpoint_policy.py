from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def test_unannotated_checkpoint_uses_atomic_publish_without_temp_residue(
    tmp_path: Path,
    monkeypatch,
):
    fake_torch = SimpleNamespace(
        save=lambda _payload, path: Path(path).write_bytes(b"checkpoint")
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    spec = importlib.util.spec_from_file_location(
        "georoute_checkpoint_writer_under_test",
        ROOT / "opentad" / "utils" / "checkpoint.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stateful = SimpleNamespace(state_dict=lambda: {"value": 1})

    module.save_checkpoint(
        stateful,
        None,
        stateful,
        stateful,
        19,
        work_dir=str(tmp_path),
    )

    checkpoint_dir = tmp_path / "checkpoint"
    assert (checkpoint_dir / "epoch_19.pth").read_bytes() == b"checkpoint"
    assert not list(checkpoint_dir.glob("*.tmp*"))


def test_georoute_config_and_runner_enforce_one_final_checkpoint():
    config = (
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "georoute_adatad_development_base.py"
    ).read_text(encoding="utf-8")
    train = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")
    runner = (
        ROOT / "tools" / "bata" / "georoute_stage_runner.py"
    ).read_text(encoding="utf-8")

    assert 'checkpoint_policy="final_only"' in config
    assert 'policy == "final_only"' in train
    assert 'checkpoint_dir.glob("*.pth")' in runner
    assert 'checkpoint_dir.glob("*.tmp*")' in runner
