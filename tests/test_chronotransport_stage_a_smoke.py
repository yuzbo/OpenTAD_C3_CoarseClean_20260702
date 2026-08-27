from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from opentad.cores.test_engine import iter_limited_batches


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py"


def test_iter_limited_batches_stops_after_requested_smoke_budget() -> None:
    assert list(iter_limited_batches(range(5), max_batches=1)) == [0]
    assert list(iter_limited_batches(range(3), max_batches=None)) == [0, 1, 2]


def test_iter_limited_batches_rejects_nonpositive_explicit_limit() -> None:
    with pytest.raises(ValueError, match="max_batches must be positive"):
        list(iter_limited_batches(range(3), max_batches=0))


def test_stage_a_config_is_mmengine_pretty_printable(monkeypatch) -> None:
    from mmengine.config import Config

    monkeypatch.setenv("YUZIBO_ROOT", "/data/run01/sczc063/yuzibo")
    cfg = Config.fromfile(str(CONFIG))
    assert "chronotransport_contract" in cfg.pretty_text


def test_checkpoint_key_classifier_allows_only_chronotransport_parameters() -> None:
    from tools.bata.check_chronotransport_checkpoint import classify_incompatible_keys

    result = classify_incompatible_keys(
        missing=(
            "backbone.backbone.chronotransport.transport.up.weight",
            "backbone.backbone.chronotransport.risk.head.bias",
        ),
        unexpected=(),
        allow_chronotransport_missing=True,
    )
    assert result["status"] == "PASS"
    assert result["allowed_chronotransport_missing"] == 2
    assert result["forbidden_missing"] == []


def test_checkpoint_key_classifier_rejects_non_chronotransport_drift() -> None:
    from tools.bata.check_chronotransport_checkpoint import classify_incompatible_keys

    result = classify_incompatible_keys(
        missing=("backbone.backbone.blocks.0.norm1.weight",),
        unexpected=("selector.private_head.weight",),
        allow_chronotransport_missing=True,
    )
    assert result["status"] == "FAIL"
    assert result["forbidden_missing"] == ["backbone.backbone.blocks.0.norm1.weight"]
    assert result["unexpected"] == ["selector.private_head.weight"]


def test_checkpoint_key_classifier_requires_dynamic_parameters_for_learned_mode() -> None:
    from tools.bata.check_chronotransport_checkpoint import classify_incompatible_keys

    key = "backbone.backbone.chronotransport.transport.up.weight"
    result = classify_incompatible_keys(
        missing=(key,),
        unexpected=(),
        allow_chronotransport_missing=False,
    )
    assert result["status"] == "FAIL"
    assert result["forbidden_missing"] == [key]


def test_checkpoint_state_selection_matches_configured_ema_policy() -> None:
    from tools.bata.check_chronotransport_checkpoint import select_checkpoint_state

    with pytest.raises(KeyError, match="state_dict_ema"):
        select_checkpoint_state({"state_dict": {}}, use_ema=True)
    state, state_key = select_checkpoint_state(
        {"state_dict": {"weight": 1}}, use_ema=False
    )
    assert state == {"weight": 1}
    assert state_key == "state_dict"


def test_stage_a_launcher_checks_checkpoint_before_one_batch_smoke() -> None:
    launcher = (ROOT / "scripts/run_chronotransport_adatad_gpu1.sh").read_text(
        encoding="utf-8"
    )
    assert "tools/bata/check_chronotransport_checkpoint.py" in launcher
    assert 'TEST_ARGS+=(--not_eval --max-batches "1")' in launcher


def test_checkpoint_checker_is_directly_executable_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/bata/check_chronotransport_checkpoint.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_checkpoint_checker_builds_real_stage_a_model_before_loading(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/bata/check_chronotransport_checkpoint.py"),
            "--config",
            str(CONFIG),
            "--checkpoint",
            str(tmp_path / "missing.pth"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert "Rearrange is not in the mmengine::transform registry" not in result.stderr
