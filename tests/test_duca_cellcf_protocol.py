from __future__ import annotations

from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.duca_cellcf_protocol import (
    protocol_for_name,
    protocol_from_workflow,
)
from tools.bata.duca_cellcf_training import formal_training_contract
from tools.bata.finalize_duca_cellcf_run import _resolve_artifact_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/adatad/thumos/"
    "duca_cellcf_fixed384_official_adatad_backend_full_train.py"
)


def test_protocol_definitions_separate_exposure_and_official_training() -> None:
    exposure = protocol_for_name("exposure132")
    official = protocol_for_name("official60")

    assert exposure.end_epoch == 132
    assert exposure.expected_successful_optimizer_updates == 13200
    assert exposure.terminal_epoch == 131
    assert exposure.purpose.startswith("sufficient_exposure")
    assert official.end_epoch == 60
    assert official.expected_successful_optimizer_updates == 6000
    assert official.terminal_epoch == 59
    assert official.checkpoint_criterion == "terminal_epoch_59_state_dict_ema"


def test_default_config_remains_frozen_exposure132(monkeypatch) -> None:
    monkeypatch.delenv("DUCA_CELLCF_TRAINING_PROFILE", raising=False)
    monkeypatch.delenv("DUCA_OFFICIAL_ADATAD_END_EPOCH", raising=False)
    monkeypatch.delenv("DUCA_LOSS_SCHEDULE_TOTAL_STEPS", raising=False)
    cfg = Config.fromfile(str(CONFIG))

    protocol = protocol_from_workflow(cfg.workflow)
    contract = formal_training_contract(cfg)

    assert protocol.name == "exposure132"
    assert contract["training_profile"] == "exposure132"
    assert contract["end_epoch"] == 132
    assert cfg.scheduler.max_epoch == 132
    assert "duca_training_protocol" not in cfg


def test_official60_config_has_native_scheduler_and_terminal_contract(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    monkeypatch.setenv("DUCA_OFFICIAL_ADATAD_END_EPOCH", "60")
    monkeypatch.setenv("DUCA_LOSS_SCHEDULE_STEPS_PER_EPOCH", "100")
    monkeypatch.setenv("DUCA_LOSS_SCHEDULE_TOTAL_STEPS", "6000")
    cfg = Config.fromfile(str(CONFIG))

    protocol = protocol_from_workflow(cfg.workflow)
    contract = formal_training_contract(cfg)

    assert protocol.name == "official60"
    assert cfg.workflow.end_epoch == 60
    assert cfg.scheduler.max_epoch == 60
    assert contract["expected_successful_optimizer_updates"] == 6000
    assert contract["primary_checkpoint_epoch"] == 59
    assert contract["checkpoint_criterion"] == "terminal_epoch_59_state_dict_ema"
    assert "duca_training_protocol" not in cfg


def test_profile_rejects_environment_and_resolved_config_drift(monkeypatch) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    monkeypatch.setenv("DUCA_OFFICIAL_ADATAD_END_EPOCH", "132")

    with pytest.raises(ValueError, match="fixed to 60"):
        Config.fromfile(str(CONFIG))


def test_ddp_pilot_resolves_profile_at_validation_time(monkeypatch) -> None:
    from tools.bata import validate_duca_cellcf_ddp_pilot as pilot

    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "exposure132")
    assert pilot._formal_protocol().end_epoch == 132
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    monkeypatch.setenv("DUCA_OFFICIAL_ADATAD_END_EPOCH", "60")
    monkeypatch.setenv("DUCA_LOSS_SCHEDULE_STEPS_PER_EPOCH", "100")
    monkeypatch.setenv("DUCA_LOSS_SCHEDULE_TOTAL_STEPS", "6000")
    assert pilot._formal_protocol().end_epoch == 60
    contract = pilot._validate_config_contract(ROOT, "cellcf")
    assert contract["training_profile"] == "official60"
    assert contract["formal_end_epoch"] == 60


def test_finalizer_profile_compatibility_is_scoped_to_exact_legacy_commit() -> None:
    with pytest.raises(ValueError, match="audited legacy"):
        _resolve_artifact_protocol(
            {"git_commit": "a" * 40},
            {"git_commit": "a" * 40},
        )

    legacy_commit = "1642f265e48391418a7c8a4a087e33e2b7bf6899"
    assert (
        _resolve_artifact_protocol(
            {"git_commit": legacy_commit},
            {"git_commit": legacy_commit},
        ).name
        == "exposure132"
    )
    assert (
        _resolve_artifact_protocol(
            {"git_commit": "b" * 40, "training_profile": "official60"},
            {"git_commit": "b" * 40, "training_profile": "official60"},
        ).name
        == "official60"
    )
