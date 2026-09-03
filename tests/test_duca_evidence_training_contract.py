from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mmengine.config import Config

from tools.bata import duca_evidence_training


ROOT = Path(__file__).resolve().parents[1]
C0_CONFIG = ROOT / "configs/adatad/thumos/duca_evidence_recovery_matched_h65_60.py"
FULL_CONFIG = ROOT / "configs/adatad/thumos/duca_evidence_recovery_full.py"


@pytest.mark.parametrize(
    ("config_path", "arm_id", "arm_name"),
    ((C0_CONFIG, "C0", "MATCHED_H65_60"), (FULL_CONFIG, "F", "FULL")),
)
def test_evidence_formal_contract_is_not_duca_p0(config_path, arm_id, arm_name):
    cfg = Config.fromfile(config_path)

    contract = duca_evidence_training.formal_training_contract(cfg)

    assert contract["formal_protocol"] == duca_evidence_training.FORMAL_PROTOCOL
    assert contract["arm_id"] == arm_id
    assert contract["arm_name"] == arm_name
    assert contract["expected_successful_optimizer_updates"] == 6000
    assert contract["selector_schedule_enabled"] is False


def test_evidence_update_contract_is_schedule_free():
    audit = duca_evidence_training.new_update_audit()
    audit.update(
        {
            "attempted_batches": 100,
            "optimizer_attempts": 101,
            "successful_optimizer_updates": 100,
            "amp_skipped_attempts": 1,
            "replayed_batches": 1,
            "scheduler_updates": 100,
            "ema_updates": 100,
            "max_amp_retries_observed": 1,
            "replay_state_restorations": 1,
        }
    )
    contract = duca_evidence_training.formal_training_contract(Config.fromfile(C0_CONFIG))

    duca_evidence_training.validate_update_state(
        contract=contract,
        epoch=0,
        train_batches_per_epoch=100,
        update_audit=audit,
        scheduler_last_epoch=100,
        selector_step=0,
        uses_ema=True,
    )

    audit["duca_schedule_updates"] = 1
    with pytest.raises(RuntimeError, match="must not advance"):
        duca_evidence_training.validate_update_state(
            contract=contract,
            epoch=0,
            train_batches_per_epoch=100,
            update_audit=audit,
            scheduler_last_epoch=100,
            selector_step=0,
            uses_ema=True,
        )


def test_evidence_cfg_override_allowlist():
    duca_evidence_training.assert_safe_cfg_options(
        {"seed": 8261, "work_dir": "/tmp/evidence"},
        entrypoint="tools/train.py",
    )
    with pytest.raises(RuntimeError, match="rejected"):
        duca_evidence_training.assert_safe_cfg_options(
            {"model": {"frame_selector": {"use_coverage": False}}},
            entrypoint="tools/train.py",
        )


def test_evidence_loader_contract_exposes_exact_epoch_prefix():
    class Dataset:
        def __len__(self):
            return 438

    class Loader:
        sampler = object()

        def __len__(self):
            return 219

    cfg = SimpleNamespace(
        solver=SimpleNamespace(
            train=SimpleNamespace(batch_size=2, num_workers=2)
        )
    )
    contract = duca_evidence_training.formal_training_contract(
        Config.fromfile(C0_CONFIG)
    )

    bound, manifest = duca_evidence_training.bind_train_loader_contract(
        contract,
        cfg=cfg,
        train_dataset=Dataset(),
        train_loader=Loader(),
        world_size=1,
    )

    assert bound["allow_train_loader_prefix"] is True
    assert bound["available_train_batches_per_epoch"] == 219
    assert manifest["available_batches_per_epoch"] == 219
    assert manifest["exposed_batches_per_epoch"] == 100


def test_evidence_loader_contract_rejects_too_few_batches():
    class TooShortLoader:
        sampler = object()

        def __len__(self):
            return 99

    cfg = SimpleNamespace(
        solver=SimpleNamespace(
            train=SimpleNamespace(batch_size=2, num_workers=2)
        )
    )
    contract = duca_evidence_training.formal_training_contract(
        Config.fromfile(C0_CONFIG)
    )

    with pytest.raises(RuntimeError, match="only 99 batches"):
        duca_evidence_training.bind_train_loader_contract(
            contract,
            cfg=cfg,
            train_dataset=range(198),
            train_loader=TooShortLoader(),
            world_size=1,
        )
