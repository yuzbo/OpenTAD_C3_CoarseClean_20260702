from __future__ import annotations

from array import array
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


def _coverage_dataset(*, ledger, exposure_ids, allow_missing=False):
    class DucaH65PositionsFromLedger:
        target_len = 4
        dense_len = 8
        use_expanded_positions = False

        def __init__(self):
            self.allow_missing = allow_missing

        def _value_transport_ledger(self):
            return ledger

        def _required_count(self, _valid_len):
            return 4

    transform = DucaH65PositionsFromLedger()
    data_list = [
        (video, None, None, [start])
        for video, start in exposure_ids
    ]
    return SimpleNamespace(
        pipeline=SimpleNamespace(transforms=[transform]),
        data_list=data_list,
    )


def _coverage_row(sample_id):
    return {
        "sample_id": sample_id,
        "valid_len": 8,
        "dense_len": 8,
        "target_len": 4,
        "selected_positions": array("q", [0, 2, 4, 6]),
    }


def test_evidence_ledger_coverage_uses_unique_physical_windows():
    sample_id = "video|0"
    dataset = _coverage_dataset(
        ledger={sample_id: _coverage_row(sample_id)},
        exposure_ids=[("video", 0), ("video", 0)],
    )

    result = duca_evidence_training.validate_ledger_coverage(dataset, "test")

    assert result == {
        "split": "test",
        "loader_exposures": 2,
        "unique_physical_windows": 1,
        "ledger_rows": 1,
    }


def test_evidence_ledger_coverage_rejects_missing_and_permissive_lookup():
    missing = _coverage_dataset(ledger={}, exposure_ids=[("video", 0)])
    with pytest.raises(RuntimeError, match="missing 1 windows"):
        duca_evidence_training.validate_ledger_coverage(missing, "test")

    permissive = _coverage_dataset(
        ledger={"video|0": _coverage_row("video|0")},
        exposure_ids=[("video", 0)],
        allow_missing=True,
    )
    with pytest.raises(RuntimeError, match="must fail on missing rows"):
        duca_evidence_training.validate_ledger_coverage(permissive, "test")


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


def _terminal_binding_case(tmp_path: Path, monkeypatch, *, changed: bool):
    training_config = tmp_path / "training" / C0_CONFIG.name
    evaluation_config = tmp_path / "evaluation" / C0_CONFIG.name
    pretrain = tmp_path / "pretrain.pth"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    checkpoint = tmp_path / "checkpoint.pth"
    training_config.parent.mkdir(parents=True)
    evaluation_config.parent.mkdir(parents=True)
    training_config.write_text("# config\n", encoding="utf-8")
    evaluation_config.write_text(
        "# changed config\n" if changed else "# config\n", encoding="utf-8"
    )
    pretrain.write_bytes(b"pretrain")
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("action\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    source_sha = duca_evidence_training.sha256_file(training_config)
    audit = {
        "formal_protocol": duca_evidence_training.FORMAL_PROTOCOL,
        "git_commit": "a" * 40,
        "arm_id": "C0",
        "arm_name": "MATCHED_H65_60",
        "seed": 8261,
        "source_config_path": str(training_config.resolve()),
        "source_config_sha256": source_sha,
        "resolved_config_sha256": "b" * 64,
        "runtime_pretrain_path": str(pretrain.resolve()),
        "runtime_pretrain_sha256": duca_evidence_training.sha256_file(pretrain),
        "evaluation_annotation_path": str(annotation.resolve()),
        "evaluation_annotation_sha256": duca_evidence_training.sha256_file(annotation),
        "evaluation_class_map_path": str(class_map.resolve()),
        "evaluation_class_map_sha256": duca_evidence_training.sha256_file(class_map),
        "update_audit": {
            "successful_optimizer_updates": 6000,
            "duca_schedule_updates": 0,
        },
        "status": "complete",
        "last_completed_epoch": 59,
        "expected_successful_optimizer_updates": 6000,
        "audit_sha256": "c" * 64,
        "slurm_job_id": "7",
    }
    monkeypatch.setattr(
        duca_evidence_training, "_validated_checkpoint_audit", lambda _: audit
    )
    return evaluation_config, pretrain, annotation, class_map, checkpoint


def test_terminal_binding_allows_verified_config_in_another_checkout(
    tmp_path: Path, monkeypatch
):
    config, pretrain, annotation, class_map, checkpoint = _terminal_binding_case(
        tmp_path, monkeypatch, changed=False
    )

    identity = duca_evidence_training.validate_terminal_checkpoint_binding(
        checkpoint_path=checkpoint,
        checkpoint={"successful_optimizer_updates": 6000},
        git_commit="a" * 40,
        arm_id="C0",
        arm_name="MATCHED_H65_60",
        seed=8261,
        source_config_path=config,
        source_config_sha256=duca_evidence_training.sha256_file(config),
        resolved_config_sha256="b" * 64,
        checkpoint_epoch=59,
        checkpoint_state_key="state_dict_ema",
        evaluation_annotation_path=annotation,
        evaluation_class_map_path=class_map,
        runtime_pretrain_path=pretrain,
    )

    assert identity["training_slurm_job_id"] == "7"


def test_terminal_binding_rejects_changed_config_in_another_checkout(
    tmp_path: Path, monkeypatch
):
    config, pretrain, annotation, class_map, checkpoint = _terminal_binding_case(
        tmp_path, monkeypatch, changed=True
    )

    with pytest.raises(RuntimeError, match="source_config_sha256"):
        duca_evidence_training.validate_terminal_checkpoint_binding(
            checkpoint_path=checkpoint,
            checkpoint={"successful_optimizer_updates": 6000},
            git_commit="a" * 40,
            arm_id="C0",
            arm_name="MATCHED_H65_60",
            seed=8261,
            source_config_path=config,
            source_config_sha256=duca_evidence_training.sha256_file(config),
            resolved_config_sha256="b" * 64,
            checkpoint_epoch=59,
            checkpoint_state_key="state_dict_ema",
            evaluation_annotation_path=annotation,
            evaluation_class_map_path=class_map,
            runtime_pretrain_path=pretrain,
        )
