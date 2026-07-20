from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config, ConfigDict

from opentad.datasets.duca_stateless import DucaStatelessThumosPaddingDataset
from opentad.datasets.thumos import ThumosPaddingDataset
from tools.bata.duca_protected_physical_training import (
    assert_safe_cfg_options,
    atomic_write_json,
    bind_train_loader_contract,
    build_runtime_bindings,
    canonical_sha256,
    derive_train_loader_contract,
    formal_training_contract,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "duca_protected_physical_e2e_fixed384_official60.py"
)


def _bare_stateless_dataset():
    dataset = DucaStatelessThumosPaddingDataset.__new__(
        DucaStatelessThumosPaddingDataset
    )
    dataset.data_list = [
        ["video_a", {}, {}],
        ["video_b", {}, {}],
    ]
    dataset.stateless_seed = 3407
    dataset._duca_epoch = 0
    return dataset


def test_stateless_seed_is_a_function_of_seed_epoch_and_sample():
    dataset = _bare_stateless_dataset()
    seed_epoch0 = dataset.sample_seed(0)
    assert seed_epoch0 == dataset.sample_seed(0)
    assert seed_epoch0 != dataset.sample_seed(1)
    dataset.set_epoch(1)
    assert seed_epoch0 != dataset.sample_seed(0)


def test_stateless_getitem_restores_worker_rng_and_replays_augmentation(
    monkeypatch,
):
    dataset = _bare_stateless_dataset()

    def fake_parent_getitem(_self, index):
        return {
            "inputs": torch.tensor(
                [random.random(), np.random.rand(), torch.rand(()).item()]
            ),
            "metas": {"video_name": _self.data_list[index][0]},
        }

    monkeypatch.setattr(ThumosPaddingDataset, "__getitem__", fake_parent_getitem)
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(())
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)

    first = dataset[0]
    second = dataset[0]
    assert torch.equal(first["inputs"], second["inputs"])
    assert first["metas"]["duca_stateless_seed"] == dataset.sample_seed(0)
    assert first["metas"]["duca_stateless_epoch"] == 0
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(()), expected_torch)


def test_formal_config_seals_validation_and_derives_loader_exposure():
    cfg = Config.fromfile(str(CONFIG))
    contract = formal_training_contract(cfg)
    assert cfg.dataset.val is None
    assert cfg.workflow.seal_eval_dataloaders_during_training is True
    assert contract["end_epoch"] == 60
    assert contract["expected_train_batches_per_epoch"] is None
    assert contract["expected_successful_optimizer_updates"] is None


def test_loader_contract_hashes_realized_dataset_sampler_and_drop_last(
    tmp_path,
):
    annotation = tmp_path / "ann.json"
    annotation.write_text('{"database": {}}\n', encoding="utf-8")
    dataset = _bare_stateless_dataset()
    dataset.ann_file = str(annotation)
    dataset.subset_name = "training"

    class _Sampler:
        pass

    class _Loader:
        sampler = _Sampler()

        def __len__(self):
            return 1

    cfg = Config(
        dict(
            solver=dict(train=dict(batch_size=2, num_workers=2)),
            dataset=dict(
                train=ConfigDict(
                    type="DucaStatelessThumosPaddingDataset",
                    stateless_seed=3407,
                )
            ),
        )
    )
    manifest = derive_train_loader_contract(
        cfg=cfg,
        train_dataset=dataset,
        train_loader=_Loader(),
        world_size=1,
    )
    assert manifest["loader_length"] == 1
    assert manifest["dataset"]["dataset_length"] == 2
    assert manifest["dataset"]["stateless_seed"] == 3407
    assert manifest["drop_last"] is True
    assert len(manifest["contract_sha256"]) == 64


def test_bind_loader_contract_uses_realized_loader_length(
    tmp_path,
    monkeypatch,
):
    annotation = tmp_path / "ann.json"
    annotation.write_text('{"database": {}}\n', encoding="utf-8")
    dataset = _bare_stateless_dataset()
    dataset.ann_file = str(annotation)
    dataset.subset_name = "training"

    class _Sampler:
        pass

    class _Loader:
        sampler = _Sampler()

        def __len__(self):
            return 3

    cfg = Config(
        dict(
            solver=dict(train=dict(batch_size=2, num_workers=2)),
            dataset=dict(
                train=ConfigDict(
                    type="DucaStatelessThumosPaddingDataset",
                    stateless_seed=3407,
                )
            ),
        )
    )
    loader = _Loader()
    expected = derive_train_loader_contract(
        cfg=cfg,
        train_dataset=dataset,
        train_loader=loader,
        world_size=1,
    )
    protocol = tmp_path / "protocol.json"
    atomic_write_json(
        protocol,
        {
            "schema": "duca_protected_physical_protocol_manifest_v1",
            "train_loader_contract": expected,
        },
    )
    monkeypatch.setenv(
        "DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON",
        str(protocol),
    )
    monkeypatch.setenv(
        "DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256",
        sha256_file(protocol),
    )
    bound, observed = bind_train_loader_contract(
        {},
        cfg=cfg,
        train_dataset=dataset,
        train_loader=loader,
        world_size=1,
    )
    assert observed == expected
    assert bound["expected_train_batches_per_epoch"] == 3
    assert bound["expected_successful_optimizer_updates"] == 180


def test_protected_cfg_override_allowlist_is_fail_closed():
    assert_safe_cfg_options(
        {
            "work_dir": "/external/run",
            "model": {
                "backbone": {
                    "custom": {"pretrain": "/external/pretrain.pth"}
                }
            },
        },
        entrypoint="tools/train.py",
    )
    assert_safe_cfg_options(
        {
            "work_dir": "/external/eval",
            "post_processing": {"save_dict": True},
            "inference": {"load_from_raw_predictions": False},
        },
        entrypoint="tools/test.py",
    )
    with pytest.raises(RuntimeError, match="workflow.end_epoch"):
        assert_safe_cfg_options(
            {"workflow": {"end_epoch": 1}},
            entrypoint="tools/train.py",
        )
    with pytest.raises(RuntimeError, match="post_processing.save_dict"):
        assert_safe_cfg_options(
            {"post_processing": {"save_dict": False}},
            entrypoint="tools/test.py",
        )


def _runtime_binding_files(tmp_path):
    config = tmp_path / "config.py"
    pretrain = tmp_path / "pretrain.pth"
    annotation = tmp_path / "ann.json"
    class_map = tmp_path / "classes.txt"
    config.write_text("model = dict(type='ActionFormer')\n", encoding="utf-8")
    pretrain.write_bytes(b"videomae")
    annotation.write_text('{"database": {}}\n', encoding="utf-8")
    class_map.write_text("0 action\n", encoding="utf-8")
    resolved_sha = canonical_sha256({"resolved": True})
    protocol = {
        "schema": "duca_protected_physical_protocol_manifest_v1",
        "ok": True,
        "git_commit": "a" * 40,
        "configs": {
            "arms": {
                "protected_e2e": {
                    "source_sha256": sha256_file(config),
                    "resolved_sha256": resolved_sha,
                }
            }
        },
        "videomae_pretrain": {"sha256": sha256_file(pretrain)},
        "data_files": {
            "annotation_sha256": sha256_file(annotation),
            "class_map_sha256": sha256_file(class_map),
        },
    }
    protocol_path = tmp_path / "protocol.json"
    atomic_write_json(protocol_path, protocol)
    protocol_sha = sha256_file(protocol_path)
    authorization = {
        "schema": "duca_protected_physical_authorization_v1",
        "ok": True,
        "git_commit": "a" * 40,
        "protocol_manifest_sha256": protocol_sha,
        "authorized_scope": {
            "official60_four_arm_training": True,
            "paper_claim": False,
        },
        "paper_claim_allowed": False,
    }
    authorization_path = tmp_path / "authorization.json"
    atomic_write_json(authorization_path, authorization)
    return {
        "config": config,
        "pretrain": pretrain,
        "annotation": annotation,
        "class_map": class_map,
        "resolved_sha": resolved_sha,
        "protocol": protocol_path,
        "protocol_sha": protocol_sha,
        "authorization": authorization_path,
        "authorization_sha": sha256_file(authorization_path),
    }


def test_runtime_bindings_rehash_all_p0_inputs(tmp_path, monkeypatch):
    paths = _runtime_binding_files(tmp_path)
    monkeypatch.setenv(
        "DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON",
        str(paths["protocol"]),
    )
    monkeypatch.setenv(
        "DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256",
        paths["protocol_sha"],
    )
    monkeypatch.setenv(
        "DUCA_PROTECTED_AUTHORIZATION_JSON",
        str(paths["authorization"]),
    )
    monkeypatch.setenv(
        "DUCA_PROTECTED_AUTHORIZATION_SHA256",
        paths["authorization_sha"],
    )
    monkeypatch.setenv("DUCA_RESOLVED_CONFIG_SHA256", paths["resolved_sha"])
    bindings = build_runtime_bindings(
        git_commit="a" * 40,
        variant="protected_e2e",
        seed=3407,
        slurm_job_id="123",
        source_config_path=paths["config"],
        source_config_sha256=sha256_file(paths["config"]),
        resolved_config_sha256=paths["resolved_sha"],
        runtime_config_sha256="b" * 64,
        evaluation_annotation_path=paths["annotation"],
        evaluation_class_map_path=paths["class_map"],
        evaluation_config={"tiou_thresholds": [0.5]},
        runtime_pretrain_path=paths["pretrain"],
    )
    assert bindings["pretrain_sha256"] == sha256_file(paths["pretrain"])
    paths["annotation"].write_text('{"database": {"drift": {}}}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="annotation differs from P0"):
        build_runtime_bindings(
            git_commit="a" * 40,
            variant="protected_e2e",
            seed=3407,
            slurm_job_id="123",
            source_config_path=paths["config"],
            source_config_sha256=sha256_file(paths["config"]),
            resolved_config_sha256=paths["resolved_sha"],
            runtime_config_sha256="b" * 64,
            evaluation_annotation_path=paths["annotation"],
            evaluation_class_map_path=paths["class_map"],
            evaluation_config={"tiou_thresholds": [0.5]},
            runtime_pretrain_path=paths["pretrain"],
        )
