from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.bata import duca_rime_training


torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    capture_output=True,
    text=True,
    check=False,
)
TORCH_AVAILABLE = torch_probe.returncode == 0


class DucaStatelessThumosPaddingDataset:
    def __init__(self):
        self.data_list = [(f"video_{index:03d}",) for index in range(100)]
        self.stateless_seed = 3407


class DistributedSampler:
    seed = 17
    shuffle = True
    drop_last = True
    num_replicas = 1
    rank = 0
    epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        offset = self.epoch % 100
        return iter(list(range(offset, 100)) + list(range(offset)))


class _Loader:
    sampler = DistributedSampler()

    def __len__(self):
        return 100


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch runtime unavailable")
def test_training_exposure_contract_is_exact_and_deterministic(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        solver=SimpleNamespace(train=SimpleNamespace(batch_size=1)),
    )
    dataset = DucaStatelessThumosPaddingDataset()
    first = duca_rime_training.derive_train_loader_contract(
        cfg=cfg,
        train_dataset=dataset,
        train_loader=_Loader(),
        world_size=1,
    )
    second = duca_rime_training.derive_train_loader_contract(
        cfg=cfg,
        train_dataset=dataset,
        train_loader=_Loader(),
        world_size=1,
    )
    assert first == second
    assert first["successful_detector_updates"] == 6000
    assert first["per_video_exposure_count"] == 60
    assert len(first["ordered_video_ids"]) == 100
    assert len(first["epoch_index_sha256"]) == 60
    assert first["sampler_seed"] == 17

    exposure = tmp_path / "exposure.json"
    exposure.write_text(
        json.dumps(
            {
                "schema_version": "duca_rime_phase3_training_exposure_v1",
                "train_loader_contract": first,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(exposure.read_bytes()).hexdigest()
    monkeypatch.setenv("DUCA_RIME_TRAINING_EXPOSURE_JSON", str(exposure))
    monkeypatch.setenv("DUCA_RIME_TRAINING_EXPOSURE_SHA256", digest)
    bound, actual = duca_rime_training.bind_train_loader_contract(
        {"expected_successful_optimizer_updates": 6000},
        cfg=cfg,
        train_dataset=dataset,
        train_loader=_Loader(),
        world_size=1,
    )
    assert actual == first
    assert bound["train_loader_contract"]["contract_sha256"] == first["contract_sha256"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch runtime unavailable")
def test_training_exposure_rejects_an_unsealed_sampler():
    cfg = SimpleNamespace(
        solver=SimpleNamespace(train=SimpleNamespace(batch_size=1)),
    )
    dataset = DucaStatelessThumosPaddingDataset()

    class _UnsafeLoader:
        sampler = SimpleNamespace(shuffle=True)

        def __len__(self):
            return 100

    with pytest.raises(RuntimeError, match="DistributedSampler"):
        duca_rime_training.derive_train_loader_contract(
            cfg=cfg,
            train_dataset=dataset,
            train_loader=_UnsafeLoader(),
            world_size=1,
        )
