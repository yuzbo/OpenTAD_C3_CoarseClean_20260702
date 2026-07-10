from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from opentad.models.chronotransport.replay import paired_detector_losses
from tools.bata.chronotransport_opentad_factory import (
    move_batch_to_device,
    prepare_replay_batch,
)


ROOT = Path(__file__).resolve().parents[1]


class ChronoTransportRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forced_schedule = None


class _GradDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chronotransport = ChronoTransportRuntime()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, inputs, **kwargs):
        del kwargs
        offset = 0.0 if self.chronotransport.forced_schedule == "dense" else 0.25
        return {"loss_cls": ((inputs * self.weight) + offset).square().mean()}


def test_paired_replay_can_disable_counterfactual_autograd_for_ledger_generation() -> None:
    model = _GradDetector()
    result = paired_detector_losses(
        model,
        {"inputs": torch.ones(2, 2)},
        counterfactual_schedule="periodic2_transport",
        track_counterfactual_grad=False,
    )
    assert result.counterfactual_total.requires_grad is False
    assert model.weight.grad is None


def test_prepare_replay_batch_uses_deploy_safe_diagnostic_identity() -> None:
    source = {
        "inputs": torch.ones(1, 3),
        "masks": torch.ones(1, 3, dtype=torch.bool),
        "metas": [{"video_name": "video_validation_0000051"}],
    }
    batch = prepare_replay_batch(source, batch_index=7, split="diagnostic")
    assert batch["sample_id"] == "video_validation_0000051:000007"
    assert batch["split"] == "diagnostic"
    assert batch["return_loss"] is True
    assert "raw_predictions" not in batch


def test_prepare_replay_batch_rejects_validation_or_test_ledger_generation() -> None:
    source = {"metas": [{"video_name": "x"}]}
    with pytest.raises(ValueError, match="train or diagnostic"):
        prepare_replay_batch(source, batch_index=0, split="validation")


def test_factory_moves_nested_batch_tensors_to_model_device() -> None:
    source = {
        "inputs": torch.ones(1),
        "gt_segments": [torch.ones(2, 2)],
        "metas": [{"video_name": "x", "duration": 1.0}],
    }
    moved = move_batch_to_device(source, torch.device("cpu"))
    assert moved["inputs"].device.type == "cpu"
    assert moved["gt_segments"][0].device.type == "cpu"
    assert moved["metas"][0]["video_name"] == "x"


def test_real_paired_replay_launcher_is_gpu1_and_allocation_guarded() -> None:
    launcher = ROOT / "scripts/run_chronotransport_paired_replay_gpu1.sh"
    text = launcher.read_text(encoding="utf-8")
    assert '[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]]' in text
    assert "SLURM_JOB_ID" in text
    assert "check_chronotransport_checkpoint.py" in text
    assert "tools.bata.chronotransport_opentad_factory:paired_replay_factory" in text
    assert "raw_predictions" not in text
