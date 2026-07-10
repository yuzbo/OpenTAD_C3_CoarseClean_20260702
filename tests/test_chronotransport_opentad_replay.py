from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from opentad.models.chronotransport.replay import paired_detector_losses
from tools.bata.run_chronotransport_paired_replay import _compact_runtime_payload
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
        feature = (inputs * self.weight) + offset
        self.chronotransport.latest_output = feature
        return {"loss_cls": feature.square().mean()}


class _SdpProbeDetector(_GradDetector):
    def __init__(self) -> None:
        super().__init__()
        self.sdp_states = []

    def forward(self, inputs, **kwargs):
        self.sdp_states.append(
            (
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),
            )
        )
        return super().forward(inputs, **kwargs)


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


def test_paired_replay_uses_deterministic_math_sdp_for_both_branches() -> None:
    model = _SdpProbeDetector()
    paired_detector_losses(
        model,
        {"inputs": torch.ones(2, 2)},
        counterfactual_schedule="dense",
        track_counterfactual_grad=False,
    )
    assert model.sdp_states == [(False, False, True), (False, False, True)]


def test_paired_training_preserves_gradient_capable_sdp_backend() -> None:
    model = _SdpProbeDetector()
    expected = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    paired_detector_losses(
        model,
        {"inputs": torch.ones(2, 2)},
        counterfactual_schedule="periodic2_transport",
        track_counterfactual_grad=True,
    )
    assert model.sdp_states == [expected, expected]


def test_paired_replay_exposes_ephemeral_dense_and_counterfactual_features() -> None:
    model = _GradDetector()
    result = paired_detector_losses(
        model,
        {"inputs": torch.ones(2, 2)},
        counterfactual_schedule="periodic2_transport",
        track_counterfactual_grad=True,
    )
    assert result.dense_features.requires_grad is False
    assert result.counterfactual_features.requires_grad is True
    assert result.dense_features.shape == result.counterfactual_features.shape


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


def test_compact_replay_cost_excludes_volatile_profile_latency() -> None:
    model = _GradDetector()
    model.chronotransport.latest_summary = {
        "recompute_rows": 2,
        "transport_rows": 1,
        "hold_rows": 0,
        "profile": {"latency_ms": {"recompute": {"p50": 1.23}}},
    }
    _, cost = _compact_runtime_payload(model)
    assert cost == {"recompute_rows": 2, "transport_rows": 1, "hold_rows": 0}


def test_real_paired_replay_launcher_is_gpu1_and_allocation_guarded() -> None:
    launcher = ROOT / "scripts/run_chronotransport_paired_replay_gpu1.sh"
    text = launcher.read_text(encoding="utf-8")
    assert '[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]]' in text
    assert "SLURM_JOB_ID" in text
    assert "check_chronotransport_checkpoint.py" in text
    assert "tools.bata.chronotransport_opentad_factory:paired_replay_factory" in text
    assert "raw_predictions" not in text


def test_real_stage_b_factory_launcher_and_checkpoint_contract_exist() -> None:
    from tools.bata.chronotransport_opentad_factory import stage_b_factory

    assert callable(stage_b_factory)
    launcher = (ROOT / "scripts/run_chronotransport_stage_b_gpu1.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]]' in launcher
    assert "SLURM_JOB_ID" in launcher
    assert "CHRONOTRANSPORT_DENSE_GATE_A" in launcher
    assert "CHRONOTRANSPORT_DENSE_GATE_B" in launcher
    assert "validate_chronotransport_dense_gate.py" in launcher
    trainer = (ROOT / "tools/bata/train_chronotransport_stage_b.py").read_text(
        encoding="utf-8"
    )
    assert '"state_dict"' in trainer
    assert 'torch.save({"model"' not in trainer


def test_remote_verify_supports_noninteractive_ssh_without_module_function() -> None:
    verify = (ROOT / "scripts/verify_chronotransport_n16r4.sh").read_text(
        encoding="utf-8"
    )
    assert "if command -v module" in verify
    assert 'source "${BASE}/conda_envs/opentad/bin/activate"' in verify


def test_dense_gate_validator_requires_repeatable_near_zero_dense_regret(
    tmp_path: Path,
) -> None:
    from tools.bata.validate_chronotransport_dense_gate import validate

    record = {
        "sample_id": "v:000000",
        "split": "diagnostic",
        "schedule": "dense",
        "signals": {},
        "pooled_targets": {"dense_loss": 1.0, "counterfactual_loss": 1.0000001},
        "cost": {},
        "regret": 1e-7,
    }
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")
    result = validate(first, second, tolerance=1e-6)
    assert result["status"] == "PASS"

    changed = dict(record, regret=1e-3)
    second.write_text(
        json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="byte-identical"):
        validate(first, second, tolerance=1e-6)


def test_stage_b_trainer_is_directly_executable_from_repository_root() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/bata/train_chronotransport_stage_b.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
