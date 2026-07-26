from __future__ import annotations

from pathlib import Path

import torch

from opentad.models.selectors.duca_online_frame_selector import (
    _contribution_leaf_with_st_route,
)

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "duca_sampling_rate_both_asformer_full_mini_visual.py"
RUNNER = ROOT / "scripts" / "run_duca_sampling_rate_mini_visual_gpu1.sh"


def test_contribution_teacher_promotes_uint8_observations_without_value_change() -> None:
    observations = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8)

    routed, teacher = _contribution_leaf_with_st_route(observations)

    assert routed.dtype == torch.float32
    assert teacher.dtype == torch.float32
    assert routed.requires_grad and teacher.requires_grad
    assert torch.equal(routed.detach(), observations.float())
    (routed.square().sum()).backward()
    assert teacher.grad is not None
    assert torch.isfinite(teacher.grad).all()


def test_mini_visual_config_performs_real_bounded_training() -> None:
    cfg = Config.fromfile(str(CONFIG))
    contract = cfg.duca_mini_visual_contract
    assert contract.purpose == "trained_small_sample_mechanism_diagnostic_not_official_map"
    assert int(contract.train_epochs) == 30
    assert int(contract.train_updates_per_epoch) == 4
    assert int(contract.expected_optimizer_updates) == 120
    assert cfg.workflow.formal_protocol == ""
    assert int(cfg.workflow.end_epoch) == 30
    assert int(cfg.workflow.max_train_iters) == 4
    assert int(cfg.workflow.checkpoint_interval) == 10
    assert int(cfg.solver.train.batch_size) == 1
    schedule = cfg.model.frame_selector.loss_weight_schedule
    assert float(schedule.policy_alpha.end) == 1.0
    assert float(schedule.detector_contribution.end) == 1.0
    assert float(schedule.asformer_adapt.end) == 1.0


def test_mini_visual_runner_exports_trained_and_inference_evidence() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "tools/train.py" in source
    assert "max_train_iters=4" not in source
    assert "export_duca_training_attribution" in source
    assert "export_duca_selection_quality" in source
    assert "plot_duca_training_attribution" in source
    assert "plot_duca_inference_selection" in source
    assert "analyze_duca_selection_quality" not in source
    assert "--max-samples 2 --min-valid-length 384" in source
    assert "--require-valid-boundary" in source
    assert "epoch in 9 19 29" in source
    assert "for batch_index in 0 1" in source
    assert "--all-fixed-samples" in source
    assert "official_map_reported\": False" in source
