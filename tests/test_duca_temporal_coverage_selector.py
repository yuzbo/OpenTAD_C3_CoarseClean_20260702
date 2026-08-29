import inspect
from pathlib import Path

import torch
from mmengine.config import Config

from opentad.models.duca.acquisition import (
    DucaAcquisitionAdapter,
    TemporalCoverageSelector,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "adatad" / "thumos"


def _timestamps(batch: int, temporal_len: int) -> torch.Tensor:
    return torch.arange(temporal_len, dtype=torch.float32).repeat(batch, 1)


def test_temporal_coverage_selector_exact_sorted_valid_and_deterministic():
    selector = TemporalCoverageSelector(target_k=8, anchor_count=8)
    scores = torch.stack(
        (
            torch.linspace(-2.0, 2.0, 20),
            torch.cos(torch.linspace(0.0, 3.0, 20)),
        )
    )
    valid = torch.ones(2, 20, dtype=torch.bool)
    valid[1, 18:] = False
    rng_before = torch.random.get_rng_state().clone()

    first = selector(scores, _timestamps(2, 20), valid)
    second = selector(scores, _timestamps(2, 20), valid)

    assert first.shape == (2, 8)
    assert torch.equal(first, second)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert torch.all(first[:, 1:] > first[:, :-1])
    assert torch.all(torch.gather(valid, 1, first))
    assert not first.requires_grad


def test_temporal_coverage_selector_constant_scores_cover_time_and_remain_finite():
    selector = TemporalCoverageSelector(target_k=8, anchor_count=8)
    scores = torch.full((1, 32), 7.0)
    selected = selector(scores, _timestamps(1, 32), torch.ones_like(scores, dtype=torch.bool))
    gaps = selected[:, 1:] - selected[:, :-1]

    assert torch.isfinite(selected.float()).all()
    assert selected[0, 0] <= 2
    assert selected[0, -1] >= 29
    assert gaps.max() <= 6


def test_temporal_coverage_selector_preserves_distant_priority_peaks():
    selector = TemporalCoverageSelector(target_k=8, anchor_count=8)
    scores = torch.zeros(1, 40)
    scores[0, 3] = 20.0
    scores[0, 35] = 19.0
    selected = selector(scores, _timestamps(1, 40), torch.ones_like(scores, dtype=torch.bool))

    assert bool((selected == 3).any())
    assert bool((selected == 35).any())


def test_temporal_coverage_selector_t_equals_k_returns_every_valid_position():
    selector = TemporalCoverageSelector(target_k=16, anchor_count=8)
    scores = torch.randn(3, 16)
    selected = selector(scores, _timestamps(3, 16), torch.ones(3, 16, dtype=torch.bool))

    expected = torch.arange(16).repeat(3, 1)
    assert torch.equal(selected, expected)


def test_temporal_coverage_selector_pads_short_rows_without_duplicates():
    selector = TemporalCoverageSelector(target_k=8, anchor_count=8)
    scores = torch.randn(2, 12)
    valid = torch.zeros(2, 12, dtype=torch.bool)
    valid[0, :12] = True
    valid[1, :5] = True

    selected = selector(scores, _timestamps(2, 12), valid)

    assert torch.all(selected[0, 1:] > selected[0, :-1])
    assert torch.equal(selected[1, :5], torch.arange(5))
    assert torch.equal(selected[1, 5:], torch.full((3,), -1))


def test_temporal_coverage_forward_has_no_host_sync_or_batch_loop():
    source = inspect.getsource(TemporalCoverageSelector.forward)
    for forbidden in (".item(", ".cpu(", ".numpy(", ".tolist(", "for batch", "for b in"):
        assert forbidden not in source
    assert "for slot in range(self.target_k)" in source


def test_temporal_coverage_policy_runs_through_duca_acquisition_adapter():
    adapter = DucaAcquisitionAdapter(
        feature_dim=3,
        hidden_dim=8,
        budget=8,
        acquisition_policy="temporal_coverage",
        hard_max_gap_repair=False,
        fail_on_infeasible_max_gap=False,
    )
    observations = torch.randn(2, 20, 3)
    valid = torch.ones(2, 20, dtype=torch.bool)
    valid[1, 18:] = False

    grid, outputs = adapter.acquire(observations, valid_mask=valid)

    assert grid.selected_positions.shape == (2, 8)
    assert torch.all(grid.selected_positions[:, 1:] > grid.selected_positions[:, :-1])
    assert torch.all(torch.gather(valid, 1, grid.selected_positions))
    assert outputs["structured_soft_slot_assignment"].shape == (2, 8, 20)
    assert outputs["selection_path"] == "h65_priority_temporal_facility_location"


def test_temporal_coverage_adapter_reports_effective_budget_for_short_rows():
    adapter = DucaAcquisitionAdapter(
        feature_dim=3,
        hidden_dim=8,
        budget=8,
        acquisition_policy="temporal_coverage",
        hard_max_gap_repair=False,
        fail_on_infeasible_max_gap=False,
    )
    observations = torch.randn(2, 12, 3)
    valid = torch.zeros(2, 12, dtype=torch.bool)
    valid[0, :12] = True
    valid[1, :5] = True

    grid, outputs = adapter.acquire(observations, valid_mask=valid)

    assert torch.equal(grid.effective_budget, torch.tensor([8, 5]))
    assert torch.equal(grid.detector_input_length, torch.tensor([8, 5]))
    assert torch.equal(grid.selected_positions[1, :5], torch.arange(5))
    assert torch.equal(grid.selected_positions[1, 5:], torch.full((3,), -1))
    assert int(grid.selected_mask[1].sum()) == 5
    assert torch.equal(outputs["soft_coverage"].bool(), grid.selected_mask)


def test_coverage_and_control_configs_change_only_allocation_policy(monkeypatch):
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT", "/tmp/stage1_epoch29.pth")
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_SHA256", "b" * 64)
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_EPOCH", "29")
    control = Config.fromfile(str(CONFIG_ROOT / "duca_coverage_v1_matched_h65_control.py"))
    coverage = Config.fromfile(str(CONFIG_ROOT / "duca_coverage_v1_candidate.py"))

    control_selector = control.model.frame_selector.to_dict()
    coverage_selector = coverage.model.frame_selector.to_dict()
    assert control_selector.pop("acquisition_policy") == "budget_calibrated_sampling_rate"
    assert coverage_selector.pop("acquisition_policy") == "temporal_coverage"
    assert control_selector == coverage_selector
    assert control.model.frame_selector.freeze_priority_path is True
    assert control.model.frame_selector.actionness_source_cfg.frozen is True
    assert control.model.frame_selector.actionness_source_cfg.trainable is False
    assert control.seed == coverage.seed == 3407
    assert control.max_updates == coverage.max_updates == 6000
    assert control.workflow.formal_successful_update_contract is True
    assert coverage.workflow.formal_successful_update_contract is True
    assert control.workflow.checkpoint_interval == coverage.workflow.checkpoint_interval == 5
    assert control.workflow.primary_checkpoint_state_key == "state_dict_ema"
    assert coverage.workflow.primary_checkpoint_state_key == "state_dict_ema"
