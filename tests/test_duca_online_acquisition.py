from __future__ import annotations

import inspect

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.duca import (  # noqa: E402
    DucaAcquisitionAdapter,
    SparseTemporalGrid,
    ZeroShotActionnessSource,
    budgeted_center_radius_decode,
    duca_forward_test,
    duca_forward_train,
    duca_losses,
    gather_selected_observations,
    hard_topk_st,
    make_audit_record,
)


class RecordingSparseDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_sparse(self, observations, sparse_grid=None, **kwargs):
        self.calls.append({"observations": observations, "sparse_grid": sparse_grid, "kwargs": kwargs})
        return {"loss": observations.float().mean(), "num_observations": observations.shape[1]}


def _manual_source(p_action: torch.Tensor) -> ZeroShotActionnessSource:
    uncertainty = 1.0 - torch.abs(2.0 * p_action - 1.0)
    return ZeroShotActionnessSource.from_manual(p_action=p_action, uncertainty=uncertainty)


def test_zero_shot_actionness_fallback_has_no_thumos_label_provenance() -> None:
    features = torch.randn(2, 12, 4)
    source = ZeroShotActionnessSource(mode="motion")

    output = source(features)

    assert output["p_action"].shape == (2, 12)
    assert output["uncertainty"].shape == (2, 12)
    assert output["features"].shape == (2, 12, 3)
    assert output["provenance"]["thumos_trained"] is False
    assert output["provenance"]["uses_labels"] is False
    assert output["provenance"]["uses_teacher"] is False
    assert "labels" not in output
    assert "gt_segments" not in output


def test_sparse_temporal_grid_validate_fail_closed() -> None:
    selected_positions = torch.tensor([[1, 3, 5]])
    selected_mask = torch.zeros(1, 8, dtype=torch.bool)
    selected_mask[0, [1, 3, 5]] = True

    grid = SparseTemporalGrid(
        selected_positions=selected_positions,
        selected_mask=selected_mask,
        original_length=8,
        valid_len=torch.tensor([8]),
        budget=3,
        detector_input_length=torch.tensor([3]),
    )
    assert grid.validate() is grid
    assert make_audit_record(grid, uses_teacher=False, mode="test")["uses_ledger_for_decision"] is False

    with pytest.raises(ValueError, match="budget"):
        SparseTemporalGrid(
            selected_positions=selected_positions,
            selected_mask=selected_mask,
            original_length=8,
            valid_len=torch.tensor([8]),
            budget=2,
        ).validate()

    with pytest.raises(ValueError, match="coordinate"):
        SparseTemporalGrid(
            selected_positions=selected_positions,
            selected_mask=selected_mask,
            original_length=8,
            valid_len=torch.tensor([8]),
            budget=3,
            coordinate="selected_rank",
        ).validate()

    bad_mask = selected_mask.clone()
    bad_mask[0, 2] = True
    with pytest.raises(ValueError, match="selected_positions count"):
        SparseTemporalGrid(
            selected_positions=selected_positions,
            selected_mask=bad_mask,
            original_length=8,
            valid_len=torch.tensor([8]),
            budget=4,
        ).validate()


def test_hard_topk_st_returns_hard_forward_and_surrogate_gradient() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.4, 0.8, 0.2]], requires_grad=True)
    valid = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)

    mask, indices, aux = hard_topk_st(scores, k=3, valid_mask=valid, return_aux=True)

    assert mask.detach().tolist() == [[0.0, 1.0, 1.0, 1.0, 0.0]]
    assert indices.detach().tolist() == [[1, 2, 3]]
    assert aux["hard_mask"].sum().item() == 3
    assert aux["hard_mask"][0, 4].item() == 0
    (mask * scores).sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0


def test_budgeted_center_radius_decode_outputs_consumed_original_positions() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.4, 0.8, 0.2, 0.7, 0.6, 0.3]])
    radius = torch.tensor([[0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0]])

    decoded = budgeted_center_radius_decode(center_scores=scores, radius=radius, budget=4, max_radius=16)

    positions = decoded["selected_positions"]
    assert positions.shape == (1, 4)
    assert positions.tolist()[0] == sorted(set(positions.tolist()[0]))
    assert positions.min().item() >= 0
    assert positions.max().item() < scores.shape[1]
    assert decoded["selected_mask"].sum().item() == 4
    assert decoded["selected_centers"].shape[1] <= 4
    assert decoded["selected_radius"].shape[1] <= 4


def test_budgeted_decode_short_window_selects_no_more_than_valid_len() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.7, -99.0, -99.0]])
    valid = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

    decoded = budgeted_center_radius_decode(center_scores=scores, radius=torch.zeros_like(scores), budget=4, valid_mask=valid)

    assert decoded["selected_positions"].shape == (1, 3)
    assert decoded["selected_mask"].sum().item() == 3
    assert decoded["selected_positions"].tolist()[0] == [0, 1, 2]


def test_gather_selected_observations_uses_only_original_time_positions() -> None:
    dense = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    selected_positions = torch.tensor([[0, 2, 5], [1, 3, 4]])
    selected_mask = torch.zeros(2, 6, dtype=torch.bool)
    selected_mask[0, [0, 2, 5]] = True
    selected_mask[1, [1, 3, 4]] = True

    sparse = gather_selected_observations(dense, selected_positions, selected_mask)

    assert sparse["observations"].shape == (2, 3, 3)
    assert sparse["observations"][0].tolist() == dense[0, [0, 2, 5]].tolist()
    assert sparse["observations"][1].tolist() == dense[1, [1, 3, 4]].tolist()

    channel_first = dense.movedim(1, 2)
    sparse_cf = gather_selected_observations(channel_first, selected_positions, selected_mask, time_dim=2)
    assert sparse_cf["observations"].shape == (2, 3, 3)
    assert sparse_cf["observations"].movedim(2, 1)[0].tolist() == dense[0, [0, 2, 5]].tolist()


def test_adapter_acquire_768_to_384_returns_valid_original_time_grid() -> None:
    dense = torch.randn(1, 768, 4)
    p_action = torch.linspace(0.0, 1.0, 768).view(1, 768)
    adapter = DucaAcquisitionAdapter(actionness_source=_manual_source(p_action), budget=384, max_radius=16)

    grid, scores = adapter.acquire(dense)

    assert isinstance(grid, SparseTemporalGrid)
    assert grid.selected_positions.shape == (1, 384)
    assert grid.selected_count.tolist() == [384]
    assert grid.coordinate == "original_time"
    assert grid.budget_unit == "detector_temporal_observation"
    assert grid.detector_consumes_selected_positions is True
    assert scores["radius"].max().item() <= 16.0
    grid.validate()


def test_train_and_test_forward_are_hard_sparse_and_teacher_free_at_inference() -> None:
    dense = torch.randn(1, 12, 3)
    p_action = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.05, 0.55, 0.45, 0.35]])
    adapter = DucaAcquisitionAdapter(feature_dim=3, actionness_source=_manual_source(p_action), budget=4, max_radius=2)
    detector = RecordingSparseDetector()
    teacher_utility = torch.rand(1, 12)
    batch = {
        "observations": dense,
        "teacher_utility": teacher_utility,
        "boundary_target": torch.zeros(1, 12),
        "action_target": torch.ones(1, 12),
    }

    train_output = duca_forward_train(detector=detector, adapter=adapter, batch=batch)

    assert len(detector.calls) == 1
    assert detector.calls[0]["observations"].shape[1] == 4
    assert detector.calls[0]["sparse_grid"].selected_count.tolist() == [4]
    assert "teacher_utility" not in detector.calls[0]["kwargs"]
    assert train_output["losses"]["teacher_utility_loss"].requires_grad
    assert train_output["audit"]["uses_teacher"] is False

    with pytest.raises(ValueError, match="forbids"):
        duca_forward_test(detector=detector, adapter=adapter, batch=batch)

    test_output = duca_forward_test(detector=detector, adapter=adapter, batch={"observations": dense})
    assert test_output["detector_input"].shape[1] == 4
    assert test_output["audit"]["uses_teacher"] is False
    assert test_output["audit"]["uses_ledger_for_decision"] is False


def test_duca_losses_expose_required_components() -> None:
    dense = torch.randn(1, 10, 3)
    adapter = DucaAcquisitionAdapter(feature_dim=3, budget=4, max_radius=4)
    output = adapter.forward_acquire(dense)
    losses = duca_losses(
        output,
        teacher_utility=torch.rand(1, 10),
        boundary_target=torch.zeros(1, 10),
        action_target=torch.ones(1, 10),
        detector_loss=torch.tensor(0.5, requires_grad=True),
    )

    expected = {
        "detector_loss",
        "budget_loss",
        "boundary_coverage_loss",
        "action_local_hole_loss",
        "redundancy_loss",
        "radius_cost_loss",
        "entropy_anti_collapse_loss",
        "teacher_utility_loss",
        "total_loss",
    }
    assert expected <= set(losses)
    assert losses["total_loss"].requires_grad


def test_online_interfaces_do_not_accept_ledger_or_teacher_in_test_signature() -> None:
    for func in (duca_forward_train, duca_forward_test):
        assert "ledger" not in inspect.signature(func).parameters
    assert "teacher_utility" not in inspect.signature(duca_forward_test).parameters
    assert "ledger" not in inspect.signature(DucaAcquisitionAdapter.forward_acquire).parameters
    assert hasattr(DucaAcquisitionAdapter, "acquire")
