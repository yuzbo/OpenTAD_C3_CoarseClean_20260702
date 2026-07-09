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
    temporal_max_gap_hole_loss,
)


class RecordingSparseDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_sparse(self, observations, sparse_grid=None, **kwargs):
        self.calls.append({"observations": observations, "sparse_grid": sparse_grid, "kwargs": kwargs})
        return {"loss": observations.float().mean(), "num_observations": observations.shape[1]}


class DetectorLossOnly(torch.nn.Module):
    def forward_sparse(self, observations, sparse_grid=None, **kwargs):
        return {"loss": observations.pow(2).mean()}


class ForbiddenPayloadDetector(torch.nn.Module):
    forbidden = {"teacher_utility", "teacher_points", "dense_teacher", "dense_teacher_payload", "prediction_cache"}

    def forward_sparse(self, observations, sparse_grid=None, batch=None, **kwargs):
        hits = []

        def walk(obj, path="batch"):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in self.forbidden or key.startswith("dense_teacher"):
                        hits.append(f"{path}.{key}")
                    walk(value, f"{path}.{key}")
            elif isinstance(obj, (list, tuple)):
                for idx, value in enumerate(obj):
                    walk(value, f"{path}[{idx}]")

        walk(batch)
        if hits:
            raise AssertionError(f"forbidden detector payload leaked: {hits}")
        return {"loss": observations.float().mean()}


def _manual_source(p_action: torch.Tensor) -> ZeroShotActionnessSource:
    uncertainty = 1.0 - torch.abs(2.0 * p_action - 1.0)
    return ZeroShotActionnessSource.from_manual(p_action=p_action, uncertainty=uncertainty)


def test_zero_shot_actionness_fallback_has_no_thumos_label_provenance() -> None:
    features = torch.randn(2, 12, 4)
    source = ZeroShotActionnessSource(mode="motion")

    output = source(features)

    assert output["p_action"].shape == (2, 12)
    assert output["uncertainty"].shape == (2, 12)
    assert output["delta_p_action"].shape == (2, 12)
    assert output["abs_delta_p_action"].shape == (2, 12)
    assert output["uncertainty_peak"].shape == (2, 12)
    assert output["transition_score"].shape == (2, 12)
    assert output["features"].shape == (2, 12, 7)
    assert output["provenance"]["thumos_trained"] is False
    assert output["provenance"]["uses_labels"] is False
    assert output["provenance"]["uses_teacher"] is False
    assert "labels" not in output
    assert "gt_segments" not in output


def test_actionness_source_exposes_state_transition_features() -> None:
    p_action = torch.tensor([[0.05, 0.05, 0.95, 0.95, 0.05, 0.05]], dtype=torch.float32)
    source = _manual_source(p_action)

    output = source(torch.zeros(1, 6, 1))

    assert output["delta_p_action"].tolist()[0] == pytest.approx([0.0, 0.0, 0.90, 0.0, -0.90, 0.0])
    assert output["abs_delta_p_action"].tolist()[0] == pytest.approx([0.0, 0.0, 0.90, 0.0, 0.90, 0.0])
    assert output["transition_score"][0, 2].item() > output["transition_score"][0, 3].item()
    assert output["transition_score"][0, 4].item() > output["transition_score"][0, 3].item()
    assert output["features"][0, :, 3].tolist() == pytest.approx(output["delta_p_action"][0].tolist())
    assert output["features"][0, :, 4].tolist() == pytest.approx(output["abs_delta_p_action"][0].tolist())


def test_manual_actionness_uncertainty_drives_uncertainty_peak_features() -> None:
    p_action = torch.tensor([[0.10, 0.20, 0.30]], dtype=torch.float32)
    manual_uncertainty = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    source = ZeroShotActionnessSource.from_manual(p_action=p_action, uncertainty=manual_uncertainty)

    output = source(torch.zeros(1, 3, 1))

    assert output["uncertainty"].tolist()[0] == pytest.approx([0.0, 1.0, 0.0])
    assert output["features"][0, :, 1].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert output["uncertainty_peak"].tolist()[0] == pytest.approx([0.0, 1.0, 0.0])


def test_adapter_default_scoring_is_transition_first_with_actionness_auxiliary() -> None:
    p_action = torch.tensor([[0.05, 0.05, 0.60, 0.99, 0.60, 0.05, 0.05]], dtype=torch.float32)
    adapter = DucaAcquisitionAdapter(actionness_source=_manual_source(p_action), budget=2, max_radius=0)

    scores = adapter.forward_scores(torch.zeros(1, 7, 1))

    assert adapter.actionness_weight < adapter.transition_weight
    assert adapter.actionness_weight < adapter.boundary_weight
    assert scores["center_scores"][0, 2].item() > scores["center_scores"][0, 3].item()
    assert scores["center_scores"][0, 4].item() > scores["center_scores"][0, 3].item()


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
        requested_budget=torch.tensor([3]),
        effective_budget=torch.tensor([3]),
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
            requested_budget=torch.tensor([2]),
            effective_budget=torch.tensor([2]),
        ).validate()

    with pytest.raises(ValueError, match="coordinate"):
        SparseTemporalGrid(
            selected_positions=selected_positions,
            selected_mask=selected_mask,
            original_length=8,
            valid_len=torch.tensor([8]),
            budget=3,
            requested_budget=torch.tensor([3]),
            effective_budget=torch.tensor([3]),
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
            requested_budget=torch.tensor([4]),
            effective_budget=torch.tensor([4]),
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


def test_temporal_max_gap_hole_loss_penalizes_empty_windows_and_backprops() -> None:
    clustered = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    covered = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]], requires_grad=True)
    valid = torch.ones(1, 7, dtype=torch.bool)

    clustered_loss = temporal_max_gap_hole_loss(
        clustered,
        valid,
        max_unselected_hole=2,
        min_window_mass=0.5,
    )
    covered_loss = temporal_max_gap_hole_loss(
        covered,
        valid,
        max_unselected_hole=2,
        min_window_mass=0.5,
    )

    assert clustered_loss.item() > covered_loss.item()
    clustered_loss.backward()
    assert clustered.grad is not None
    assert torch.isfinite(clustered.grad).all()
    assert clustered.grad.abs().sum().item() > 0.0


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


def test_budgeted_center_radius_decode_repairs_max_unselected_hole_without_changing_budget() -> None:
    scores = torch.tensor([[12.0, 11.0, 10.0, 9.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2]])
    radius = torch.zeros_like(scores)

    decoded = budgeted_center_radius_decode(
        center_scores=scores,
        radius=radius,
        budget=4,
        max_unselected_hole=2,
    )

    positions = [int(item) for item in decoded["selected_positions"][0].tolist() if int(item) >= 0]
    assert len(positions) == 4
    assert positions == sorted(set(positions))
    assert decoded["selected_mask"].sum().item() == 4
    holes = []
    run = 0
    for idx in range(scores.shape[1]):
        if idx in positions:
            run = 0
        else:
            run += 1
            holes.append(run)
    assert max(holes) <= 2
    assert decoded["max_gap_repair"][0]["repair_count"] > 0
    assert decoded["max_gap_repair"][0]["satisfied"] is True


def test_budgeted_center_radius_decode_fails_closed_when_max_gap_is_infeasible() -> None:
    scores = torch.linspace(1.0, 0.0, 10).view(1, 10)

    with pytest.raises(ValueError, match="max_unselected_hole is infeasible"):
        budgeted_center_radius_decode(
            center_scores=scores,
            radius=torch.zeros_like(scores),
            budget=4,
            max_unselected_hole=1,
        )


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
    assert grid.budget_unit == "detector_consumed_temporal_observation"
    assert grid.detector_consumes_selected_positions is True
    assert scores["radius"].max().item() <= 16.0
    grid.validate()


def test_detector_loss_only_backpropagates_to_adapter_parameters() -> None:
    torch.manual_seed(7)
    dense = torch.randn(2, 24, 3)
    adapter = DucaAcquisitionAdapter(feature_dim=3, budget=8, max_radius=4)
    detector = DetectorLossOnly()

    output = duca_forward_train(
        detector=detector,
        adapter=adapter,
        batch={"observations": dense},
        loss_weights={
            "teacher": 0.0,
            "boundary": 0.0,
            "hole": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
            "budget": 0.0,
            "detector": 1.0,
        },
    )
    output["losses"]["total_loss"].backward()

    grads = [
        param.grad.detach().abs().sum().item()
        for name, param in adapter.named_parameters()
        if ("center_head" in name or "encoder" in name) and param.grad is not None
    ]
    assert grads
    assert sum(grads) > 0.0


def test_selected_mask_st_hard_forward_matches_actual_decoded_union() -> None:
    dense = torch.randn(1, 16, 2)
    p_action = torch.tensor([[0.05, 0.10, 0.50, 0.95, 0.55, 0.10, 0.05, 0.90, 0.52, 0.08, 0.03, 0.02, 0.01, 0.0, 0.0, 0.0]])
    adapter = DucaAcquisitionAdapter(actionness_source=_manual_source(p_action), budget=6, max_radius=4)

    out = adapter.forward_acquire(dense)

    assert torch.equal(out["selected_mask_st"].detach().bool(), out["grid"].selected_mask)


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


def test_optional_coarse_hidden_dim_uses_zero_fallback_without_shape_mismatch() -> None:
    dense = torch.randn(1, 12, 3)
    adapter = DucaAcquisitionAdapter(
        feature_dim=3,
        budget=4,
        max_radius=2,
        coarse_hidden_dim=5,
        require_coarse_hidden_features=False,
    )

    output = adapter.forward_acquire(dense)

    assert output["detector_input"].shape == (1, 4, 3)
    assert output["uses_coarse_hidden_features"] is False
    assert output["coarse_hidden_features"].shape == (1, 12, 5)
    assert output["coarse_hidden_features"].abs().sum().item() == 0.0


def test_train_detector_batch_is_sanitized_of_teacher_payload() -> None:
    dense = torch.randn(1, 12, 3)
    adapter = DucaAcquisitionAdapter(feature_dim=3, budget=4, max_radius=2)
    detector = ForbiddenPayloadDetector()
    batch = {
        "observations": dense,
        "teacher_utility": torch.rand(1, 12),
        "nested": {"teacher_points": torch.rand(1, 12), "dense_teacher_payload": {"score": 1.0}},
    }

    duca_forward_train(detector=detector, adapter=adapter, batch=batch)


def test_test_forward_recursively_rejects_forbidden_payloads() -> None:
    dense = torch.randn(1, 12, 3)
    adapter = DucaAcquisitionAdapter(feature_dim=3, budget=4, max_radius=2)
    batch = {
        "observations": dense,
        "metas": [{"video_name": "x", "teacher_utility": [1, 2, 3]}],
    }

    with pytest.raises(ValueError, match="teacher_utility"):
        duca_forward_test(adapter=adapter, batch=batch)

    with pytest.raises(ValueError, match="prediction_cache"):
        duca_forward_test(adapter=adapter, batch={"observations": dense, "meta": {"prediction_cache": {"x": 1}}})


def test_manual_actionness_provenance_is_conservative_unless_declared() -> None:
    source = _manual_source(torch.tensor([[0.2, 0.8]]))
    provenance = source(torch.zeros(1, 2, 3))["provenance"]

    assert provenance["source_type"] == "manual"
    assert provenance["thumos_trained"] in {None, "unknown", True}
    assert provenance["uses_labels"] in {None, "unknown", True}


def test_dynamic_per_sample_budget_is_validated_per_row() -> None:
    dense = torch.randn(2, 12, 3)
    adapter = DucaAcquisitionAdapter(feature_dim=3, budget=8, max_radius=2)

    grid, _ = adapter.acquire(dense, budget=torch.tensor([4, 7]))

    assert grid.selected_count.tolist() == [4, 7]
    assert grid.requested_budget.tolist() == [4, 7]
    assert grid.effective_budget.tolist() == [4, 7]
    grid.validate()


def test_non_arange_dense_positions_fail_closed() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    dense_positions = torch.tensor([[0, 2, 4, 8]])

    with pytest.raises(ValueError, match="dense_positions"):
        budgeted_center_radius_decode(
            center_scores=scores,
            radius=torch.zeros_like(scores),
            budget=2,
            dense_positions=dense_positions,
        )


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


def test_duca_losses_include_temporal_max_gap_hole_loss() -> None:
    dense = torch.randn(1, 12, 3)
    adapter = DucaAcquisitionAdapter(feature_dim=3, budget=4, max_radius=1, max_unselected_hole=2)
    output = adapter.forward_acquire(dense)

    losses = duca_losses(
        output,
        loss_weights={
            "detector": 0.0,
            "teacher": 0.0,
            "boundary": 0.0,
            "actionness": 0.0,
            "hole": 0.0,
            "max_gap_hole": 1.0,
            "budget": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
        },
    )

    assert "temporal_max_gap_hole_loss" in losses
    assert torch.isfinite(losses["temporal_max_gap_hole_loss"])
    assert losses["temporal_max_gap_hole_loss"].requires_grad


def test_duca_inactive_zero_losses_do_not_overflow_when_score_sum_overflows() -> None:
    scores = torch.full((1, 768), -torch.finfo(torch.float32).max / 4.0, dtype=torch.float32)
    selected_mask = torch.zeros_like(scores)
    selected_mask[:, :384] = 1.0
    losses = duca_losses(
        scores=scores,
        selected_mask_st=selected_mask,
        budget=384,
        p_action=torch.full_like(scores, 0.5),
        loss_weights={
            "detector": 1.0,
            "teacher": 0.0,
            "boundary": 0.0,
            "actionness": 0.0,
            "hole": 0.0,
            "budget": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
        },
    )

    assert torch.isfinite(losses["detector_loss"])
    assert torch.isfinite(losses["teacher_utility_loss"])
    assert torch.isfinite(losses["boundary_coverage_loss"])
    assert torch.isfinite(losses["total_loss"])


def test_signed_teacher_utility_does_not_reward_negative_points() -> None:
    scores = torch.tensor([[2.0, 1.0, 0.5, 0.1]], requires_grad=True)
    selected_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], requires_grad=True)
    utility = torch.tensor([[-2.0, 1.0, 0.5, -0.5]])

    losses = duca_losses(
        scores=scores,
        selected_mask_st=selected_mask,
        budget=2,
        teacher_utility=utility,
        loss_weights={"teacher": 1.0, "budget": 0.0, "entropy": 0.0},
    )

    assert losses["teacher_utility_loss"].item() >= 0.0


def test_online_interfaces_do_not_accept_ledger_or_teacher_in_test_signature() -> None:
    for func in (duca_forward_train, duca_forward_test):
        assert "ledger" not in inspect.signature(func).parameters
    assert "teacher_utility" not in inspect.signature(duca_forward_test).parameters
    assert "ledger" not in inspect.signature(DucaAcquisitionAdapter.forward_acquire).parameters
    assert hasattr(DucaAcquisitionAdapter, "acquire")
