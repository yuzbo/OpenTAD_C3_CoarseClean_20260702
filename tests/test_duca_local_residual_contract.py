from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip(
        "local Windows torch/c10.dll import is unstable; Linux remote runs this suite",
        allow_module_level=True,
    )

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.duca.acquisition import DucaAcquisitionAdapter
from opentad.models.duca.structured_selection import (
    exact_uniform_positions,
    normalize_scores_within_exact_uniform_cells,
)
from opentad.models.duca.transition_only import continuous_policy_logits
from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector


def _local_adapter() -> DucaAcquisitionAdapter:
    return DucaAcquisitionAdapter(
        feature_dim=3,
        hidden_dim=8,
        budget=4,
        acquisition_policy="local_cell_deformation",
        structured_temperature=0.7,
        selector_variant="transition_only",
        coarse_hidden_dim=4,
        require_coarse_hidden_features=True,
        hard_max_gap_repair=False,
    )


def _delta_residual_adapter(*, residual_scale: float = 0.25) -> DucaAcquisitionAdapter:
    return DucaAcquisitionAdapter(
        feature_dim=3,
        hidden_dim=8,
        budget=4,
        acquisition_policy="local_cell_deformation",
        structured_temperature=0.7,
        selector_variant="transition_only",
        coarse_hidden_dim=4,
        require_coarse_hidden_features=True,
        local_cell_base_policy="abs_delta_actionness",
        local_cell_residual_scale=residual_scale,
        hard_max_gap_repair=False,
    )


def _frozen_official_asformer_cfg() -> dict:
    return {
        "type": "C3CoarseProbeActionnessSource",
        "source_name": "frozen_p0_official_asformer",
        "probe_model": "official-action-seg",
        "official_action_seg_backend": "official_asformer",
        "spatial_size": 16,
        "tcn_hidden_dim": 16,
        "official_num_layers": 1,
        "dropout": 0.0,
        "frozen": True,
        "trainable": False,
        "thumos_trained": True,
        "uses_labels": True,
        "uses_teacher": False,
        "uses_gt": True,
        "uses_prediction_cache": False,
        "trained_with_thumos_labels": True,
        "trained_with_gt_segments": True,
        "training_dataset": "THUMOS14",
        "training_supervision_scope": "train_only",
        "uses_labels_at_inference": False,
        "uses_gt_at_inference": False,
        "uses_teacher_at_inference": False,
        "uses_prediction_cache_at_inference": False,
        "calibration_split": "none",
    }


def _local_selector(*, companion_fraction: float = 0.0) -> DucaOnlineFrameSelector:
    return DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        dense_window_size=8,
        selector_hidden_channels=8,
        acquisition_policy="local_cell_deformation",
        structured_temperature=0.7,
        local_cell_force_exact_uniform=False,
        local_cell_base_policy="abs_delta_actionness",
        local_cell_residual_scale=0.25,
        local_cell_detector_grid_mode="selected",
        inference_policy_alpha=1.0,
        training_uniform_companion_fraction=companion_fraction,
        selector_variant="transition_only",
        coarse_hidden_dim=16,
        use_coarse_hidden_features=True,
        require_coarse_hidden_features=True,
        allow_frozen_coarse_probe=True,
        policy_hidden_gradient_scale=0.0,
        auxiliary_hidden_gradient_scale=0.0,
        max_unselected_hole=None,
        hard_max_gap_repair=False,
        soft_max_gap_loss_enabled=False,
        detector_gradient_mode="local_cell_straight_through",
        coordinate_space="original_time",
        detector_output_coordinate_space="selected_axis_index",
        forbid_external_actionness=True,
        actionness_source_cfg=_frozen_official_asformer_cfg(),
        loss_weights={
            "actionness": 0.0,
            "transition": 0.0,
            "transition_boundary": 0.0,
            "teacher": 0.0,
            "boundary": 0.0,
            "hole": 0.0,
            "max_gap_hole": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
            "budget": 0.0,
        },
    )


def _gather_6d(inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    index = positions[:, None, None, :, None, None].expand(
        inputs.shape[0],
        inputs.shape[1],
        inputs.shape[2],
        positions.shape[1],
        inputs.shape[4],
        inputs.shape[5],
    )
    return torch.gather(inputs, 3, index)


def test_local_cell_homotopy_starts_at_exact_uniform_without_scorer_gradient() -> None:
    adapter = _local_adapter().train()
    learned = torch.tensor(
        [[0.0, 5.0, 0.0, 4.0, 3.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    valid = torch.ones(1, 8, dtype=torch.bool)
    budgets = torch.tensor([4])

    decoded = adapter._decode_local_cell(
        learned,
        None,
        valid,
        budgets,
        stable_selection=False,
        policy_mix_alpha=0.0,
    )

    expected = exact_uniform_positions(8, 4)
    assert torch.equal(decoded["selected_positions"][0], expected)
    assert decoded["selection_path"] == "local_cell_uniform_reference"
    assert decoded["policy_mix_alpha"] == pytest.approx(0.0)
    weighted = torch.arange(8, dtype=torch.float32)
    (decoded["soft_coverage"] * weighted).sum().backward()
    assert torch.equal(learned.grad, torch.zeros_like(learned.grad))


def test_local_cell_homotopy_reaches_learned_policy_and_reports_mixed_logits() -> None:
    adapter = _local_adapter().train()
    learned = torch.tensor(
        [[0.0, 5.0, 0.0, 4.0, 3.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    valid = torch.ones(1, 8, dtype=torch.bool)
    budgets = torch.tensor([4])

    midpoint = adapter._decode_local_cell(
        learned,
        None,
        valid,
        budgets,
        stable_selection=False,
        policy_mix_alpha=0.5,
    )
    endpoint = adapter._decode_local_cell(
        learned,
        None,
        valid,
        budgets,
        stable_selection=False,
        policy_mix_alpha=1.0,
    )

    expected_midpoint = continuous_policy_logits(
        learned,
        valid,
        k=4,
        alpha=0.5,
    )
    assert midpoint["selection_path"] == "local_cell_continuous_homotopy"
    assert torch.allclose(midpoint["decode_policy_logits"], expected_midpoint)
    assert endpoint["selection_path"] == "local_cell_learned"
    assert endpoint["selected_positions"].tolist() == [[1, 3, 4, 7]]
    weighted = torch.arange(8, dtype=torch.float32)
    (endpoint["soft_coverage"] * weighted).sum().backward()
    assert float(learned.grad.abs().sum().item()) > 0.0


def test_delta_residual_policy_has_detached_base_and_bounded_residual() -> None:
    adapter = _delta_residual_adapter(residual_scale=0.25).train()
    residual = torch.tensor(
        [[0.0, 2.0, -1.0, 1.0, 0.5, -2.0, 3.0, -0.5]],
        requires_grad=True,
    )
    delta = torch.tensor(
        [[0.1, 0.9, 0.8, 0.2, 0.1, 0.7, 0.2, 0.6]],
        requires_grad=True,
    )
    valid = torch.ones(1, 8, dtype=torch.bool)
    decoded = adapter._decode_local_cell(
        residual,
        delta,
        valid,
        torch.tensor([4]),
        stable_selection=False,
        policy_mix_alpha=1.0,
    )

    expected_base = normalize_scores_within_exact_uniform_cells(delta.detach(), k=4)
    expected_residual = 0.25 * torch.tanh(residual)
    assert torch.allclose(decoded["local_cell_normalized_base_scores"], expected_base)
    assert torch.allclose(
        decoded["local_cell_bounded_residual_scores"], expected_residual
    )
    assert torch.allclose(
        decoded["local_cell_utility_scores"], expected_base + expected_residual
    )
    assert float(decoded["local_cell_bounded_residual_scores"].abs().max()) <= 0.25

    decoded["soft_coverage"].square().sum().backward()
    assert residual.grad is not None and float(residual.grad.abs().sum()) > 0.0
    assert delta.grad is None


def test_pure_delta_endpoint_selects_cellwise_delta_maxima() -> None:
    adapter = _delta_residual_adapter(residual_scale=0.0).train()
    residual = torch.zeros(1, 8, requires_grad=True)
    delta = torch.tensor([[0.1, 0.9, 0.8, 0.2, 0.1, 0.7, 0.2, 0.6]])
    valid = torch.ones(1, 8, dtype=torch.bool)

    decoded = adapter._decode_local_cell(
        residual,
        delta,
        valid,
        torch.tensor([4]),
        stable_selection=False,
        policy_mix_alpha=1.0,
    )

    assert decoded["selected_positions"].tolist() == [[1, 2, 5, 7]]
    assert decoded["selection_path"] == "local_cell_learned"
    assert decoded["local_cell_base_policy"] == "abs_delta_actionness"


def test_delta_residual_alpha_zero_is_exact_uniform() -> None:
    adapter = _delta_residual_adapter().train()
    decoded = adapter._decode_local_cell(
        torch.tensor([[0.0, 4.0, 0.0, 3.0, 2.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 0.9, 0.0, 0.0, 0.8, 0.1, 0.7]]),
        torch.ones(1, 8, dtype=torch.bool),
        torch.tensor([4]),
        stable_selection=False,
        policy_mix_alpha=0.0,
    )

    assert torch.equal(decoded["selected_positions"][0], exact_uniform_positions(8, 4))
    assert decoded["selection_path"] == "local_cell_uniform_reference"


def test_local_cell_detector_bridge_is_hard_forward_and_scorer_only_backward() -> None:
    torch.manual_seed(17)
    selector = _local_selector().train()
    inputs = torch.randn(1, 1, 3, 8, 16, 16)
    out = selector.forward_train(
        inputs=inputs,
        masks=torch.ones(1, 8, dtype=torch.bool),
        metas=[{"video_name": "local_bridge"}],
        gt_segments=[torch.tensor([[1.0, 6.0]])],
        gt_labels=[torch.tensor([1])],
    )
    positions = out["selector_outputs"]["grid"].selected_positions

    assert torch.equal(out["inputs"].detach(), _gather_6d(inputs, positions))
    assert torch.equal(
        out["selector_outputs"]["detector_grid_positions"],
        positions,
    )
    assert out["selector_outputs"]["selected_input_st_gradient_path"] == (
        "local_cell_straight_through"
    )
    out["inputs"].square().mean().backward()
    scorer_grad = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in selector.adapter.transition_scorer.parameters()
        if parameter.grad is not None
    )
    assert scorer_grad > 0.0
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in selector.raw_actionness_source.parameters()
    )


class _PositionScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dim = 37
        self.scorer_hidden_dim = 1
        self.logits = nn.Parameter(
            torch.tensor([0.0, 5.0, 0.0, 4.0, 3.0, 0.0, 0.0, 0.0])
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.logits[None, : descriptors.shape[1]].expand(
            descriptors.shape[0], -1
        ) + descriptors.sum(dim=-1) * 0.0


def test_local_cell_uniform_companion_blocks_only_companion_bridge_row() -> None:
    torch.manual_seed(23)
    selector = _local_selector(companion_fraction=0.5).train()
    selector.adapter.transition_scorer = _PositionScorer()
    inputs = torch.randn(2, 1, 3, 8, 16, 16)
    out = selector.forward_train(
        inputs=inputs,
        masks=torch.ones(2, 8, dtype=torch.bool),
        metas=[{"video_name": "companion"}, {"video_name": "learned"}],
        gt_segments=[torch.tensor([[1.0, 6.0]]), torch.tensor([[1.0, 6.0]])],
        gt_labels=[torch.tensor([1]), torch.tensor([1])],
    )
    state = out["selector_outputs"]
    companion = state["training_uniform_companion_mask"]
    center_scores = state["center_scores"]
    center_scores.retain_grad()

    assert int(companion.sum().item()) == 1
    companion_row = int(torch.nonzero(companion, as_tuple=False).item())
    learned_row = 1 - companion_row
    assert torch.equal(
        state["grid"].selected_positions[companion_row],
        exact_uniform_positions(8, 4),
    )
    assert state["grid"].selected_positions[learned_row].tolist() == [1, 3, 4, 7]

    out["inputs"].square().mean().backward()
    assert center_scores.grad[companion_row].abs().sum().item() == pytest.approx(0.0)
    assert center_scores.grad[learned_row].abs().sum().item() > 0.0


def test_local_cell_bridge_rejects_global_policy() -> None:
    with pytest.raises(ValueError, match="requires local_cell_deformation"):
        DucaOnlineFrameSelector(
            in_channels=3,
            budget=4,
            dense_window_size=8,
            selector_hidden_channels=8,
            acquisition_policy="global_structured_topk",
            selector_variant="transition_only",
            coarse_hidden_dim=16,
            max_unselected_hole=3,
            hard_max_gap_repair=False,
            detector_gradient_mode="local_cell_straight_through",
            forbid_external_actionness=True,
            actionness_source_cfg=_frozen_official_asformer_cfg(),
            allow_frozen_coarse_probe=True,
        )
