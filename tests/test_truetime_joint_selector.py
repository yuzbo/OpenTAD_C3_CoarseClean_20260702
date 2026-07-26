from __future__ import annotations

import pytest
import torch

from opentad.models.selectors.truetime_joint_selector import (
    TrueTimeRelaxedHardTopKSelector,
    selector_grad_norm,
)


def test_st_topk_selects_exact_k_and_reports_metrics() -> None:
    torch.manual_seed(7)
    selector = TrueTimeRelaxedHardTopKSelector(in_channels=3, selected_count=2, temperature=0.7)
    features = torch.randn(2, 3, 5, requires_grad=True)
    masks = torch.ones(2, 5, dtype=torch.bool)

    out = selector.forward_features(features, masks=masks, phase="joint_finetune")

    assert out["hard_mask"].shape == (2, 5)
    assert torch.equal(out["hard_mask"].sum(dim=1), torch.tensor([2.0, 2.0]))
    assert out["selected_indices"].shape == (2, 2)
    for row in out["selected_indices"]:
        assert torch.equal(row, row.sort().values)
        assert int(row.unique().numel()) == 2
    assert out["selected_count_mean"].item() == pytest.approx(2.0)
    assert out["selected_count_std"].item() == pytest.approx(0.0)
    assert out["entropy"].item() > 0.0
    assert out["coordinate_space"] == "selected_axis_index"
    assert out["true_time_source_axis"] == "true_time_dense_index"
    assert out["detector_gradient_mode"] == "st_sparse_gather"


def test_forward_train_selected_inputs_are_hard_values_with_selector_gradient() -> None:
    torch.manual_seed(11)
    selector = TrueTimeRelaxedHardTopKSelector(
        in_channels=4,
        selected_count=3,
        temperature=0.5,
        detector_gradient_mode="st_sparse_gather",
        slot_softmax_temperature=0.8,
        slot_distance_penalty=2.0,
    )
    features = torch.randn(2, 4, 6, requires_grad=True)
    masks = torch.ones(2, 6, dtype=torch.bool)
    detector_weights = torch.linspace(0.1, 1.3, 3).view(1, 1, 3)

    out = selector.forward_train(inputs=features, masks=masks, metas=[{}, {}])
    hard_inputs = torch.gather(
        features,
        dim=2,
        index=out["selector_outputs"]["selected_indices"][:, None, :].expand(-1, features.shape[1], -1),
    )
    detector_loss = (out["inputs"] * detector_weights).sum()
    detector_loss.backward()

    assert torch.allclose(out["inputs"].detach(), hard_inputs.detach(), atol=1e-6)
    assert out["metas"][0]["irregular_selected_count"] == 3
    assert out["metas"][0]["irregular_dense_valid_len"] == 6
    assert out["metas"][0]["irregular_selected_valid_len"] == 3
    assert selector_grad_norm(selector) > 0.0
    assert out["selector_outputs"]["selected_input_st_gradient_path"] == "st_sparse_gather"


def test_sparse_metadata_separates_dense_and_selected_valid_lengths() -> None:
    torch.manual_seed(17)
    selector = TrueTimeRelaxedHardTopKSelector(in_channels=2, selected_count=2, dense_len=6)
    features = torch.randn(1, 2, 6)
    masks = torch.tensor([[True, True, True, True, True, False]])

    out = selector.forward_test(features, masks=masks, metas=[{"video_name": "sample"}])
    meta = out["metas"][0]

    assert len(meta["selected_axis_to_true_time_dense_index"]) == 2
    assert meta["irregular_selected_count"] == 2
    assert meta["irregular_dense_valid_len"] == 5
    assert meta["irregular_selected_valid_len"] == 2
    assert meta["truetime_dense_valid_len"] == 5
    assert meta["detector_prediction_inverse_map_required"] is True


@pytest.mark.parametrize(
    ("shape", "expected_shape"),
    [
        ((2, 3, 6), (2, 3, 2)),
        ((2, 3, 6, 4, 5), (2, 3, 2, 4, 5)),
        ((2, 1, 3, 6, 4, 5), (2, 1, 3, 2, 4, 5)),
    ],
)
def test_forward_train_supports_3d_5d_and_6d_inputs(shape, expected_shape) -> None:
    torch.manual_seed(23)
    selector = TrueTimeRelaxedHardTopKSelector(in_channels=3, selected_count=2, dense_len=6)
    inputs = torch.randn(*shape, requires_grad=True)
    masks = torch.ones(shape[0], 6, dtype=torch.bool)

    out = selector.forward_train(inputs=inputs, masks=masks, metas=[{} for _ in range(shape[0])])
    loss = out["inputs"].sum()
    loss.backward()

    assert tuple(out["inputs"].shape) == expected_shape
    assert tuple(out["masks"].shape) == (shape[0], 2)
    assert selector_grad_norm(selector) > 0.0


def test_val_test_selection_rejects_gt_or_teacher_metadata_leakage() -> None:
    selector = TrueTimeRelaxedHardTopKSelector(in_channels=2, selected_count=1)
    features = torch.randn(1, 2, 3)
    masks = torch.ones(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="forbids GT"):
        selector.forward_test(features, masks=masks, metas=[{"video_name": "v", "gt_segments": [[0, 1]]}])

    with pytest.raises(ValueError, match="forbids teacher"):
        selector.forward_test(features, masks=masks, metas=[{"video_name": "v", "teacher_utility": [0.2]}])


def test_unsupported_detector_gradient_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="detector_gradient_mode"):
        TrueTimeRelaxedHardTopKSelector(in_channels=2, selected_count=1, detector_gradient_mode="hard_gather_only")
