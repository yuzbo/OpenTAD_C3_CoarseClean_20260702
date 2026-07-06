from __future__ import annotations

import torch

from opentad.models.utils.truetime_geometry import TrueTimeMap, inverse_map_prediction_segments


def test_truetime_selected_dense_roundtrip_preserves_fractional_positions() -> None:
    time_map = TrueTimeMap(selected_positions=[0, 2, 5, 9], dense_len=10, valid_len=10)

    selected_axis = torch.tensor([0.0, 0.5, 1.5, 3.0])
    true_time = time_map.selected_to_true(selected_axis)
    roundtrip = time_map.true_to_selected(true_time)

    assert torch.allclose(true_time, torch.tensor([0.0, 1.0, 3.5, 9.0]))
    assert torch.allclose(roundtrip, selected_axis, atol=1e-5)
    assert time_map.selected_axis_name == "selected_axis_index"
    assert time_map.true_time_axis_name == "true_time_dense_index"


def test_truetime_segment_remap_is_explicit_about_coordinate_spaces() -> None:
    time_map = TrueTimeMap(selected_positions=[0, 2, 5, 9], dense_len=10, valid_len=10)
    selected_segments = torch.tensor([[0.0, 1.0], [1.5, 3.0]])

    true_segments = time_map.remap_segments(
        selected_segments,
        source_coordinate_space="selected_axis_index",
        target_coordinate_space="true_time_dense_index",
    )
    selected_roundtrip = time_map.remap_segments(
        true_segments,
        source_coordinate_space="true_time_dense_index",
        target_coordinate_space="selected_axis_index",
    )

    assert torch.allclose(true_segments, torch.tensor([[0.0, 2.0], [3.5, 9.0]]))
    assert torch.allclose(selected_roundtrip, selected_segments, atol=1e-5)


def test_prediction_inverse_map_records_selected_axis_source() -> None:
    time_map = TrueTimeMap(selected_positions=[1, 3, 6, 7], dense_len=8, valid_len=8)
    predictions = {
        "segments": torch.tensor([[0.0, 2.0], [1.0, 3.0]]),
        "scores": torch.tensor([0.9, 0.2]),
        "coordinate_space": "selected_axis_index",
    }

    mapped = inverse_map_prediction_segments(predictions, time_map)

    assert mapped["coordinate_space"] == "true_time_dense_index"
    assert mapped["source_coordinate_space"] == "selected_axis_index"
    assert torch.allclose(mapped["segments"], torch.tensor([[1.0, 6.0], [3.0, 7.0]]))
    assert torch.equal(mapped["scores"], predictions["scores"])
