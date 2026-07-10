from __future__ import annotations

import torch

import pytest

from opentad.models.utils.truetime_geometry import (
    TrueTimeMap,
    inverse_map_prediction_segments,
    remap_selected_axis_segments_to_true_time,
    truetime_map_from_metadata,
)


def test_truetime_selected_dense_roundtrip_preserves_fractional_positions() -> None:
    time_map = TrueTimeMap(selected_positions=[0, 2, 5, 9], dense_len=10, valid_len=10)

    selected_axis = torch.tensor([0.0, 0.5, 1.5, 3.0])
    true_time = time_map.selected_to_true(selected_axis)
    roundtrip = time_map.true_to_selected(true_time)

    assert torch.allclose(true_time, torch.tensor([0.0, 1.0, 3.5, 9.0]))
    assert torch.allclose(roundtrip, selected_axis, atol=1e-5)
    assert time_map.selected_axis_name == "selected_axis_index"
    assert time_map.true_time_axis_name == "true_time_dense_index"


def test_truetime_identity_map_preserves_half_open_right_edge() -> None:
    time_map = TrueTimeMap(selected_positions=range(8), dense_len=8, valid_len=8)
    segment = torch.tensor([[0.0, 8.0]])

    assert torch.equal(time_map.selected_to_true(segment), segment)


def test_truetime_sparse_map_roundtrip_preserves_head_and_tail_boundaries() -> None:
    time_map = TrueTimeMap(selected_positions=[1, 2, 3, 4], dense_len=8, valid_len=8)
    segments = torch.tensor([[0.0, 1.0], [1.0, 7.0], [4.0, 8.0]])

    selected = time_map.true_to_selected(segments)
    roundtrip = time_map.selected_to_true(selected)

    assert torch.allclose(roundtrip, segments, atol=1e-6)
    assert selected[0, 0].item() == -1.0
    assert selected[-1, 1].item() == 4.0


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


def test_metadata_selected_axis_remap_preserves_ordering() -> None:
    meta = {
        "detector_prediction_inverse_map_required": True,
        "selected_axis_to_true_time_dense_index": [1, 4, 8, 9],
        "truetime_dense_len": 10,
        "irregular_dense_valid_len": torch.tensor([10.0]),
        "irregular_selected_valid_len": [4.0],
        "irregular_selected_count": 4,
    }
    selected_segments = torch.tensor([[0.0, 1.0], [1.25, 2.5], [2.5, 3.0]])

    time_map = truetime_map_from_metadata(meta)
    true_segments = remap_selected_axis_segments_to_true_time(selected_segments, meta)

    assert torch.allclose(true_segments, torch.tensor([[1.0, 4.0], [5.0, 8.5], [8.5, 9.0]]))
    assert torch.all(true_segments[1:, 0] >= true_segments[:-1, 0])
    assert time_map.selected_len == 4


def test_metadata_selected_axis_remap_fails_closed_when_required_mapping_is_missing() -> None:
    meta = {
        "detector_prediction_inverse_map_required": True,
        "truetime_dense_len": 10,
        "irregular_dense_valid_len": 10,
        "irregular_selected_valid_len": 4,
    }

    with pytest.raises(ValueError, match="selected_axis_to_true_time_dense_index"):
        remap_selected_axis_segments_to_true_time(torch.tensor([[0.0, 1.0]]), meta)
