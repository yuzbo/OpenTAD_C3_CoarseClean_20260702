from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("torch post-processing tests run in the Linux/remote OpenTAD environment", allow_module_level=True)

torch = pytest.importorskip("torch")

from opentad.models.utils.post_processing import utils as post_utils


def test_nonuniform_selected_axis_segments_convert_back_to_dense_seconds() -> None:
    meta = {
        "fps": 30.0,
        "snippet_stride": 4,
        "offset_frames": 0,
        "window_start_frame": 120,
        "duration": 100.0,
        "irregular_native_axis": False,
        "irregular_selected_positions": [0.0, 2.0, 10.0, 20.0],
        "irregular_selected_valid_len": [24.0],
    }
    selected_axis_segments = torch.tensor([[1.0, 3.0]])

    seconds = post_utils.convert_to_seconds(selected_axis_segments.clone(), meta)

    expected_dense = torch.tensor([[2.0, 20.0]])
    expected_seconds = (expected_dense * 4.0 + 120.0) / 30.0
    assert torch.allclose(seconds, expected_seconds)
