import numpy as np
import pytest
import torch

from opentad.datasets.transforms.phystime_raw import BuildPhysTimeRawFrameGeometry


def make_sample(with_gt=True):
    frame_indices = np.array([100, 108, 132, 140], dtype=np.int64)
    sample = {
        "frame_inds": frame_indices.copy(),
        "selected_raw_frame_indices": frame_indices.copy(),
        "selected_dense_indices": np.array([0, 2, 8, 10], dtype=np.float32),
        "masks": torch.ones(4, dtype=torch.bool),
        "snippet_stride": 4,
        "fps": 20.0,
        "avg_fps": 20.0,
        "total_frames": 400,
        "duration": 20.0,
        "irregular_dense_valid_len": 11,
        "irregular_sampling_strategy": "random_fixed_subsample",
        "irregular_sampling_scope": "within_accepted_window",
        "irregular_window_crop_uses_gt": bool(with_gt),
        "irregular_subsample_uses_gt": False,
        "irregular_native_axis": True,
        "remap_gt_to_selected_axis": False,
        "gt_remapped_to_selected_axis": False,
    }
    if with_gt:
        sample["gt_segments"] = np.array([[1.0, 6.0]], dtype=np.float32)
    return sample


def test_raw_geometry_uses_original_video_time_and_dense_cell_support():
    out = BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=True)(make_sample())

    assert out["phystime_timestamps_sec"] == pytest.approx([5.0, 5.4, 6.6, 7.0])
    assert out["phystime_support_intervals_sec"][0] == pytest.approx([5.0, 5.1])
    assert out["phystime_support_intervals_sec"][1] == pytest.approx([5.3, 5.5])
    assert torch.allclose(out["gt_segments"], torch.tensor([[5.2, 6.2]]))
    assert out["prediction_time_unit"] == "seconds"
    assert out["gt_time_unit"] == "seconds"
    assert out["phystime_support_provenance"] == "original_raw_dense_cells"
    assert out["phystime_sampling_uses_gt"] is False


def test_raw_geometry_forbids_selected_axis_ground_truth():
    sample = make_sample()
    sample["remap_gt_to_selected_axis"] = True

    with pytest.raises(ValueError, match="selected-axis"):
        BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=True)(sample)


def test_raw_geometry_rejects_misaligned_selected_positions():
    sample = make_sample()
    sample["frame_inds"][2] += 1

    with pytest.raises(ValueError, match="audit metadata"):
        BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=True)(sample)


def test_raw_geometry_test_mode_does_not_require_ground_truth():
    out = BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=False)(make_sample(with_gt=False))

    assert "gt_time_unit" not in out
    assert out["prediction_time_unit"] == "seconds"
