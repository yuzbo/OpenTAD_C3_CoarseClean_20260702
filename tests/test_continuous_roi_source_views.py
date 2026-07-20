import numpy as np
import pytest

pytest.importorskip("torch")

from opentad.datasets.transforms.native_crop import (
    CONTINUOUS_ROI_INPUT_SCHEMA,
    ContinuousRoiSourceViews,
    stable_video_key,
)


def _frame(height=180, width=320, offset=0):
    y, x = np.mgrid[:height, :width]
    return np.stack(
        (
            (x + offset) % 256,
            (y + offset) % 256,
            (x + y + offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_continuous_roi_source_views_preserve_source_pixels_and_add_global_only():
    frames = [_frame(offset=0), _frame(offset=7)]
    transform = ContinuousRoiSourceViews(global_size=96)
    output = transform(
        {
            "imgs": frames,
            "video_name": "video_validation_0000001",
            "window_start_frame": 64,
        }
    )
    inputs = output["continuous_roi_inputs"]
    assert set(inputs) == {"global", "source", "sample_key", "window_start"}
    assert inputs["source"].shape == (1, 3, 2, 180, 320)
    assert inputs["global"].shape == (1, 3, 2, 96, 96)
    assert inputs["source"].dtype == np.uint8
    assert np.array_equal(inputs["source"][0, :, 0].transpose(1, 2, 0), frames[0])
    assert int(inputs["sample_key"]) == int(
        stable_video_key("video_validation_0000001")
    )
    assert int(inputs["window_start"]) == 64
    audit = output["continuous_roi_geometry"]
    assert audit["schema_version"] == CONTINUOUS_ROI_INPUT_SCHEMA
    assert audit["source_resized_before_crop"] is False
    assert audit["decision_inputs"] == []
    assert audit["uses_gt"] is False


def test_continuous_roi_source_views_do_not_consume_gt():
    transform = ContinuousRoiSourceViews()
    base = {
        "imgs": [_frame()],
        "video_name": "video_validation_0000002",
        "window_start_frame": 0,
    }
    first = transform(dict(base, gt_segments=np.array([[1.0, 3.0]])))
    second = transform(dict(base, gt_segments=np.array([[100.0, 200.0]])))
    for key in ("global", "source"):
        assert np.array_equal(
            first["continuous_roi_inputs"][key],
            second["continuous_roi_inputs"][key],
        )


def test_continuous_roi_source_views_fail_closed_on_geometry_drift():
    transform = ContinuousRoiSourceViews()
    with pytest.raises(ValueError, match="changed source geometry"):
        transform(
            {
                "imgs": [_frame(), _frame(height=179)],
                "video_name": "video_validation_0000003",
            }
        )


def test_continuous_roi_source_views_fail_closed_on_wrong_frozen_geometry():
    transform = ContinuousRoiSourceViews()
    with pytest.raises(ValueError, match="frozen to source geometry"):
        transform(
            {
                "imgs": [_frame(height=160, width=284)],
                "video_name": "video_validation_0000004",
            }
        )
