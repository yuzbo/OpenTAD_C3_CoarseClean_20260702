from __future__ import annotations

import json

import numpy as np
import pytest

try:
    import torch

    from opentad.datasets.base.sliding_dataset import SlidingWindowDataset
    from opentad.models.selectors.duca_allocation_artifact_replay import (
        DucaAllocationArtifactReplaySelector,
    )
    from opentad.models.utils.truetime_geometry import SELECTED_AXIS
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"OpenTAD dataset dependencies are unavailable: {exc}", allow_module_level=True)


class _ToySlidingDataset(SlidingWindowDataset):
    def get_gt(self, _video_info, thresh=0.0):
        del thresh
        return {
            "gt_segments": np.asarray([[8.0, 10.0]], dtype=np.float32),
            "gt_labels": np.asarray([0], dtype=np.int32),
        }


def _dataset(tmp_path, *, include_background_windows: bool):
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "video": {
                        "subset": "training",
                        "frame": 20,
                        "duration": 20.0,
                        "annotations": [],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    class_map = tmp_path / "classes.txt"
    class_map.write_text("action\n", encoding="utf-8")
    return _ToySlidingDataset(
        ann_file=annotation,
        subset_name="training",
        data_path=tmp_path,
        pipeline=[],
        class_map=class_map,
        test_mode=False,
        feature_stride=1,
        sample_stride=1,
        window_size=8,
        window_overlap_ratio=0.5,
        ioa_thresh=1.0e-8,
        include_background_windows=include_background_windows,
    )


def test_r0_holdout_export_can_include_background_windows_without_fake_gt(tmp_path) -> None:
    foreground_only = _dataset(tmp_path / "foreground", include_background_windows=False)
    exhaustive = _dataset(tmp_path / "exhaustive", include_background_windows=True)

    assert len(exhaustive) > len(foreground_only)
    assert any(row[2]["gt_segments"].shape == (0, 2) for row in exhaustive.data_list)
    assert all(row[2]["gt_segments"].shape[1:] == (2,) for row in exhaustive.data_list)


def test_r0_selected_axis_replay_gathers_exact_positions_and_emits_inverse_map() -> None:
    selector = DucaAllocationArtifactReplaySelector.__new__(
        DucaAllocationArtifactReplaySelector
    )
    torch.nn.Module.__init__(selector)
    selector.budget = 2
    selector.detector_output_coordinate_space = SELECTED_AXIS
    selector.family_key = "A_exact_uniform"
    selector.artifact_sha256 = "a" * 64
    selector._allocation_rows = {
        "video|0": {
            "valid_len": 4,
            "positions": [0, 3],
            "privileged": False,
            "deployable": True,
        }
    }

    inputs = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1, 1)
    output = selector.forward_test(
        inputs=inputs,
        masks=torch.ones(1, 4, dtype=torch.bool),
        metas=[{"video_name": "video", "window_start_frame": 0}],
    )

    assert output["inputs"].flatten().tolist() == [0.0, 3.0]
    assert output["masks"].tolist() == [[True, True]]
    meta = output["metas"][0]
    assert meta["detector_output_coordinate_space"] == SELECTED_AXIS
    assert meta["detector_prediction_inverse_map_required"] is True
    assert meta["selected_axis_to_true_time_dense_index"] == [0, 3]
    assert meta["duca_online_selected_axis_remap"]["selected_to_original"] == {
        0: 0,
        1: 3,
    }
