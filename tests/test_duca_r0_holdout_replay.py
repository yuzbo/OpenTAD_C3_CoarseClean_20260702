from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    from opentad.datasets.base.sliding_dataset import SlidingWindowDataset
    from opentad.models.selectors.duca_allocation_artifact_replay import (
        DucaAllocationArtifactReplaySelector,
    )
    from opentad.models.utils.truetime_geometry import SELECTED_AXIS
    from tools.bata.duca_allocation_families import (
        PhysicalAxis,
        resolve_physical_cap,
    )
    from tools.bata.diagnose_duca_allocation_family_ceiling import allocation_metrics
    from tools.bata.duca_exact_physical_solver import solve_boundary_burst_oracle
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"OpenTAD dataset dependencies are unavailable: {exc}", allow_module_level=True)


class _ToySlidingDataset(SlidingWindowDataset):
    def get_gt(self, _video_info, thresh=0.0):
        del thresh
        return {
            "gt_segments": np.asarray([[6.0, 10.0]], dtype=np.float32),
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
    assert all(row[2]["gt_boundary_validity"].shape[1:] == (2,) for row in exhaustive.data_list)


def test_sliding_windows_mark_crop_cut_endpoints_invalid(tmp_path) -> None:
    dataset = _dataset(tmp_path, include_background_windows=True)
    rows = {
        int(window[0]): annotation
        for _, _, annotation, window in dataset.data_list
        if len(annotation["gt_segments"])
    }

    assert rows[0]["gt_boundary_validity"].tolist() == [[True, False]]
    assert rows[4]["gt_boundary_validity"].tolist() == [[True, True]]
    assert rows[8]["gt_boundary_validity"].tolist() == [[False, True]]


@pytest.mark.parametrize("radius,quota", [(2, 3), (4, 5)])
def test_r0_boundary_burst_oracle_enforces_exact_k_g_bilateral_and_quota(
    radius: int,
    quota: int,
) -> None:
    axis = PhysicalAxis.from_source_frames(
        [4 * index for index in range(48)],
        decoder_fps=30.0,
        annotation_fps=30.0,
    )
    cap = resolve_physical_cap(
        axis,
        requested_budget=24,
        policy="explicit_frames",
        value=12,
    )
    solved = solve_boundary_burst_oracle(
        axis,
        [[14.2, 33.4]],
        [[True, True]],
        requested_budget=24,
        cap=cap,
        radius=radius,
        quota=quota,
        max_unselected_hole=2,
    )

    assert len(solved.positions) == 24
    assert solved.max_unselected_hole <= 2
    assert solved.residual_fill_count > 0
    assert solved.background_selected_count > 0
    assert len(solved.endpoint_contracts) == 2
    assert all(row["quota_pass"] for row in solved.endpoint_contracts)
    assert all(row["bilateral_pass"] for row in solved.endpoint_contracts)
    assert all(row["selected_in_radius"] >= quota for row in solved.endpoint_contracts)


def test_r0_boundary_burst_oracle_excludes_invalid_crop_endpoint() -> None:
    axis = PhysicalAxis.from_source_frames(
        [4 * index for index in range(16)],
        decoder_fps=30.0,
        annotation_fps=30.0,
    )
    cap = resolve_physical_cap(
        axis,
        requested_budget=8,
        policy="explicit_frames",
        value=12,
    )
    solved = solve_boundary_burst_oracle(
        axis,
        [[0.0, 8.0]],
        [[False, True]],
        requested_budget=8,
        cap=cap,
        radius=2,
        quota=3,
        max_unselected_hole=2,
    )

    assert solved.invalid_endpoint_count == 1
    assert len(solved.endpoint_contracts) == 1
    assert solved.endpoint_contracts[0]["endpoint"] == "end"


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


def test_r0_launcher_is_headroom_gated_and_uses_constrained_burst_families() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_duca_boundary_burst_r0_holdout_map_gpu1.sh").read_text(
        encoding="utf-8"
    )

    assert "build_duca_r0_boundary_burst_oracles" in script
    assert "R2Q3_privileged_boundary_burst" in script
    assert "R4Q5_privileged_boundary_burst" in script
    assert "Z_unrestricted_gt_oracle" in script
    assert "finalize_duca_r0_boundary_burst" in script
    assert '"${TRAIN_BLOCK_LIST}" "${EVAL_BLOCKED}"' in script
    assert '"${HOLDOUT_BLOCK_LIST}" "${EVAL_BLOCKED}"' not in script
    assert "--bootstrap-samples 1000" in script
    assert "--required-headroom-percentage-points 0.20" in script


def test_r0_allocation_metrics_exclude_crop_cut_endpoint() -> None:
    metrics = allocation_metrics(
        [0, 5, 10, 15],
        [[0.0, 15.0]],
        valid_len=16,
        radii=(0, 1),
        short_action_max_length=16.0,
        gt_boundary_validity=[[False, True]],
    )

    assert metrics["endpoint_count"] == 1
    assert metrics["invalid_crop_endpoint_count"] == 1
    assert metrics["mean_endpoint_distance"] == pytest.approx(0.0)
    assert metrics["endpoint_recall_r0"] == pytest.approx(1.0)
    assert metrics["both_boundary_recall_r0"] is None
