import copy

import pytest

torch = pytest.importorskip("torch")

from opentad.datasets.transforms.loading import RandomTrunc
from opentad.datasets.transforms.phystime import (
    BuildSelectedAxisFeatureBaseline,
    BuildPairedPhysTimeFeatureViews,
    BuildPhysTimeFeatureGeometry,
    SampleIrregularFeatureObservations,
)
from opentad.datasets.transforms.formatting import Collect
from opentad.datasets.builder import collate


def test_random_trunc_records_absolute_feature_crop_origin(monkeypatch):
    transform = RandomTrunc(trunc_len=4, trunc_thresh=0.0, has_action=False)
    monkeypatch.setattr("opentad.datasets.transforms.loading.random.randint", lambda low, high: 2)
    results = {
        "feats": torch.arange(12, dtype=torch.float32).reshape(6, 2),
        "gt_segments": torch.tensor([[1.0, 5.0]]),
        "gt_labels": torch.tensor([0]),
        "feature_start_idx": 10,
    }

    output = transform(results)

    assert output["phystime_window_start_feature_idx"] == 12
    assert output["phystime_window_valid_feature_count"] == 4
    assert torch.equal(output["feats"], torch.arange(4, 12, dtype=torch.float32).reshape(4, 2))


def test_uniform_sampling_preserves_original_indices_and_exposes_gaps():
    sampler = SampleIrregularFeatureObservations(num_observations=3, strategy="uniform")
    original_features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    results = {
        "feats": original_features.clone(),
        "masks": torch.ones(6, dtype=torch.bool),
        "phystime_window_start_feature_idx": 10,
        "video_name": "video_test",
    }

    output = sampler(results)

    assert output["phystime_selected_feature_indices"] == [10, 12, 15]
    assert torch.equal(output["masks"], torch.ones(3, dtype=torch.bool))
    assert torch.equal(output["feats"], original_features[[0, 2, 5]])


def test_sampling_decision_does_not_read_ground_truth():
    sampler = SampleIrregularFeatureObservations(
        num_observations=4,
        strategy="random",
        seed=17,
        stochastic=False,
    )
    common = {
        "feats": torch.randn(8, 3),
        "masks": torch.ones(8, dtype=torch.bool),
        "phystime_window_start_feature_idx": 3,
        "video_name": "same_video",
    }
    first = dict(common, gt_segments=torch.tensor([[0.0, 1.0]]), gt_labels=torch.tensor([0]))
    second = dict(common, gt_segments=torch.tensor([[6.0, 8.0]]), gt_labels=torch.tensor([9]))

    first_output = sampler(copy.deepcopy(first))
    second_output = sampler(copy.deepcopy(second))

    assert first_output["phystime_selected_feature_indices"] == second_output["phystime_selected_feature_indices"]
    assert torch.equal(first_output["feats"], second_output["feats"])


def test_feature_geometry_converts_gt_to_absolute_seconds_without_filling_dropped_cells():
    transform = BuildPhysTimeFeatureGeometry(convert_gt_to_seconds=True)
    results = {
        "feats": torch.randn(3, 4),
        "masks": torch.ones(3, dtype=torch.bool),
        "phystime_selected_feature_indices": [10, 12, 15],
        "phystime_window_start_feature_idx": 10,
        "phystime_window_valid_feature_count": 6,
        "gt_segments": torch.tensor([[1.0, 4.0]]),
        "fps": 4.0,
        "snippet_stride": 4,
        "offset_frames": 0,
        "duration": 20.0,
    }

    output = transform(results)

    assert output["phystime_timestamps_sec"] == pytest.approx([10.0, 12.0, 15.0])
    assert torch.allclose(
        torch.tensor(output["phystime_support_intervals_sec"]),
        torch.tensor([[9.5, 10.5], [11.5, 12.5], [14.5, 15.5]]),
    )
    assert output["phystime_domain_start_sec"] == pytest.approx(9.5)
    assert output["phystime_domain_end_sec"] == pytest.approx(15.5)
    assert torch.allclose(output["gt_segments"], torch.tensor([[11.0, 14.0]]))
    assert output["gt_time_unit"] == "seconds"
    assert output["prediction_time_unit"] == "seconds"
    assert output["phystime_support_provenance"] == "original_feature_ownership_cells"


def test_feature_geometry_rejects_rank_adjacent_sparse_frame_provenance():
    transform = BuildPhysTimeFeatureGeometry(convert_gt_to_seconds=False)
    results = {
        "feats": torch.randn(2, 4),
        "masks": torch.ones(2, dtype=torch.bool),
        "phystime_selected_feature_indices": [0, 1],
        "phystime_window_start_feature_idx": 0,
        "phystime_window_valid_feature_count": 2,
        "fps": 4.0,
        "snippet_stride": 4,
        "offset_frames": 0,
        "duration": 2.0,
        "phystime_input_support_provenance": "rank_adjacent_sparse_frames",
    }

    with pytest.raises(ValueError, match="contiguous feature-token ownership"):
        transform(results)


def test_paired_view_transform_emits_two_audited_views_with_shared_seconds_gt():
    transform = BuildPairedPhysTimeFeatureViews(
        first_view=dict(num_observations=4, strategy="uniform", stochastic=False, seed=3),
        second_view=dict(num_observations=4, strategy="random", stochastic=False, seed=11),
    )
    results = {
        "video_name": "paired_video",
        "feats": torch.arange(24, dtype=torch.float32).reshape(8, 3),
        "masks": torch.ones(8, dtype=torch.bool),
        "gt_segments": torch.tensor([[1.0, 6.0]]),
        "gt_labels": torch.tensor([1]),
        "phystime_window_start_feature_idx": 2,
        "phystime_window_valid_feature_count": 8,
        "fps": 4.0,
        "snippet_stride": 4,
        "offset_frames": 0,
        "duration": 20.0,
    }

    output = transform(results)

    assert output["feats"].shape == output["paired_feats"].shape == (4, 3)
    assert output["masks"].shape == output["paired_masks"].shape == (4,)
    assert torch.allclose(output["gt_segments"], torch.tensor([[3.0, 8.0]]))
    assert output["gt_time_unit"] == "seconds"
    assert output["paired_metas"]["gt_time_unit"] == "seconds"
    assert output["paired_metas"]["prediction_time_unit"] == "seconds"
    assert output["paired_metas"]["phystime_support_provenance"] == "original_feature_ownership_cells"
    assert output["phystime_sampling_uses_gt"] is False
    assert output["paired_metas"]["phystime_sampling_uses_gt"] is False


def test_collect_and_dataloader_collate_stack_paired_inputs_and_masks():
    collect_transform = Collect(
        inputs="feats",
        keys=["masks", "gt_segments", "gt_labels"],
        paired_inputs="paired_feats",
        paired_masks="paired_masks",
        paired_metas="paired_metas",
    )
    base = {
        "feats": torch.randn(4, 3),
        "paired_feats": torch.randn(4, 3),
        "masks": torch.ones(3, dtype=torch.bool),
        "paired_masks": torch.ones(3, dtype=torch.bool),
        "gt_segments": torch.tensor([[0.5, 1.5]]),
        "gt_labels": torch.tensor([0]),
        "video_name": "paired_video",
        "paired_metas": {"prediction_time_unit": "seconds"},
    }

    sample = collect_transform(base)
    batch = collate([sample, copy.deepcopy(sample)])

    assert batch["inputs"].shape == (2, 4, 3)
    assert batch["paired_inputs"].shape == (2, 4, 3)
    assert batch["paired_masks"].shape == (2, 3)
    assert isinstance(batch["paired_metas"], list) and len(batch["paired_metas"]) == 2


def test_selected_axis_baseline_remaps_gt_and_records_inverse_map_metadata():
    transform = BuildSelectedAxisFeatureBaseline(append_timestamp_channels=False)
    results = {
        "feats": torch.randn(3, 4),
        "masks": torch.ones(3, dtype=torch.bool),
        "phystime_selected_feature_indices": [10, 12, 15],
        "phystime_window_start_feature_idx": 10,
        "phystime_window_valid_feature_count": 6,
        "gt_segments": torch.tensor([[1.0, 4.0]]),
        "fps": 4.0,
        "snippet_stride": 4,
        "offset_frames": 0,
        "duration": 20.0,
    }

    output = transform(results)

    assert torch.allclose(output["gt_segments"], torch.tensor([[0.5, 1.6666667]]), atol=1.0e-5)
    assert output["irregular_selected_positions"] == pytest.approx([0.0, 2.0, 5.0])
    assert output["irregular_selected_valid_len"] == pytest.approx(6.0)
    assert output["irregular_native_axis"] is False
    assert output["remap_gt_to_selected_axis"] is True
    assert output["gt_remapped_to_selected_axis"] is True


def test_timestamp_selected_axis_baseline_appends_four_physical_time_channels():
    transform = BuildSelectedAxisFeatureBaseline(append_timestamp_channels=True)
    results = {
        "feats": torch.randn(3, 4),
        "masks": torch.ones(3, dtype=torch.bool),
        "phystime_selected_feature_indices": [0, 2, 5],
        "phystime_window_start_feature_idx": 0,
        "phystime_window_valid_feature_count": 6,
        "gt_segments": torch.tensor([[1.0, 4.0]]),
        "fps": 4.0,
        "snippet_stride": 4,
        "offset_frames": 0,
        "duration": 20.0,
    }

    output = transform(results)

    assert output["feats"].shape == (3, 8)
    assert torch.isfinite(output["feats"][:, -4:]).all()
