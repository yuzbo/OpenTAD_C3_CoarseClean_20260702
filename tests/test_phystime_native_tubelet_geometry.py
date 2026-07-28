import copy

import numpy as np
import pytest
import torch

from opentad.datasets.transforms.formatting import Collect
from opentad.datasets.transforms.phystime_raw import (
    BuildPhysTimeNativeTubeletGeometry,
    BuildPhysTimeRawFrameGeometry,
)
from opentad.models.detectors.actionformer import ActionFormer


def make_raw_sample(*, valid_count=4, padded_count=4, with_gt=True, apply_geometry=True):
    frame_indices = np.array([100, 108, 132, 140], dtype=np.int64)[:valid_count]
    dense_indices = np.array([0, 2, 8, 10], dtype=np.float32)[:valid_count]
    if padded_count > valid_count:
        frame_indices = np.concatenate(
            [frame_indices, np.repeat(frame_indices[-1], padded_count - valid_count)]
        )
    sample = {
        "frame_inds": frame_indices,
        "selected_raw_frame_indices": frame_indices[:valid_count].copy(),
        "selected_dense_indices": dense_indices,
        "masks": torch.arange(padded_count) < valid_count,
        "snippet_stride": 4,
        "fps": 20.0,
        "avg_fps": 20.0,
        "total_frames": 400,
        "duration": 20.0,
        "irregular_dense_valid_len": 11,
        "irregular_window_crop_uses_gt": bool(with_gt),
        "irregular_subsample_uses_gt": False,
        "irregular_sampling_strategy": "random_fixed_subsample",
        "irregular_sampling_scope": "within_accepted_window",
        "irregular_native_axis": True,
        "remap_gt_to_selected_axis": False,
        "gt_remapped_to_selected_axis": False,
    }
    if with_gt:
        sample["gt_segments"] = np.array([[1.0, 6.0]], dtype=np.float32)
        sample["gt_labels"] = np.array([3], dtype=np.int32)
    if not apply_geometry:
        return sample
    return BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=with_gt)(sample)


def build_native(sample, coordinate_mode="physical_time_seconds", **kwargs):
    return BuildPhysTimeNativeTubeletGeometry(
        tubelet_size=2,
        chunk_size=4,
        transformer_depth=2,
        adapter_indices=[0, 1],
        adapter_kernel_size=3,
        adapter_dilation=1,
        coordinate_mode=coordinate_mode,
        **kwargs,
    )(sample)


def test_native_tubelet_geometry_keeps_disconnected_support_atoms_without_filling_gap():
    out = build_native(make_raw_sample())

    assert out["phystime_raw_observation_count"] == 4
    assert out["phystime_raw_valid_count"] == 4
    assert out["phystime_native_token_count"] == 2
    assert out["phystime_native_valid_count"] == 2
    assert out["phystime_native_token_dense_positions"] == pytest.approx([1.0, 9.0])
    assert out["phystime_native_token_timestamps_sec"] == pytest.approx([5.2, 6.8])
    np.testing.assert_allclose(
        out["phystime_patch_embed_support_atoms_sec"][0],
        [[5.0, 5.1], [5.3, 5.5]],
        atol=1.0e-6,
    )
    assert out["phystime_patch_embed_semantic_atom_mask"] == [[True, True], [True, True]]
    assert out["phystime_patch_embed_compute_atom_mask"] == [[True, True], [True, True]]
    assert out["phystime_patch_embed_support_envelopes_sec"][0] == pytest.approx([5.0, 5.5])
    assert out["phystime_patch_embed_lineage_provenance"] == "raw_atoms_exact_at_patch_embed_input"
    assert out["phystime_patch_embed_envelope_inflation_sec"][0] == pytest.approx(0.2, abs=1.0e-6)
    assert out["phystime_native_final_feature_support_is_exact"] is False
    assert out["phystime_native_final_feature_raw_slot_upper_bound"] == 4
    assert out["phystime_native_final_feature_lineage"] == (
        "structural_upper_bound_chunk_attention_global_temporal_adapter"
    )
    assert out["phystime_native_final_feature_raw_slot_ranges_exclusive"] == [[0, 4], [0, 4]]


def test_native_tubelet_geometry_handles_odd_valid_prefix_with_single_atom_tail():
    out = build_native(make_raw_sample(valid_count=3, padded_count=4))

    assert out["phystime_native_token_count"] == 2
    assert out["phystime_native_valid_count"] == 2
    assert out["phystime_patch_embed_semantic_atom_mask"] == [[True, True], [True, False]]
    assert out["phystime_patch_embed_compute_atom_mask"] == [[True, True], [True, True]]
    assert out["phystime_patch_embed_atom_kind"] == [
        ["observed", "observed"],
        ["observed", "padding_repeat"],
    ]
    assert out["phystime_native_token_dense_positions"] == pytest.approx([1.0, 8.0])
    assert out["phystime_patch_embed_support_atoms_sec"][1][1] == pytest.approx([6.5, 6.7])


def test_native_tubelet_geometry_audits_all_padding_repeat_slots_consumed_by_backbone():
    out = build_native(make_raw_sample(valid_count=1, padded_count=4))

    assert out["phystime_native_token_count"] == 2
    assert out["phystime_native_valid_count"] == 1
    assert out["phystime_patch_embed_semantic_atom_mask"] == [[True, False], [False, False]]
    assert out["phystime_patch_embed_compute_atom_mask"] == [[True, True], [True, True]]
    assert out["phystime_patch_embed_atom_kind"] == [
        ["observed", "padding_repeat"],
        ["padding_repeat", "padding_repeat"],
    ]
    assert out["phystime_patch_embed_padding_repeat_count"] == 3
    assert out["phystime_native_final_feature_valid_tokens_may_depend_on_padding_repeats"] is True
    assert out["phystime_native_final_feature_padding_dependency_upper_bound_mask"] == [True, True]
    assert len(out["phystime_patch_embed_support_atoms_sec"]) == 2


def test_g1a_modes_keep_gt_predictions_and_nms_coordinates_in_seconds():
    physical = build_native(make_raw_sample(), coordinate_mode="physical_time_seconds")
    selected = build_native(make_raw_sample(), coordinate_mode="uniform_rank_seconds")

    np.testing.assert_allclose(np.asarray(physical["gt_segments"]), [[5.2, 6.2]], atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(selected["gt_segments"]), [[5.2, 6.2]], atol=1.0e-6)
    assert physical["irregular_native_axis"] is True
    assert selected["irregular_native_axis"] is True
    assert physical["gt_remapped_to_selected_axis"] is False
    assert selected["gt_remapped_to_selected_axis"] is False
    assert physical["phystime_g1a_axis_positions_sec"] == pytest.approx([5.2, 6.8])
    assert selected["phystime_g1a_axis_positions_sec"] == pytest.approx([5.55, 6.65])
    assert selected["phystime_g1a_axis_start_sec"] == pytest.approx(5.0)
    assert selected["phystime_g1a_axis_end_sec"] == pytest.approx(7.2)
    assert selected["prediction_time_unit"] == "seconds"
    assert physical["prediction_time_unit"] == "seconds"
    assert len(selected["selected_dense_indices"]) == 4


def test_native_tubelet_geometry_rejects_gt_dependent_sampling_and_bad_provenance():
    gt_dependent = make_raw_sample()
    gt_dependent["phystime_subsample_uses_gt"] = True
    with pytest.raises(ValueError, match="GT-independent"):
        build_native(gt_dependent)

    unaudited = make_raw_sample()
    unaudited["phystime_support_provenance"] = "convex_hull_guess"
    with pytest.raises(ValueError, match="provenance"):
        build_native(unaudited)


def test_raw_geometry_rejects_annotation_decoder_fps_mismatch():
    sample = make_raw_sample(apply_geometry=False)
    sample["avg_fps"] = 25.0

    with pytest.raises(ValueError, match="annotation and decoder FPS"):
        BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=True)(sample)


def test_raw_geometry_accepts_audited_rounding_tolerance_and_records_timebase_error():
    sample = make_raw_sample(apply_geometry=False)
    sample["avg_fps"] = 20.1
    out = BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=True)(sample)

    assert out["phystime_canonical_fps"] == pytest.approx(20.0)
    assert out["phystime_decoder_avg_fps"] == pytest.approx(20.1)
    assert out["phystime_fps_relative_error"] == pytest.approx(0.1 / 20.1)
    assert out["phystime_duration_relative_error"] == pytest.approx(
        (20.0 - 400.0 / 20.1) / 20.0
    )


def test_raw_geometry_rejects_decoder_duration_or_frame_count_mismatch():
    sample = make_raw_sample(apply_geometry=False)
    sample["total_frames"] = 350

    with pytest.raises(ValueError, match="duration|frame count"):
        BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=True)(sample)


def test_raw_geometry_domain_matches_end_exclusive_gt_coordinate_contract():
    out = make_raw_sample()

    assert out["phystime_domain_start_sec"] == pytest.approx(5.0)
    assert out["phystime_domain_end_sec"] == pytest.approx(7.2)
    assert max(float(value) for segment in np.asarray(out["gt_segments"]) for value in segment) <= 7.2


@pytest.mark.parametrize("missing_key", ["selected_raw_frame_indices", "irregular_subsample_uses_gt"])
def test_raw_geometry_fails_closed_when_sampling_provenance_is_missing(missing_key):
    sample = make_raw_sample(with_gt=False, apply_geometry=False)
    sample.pop(missing_key)

    with pytest.raises(ValueError, match="audit|provenance|GT-independent"):
        BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=False)(sample)


def test_native_tubelet_geometry_rejects_invalid_chunk_contract():
    with pytest.raises(ValueError, match="chunk_size"):
        BuildPhysTimeNativeTubeletGeometry(
            tubelet_size=3,
            chunk_size=4,
            coordinate_mode="physical_time_seconds",
        )


def test_collect_preserves_native_tubelet_audit_metadata():
    out = build_native(make_raw_sample(with_gt=False))
    out["imgs"] = torch.zeros(3, 4, 2, 2)
    collected = Collect(inputs="imgs", keys=["masks"])(copy.deepcopy(out))
    meta = collected["metas"]

    assert meta["phystime_native_token_count"] == 2
    assert meta["phystime_patch_embed_semantic_atom_mask"] == [[True, True], [True, True]]
    assert meta["phystime_patch_embed_lineage_provenance"] == "raw_atoms_exact_at_patch_embed_input"


def test_actionformer_consumes_native_geometry_without_changing_detector_family():
    class StrictPaddingBackbone(torch.nn.Module):
        latest_temporal_padding_mask_summary = {"strict_isolation_verified": True}

    meta = build_native(make_raw_sample(with_gt=False))
    raw_masks = meta["masks"].unsqueeze(0)
    features = torch.ones(1, 8, 2)
    detector = ActionFormer.__new__(ActionFormer)
    torch.nn.Module.__init__(detector)
    detector.backbone = StrictPaddingBackbone()
    detector.native_temporal_geometry = ActionFormer._normalize_native_temporal_geometry(
        {
            "tubelet_size": 2,
            "expected_raw_count": 4,
            "expected_token_count": 2,
            "expected_transformer_depth": 2,
            "expected_adapter_indices": [0, 1],
            "expected_adapter_kernel_size": 3,
            "expected_adapter_dilation": 1,
        }
    )
    detector._last_native_temporal_geometry_audit = None

    aligned, native_masks, aligned_metas = detector._align_native_temporal_geometry(
        features,
        raw_masks,
        [meta],
    )

    assert type(detector) is ActionFormer
    assert torch.allclose(aligned, features)
    assert native_masks.tolist() == [[True, True]]
    assert aligned_metas[0]["phystime_native_token_count"] == 2
    audit = detector.collect_native_temporal_geometry_audit()
    assert audit["feature_interpolation"] is False
    assert audit["backbone_temporal_padding_isolation"]["strict_isolation_verified"] is True
