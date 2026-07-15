import pytest
import torch

from opentad.models.dense_heads.anchor_free_head import AnchorFreeHead
from opentad.models.utils.phystime_geometry import geometry_from_metas
from opentad.models.utils.native_temporal_geometry import align_native_tubelet_geometry


class _PhysicalGridHarness:
    _physical_selected_count_from_meta_with_keys = AnchorFreeHead._physical_selected_count_from_meta_with_keys
    _physical_selected_count_from_meta = AnchorFreeHead._physical_selected_count_from_meta
    _physical_positions_from_meta = AnchorFreeHead._physical_positions_from_meta
    _physical_assignment_positions_from_meta = AnchorFreeHead._physical_assignment_positions_from_meta
    _selected_axis_to_physical_axis = AnchorFreeHead._selected_axis_to_physical_axis

    physical_grid_positions_key = "phystime_g1a_axis_positions_sec"
    physical_grid_selected_count_keys = ("phystime_native_valid_count",)
    physical_grid_dense_valid_len_key = "irregular_dense_valid_len"
    physical_grid_axis_start_key = "phystime_g1a_axis_start_sec"
    physical_grid_axis_end_key = "phystime_g1a_axis_end_sec"
    physical_grid_eps = 1.0e-6
    physical_grid_required = True
    physical_grid_assignment_positions_key = "phystime_uniform_rank_timestamps_sec"
    physical_grid_assignment_count_keys = ("phystime_native_valid_count",)


def make_metas(valid_counts=(4, 3)):
    metas = []
    for valid_count in valid_counts:
        native_valid = (valid_count + 1) // 2
        metas.append(
            {
                "phystime_raw_observation_count": 4,
                "phystime_raw_valid_count": valid_count,
                "phystime_native_token_count": 2,
                "phystime_native_valid_count": native_valid,
                "phystime_native_tubelet_size": 2,
                "phystime_native_attention_chunk_token_count": 2,
                "phystime_native_transformer_depth": 2,
                "phystime_native_adapter_indices": [0, 1],
                "phystime_native_adapter_kernel_size": 3,
                "phystime_native_adapter_dilation": 1,
                "phystime_patch_embed_lineage_provenance": "raw_atoms_exact_at_patch_embed_input",
                "phystime_native_final_feature_lineage": (
                    "structural_upper_bound_chunk_attention_global_temporal_adapter"
                ),
                "phystime_native_final_feature_support_is_exact": False,
                "phystime_native_final_feature_raw_slot_ranges_exclusive": [[0, 4], [0, 4]],
                "phystime_native_final_feature_padding_dependency_upper_bound_mask": [
                    valid_count < 4,
                    valid_count < 4,
                ],
                "phystime_patch_embed_semantic_atom_mask": [
                    [slot * 2 < valid_count, slot * 2 + 1 < valid_count] for slot in range(2)
                ],
                "phystime_patch_embed_compute_atom_mask": [[True, True], [True, True]],
                "phystime_patch_embed_padding_repeat_count": 4 - valid_count,
                "phystime_timestamps_sec": [float(i) + 0.5 for i in range(valid_count)],
                "phystime_support_intervals_sec": [
                    [float(i), float(i) + 1.0] for i in range(valid_count)
                ],
                "phystime_duration_sec": 4.0,
                "phystime_domain_start_sec": 0.0,
                "phystime_domain_end_sec": 4.0,
                "phystime_native_token_timestamps_sec": [
                    float(
                        sum(float(i) + 0.5 for i in range(2 * token, min(2 * token + 2, valid_count)))
                    )
                    / float(max(min(2 * token + 2, valid_count) - 2 * token, 1))
                    for token in range(native_valid)
                ],
                "phystime_patch_embed_support_envelopes_sec": [
                    [float(2 * token), float(min(2 * token + 2, valid_count))]
                    if 2 * token < valid_count
                    else [float(valid_count - 1), float(valid_count)]
                    for token in range(2)
                ],
                "phystime_g1a_axis_positions_sec": list(range(native_valid)),
                "selected_dense_indices": list(range(valid_count)),
                "phystime_subsample_uses_gt": False,
            }
        )
    return metas


def test_align_native_tubelet_geometry_reduces_raw_mask_without_interpolating_features():
    features = torch.randn(2, 8, 2)
    raw_masks = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)

    aligned, token_masks, metas, audit = align_native_tubelet_geometry(
        features,
        raw_masks,
        make_metas(),
        tubelet_size=2,
        expected_raw_count=4,
        expected_token_count=2,
        expected_transformer_depth=2,
        expected_adapter_indices=[0, 1],
        expected_adapter_kernel_size=3,
        expected_adapter_dilation=1,
    )

    assert torch.allclose(aligned, features)
    assert token_masks.tolist() == [[True, True], [True, True]]
    assert audit["raw_observation_count"] == 4
    assert audit["native_token_count"] == 2
    assert audit["feature_interpolation"] is False
    assert audit["raw_valid_counts"] == [4, 3]
    assert audit["native_valid_counts"] == [2, 2]
    assert audit["base_candidate_tensor_count"] == 2
    assert audit["base_candidate_valid_counts"] == [2, 2]
    assert audit["candidate_mask_policy"] == "semantic_anchor_prefix"
    assert metas[0]["phystime_native_token_count"] == 2
    assert metas[0]["phystime_support_provenance"] == "native_patch_embed_input_envelopes"
    assert len(metas[0]["phystime_timestamps_sec"]) == 2


def test_align_native_tubelet_geometry_rewrites_projection_metadata_to_j_axis():
    features = torch.randn(1, 8, 2)
    raw_masks = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    _aligned, token_masks, metas, _audit = align_native_tubelet_geometry(
        features,
        raw_masks,
        make_metas((3,))[:1],
        tubelet_size=2,
        expected_raw_count=4,
        expected_token_count=2,
        expected_transformer_depth=2,
        expected_adapter_indices=[0, 1],
        expected_adapter_kernel_size=3,
        expected_adapter_dilation=1,
    )
    geometry = geometry_from_metas(metas, token_masks, dtype=torch.float32, device=torch.device("cpu"))

    assert token_masks.tolist() == [[True, True]]
    assert geometry["timestamps_sec"].shape == (1, 2)
    assert geometry["timestamps_sec"][0].tolist() == pytest.approx([1.0, 2.5])
    assert torch.allclose(
        geometry["support_intervals_sec"][0],
        torch.tensor([[0.0, 2.0], [2.0, 3.0]]),
    )
    assert metas[0]["phystime_raw_timestamps_sec"] == pytest.approx([0.5, 1.5, 2.5])


def test_align_native_tubelet_geometry_zeros_invalid_backbone_tokens_before_temporal_convolution():
    features = torch.ones(1, 8, 2)
    raw_masks = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    aligned, token_masks, _metas, audit = align_native_tubelet_geometry(
        features,
        raw_masks,
        make_metas((2,))[:1],
        tubelet_size=2,
        expected_raw_count=4,
        expected_token_count=2,
        expected_transformer_depth=2,
        expected_adapter_indices=[0, 1],
        expected_adapter_kernel_size=3,
        expected_adapter_dilation=1,
    )

    assert token_masks.tolist() == [[True, False]]
    assert torch.all(aligned[:, :, 0] == 1)
    assert torch.all(aligned[:, :, 1] == 0)
    assert audit["invalid_native_features_zeroed"] is True


def test_align_native_tubelet_geometry_rejects_k_j_or_metadata_mismatch():
    features = torch.randn(1, 8, 3)
    raw_masks = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="K/J"):
        align_native_tubelet_geometry(
            features,
            raw_masks,
            make_metas((4,))[:1],
            tubelet_size=2,
            expected_raw_count=4,
            expected_token_count=2,
        )

    bad_metas = make_metas((4,))[:1]
    bad_metas[0]["phystime_patch_embed_semantic_atom_mask"] = [[True, True]]
    with pytest.raises(ValueError, match="patch-input atom audit"):
        align_native_tubelet_geometry(
            torch.randn(1, 8, 2),
            raw_masks,
            bad_metas,
            tubelet_size=2,
            expected_raw_count=4,
            expected_token_count=2,
        )

    features = torch.randn(1, 8, 2)
    bad_metas = make_metas((4,))[:1]
    bad_metas[0]["phystime_g1a_axis_positions_sec"] = [0.0]
    with pytest.raises(ValueError, match="positions"):
        align_native_tubelet_geometry(
            features,
            raw_masks,
            bad_metas,
            tubelet_size=2,
            expected_raw_count=4,
            expected_token_count=2,
        )


def test_align_native_tubelet_geometry_rejects_non_prefix_mask_and_unaudited_support():
    features = torch.randn(1, 8, 2)
    with pytest.raises(ValueError, match="prefix"):
        align_native_tubelet_geometry(
            features,
            torch.tensor([[1, 0, 1, 0]], dtype=torch.bool),
            make_metas((2,))[:1],
            tubelet_size=2,
            expected_raw_count=4,
            expected_token_count=2,
        )

    bad_metas = make_metas((4,))[:1]
    bad_metas[0]["phystime_patch_embed_lineage_provenance"] = "filled_envelope"
    with pytest.raises(ValueError, match="provenance"):
        align_native_tubelet_geometry(
            features,
            torch.ones(1, 4, dtype=torch.bool),
            bad_metas,
            tubelet_size=2,
            expected_raw_count=4,
            expected_token_count=2,
        )


def test_seconds_grid_uses_explicit_domain_end_without_index_plus_one_expansion():
    meta = {
        "phystime_g1a_axis_positions_sec": [4.2, 4.6, 4.96],
        "phystime_native_valid_count": 3,
        "phystime_g1a_axis_start_sec": 4.0,
        "phystime_g1a_axis_end_sec": 5.0,
    }

    positions, domain_start, domain_end = _PhysicalGridHarness()._physical_positions_from_meta(
        meta, torch.device("cpu"), torch.float32
    )

    assert positions.tolist() == pytest.approx([4.2, 4.6, 4.96])
    assert domain_start == pytest.approx(4.0)
    assert domain_end == pytest.approx(5.0)


def test_seconds_grid_mapping_uses_both_domain_edges_in_local_geometry():
    harness = _PhysicalGridHarness()
    positions = torch.tensor([4.2, 4.6, 4.96])
    coords = torch.tensor([-0.5, 0.0, 0.5, 1.0, 2.0, 2.5])

    mapped = harness._selected_axis_to_physical_axis(
        coords,
        positions,
        domain_start=4.0,
        domain_end=5.0,
    )

    assert mapped.tolist() == pytest.approx([4.0, 4.2, 4.4, 4.6, 4.96, 5.0])


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"phystime_g1a_axis_end_sec": 4.7}, "outside the explicit physical domain"),
        ({"phystime_g1a_axis_start_sec": 5.0}, "domain must have positive extent"),
        ({"phystime_g1a_axis_positions_sec": [4.2, 4.2, 4.8]}, "strictly increasing"),
    ],
)
def test_seconds_grid_rejects_invalid_explicit_domain(updates, match):
    meta = {
        "phystime_g1a_axis_positions_sec": [4.2, 4.6, 4.8],
        "phystime_native_valid_count": 3,
        "phystime_g1a_axis_start_sec": 4.0,
        "phystime_g1a_axis_end_sec": 5.0,
    }
    meta.update(updates)

    with pytest.raises(ValueError, match=match):
        _PhysicalGridHarness()._physical_positions_from_meta(
            meta, torch.device("cpu"), torch.float32
        )


def test_assignment_positions_can_be_distinct_from_physical_decode_positions():
    meta = {
        "phystime_g1a_axis_positions_sec": [0.1, 1.5, 4.8],
        "phystime_uniform_rank_timestamps_sec": [0.8, 2.4, 4.0],
        "phystime_native_valid_count": 3,
        "phystime_g1a_axis_start_sec": 0.0,
        "phystime_g1a_axis_end_sec": 5.0,
    }

    positions = _PhysicalGridHarness()._physical_assignment_positions_from_meta(
        meta, torch.device("cpu"), torch.float32
    )

    assert positions.tolist() == pytest.approx([0.8, 2.4, 4.0])


def test_rank_assignment_never_creates_positive_for_physical_center_outside_gt():
    head = object.__new__(AnchorFreeHead)
    head.num_classes = 2
    head.center_sample = "radius"
    head.center_sample_radius = 1.5
    head.filter_similar_gt = True

    physical_points = [
        torch.tensor(
            [
                [0.1, 0.0, 10.0, 0.2],
                [4.8, 0.0, 10.0, 0.2],
            ],
            dtype=torch.float32,
        )
    ]
    assignment_points = [
        torch.tensor(
            [
                [1.5, 0.0, 10.0, 1.0],
                [3.5, 0.0, 10.0, 1.0],
            ],
            dtype=torch.float32,
        )
    ]
    gt_segments = [torch.tensor([[1.4, 1.6]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    gt_cls, gt_reg = head.prepare_targets(
        physical_points,
        gt_segments,
        gt_labels,
        assignment_points=assignment_points,
    )

    assert gt_cls[0].sum().item() == pytest.approx(0.0)
    assert gt_reg[0].shape == (2, 2)


def test_rank_assignment_can_restore_center_sampling_when_physical_center_is_inside_gt():
    head = object.__new__(AnchorFreeHead)
    head.num_classes = 2
    head.center_sample = "radius"
    head.center_sample_radius = 0.1
    head.filter_similar_gt = True

    physical_points = [torch.tensor([[1.52, 0.0, 10.0, 0.05]], dtype=torch.float32)]
    gt_segments = [torch.tensor([[1.4, 1.8]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]
    physical_only_cls, _physical_only_reg = head.prepare_targets(
        physical_points,
        gt_segments,
        gt_labels,
    )
    rank_assignment_cls, rank_assignment_reg = head.prepare_targets(
        physical_points,
        gt_segments,
        gt_labels,
        assignment_points=[torch.tensor([[1.6, 0.0, 10.0, 2.0]], dtype=torch.float32)],
    )

    assert physical_only_cls[0].sum().item() == pytest.approx(0.0)
    assert rank_assignment_cls[0].sum().item() == pytest.approx(1.0)
    assert rank_assignment_reg[0].reshape(-1).tolist() == pytest.approx([2.4, 5.6], rel=1e-4)
