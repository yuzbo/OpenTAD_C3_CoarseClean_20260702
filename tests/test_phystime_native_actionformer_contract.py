import pytest
import torch

from opentad.models.dense_heads.anchor_free_head import AnchorFreeHead
from opentad.models.utils.native_temporal_geometry import align_native_tubelet_geometry


class _PhysicalGridHarness:
    _physical_selected_count_from_meta = AnchorFreeHead._physical_selected_count_from_meta
    _physical_positions_from_meta = AnchorFreeHead._physical_positions_from_meta
    _selected_axis_to_physical_axis = AnchorFreeHead._selected_axis_to_physical_axis

    physical_grid_positions_key = "phystime_g1a_axis_positions_sec"
    physical_grid_selected_count_keys = ("phystime_native_valid_count",)
    physical_grid_dense_valid_len_key = "irregular_dense_valid_len"
    physical_grid_axis_start_key = "phystime_g1a_axis_start_sec"
    physical_grid_axis_end_key = "phystime_g1a_axis_end_sec"
    physical_grid_eps = 1.0e-6
    physical_grid_required = True


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
