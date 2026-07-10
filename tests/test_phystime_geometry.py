import pytest

torch = pytest.importorskip("torch")

from opentad.models.utils.phystime_geometry import (
    build_physical_query_pyramid,
    clip_to_ownership_intervals,
    geometry_from_metas,
    support_overlap_mass,
    validate_physical_observations,
)


def _base_geometry():
    timestamps = torch.tensor([[0.5, 1.5, 3.5, 0.0]], dtype=torch.float32)
    supports = torch.tensor(
        [[[0.0, 1.0], [1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor([[True, True, True, False]])
    duration = torch.tensor([4.0])
    return timestamps, supports, valid_mask, duration


def test_validation_rejects_non_increasing_valid_timestamps():
    timestamps, supports, valid_mask, duration = _base_geometry()
    timestamps[0, 2] = 1.5

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_physical_observations(timestamps, supports, valid_mask, duration)


def test_validation_rejects_support_that_does_not_contain_timestamp():
    timestamps, supports, valid_mask, duration = _base_geometry()
    supports[0, 1] = torch.tensor([1.6, 1.8])

    with pytest.raises(ValueError, match="contain its timestamp"):
        validate_physical_observations(timestamps, supports, valid_mask, duration)


def test_ownership_clipping_removes_overlap_without_expanding_support():
    timestamps = torch.tensor([[1.0, 3.0]])
    supports = torch.tensor([[[0.5, 2.6], [1.4, 3.5]]])
    valid_mask = torch.tensor([[True, True]])
    duration = torch.tensor([4.0])

    owned = clip_to_ownership_intervals(timestamps, supports, valid_mask, duration)

    assert torch.allclose(owned[0, 0], torch.tensor([0.5, 2.0]))
    assert torch.allclose(owned[0, 1], torch.tensor([2.0, 3.5]))
    assert torch.all(owned[..., 0] >= supports[..., 0])
    assert torch.all(owned[..., 1] <= supports[..., 1])


def test_true_gap_has_zero_overlap_mass():
    timestamps, supports, valid_mask, duration = _base_geometry()
    owned = clip_to_ownership_intervals(timestamps, supports, valid_mask, duration)
    query_intervals = torch.tensor([[[2.0, 3.0]]])

    mass = support_overlap_mass(owned, query_intervals, valid_mask)

    assert mass.shape == (1, 1, 4)
    assert mass.sum().item() == pytest.approx(0.0)


def test_query_pyramid_is_globally_aligned_and_independent_of_observation_count():
    duration = torch.tensor([4.2, 4.2])
    domain_start = torch.tensor([0.6, 0.6])
    domain_end = torch.tensor([3.2, 3.2])

    pyramid = build_physical_query_pyramid(
        duration,
        domain_start,
        domain_end,
        base_spacing_sec=1.0,
        num_levels=2,
    )

    assert len(pyramid) == 2
    assert torch.equal(pyramid[0]["valid_mask"].sum(dim=1), torch.tensor([4, 4]))
    assert torch.equal(pyramid[1]["valid_mask"].sum(dim=1), torch.tensor([2, 2]))
    assert torch.allclose(pyramid[0]["centers_sec"][0, :4], torch.tensor([0.5, 1.5, 2.5, 3.5]))
    assert torch.allclose(pyramid[1]["centers_sec"][0, :2], torch.tensor([1.0, 3.0]))


def test_geometry_from_metas_ignores_padding_values():
    masks = torch.tensor([[True, True, False], [True, False, False]])
    metas = [
        {
            "phystime_timestamps_sec": [0.5, 1.5],
            "phystime_support_intervals_sec": [[0.0, 1.0], [1.0, 2.0]],
            "phystime_duration_sec": 3.0,
            "phystime_domain_start_sec": 0.0,
            "phystime_domain_end_sec": 2.0,
            "phystime_support_provenance": "original_feature_ownership_cells",
        },
        {
            "phystime_timestamps_sec": [1.0],
            "phystime_support_intervals_sec": [[0.75, 1.25]],
            "phystime_duration_sec": 3.0,
            "phystime_domain_start_sec": 0.5,
            "phystime_domain_end_sec": 1.5,
            "phystime_support_provenance": "original_feature_ownership_cells",
        },
    ]

    geometry = geometry_from_metas(metas, masks, dtype=torch.float32, device=masks.device)

    assert geometry["timestamps_sec"].shape == (2, 3)
    assert geometry["support_intervals_sec"].shape == (2, 3, 2)
    assert geometry["timestamps_sec"][0, 2].item() == 0.0
    assert geometry["support_intervals_sec"][1, 1:].sum().item() == 0.0
    assert torch.equal(geometry["valid_mask"], masks)


def test_geometry_from_metas_rejects_unaudited_support_provenance():
    masks = torch.tensor([[True]])
    metas = [
        {
            "phystime_timestamps_sec": [0.5],
            "phystime_support_intervals_sec": [[0.0, 1.0]],
            "phystime_duration_sec": 1.0,
            "phystime_domain_start_sec": 0.0,
            "phystime_domain_end_sec": 1.0,
            "phystime_support_provenance": "rank_adjacent_sparse_frames",
        }
    ]

    with pytest.raises(ValueError, match="support provenance"):
        geometry_from_metas(metas, masks, dtype=torch.float32, device=masks.device)
