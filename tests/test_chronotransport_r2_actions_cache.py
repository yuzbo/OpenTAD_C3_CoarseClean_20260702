from collections import Counter

import pytest
import torch

from opentad.models.chronotransport import ChronoAction, ChronoCacheBank, LayerGroup
from opentad.models.chronotransport.controls import (
    InvalidImplementationError,
    motion_topk_actions,
    random_exact_count_actions,
)
from opentad.models.chronotransport.scheduler import R2_NON_DENSE_NAMES, ScheduleLibrary


EXPECTED_NAMES = (
    "periodic2_transport",
    "periodic2_hold",
    "periodic4_transport",
    "periodic4_hold",
    "periodic8_transport",
    "periodic8_hold",
    "transport_only",
    "hold_only",
    "layer_only_early_recompute",
    "layer_only_early_recompute_hold",
    "layer_only_late_recompute",
    "layer_only_late_recompute_hold",
    "joint_progressive_transport",
    "joint_progressive_hold",
    "joint_reverse_transport",
    "joint_reverse_hold",
)


def test_r2_candidate_library_has_exact_names_actions_and_hash():
    groups = (LayerGroup(0, 4), LayerGroup(4, 8), LayerGroup(8, 12))
    library = ScheduleLibrary.r2(num_chunks=48, layer_groups=groups)
    assert R2_NON_DENSE_NAMES == EXPECTED_NAMES
    assert library.names == ("dense",) + EXPECTED_NAMES
    assert library.canonical_names == EXPECTED_NAMES + ("dense",)
    assert library.stacked_actions().shape == (17, 48, 3)
    assert len(library.library_sha256) == 64
    assert library.canonical_payload()["library_sha256"] == library.library_sha256
    for candidate in library.candidates:
        assert torch.equal(candidate.actions[0], torch.zeros(3, dtype=torch.long))

    assert (library.find("periodic2_transport").actions == int(ChronoAction.RECOMPUTE)).sum(0).tolist() == [24, 24, 24]
    assert (library.find("periodic4_hold").actions == int(ChronoAction.RECOMPUTE)).sum(0).tolist() == [12, 12, 12]
    assert (library.find("periodic8_transport").actions == int(ChronoAction.RECOMPUTE)).sum(0).tolist() == [6, 6, 6]
    assert (library.find("joint_progressive_transport").actions == int(ChronoAction.RECOMPUTE)).sum(0).tolist() == [6, 12, 24]
    assert (library.find("joint_reverse_hold").actions == int(ChronoAction.RECOMPUTE)).sum(0).tolist() == [24, 12, 6]


@pytest.mark.parametrize("period,expected", [(2, 24), (4, 12), (8, 6)])
def test_motion_topk_has_exact_count_and_stable_ties(period, expected):
    motion = torch.ones(2, 48, 3)
    actions = motion_topk_actions(motion, period=period)
    counts = (actions == int(ChronoAction.RECOMPUTE)).sum(1)
    assert counts.tolist() == [[expected] * 3, [expected] * 3]
    assert torch.all(actions[:, 0, :] == int(ChronoAction.RECOMPUTE))
    assert torch.all(actions[:, 1:expected, :] == int(ChronoAction.RECOMPUTE))
    assert torch.all(actions[:, expected:, :] == int(ChronoAction.HOLD))


def test_motion_topk_nonfinite_is_invalid_implementation():
    motion = torch.zeros(1, 48, 3)
    motion[0, 7, 1] = float("nan")
    with pytest.raises(InvalidImplementationError, match="non-finite"):
        motion_topk_actions(motion, period=4)


@pytest.mark.parametrize("period,expected", [(2, 24), (4, 12), (8, 6)])
def test_random_control_is_deterministic_and_exact_count(period, expected):
    a = random_exact_count_actions("vide\u0301o-window", seed=3407, num_groups=3, period=period)
    b = random_exact_count_actions("vidéo-window", seed=3407, num_groups=3, period=period)
    assert torch.equal(a, b)
    assert a.shape == (48, 3)
    assert (a == int(ChronoAction.RECOMPUTE)).sum(0).tolist() == [expected] * 3
    assert torch.all(a[0] == int(ChronoAction.RECOMPUTE))


def test_cache_tracks_actual_age_separately_from_embedding_cap_and_detaches_history():
    cache = ChronoCacheBank(1, detach_policy="always", training=True)
    cache.reset(1)
    state = torch.tensor([1.0], requires_grad=True)
    cache.commit(0, 0, ChronoAction.RECOMPUTE, state, chunk_index=0)
    assert cache.actual_age(0, 0) == 0
    assert cache.transport_embedding_age(0, 0) == 0
    assert cache.read_latest(0, 0).grad_fn is None
    for chunk in range(1, 12):
        cache.commit(0, 0, ChronoAction.HOLD, cache.read_latest(0, 0), chunk_index=chunk)
    assert cache.actual_age(0, 0) == 11
    assert cache.transport_embedding_age(0, 0) == 8
    assert cache.normalized_actual_age(0, 0) == pytest.approx(11 / 12)


def test_cache_accepts_full_window_age_but_rejects_past_hard_limit():
    cache = ChronoCacheBank(1, hard_cache_validity_age=47)
    cache.reset(1)
    cache.commit(0, 0, ChronoAction.RECOMPUTE, torch.zeros(1), chunk_index=0)
    for chunk in range(1, 48):
        cache.commit(0, 0, ChronoAction.HOLD, cache.read_latest(0, 0), chunk_index=chunk)
    assert cache.actual_age(0, 0) == 47
    with pytest.raises(RuntimeError, match="hard cache validity"):
        cache.commit(0, 0, ChronoAction.HOLD, cache.read_latest(0, 0), chunk_index=48)
