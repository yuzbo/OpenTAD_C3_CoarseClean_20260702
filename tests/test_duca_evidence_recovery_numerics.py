"""Numerical and replay contracts for the corrected Evidence Recovery path."""
from __future__ import annotations

import copy
import random

import pytest

try:
    import numpy as np
    import torch
except OSError as exc:  # pragma: no cover - Windows CI without a CUDA runtime
    pytest.skip(f"torch runtime unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.bricks.bounded_interval_adapter import BoundedTubeletIntervalAdapter
from opentad.models.bricks.dense_temporal_recovery import DenseTemporalRecovery
from opentad.models.bricks.temporal_token_merge import BoundaryProtectedTemporalTokenMerge
from opentad.models.duca.evidence_recovery import (
    DucaEvidenceRecoveryModule,
    EvidenceRecoverySelector,
    largest_remainder_quota,
)
from opentad.cores.train_engine import _capture_rng_state, _restore_rng_state


def test_nan_utility_fails_before_python_topk_conversion():
    selector = EvidenceRecoverySelector(budget=4, window_size=8)
    utility = torch.tensor([[0.1, float("nan"), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    args = [torch.zeros_like(utility) for _ in range(4)]
    with pytest.raises(FloatingPointError, match="selector.utility"):
        selector.select(utility, *args, valid_mask=torch.ones(1, 8, dtype=torch.bool))


def test_inf_interval_offset_fails_before_grid_sample_or_normalization():
    adapter = BoundedTubeletIntervalAdapter(embed_dims=2, enabled=True)
    x = torch.zeros(1, 3, 2, 4, 4)
    weight = torch.ones(2, 3, 2, 2, 2)
    with pytest.raises(FloatingPointError, match="interval.z_condition"):
        adapter.forward_tubelet(x, weight, None, 2, 0, torch.tensor([[[float("inf"), 0.0, 0.0]]]))


def test_all_masked_and_short_video_are_explicitly_invalid_tail():
    recovery = DenseTemporalRecovery(embed_dims=2, target_grid_size=8, original_window_size=16)
    feats = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    centers = torch.tensor([[0.0, 4.0, 8.0, 12.0]])
    intervals = torch.stack((centers - 1.0, centers + 1.0), dim=-1)
    all_masked = recovery(feats, centers, intervals, valid_mask=torch.zeros(1, 4, dtype=torch.bool))
    assert torch.equal(all_masked, torch.zeros_like(all_masked))

    selector = EvidenceRecoverySelector(budget=8, window_size=16)
    values = torch.rand(1, 16)
    out = selector.select(values, values, values, values, values, torch.tensor([[1, 1, 1] + [0] * 13], dtype=torch.bool))
    assert out["selected_valid_counts"].tolist() == [3]
    assert out["selected_positions"][0, 3:].tolist() == [2, 2, 2, 2, 2]


def test_merge_keeps_feature_and_timestamp_order_synchronized():
    merge = BoundaryProtectedTemporalTokenMerge(enabled=True, protected_boundary_tubelets=0)
    x = torch.arange(3 * 2, dtype=torch.float32).view(1, 3, 2)
    mass = torch.ones(1, 3)
    centers = torch.tensor([[2.0, 0.0, 1.0]])
    intervals = torch.stack((centers - 0.1, centers + 0.1), dim=-1)
    merged, new_mass, new_centers, new_intervals, _ = merge.merge_step(x, 1, mass, centers, intervals)
    assert torch.all(new_centers[0, 1:] >= new_centers[0, :-1])
    assert torch.all(new_intervals[..., 1] >= new_intervals[..., 0])
    assert torch.all(new_intervals[..., 0][0, 1:] >= new_intervals[..., 0][0, :-1])
    assert merged.shape[1] == new_mass.shape[1]


def test_recovery_tail_is_strictly_zero_after_masking():
    recovery = DenseTemporalRecovery(embed_dims=2, target_grid_size=8, original_window_size=16)
    feats = torch.ones(1, 2, 4)
    centers = torch.tensor([[0.0, 4.0, 8.0, 12.0]])
    intervals = torch.stack((centers - 1.0, centers + 1.0), dim=-1)
    out = recovery(
        feats,
        centers,
        intervals,
        valid_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        dense_valid_len=torch.tensor([8]),
    )
    assert torch.equal(out[..., 4:], torch.zeros_like(out[..., 4:]))


def test_largest_remainder_rejects_nonfinite_before_small_budget_branch():
    with pytest.raises(FloatingPointError, match="segment weight"):
        largest_remainder_quota([float("nan"), 1.0], total_budget=2, min_per_seg=2)


def test_gradient_replay_restores_all_rng_streams():
    random.seed(8261)
    np.random.seed(8261)
    torch.manual_seed(8261)
    snapshot = _capture_rng_state()
    first = (random.random(), float(np.random.rand()), torch.rand(3))
    _restore_rng_state(snapshot)
    second = (random.random(), float(np.random.rand()), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_checkpoint_resume_next_update_is_identical():
    torch.manual_seed(11)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    x = torch.randn(2, 3)
    target = torch.randn(2, 2)
    model.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(x), target).backward()
    optimizer.step()
    expected = copy.deepcopy(model.state_dict())

    model.load_state_dict(state[0])
    optimizer.load_state_dict(state[1])
    model.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(x), target).backward()
    optimizer.step()
    for key, value in expected.items():
        assert torch.equal(value, model.state_dict()[key])


def test_scalar_optimizer_update_loop_remains_finite_for_1000_steps():
    """A small optimizer sanity check; this is not a full model gate."""
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    for _ in range(1000):
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter - 1.0).square()
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(parameter.grad).all()
        optimizer.step()
        assert torch.isfinite(parameter).all()
    assert abs(float(parameter) - 1.0) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_f_and_a6_real_shape_cuda_one_step():
    device = torch.device("cuda")
    module = DucaEvidenceRecoveryModule(budget=384, window_size=768).to(device)
    lowres = torch.randn(1, 3, 768, 64, 64, device=device)
    masks = torch.ones(1, 768, dtype=torch.bool, device=device)
    with torch.no_grad():
        full = module.acquire(lowres, valid_mask=masks)
        assert full["selection"]["selected_positions"].shape == (1, 384)
        h65 = module.__class__(budget=384, window_size=768, use_h65_selection=True).to(device)
        positions = torch.linspace(0, 767, 384, device=device).round().long().view(1, -1)
        replay = h65.acquire(lowres, valid_mask=masks, h65_positions=positions)
        assert replay["selection"]["selected_positions"].shape == (1, 384)
