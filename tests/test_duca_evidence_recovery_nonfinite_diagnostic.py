"""D0 diagnostic and fail-fast tests for DUCA Evidence Recovery."""
from __future__ import annotations

import logging

import pytest

try:
    import torch
    import torch.nn as nn
except OSError as exc:
    pytest.skip(f"torch runtime unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.bricks.dense_temporal_recovery import DenseTemporalRecovery
from opentad.models.bricks.temporal_token_merge import BoundaryProtectedTemporalTokenMerge
from opentad.models.duca.evidence_recovery import (
    DucaEvidenceRecoveryModule,
    largest_remainder_quota,
)
from opentad.cores.train_engine import train_one_epoch


def test_h65_nan_is_rejected_before_float_to_integer_cast():
    module = DucaEvidenceRecoveryModule(
        budget=4,
        window_size=8,
        use_h65_selection=True,
    )
    lowres = torch.zeros(1, 3, 8, 4, 4)
    masks = torch.ones(1, 8, dtype=torch.bool)
    positions = torch.tensor([[0.0, 2.0, float("nan"), 6.0]])
    with pytest.raises(FloatingPointError, match="h65_positions"):
        module.acquire(lowres, valid_mask=masks, h65_positions=positions)


def test_largest_remainder_rejects_nonfinite_weight_before_floor():
    with pytest.raises(FloatingPointError, match="segment weight"):
        largest_remainder_quota([1.0, float("nan")], total_budget=8)


def test_recovery_rejects_nonfinite_support_before_normalization():
    recovery = DenseTemporalRecovery(embed_dims=2, target_grid_size=4, original_window_size=8)
    feats = torch.zeros(1, 2, 2)
    centers = torch.tensor([[0.0, float("inf")]])
    intervals = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    with pytest.raises(FloatingPointError, match="recovery.centers"):
        recovery.scatter_triangular(feats, centers, intervals)


def test_token_merge_rejects_nonfinite_mass_before_division():
    merger = BoundaryProtectedTemporalTokenMerge(enabled=True)
    x = torch.zeros(1, 4, 2)
    mass = torch.tensor([[2.0, float("nan")]])
    centers = torch.tensor([[0.0, 1.0]])
    intervals = torch.tensor([[[0.0, 1.0], [1.0, 2.0]]])
    with pytest.raises(FloatingPointError, match="merge.support_mass"):
        merger.merge_step(x, 2, mass, centers, intervals)


class _NanLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, **kwargs):
        return {"cost": self.weight * torch.tensor(float("nan"))}


class _Scheduler:
    def get_last_lr(self):
        return [1.0]

    def step(self):
        return None


def test_forward_nan_fails_without_deterministic_replay():
    model = _NanLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(FloatingPointError, match="before AMP scaling"):
        train_one_epoch(
            train_loader=[{}],
            model=model,
            optimizer=optimizer,
            scheduler=_Scheduler(),
            curr_epoch=0,
            logger=logging.getLogger("duca-d0-test"),
            require_finite_loss=True,
            max_nonfinite_loss_retries=0,
        )
