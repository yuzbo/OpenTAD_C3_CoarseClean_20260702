"""Unit tests for DUCA Evidence Recovery architecture, invariants, and mathematical parity."""
from __future__ import annotations

import math
import pytest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except OSError as exc:
    pytest.skip(f"torch runtime unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.bricks.bounded_interval_adapter import (
    BoundedTubeletIntervalAdapter,
    ContinuousTimestampConditioner,
)
from opentad.models.bricks.temporal_token_merge import BoundaryProtectedTemporalTokenMerge
from opentad.models.bricks.dense_temporal_recovery import DenseTemporalRecovery
from opentad.models.duca.evidence_recovery import (
    ASFormerDenseSemanticScout,
    EvidenceRecoverySelector,
    DucaEvidenceRecoveryModule,
    DucaEvidenceRecoveryFrameSelector,
    compute_distillation_loss,
    compute_two_view_consistency_loss,
    largest_remainder_quota,
    partition_semantic_segments,
)
from opentad.models.duca.structured_selection import exact_uniform_positions
from opentad.models.utils.truetime_geometry import SELECTED_AXIS, TRUE_TIME_AXIS, TrueTimeMap


def test_bounded_interval_adapter_parity():
    """Verify BoundedTubeletIntervalAdapter at g=1 is mathematically identical to standard 3D Conv."""
    adapter = BoundedTubeletIntervalAdapter(
        embed_dims=16,
        in_channels=3,
        patch_size=4,
        tubelet_size=2,
        enabled=True,
    )
    B, C_in, T_frames, H, W = 2, 3, 16, 16, 16
    x = torch.randn(B, C_in, T_frames, H, W)
    weight_3d = torch.randn(16, 3, 2, 4, 4)
    bias_3d = torch.randn(16)

    # Standard Conv3D
    std_out = F.conv3d(x, weight_3d, bias=bias_3d, stride=(2, 4, 4), padding=(0, 0, 0))

    # Adapter with default zero-initialized MLP -> g(z) == 1.0
    z = torch.randn(B, 8, 3)
    adapter_out = adapter.forward_tubelet(
        x, weight_3d, bias_3d, stride_spatial=4, padding_spatial=0, z_condition=z
    )

    diff = torch.max(torch.abs(std_out - adapter_out)).item()
    assert diff < 1e-5, f"Adapter at initialization must match Conv3d, max diff={diff}"


def test_semantic_scout_context_novelty_preserves_temporal_shape():
    scout = ASFormerDenseSemanticScout(in_channels=3, hidden_dim=16, num_layers=1, window_size=32)
    rgb = torch.randn(2, 3, 32, 16, 16)
    out = scout(rgb, valid_mask=torch.ones(2, 32, dtype=torch.bool))
    assert out["context_novelty"].shape == (2, 32)
    assert out["utility"].shape == (2, 32)


def test_bounded_interval_adapter_range():
    """Verify g(z) is strictly bounded in [0.5, 1.5]."""
    adapter = BoundedTubeletIntervalAdapter(enabled=True)
    # Feed extreme z values
    extreme_z = torch.tensor([[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]])
    g = adapter.compute_g(extreme_z)
    assert (g >= 0.5).all(), "g(z) must be >= 0.5"
    assert (g <= 1.5).all(), "g(z) must be <= 1.5"


def test_bounded_interval_adapter_rejects_non_two_frame_decomposition():
    adapter = BoundedTubeletIntervalAdapter(tubelet_size=4, enabled=True)
    x = torch.randn(1, 3, 8, 8, 8)
    weight = torch.randn(8, 3, 4, 2, 2)
    z = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="tubelet_size=2"):
        adapter.forward_tubelet(x, weight, None, stride_spatial=2, padding_spatial=0, z_condition=z)


def test_continuous_timestamp_conditioner_zero_init():
    """Verify ContinuousTimestampConditioner output is strictly zero at initialization."""
    conditioner = ContinuousTimestampConditioner(num_heads=6, enabled=True)
    timestamps = torch.linspace(0, 1, 8).unsqueeze(0)  # [1, 8]
    bias = conditioner(timestamps, spatial_tokens_per_tubelet=16)
    assert bias is not None
    assert torch.max(torch.abs(bias)).item() == 0.0, "Initial timestamp attention bias must be exactly zero"


def test_continuous_timestamp_conditioner_is_finite_and_bounded_after_growth():
    """Non-uniform replay timestamps must not produce an unbounded attention mask."""
    conditioner = ContinuousTimestampConditioner(num_heads=6, enabled=True)
    with torch.no_grad():
        conditioner.bias_mlp[-1].weight.fill_(1000.0)
        conditioner.bias_mlp[-1].bias.fill_(1000.0)
    timestamps = torch.tensor([[0.0, 1.0, 0.125, 0.875]])
    bias = conditioner(timestamps, spatial_tokens_per_tubelet=4)
    assert torch.isfinite(bias).all()
    assert torch.max(torch.abs(bias)).item() <= 4.0


def test_temporal_token_merge_schedule_and_mass_conservation():
    """Verify BoundaryProtectedTemporalTokenMerge reduces tokens exactly 8 -> 7 -> 6 -> 5 and preserves mass."""
    merger = BoundaryProtectedTemporalTokenMerge(enabled=True, protected_boundary_tubelets=2)
    assert merger.should_merge(2)  # Block 3 (index 2)
    assert merger.should_merge(5)  # Block 6 (index 5)
    assert merger.should_merge(8)  # Block 9 (index 8)
    assert not merger.should_merge(0)
    assert not merger.should_merge(1)

    B = 2
    S = 16  # spatial tokens
    T = 8
    C = 32
    x = torch.randn(B, T * S, C)
    mass = torch.ones(B, T) * 2.0
    centers = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    intervals = torch.stack([centers - 0.05, centers + 0.05], dim=-1)
    b_scores = torch.rand(B, T)

    # Initial total mass per sample
    initial_total_mass = mass.sum(dim=-1)

    # Merge step 1: 8 -> 7
    x1, m1, c1, int1, b1 = merger.merge_step(x, S, mass, centers, intervals, b_scores)
    assert x1.shape == (B, 7 * S, C)
    assert torch.allclose(m1.sum(dim=-1), initial_total_mass)
    assert b1.shape == (B, 7)

    # Merge step 2: 7 -> 6
    x2, m2, c2, int2, b2 = merger.merge_step(x1, S, m1, c1, int1, b1)
    assert x2.shape == (B, 6 * S, C)
    assert torch.allclose(m2.sum(dim=-1), initial_total_mass)
    assert b2.shape == (B, 6)

    # Merge step 3: 6 -> 5
    x3, m3, c3, int3, b3 = merger.merge_step(x2, S, m2, c2, int2, b2)
    assert x3.shape == (B, 5 * S, C)
    assert torch.allclose(m3.sum(dim=-1), initial_total_mass)
    assert b3.shape == (B, 5)


def test_temporal_token_merge_no_unprotected_pair_preserves_mass():
    merger = BoundaryProtectedTemporalTokenMerge(enabled=True, protected_boundary_tubelets=8)
    x = torch.randn(1, 3, 4)
    mass = torch.tensor([[1.0, 2.0, 3.0]])
    centers = torch.tensor([[0.0, 1.0, 2.0]])
    intervals = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
    boundary_scores = torch.tensor([[0.0, 10.0, 0.0]])

    x1, mass1, centers1, intervals1, scores1 = merger.merge_step(
        x,
        spatial_tokens=1,
        support_mass=mass,
        support_centers=centers,
        support_intervals=intervals,
        boundary_scores=boundary_scores,
    )

    assert x1.shape == (1, 2, 4)
    assert torch.allclose(mass1.sum(dim=-1), mass.sum(dim=-1))
    assert centers1.shape == (1, 2)
    assert intervals1.shape == (1, 2, 2)
    assert scores1.shape == (1, 2)



def test_dense_temporal_recovery_parity():
    """Verify DenseTemporalRecovery at initialization is strictly non-parametric interpolation."""
    recovery = DenseTemporalRecovery(embed_dims=16, target_grid_size=384, enabled=True)
    B, C, N = 2, 16, 192
    feats = torch.randn(B, C, N)
    centers = torch.linspace(0, 767, N).unsqueeze(0).expand(B, -1)
    intervals = torch.stack([centers - 2.0, centers + 2.0], dim=-1)

    out = recovery(feats, centers, intervals)
    assert out.shape == (B, 16, 384)

    # Non-parametric reference
    scatter_ref = recovery.scatter_triangular(feats, centers, intervals)
    diff = torch.max(torch.abs(out - scatter_ref)).item()
    assert diff < 1e-6, "Recovery output at initialization must equal triangular scatter reference"


def test_largest_remainder_quota_exact_sum():
    """Verify largest_remainder_quota always sums to exact target budget."""
    weights = [10.5, 2.3, 8.7, 15.1, 0.4, 6.2]
    total_budget = 384
    quotas = largest_remainder_quota(weights, total_budget, min_per_seg=2)
    assert sum(quotas) == total_budget
    assert all(q >= 2 for q in quotas)


def test_evidence_recovery_selector_exact_k():
    """Verify EvidenceRecoverySelector always returns exact K=384 unique, monotonic indices."""
    selector = EvidenceRecoverySelector(budget=384, window_size=768, use_coverage=True)
    B, T = 2, 768
    utility = torch.rand(B, T)
    boundary_prob = torch.rand(B, T)
    delta_action = torch.rand(B, T)
    feat_res = torch.rand(B, T)
    novelty = torch.rand(B, T)
    valid_mask = torch.ones(B, T, dtype=torch.bool)

    out = selector.select(utility, boundary_prob, delta_action, feat_res, novelty, valid_mask)
    sel_pos = out["selected_positions"]
    assert sel_pos.shape == (B, 384)

    for b in range(B):
        row = sel_pos[b].tolist()
        assert len(set(row)) == 384, "All 384 selected positions must be unique"
        assert row == sorted(row), "Selected positions must be strictly monotonically increasing"
        assert all(0 <= p < 768 for p in row)


def test_evidence_recovery_selector_rejects_all_masked_rows():
    selector = EvidenceRecoverySelector(budget=4, window_size=8, use_coverage=True)
    values = torch.zeros(1, 8)
    with pytest.raises(ValueError, match="all-masked"):
        selector.select(values, values, values, values, values, torch.zeros(1, 8, dtype=torch.bool))


def test_evidence_recovery_selector_enforces_max_unselected_hole():
    selector = EvidenceRecoverySelector(budget=16, window_size=64, use_coverage=True, max_hole=3)
    utility = torch.zeros(1, 64)
    utility[:, :16] = 100.0
    boundary_prob = torch.zeros(1, 64)
    delta_action = torch.zeros(1, 64)
    feat_res = torch.zeros(1, 64)
    novelty = torch.zeros(1, 64)
    valid_mask = torch.ones(1, 64, dtype=torch.bool)

    out = selector.select(utility, boundary_prob, delta_action, feat_res, novelty, valid_mask)

    assert out["selected_positions"].shape == (1, 16)
    assert int(out["observed_max_unselected_hole"].max().item()) <= 3


def test_h65_replay_requires_supplied_positions():
    module = DucaEvidenceRecoveryModule(budget=8, window_size=16, use_h65_selection=True)
    lowres = torch.randn(1, 3, 16, 8, 8)
    with pytest.raises(ValueError, match="requires H65 selected positions"):
        module.acquire(lowres, valid_mask=torch.ones(1, 16, dtype=torch.bool))


def test_h65_replay_uses_supplied_positions_exactly():
    module = DucaEvidenceRecoveryModule(budget=8, window_size=16, use_h65_selection=True)
    lowres = torch.randn(1, 3, 16, 8, 8)
    positions = exact_uniform_positions(16, 8).unsqueeze(0)

    out = module.acquire(lowres, valid_mask=torch.ones(1, 16, dtype=torch.bool), h65_positions=positions)

    selection = out["selection"]
    assert out["scout"] is None
    assert torch.equal(selection["selected_positions"], positions)
    assert selection["selected_valid_counts"].tolist() == [8]


def test_h65_position_parser_pads_short_tail_without_marking_padding_valid():
    selector = DucaEvidenceRecoveryFrameSelector(budget=8, window_size=16, use_h65_selection=True)
    positions = selector._extract_h65_positions(
        [{"irregular_selected_positions": [0, 2, 4, 6]}],
        torch.tensor([8]),
        device=torch.device("cpu"),
    )

    assert positions.shape == (1, 8)
    assert positions[0, :4].tolist() == [0, 2, 4, 6]
    assert positions[0, 4:].tolist() == [8, 8, 8, 8]

    lowres = torch.randn(1, 3, 16, 8, 8)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, :8] = True
    out = selector.module.acquire(lowres, valid_mask=mask, h65_positions=positions)
    assert out["selection"]["selected_valid_counts"].tolist() == [4]


def test_no_recovery_updates_gt_from_original_time_axis_after_backbone_merge():
    selector = DucaEvidenceRecoveryFrameSelector(
        budget=8,
        window_size=16,
        use_dense_recovery=False,
        use_temporal_merge=True,
    )
    original_segments = [torch.tensor([[2.0, 10.0]])]
    stale_segments = [torch.tensor([[100.0, 101.0]])]
    centers = torch.tensor([[0.5, 2.5, 6.5, 10.5]])
    selector_outputs = {
        "metas": [{"video_name": "sample"}],
        "selected_positions": torch.arange(8).unsqueeze(0),
        "dense_valid_len": torch.tensor([16]),
        "selected_valid_counts": torch.tensor([8]),
        "support_centers": centers,
        "gt_segments": stale_segments,
        "gt_segments_original": original_segments,
    }
    masks = torch.ones(1, 4, dtype=torch.bool)
    selector._update_no_recovery_axis_after_backbone(
        selector_outputs=selector_outputs,
        masks=masks,
        backbone_support_metadata={"support_centers": centers},
    )

    expected = TrueTimeMap(centers[0], dense_len=16, valid_len=16).remap_segments(
        original_segments[0],
        source_coordinate_space=TRUE_TIME_AXIS,
        target_coordinate_space=SELECTED_AXIS,
    )
    assert torch.allclose(selector_outputs["gt_segments"][0], expected)
    assert selector_outputs["metas"][0]["gt_remapped_to_selected_axis"] is True


def test_distillation_and_consistency_losses():
    """Verify distillation and consistency loss computation."""
    B, C, T = 2, 32, 384
    s_feat = torch.randn(B, C, T)
    t_feat = torch.randn(B, C, T)
    loss_dist = compute_distillation_loss(s_feat, t_feat)
    assert loss_dist.item() >= 0.0
    assert math.isfinite(loss_dist.item())

    # Self distillation should be minimal
    loss_self = compute_distillation_loss(s_feat, s_feat)
    assert loss_self.item() < 1e-4

    logits1 = torch.randn(B, 384, 20)
    logits2 = torch.randn(B, 384, 20)
    reg1 = torch.randn(B, 384, 2)
    reg2 = torch.randn(B, 384, 2)
    loss_cons = compute_two_view_consistency_loss(logits1, logits2, reg1, reg2)
    assert loss_cons.item() >= 0.0
    assert math.isfinite(loss_cons.item())


def test_frame_selector_forward_train_and_test():
    """Verify DucaEvidenceRecoveryFrameSelector forward_train and forward_test pipelines."""
    selector = DucaEvidenceRecoveryFrameSelector(
        budget=384,
        window_size=768,
        use_coverage=True,
        use_time_conditioning=True,
        use_temporal_merge=True,
        use_dense_recovery=True,
        use_robust_training=True,
    )
    B = 2
    inputs = torch.randn(B, 1, 3, 768, 64, 64)
    masks = torch.ones(B, 768, dtype=torch.bool)
    metas = [{"video_name": f"video_{i}"} for i in range(B)]
    gt_segments = [torch.tensor([[10.0, 50.0], [100.0, 200.0]]) for _ in range(B)]
    gt_labels = [torch.tensor([1, 2]) for _ in range(B)]

    # Test forward_train
    train_out = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    assert train_out["inputs"].shape == (B, 1, 3, 384, 64, 64)
    assert train_out["masks"].shape == (B, 384)
    assert "scout_action_loss" in train_out["losses"]
    assert "scout_boundary_loss" in train_out["losses"]
    assert "scout_robust_consistency_loss" in train_out["losses"]

    no_robust = DucaEvidenceRecoveryFrameSelector(
        budget=16, window_size=32, use_dense_recovery=True, use_robust_training=False
    )
    tiny_inputs = torch.randn(1, 1, 3, 32, 16, 16)
    tiny_masks = torch.ones(1, 32, dtype=torch.bool)
    tiny_gt = [torch.tensor([[4.0, 12.0]])]
    tiny_out = no_robust.forward_train(tiny_inputs, tiny_masks, [{"video_name": "tiny"}], tiny_gt, [torch.tensor([1])])
    assert "scout_action_loss" in tiny_out["losses"]
    assert "scout_robust_consistency_loss" not in tiny_out["losses"]

    # Test forward_test
    test_out = selector.forward_test(inputs, masks, metas)
    assert test_out["inputs"].shape == (B, 1, 3, 384, 64, 64)
    assert test_out["masks"].shape == (B, 384)

    # Test recover_features
    feats = torch.randn(B, 384, 192)
    rec_feats, rec_masks = selector.recover_features(feats, train_out["masks"], train_out)
    assert rec_feats.shape == (B, 384, 384)
    assert rec_masks.shape == (B, 384)
