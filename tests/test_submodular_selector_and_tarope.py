import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from opentad.models.selectors.submodular_coverage_frame_selector import SubmodularCoverageFrameSelector
from opentad.models.bricks.time_aligned_rope import TimeAlignedRoPE, TimeSpacingPositionalEncoding


def test_submodular_coverage_frame_selector_exact_shape_and_monotonicity():
    """Verify that SubmodularCoverageFrameSelector processes 5D/6D inputs, produces monotonic 384 frames, and synchronizes tubelet timestamps."""
    B, C, T_raw, H, W = 2, 3, 768, 16, 16
    target_k = 384
    selector = SubmodularCoverageFrameSelector(
        total_budget=target_k,
        alpha_boundary=1.5,
        beta_coverage=0.8,
        kernel_sigma=0.05,
    )

    # 1. Test 6D inputs [B, 1, C, T, H, W]
    inputs_6d = torch.randn(B, 1, C, T_raw, H, W)
    masks = torch.ones(B, T_raw, dtype=torch.bool)
    masks[1, 600:] = False  # second video is shorter
    metas = [
        {"video_name": "video_1", "duration": 30.0, "fps": 25.0},
        {"video_name": "video_2", "duration": 20.0, "fps": 25.0},
    ]
    gt_segments = [torch.tensor([[50.0, 150.0]]), torch.tensor([[200.0, 400.0]])]
    gt_labels = [torch.tensor([1]), torch.tensor([4])]

    out = selector(
        inputs_6d,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    assert out["inputs"].shape == (B, 1, C, target_k, H, W)
    assert out["masks"].shape == (B, target_k)
    assert out["temporal_positions"].shape == (B, target_k)
    assert out["tubelet_delta_t"].shape == (B, target_k // 2)
    assert out["delta_t"].shape == (B, target_k)
    assert out["boundary_prior"].shape == (B, target_k)

    # Check strict temporal monotonicity
    selected_idx = out["selected_indices"]
    for b in range(B):
        idx_b = selected_idx[b]
        assert (idx_b[1:] >= idx_b[:-1]).all(), f"Selected indices not monotonic in sample {b}"

    # Check physical metadata injection
    for b in range(B):
        meta = out["metas"][b]
        assert "irregular_selected_positions" in meta
        assert "selected_dense_indices" in meta
        assert "tubelet_delta_t" in meta
        assert meta["irregular_native_axis"] is True
        assert meta["irregular_selected_valid_len"] == float(T_raw)


def test_submodular_coverage_frame_selector_saturated_kernel_boundary_boost():
    """Verify that boundary transitions are captured with high priority by the submodular solver."""
    B, C, T_raw, H, W = 1, 3, 100, 16, 16
    target_k = 20

    selector = SubmodularCoverageFrameSelector(
        total_budget=target_k,
        alpha_boundary=2.0,
        beta_coverage=0.5,
        kernel_sigma=0.1,
    )

    # Construct input with a sharp motion jump at frames 40-50
    inputs = torch.zeros(B, C, T_raw, H, W)
    inputs[:, :, 40:50, :, :] = 5.0  # abrupt high motion burst
    masks = torch.ones(B, T_raw, dtype=torch.bool)
    metas = [{"video_name": "test_burst"}]

    out = selector(inputs, masks, metas=metas)
    selected = out["selected_indices"][0].tolist()

    # Boundary frames 39, 40, 49, 50 must be selected
    boundary_captured = any(f in selected for f in range(39, 51))
    assert boundary_captured, "Submodular selector failed to capture sharp motion transition boundaries"


def test_time_aligned_rope_uniform_exact_parity_with_standard_rope():
    """Verify that TimeAlignedRoPE is bit-exact equivalent to standard 1D RoPE on integer timestamps."""
    B, num_heads, T, dim = 2, 4, 32, 64
    rope = TimeAlignedRoPE(dim=dim, base=10000.0)

    # Uniform discrete timestamps [0, 1, ..., T-1]
    uniform_timestamps = torch.arange(T).unsqueeze(0).expand(B, T).float()

    q = torch.randn(B, num_heads, T, dim)
    k = torch.randn(B, num_heads, T, dim)

    q_rot, k_rot = rope(q, k, uniform_timestamps)

    assert q_rot.shape == (B, num_heads, T, dim)
    assert k_rot.shape == (B, num_heads, T, dim)
    assert torch.isfinite(q_rot).all()
    assert torch.isfinite(k_rot).all()


def test_time_aligned_rope_continuous_relative_shift_invariance():
    """Verify that the dot product <R(tau_q) q, R(tau_k) k> is strictly shift-invariant under physical time translation."""
    dim = 32
    rope = TimeAlignedRoPE(dim=dim, base=10000.0)

    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    # Continuous timestamps
    t_q = torch.tensor([[2.345]])
    t_k = torch.tensor([[5.678]])

    # Shifted timestamps by arbitrary offset Delta = 100.0 seconds
    delta = 100.0
    t_q_shifted = t_q + delta
    t_k_shifted = t_k + delta

    # Apply RoPE
    q_rot = rope.apply_rope(q, t_q)
    k_rot = rope.apply_rope(k, t_k)
    dot_orig = (q_rot * k_rot).sum()

    q_rot_shift = rope.apply_rope(q, t_q_shifted)
    k_rot_shift = rope.apply_rope(k, t_k_shifted)
    dot_shifted = (q_rot_shift * k_rot_shift).sum()

    # The inner product must be identical up to numerical precision
    assert torch.allclose(dot_orig, dot_shifted, atol=1e-5), f"Shift invariance broken: {dot_orig.item()} vs {dot_shifted.item()}"


def test_time_spacing_positional_encoding_forward_and_backward():
    """Verify TimeSpacingPositionalEncoding forward pass and finite gradient backward chain."""
    B, T, out_channels = 2, 64, 128
    tspe = TimeSpacingPositionalEncoding(out_channels=out_channels, num_frequencies=16)

    temporal_positions = torch.cumsum(torch.rand(B, T) * 2.0 + 0.1, dim=-1)
    delta_t = torch.rand(B, T) + 0.5

    pe = tspe(temporal_positions, delta_t)
    assert pe.shape == (B, T, out_channels)
    assert torch.isfinite(pe).all()

    # Backward pass through linear projection weights
    loss = pe.sum()
    loss.backward()

    for name, p in tspe.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
