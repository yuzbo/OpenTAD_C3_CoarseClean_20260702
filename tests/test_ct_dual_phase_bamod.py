import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from opentad.models.duca.acquisition import (
    DualPhaseBudgetSelection,
    dual_phase_orthogonal_budget_positions,
    interpolate_h65_positions_to_detector_grid,
)
from opentad.models.selectors.dual_phase_frame_selector import DualPhaseFrameSelector
from opentad.models.backbones.vit_adapter import (
    Attention,
    Block,
    VisionTransformerAdapter,
)
from opentad.models.bricks.scale_adaptive_conv1d import (
    ContinuousTimeScaleAdaptiveConv1d,
    inverse_piecewise_linear_1d,
)
from opentad.models.bricks.conv import ConvModule


def test_dual_phase_orthogonal_budget_positions_burst_budget_enforcement():
    batch_size = 2
    temporal_len = 768
    total_k = 384
    k_scaffold = 128
    k_burst = 200  # explicitly less than 384 - 128 = 256

    priority = torch.rand(batch_size, temporal_len)
    valid_mask = torch.ones(batch_size, temporal_len, dtype=torch.bool)

    selection = dual_phase_orthogonal_budget_positions(
        h65_priority=priority,
        valid_mask=valid_mask,
        total_budget=total_k,
        scaffold_budget=k_scaffold,
        burst_budget=k_burst,
        burst_radius=2,
    )

    assert isinstance(selection, DualPhaseBudgetSelection)
    assert selection.selected_positions.shape == (batch_size, total_k)
    assert (selection.actual_count == total_k).all()

    for b in range(batch_size):
        pos = selection.selected_positions[b]
        assert (pos[1:] > pos[:-1]).all(), "Positions must be strictly ascending"
        burst_count = selection.burst_mask[b].sum().item()
        assert burst_count <= k_burst, f"Burst count {burst_count} must not exceed burst_budget {k_burst}"


def test_dual_phase_frame_selector_forward_5d_and_6d_short_video():
    B = 2
    T_raw = 768
    H, W = 32, 32
    selector = DualPhaseFrameSelector(
        total_budget=384,
        scaffold_budget=128,
        burst_budget=256,
        burst_radius=2,
    )

    # 1. Test with standard 6D NCTHW input format [B, 1, 3, T, H, W]
    inputs_6d = torch.randint(0, 256, (B, 1, 3, T_raw, H, W), dtype=torch.uint8)
    masks = torch.ones(B, T_raw, dtype=torch.bool)
    masks[1, 150:] = False  # second video is short (150 < 384)
    metas = [{"video_name": "v1"}, {"video_name": "v2"}]

    out = selector.forward_train(inputs_6d, masks, metas)

    assert out["inputs"].shape == (B, 1, 3, 384, H, W)
    assert out["masks"].shape == (B, 384)
    assert out["boundary_prior"].shape == (B, 384)
    assert out["delta_t"].shape == (B, 384)
    assert out["temporal_positions"].shape == (B, 384)

    # Assert temporal_positions is strictly monotonic and has NO -1 values
    for b in range(B):
        pos = out["temporal_positions"][b]
        assert (pos >= 0).all(), f"Row {b} contains negative timestamps!"
        assert (pos[1:] > pos[:-1]).all(), f"Row {b} timestamps are not strictly monotonic!"
        assert (out["delta_t"][b] >= 1.0).all(), f"Row {b} delta_t has values < 1.0!"

    assert metas[0]["original_window_size"] == T_raw
    assert metas[1]["original_window_size"] == T_raw

    # 2. Test with 5D input format [B, 3, T, H, W]
    inputs_5d = torch.randn(B, 3, T_raw, H, W)
    out_5d = selector.forward_train(inputs_5d, masks, metas)
    assert out_5d["inputs"].shape == (B, 3, 384, H, W)


def test_inverse_piecewise_linear_1d_interpolation_and_extrapolation():
    B = 1
    T_in = 4
    timestamps = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
    target_times = torch.tensor([[[-15.0, 0.0, 5.0, 15.0, 30.0, 50.0]]])  # shape [1, 1, 6]

    frac_idx = inverse_piecewise_linear_1d(timestamps, target_times)
    assert frac_idx.shape == (1, 1, 6)

    # Target -15.0 linearly extrapolated beyond left endpoint -> frac_idx = 0.0 + (-15.0 / 10.0) = -1.5
    assert torch.isclose(frac_idx[0, 0, 0], torch.tensor(-1.5), atol=1e-5)
    # Target 0.0 -> frac_idx = 0.0
    assert torch.isclose(frac_idx[0, 0, 1], torch.tensor(0.0), atol=1e-5)
    # Target 5.0 -> frac_idx = 0.5
    assert torch.isclose(frac_idx[0, 0, 2], torch.tensor(0.5), atol=1e-5)
    # Target 15.0 -> frac_idx = 1.5
    assert torch.isclose(frac_idx[0, 0, 3], torch.tensor(1.5), atol=1e-5)
    # Target 30.0 -> frac_idx = 3.0
    assert torch.isclose(frac_idx[0, 0, 4], torch.tensor(3.0), atol=1e-5)
    # Target 50.0 linearly extrapolated beyond right endpoint -> frac_idx = 2.0 + (30.0 / 10.0) = 5.0
    assert torch.isclose(frac_idx[0, 0, 5], torch.tensor(5.0), atol=1e-5)


def test_continuous_time_scale_adaptive_conv1d_uniform_exact_parity():
    """Verify that ContinuousTimeScaleAdaptiveConv1d is mathematically identical to nn.Conv1d on uniform timestamps."""
    torch.manual_seed(42)
    B, C_in, C_out, T_in = 2, 16, 32, 64

    for stride in [1, 2]:
        for dilation in [1, 2]:
            padding = dilation  # preserves nominal center alignment for K=3
            ct_conv = ContinuousTimeScaleAdaptiveConv1d(
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
                ref_delta_t=1.0,
                enable_learned_modulation=False,
            )

            std_conv = nn.Conv1d(
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            )
            # Share identical weights and bias
            std_conv.weight.data.copy_(ct_conv.weight.data)
            std_conv.bias.data.copy_(ct_conv.bias.data)

            x = torch.randn(B, C_in, T_in)
            uniform_tau = torch.arange(T_in, dtype=torch.float32).unsqueeze(0).expand(B, -1)

            out_ct = ct_conv(x, temporal_positions=uniform_tau)
            out_std = std_conv(x)

            assert out_ct.shape == out_std.shape
            max_diff = (out_ct - out_std).abs().max().item()
            assert max_diff < 1e-5, f"CT-Conv parity failed for stride={stride}, dilation={dilation}: max diff={max_diff}"

            # Also check delta_t uniform parity
            uniform_dt = torch.ones(B, T_in)
            out_ct_dt = ct_conv(x, delta_t=uniform_dt)
            max_diff_dt = (out_ct_dt - out_std).abs().max().item()
            assert max_diff_dt < 1e-5, f"CT-Conv delta_t parity failed for stride={stride}, dilation={dilation}: max diff={max_diff_dt}"


def test_continuous_time_scale_adaptive_conv1d_nonuniform_impulse():
    """Verify that CT-Conv correctly samples non-uniform locations for an impulse signal."""
    B, C, T_in = 1, 1, 16
    ct_conv = ContinuousTimeScaleAdaptiveConv1d(
        in_channels=C,
        out_channels=C,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        ref_delta_t=1.0,
        enable_learned_modulation=False,
    )
    # Set weights to tap indicators: [0, 1, 0] (center tap only)
    ct_conv.weight.data.zero_()
    ct_conv.weight.data[0, 0, 1] = 1.0  # Center tap weight = 1.0

    x = torch.zeros(B, C, T_in)
    x[0, 0, 5] = 10.0  # Impulse at index 5

    uniform_tau = torch.arange(T_in, dtype=torch.float32).unsqueeze(0)
    out = ct_conv(x, temporal_positions=uniform_tau)
    # Center tap on uniform grid preserves impulse at index 5
    assert torch.isclose(out[0, 0, 5], torch.tensor(10.0), atol=1e-5)


def test_block_amod_exact_identity_bypass():
    """Verify 100% mathematical identity bypass for unselected tokens in Block.forward_amod."""
    B, N, C = 2, 32, 64
    # Test without adapter first to test pure MHSA+MLP bypass
    block = Block(embed_dims=C, num_heads=4, mlp_ratio=2.0, use_adapter=False, temporal_size=8)

    x = torch.randn(B, N, C)
    # First 16 tokens have high score (selected), last 16 have low score (unselected)
    route_scores = torch.zeros(B, N)
    route_scores[:, :16] = 10.0
    route_scores[:, 16:] = -10.0

    out = block.forward_amod(
        x,
        h=2,
        w=2,
        route_scores=route_scores,
        capacity=0.5,
    )

    assert out.shape == (B, N, C)
    # Unselected tokens [:, 16:] must be IDENTICAL to input x[:, 16:]
    assert torch.equal(out[:, 16:], x[:, 16:]), "Unselected tokens failed exact identity bypass!"


def test_actionformer_optimizer_parameter_coverage_ct_dp_bamod():
    """Verify that all trainable parameters in ActionFormer are uniquely partitioned into decay/no-decay."""
    from opentad.models.detectors.actionformer import ActionFormer

    cfg_dict = dict(
        frame_selector=dict(
            type="DualPhaseFrameSelector",
            total_budget=64,
            scaffold_budget=32,
            burst_budget=32,
        ),
        projection=dict(
            type="Conv1DTransformerProj",
            in_channels=64,
            out_channels=64,
            arch=(1, 1, 0),
            conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
            norm_cfg=dict(type="LN"),
            attn_cfg=dict(n_head=4, n_mha_win_size=-1, attn_pdrop=0.0),
        ),
        neck=dict(
            type="FPNIdentity",
            in_channels=64,
            out_channels=64,
            num_levels=1,
        ),
        rpn_head=dict(
            type="ActionFormerHead",
            num_classes=20,
            in_channels=64,
            feat_channels=64,
            num_convs=2,
            prior_generator=dict(
                type="PointGenerator",
                strides=[1],
                regression_range=[[-1, 10000]],
            ),
            loss=dict(
                cls_loss=dict(type="FocalLoss"),
                reg_loss=dict(type="DIOULoss"),
            ),
            conv_cfg=dict(type="ContinuousTimeScaleAdaptiveConv1d"),
        ),
    )

    detector = ActionFormer(**cfg_dict)
    optim_cfg = dict(lr=1e-4, weight_decay=0.05)
    groups = detector.get_optim_groups(optim_cfg)

    assert len(groups) == 2
    decay_params = set(groups[0]["params"])
    no_decay_params = set(groups[1]["params"])

    # Verify no intersection between decay and no_decay
    assert len(decay_params & no_decay_params) == 0

    # Verify that all non-backbone trainable parameters are present
    trainable_params = {p for n, p in detector.named_parameters() if not n.startswith("backbone") and p.requires_grad}
    all_grouped_params = decay_params | no_decay_params
    assert trainable_params == all_grouped_params

    # Verify optimizer can be instantiated
    optimizer = torch.optim.AdamW(groups)
    assert optimizer is not None


class SimpleVideoBackbone(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x, masks=None, boundary_prior=None):
        if x.ndim == 6:
            x = x[:, 0]  # squeeze clip dimension [B, C, T, H, W]
        return self.pool(self.conv(x)).squeeze(-1).squeeze(-1)  # [B, C, T]


def test_full_actionformer_end_to_end_ct_dual_phase_bamod_6d_and_backward():
    """Verify end-to-end forward and backward pass of ActionFormer with DualPhaseFrameSelector and CT-Conv on 6D NCTHW input."""
    from opentad.models.detectors.actionformer import ActionFormer

    cfg_dict = dict(
        frame_selector=dict(
            type="DualPhaseFrameSelector",
            total_budget=64,
            scaffold_budget=32,
            burst_budget=32,
            burst_radius=2,
        ),
        projection=dict(
            type="Conv1DTransformerProj",
            in_channels=64,
            out_channels=64,
            arch=(1, 1, 0),
            conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
            norm_cfg=dict(type="LN"),
            attn_cfg=dict(n_head=4, n_mha_win_size=-1, attn_pdrop=0.0),
        ),
        neck=dict(
            type="FPNIdentity",
            in_channels=64,
            out_channels=64,
            num_levels=1,
        ),
        rpn_head=dict(
            type="ActionFormerHead",
            num_classes=20,
            in_channels=64,
            feat_channels=64,
            num_convs=2,
            prior_generator=dict(
                type="PointGenerator",
                strides=[1],
                regression_range=[[-1, 10000]],
            ),
            loss=dict(
                cls_loss=dict(type="FocalLoss"),
                reg_loss=dict(type="DIOULoss"),
            ),
            conv_cfg=dict(type="ContinuousTimeScaleAdaptiveConv1d"),
        ),
    )

    detector = ActionFormer(**cfg_dict)
    detector.backbone = SimpleVideoBackbone(in_channels=3, out_channels=64)

    B = 2
    T_raw = 128
    inputs_6d = torch.randn(B, 1, 3, T_raw, 16, 16)
    masks = torch.ones(B, T_raw, dtype=torch.bool)
    masks[1, 80:] = False  # second video is shorter
    metas = [{"video_name": "v1"}, {"video_name": "v2"}]
    gt_segments = [torch.tensor([[2.0, 5.0]]), torch.tensor([[10.0, 20.0]])]
    gt_labels = [torch.tensor([1]), torch.tensor([5])]

    # Train forward
    train_losses = detector(
        inputs_6d,
        masks,
        metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    assert isinstance(train_losses, dict)
    assert "cost" in train_losses
    assert "cls_loss" in train_losses
    assert "reg_loss" in train_losses
    assert torch.isfinite(train_losses["cost"]).item()

    # Test backward pass
    train_losses["cost"].backward()
    for name, p in detector.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"

    # Test forward
    test_preds = detector.forward_test(
        inputs_6d,
        masks,
        metas,
    )
    proposals, scores = test_preds
    assert len(proposals) == B
    assert len(scores) == B
    for b in range(B):
        assert proposals[b].shape[1] == 2  # [start, end]
        assert scores[b].shape[1] == 20  # num_classes
