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


def test_dual_phase_frame_selector_forward():
    B = 2
    T_raw = 768
    H, W = 32, 32
    selector = DualPhaseFrameSelector(
        total_budget=384,
        scaffold_budget=128,
        burst_budget=256,
        burst_radius=2,
    )

    inputs = torch.randn(B, 3, T_raw, H, W)
    masks = torch.ones(B, T_raw, dtype=torch.bool)
    masks[1, 500:] = False  # second video is shorter
    metas = [{"video_name": "v1"}, {"video_name": "v2"}]

    out = selector.forward_train(inputs, masks, metas)

    assert out["inputs"].shape == (B, 3, 384, H, W)
    assert out["masks"].shape == (B, 384)
    assert out["boundary_prior"].shape == (B, 384)
    assert out["delta_t"].shape == (B, 384)
    assert torch.isfinite(out["inputs"]).all()
    assert torch.isfinite(out["boundary_prior"]).all()
    assert torch.isfinite(out["delta_t"]).all()


def test_interpolate_h65_positions_to_detector_grid():
    B = 2
    K = 256
    selected_pos = torch.stack([
        torch.linspace(0, 500, K),
        torch.linspace(10, 700, K),
    ], dim=0)
    actual_count = torch.tensor([K, K], dtype=torch.long)

    grid = interpolate_h65_positions_to_detector_grid(
        selected_pos,
        actual_count,
        detector_length=384,
    )

    assert grid.shape == (B, 384)
    assert torch.isfinite(grid).all()
    assert (grid[:, 1:] >= grid[:, :-1]).all()


def test_amod_attention_column_mean_chunked():
    B, N, C = 2, 64, 64
    num_heads = 4
    attn_mod = Attention(embed_dims=C, num_heads=num_heads, qkv_bias=True)

    x = torch.randn(B, N, C)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 48:] = False  # padding in last 16 tokens

    out, col_mean = attn_mod.forward_with_column_mean(x, token_mask=mask, chunk_size=16)

    assert out.shape == (B, N, C)
    assert col_mean.shape == (B, N)
    assert torch.isfinite(col_mean).all()
    assert (col_mean[:, :48].mean() > col_mean[:, 48:].mean()).item()


def test_block_amod_with_mask_and_boundary_prior():
    B, N, C = 2, 32, 64
    block = Block(embed_dims=C, num_heads=4, mlp_ratio=2.0, use_adapter=True, temporal_size=8)

    x = torch.randn(B, N, C)
    route_scores = torch.rand(B, N)
    boundary_prior = torch.zeros(B, N)
    boundary_prior[:, :8] = 5.0

    temporal_token_mask = torch.ones(B, N, dtype=torch.bool)
    temporal_token_mask[:, 24:] = False

    capacity = 0.5
    out = block.forward_amod(
        x,
        h=2,
        w=2,
        route_scores=route_scores,
        capacity=capacity,
        boundary_prior=boundary_prior,
        boundary_prior_scale=0.5,
        temporal_token_mask=temporal_token_mask,
    )

    assert out.shape == (B, N, C)
    assert torch.isfinite(out).all()


def test_continuous_time_scale_adaptive_conv1d_physical_direction():
    B, C, T_in = 2, 16, 32
    conv = ContinuousTimeScaleAdaptiveConv1d(
        in_channels=C,
        out_channels=C,
        kernel_size=3,
        stride=1,
        padding=1,
        ref_delta_t=1.0,
        enable_learned_modulation=False,
    )

    x = torch.randn(B, C, T_in)

    # Dense sampling (delta_t = 0.5 < 1.0) -> scale = ref/delta_t = 2.0 -> taps expanded
    delta_t_dense = torch.full((B, T_in), 0.5)
    out_dense = conv(x, delta_t=delta_t_dense)
    assert out_dense.shape == (B, C, T_in)

    # Sparse sampling (delta_t = 2.0 > 1.0) -> scale = ref/delta_t = 0.5 -> taps contracted
    delta_t_sparse = torch.full((B, T_in), 2.0)
    out_sparse = conv(x, delta_t=delta_t_sparse)
    assert out_sparse.shape == (B, C, T_in)


def test_conv_module_with_scale_adaptive_conv():
    B, C, T = 2, 32, 64
    conv_mod = ConvModule(
        in_channels=C,
        out_channels=C,
        kernel_size=3,
        stride=1,
        padding=1,
        conv_cfg=dict(type="ContinuousTimeScaleAdaptiveConv1d"),
        norm_cfg=dict(type="LN"),
        act_cfg=dict(type="relu"),
    )

    x = torch.randn(B, C, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    delta_t = torch.ones(B, T)

    out, out_mask = conv_mod(x, mask=mask, delta_t=delta_t)
    assert out.shape == (B, C, T)
    assert out_mask.shape == (B, T)
    assert torch.isfinite(out).all()


def test_vit_adapter_12_layers_amod_end_to_end():
    B = 2
    C = 384
    num_frames = 16
    tubelet_size = 2
    patch_size = 16
    img_size = 32

    vit = VisionTransformerAdapter(
        img_size=img_size,
        patch_size=patch_size,
        embed_dims=C,
        depth=12,  # Full 12-layer ViT
        num_heads=6,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        total_frames=num_frames,
        return_feat_map=True,
        amod_config=dict(
            enabled=True,
            capacity=0.5,
            amod_layers=[1, 3, 5, 7, 9, 11],
            boundary_prior_scale=0.25,
        ),
    )

    x = torch.randn(B, 3, num_frames, img_size, img_size)
    temporal_mask = torch.ones(B, num_frames, dtype=torch.bool)
    temporal_mask[1, 10:] = False

    boundary_prior = torch.rand(B, num_frames // tubelet_size)

    out = vit(x, temporal_mask=temporal_mask, boundary_prior=boundary_prior)

    assert out.shape == (B, C, num_frames // tubelet_size, 2, 2)
    assert torch.isfinite(out).all()


def test_continuous_time_piecewise_linear_inverse_mapping():
    B, C, T_in = 2, 16, 32
    conv = ContinuousTimeScaleAdaptiveConv1d(
        in_channels=C,
        out_channels=C,
        kernel_size=3,
        stride=1,
        padding=1,
        ref_delta_t=1.0,
        enable_learned_modulation=True,
    )

    x = torch.randn(B, C, T_in)
    # Non-uniform timestamps with rapid burst at the beginning and sparse at the end
    tau = torch.stack([
        torch.cat([torch.linspace(0, 5, 16), torch.linspace(5, 50, 16)]),
        torch.linspace(0, 100, 32),
    ], dim=0)

    out = conv(x, temporal_positions=tau)
    assert out.shape == (B, C, T_in)
    assert torch.isfinite(out).all()


def test_anchor_free_head_with_continuous_time_scale_adaptive_conv():
    from opentad.models.dense_heads.actionformer_head import ActionFormerHead

    B, C = 2, 64
    num_classes = 20
    head = ActionFormerHead(
        num_classes=num_classes,
        in_channels=C,
        feat_channels=C,
        num_convs=2,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4],
            regression_range=[[-1, 10000], [-1, 10000], [-1, 10000]],
        ),
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        conv_cfg=dict(type="ContinuousTimeScaleAdaptiveConv1d"),
    )

    feat_list = [
        torch.randn(B, C, 32),
        torch.randn(B, C, 16),
        torch.randn(B, C, 8),
    ]
    mask_list = [
        torch.ones(B, 32, dtype=torch.bool),
        torch.ones(B, 16, dtype=torch.bool),
        torch.ones(B, 8, dtype=torch.bool),
    ]
    delta_t = torch.ones(B, 32)
    temporal_positions = torch.linspace(0, 31, 32).unsqueeze(0).expand(B, -1)

    gt_segments = [torch.tensor([[2.0, 5.0]]), torch.tensor([[10.0, 20.0]])]
    gt_labels = [torch.tensor([1]), torch.tensor([5])]

    losses = head.forward_train(
        feat_list,
        mask_list,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        delta_t=delta_t,
        temporal_positions=temporal_positions,
    )
    assert isinstance(losses, dict)
    assert "cls_loss" in losses
    assert "reg_loss" in losses

    proposals, scores = head.forward_test(
        feat_list,
        mask_list,
        delta_t=delta_t,
        temporal_positions=temporal_positions,
    )
    assert len(proposals) == B
    assert len(scores) == B


class SimpleVideoBackbone(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x, masks=None, boundary_prior=None):
        return self.pool(self.conv(x)).squeeze(-1).squeeze(-1)  # [B, C, T]


def test_full_actionformer_end_to_end_ct_dual_phase_bamod():
    from opentad.models.detectors.actionformer import ActionFormer

    cfg_dict = dict(
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

    detector = ActionFormer(
        projection=cfg_dict["projection"],
        neck=cfg_dict["neck"],
        rpn_head=cfg_dict["rpn_head"],
    )
    detector.backbone = SimpleVideoBackbone(in_channels=3, out_channels=64)

    B = 2
    T = 64
    inputs = torch.randn(B, 3, T, 16, 16)
    masks = torch.ones(B, T, dtype=torch.bool)
    metas = [{"video_name": "v1"}, {"video_name": "v2"}]
    gt_segments = [torch.tensor([[2.0, 5.0]]), torch.tensor([[10.0, 20.0]])]
    gt_labels = [torch.tensor([1]), torch.tensor([5])]

    # Train forward
    train_losses = detector(
        inputs,
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

    # Test forward
    test_preds = detector.forward_test(
        inputs,
        masks,
        metas,
    )
    proposals, scores = test_preds
    assert len(proposals) == B
    assert len(scores) == B


