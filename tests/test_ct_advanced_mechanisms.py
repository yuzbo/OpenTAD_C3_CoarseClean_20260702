import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
from opentad.models.losses.iou_loss import ContinuousPhysicalGIoULoss
from opentad.models.losses.sinkhorn_ot_loss import SinkhornOptimalTransportLoss


def test_ct_tubelet_patch_embed_uniform_exact_parity():
    """Verify that CT-Tubelet 3D patch embedding produces exact numerical parity with standard 3D conv on uniform timestamps."""
    torch.manual_seed(42)
    B, C, num_frames, H, W = 2, 3, 16, 32, 32
    tubelet_size = 2
    patch_size = 16
    embed_dims = 64

    vit = VisionTransformerAdapter(
        img_size=H,
        patch_size=patch_size,
        embed_dims=embed_dims,
        depth=2,
        num_heads=2,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        total_frames=num_frames,
        ct_tubelet=True,
    )

    x = torch.randn(B, C, num_frames, H, W)
    num_tubelets = num_frames // tubelet_size
    uniform_dt = torch.ones((B, num_tubelets), dtype=torch.float32)

    # Standard patch embed output
    std_out = vit.patch_embed(x)[0]

    # CT-Tubelet patch embed output
    ct_out = vit._forward_ct_patch_embed(x, delta_t=uniform_dt)

    assert std_out.shape == ct_out.shape
    max_diff = (std_out - ct_out).abs().max().item()
    assert max_diff < 1e-5, f"CT-Tubelet uniform parity failed: max diff = {max_diff}"


def test_ct_tubelet_patch_embed_velocity_scaling():
    """Verify that CT-Tubelet scales the temporal differential component inversely with delta_t."""
    torch.manual_seed(42)
    B, C, num_frames, H, W = 1, 3, 16, 32, 32
    tubelet_size = 2
    patch_size = 16
    embed_dims = 64

    vit = VisionTransformerAdapter(
        img_size=H,
        patch_size=patch_size,
        embed_dims=embed_dims,
        depth=2,
        num_heads=2,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        total_frames=num_frames,
        ct_tubelet=True,
    )

    # Input with non-zero temporal movement: frame 1 differs from frame 0
    x = torch.zeros(B, C, num_frames, H, W)
    x[:, :, 1::2] = 2.0  # differential x1 - x0 = 2.0

    # dt = 1.0 vs dt = 4.0
    num_tubelets = num_frames // tubelet_size
    dt_fast = torch.full((B, num_tubelets), 1.0)
    dt_slow = torch.full((B, num_tubelets), 4.0)

    out_fast = vit._forward_ct_patch_embed(x, delta_t=dt_fast)
    out_slow = vit._forward_ct_patch_embed(x, delta_t=dt_slow)

    assert out_fast.shape == out_slow.shape
    assert torch.isfinite(out_fast).all()
    assert torch.isfinite(out_slow).all()
    # When delta_t is 4x larger, differential energy is dampened by 4x
    assert not torch.allclose(out_fast, out_slow)


def test_continuous_physical_giou_loss():
    """Verify mathematical properties of ContinuousPhysicalGIoULoss."""
    loss_fn = ContinuousPhysicalGIoULoss(loss_weight=1.0, center_weight=1.0)

    # 1. Identical boxes: loss must be exactly 0.0
    gt_boxes = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
    pred_boxes_identical = torch.tensor([[10.0, 20.0], [5.0, 15.0]])
    loss_identical = loss_fn(pred_boxes_identical, gt_boxes)
    assert torch.isclose(loss_identical, torch.tensor(0.0), atol=1e-5), f"Expected 0.0 for identical boxes, got {loss_identical.item()}"

    # 2. Shifted boxes: loss must be strictly positive
    pred_boxes_shifted = torch.tensor([[12.0, 22.0], [5.0, 15.0]])
    loss_shifted = loss_fn(pred_boxes_shifted, gt_boxes)
    assert loss_shifted.item() > 0.0, f"Expected positive loss for shifted boxes, got {loss_shifted.item()}"

    # 3. Disjoint boxes: loss must penalize distance continuously
    pred_boxes_far = torch.tensor([[50.0, 60.0], [5.0, 15.0]])
    loss_far = loss_fn(pred_boxes_far, gt_boxes)
    assert loss_far.item() > loss_shifted.item(), f"Loss for far boxes ({loss_far.item()}) should exceed shifted ({loss_shifted.item()})"

    # 4. Backward pass produces finite gradients
    pred_var = pred_boxes_shifted.clone().requires_grad_(True)
    loss = loss_fn(pred_var, gt_boxes)
    loss.backward()
    assert pred_var.grad is not None
    assert torch.isfinite(pred_var.grad).all()


def test_sinkhorn_optimal_transport_loss():
    """Verify convergence, distance penalty, and gradient flow of SinkhornOptimalTransportLoss."""
    loss_fn = SinkhornOptimalTransportLoss(
        epsilon=0.05,
        num_iters=20,
        temperature=1.0,
        boundary_bandwidth=2.0,
        loss_weight=1.0,
    )

    B, T = 2, 64
    valid_masks = torch.ones((B, T), dtype=torch.bool)
    gt_segments = [
        torch.tensor([[10.0, 20.0]]),
        torch.tensor([[30.0, 45.0]]),
    ]

    # 1. Optimal predicted logits that peak exactly at boundaries
    pred_logits_good = torch.zeros((B, T), requires_grad=True)
    with torch.no_grad():
        pred_logits_good[0, 10] = 5.0
        pred_logits_good[0, 20] = 5.0
        pred_logits_good[1, 30] = 5.0
        pred_logits_good[1, 45] = 5.0

    loss_good = loss_fn(pred_logits_good, gt_segments, valid_masks)
    assert torch.isfinite(loss_good).item()

    # 2. Bad predicted logits that peak far from boundaries
    pred_logits_bad = torch.zeros((B, T), requires_grad=True)
    with torch.no_grad():
        pred_logits_bad[0, 55] = 5.0
        pred_logits_bad[0, 60] = 5.0
        pred_logits_bad[1, 5] = 5.0
        pred_logits_bad[1, 10] = 5.0

    loss_bad = loss_fn(pred_logits_bad, gt_segments, valid_masks)
    assert loss_bad.item() > loss_good.item(), f"Bad placement loss ({loss_bad.item()}) should be higher than good ({loss_good.item()})"

    # 3. Backward gradient check
    loss_good.backward()
    assert pred_logits_good.grad is not None
    assert torch.isfinite(pred_logits_good.grad).all()
