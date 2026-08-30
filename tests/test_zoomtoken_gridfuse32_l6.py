from pathlib import Path

import pytest
import torch
from mmengine import Config

from opentad.models.backbones.vit_adapter import Block, VisionTransformerAdapter


ROOT = Path(__file__).resolve().parents[1]


def _lineage():
    tubelets = torch.arange(8, dtype=torch.long).repeat_interleave(64).view(1, 512)
    rectangle = (
        torch.arange(8, dtype=torch.long).view(8, 1) * 10
        + torch.arange(8, dtype=torch.long).view(1, 8)
    ).reshape(-1)
    spatial = rectangle.repeat(8).view(1, 512)
    positions = [torch.arange(512, dtype=torch.long).view(1, 512)]
    return tubelets, spatial, positions


def _stats():
    return {
        "ragged_attention_bucket_call_count": 0,
        "ragged_mlp_bucket_call_count": 0,
        "executed_attention_tokens": 0,
        "executed_kv_tokens": 0,
        "executed_attention_pairs": 0,
        "executed_mlp_tokens": 0,
        "gridfuse_bucket_call_count": 0,
    }


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_gridfuse_broadcasts_one_delta_without_collapsing_native_identity(orientation):
    torch.manual_seed(42)
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    inputs = torch.randn(1, 512, 8)
    tubelets, spatial, positions = _lineage()
    stats = _stats()
    output = block.forward_native_ragged(
        inputs,
        bucket_positions=positions,
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=8,
        grid_height=10,
        grid_width=10,
        packed_stats=stats,
        gridfuse_orientation=orientation,
    )

    if orientation == "horizontal":
        input_pairs = inputs.reshape(1, 8, 8, 4, 2, 8)
        output_pairs = output.reshape(1, 8, 8, 4, 2, 8)
        before = input_pairs[..., 0, :] - input_pairs[..., 1, :]
        after = output_pairs[..., 0, :] - output_pairs[..., 1, :]
    else:
        input_pairs = inputs.reshape(1, 8, 4, 2, 8, 8)
        output_pairs = output.reshape(1, 8, 4, 2, 8, 8)
        before = input_pairs[:, :, :, 0] - input_pairs[:, :, :, 1]
        after = output_pairs[:, :, :, 0] - output_pairs[:, :, :, 1]
    assert torch.allclose(after, before, atol=1e-6, rtol=1e-5)
    assert stats["executed_attention_tokens"] == 256
    assert stats["executed_kv_tokens"] == 256
    assert stats["executed_attention_pairs"] == 256 * 256
    assert stats["executed_mlp_tokens"] == 256
    assert stats["gridfuse_bucket_call_count"] == 1


def test_gridfuse_matches_dense_block_when_each_horizontal_pair_is_identical():
    torch.manual_seed(7)
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    pair_values = torch.randn(1, 8, 8, 4, 1, 8)
    inputs = pair_values.expand(-1, -1, -1, -1, 2, -1).reshape(1, 512, 8)
    tubelets, spatial, positions = _lineage()
    dense = block.forward_native_ragged(
        inputs,
        bucket_positions=positions,
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=8,
        grid_height=10,
        grid_width=10,
    )
    fused = block.forward_native_ragged(
        inputs,
        bucket_positions=positions,
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=8,
        grid_height=10,
        grid_width=10,
        gridfuse_orientation="horizontal",
    )
    assert torch.allclose(fused, dense, atol=2e-6, rtol=2e-5)


def test_gridfuse_rejects_nonrectangular_or_incomplete_native_support():
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    inputs = torch.randn(1, 512, 8)
    tubelets, spatial, positions = _lineage()
    broken = spatial.clone()
    broken[0, 63] = 99
    with pytest.raises(ValueError, match="hole-free 8x8"):
        block.forward_native_ragged(
            inputs,
            bucket_positions=positions,
            tubelet_indices=tubelets,
            spatial_indices=broken,
            total_tubelets=8,
            grid_height=10,
            grid_width=10,
            gridfuse_orientation="horizontal",
        )


def test_gridfuse_checkpoint_backpropagates_without_double_counting():
    torch.manual_seed(9)
    block = Block(
        embed_dims=8,
        num_heads=2,
        use_adapter=False,
        with_cp=True,
    ).train()
    inputs = torch.randn(1, 512, 8, requires_grad=True)
    tubelets, spatial, positions = _lineage()
    stats = _stats()
    output = block.forward_native_ragged(
        inputs,
        bucket_positions=positions,
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=8,
        grid_height=10,
        grid_width=10,
        packed_stats=stats,
        gridfuse_orientation="vertical",
    )
    output.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum().item() > 0
    assert stats["gridfuse_bucket_call_count"] == 1
    assert stats["executed_attention_pairs"] == 256 * 256


def test_gridfuse_backbone_executes_frozen_6_dense_plus_6_fused_ledger():
    torch.manual_seed(11)
    backbone = VisionTransformerAdapter(
        img_size=160,
        patch_size=16,
        embed_dims=8,
        depth=12,
        num_heads=2,
        mlp_ratio=2.0,
        num_frames=16,
        tubelet_size=2,
        total_frames=16,
        adapter_index=list(range(12)),
        use_mean_pooling=False,
        gridfuse32_l6=dict(enabled=True),
    ).eval()
    rectangle = (
        torch.arange(8, dtype=torch.long).view(8, 1) * 10
        + torch.arange(8, dtype=torch.long).view(1, 8)
    ).reshape(-1)
    physical = torch.cat([tubelet * 100 + rectangle for tubelet in range(8)]).view(1, 512)
    selected_native = torch.randn(1, 512, 3, 2, 16, 16)
    output = backbone.forward_native_ragged(
        selected_native,
        physical,
        total_tubelets=8,
        source_grid_hw=(10, 10),
        use_absolute_position=False,
        refresh_mode="full64",
    )
    summary = backbone.latest_gridfuse32_l6_summary
    assert output.shape == (1, 512, 8)
    assert summary["gridfuse_schema_version"] == "zoomtoken_gridfuse32_l6_v001"
    assert summary["dense_block_count"] == 6
    assert summary["gridfuse_block_count"] == 6
    assert summary["executed_attention_tokens_all_blocks"] == 4608
    assert summary["executed_kv_tokens_all_blocks"] == 4608
    assert summary["executed_mlp_tokens_all_blocks"] == 4608
    assert summary["executed_adapter_tokens_all_blocks"] == 6144
    assert summary["attention_pairs_all_blocks"] == 1_966_080
    assert summary["gridfuse_bucket_call_count"] == 6
    assert summary["padded_heavy_tokens_per_window"] == 0


def test_gridfuse_adds_no_parameters_and_config_freezes_all_gates():
    kwargs = dict(
        img_size=160,
        patch_size=16,
        embed_dims=8,
        depth=12,
        num_heads=2,
        num_frames=16,
        tubelet_size=2,
        total_frames=16,
        adapter_index=list(range(12)),
    )
    dense = VisionTransformerAdapter(**kwargs)
    fused = VisionTransformerAdapter(**kwargs, gridfuse32_l6=dict(enabled=True))
    assert dense.state_dict().keys() == fused.state_dict().keys()
    assert sum(parameter.numel() for parameter in dense.parameters()) == sum(
        parameter.numel() for parameter in fused.parameters()
    )

    config = Config.fromfile(
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "georoute_official_r1_gridfuse32_l6_prebackbone_seed42_v001.py"
    )
    route = config.model.backbone.backbone.gridfuse32_l6
    assert tuple(route.dense_block_indices) == tuple(range(6))
    assert tuple(route.fused_block_indices) == tuple(range(6, 12))
    assert route.native_tokens_per_clip == 512
    assert route.merged_tokens_per_clip == 256
    assert route.completion == "broadcast_residual_delta"
    assert config.gridfuse32_l6_gates.g0.p50_speedup_min == 1.35
    assert config.gridfuse32_l6_gates.g1.seed == 42
    assert tuple(config.gridfuse32_l6_gates.g2.pass_order) == (
        "R1",
        "C",
        "C",
        "R1",
        "C",
        "R1",
        "R1",
        "C",
    )
    assert config.gridfuse32_l6_contract.new_trainable_parameters is False
    assert config.gridfuse32_l6_contract.gt_for_route_allowed is False
    assert config.gridfuse32_l6_contract.teacher_for_route_allowed is False
