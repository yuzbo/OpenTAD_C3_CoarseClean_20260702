from pathlib import Path
from types import MethodType
import subprocess

import torch
from mmengine import Config

from opentad.models.backbones.vit_adapter import Block, VisionTransformerAdapter
from tools import train as train_module


ROOT = Path(__file__).resolve().parents[1]


def _tar32_config():
    return dict(
        enabled=True,
        support_tokens_per_tubelet=64,
        selected_tokens_per_tubelet=32,
        dense_block_indices=(0, 2, 4, 6, 8, 10),
        tar32_block_indices=(1, 3, 5, 7, 9, 11),
        query_chunk_size=32,
        routing_score="preceding_dense_attention_column_mean",
    )


def _tiny_backbone(*, tar32=True, adapter_index=(), with_cp=False):
    return VisionTransformerAdapter(
        img_size=160,
        patch_size=16,
        embed_dims=8,
        depth=12,
        num_heads=2,
        mlp_ratio=2.0,
        num_frames=2,
        tubelet_size=2,
        total_frames=4,
        adapter_index=list(adapter_index),
        use_mean_pooling=False,
        with_cp=with_cp,
        tar32_fkv=_tar32_config() if tar32 else None,
    )


def _strict_r1_two_tubelet_indices():
    spatial = torch.tensor(
        [row * 10 + col for row in range(8) for col in range(8)],
        dtype=torch.long,
    )
    return torch.cat((spatial, spatial + 100)).view(1, -1)


def _empty_stats():
    return {
        "ragged_attention_bucket_call_count": 0,
        "ragged_mlp_bucket_call_count": 0,
        "ragged_adapter_forward_count": 0,
        "executed_attention_tokens": 0,
        "executed_kv_tokens": 0,
        "executed_attention_pairs": 0,
        "executed_mlp_tokens": 0,
        "executed_adapter_tokens": 0,
    }


def test_tar32_mask_is_exact_per_tubelet_and_stable_on_ties():
    scores = torch.zeros(1, 128)
    scores[:, 40:48] = 2.0
    scores[:, 64 + 20 : 64 + 28] = 3.0
    lineage = torch.arange(2).repeat_interleave(64).view(1, -1)
    mask = VisionTransformerAdapter._tar32_fkv_mask(
        scores,
        lineage,
        total_tubelets=2,
    )
    assert mask.reshape(1, 2, 64).sum(dim=-1).tolist() == [[32, 32]]
    assert mask[0, 40:48].all()
    assert mask[0, :24].all()
    assert mask[0, 64 + 20 : 64 + 28].all()
    assert mask[0, 64 : 64 + 24].all()


def test_dense_score_then_tar32_short_query_uses_full_k64_context():
    torch.manual_seed(41)
    dense = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    sparse = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    inputs = torch.randn(1, 128, 8, requires_grad=True)
    lineage = torch.arange(2).repeat_interleave(64).view(1, -1)
    spatial = torch.arange(64).repeat(2).view(1, -1)
    buckets = [torch.arange(128).view(2, 64)]
    stats = _empty_stats()
    dense_output, scores = dense.forward_native_ragged_with_column_score(
        inputs,
        bucket_positions=buckets,
        tubelet_indices=lineage,
        spatial_indices=spatial,
        total_tubelets=2,
        grid_height=8,
        grid_width=8,
        query_chunk_size=17,
        packed_stats=stats,
    )
    assert scores.shape == (1, 128)
    assert torch.allclose(scores.reshape(1, 2, 64).sum(-1), torch.ones(1, 2))
    assert stats["executed_attention_tokens"] == 128
    assert stats["executed_kv_tokens"] == 128
    assert stats["executed_attention_pairs"] == 2 * 64 * 64

    refresh = VisionTransformerAdapter._tar32_fkv_mask(
        scores,
        lineage,
        total_tubelets=2,
    )
    stats = _empty_stats()
    output = sparse.forward_native_ragged(
        dense_output,
        bucket_positions=buckets,
        tubelet_indices=lineage,
        spatial_indices=spatial,
        total_tubelets=2,
        grid_height=8,
        grid_width=8,
        packed_stats=stats,
        refresh_mask=refresh,
        refresh_mode="mod32_kv",
    )
    assert torch.equal(output[~refresh], dense_output[~refresh])
    assert stats["executed_attention_tokens"] == 64
    assert stats["executed_kv_tokens"] == 128
    assert stats["executed_attention_pairs"] == 2 * 32 * 64
    assert stats["executed_mlp_tokens"] == 64
    gradient = torch.autograd.grad(output[refresh].sum(), dense_output)[0]
    assert gradient[~refresh].abs().sum().item() > 0


def test_tar32_end_to_end_alternates_immediate_scores_and_keeps_full_adapter():
    torch.manual_seed(43)
    model = _tiny_backbone(tar32=True, adapter_index=range(12)).eval()
    dense = _tiny_backbone(tar32=False, adapter_index=range(12)).eval()
    assert [name for name, _ in model.named_parameters()] == [
        name for name, _ in dense.named_parameters()
    ]

    observed_masks = []
    expected_masks = []
    for dense_index, sparse_index in zip(range(0, 12, 2), range(1, 12, 2)):
        dense_block = model.blocks[dense_index]
        sparse_block = model.blocks[sparse_index]
        dense_forward = dense_block.forward_native_ragged_with_column_score
        sparse_forward = sparse_block.forward_native_ragged

        def dense_with_marker(
            self,
            x,
            *,
            original=dense_forward,
            pair_index=dense_index // 2,
            **kwargs,
        ):
            output, _ = original(x, **kwargs)
            base = torch.arange(64, device=x.device, dtype=torch.float32)
            score = torch.roll(base, shifts=pair_index).repeat(
                int(x.shape[0]), 2
            )
            return output, score

        def sparse_with_observer(
            self,
            x,
            *,
            refresh_mask,
            original=sparse_forward,
            pair_index=dense_index // 2,
            **kwargs,
        ):
            base = torch.arange(64, device=x.device, dtype=torch.float32)
            score = torch.roll(base, shifts=pair_index).repeat(
                int(x.shape[0]), 2
            )
            lineage = torch.arange(2, device=x.device).repeat_interleave(64)
            expected = VisionTransformerAdapter._tar32_fkv_mask(
                score,
                lineage.view(1, -1).expand(int(x.shape[0]), -1),
                total_tubelets=2,
            )
            observed_masks.append(refresh_mask.detach().cpu())
            expected_masks.append(expected.detach().cpu())
            return original(x, refresh_mask=refresh_mask, **kwargs)

        dense_block.forward_native_ragged_with_column_score = MethodType(
            dense_with_marker,
            dense_block,
        )
        sparse_block.forward_native_ragged = MethodType(
            sparse_with_observer,
            sparse_block,
        )

    adapter_shapes = []
    handles = [
        block.adapter.register_forward_pre_hook(
            lambda _module, args: adapter_shapes.append(tuple(args[0].shape))
        )
        for block in model.blocks
    ]
    try:
        with torch.no_grad():
            output = model.forward_native_ragged(
                torch.randn(1, 128, 3, 2, 16, 16),
                _strict_r1_two_tubelet_indices(),
                total_tubelets=2,
                source_grid_hw=(10, 10),
                use_absolute_position=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (1, 128, 8)
    assert len(observed_masks) == 6
    assert all(torch.equal(actual, expected) for actual, expected in zip(observed_masks, expected_masks))
    assert adapter_shapes == [(1, 128, 8)] * 12
    summary = model.latest_tar32_fkv_summary
    assert summary["composite_execution_mode"] == "tar32_fkv"
    assert summary["dense_block_indices"] == [0, 2, 4, 6, 8, 10]
    assert summary["tar32_block_indices"] == [1, 3, 5, 7, 9, 11]
    assert summary["refresh_query_tokens_per_window"] == 64
    assert summary["kv_context_tokens_per_window"] == 128
    assert summary["executed_attention_tokens_all_blocks"] == 1152
    assert summary["executed_kv_tokens_all_blocks"] == 1536
    assert summary["executed_mlp_tokens_all_blocks"] == 1152
    assert summary["executed_adapter_tokens_all_blocks"] == 1536
    assert summary["attention_pairs_all_blocks"] == 73728
    assert summary["temporal_state_reuse"] is False
    assert summary["new_trainable_parameters"] is False


def test_tar32_checkpoint_path_backpropagates_through_every_adapter():
    torch.manual_seed(47)
    model = _tiny_backbone(
        tar32=True,
        adapter_index=range(12),
        with_cp=True,
    ).train()
    output = model.forward_native_ragged(
        torch.randn(1, 128, 3, 2, 16, 16, requires_grad=True),
        _strict_r1_two_tubelet_indices(),
        total_tubelets=2,
        source_grid_hw=(10, 10),
        use_absolute_position=False,
    )
    output.sum().backward()
    adapter_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "adapter" in name and parameter.requires_grad
    ]
    assert adapter_gradients
    assert all(gradient is not None for gradient in adapter_gradients)


def test_tar32_config_launcher_and_training_recovery_are_bound(tmp_path):
    config_path = (
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "georoute_official_r1_tar32_fkv_prebackbone_seed42_v001.py"
    )
    config = Config.fromfile(config_path)
    tar32 = config.model.backbone.backbone.tar32_fkv
    assert config.official_bc_arm == "R1-TAR32-FKV"
    assert tuple(tar32.dense_block_indices) == (0, 2, 4, 6, 8, 10)
    assert tuple(tar32.tar32_block_indices) == (1, 3, 5, 7, 9, 11)
    assert tar32.selected_tokens_per_tubelet == 32
    assert config.model.backbone.custom.zoomtoken_refresh_carry_mode == "full64"
    assert config.zoomtoken_p1_config.temporal_state_reuse is False
    assert config.zoomtoken_p1_config.new_trainable_router is False
    assert "R1-TAR32-FKV" in train_module.ZOOMTOKEN_RECOVERY_ARMS

    config.work_dir = str(tmp_path / "tar32")
    config.zoomtoken_p1_config.source_commit = "1" * 40
    recovery = train_module._zoomtoken_recovery_contract(config)
    assert recovery["arm_surface"] == "R1-TAR32-FKV"
    assert recovery["interval_epochs"] == 5
    assert recovery["keep_latest"] == 3

    launcher_path = ROOT / "scripts" / "run_zoomtoken_r1_tar32_fkv_n16r4.sh"
    launcher = launcher_path.read_text(encoding="utf-8")
    assert "PRECHECK_READY" in launcher
    assert "--nproc_per_node=2" in launcher
    assert "georoute_official_r1_tar32_fkv_prebackbone_seed42_v001.py" in launcher
    syntax = subprocess.run(
        ["bash", "-n", str(launcher_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
