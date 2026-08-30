from pathlib import Path
from types import MethodType

import pytest
import torch
from mmengine import Config

from opentad.models.backbones.vit_adapter import Attention, Block, VisionTransformerAdapter


ROOT = Path(__file__).resolve().parents[1]


def _spatial_k64():
    return torch.tensor(
        [row * 10 + col for row in range(8) for col in range(8)],
        dtype=torch.long,
    )


def _lineage(batch=1):
    spatial = _spatial_k64().repeat(8).view(1, 512).repeat(batch, 1)
    tubelets = torch.arange(8).repeat_interleave(64).view(1, 512).repeat(batch, 1)
    return tubelets, spatial


def _stats():
    return {
        "ragged_attention_bucket_call_count": 0,
        "ragged_mlp_bucket_call_count": 0,
        "ragged_adapter_forward_count": 0,
        "executed_attention_tokens": 0,
        "executed_kv_tokens": 0,
        "executed_attention_pairs": 0,
        "executed_mlp_tokens": 0,
        "executed_adapter_tokens": 0,
        "racer24_block_forward_count": 0,
        "racer24_clip_count": 0,
        "racer24_selected_query_tokens": 0,
    }


def _racer_config():
    return dict(
        enabled=True,
        racer_blocks=(4, 6, 8, 10),
        tubelets_per_clip=8,
        spatial_tokens_per_tubelet=64,
        selected_per_tubelet=24,
        selected_query_tokens_per_clip=192,
        full_kv_tokens_per_clip=512,
        completion="parameter_free_key_residual_entropy",
        router="preceding_dense_residual_plus_adjacent_surprise",
    )


def _tiny_backbone(*, racer24=None, with_cp=False):
    return VisionTransformerAdapter(
        img_size=160,
        patch_size=16,
        embed_dims=8,
        depth=12,
        num_heads=2,
        mlp_ratio=1.0,
        num_frames=16,
        tubelet_size=2,
        total_frames=16,
        adapter_index=list(range(12)),
        use_mean_pooling=False,
        with_cp=with_cp,
        racer24=racer24,
    )


def test_selected_query_full_kv_matches_dense_corresponding_queries():
    torch.manual_seed(42)
    attention = Attention(embed_dims=8, num_heads=2).eval()
    carrier = torch.randn(2, 17, 8)
    selected_indices = torch.tensor([0, 3, 5, 11, 16])
    selected = carrier.index_select(1, selected_indices)

    expected = attention(carrier).index_select(1, selected_indices)
    actual, full_keys = attention.forward_selected_query_full_kv(
        selected,
        carrier,
        return_full_keys=True,
    )
    assert actual.shape == (2, 5, 8)
    assert full_keys.shape == (2, 2, 17, 4)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert set(dict(attention.named_parameters())) == {
        "q_bias",
        "v_bias",
        "qkv.weight",
        "proj.weight",
        "proj.bias",
    }


def test_router_is_per_tubelet_exact_k24_stable_and_stopgrad():
    current = torch.ones(1, 8, 64, 4, requires_grad=True)
    previous = torch.ones_like(current, requires_grad=True)
    spatial = _spatial_k64().view(1, 1, 64).repeat(1, 8, 1)
    selected, score = Block._racer24_route_indices(current, previous, spatial)

    assert selected.shape == (1, 8, 24)
    assert selected.tolist() == [[list(range(24)) for _ in range(8)]]
    assert not score.requires_grad
    assert not selected.requires_grad
    assert torch.equal(selected, selected.detach())


def test_router_missing_adjacent_coordinate_receives_higher_surprise_rank():
    current = torch.ones(1, 8, 64, 4)
    previous = torch.ones_like(current)
    aligned = _spatial_k64().view(1, 1, 64).repeat(1, 8, 1)
    shifted = aligned.clone()
    shifted[:, 1] += 1

    _, aligned_score = Block._racer24_route_indices(current, previous, aligned)
    _, shifted_score = Block._racer24_route_indices(current, previous, shifted)
    assert shifted_score[0, 1, -1] > aligned_score[0, 1, -1]
    assert shifted_score[0, 0, 0] >= aligned_score[0, 0, 0]


def test_completion_is_finite_and_scatter_preserves_selected_exact_residuals():
    torch.manual_seed(7)
    selected = torch.arange(24).view(1, 1, 24).repeat(1, 8, 1)
    selected_residual = torch.randn(1, 192, 8)
    previous = torch.zeros(1, 8, 64, 8)
    full_keys = torch.zeros(1, 2, 512, 4)
    completed = Block._racer24_complete_residual(
        full_keys,
        selected,
        selected_residual,
        previous,
        attention_scale=0.5,
    )
    gathered = torch.gather(
        completed,
        2,
        selected.unsqueeze(-1).expand(1, 8, 24, 8),
    )
    assert completed.shape == (1, 8, 64, 8)
    assert torch.isfinite(completed).all()
    assert torch.equal(gathered, selected_residual.reshape(1, 8, 24, 8))


def test_racer_block_executes_q192_kv512_mlp192_and_restores_dense_carrier():
    torch.manual_seed(9)
    block = Block(embed_dims=8, num_heads=2, mlp_ratio=1.0, use_adapter=True, temporal_size=8).eval()
    carrier = torch.randn(1, 512, 8, requires_grad=True)
    previous = torch.randn_like(carrier, requires_grad=True)
    tubelets, spatial = _lineage()
    stats = _stats()
    output = block.forward_native_ragged(
        carrier,
        bucket_positions=[torch.arange(512).view(1, 512)],
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=8,
        grid_height=10,
        grid_width=10,
        packed_stats=stats,
        racer24_previous_dense_residual=previous,
        count_full_kv_tokens=True,
    )
    assert output.shape == carrier.shape
    assert torch.isfinite(output).all()
    assert stats["executed_attention_tokens"] == 192
    assert stats["executed_kv_tokens"] == 512
    assert stats["executed_attention_pairs"] == 192 * 512
    assert stats["executed_mlp_tokens"] == 192
    assert stats["executed_adapter_tokens"] == 512
    assert stats["racer24_block_forward_count"] == 1
    output.square().mean().backward()
    assert carrier.grad is not None
    assert previous.grad is not None


def test_backbone_schedule_adapter_dense512_no_new_parameters_and_no_state():
    torch.manual_seed(11)
    dense = _tiny_backbone()
    torch.manual_seed(11)
    racer = _tiny_backbone(racer24=_racer_config(), with_cp=True).train()
    assert [
        (name, tuple(parameter.shape), parameter.requires_grad)
        for name, parameter in racer.named_parameters()
    ] == [
        (name, tuple(parameter.shape), parameter.requires_grad)
        for name, parameter in dense.named_parameters()
    ]
    assert list(racer.state_dict()) == list(dense.state_dict())

    adapter_inputs = []
    originals = []
    for adapter in (block.adapter for block in racer.blocks):
        original = adapter.forward_native_ragged
        originals.append((adapter, original))

        def recorded(self, inputs, *args, _original=original, **kwargs):
            adapter_inputs.append(tuple(inputs.shape))
            return _original(inputs, *args, **kwargs)

        adapter.forward_native_ragged = MethodType(recorded, adapter)

    selected_native = torch.randn(1, 512, 3, 2, 16, 16)
    _, spatial = _lineage()
    physical = (
        torch.arange(8).view(1, 8, 1) * 100
        + spatial.view(1, 8, 64)
    ).reshape(1, 512)
    output = racer.forward_native_ragged(
        selected_native,
        physical,
        total_tubelets=8,
        source_grid_hw=(10, 10),
        use_absolute_position=False,
    )
    summary = racer.latest_native_packed_summary
    assert output.shape == (1, 512, 8)
    assert summary["dense_block_indices"] == [0, 1, 2, 3, 5, 7, 9, 11]
    assert summary["racer_block_indices"] == [4, 6, 8, 10]
    assert summary["racer24_selected_per_tubelet"] == 24
    assert summary["racer24_query_tokens_per_clip"] == 192
    assert summary["racer24_kv_tokens_per_clip"] == 512
    assert summary["executed_attention_tokens_all_blocks"] == 4864
    assert summary["executed_kv_tokens_all_blocks"] == 6144
    assert summary["executed_mlp_tokens_all_blocks"] == 4864
    assert summary["attention_pairs_all_blocks"] == 2_490_368
    assert summary["executed_adapter_tokens_all_blocks"] == 6144
    assert summary["racer24_cross_clip_state"] is False
    assert adapter_inputs == [(1, 512, 8)] * 12
    output.square().mean().backward()
    adapter_gradients = [
        parameter.grad
        for name, parameter in racer.named_parameters()
        if "adapter" in name and parameter.requires_grad
    ]
    assert any(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in adapter_gradients
    )

    for adapter, original in originals:
        adapter.forward_native_ragged = original
    racer.eval()
    adapter_inputs.clear()
    with torch.no_grad():
        first = racer.forward_native_ragged(
            selected_native,
            physical,
            total_tubelets=8,
            source_grid_hw=(10, 10),
            use_absolute_position=False,
        )
        second = racer.forward_native_ragged(
            selected_native,
            physical,
            total_tubelets=8,
            source_grid_hw=(10, 10),
            use_absolute_position=False,
        )
    assert torch.equal(first, second)


def test_config_recipe_and_iteration0_launcher_are_frozen_and_nontraining():
    config_dir = ROOT / "configs" / "adatad" / "thumos"
    config = Config.fromfile(
        config_dir / "georoute_official_r1_racer24_prebackbone_seed42_v001.py"
    )
    base = Config.fromfile(
        config_dir / "georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"
    )
    racer = config.model.backbone.backbone.racer24
    assert tuple(racer.racer_blocks) == (4, 6, 8, 10)
    assert (racer.selected_per_tubelet, racer.spatial_tokens_per_tubelet) == (24, 64)
    assert (racer.selected_query_tokens_per_clip, racer.full_kv_tokens_per_clip) == (192, 512)
    assert config.model.backbone.custom.zoomtoken_refresh_carry_mode == "full64"
    assert config.zoomtoken_p1_config.arm_surface == "RACER24"
    for field in ("dataset", "solver", "optimizer", "scheduler", "workflow", "post_processing"):
        assert config[field] == base[field]

    launcher = (
        ROOT / "scripts" / "run_zoomtoken_racer24_iteration0_n16r4.sh"
    ).read_text(encoding="utf-8")
    assert "tests/test_zoomtoken_racer24.py" in launcher
    assert "profile_zoomtoken_racer24_block.py" in launcher
    assert "--measurements 200" in launcher
    assert "--min-speedup 1.08" in launcher
    assert "--max-memory-ratio 1.05" in launcher
    assert "tools/train.py" not in launcher
    assert "sbatch" not in launcher


def test_racer_config_rejects_science_drift():
    bad = _racer_config()
    bad["racer_blocks"] = (5, 7, 9, 11)
    with pytest.raises(ValueError, match="racer_blocks"):
        _tiny_backbone(racer24=bad)
