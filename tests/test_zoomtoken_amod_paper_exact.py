from pathlib import Path

import pytest
import torch
from mmengine import Config

from opentad.models.backbones.vit_adapter import Attention, Block, VisionTransformerAdapter


ROOT = Path(__file__).resolve().parents[1]


def test_attention_column_mean_matches_full_attention_and_chunked_output():
    torch.manual_seed(5)
    attention = Attention(embed_dims=8, num_heads=2).eval()
    inputs = torch.randn(2, 7, 8)
    output, score = attention.forward_with_column_mean(
        inputs,
        query_chunk_size=3,
    )
    q, k, v = attention._project_qkv(inputs)
    probabilities = ((q * attention.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    expected_output = (probabilities @ v).transpose(1, 2).reshape(2, 7, 8)
    expected_output = attention.proj_drop(attention.proj(expected_output))
    expected_score = probabilities.float().sum(dim=-2).mean(dim=1) / 7.0
    assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(score, expected_score, atol=1e-7, rtol=1e-6)
    assert torch.allclose(score.sum(dim=1), torch.ones(2), atol=1e-6)


def test_amod_topk_is_exact_and_prefers_lower_token_index_on_ties():
    scores = torch.tensor([[0.5, 0.5, 0.1, 0.5, 0.2, 0.5]])
    indices = Block._stable_amod_topk_indices(scores, selected_count=3)
    assert indices.tolist() == [[0, 1, 3]]


def test_amod_block_identity_bypasses_unselected_attention_and_mlp():
    torch.manual_seed(7)
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    inputs = torch.randn(1, 6, 8)
    scores = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    output, selected = block.forward_amod(
        inputs,
        1,
        6,
        scores=scores,
        capacity=0.5,
    )
    assert selected.tolist() == [[0, 1, 2]]
    assert torch.equal(output[:, 3:], inputs[:, 3:])
    assert not torch.equal(output[:, :3], inputs[:, :3])


def test_amod_capacity_one_matches_dense_block_in_eval_mode():
    torch.manual_seed(11)
    block = Block(embed_dims=8, num_heads=2, use_adapter=False).eval()
    inputs = torch.randn(2, 5, 8)
    expected = block(inputs, 1, 5)
    actual, selected = block.forward_amod(
        inputs,
        1,
        5,
        scores=torch.randn(2, 5),
        capacity=1.0,
    )
    assert selected.tolist() == [list(range(5)), list(range(5))]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def _tiny_backbone(*, amod=None, with_cp=False):
    return VisionTransformerAdapter(
        img_size=32,
        patch_size=16,
        embed_dims=8,
        depth=12,
        num_heads=2,
        mlp_ratio=2.0,
        num_frames=2,
        tubelet_size=2,
        total_frames=2,
        adapter_index=list(range(12)),
        use_mean_pooling=False,
        with_cp=with_cp,
        amod=amod,
    )


def _amod_config(capacity=0.5):
    return dict(
        enabled=True,
        capacity=capacity,
        dense_block_indices=(0, 2, 4, 6, 8, 10),
        amod_block_indices=(1, 3, 5, 7, 9, 11),
        query_chunk_size=2,
        routing_score="preceding_dense_attention_column_mean",
        unselected_update="identity_bypass",
    )


def test_backbone_amod_schedule_has_no_new_parameters_or_temporal_state():
    torch.manual_seed(13)
    dense = _tiny_backbone()
    torch.manual_seed(13)
    amod = _tiny_backbone(amod=_amod_config()).eval()
    assert [name for name, _ in amod.named_parameters()] == [
        name for name, _ in dense.named_parameters()
    ]
    output = amod(torch.randn(1, 3, 2, 32, 32))
    assert output.shape == (1, 4, 8)
    summary = amod.latest_amod_summary
    assert summary["dense_block_indices"] == [0, 2, 4, 6, 8, 10]
    assert summary["amod_block_indices"] == [1, 3, 5, 7, 9, 11]
    assert summary["token_count"] == 4
    assert summary["selected_tokens_per_amod_block"] == 2
    assert summary["adapter_execution"] == "dense_full_token_grid"
    assert summary["temporal_state_reuse"] is False


def test_backbone_amod_checkpoint_path_backpropagates_through_dense_adapters():
    torch.manual_seed(17)
    model = _tiny_backbone(amod=_amod_config(), with_cp=True).train()
    output = model(torch.randn(1, 3, 2, 32, 32, requires_grad=True))
    output.sum().backward()
    adapter_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "adapter" in name and parameter.requires_grad
    ]
    assert adapter_grads and all(gradient is not None for gradient in adapter_grads)


def test_amod_rejects_non_alternating_or_stateful_route_combinations():
    bad = _amod_config()
    bad["amod_block_indices"] = (6, 7, 8, 9, 10, 11)
    with pytest.raises(ValueError, match="alternate"):
        _tiny_backbone(amod=bad)
    with pytest.raises(ValueError, match="mutually exclusive"):
        VisionTransformerAdapter(
            img_size=32,
            patch_size=16,
            embed_dims=8,
            depth=12,
            num_heads=2,
            num_frames=2,
            tubelet_size=2,
            total_frames=2,
            adapter_index=[],
            amod=_amod_config(),
            tubelet_token_redundancy_aux=dict(enabled=True),
        )


def test_official_amod_config_and_launcher_bind_the_frozen_reference_arm():
    config_path = (
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "georoute_official_amod50_prebackbone_seed42_v001.py"
    )
    config = Config.fromfile(config_path)
    amod = config.model.backbone.backbone.amod
    source = config_path.read_text(encoding="utf-8")
    assert '_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]' in source
    assert config.official_bc_arm == "AMOD50"
    assert amod.capacity == 0.5
    assert tuple(amod.dense_block_indices) == (0, 2, 4, 6, 8, 10)
    assert tuple(amod.amod_block_indices) == (1, 3, 5, 7, 9, 11)
    assert config.zoomtoken_p1_config.selected_tokens_per_amod_block == 400
    assert config.zoomtoken_p1_config.temporal_state_reuse is False
    assert config.workflow.checkpoint_interval == 5
    assert config.workflow.checkpoint_policy == "recovery_latest3_plus_final"

    launcher = (
        ROOT / "scripts" / "run_zoomtoken_official_prebackbone_bc_n16r4.sh"
    ).read_text(encoding="utf-8")
    assert "AMOD50)" in launcher
    assert 'CONFIG_NAME="georoute_official_amod50_prebackbone_seed42_v001.py"' in launcher
    train_source = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")
    assert '"AMOD50"' in train_source
