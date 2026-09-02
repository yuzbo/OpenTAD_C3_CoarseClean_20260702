import torch

from opentad.models.backbones.vit_adapter import (
    Block,
    mixture_of_depths_capacity_for_successful_update,
)


def test_mixture_of_depths_segment_capacity_schedule_matches_contract():
    schedule = [
        {"start": 0, "end": 1500, "start_value": 1.0, "end_value": 1.0},
        {"start": 1500, "end": 3500, "start_value": 1.0, "end_value": 0.75, "interpolation": "cosine"},
        {"start": 3500, "end": 5000, "start_value": 0.75, "end_value": 0.65, "interpolation": "cosine"},
        {"start": 5000, "end": 6000, "start_value": 0.65, "end_value": 0.65},
    ]

    assert mixture_of_depths_capacity_for_successful_update(0, schedule, 0.65) == 1.0
    assert mixture_of_depths_capacity_for_successful_update(1500, schedule, 0.65) == 1.0
    mid = mixture_of_depths_capacity_for_successful_update(2500, schedule, 0.65)
    assert 0.75 < mid < 1.0
    assert mixture_of_depths_capacity_for_successful_update(5000, schedule, 0.65) == 0.65


def test_block_amod_capacity_one_is_dense_parity_without_padding():
    torch.manual_seed(11)
    block = Block(
        embed_dims=16,
        num_heads=4,
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_adapter=False,
    )
    block.eval()
    x = torch.randn(2, 16, 16)
    route_scores = torch.randn(2, 16)

    dense = block(x.clone(), h=2, w=2)
    amod = block.forward_amod(x.clone(), h=2, w=2, route_scores=route_scores, capacity=1.0)

    assert torch.allclose(amod, dense, atol=1e-6, rtol=1e-6)
    assert block.last_amod_summary["all_dense"] is True
    assert block.last_amod_summary["selected_counts"] == [16, 16]


def test_block_amod_uses_per_sample_valid_token_capacity_and_never_selects_padding():
    torch.manual_seed(12)
    block = Block(
        embed_dims=16,
        num_heads=4,
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_adapter=False,
    )
    x = torch.randn(2, 10, 16)
    route_scores = torch.zeros(2, 10)
    route_scores[:, 6:] = 100.0
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False, False, False],
        ]
    )

    out = block.forward_amod(
        x,
        h=1,
        w=1,
        route_scores=route_scores,
        capacity=0.5,
        temporal_token_mask=mask,
    )

    assert out.shape == x.shape
    summary = block.last_amod_summary
    assert summary["valid_counts"] == [6, 4]
    assert summary["selected_counts"] == [3, 2]
    assert summary["padding_selected_count"] == 0
    assert summary["hard_route"] == "topk_on_router_logits"
