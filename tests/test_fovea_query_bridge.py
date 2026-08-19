"""Focused unit tests for FoveaSampler / Query-Bridge.

Run with the project conda env:
    python -m pytest tests/test_fovea_query_bridge.py -v
"""

import math

import pytest
import torch

from opentad.models.losses.fovea_losses import build_fovea_losses, build_gt_geometry_mask
from opentad.models.selectors.fovea_heads import CoarseProposalHead, FoveaHeads
from opentad.models.selectors.fovea_query_bridge_selector import (
    FoveaQueryBridgeFrameSelector,
    _transport_inputs,
)
from opentad.models.selectors.fovea_sampler import FoveatedSampler, GumbelTopK
from opentad.models.selectors.fovea_scout import FoveaScout
from opentad.models.selectors.query_bridge import QueryBridgeWithDecoder

T = 64
D = 32


def _valid_mask(batch=1, length=T, valid_len=60):
    mask = torch.zeros(batch, length, dtype=torch.float32)
    mask[:, :valid_len] = 1.0
    return mask


def _observations(batch=1, length=T, size=32):
    return torch.randn(batch, 3, length, size, size)


def test_scout_shapes_and_mask():
    scout = FoveaScout(hidden_dim=D, temporal_layers=2, dilations=(1, 2))
    obs = _observations()
    valid = _valid_mask().bool()
    z = scout(obs, valid)
    assert tuple(z.shape) == (1, T, D)
    assert bool(torch.isfinite(z[:, :60]).all().item())
    assert bool((z[:, 60:] == 0).all().item())


def test_query_bridge_contribution_is_masked_softmax():
    bridge = QueryBridgeWithDecoder(hidden_dim=D, num_queries=4, num_decoder_layers=1, num_heads=4)
    valid = _valid_mask().bool()
    z = torch.randn(1, T, D)
    out = bridge(z, valid)
    assert tuple(out.contribution.shape) == (1, 4, T)
    assert bool((out.contribution[:, :, 60:] == 0).all().item())
    sums = out.contribution.sum(dim=-1)
    assert bool(torch.allclose(sums, torch.ones_like(sums), atol=1.0e-5))
    # Q1 is a selector-internal memory with the documented contract.
    assert out.metadata["query_memory_enters_heavy_detector"] is False
    assert tuple(out.query_memory.shape) == (1, 4, D)


def test_three_manual_branches_stay_separate_and_fuse():
    heads = FoveaHeads(hidden_dim=D, num_queries=4)
    valid = _valid_mask().bool()
    z = torch.randn(1, T, D)
    a = torch.rand(1, 4, T)
    a = a.masked_fill(~valid.unsqueeze(1), 0.0)
    a = a / a.sum(dim=-1, keepdim=True)
    q1 = torch.randn(1, 4, D)
    out = heads(z, a, q1, valid)
    for key in ("saliency", "boundary_start", "boundary_end", "boundary_edge", "uncertainty", "uncertainty_context"):
        assert key in out
    expected = out["saliency"] + out["boundary_edge"] + out["uncertainty_context"]
    assert bool(torch.allclose(out["frame_score"][:, :60], expected[:, :60], atol=1.0e-5))
    assert bool(torch.isneginf(out["frame_score"][:, 60:]).all().item())


def test_coarse_head_outputs():
    head = CoarseProposalHead(hidden_dim=D)
    valid = _valid_mask().bool()
    z = torch.randn(1, T, D)
    a = torch.zeros(1, 4, T)
    a[:, :, :60] = 1.0 / 60.0
    out = head(z, a, valid)
    assert tuple(out["coarse_logits"].shape) == (1, T)
    assert bool((out["coarse_width"][:, 60:] == 0).all().item())


def test_gumbel_topk_exact_k_and_straight_through():
    topk = GumbelTopK(tau=1.0)
    logits = torch.randn(1, T, requires_grad=True)
    valid = _valid_mask().bool()
    out = topk(logits, valid, k=32)
    assert int(out["indices"].numel()) == 32
    assert bool((out["indices"] < 60).all().item())
    assert float(out["onehot"].sum().item()) == pytest.approx(32.0)
    assert float(out["soft_sample"].sum().item()) == pytest.approx(32.0, abs=1.0e-4)
    assert bool((out["soft_sample"][:, 60:] == 0).all().item())
    transport = out["onehot"] + (out["soft_sample"] - out["soft_sample"].detach())
    assert bool(torch.allclose(transport.sum(dim=-1), torch.full((1,), 32.0), atol=1.0e-4))
    assert bool(transport.requires_grad)


def test_foveated_sampler_inference_deterministic_exact_k():
    sampler = FoveatedSampler(
        target_k=32,
        min_k=32,
        max_k=32,
        budget_step=16,
        boundary_quota=8,
        boundary_center_top_m=4,
        boundary_radius=2,
        boundary_pair_max_gap=8,
        mmr_lambda=0.10,
        dynamic_budget=False,
    )
    score = torch.randn(1, T)
    score[:, 20] += 5.0
    valid = _valid_mask().bool()
    r1 = sampler(score, valid, training=False)
    r2 = sampler(score, valid, training=False)
    assert torch.equal(r1["indices"], r2["indices"])
    idx = r1["indices"][0].tolist()
    assert len(idx) == len(set(idx)) == 32
    assert idx == sorted(idx)
    assert all(i < 60 for i in idx)
    assert int(r1["metadata"]["budget"]) == 32
    assert r1["metadata"]["mode"] == "inference_greedy_mmr"
    assert bool(torch.all(r1["transport"].sum(dim=-1) == 32))
    assert bool((r1["probs"][:, 60:] == 0).all().item())


def test_foveated_sampler_training_transport_sum():
    sampler = FoveatedSampler(
        target_k=32,
        min_k=32,
        max_k=32,
        boundary_quota=8,
        boundary_center_top_m=4,
        dynamic_budget=False,
    )
    score = torch.randn(1, T)
    valid = _valid_mask().bool()
    r = sampler(score, valid, training=True)
    assert int(r["indices"].numel()) == 32
    assert float(r["transport"].sum(dim=-1).item()) == pytest.approx(32.0, abs=1.0e-4)
    assert bool((r["probs"][:, 60:] == 0).all().item())


def test_gt_geometry_mask_values():
    valid = torch.ones(1, 16, dtype=torch.bool)
    gt = [torch.tensor([[4.0, 9.0]])]
    mask = build_gt_geometry_mask(gt_segments=gt, valid=valid, boundary_radius=2)
    assert int(mask[0, 0].item()) == 0
    assert int(mask[0, 4].item()) == 2
    assert int(mask[0, 5].item()) == 1
    assert int(mask[0, 7].item()) == 2
    assert int(mask[0, 9].item()) == 2
    assert int(mask[0, 12].item()) == 0


def test_build_fovea_losses_keys_and_finite():
    valid = _valid_mask().bool()
    contribution = torch.rand(1, 4, T)
    contribution = contribution.masked_fill(~valid.unsqueeze(1), 0.0)
    contribution = contribution / contribution.sum(dim=-1, keepdim=True)
    frame_score = torch.randn(1, T).masked_fill(~valid, -torch.inf)
    coarse_logits = torch.randn(1, T).masked_fill(~valid, -10.0)
    coarse_center = torch.zeros(1, T)
    coarse_width = torch.ones(1, T)
    gt = [torch.tensor([[10.0, 30.0]])]
    bundle = build_fovea_losses(
        contribution=contribution,
        frame_score=frame_score,
        coarse_logits=coarse_logits,
        coarse_center=coarse_center,
        coarse_width=coarse_width,
        valid=valid,
        gt_segments=gt,
        boundary_radius=2,
        budget_target=32,
        selected_count=torch.tensor([32.0]),
        weights={"mask": 1.0, "coarse": 1.0, "cycle": 0.0, "budget": 0.05, "diversity": 0.05},
    )
    for value in (bundle.mask_loss, bundle.coarse_loss, bundle.budget_loss, bundle.diversity_loss):
        assert torch.isfinite(value).all()
    assert float(bundle.cycle_loss.item()) == 0.0


def test_selector_forward_test_no_inference_leakage():
    selector = FoveaQueryBridgeFrameSelector(
        scout_hidden_dim=32,
        scout_temporal_layers=2,
        scout_dilations=(1, 2),
        query_hidden_dim=32,
        num_queries=4,
        query_decoder_layers=1,
        query_num_heads=4,
        target_k=32,
        min_k=32,
        max_k=32,
        boundary_quota=0,
        boundary_center_top_m=0,
        mmr_lambda=0.0,
        cycle_enabled=False,
    )
    selector.eval()
    inputs = _observations(batch=1, length=T, size=32)
    masks = _valid_mask()
    metas = [{"fps": 30.0, "snippet_stride": 1.0, "offset_frames": 0.0}]
    out = selector.forward_test(inputs, masks, metas)
    assert int(out["selected_len"]) % 16 == 0
    assert int(out["selected_len"]) == 32
    assert tuple(out["inputs"].shape) == (1, 3, 32, 32, 32)
    positions = out["metas"][0]["duca_sparse_physical_positions"]
    assert len(positions) == 32
    assert positions == sorted(positions)
    assert all(0 <= p < 60 for p in positions)
    assert out["metas"][0]["duca_sparse_variable_compute"] is True
    assert selector._cycle_context is None and selector._pending_cycle is None
    # hard selection numerically equals a plain gather
    flat = inputs.permute(0, 2, 1, 3, 4).reshape(1, T, -1)
    expected = flat.gather(1, torch.as_tensor([positions]).unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
    actual = out["inputs"].permute(0, 2, 1, 3, 4).reshape(1, 32, -1)
    assert bool(torch.equal(actual, expected))


def test_transport_inputs_forward_equals_hard_gather():
    inputs = _observations(batch=1, length=T, size=16)
    transport = torch.zeros(1, T, 16)
    idx = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    for k, j in enumerate(idx):
        transport[0, j, k] = 1.0
    indices = torch.as_tensor([idx])
    out, _ = _transport_inputs(inputs, transport, indices, 16)
    flat = inputs.permute(0, 2, 1, 3, 4).reshape(1, T, -1)
    expected = flat.gather(1, indices.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
    expected = expected.reshape(1, 16, 3, 16, 16).permute(0, 2, 1, 3, 4)
    assert torch.equal(out, expected)
