"""Actual repository AdaTAD operators, small dimensions for CPU correctness.

These checks do not establish pretrained accuracy, GPU speed, or a completed
training capability. The full production configuration has a separate precheck.
"""
import copy
import itertools
import math
import torch
import pytest

from geosparse_ext.sparse import NativeEncoder, sparse_block, heavy_macs
from geosparse_ext.routing import ordered_log_prob, policy_losses, sample_subset
from geosparse_ext.geometry import canonical_atoms, support_distance, union_duration
from geosparse_ext.receivers import Receiver
from geosparse_ext.contracts import EvidenceBatch, RoutePlan


def source_model(depth=2, with_cp=False, total_frames=4):
    from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
    torch.manual_seed(3)
    model = VisionTransformerAdapter(img_size=32, patch_size=16, embed_dims=16, depth=depth,
                                    num_heads=2, num_frames=4, total_frames=total_frames,
                                    adapter_index=list(range(depth)), return_feat_map=True, with_cp=with_cp,
                                    drop_path_rate=0)
    for block in model.blocks:
        torch.nn.init.normal_(block.adapter.up_proj.weight, std=0.02)
    model._freeze_layers()
    return model.eval()


@pytest.mark.parametrize("route", ["A", "C"])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("with_cp", [False, True])
@pytest.mark.parametrize("implementation", ["reference", "bucket"])
def test_real_source_dense_limit_output_and_upstream_gradients(route, tail, with_cp, implementation):
    source = source_model(with_cp=with_cp, total_frames=8)
    assert all(block.with_cp == with_cp for block in source.blocks)
    candidate = copy.deepcopy(source)
    encoder = NativeEncoder(candidate, route, implementation).eval()
    frames = torch.randn(2, 3, 4, 32, 32, requires_grad=True)
    second = frames.detach().clone().requires_grad_()
    valid = torch.ones(2, 4, dtype=torch.bool)
    if tail:
        valid[1, 2:] = False
    mask = torch.ones((2, 8) if route == "A" else (2, 2, 1, 1), dtype=torch.bool)
    expected = source(frames)
    actual = encoder(second, mask)
    output_mask = valid.reshape(2, 2, 2).any(-1)[:, None, :, None, None]
    expected, actual = expected * output_mask, actual * output_mask
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    weight = torch.randn_like(expected)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(second.grad, frames.grad, atol=1e-5, rtol=1e-4)
    for (name, p), (_, q) in zip(source.named_parameters(), candidate.named_parameters()):
        if "adapter" in name:
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(q.grad, p.grad, atol=1e-5, rtol=1e-4)
        elif not p.requires_grad:
            assert p.grad is None and q.grad is None
    assert source.blocks[0].adapter.up_proj.weight.grad.abs().sum() > 0


def test_production_dense_limit_kernel_context_is_scoped_on_failure():
    from types import SimpleNamespace
    from geosparse_ext.gpu_precheck import dense_limit

    def flags():
        return (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled())

    previous = flags()
    observed = []

    class SourceProbe:
        def eval(self):
            observed.append(flags())
            raise RuntimeError("stop before model execution")

    model = SimpleNamespace(geosparse=SimpleNamespace(encoder=SimpleNamespace(source=SourceProbe())))
    with pytest.raises(RuntimeError, match="stop before model execution"):
        dense_limit(model, None, "A")
    assert observed == [(False, True, False, True, False)]
    assert flags() == previous


def test_zero_heavy_executes_light_tia_and_no_attention():
    source = source_model()
    encoder = NativeEncoder(source, "A").eval()
    frames = torch.randn(2, 3, 4, 32, 32)
    embedded, _, h, w = encoder.embed(frames)
    expected = embedded
    for block in source.blocks:
        expected = block.adapter(expected, h, w)
    called = []
    hooks = [block.attn.register_forward_pre_hook(lambda m, x: called.append(x[0].shape)) for block in source.blocks]
    actual = encoder(frames, torch.zeros(2, 8, dtype=torch.bool))
    for hook in hooks:
        hook.remove()
    assert not called
    torch.testing.assert_close(actual.flatten(2).transpose(1, 2), expected)
    assert heavy_macs(encoder.trace, 16) == 0


def test_real_compressed_operator_shapes_and_parent_isolation():
    source = source_model()
    encoder = NativeEncoder(source, "A").eval()
    frames = torch.randn(2, 3, 4, 32, 32)
    selected = torch.zeros(2, 8, dtype=torch.bool)
    selected[0, [0, 2]] = True
    selected[1, [1, 3, 5]] = True
    observed = []
    hooks = [m.register_forward_pre_hook(lambda m, args: observed.append(tuple(args[0].shape)))
             for block in source.blocks for m in (block.attn, block.mlp)]
    output = encoder(frames, selected)
    for hook in hooks:
        hook.remove()
    assert set(s[1] for s in observed) == {2, 3}
    changed = frames.clone()
    changed[1] += 10
    torch.testing.assert_close(encoder(changed, selected)[0], output[0])


def test_bucket_reference_gradients_ragged_and_empty():
    block = source_model().blocks[0]
    other = copy.deepcopy(block)
    x = torch.randn(3, 8, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mask = torch.zeros(3, 8, dtype=torch.bool)
    mask[:2, [1, 2, 6]] = True
    a = sparse_block(block, x, mask, 2, 2, implementation="reference")
    b = sparse_block(other, y, mask, 2, 2, implementation="bucket")
    torch.testing.assert_close(a, b)
    weight = torch.randn_like(a)
    (a * weight).sum().backward()
    (b * weight).sum().backward()
    torch.testing.assert_close(x.grad, y.grad)


def test_all_coarse_still_executes_one_quarter_tokens():
    encoder = NativeEncoder(source_model(), "C").eval()
    encoder(torch.randn(2, 3, 4, 32, 32), torch.zeros(2, 2, 1, 1, dtype=torch.bool))
    assert all(x["qkv_tokens"] == 2 for x in encoder.trace)
    assert heavy_macs(encoder.trace, 16) > 0


def test_mixed_bucket_matches_reference_updates_and_gradients():
    from geosparse_ext.sparse import mixed_block, mixed_block_bucket, MixedScaleProjection
    source = source_model()
    block, other = source.blocks[0], copy.deepcopy(source.blocks[0])
    projection = MixedScaleProjection(16, learned=True)
    second_projection = copy.deepcopy(projection)
    x = torch.randn(3, 8, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    fine = torch.tensor([[True, False], [False, True], [False, False]]).reshape(3, 2, 1, 1)
    valid = torch.ones(3, 8, dtype=torch.bool)
    valid[1, 4:] = False
    valid[2] = False
    a = mixed_block(block, x, fine, 2, 2, projection, valid)
    b = mixed_block_bucket(other, y, fine, 2, 2, second_projection, valid)
    torch.testing.assert_close(a, b)
    weight = torch.randn_like(a)
    (a * weight).sum().backward()
    (b * weight).sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    torch.testing.assert_close(projection.projection.weight.grad, second_projection.projection.weight.grad)


@pytest.mark.parametrize("route", ["A", "C"])
def test_training_dense_limit_preserves_parent_drop_path_at_padded_tails(route):
    from mmcv.cnn.bricks.drop import DropPath
    source = source_model(total_frames=12)
    for block in source.blocks:
        block.drop_path = DropPath(.4)
    source.train()
    candidate = NativeEncoder(copy.deepcopy(source), route).train()
    x = torch.randn(3, 3, 4, 32, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    valid = torch.ones(3, 4, dtype=torch.bool)
    valid[0, 2:] = False
    valid[2] = False
    mask = torch.ones((3, 8) if route == "A" else (3, 2, 1, 1), dtype=torch.bool)
    rng = torch.get_rng_state()
    expected = source(x)
    torch.set_rng_state(rng)
    actual = candidate(y, mask)
    output_mask = valid.reshape(3, 2, 2).any(-1)[:, None, :, None, None]
    expected, actual = expected * output_mask, actual * output_mask
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-4)


def test_static_depth_uses_preregistered_source_layers():
    source = source_model(depth=12)
    indices = [0, 4, 7, 11]
    encoder = NativeEncoder(source, "DENSE", active_layers=indices).eval()
    frames = torch.randn(1, 3, 4, 32, 32)
    expected, _, h, w = encoder.embed(frames)
    for i in indices:
        expected = source.blocks[i](expected, h, w)
    actual = encoder(frames)
    torch.testing.assert_close(actual.flatten(2).transpose(1, 2), source.norm(expected))
    assert sorted({row["layer"] for row in encoder.trace}) == indices


def test_bcr_dense_prefix_and_sparse_suffix_are_both_counted():
    source = source_model()
    encoder = NativeEncoder(source, "A", prefix_depth=1).eval()
    selected = torch.zeros(1, 8, dtype=torch.bool)
    selected[0, :2] = True
    encoder(torch.randn(1, 3, 4, 32, 32), selected)
    assert [row["qkv_tokens"] for row in encoder.trace] == [8, 2]


def test_native_atoms_do_not_repair_or_cross_parent():
    atoms = canonical_atoms(16, 14, 14, temporal_atom=4, spatial_group=2)
    parents = atoms // (8 * 196)
    assert torch.equal(parents[:, :1].expand_as(parents), parents)
    assert atoms.flatten().unique().numel() == 16 * 196


def test_pl_normalization_and_sign():
    logits = torch.tensor([0.2, 0.7, -0.4, 1.2], dtype=torch.float64, requires_grad=True)
    mass = sum(ordered_log_prob(logits, torch.tensor(order)).exp() for order in itertools.permutations(range(4), 2))
    assert abs(float(mass) - 1) < 1e-12
    logp = ordered_log_prob(logits, torch.tensor([0]))
    (logp * 2).backward()
    assert logits.grad[0] > 0  # minimizing positive cost lowers its probability
    for k in (0, 4):
        order, lp = sample_subset(logits, k)
        assert len(order) == k and float(lp) == 0


@pytest.mark.parametrize("quota", ["per_clip_equal_quota", "global_nonzero_per_clip"])
def test_constrained_policy_records_its_actual_sampling_likelihood(quota):
    from geosparse_ext.routing import quota_order
    logits = torch.tensor([.1, .7, -.2, .8], dtype=torch.float64, requires_grad=True)
    parents = torch.tensor([0, 0, 1, 1])
    fraction = .5 if quota == "per_clip_equal_quota" else .75
    order, lp = quota_order(logits, logits, parents, fraction, quota, "hybrid", False, True,
                            torch.Generator().manual_seed(8))
    assert len(order.unique()) == len(order) == (2 if fraction == .5 else 3)
    assert parents[order[:2]].tolist() == [0, 1]
    expected = logits[:2].log_softmax(0)[order[0]] + logits[2:].log_softmax(0)[order[1] - 2]
    if len(order) == 3:
        remaining = [i for i in range(4) if i not in order[:2].tolist()]
        expected = expected + logits[remaining].log_softmax(0)[remaining.index(int(order[2]))]
    torch.testing.assert_close(lp, expected)
    lp.backward()
    assert torch.isfinite(logits.grad).all()
    all_order, all_lp = quota_order(logits, logits, parents, 1., quota, "hybrid", False, True, None)
    assert len(all_order) == 4 and float(all_lp) == 0
    if quota == "global_nonzero_per_clip":
        with pytest.raises(ValueError, match="INFEASIBLE"):
            quota_order(logits, logits, parents, .1, quota, "hybrid", False, True, None)


def evidence():
    return EvidenceBatch(torch.randn(1, 2, 8), torch.tensor([[[[10., 11.], [30., 31.]], [[50., 51.], [52., 53.]]]]),
                         torch.ones(1, 2, 2, dtype=torch.bool), torch.tensor([[20., 51.]]),
                         torch.tensor([[[0., 0., 1., 1.], [0., 0., 1., 1.]]]),
                         torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2))


def test_support_union_does_not_fill_gap_or_count_duplicate_exposure():
    e = evidence()
    distance = support_distance(torch.tensor([[20.]]), e.source_support, e.support_valid)
    assert float(distance[0, 0, 0]) == 9
    torch.testing.assert_close(union_duration(e.source_support, e.support_valid), torch.tensor([[2., 2.]]))
    intervals = torch.tensor([[[[1., 2.], [1., 2.], [1.5, 3.]]]])
    assert union_duration(intervals, torch.ones(1, 1, 3, dtype=torch.bool)).item() == 2
    intervals = torch.tensor([[[[1., 10.], [2., 3.], [5., 12.]]]])
    assert union_duration(intervals, torch.ones(1, 1, 3, dtype=torch.bool)).item() == 11
    assert union_duration(intervals, torch.zeros(1, 1, 3, dtype=torch.bool)).item() == 0


def test_evidence_slots_pool_only_observed_members_on_odd_patch_grid():
    from types import SimpleNamespace
    from geosparse_ext.evidence import spatial_slots
    x = torch.arange(49.).reshape(1, 1, 1, 7, 7).requires_grad_()
    selected = torch.zeros(1, 1, 7, 7, dtype=torch.bool)
    selected[0, 0, 0, 0] = selected[0, 0, 2, 2] = selected[0, 0, 3, 3] = True
    layout = SimpleNamespace(source_support=torch.tensor([[[[0., 1.], [2., 3.]]]]),
                             support_valid=torch.ones(1, 1, 2, dtype=torch.bool), parent_tubelets=8)
    e = spatial_slots(x, selected, layout, 4)
    torch.testing.assert_close(e.features.flatten(), torch.tensor([8., 24.]))
    assert e.valid.all() and e.source_support.shape == (1, 2, 2, 2)
    e.features.sum().backward()
    assert x.grad[0, 0, 0, 0, 0] == .5 and x.grad[0, 0, 0, 3, 3] == 1
    assert x.grad[~selected[:, None]].count_nonzero() == 0


@pytest.mark.parametrize("mode", sorted(Receiver.modes))
def test_receiver_empty_evidence_exact_bypass(mode):
    model = Receiver(8, width=8, mode=mode, heads=2)
    e = evidence()
    e.valid.zero_()
    e.support_valid.zero_()
    query = torch.randn(1, 4, 8)
    actual = model(query, torch.arange(4.)[None], e, torch.tensor([2.]))
    torch.testing.assert_close(actual, query, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["support_attention", "timestamp_attention", "physical_interp", "concat_scatter"])
def test_receiver_storage_permutation(mode):
    model = Receiver(8, width=8, mode=mode, heads=2).eval()
    e = evidence()
    query, times = torch.randn(1, 4, 8), torch.tensor([[10., 20., 30., 51.]])
    expected = model(query, times, e, torch.tensor([10.]))
    other = EvidenceBatch(**{name: value[:, [1, 0]] for name, value in vars(e).items()})
    actual = model(query, times, other, torch.tensor([10.]))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_support_attention_null_inside_unobserved_hull():
    model = Receiver(8, width=8, heads=2)
    query = torch.randn(1, 1, 8)
    actual = model(query, torch.tensor([[20.]]), evidence(), torch.tensor([1.]))
    torch.testing.assert_close(actual, query, atol=0, rtol=0)
