import copy

import torch

from libs.modeling.meta_archs import PtTransformerClsHead, PtTransformerRegHead
from libs.modeling.sparse_heads import (
    NativeGridSparseQuerySelector,
    build_sparse_head_execution_receipt,
    estimate_sparse_head_macs,
    run_sparse_cls_head,
    run_sparse_reg_head,
)


def _make_mask(lengths, time_size):
    positions = torch.arange(time_size)[None, None, :]
    return positions < torch.as_tensor(lengths)[:, None, None]


def test_stratified_selector_is_exact_deterministic_and_native_grid():
    masks = (
        _make_mask([600, 503], 640),
        _make_mask([300, 252], 320),
        _make_mask([150, 126], 160),
    )
    selector = NativeGridSparseQuerySelector(
        384, policy="stratified_uniform"
    )
    selected_a = selector(masks, ["video_a", "video_b"])
    selected_b = selector(masks, ["video_a", "video_b"])
    for batch_idx in range(2):
        assert sum(x[batch_idx].sum().item() for x in selected_a) == 384
    for source, first, second in zip(masks, selected_a, selected_b):
        assert torch.equal(first, second)
        assert not torch.logical_and(first, torch.logical_not(source)).any()


def test_video_hash_selector_is_fixed_per_video_and_not_step_random():
    masks = (_make_mask([64], 64), _make_mask([32], 32))
    selector = NativeGridSparseQuerySelector(
        16, policy="video_hash_random", hash_seed=1234567891
    )
    first = selector(masks, ["video_a"])
    second = selector(masks, ["video_a"])
    other = selector(masks, ["video_b"])
    assert all(torch.equal(x, y) for x, y in zip(first, second))
    assert any(not torch.equal(x, y) for x, y in zip(first, other))


def test_sparse_heads_equal_dense_heads_at_selected_physical_indices():
    torch.manual_seed(7)
    feats = (
        torch.randn(2, 4, 31),
        torch.randn(2, 4, 17),
    )
    masks = (
        _make_mask([29, 23], 31),
        _make_mask([15, 11], 17),
    )
    feats = tuple(feat * mask.to(feat.dtype) for feat, mask in zip(feats, masks))
    selector = NativeGridSparseQuerySelector(
        12, policy="stratified_uniform"
    )
    selected = selector(masks, ["video_a", "video_b"])

    cls_head = PtTransformerClsHead(
        4, 6, 3, num_layers=3, kernel_size=3, with_ln=True
    )
    reg_head = PtTransformerRegHead(
        4, 6, 2, num_layers=3, kernel_size=3, with_ln=True
    )
    dense_cls = cls_head(feats, masks)
    dense_reg = reg_head(feats, masks)
    sparse_cls = run_sparse_cls_head(cls_head, feats, masks, selected)
    sparse_reg = run_sparse_reg_head(reg_head, feats, masks, selected)

    for dense, sparse, sparse_mask in zip(dense_cls, sparse_cls, selected):
        expanded = sparse_mask.expand_as(dense)
        torch.testing.assert_close(
            sparse[expanded], dense[expanded], rtol=1e-5, atol=1e-6
        )
        assert torch.count_nonzero(sparse[~expanded]).item() == 0
    for dense, sparse, sparse_mask in zip(dense_reg, sparse_reg, selected):
        expanded = sparse_mask.expand_as(dense)
        torch.testing.assert_close(
            sparse[expanded], dense[expanded], rtol=1e-5, atol=1e-6
        )
        assert torch.count_nonzero(sparse[~expanded]).item() == 0


def test_sparse_selected_loss_has_dense_selected_gradient():
    torch.manual_seed(11)
    dense_leaf = torch.randn(1, 4, 37, requires_grad=True)
    dense_mask = _make_mask([35], 37)
    dense_feat = dense_leaf * dense_mask.to(dense_leaf.dtype)
    sparse_leaf = dense_leaf.detach().clone().requires_grad_(True)
    sparse_feat = sparse_leaf * dense_mask.to(sparse_leaf.dtype)
    selector = NativeGridSparseQuerySelector(
        9, policy="stratified_uniform"
    )
    selected = selector((dense_mask,), ["video"])

    dense_head = PtTransformerClsHead(
        4, 5, 3, num_layers=3, kernel_size=3, with_ln=True
    )
    sparse_head = copy.deepcopy(dense_head)
    dense_out = dense_head((dense_feat,), (dense_mask,))[0]
    sparse_out = run_sparse_cls_head(
        sparse_head, (sparse_feat,), (dense_mask,), selected
    )[0]
    dense_loss = dense_out[selected[0].expand_as(dense_out)].sum()
    sparse_loss = sparse_out[selected[0].expand_as(sparse_out)].sum()
    dense_loss.backward()
    sparse_loss.backward()

    torch.testing.assert_close(
        dense_leaf.grad, sparse_leaf.grad, rtol=1e-5, atol=1e-6
    )
    for dense_param, sparse_param in zip(
        dense_head.parameters(), sparse_head.parameters()
    ):
        torch.testing.assert_close(
            dense_param.grad, sparse_param.grad, rtol=1e-5, atol=1e-6
        )


def test_sparse_equivalence_preserves_dense_mask_semantics_with_holes():
    torch.manual_seed(13)
    feat = torch.randn(1, 4, 17)
    mask = torch.ones(1, 1, 17, dtype=torch.bool)
    mask[:, :, [3, 8, 14]] = False
    selected = torch.zeros_like(mask)
    selected[:, :, [2, 4, 7, 9, 13, 15]] = True
    head = PtTransformerClsHead(
        4, 5, 3, num_layers=3, kernel_size=3, with_ln=True
    )
    dense = head((feat,), (mask,))[0]
    sparse = run_sparse_cls_head(head, (feat,), (mask,), (selected,))[0]
    expanded = selected.expand_as(dense)
    torch.testing.assert_close(
        sparse[expanded], dense[expanded], rtol=1e-5, atol=1e-6
    )


def test_sparse_selected_all_is_dense_equivalent():
    torch.manual_seed(17)
    feat = torch.randn(1, 4, 19)
    mask = _make_mask([19], 19)
    selector = NativeGridSparseQuerySelector(
        128, policy="stratified_uniform"
    )
    selected = selector((mask,), None)
    head = PtTransformerClsHead(
        4, 5, 3, num_layers=3, kernel_size=3, with_ln=True
    )
    dense = head((feat,), (mask,))[0]
    sparse = run_sparse_cls_head(head, (feat,), (mask,), selected)[0]
    torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-6)


def test_sparse_mac_ledger_counts_only_required_physical_outputs():
    masks = (
        _make_mask([64], 64),
        _make_mask([32], 32),
        _make_mask([16], 16),
    )
    selector = NativeGridSparseQuerySelector(
        8, policy="stratified_uniform"
    )
    selected = selector(masks, ["video"])
    cls_head = PtTransformerClsHead(
        4, 6, 3, num_layers=3, kernel_size=3, with_ln=True
    )
    reg_head = PtTransformerRegHead(
        4, 6, 3, num_layers=3, kernel_size=3, with_ln=True
    )
    cls_cost = estimate_sparse_head_macs(
        cls_head, masks, selected, "cls_head"
    )
    reg_cost = estimate_sparse_head_macs(
        reg_head, masks, selected, "offset_head"
    )
    assert 0 < cls_cost["sparse_macs"] < cls_cost["dense_macs"]
    assert 0 < reg_cost["sparse_macs"] < reg_cost["dense_macs"]
    receipt = build_sparse_head_execution_receipt(
        cls_head,
        reg_head,
        masks,
        selected,
        budget=8,
        policy="stratified_uniform",
        training_loss_support="selected_native_grid_queries",
    )
    assert receipt["selected_count_contract_pass"] is True
    assert (
        receipt["training_loss_support"]
        == "selected_native_grid_queries"
    )
    assert receipt["selected_counts_per_sample_level"] == [[5, 2, 1]]
    assert 0.0 < receipt["theoretical_head_mac_fraction"] < 1.0
    assert receipt["wall_clock_claim_allowed"] is False
