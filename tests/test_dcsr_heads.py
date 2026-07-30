import copy

import pytest
import torch

from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.modeling.meta_archs import (
    PtTransformerClsHead,
    PtTransformerRegHead,
)
from libs.modeling.sparse_heads import (
    NativeGridSparseQuerySelector,
    build_dcsr_head_execution_receipt,
    run_dcsr_heads,
)
from libs.utils import make_optimizer


def _make_mask(lengths, time_size):
    positions = torch.arange(time_size)[None, None, :]
    return positions < torch.as_tensor(lengths)[:, None, None]


def _make_heads(num_layers=3):
    cls_head = PtTransformerClsHead(
        4,
        4,
        3,
        num_layers=num_layers,
        kernel_size=3,
        with_ln=True,
    )
    reg_head = PtTransformerRegHead(
        4,
        4,
        2,
        num_layers=num_layers,
        kernel_size=3,
        with_ln=True,
    )
    return cls_head, reg_head


def test_dcsr_g0_identity_is_bitwise_official_dense():
    torch.manual_seed(101)
    feats = (
        torch.randn(2, 4, 31),
        torch.randn(2, 4, 17),
    )
    masks = (
        _make_mask([29, 23], 31),
        _make_mask([15, 11], 17),
    )
    selector = NativeGridSparseQuerySelector(
        12, policy="stratified_uniform"
    )
    selected = selector(masks, ["video_a", "video_b"])
    cls_head, reg_head = _make_heads(num_layers=3)

    official_cls = cls_head(feats, masks)
    official_reg = reg_head(feats, masks)
    dcsr_cls, dcsr_reg = run_dcsr_heads(
        cls_head,
        reg_head,
        feats,
        masks,
        selected,
        residual_enabled=False,
    )

    assert all(
        torch.equal(official, routed)
        for official, routed in zip(official_cls, dcsr_cls)
    )
    assert all(
        torch.equal(official, routed)
        for official, routed in zip(official_reg, dcsr_reg)
    )


def test_dcsr_g1_keeps_unselected_dense_scaffold_and_signed_refinement():
    torch.manual_seed(103)
    feats = (
        torch.randn(1, 4, 31),
        torch.randn(1, 4, 17),
    )
    masks = (
        _make_mask([29], 31),
        _make_mask([15], 17),
    )
    selected = NativeGridSparseQuerySelector(
        8, policy="stratified_uniform"
    )(masks, ["video"])
    scaffold_cls, scaffold_reg = _make_heads(num_layers=1)
    residual_cls, residual_reg = _make_heads(num_layers=3)

    torch.nn.init.constant_(scaffold_reg.offset_head.conv.weight, 0.0)
    torch.nn.init.constant_(scaffold_reg.offset_head.conv.bias, 1.0)
    torch.nn.init.constant_(residual_cls.cls_head.conv.weight, 0.0)
    torch.nn.init.constant_(residual_cls.cls_head.conv.bias, 0.25)
    torch.nn.init.constant_(residual_reg.offset_head.conv.weight, 0.0)
    torch.nn.init.constant_(residual_reg.offset_head.conv.bias, -0.25)

    base_cls = scaffold_cls(feats, masks)
    base_reg = scaffold_reg(feats, masks)
    out_cls, out_reg = run_dcsr_heads(
        scaffold_cls,
        scaffold_reg,
        feats,
        masks,
        selected,
        residual_cls_head=residual_cls,
        residual_reg_head=residual_reg,
        residual_enabled=True,
    )

    for level_idx, selected_mask in enumerate(selected):
        cls_selected = selected_mask.expand_as(out_cls[level_idx])
        reg_selected = selected_mask.expand_as(out_reg[level_idx])
        assert torch.equal(
            out_cls[level_idx][~cls_selected],
            base_cls[level_idx][~cls_selected],
        )
        assert torch.equal(
            out_reg[level_idx][~reg_selected],
            base_reg[level_idx][~reg_selected],
        )
        torch.testing.assert_close(
            out_cls[level_idx][cls_selected],
            base_cls[level_idx][cls_selected] + 0.25,
        )
        torch.testing.assert_close(
            out_reg[level_idx][reg_selected],
            base_reg[level_idx][reg_selected] - 0.25,
        )


def test_dcsr_full_grid_loss_support_observes_unselected_queries():
    cfg = load_config("configs/thumos_i3d_dcsr_g1_uniform.yaml")
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    time_size = 4
    num_classes = model.num_classes
    full_mask = torch.ones(1, time_size, dtype=torch.bool)
    offsets = [torch.ones(1, time_size, 2)]
    labels = [torch.zeros(time_size, num_classes)]
    target_offsets = [torch.zeros(time_size, 2)]
    logits = torch.zeros(1, time_size, num_classes)

    model.loss_normalizer = 100.0
    reference = model.losses(
        [full_mask],
        [logits.clone()],
        offsets,
        labels,
        target_offsets,
    )["cls_loss"]
    changed = logits.clone()
    changed[:, 3, :] = 100.0
    model.loss_normalizer = 100.0
    observed = model.losses(
        [full_mask],
        [changed],
        offsets,
        labels,
        target_offsets,
    )["cls_loss"]

    assert not torch.isclose(observed, reference)
    assert (
        model.dcsr_training_loss_support
        == "official_all_valid_fpn_queries"
    )


@pytest.mark.parametrize(
    "config_path,expected_intervention",
    [
        (
            "configs/thumos_i3d_dcsr_g0_identity.yaml",
            {
                "enabled": True,
                "mode": "official_identity",
                "budget": 384,
                "policy": "stratified_uniform",
                "hash_seed": 1234567891,
                "scaffold_num_layers": 3,
                "residual_enabled": False,
                "residual_scale": 1.0,
                "training_loss_support": "official_all_valid_fpn_queries",
            },
        ),
        (
            "configs/thumos_i3d_dcsr_g1_uniform.yaml",
            {
                "enabled": True,
                "mode": "cheap_dense_scaffold",
                "budget": 384,
                "policy": "stratified_uniform",
                "hash_seed": 1234567891,
                "scaffold_num_layers": 1,
                "residual_enabled": True,
                "residual_scale": 1.0,
                "training_loss_support": "official_all_valid_fpn_queries",
            },
        ),
    ],
)
def test_dcsr_configs_change_only_declared_model_intervention(
    config_path, expected_intervention
):
    official = load_config("configs/thumos_i3d.yaml")
    dcsr = load_config(config_path)
    intervention = copy.deepcopy(dcsr["model"].pop("dcsr_head"))
    assert dcsr == official
    assert intervention == expected_intervention


def test_dcsr_development_pair_differs_only_by_model_intervention():
    dense = load_config("configs/thumos_i3d_dcsr_dev_dense.yaml")
    g1 = load_config("configs/thumos_i3d_dcsr_dev_g1_uniform.yaml")
    intervention = copy.deepcopy(g1["model"].pop("dcsr_head"))
    assert g1 == dense
    assert dense["train_split"] == ["validation"]
    assert dense["val_split"] == ["validation"]
    assert (
        dense["dataset"]["video_id_manifest"]
        == "$DCSR_INTERNAL_HOLDOUT_MANIFEST"
    )
    assert intervention["mode"] == "cheap_dense_scaffold"
    assert (
        intervention["training_loss_support"]
        == "official_all_valid_fpn_queries"
    )


def test_dcsr_g0_strict_state_dict_is_official_compatible():
    official_cfg = load_config("configs/thumos_i3d.yaml")
    g0_cfg = load_config("configs/thumos_i3d_dcsr_g0_identity.yaml")
    official = make_meta_arch(
        official_cfg["model_name"], **official_cfg["model"]
    )
    g0 = make_meta_arch(g0_cfg["model_name"], **g0_cfg["model"])
    assert set(official.state_dict()) == set(g0.state_dict())
    g0.load_state_dict(official.state_dict(), strict=True)
    assert g0.dcsr_scaffold_cls_head is None
    assert g0.dcsr_scaffold_reg_head is None


def test_dcsr_g1_registers_scaffold_and_zero_initializes_residual_api():
    cfg = load_config("configs/thumos_i3d_dcsr_g1_uniform.yaml")
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    assert model.dcsr_scaffold_cls_head is not None
    assert model.dcsr_scaffold_reg_head is not None
    assert torch.count_nonzero(
        model.cls_head.cls_head.conv.weight
    ).item() == 0
    assert torch.count_nonzero(
        model.cls_head.cls_head.conv.bias
    ).item() == 0
    assert torch.count_nonzero(
        model.reg_head.offset_head.conv.weight
    ).item() == 0
    assert torch.count_nonzero(
        model.reg_head.offset_head.conv.bias
    ).item() == 0

    optimizer = make_optimizer(model, cfg["opt"])
    optimized_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert optimized_ids == {id(param) for param in model.parameters()}


def test_dcsr_head_receipt_includes_scaffold_and_sparse_residual_cost():
    masks = (
        _make_mask([64], 64),
        _make_mask([32], 32),
        _make_mask([16], 16),
    )
    selected = NativeGridSparseQuerySelector(
        8, policy="stratified_uniform"
    )(masks, ["video"])
    scaffold_cls, scaffold_reg = _make_heads(num_layers=1)
    residual_cls, residual_reg = _make_heads(num_layers=3)
    receipt = build_dcsr_head_execution_receipt(
        scaffold_cls,
        scaffold_reg,
        residual_cls,
        residual_reg,
        masks,
        selected,
        budget=8,
        policy="stratified_uniform",
        training_loss_support="official_all_valid_fpn_queries",
    )
    assert receipt["unselected_queries_keep_dense_scaffold"] is True
    assert receipt["selected_counts_per_sample_level"] == [[5, 2, 1]]
    assert receipt["dense_scaffold_macs"] > 0
    assert receipt["sparse_residual_macs"] > 0
    assert receipt["combined_dcsr_head_macs"] < receipt[
        "official_dense_head_macs"
    ]
    assert 0.0 < receipt["theoretical_head_mac_fraction"] < 1.0
    assert receipt["wall_clock_claim_allowed"] is False


def test_dcsr_rejects_selected_only_training_support():
    cfg = load_config("configs/thumos_i3d_dcsr_g1_uniform.yaml")
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["dcsr_head"][
        "training_loss_support"
    ] = "selected_native_grid_queries"
    with pytest.raises(
        ValueError,
        match="official_all_valid_fpn_queries",
    ):
        make_meta_arch(cfg["model_name"], **model_cfg)
