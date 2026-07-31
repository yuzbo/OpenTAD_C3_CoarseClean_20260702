import copy

import pytest
import torch

from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.modeling.sparse_heads import run_dcsr_heads
from libs.utils import make_optimizer


CONFIGS = {
    "d1_off": "configs/thumos_i3d_odfcr_dev_d1_off.yaml",
    "d1_all": "configs/thumos_i3d_odfcr_dev_d1_all.yaml",
    "d3_off": "configs/thumos_i3d_odfcr_dev_d3_off.yaml",
    "d3_all": "configs/thumos_i3d_odfcr_dev_d3_all.yaml",
}
OFFICIAL_REFERENCE = "configs/thumos_i3d_odfcr_dev_dense_reference.yaml"


def _make_model(arm, seed=2026073101):
    torch.manual_seed(seed)
    cfg = load_config(CONFIGS[arm])
    return cfg, make_meta_arch(cfg["model_name"], **cfg["model"])


def _assert_state_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key


def _assert_tensor_tuple_equal(left, right):
    assert len(left) == len(right)
    for expected, actual in zip(left, right):
        assert torch.equal(expected, actual)


def _assert_final_equal(left, right):
    assert len(left) == len(right)
    for expected, actual in zip(left, right):
        assert expected["video_id"] == actual["video_id"]
        for key in ("segments", "scores", "labels"):
            assert torch.equal(expected[key], actual[key]), key


@pytest.mark.parametrize(
    "arm,depth,residual,support",
    [
        ("d1_off", 1, False, "off"),
        ("d1_all", 1, True, "all_valid"),
        ("d3_off", 3, False, "off"),
        ("d3_all", 3, True, "all_valid"),
    ],
)
def test_odfcr_configs_change_only_the_frozen_intervention(
    arm, depth, residual, support
):
    official = load_config(OFFICIAL_REFERENCE)
    candidate = load_config(CONFIGS[arm])
    intervention = copy.deepcopy(candidate["model"].pop("odfcr_head"))
    assert candidate == official
    assert intervention == {
        "enabled": True,
        "mode": "official_dense_floor_factorial",
        "scaffold_num_layers": depth,
        "residual_enabled": residual,
        "residual_execution_support": support,
        "residual_num_layers": 3,
        "residual_scale": 1.0,
        "training_loss_support": "official_all_valid_fpn_queries",
    }


def test_odfcr_d3_off_state_dict_is_bitwise_official_dense():
    torch.manual_seed(2026073101)
    official_cfg = load_config(OFFICIAL_REFERENCE)
    official = make_meta_arch(
        official_cfg["model_name"], **official_cfg["model"]
    )
    _, d3_off = _make_model("d3_off")

    _assert_state_equal(official.state_dict(), d3_off.state_dict())
    assert d3_off.odfcr_scaffold_cls_head is None
    assert d3_off.odfcr_scaffold_reg_head is None
    assert d3_off.odfcr_residual_cls_head is None
    assert d3_off.odfcr_residual_reg_head is None
    d3_off.load_state_dict(official.state_dict(), strict=True)


def test_odfcr_d3_all_preserves_floor_and_starts_as_exact_noop():
    torch.manual_seed(2026073101)
    official_cfg = load_config(OFFICIAL_REFERENCE)
    official = make_meta_arch(
        official_cfg["model_name"], **official_cfg["model"]
    )
    _, d3_all = _make_model("d3_all")

    _assert_state_equal(
        official.cls_head.state_dict(), d3_all.cls_head.state_dict()
    )
    _assert_state_equal(
        official.reg_head.state_dict(), d3_all.reg_head.state_dict()
    )
    assert torch.count_nonzero(
        d3_all.odfcr_residual_cls_head.cls_head.conv.weight
    ).item() == 0
    assert torch.count_nonzero(
        d3_all.odfcr_residual_cls_head.cls_head.conv.bias
    ).item() == 0
    assert torch.count_nonzero(
        d3_all.odfcr_residual_reg_head.offset_head.conv.weight
    ).item() == 0
    assert torch.count_nonzero(
        d3_all.odfcr_residual_reg_head.offset_head.conv.bias
    ).item() == 0

    feats = tuple(
        torch.randn(1, 512, length) for length in (17, 9, 5, 3, 2, 1)
    )
    masks = tuple(
        torch.ones(1, 1, length, dtype=torch.bool)
        for length in (17, 9, 5, 3, 2, 1)
    )
    official_cls = d3_all.cls_head(feats, masks)
    official_reg = d3_all.reg_head(feats, masks)
    routed_cls, routed_reg = run_dcsr_heads(
        d3_all.cls_head,
        d3_all.reg_head,
        feats,
        masks,
        masks,
        residual_cls_head=d3_all.odfcr_residual_cls_head,
        residual_reg_head=d3_all.odfcr_residual_reg_head,
        residual_enabled=True,
        residual_scale=1.0,
        residual_execution="dense",
    )
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(official_cls, routed_cls)
    )
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(official_reg, routed_reg)
    )


def test_odfcr_d3_component_outputs_are_bitwise_official():
    torch.manual_seed(2026073101)
    official_cfg = load_config(OFFICIAL_REFERENCE)
    official = make_meta_arch(
        official_cfg["model_name"], **official_cfg["model"]
    ).eval()
    _, d3_off = _make_model("d3_off")
    _, d3_all = _make_model("d3_all")
    d3_off.eval()
    d3_all.eval()
    feats = tuple(
        torch.randn(1, 512, length) for length in (17, 9, 5, 3, 2, 1)
    )
    masks = tuple(
        torch.ones(1, 1, length, dtype=torch.bool)
        for length in (17, 9, 5, 3, 2, 1)
    )
    official_points = official.point_generator(feats)
    d3_off_points = d3_off.point_generator(feats)
    d3_all_points = d3_all.point_generator(feats)
    official_cls = official.cls_head(feats, masks)
    official_reg = official.reg_head(feats, masks)
    d3_off_cls = d3_off.cls_head(feats, masks)
    d3_off_reg = d3_off.reg_head(feats, masks)
    d3_all_cls, d3_all_reg = run_dcsr_heads(
        d3_all.cls_head,
        d3_all.reg_head,
        feats,
        masks,
        masks,
        residual_cls_head=d3_all.odfcr_residual_cls_head,
        residual_reg_head=d3_all.odfcr_residual_reg_head,
        residual_enabled=True,
        residual_scale=1.0,
        residual_execution="dense",
    )
    _assert_tensor_tuple_equal(official_points, d3_off_points)
    _assert_tensor_tuple_equal(official_points, d3_all_points)
    _assert_tensor_tuple_equal(official_cls, d3_off_cls)
    _assert_tensor_tuple_equal(official_reg, d3_off_reg)
    _assert_tensor_tuple_equal(official_cls, d3_all_cls)
    _assert_tensor_tuple_equal(official_reg, d3_all_reg)

    video = {
        "video_id": "odfcr_component_identity",
        "fps": 30.0,
        "duration": 10.0,
        "feat_stride": 4,
        "feat_num_frames": 16,
    }

    def decode(model, points, cls_logits, offsets):
        return model.inference(
            [video],
            points,
            [mask.squeeze(1) for mask in masks],
            [tensor.permute(0, 2, 1) for tensor in cls_logits],
            [tensor.permute(0, 2, 1) for tensor in offsets],
        )

    official_final = decode(
        official, official_points, official_cls, official_reg
    )
    _assert_final_equal(
        official_final,
        decode(d3_off, d3_off_points, d3_off_cls, d3_off_reg),
    )
    _assert_final_equal(
        official_final,
        decode(d3_all, d3_all_points, d3_all_cls, d3_all_reg),
    )


def test_odfcr_depth_one_scaffold_initialization_is_paired():
    _, d1_off = _make_model("d1_off")
    _, d1_all = _make_model("d1_all")

    _assert_state_equal(
        d1_off.odfcr_scaffold_cls_head.state_dict(),
        d1_all.odfcr_scaffold_cls_head.state_dict(),
    )
    _assert_state_equal(
        d1_off.odfcr_scaffold_reg_head.state_dict(),
        d1_all.odfcr_scaffold_reg_head.state_dict(),
    )
    assert d1_off.odfcr_residual_cls_head is None
    assert d1_all.odfcr_residual_cls_head is not None


@pytest.mark.parametrize("arm", ["d1_all", "d3_all"])
def test_odfcr_residual_parameters_are_optimized(arm):
    cfg, model = _make_model(arm)
    optimizer = make_optimizer(model, cfg["opt"])
    optimized_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    residual_ids = {
        id(param)
        for name, param in model.named_parameters()
        if name.startswith("odfcr_residual_")
    }
    assert residual_ids
    assert residual_ids <= optimized_ids
    assert optimized_ids == {id(param) for param in model.parameters()}


def test_odfcr_frozen_k384_replay_is_eval_only_and_exact_budget():
    _, model = _make_model("d3_all")
    with pytest.raises(RuntimeError, match="eval mode"):
        model.configure_odfcr_frozen_replay(384)
    model.eval()
    model.configure_odfcr_frozen_replay(
        384,
        policy="stratified_uniform",
        hash_seed=2026073100,
    )
    masks = (
        torch.ones(1, 1, 300, dtype=torch.bool),
        torch.ones(1, 1, 200, dtype=torch.bool),
    )
    selected = model.odfcr_replay_query_selector(masks)
    assert sum(int(mask.sum()) for mask in selected) == 384
    assert model.odfcr_replay_query_selector.hash_seed == 2026073100
    model.train()
    with pytest.raises(RuntimeError, match="cannot remain active"):
        model([{}])
    model.eval()
    model.clear_odfcr_frozen_replay()
    assert model.odfcr_replay_query_selector is None
    with pytest.raises(RuntimeError, match="fixed to"):
        model.configure_odfcr_frozen_replay(383)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("scaffold_num_layers", 2, "scaffold_num_layers"),
        ("residual_num_layers", 2, "residual_num_layers"),
        ("residual_scale", 0.5, "exactly 1.0"),
        ("training_loss_support", "selected_native_grid_queries", "all_valid"),
    ],
)
def test_odfcr_rejects_contract_drift(field, value, match):
    cfg = load_config(CONFIGS["d3_all"])
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["odfcr_head"][field] = value
    with pytest.raises(ValueError, match=match):
        make_meta_arch(cfg["model_name"], **model_cfg)


def test_odfcr_rejects_residual_support_mismatch():
    cfg = load_config(CONFIGS["d3_all"])
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["odfcr_head"]["residual_execution_support"] = "off"
    with pytest.raises(ValueError, match="all_valid"):
        make_meta_arch(cfg["model_name"], **model_cfg)


def test_odfcr_disabled_contract_is_fail_closed():
    cfg = load_config(CONFIGS["d3_all"])
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["odfcr_head"] = {
        "enabled": False,
        "mode": "silent_contract_drift",
    }
    with pytest.raises(ValueError, match="accepts only"):
        make_meta_arch(cfg["model_name"], **model_cfg)
