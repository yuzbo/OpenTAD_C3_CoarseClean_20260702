from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "tests" / "test_c3_physical_grid_actionformer_candidate.py"


def _load_helper():
    name = "duca_rime_tridet_physical_grid_helpers"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helper = _load_helper()
TORCH_AVAILABLE = helper.TORCH_AVAILABLE
if TORCH_AVAILABLE:
    import torch


def _make_tridet_head():
    helper._install_head_runtime_or_skip()
    builder = sys.modules[f"{helper.RUNTIME_PACKAGE}.models.builder"]
    default_build_loss = builder.build_loss
    build_calls = 0

    class _ElementwiseDummyLoss(torch.nn.Module):
        def forward(self, inputs, targets, reduction="none", **kwargs):
            loss = inputs.sum() * 0.0
            if reduction in {"mean", "sum"}:
                return loss
            return torch.zeros_like(inputs)

    def _build_tridet_loss(cfg):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            return _ElementwiseDummyLoss()
        return default_build_loss(cfg)

    builder.build_loss = _build_tridet_loss
    misc_name = f"{helper.RUNTIME_PACKAGE}.models.bricks.misc"
    if misc_name not in sys.modules:
        misc = types.ModuleType(misc_name)
        misc.Scale = helper._Scale
        sys.modules[misc_name] = misc
    module = helper._load_module(
        f"{helper.RUNTIME_PACKAGE}.models.dense_heads.tridet_head",
        ROOT / "opentad" / "models" / "dense_heads" / "tridet_head.py",
    )
    head = module.TriDetHead(
        num_classes=2,
        in_channels=2,
        feat_channels=2,
        num_convs=0,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1],
            regression_range=[(0, 10000)],
        ),
        loss=types.SimpleNamespace(
            cls_loss=dict(type="DummyLoss"),
            reg_loss=dict(type="DummyLoss"),
            iou_rate=dict(type="DummyLoss"),
        ),
        num_bins=2,
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            contract="duca_rime_physical_dynamic_k_v1",
        ),
    )
    with torch.no_grad():
        for module_ in (
            head.cls_head,
            head.reg_head,
            head.cls_start_head,
            head.cls_end_head,
        ):
            module_.weight.zero_()
            module_.bias.zero_()
        head.scale[0].scale.fill_(1.0)
    return head


def _rime_meta():
    return {
        "video_name": "synthetic_rime_tridet",
        "irregular_selected_positions": [0, 2, 5, 8],
        "selected_dense_indices": [0, 2, 5, 8],
        "selected_valid_len": 4,
        "irregular_selected_count": 4,
        "irregular_selected_valid_len": 9,
        "irregular_dense_valid_len": 9,
        "irregular_native_axis": True,
        "remap_gt_to_selected_axis": False,
        "gt_remapped_to_selected_axis": False,
        "pc_ot_mras_prebackbone_remap_gt_to_selected_axis": False,
        "selected_axis_remap_required": False,
        "detector_prediction_inverse_map_required": False,
        "detector_output_coordinate_space": "dense_physical",
        "proposal_axis": "dense_physical",
        "duca_contract": "duca_rime_physical_dynamic_k_v1",
        "physical_grid_contract": "duca_rime_physical_dynamic_k_v1",
        "duca_backbone_tail_padding_mode": "none_exact_k_bucket",
    }


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch runtime unavailable")
def test_tridet_head_decodes_and_assigns_on_rime_physical_grid():
    head = _make_tridet_head()
    features = [torch.zeros(1, 2, 4)]
    masks = [torch.ones(1, 4, dtype=torch.bool)]
    meta = _rime_meta()

    points, regression, scores = head.forward_test(
        features,
        masks,
        metas=[meta],
    )

    assert points.shape == (1, 4, 4)
    assert torch.allclose(points[0, :, 0], torch.tensor([0.0, 2.0, 5.0, 8.0]))
    assert regression.shape == (1, 2, 4, 2)
    assert scores.shape == (1, 4, 2)

    losses = head.forward_train(
        features,
        masks,
        gt_segments=[torch.tensor([[1.5, 2.5]], dtype=torch.float32)],
        gt_labels=[torch.tensor([1], dtype=torch.long)],
        metas=[_rime_meta()],
    )
    assert set(losses) == {"cls_loss", "reg_loss"}
    assert all(torch.isfinite(value) for value in losses.values())


def test_tridet_remaps_selected_axis_before_nms_in_source_contract():
    source = (ROOT / "opentad" / "models" / "detectors" / "tridet.py").read_text(
        encoding="utf-8"
    )
    post = source[source.index("    def post_processing") :]
    remap = post.index("_remap_selector_segments_for_post_processing")
    nms = post.index("batched_nms", remap)
    seconds = post.index("convert_to_seconds", nms)

    assert remap < nms < seconds
