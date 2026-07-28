import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _load_audit_with_stubs(name):
    torch_stub = ModuleType("torch")
    torch_stub.no_grad = lambda: (lambda func: func)
    torch_stub.is_tensor = lambda obj: False
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None)
    torch_stub.manual_seed = lambda seed: None
    torch_stub.device = lambda value: value

    mmengine_stub = ModuleType("mmengine")
    mmengine_config_stub = ModuleType("mmengine.config")
    mmengine_config_stub.Config = SimpleNamespace(fromfile=lambda path: None)
    mmengine_stub.config = mmengine_config_stub

    inserted = {
        "torch": torch_stub,
        "mmengine": mmengine_stub,
        "mmengine.config": mmengine_config_stub,
        "opentad": ModuleType("opentad"),
        "opentad.datasets": ModuleType("opentad.datasets"),
        "opentad.models": ModuleType("opentad.models"),
        "opentad.models.utils": ModuleType("opentad.models.utils"),
        "opentad.models.utils.post_processing": ModuleType("opentad.models.utils.post_processing"),
    }
    inserted["opentad.datasets"].build_dataloader = lambda *args, **kwargs: None
    inserted["opentad.datasets"].build_dataset = lambda *args, **kwargs: None
    inserted["opentad.models"].build_detector = lambda *args, **kwargs: None
    inserted["opentad.models.utils.post_processing"].convert_to_seconds = lambda segments, meta, **kwargs: segments
    inserted["opentad.models.utils.post_processing"].selected_axis_to_dense_axis = (
        lambda segments, meta, **kwargs: segments
    )
    old_modules = {module_name: sys.modules.get(module_name) for module_name in inserted}
    sys.modules.update(inserted)
    try:
        module_path = ROOT / "tools/bata/audit_sparse_head_assignment.py"
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, old_value in old_modules.items():
            if old_value is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_value


def _cfg_with_axis(remap, expected):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train=SimpleNamespace(
                pipeline=[
                    {"type": "PrepareVideoInfo"},
                    {"type": "LoadFrames", "method": "random_fixed_subsample", "remap_gt_to_selected_axis": remap},
                ]
            )
        ),
        model=SimpleNamespace(rpn_head=SimpleNamespace(route_contract={"expected_axis_contract": expected})),
    )


def test_sparsehead_canonical_configs_are_native_axis_and_fail_closed():
    baseline = Config.fromfile(str(ROOT / "configs/adatad/thumos/sparsehead_irregular_bridge_k384_baseline.py"))
    repair = Config.fromfile(
        str(ROOT / "configs/adatad/thumos/sparsehead_irregular_bridge_k384_balanced_repair.py")
    )

    for cfg in (baseline, repair):
        assert cfg.model.type == "IrregularActionFormer"
        assert cfg.model.projection.type == "IrregularConvTransformerProj"
        assert cfg.model.neck.type == "IrregularFPN"
        assert cfg.model.rpn_head.type == "IrregularActionFormerBridgeHead"
        assert cfg.model.rpn_head.prior_generator.type == "IrregularPointGeneratorV2"
        assert cfg.model.rpn_head.route_contract.expected_axis_contract.gt_axis == "native"
        assert cfg.model.rpn_head.route_contract.diagnostic_only is True
        assert cfg.model.rpn_head.route_contract.primary_result_allowed is False
        assert cfg.protocol_flags.metric_claim_allowed is False
        load_frames = next(step for step in cfg.dataset.train.pipeline if step.type == "LoadFrames")
        assert load_frames.remap_gt_to_selected_axis is False

    assert baseline.model.rpn_head.allow_center_fallback_inside_gt is False
    assert repair.model.rpn_head.allow_center_fallback_inside_gt is True
    assert repair.model.rpn_head.hard_min_points_per_gt == 4
    assert repair.model.rpn_head.hard_min_points_per_level == 2
    assert repair.model.rpn_head.hard_max_points_per_gt == 10


def test_sparsehead_assignment_audit_rejects_mixed_axis_batches():
    audit = _load_audit_with_stubs("sparsehead_assignment_audit_contract")
    native = {"gt_axis": "native", "proposal_axis": "native", "nms_axis": "native", "postprocess_axis": "native"}
    selected = {
        "gt_axis": "selected",
        "proposal_axis": "selected",
        "nms_axis": "native",
        "postprocess_axis": "native",
    }
    contracts = [
        audit.same_batch_config_contract(_cfg_with_axis(False, native), "train", "native.py"),
        audit.same_batch_config_contract(_cfg_with_axis(True, selected), "train", "selected.py"),
    ]
    with pytest.raises(ValueError, match="same-batch.*incompatible.*native.py.*selected.py"):
        audit.assert_same_batch_axis_compatible(contracts)


def test_sparsehead_source_surface_keeps_assignment_and_scale_guards():
    bridge = _read("opentad/models/dense_heads/irregular_actionformer_bridge_head.py")
    audit = _read("tools/bata/audit_sparse_head_assignment.py")
    point_generator = _read("opentad/models/dense_heads/prior_generator/irregular_point_generator.py")

    assert 'allow_legacy_full_cell_span=False' in bridge
    assert 'center_radius_scale="point_radius"' in bridge
    assert 'reg_denom_mode="left_right_mean"' in bridge
    assert "def _apply_hard_gt_coverage_fallback(" in bridge
    assert "bridge_hard_gt_balance_fallback_count" in bridge
    assert "hard_assignment_under_target_gt_count" in audit
    assert "official_vs_current_assignment_diff" in audit
    assert 'dense_compat_mode != "official_actionformer"' in point_generator
    assert "requires dense-like temporal grid centers" in point_generator


def test_sparsehead_balanced_repair_assigns_multiple_points_per_gt_on_linux():
    if sys.platform.startswith("win"):
        pytest.skip("Torch DLL-backed tensor checks run in the Linux training environment.")
    torch = pytest.importorskip("torch")
    from mmengine.config import ConfigDict
    from opentad.models.dense_heads.irregular_actionformer_bridge_head import IrregularActionFormerBridgeHead

    audit_path = ROOT / "tools/bata/audit_sparse_head_assignment.py"
    spec = importlib.util.spec_from_file_location("sparsehead_assignment_audit_balance", audit_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    head = IrregularActionFormerBridgeHead(
        num_classes=5,
        in_channels=512,
        feat_channels=512,
        num_convs=1,
        assignment_mode="hard",
        regression_mode="symmetric_linear",
        prior_generator=ConfigDict(
            type="IrregularPointGeneratorV2",
            strides=[1],
            regression_range=[(0, 100)],
            range_mode="absolute",
        ),
        loss=ConfigDict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
        center_sample="radius",
        center_sample_radius=10.0,
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        allow_center_fallback_inside_gt=True,
        hard_min_points_per_gt=3,
        hard_max_points_per_gt=3,
        debug_cfg=dict(enable=True),
    )
    point = torch.tensor(
        [[float(center), 0.0, 100.0, 1.0, 1.0, 1.0, 1.0] for center in range(2, 9)],
        dtype=torch.float32,
    )
    gt_segment = torch.tensor([[0.0, 10.0], [1.0, 9.0]], dtype=torch.float32)

    diag = audit.build_hard_diagnostics(head, point, gt_segment, [(0, 7)])
    assigned = diag["assigned_gt"]
    counts = torch.bincount(assigned[assigned >= 0], minlength=2)
    assert counts.tolist() == [3, 4]
    assert diag["hard_assignment_gt_balance_fallback_count"] >= 2
    assert diag["hard_assignment_under_target_gt_count"] == 0
