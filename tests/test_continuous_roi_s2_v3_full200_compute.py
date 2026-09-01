import ast
from pathlib import Path

from mmengine.config import Config

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    ARMS,
    SEEDS,
    config_path,
    validate_matrix,
    validate_parameter_fairness,
)


ROOT = Path(__file__).resolve().parents[1]


def test_full_matrix_and_parameter_surface_are_frozen():
    receipt = validate_matrix(ROOT)
    assert receipt["cell_count"] == 9
    assert [cell["arm"] for cell in receipt["cells"]] == [
        arm for arm in ARMS for _ in SEEDS
    ]
    assert [cell["seed"] for cell in receipt["cells"]] == [
        seed for _ in ARMS for seed in SEEDS
    ]
    assert receipt["training_identities"] == 200
    assert receipt["evaluation_videos"] == 211
    assert receipt["evaluation_ordered_windows"] == 792
    assert receipt["successful_updates_per_cell"] == 6000
    validate_parameter_fairness(ROOT)


def test_a0_crop_is_source_native_fixed_center_and_label_free():
    source_height, source_width, crop = 180, 320, 128
    x0 = (source_width - crop) // 2
    y0 = (source_height - crop) // 2
    assert [x0, y0, x0 + crop, y0 + crop] == [96, 26, 224, 154]
    cfg = Config.fromfile(config_path(ROOT, "U128-A0", 4407))
    binding = cfg.continuous_roi_s2_v3_full200_compute
    assert binding.canonical_crop_xyxy == [96, 26, 224, 154]
    assert binding.crop_policy == "source_native_fixed_center"
    pipeline_text = repr(cfg.dataset.train.pipeline).lower()
    assert "gt_segments" not in repr(
        next(step for step in cfg.dataset.train.pipeline if step.type == "NativeCropSourceViews")
    ).lower()
    assert "candidate" not in pipeline_text
    assert "metric" not in pipeline_text


def test_u128_a0_fusion_has_no_parameters():
    path = ROOT / "opentad/models/backbones/native_crop_wrapper.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fusion = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NativeCropFeatureFusion"
    )
    init = next(
        node
        for node in fusion.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assigned_attributes = {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert assigned_attributes == {"mode"}
