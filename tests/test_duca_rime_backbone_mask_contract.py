from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.detectors.tridet import TriDet


class _RecordingBackbone(nn.Module):
    def __init__(self, *, dynamic_temporal_bucket: bool):
        super().__init__()
        self.dynamic_temporal_bucket = bool(dynamic_temporal_bucket)
        self.received_masks = "not-called"

    def forward(self, inputs, masks=None):
        self.received_masks = masks
        return inputs


def _detector_shell(detector_type, *, dynamic_temporal_bucket, selector_variant=None):
    detector = detector_type.__new__(detector_type)
    nn.Module.__init__(detector)
    detector.backbone = _RecordingBackbone(
        dynamic_temporal_bucket=dynamic_temporal_bucket
    )
    detector.frame_selector = (
        None
        if selector_variant is None
        else SimpleNamespace(selector_variant=selector_variant)
    )
    return detector


@pytest.mark.parametrize("detector_type", [ActionFormer, TriDet])
def test_dynamic_temporal_backbone_receives_the_exact_mask(detector_type):
    detector = _detector_shell(
        detector_type,
        dynamic_temporal_bucket=True,
    )
    inputs = torch.zeros((2, 1, 3, 32, 2, 2))
    masks = torch.ones((2, 32), dtype=torch.bool)

    output = detector._forward_backbone_with_temporal_mask(inputs, masks)

    assert output is inputs
    assert detector.backbone.received_masks is masks


@pytest.mark.parametrize("detector_type", [ActionFormer, TriDet])
def test_non_dynamic_backbone_preserves_the_legacy_call_contract(detector_type):
    detector = _detector_shell(
        detector_type,
        dynamic_temporal_bucket=False,
    )
    inputs = torch.zeros((2, 4, 8))
    masks = torch.ones((2, 8), dtype=torch.bool)

    output = detector._forward_backbone_with_temporal_mask(inputs, masks)

    assert output is inputs
    assert detector.backbone.received_masks is None


@pytest.mark.parametrize("detector_type", [ActionFormer, TriDet])
def test_rime_selector_rejects_a_non_dynamic_backbone(detector_type):
    detector = _detector_shell(
        detector_type,
        dynamic_temporal_bucket=False,
        selector_variant="duca_rime_physical",
    )

    with pytest.raises(
        RuntimeError,
        match="requires a dynamic temporal backbone",
    ):
        detector._forward_backbone_with_temporal_mask(
            torch.zeros((1, 1, 3, 32, 2, 2)),
            torch.ones((1, 32), dtype=torch.bool),
        )


def _assert_method_routes_aligned_mask(source_path, class_name, method_name):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    matching_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_forward_backbone_with_temporal_mask"
        and [
            argument.id if isinstance(argument, ast.Name) else None
            for argument in node.args
        ]
        == ["inputs", "masks"]
    ]
    assert len(matching_calls) == 1


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    [
        ("opentad/models/detectors/actionformer.py", "ActionFormer"),
        ("opentad/models/detectors/tridet.py", "TriDet"),
    ],
)
@pytest.mark.parametrize("method_name", ["forward_train", "forward_test"])
def test_actionformer_and_tridet_use_the_shared_backbone_handoff(
    module_path,
    class_name,
    method_name,
):
    root = Path(__file__).resolve().parents[1]
    _assert_method_routes_aligned_mask(
        root / module_path,
        class_name,
        method_name,
    )
