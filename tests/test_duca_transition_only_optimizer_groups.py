from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("Linux remote runs Torch/OpenTAD optimizer grouping tests", allow_module_level=True)

import torch.nn as nn

from opentad.models.detectors.actionformer import ActionFormer


def _module_with_transition_selector() -> ActionFormer:
    model = ActionFormer.__new__(ActionFormer)
    nn.Module.__init__(model)
    selector = nn.Module()
    selector.selector_variant = "transition_only"
    selector.coarse_trunk_lr = 2.5e-5
    selector.action_head_lr = 5.0e-5
    selector.transition_scorer_lr = 1.0e-4
    selector.raw_actionness_source = nn.Module()
    selector.raw_actionness_source.probe_module = nn.Module()
    selector.raw_actionness_source.probe_module.spatial_stem = nn.Conv2d(3, 4, 1)
    selector.raw_actionness_source.probe_module.official_temporal = nn.Module()
    selector.raw_actionness_source.probe_module.official_temporal.encoder = nn.Module()
    selector.raw_actionness_source.probe_module.official_temporal.encoder.block = nn.Conv1d(4, 4, 1)
    selector.raw_actionness_source.probe_module.official_temporal.encoder.conv_out = nn.Conv1d(4, 2, 1)
    selector.raw_actionness_source.probe_module.official_temporal.decoders = nn.ModuleList([nn.Module()])
    selector.raw_actionness_source.probe_module.official_temporal.decoders[0].conv_out = nn.Conv1d(4, 2, 1)
    selector.adapter = nn.Module()
    selector.adapter.transition_scorer = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 1))
    model.frame_selector = selector
    model.rpn_head = nn.Linear(4, 2)
    return model


def test_transition_only_optimizer_uses_audited_component_learning_rates() -> None:
    model = _module_with_transition_selector()
    groups = model.get_optim_groups({"lr": 1.0e-4, "weight_decay": 0.05})
    lr_by_param = {
        id(param): float(group["lr"])
        for group in groups
        for param in group["params"]
    }
    named = dict(model.named_parameters())

    assert lr_by_param[id(named["frame_selector.raw_actionness_source.probe_module.spatial_stem.weight"])] == pytest.approx(2.5e-5)
    assert lr_by_param[id(named["frame_selector.raw_actionness_source.probe_module.official_temporal.encoder.block.weight"])] == pytest.approx(2.5e-5)
    assert lr_by_param[id(named["frame_selector.raw_actionness_source.probe_module.official_temporal.encoder.conv_out.weight"])] == pytest.approx(5.0e-5)
    assert lr_by_param[id(named["frame_selector.raw_actionness_source.probe_module.official_temporal.decoders.0.conv_out.weight"])] == pytest.approx(5.0e-5)
    assert lr_by_param[id(named["frame_selector.adapter.transition_scorer.1.weight"])] == pytest.approx(1.0e-4)
    assert lr_by_param[id(named["rpn_head.weight"])] == pytest.approx(1.0e-4)
