from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.config import Config

from opentad.models.duca.transition_only import DucaProtectedTransitionScorer
from opentad.models.selectors.duca_protected_e2e_frame_selector import (
    DucaProtectedE2EFrameSelector,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"


class _FakeOfficialASFormerSource(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Parameter(torch.linspace(0.5, 1.5, 96))
        self.action_head = nn.Parameter(torch.tensor(0.7))

    def forward(self, inputs, valid_mask=None):
        base = inputs.mean(dim=(1, 3, 4))
        hidden = base[:, :, None] * self.trunk[None, None, :]
        logits = hidden[:, :, 0] * self.action_head
        valid = valid_mask.bool()
        return {
            "actionness_logits": logits.masked_fill(~valid, -1.0e4),
            "p_action": torch.sigmoid(logits).masked_fill(~valid, 0.0),
            "coarse_hidden_features": hidden.masked_fill(
                ~valid[:, :, None],
                0.0,
            ),
            "hidden_kind": "official_asformer_encoder_hidden",
            "provenance": {
                "source_type": "test_official_asformer",
                "thumos_trained": True,
                "uses_labels": True,
                "uses_teacher": False,
                "uses_gt": False,
                "uses_prediction_cache": False,
            },
            "compute_profile": {"test": True},
        }


def _selector(arm: str) -> DucaProtectedE2EFrameSelector:
    selector = DucaProtectedE2EFrameSelector(
        in_channels=3,
        arm="exact_uniform",
        budget=3,
        dense_window_size=6,
    )
    selector.arm = arm
    selector.raw_actionness_source = _FakeOfficialASFormerSource()
    selector.transition_scorer = DucaProtectedTransitionScorer(96, 64)
    selector.policy_hidden_gradient_scale = 0.0
    selector.detector_bridge_gradient_scale = {
        "protected_e2e": 1.0,
        "protected_e2e_bridge025": 0.25,
        "protected_e2e_uni_companion": 0.25,
    }[arm]
    selector.uniform_companion_fraction = (
        0.50 if arm == "protected_e2e_uni_companion" else 0.0
    )
    return selector


def _batch(batch_size: int = 1):
    rows = []
    metas = []
    segments = []
    labels = []
    boundary_validity = []
    for index in range(batch_size):
        values = torch.linspace(-1.0 + index, 1.0 + index, 6)
        rows.append(values[None, :, None, None].expand(3, 6, 2, 2).clone())
        metas.append(
            {
                "video_name": f"test_video_{index}",
                "avg_fps": 10.0,
                "frame_inds": torch.arange(6, dtype=torch.long)[:, None] * 4,
            }
        )
        segments.append(torch.tensor([[1.0, 4.0]], dtype=torch.float32))
        labels.append(torch.tensor([0], dtype=torch.long))
        boundary_validity.append(torch.tensor([[True, True]]))
    inputs = torch.stack(rows)
    masks = torch.ones((batch_size, 6), dtype=torch.bool)
    return inputs, masks, metas, segments, labels, boundary_validity


def _forward(selector, batch):
    inputs, masks, metas, segments, labels, boundary_validity = batch
    return selector.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=segments,
        gt_labels=labels,
        gt_boundary_validity=boundary_validity,
    )


def _parameter_gradients(module):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def _grad_mass(parameters):
    return sum(
        0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum())
        for parameter in parameters
    )


def test_bridge025_preserves_hard_forward_and_scales_only_policy_gradient():
    torch.manual_seed(13)
    direct = _selector("protected_e2e")
    scaled = _selector("protected_e2e_bridge025")
    scaled.load_state_dict(copy.deepcopy(direct.state_dict()))
    batch = _batch()

    direct_output = _forward(direct, batch)
    scaled_output = _forward(scaled, batch)
    assert torch.equal(
        direct_output["selector_outputs"]["selected_positions"],
        scaled_output["selector_outputs"]["selected_positions"],
    )
    assert torch.equal(
        direct_output["inputs"].detach(),
        scaled_output["inputs"].detach(),
    )
    assert torch.equal(
        scaled_output["inputs"].detach(),
        scaled_output["selector_outputs"]["hard_detector_input"].detach(),
    )

    direct_output["inputs"].square().mean().backward()
    scaled_output["inputs"].square().mean().backward()
    direct_grads = _parameter_gradients(direct.transition_scorer)
    scaled_grads = _parameter_gradients(scaled.transition_scorer)
    assert direct_grads.keys() == scaled_grads.keys()
    for name in direct_grads:
        assert torch.allclose(
            scaled_grads[name],
            direct_grads[name] * 0.25,
            atol=2.0e-7,
            rtol=2.0e-5,
        ), name
    assert _grad_mass([scaled.raw_actionness_source.trunk]) == 0.0
    assert _grad_mass([scaled.raw_actionness_source.action_head]) == 0.0


def test_uni_companion_uses_one_uniform_and_one_learned_row_in_one_pass():
    torch.manual_seed(29)
    selector = _selector("protected_e2e_uni_companion")
    output = _forward(selector, _batch(batch_size=2))
    state = output["selector_outputs"]
    uniform_mask = state["uniform_companion_mask"]
    bridge_mask = state["detector_gradient_bridge_mask"]

    assert uniform_mask.dtype == torch.bool
    assert uniform_mask.tolist().count(True) == 1
    assert torch.equal(bridge_mask, ~uniform_mask)
    assert state["detector_bridge_gradient_scale"] == 0.25
    assert state["uniform_companion_fraction"] == 0.50
    uniform_row = int(torch.nonzero(uniform_mask, as_tuple=False)[0].item())
    assert state["selected_positions"][uniform_row].tolist() == [0, 2, 5]
    assert torch.equal(
        state["selected_positions"][uniform_row],
        state["uniform_companion_positions"][uniform_row],
    )
    assert torch.equal(output["inputs"].detach(), state["hard_detector_input"].detach())
    assert selector.last_forward_summary["uniform_companion_count"] == 1
    assert selector.last_forward_summary["learned_detector_count"] == 1

    output["inputs"].square().mean().backward()
    assert _grad_mass(selector.transition_scorer.parameters()) > 0.0
    assert _grad_mass([selector.raw_actionness_source.trunk]) == 0.0
    assert _grad_mass([selector.raw_actionness_source.action_head]) == 0.0


def test_uni_companion_is_training_only_and_inference_is_learned_hard_policy():
    torch.manual_seed(41)
    selector = _selector("protected_e2e_uni_companion")
    selector.eval()
    inputs, masks, metas, *_ = _batch(batch_size=2)
    output = selector.forward_test(inputs, masks, metas)
    state = output["selector_outputs"]

    assert state["detector_gradient_bridge"] is False
    assert not bool(state["uniform_companion_mask"].any().item())
    assert not bool(state["detector_gradient_bridge_mask"].any().item())
    assert "uniform_companion_positions" not in state
    assert "soft_slot_assignment" not in state
    assert torch.equal(output["inputs"], state["hard_detector_input"])


def test_new_configs_freeze_the_three_matched_learned_versions():
    direct = Config.fromfile(
        str(CONFIG_ROOT / "duca_protected_physical_e2e_fixed384_official60.py")
    )
    scaled = Config.fromfile(
        str(
            CONFIG_ROOT / "duca_protected_physical_e2e_bridge025_fixed384_official60.py"
        )
    )
    companion = Config.fromfile(
        str(
            CONFIG_ROOT
            / "duca_protected_physical_e2e_uni_companion_fixed384_official60.py"
        )
    )

    assert direct.model.frame_selector.arm == "protected_e2e"
    assert scaled.model.frame_selector.arm == "protected_e2e_bridge025"
    assert companion.model.frame_selector.arm == "protected_e2e_uni_companion"
    assert scaled.model.frame_selector.detector_bridge_gradient_scale == 0.25
    assert companion.model.frame_selector.detector_bridge_gradient_scale == 0.25
    assert companion.model.frame_selector.uniform_companion_fraction == 0.50
    for cfg in (direct, scaled, companion):
        assert cfg.workflow.end_epoch == 60
        assert cfg.workflow.primary_checkpoint_epoch == 59
        assert cfg.model.rpn_head.type == "ActionFormerHead"
        assert cfg.model.backbone.backbone.total_frames == 384
        assert cfg.dataset.to_dict() == direct.dataset.to_dict()
        assert cfg.optimizer.to_dict() == direct.optimizer.to_dict()
        assert cfg.scheduler.to_dict() == direct.scheduler.to_dict()
