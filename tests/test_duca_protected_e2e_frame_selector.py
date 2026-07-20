from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from opentad.datasets.transforms.end_to_end import LoadFrames
from opentad.models.selectors.duca_protected_e2e_frame_selector import (
    DucaProtectedE2EFrameSelector,
    _transition_target_from_gt_segments,
)


class _FakeOfficialASFormerSource(nn.Module):
    def __init__(self, *, restricted_policy_hidden: bool = False) -> None:
        super().__init__()
        self.trunk = nn.Parameter(torch.linspace(0.5, 1.5, 96))
        self.action_head = nn.Parameter(torch.tensor(0.7))
        self.restricted_policy_hidden = bool(restricted_policy_hidden)

    def forward(self, inputs, valid_mask=None):
        base = inputs.mean(dim=(1, 3, 4))
        hidden = base[:, :, None] * self.trunk[None, None, :]
        logits = hidden[:, :, 0] * self.action_head
        valid = valid_mask.bool()
        output = {
            "actionness_logits": logits.masked_fill(~valid, -1.0e4),
            "p_action": torch.sigmoid(logits).masked_fill(~valid, 0.0),
            "coarse_hidden_features": hidden.masked_fill(~valid[:, :, None], 0.0),
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
        if self.restricted_policy_hidden:
            output["policy_hidden_features"] = hidden.detach() + (
                hidden - hidden.detach()
            )
        return output


def _batch():
    values = torch.linspace(-1.0, 1.0, 6)
    inputs = values[None, None, :, None, None].expand(1, 3, 6, 2, 2).clone()
    masks = torch.ones((1, 6), dtype=torch.bool)
    metas = [
        {
            "video_name": "test_video",
            "avg_fps": 10.0,
            "frame_inds": torch.arange(6, dtype=torch.long)[:, None] * 4,
        }
    ]
    segments = [torch.tensor([[1.0, 4.0]], dtype=torch.float32)]
    labels = [torch.tensor([0], dtype=torch.long)]
    boundary_validity = [torch.tensor([[True, True]])]
    return inputs, masks, metas, segments, labels, boundary_validity


def _selector(arm: str):
    selector = DucaProtectedE2EFrameSelector(
        in_channels=3,
        arm="exact_uniform",
        budget=3,
        dense_window_size=6,
    )
    if arm != "exact_uniform":
        selector.arm = arm
        selector.raw_actionness_source = _FakeOfficialASFormerSource(
            restricted_policy_hidden=arm == "protected_e2e_rho001"
        )
        from opentad.models.duca.transition_only import DucaProtectedTransitionScorer

        selector.transition_scorer = DucaProtectedTransitionScorer(96, 64)
        selector.policy_hidden_gradient_scale = (
            0.01 if arm == "protected_e2e_rho001" else 0.0
        )
    return selector


def _zero_grad(module):
    for parameter in module.parameters():
        parameter.grad = None


def _grad_mass(parameters):
    return sum(
        0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum())
        for parameter in parameters
    )


def test_exact_uniform_skips_coarse_and_writes_native_physical_contract():
    selector = _selector("exact_uniform")
    inputs, masks, metas, segments, labels, boundary_validity = _batch()
    output = selector.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=segments,
        gt_labels=labels,
        gt_boundary_validity=boundary_validity,
    )

    assert selector.raw_actionness_source is None
    assert output["losses"] == {}
    assert output["selector_outputs"]["selected_positions"].tolist() == [[0, 2, 5]]
    assert torch.equal(output["inputs"], inputs[:, :, [0, 2, 5]])
    assert output["gt_segments"] is segments
    assert output["gt_labels"] is labels
    meta = output["metas"][0]
    assert meta["remap_gt_to_selected_axis"] is False
    assert meta["gt_remapped_to_selected_axis"] is False
    assert meta["detector_prediction_inverse_map_required"] is False
    assert meta["detector_output_coordinate_space"] == "dense_physical"
    assert meta["irregular_selected_positions"] == [0, 2, 5]
    assert meta["duca_observed_max_gap_seconds"] <= meta["duca_max_gap_seconds_cap"]


def test_short_window_replicates_last_selected_frame_for_backbone_tail():
    selector = DucaProtectedE2EFrameSelector(
        in_channels=3,
        arm="exact_uniform",
        budget=4,
        dense_window_size=6,
    )
    values = torch.arange(6, dtype=torch.float32)
    inputs = values[None, None, :, None, None].expand(1, 3, 6, 1, 1).clone()
    masks = torch.tensor([[True, True, True, False, False, False]])
    metas = [
        {
            "video_name": "short_video",
            "avg_fps": 10.0,
            "frame_inds": torch.arange(6, dtype=torch.long)[:, None],
        }
    ]
    output = selector.forward_test(inputs, masks, metas)

    assert output["selector_outputs"]["selected_positions"].tolist() == [
        [0, 1, 2, -1]
    ]
    assert output["masks"].tolist() == [[True, True, True, False]]
    assert output["inputs"][0, 0, :, 0, 0].tolist() == [0.0, 1.0, 2.0, 2.0]
    assert (
        output["selector_outputs"]["backbone_tail_padding_mode"]
        == "replicate_last_selected"
    )
    assert (
        output["metas"][0]["duca_backbone_tail_padding_mode"]
        == "replicate_last_selected"
    )


def test_main_detector_gradient_stops_at_selector_adapter_head():
    selector = _selector("protected_e2e")
    inputs, masks, metas, segments, labels, boundary_validity = _batch()
    output = selector.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=segments,
        gt_labels=labels,
        gt_boundary_validity=boundary_validity,
    )
    detector_proxy = output["inputs"].square().mean()
    detector_proxy.backward()

    source = selector.raw_actionness_source
    assert _grad_mass(selector.transition_scorer.parameters()) > 0.0
    assert _grad_mass([source.trunk]) == 0.0
    assert _grad_mass([source.action_head]) == 0.0
    assert torch.equal(
        output["inputs"].detach(),
        output["selector_outputs"]["hard_detector_input"].detach(),
    )


def test_auxiliary_loss_gradient_ownership_is_separated():
    selector = _selector("protected_e2e")
    inputs, masks, metas, segments, labels, boundary_validity = _batch()
    output = selector.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=segments,
        gt_labels=labels,
        gt_boundary_validity=boundary_validity,
    )
    source = selector.raw_actionness_source

    output["losses"]["selector_action_loss"].backward(retain_graph=True)
    assert _grad_mass([source.trunk]) > 0.0
    assert _grad_mass([source.action_head]) > 0.0
    assert _grad_mass(selector.transition_scorer.parameters()) == 0.0

    _zero_grad(selector)
    (
        output["losses"]["selector_transition_loss"]
        + output["losses"]["selector_transition_boundary_loss"]
    ).backward()
    assert _grad_mass([source.trunk]) > 0.0
    assert _grad_mass([source.action_head]) == 0.0
    assert _grad_mass(selector.transition_scorer.parameters()) > 0.0


def test_rho001_detector_gradient_reaches_only_restricted_hidden_route():
    selector = _selector("protected_e2e_rho001")
    inputs, masks, metas, segments, labels, boundary_validity = _batch()
    output = selector.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=segments,
        gt_labels=labels,
        gt_boundary_validity=boundary_validity,
    )
    output["inputs"].square().mean().backward()

    source = selector.raw_actionness_source
    assert _grad_mass(selector.transition_scorer.parameters()) > 0.0
    assert _grad_mass([source.trunk]) > 0.0
    assert _grad_mass([source.action_head]) == 0.0


def test_inference_uses_hard_viterbi_without_soft_assignment():
    selector = _selector("protected_e2e")
    selector.eval()
    inputs, masks, metas, _segments, _labels, _boundary_validity = _batch()
    output = selector.forward_test(inputs, masks, metas)

    assert output["selector_outputs"]["detector_gradient_bridge"] is False
    assert "soft_slot_assignment" not in output["selector_outputs"]
    assert torch.equal(
        output["inputs"],
        output["selector_outputs"]["hard_detector_input"],
    )


def test_transition_target_ignores_crop_created_pseudo_boundary():
    valid = torch.ones((1, 6), dtype=torch.bool)
    segments = [torch.tensor([[0.0, 4.0]], dtype=torch.float32)]
    target = _transition_target_from_gt_segments(
        segments,
        valid,
        sigma=0.5,
        radius=1,
        boundary_validity=[torch.tensor([[False, True]])],
    )
    assert float(target[0, 0]) == 0.0
    assert float(target[0, 4]) > 0.0
    assert torch.isclose(target.sum(), torch.tensor(1.0))


def test_load_frames_reports_true_boundary_validity(monkeypatch):
    loader = LoadFrames.__new__(LoadFrames)
    loader.crop_ratio = None
    loader.trunc_thresh = 0.5
    monkeypatch.setattr(
        "opentad.datasets.transforms.end_to_end.random.randint",
        lambda _start, _end: 2,
    )
    feats = np.arange(10)
    segments = np.asarray([[0.0, 5.0], [3.0, 6.0], [5.0, 9.0]])
    labels = np.asarray([0, 1, 2])
    _, clipped, kept_labels, validity = loader.random_trunc(
        feats,
        5,
        segments,
        labels,
        return_boundary_validity=True,
    )
    assert clipped.tolist() == [[0.0, 3.0], [1.0, 4.0], [3.0, 5.0]]
    assert kept_labels.tolist() == [0, 1, 2]
    assert validity.tolist() == [
        [False, True],
        [True, True],
        [True, False],
    ]
