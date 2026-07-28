from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from opentad.models.dense_heads.actionformer_head import ActionFormerHead
from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.duca.transition_only import DucaProtectedTransitionScorer
from opentad.models.selectors.duca_protected_e2e_frame_selector import (
    _emit_protected_inference_ledger,
)
from tools.bata.finalize_duca_rime_inference_ledger import finalize_ledger
from tools.bata.run_duca_protected_physical_full_model_gate import (
    _perturb_unselected,
    _remap_gt_to_selected_axis,
    _target_assignment_parity,
)


CONTRACT = "duca_protected_e2e_physical_v1"


def test_exact_uniform_protected_inference_emits_a_no_padding_ledger(
    tmp_path,
    monkeypatch,
):
    ledger_root = tmp_path / "ledger"
    monkeypatch.setenv("DUCA_RIME_INFERENCE_LEDGER_ROOT", str(ledger_root))
    monkeypatch.setenv("RANK", "0")
    _emit_protected_inference_ledger(
        arm="exact_uniform",
        budget=4,
        metas=[
            {
                "video_name": "video_0001",
                "window_start_frame": 0,
                "irregular_dense_valid_len": 8,
                "selected_dense_indices": [0, 2, 5, 7],
                "selected_valid_len": 4,
                "duca_max_gap_seconds_cap": 0.5,
                "duca_observed_max_gap_seconds": 0.5,
            }
        ],
    )
    shard = ledger_root / "inference_ledger.rank0000.jsonl"
    row = json.loads(shard.read_text(encoding="utf-8"))
    assert row["requested_k"] == row["effective_k"] == row["backbone_input_k"] == 4
    summary = finalize_ledger(
        shards=[shard],
        output_jsonl=tmp_path / "inference_ledger.jsonl",
        expected_arm="exact_uniform",
    )
    assert summary["no_padding_ledger"] is True


def test_real_loader_uint8_unselected_perturbation_preserves_hard_gather():
    inputs = torch.arange(18, dtype=torch.uint8).reshape(1, 3, 6, 1, 1)
    positions = torch.tensor([[0, 2, 5]], dtype=torch.long)
    perturbed = _perturb_unselected(inputs, positions)

    assert torch.equal(perturbed[:, :, [0, 2, 5]], inputs[:, :, [0, 2, 5]])
    assert torch.equal(
        perturbed[:, :, [1, 3, 4]],
        torch.bitwise_xor(
            inputs[:, :, [1, 3, 4]],
            torch.full_like(inputs[:, :, [1, 3, 4]], 0xFF),
        ),
    )


def _make_head() -> ActionFormerHead:
    head = ActionFormerHead(
        num_classes=2,
        in_channels=2,
        feat_channels=2,
        num_convs=0,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1],
            regression_range=[(0, 10000)],
        ),
        loss=SimpleNamespace(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            contract=CONTRACT,
        ),
    )
    with torch.no_grad():
        head.cls_head.weight.zero_()
        head.cls_head.bias.zero_()
        head.reg_head.weight.zero_()
        head.reg_head.bias.zero_()
    return head


def _protected_meta(**updates):
    meta = {
        "duca_contract": CONTRACT,
        "physical_grid_contract": CONTRACT,
        "irregular_selected_positions": [0, 2, 5],
        "selected_dense_indices": [0, 2, 5],
        "selected_valid_len": 3,
        "irregular_selected_count": 3,
        "irregular_dense_valid_len": 6,
        "irregular_native_axis": True,
        "remap_gt_to_selected_axis": False,
        "gt_remapped_to_selected_axis": False,
        "pc_ot_mras_prebackbone_remap_gt_to_selected_axis": False,
        "selected_axis_remap_required": False,
        "detector_prediction_inverse_map_required": False,
        "detector_output_coordinate_space": "dense_physical",
        "proposal_axis": "dense_physical",
        "duca_backbone_tail_padding_mode": "replicate_last_selected",
    }
    meta.update(updates)
    return meta


def test_protected_head_decodes_hard_slots_on_dense_physical_axis():
    head = _make_head()
    features = [torch.zeros(1, 2, 4)]
    masks = [torch.tensor([[True, True, True, False]])]
    proposals, scores = head.forward_test(
        features,
        masks,
        metas=[_protected_meta()],
    )

    assert torch.equal(proposals[0][:, 0], torch.tensor([0.0, 2.0, 5.0]))
    assert torch.equal(proposals[0][:, 1], torch.tensor([0.0, 2.0, 5.0]))
    assert proposals[0].shape == (3, 2)
    assert scores[0].shape == (3, 2)


def test_exact_uniform_explicit_target_assignment_matches_selected_axis():
    head = _make_head()
    features = [torch.zeros(1, 2, 4)]
    masks = [torch.tensor([[True, True, True, False]])]
    positions = torch.tensor([[0, 2, 5, -1]], dtype=torch.long)
    dense_masks = torch.ones((1, 6), dtype=torch.bool)
    physical_segments = [torch.tensor([[0.5, 5.5]], dtype=torch.float32)]
    labels = [torch.tensor([1], dtype=torch.long)]
    legacy_segments = _remap_gt_to_selected_axis(
        physical_segments,
        positions,
        dense_masks,
    )

    report = _target_assignment_parity(
        head,
        base_points=head.prior_generator(features),
        base_masks=masks,
        physical_metas=[_protected_meta()],
        dense_masks=dense_masks,
        positions=positions,
        physical_segments=physical_segments,
        legacy_segments=legacy_segments,
        gt_labels=labels,
    )

    assert report["classification_targets_equal"] is True
    assert report["positive_masks_equal"] is True
    assert report["physical_regression_targets_equal"] is True
    assert report["positive_count"] > 0


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"irregular_selected_positions": [0.0, 2.0, 5.0]},
            "integer tensors",
        ),
        (
            {"selected_valid_len": 2},
            "selected-count fields disagree",
        ),
        (
            {"selected_dense_indices": [0, 3, 5]},
            "positions disagree",
        ),
        (
            {"irregular_selected_positions": [0, 5, 2], "selected_dense_indices": [0, 5, 2]},
            "unique and increasing",
        ),
        (
            {"detector_output_coordinate_space": "selected_axis"},
            "contract mismatch",
        ),
    ],
)
def test_protected_head_fails_closed_on_coordinate_contract_errors(
    updates,
    message,
):
    head = _make_head()
    features = [torch.zeros(1, 2, 4)]
    masks = [torch.tensor([[True, True, True, False]])]
    with pytest.raises(ValueError, match=message):
        head.forward_test(features, masks, metas=[_protected_meta(**updates)])


def test_protected_head_rejects_mask_and_selected_count_disagreement():
    head = _make_head()
    features = [torch.zeros(1, 2, 4)]
    masks = [torch.tensor([[True, True, False, False]])]
    with pytest.raises(ValueError, match="first detector mask count"):
        head.forward_test(features, masks, metas=[_protected_meta()])


class _FakeOfficialTemporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.conv_out = nn.Conv1d(4, 2, kernel_size=1)
        self.encoder.layers = nn.ModuleList([nn.Conv1d(4, 4, kernel_size=1)])
        decoder = nn.Module()
        decoder.conv_out = nn.Conv1d(4, 2, kernel_size=1)
        decoder.layers = nn.ModuleList([nn.Conv1d(4, 4, kernel_size=1)])
        self.decoders = nn.ModuleList([decoder])


class _FakeProbeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
        )
        self.official_temporal = _FakeOfficialTemporal()


class _FakeActionnessSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe_module = _FakeProbeModule()


class _FakeProtectedSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.selector_variant = "protected_e2e_physical"
        self.selector_lr = 1.0e-4
        self.coarse_trunk_lr = 2.5e-5
        self.action_head_lr = 5.0e-5
        self.budget = 3
        self.separate_detector_rng = True
        self.transition_scorer = DucaProtectedTransitionScorer(96, 64)
        self.raw_actionness_source = _FakeActionnessSource()


def _make_optimizer_probe_model() -> ActionFormer:
    model = ActionFormer.__new__(ActionFormer)
    nn.Module.__init__(model)
    model.frame_selector = _FakeProtectedSelector()
    model.projection_probe = nn.Conv1d(4, 4, kernel_size=1)
    return model


def test_optimizer_covers_every_protected_parameter_with_frozen_learning_rates():
    model = _make_optimizer_probe_model()
    groups = model.get_optim_groups(
        dict(lr=2.0e-4, weight_decay=0.05)
    )
    id_to_lr = {
        id(parameter): float(group["lr"])
        for group in groups
        for parameter in group["params"]
    }
    named = dict(model.named_parameters())

    assert set(id_to_lr) == {id(parameter) for parameter in named.values()}
    for name, parameter in named.items():
        if name.startswith(
            (
                "frame_selector.transition_scorer.selector_adapter.",
                "frame_selector.transition_scorer.selector_score_head.",
            )
        ):
            expected = 1.0e-4
        elif ".official_temporal.encoder.conv_out." in name:
            expected = 5.0e-5
        elif ".official_temporal.decoders.0.conv_out." in name:
            expected = 5.0e-5
        elif name.startswith(
            "frame_selector.raw_actionness_source.probe_module."
        ):
            expected = 2.5e-5
        else:
            expected = 2.0e-4
        assert id_to_lr[id(parameter)] == expected, name


def test_detector_rng_stream_is_unchanged_by_selector_randomness():
    model = _make_optimizer_probe_model()
    inputs = torch.zeros(1, 3, 6, 2, 2)
    torch.manual_seed(91)
    random.seed(92)
    np.random.seed(93)
    state = model._capture_protected_detector_rng(inputs)

    expected = (
        torch.rand(4),
        random.random(),
        np.random.rand(4),
    )
    _ = torch.rand(19)
    _ = [random.random() for _index in range(7)]
    _ = np.random.rand(11)
    model._restore_protected_detector_rng(state)
    observed = (
        torch.rand(4),
        random.random(),
        np.random.rand(4),
    )

    assert torch.equal(observed[0], expected[0])
    assert observed[1] == expected[1]
    assert np.array_equal(observed[2], expected[2])


def test_actionformer_contract_requires_dense_gt_identity_and_exact_mask_count():
    model = _make_optimizer_probe_model()
    inputs = torch.zeros(1, 3, 3, 2, 2)
    masks = torch.ones(1, 3, dtype=torch.bool)
    metas = [_protected_meta()]
    segments = [torch.tensor([[1.0, 4.0]])]
    labels = [torch.tensor([0])]

    model._validate_protected_selector_contract(
        inputs,
        masks,
        metas,
        gt_segments=segments,
        gt_labels=labels,
        original_gt_segments=segments,
        original_gt_labels=labels,
    )

    with pytest.raises(RuntimeError, match="preserve dense GT segment objects"):
        model._validate_protected_selector_contract(
            inputs,
            masks,
            metas,
            gt_segments=list(segments),
            gt_labels=labels,
            original_gt_segments=segments,
            original_gt_labels=labels,
        )
