import copy

import pytest

torch = pytest.importorskip("torch")

from opentad.models import build_detector
from opentad.models.utils.post_processing import convert_to_seconds


def _model():
    return build_detector(
        dict(
            type="PhysTimeTAD",
            discretization_loss_weight=0.2,
            projection=dict(
                type="PhysTimeMeasureProjection",
                in_channels=4,
                out_channels=8,
                attention_channels=8,
                base_spacing_sec=1.0,
                num_levels=2,
            ),
            rpn_head=dict(
                type="PhysTimeHead",
                num_classes=2,
                in_channels=8,
                feat_channels=8,
                num_convs=1,
                regression_ranges_sec=[(0.0, 4.0), (4.0, 100.0)],
                loss_normalizer=4.0,
                endpoint_loss_weight=0.25,
                loss=dict(
                    cls_loss=dict(type="FocalLoss"),
                    reg_loss=dict(type="DIOULoss"),
                ),
            ),
        )
    )


def _batch(feature_shift=0.0):
    inputs = torch.randn(1, 4, 3) + feature_shift
    masks = torch.tensor([[True, True, True]])
    metas = [
        {
            "video_name": "video_test",
            "duration": 4.0,
            "fps": 4.0,
            "snippet_stride": 4,
            "offset_frames": 0,
            "phystime_timestamps_sec": [0.5, 1.5, 3.5],
            "phystime_support_intervals_sec": [[0.0, 1.0], [1.0, 2.0], [3.0, 4.0]],
            "phystime_duration_sec": 4.0,
            "phystime_domain_start_sec": 0.0,
            "phystime_domain_end_sec": 4.0,
            "phystime_support_provenance": "synthetic_explicit_support",
            "gt_time_unit": "seconds",
            "prediction_time_unit": "seconds",
            "irregular_native_axis": True,
            "remap_gt_to_selected_axis": False,
            "gt_remapped_to_selected_axis": False,
        }
    ]
    return inputs, masks, metas


def test_registry_model_one_step_has_single_cost_and_full_gradient_path():
    model = _model().train()
    inputs, masks, metas = _batch()
    losses = model.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=[torch.tensor([[0.5, 3.5]])],
        gt_labels=[torch.tensor([1])],
    )

    expected_cost = sum(value for key, value in losses.items() if key != "cost")
    assert torch.allclose(losses["cost"], expected_cost)
    losses["cost"].backward()

    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.projection.level_attentions[0].value_proj.weight.grad.abs().sum().item() > 0
    assert model.projection.level_attentions[0].key_proj.weight.grad is not None
    assert model.rpn_head.cls_head.weight.grad.abs().sum().item() > 0
    assert model.rpn_head.reg_head.weight.grad.abs().sum().item() > 0
    assert model.rpn_head.endpoint_head.weight.grad.abs().sum().item() > 0


def test_optimizer_groups_cover_every_trainable_non_backbone_parameter_once():
    model = _model()
    groups = model.get_optim_groups(dict(lr=1.0e-4, weight_decay=0.05))
    grouped = [parameter for group in groups for parameter in group["params"]]
    expected = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]

    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in expected}


def test_paired_views_add_common_coverage_discretization_loss():
    model = _model().train()
    inputs, masks, metas = _batch()
    paired_inputs, paired_masks, paired_metas = _batch(feature_shift=0.5)

    losses = model.forward_train(
        inputs,
        masks,
        metas,
        gt_segments=[torch.tensor([[0.5, 3.5]])],
        gt_labels=[torch.tensor([1])],
        paired_inputs=paired_inputs,
        paired_masks=paired_masks,
        paired_metas=paired_metas,
    )

    assert "discretization_loss" in losses
    assert torch.isfinite(losses["discretization_loss"])
    assert losses["discretization_loss"].item() >= 0
    losses["cost"].backward()
    assert model.projection.level_attentions[0].value_proj.weight.grad is not None


def test_discretization_consistency_weights_errors_by_common_physical_coverage():
    model = _model()
    points = (torch.tensor([[[0.5, 0.0, 4.0, 1.0], [1.5, 0.0, 4.0, 1.0]]]),)

    def raw(second_logits):
        return {
            "mask": torch.tensor([[True, True]]),
            "points": points,
            "cls_logits": (torch.tensor([[second_logits]], dtype=torch.float32),),
            "endpoint_probabilities": (torch.zeros(1, 2, 2),),
            "proposals_sec": torch.zeros(1, 2, 2),
            "cell_widths_sec": torch.ones(1, 2),
            "coverage_sec": torch.tensor([[1.0, 0.1]]),
        }

    reference = raw([0.0, 0.0])
    high_coverage_error = model._common_coverage_consistency(reference, raw([8.0, 0.0]))
    low_coverage_error = model._common_coverage_consistency(reference, raw([0.0, 8.0]))

    assert high_coverage_error > low_coverage_error * 5.0


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"gt_time_unit": "selected_index"}, "ground truth in absolute seconds"),
        ({"remap_gt_to_selected_axis": True}, "selected-axis"),
        ({"teacher_prediction_cache": "forbidden.pkl"}, "forbidden metadata"),
    ],
)
def test_training_fails_closed_on_unit_remap_or_leakage(mutation, message):
    model = _model().train()
    inputs, masks, metas = _batch()
    metas = copy.deepcopy(metas)
    metas[0].update(mutation)

    with pytest.raises(ValueError, match=message):
        model.forward_train(
            inputs,
            masks,
            metas,
            gt_segments=[torch.tensor([[0.5, 3.5]])],
            gt_labels=[torch.tensor([1])],
        )


def test_forward_test_rejects_leakage_and_returns_seconds_predictions():
    model = _model().eval()
    inputs, masks, metas = _batch()
    proposals, scores = model.forward_test(inputs, masks, metas)

    assert proposals[0].ndim == 2 and proposals[0].shape[-1] == 2
    assert scores[0].shape[-1] == 2
    assert torch.all(proposals[0][:, 1] >= proposals[0][:, 0])

    leaking = copy.deepcopy(metas)
    leaking[0]["offline_ledger"] = "forbidden.jsonl"
    with pytest.raises(ValueError, match="forbidden metadata"):
        model.forward_test(inputs, masks, leaking)


def test_seconds_post_processing_is_identity_except_duration_clamp():
    segments = torch.tensor([[-1.0, 2.0], [3.0, 8.0]])
    meta = {
        "prediction_time_unit": "seconds",
        "duration": 5.0,
        "fps": 4.0,
        "snippet_stride": 4,
        "offset_frames": 0,
    }

    converted = convert_to_seconds(segments.clone(), meta)

    assert torch.allclose(converted, torch.tensor([[0.0, 2.0], [3.0, 5.0]]))
