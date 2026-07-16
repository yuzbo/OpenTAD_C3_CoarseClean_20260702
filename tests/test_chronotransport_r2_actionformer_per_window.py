from __future__ import annotations

import types

import pytest
import torch
from mmengine.config import ConfigDict

from opentad.models import build_detector
from opentad.models.detectors.actionformer import ActionFormerPerWindowTrainOutput


def _real_tiny_actionformer():
    return build_detector(
        dict(
            type="ActionFormer",
            projection=dict(
                type="Conv1DTransformerProj",
                in_channels=8,
                out_channels=8,
                arch=(1, 0, 1),
                conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
                norm_cfg=dict(type="LN"),
                attn_cfg=dict(n_head=2, n_mha_win_size=1),
                path_pdrop=0.0,
                use_abs_pe=False,
                max_seq_len=16,
            ),
            rpn_head=dict(
                type="ActionFormerHead",
                num_classes=2,
                in_channels=8,
                feat_channels=8,
                num_convs=1,
                cls_prior_prob=0.01,
                prior_generator=dict(
                    type="PointGenerator",
                    strides=[1, 2],
                    regression_range=[(0, 4), (4, 10000)],
                ),
                loss_normalizer=10,
                loss_normalizer_momentum=0.9,
                center_sample="radius",
                center_sample_radius=1.5,
                label_smoothing=0.0,
                loss=ConfigDict(
                    cls_loss=dict(type="FocalLoss"),
                    reg_loss=dict(type="DIOULoss"),
                ),
            ),
        )
    )


def _batch():
    inputs = torch.randn(2, 8, 16, requires_grad=True)
    masks = torch.ones(2, 16, dtype=torch.bool)
    metas = [{"window_id": "fit-000"}, {"window_id": "fit-001"}]
    gt_segments = [
        torch.tensor([[1.0, 5.0]]),
        torch.tensor([[7.0, 15.0], [2.0, 4.0]]),
    ]
    gt_labels = [torch.tensor([0]), torch.tensor([1, 0])]
    return inputs, masks, metas, gt_segments, gt_labels


def test_real_actionformer_batch_two_exposes_one_same_head_loss_vector(monkeypatch):
    torch.manual_seed(3407)
    model = _real_tiny_actionformer().train()
    inputs, masks, metas, gt_segments, gt_labels = _batch()
    calls = []
    original = model.rpn_head.forward_train

    def counted(_self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        model.rpn_head,
        "forward_train",
        types.MethodType(counted, model.rpn_head),
    )
    output = model(
        inputs=inputs,
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
        chronotransport_per_window_output=True,
    )

    assert isinstance(output, ActionFormerPerWindowTrainOutput)
    assert len(calls) == 1
    assert calls[0][1]["return_per_window_task_loss"] is True
    assert output.per_window_task_loss.shape == (2,)
    assert output.per_window_task_loss.requires_grad
    assert not torch.equal(
        output.per_window_task_loss[0], output.per_window_task_loss[1]
    )
    assert torch.equal(
        output.loss_dict["cost"], output.per_window_task_loss.sum()
    )
    torch.testing.assert_close(
        output.loss_dict["cost"],
        output.loss_dict["cls_loss"] + output.loss_dict["reg_loss"],
        rtol=8 * torch.finfo(output.loss_dict["cost"].dtype).eps,
        atol=0.0,
    )
    output.loss_dict["cost"].backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.projection.parameters()
    )


def test_real_actionformer_dense_reference_uses_the_same_structured_api():
    torch.manual_seed(3407)
    model = _real_tiny_actionformer().train()
    batch = _batch()
    with torch.no_grad():
        output = model(
            inputs=batch[0],
            masks=batch[1],
            metas=batch[2],
            gt_segments=batch[3],
            gt_labels=batch[4],
            return_loss=True,
            chronotransport_per_window_output=True,
        )
    assert isinstance(output, ActionFormerPerWindowTrainOutput)
    assert output.per_window_task_loss.shape == (2,)
    assert not output.per_window_task_loss.requires_grad
    assert torch.equal(
        output.loss_dict["cost"], output.per_window_task_loss.sum()
    )


def test_structured_vector_preserves_the_unmodified_actionformer_reduction():
    torch.manual_seed(3407)
    aggregate_model = _real_tiny_actionformer().train()
    vector_model = _real_tiny_actionformer().train()
    vector_model.load_state_dict(aggregate_model.state_dict(), strict=True)
    batch = _batch()

    aggregate = aggregate_model(
        inputs=batch[0],
        masks=batch[1],
        metas=batch[2],
        gt_segments=batch[3],
        gt_labels=batch[4],
        return_loss=True,
    )
    vector = vector_model(
        inputs=batch[0],
        masks=batch[1],
        metas=batch[2],
        gt_segments=batch[3],
        gt_labels=batch[4],
        return_loss=True,
        chronotransport_per_window_output=True,
    )

    tolerance = 8 * torch.finfo(vector.loss_dict["cost"].dtype).eps
    for key in ("cls_loss", "reg_loss", "cost"):
        torch.testing.assert_close(
            vector.loss_dict[key], aggregate[key], rtol=tolerance, atol=0.0
        )
    assert torch.equal(
        vector_model.rpn_head.loss_normalizer,
        aggregate_model.rpn_head.loss_normalizer,
    )


def test_per_window_actionformer_rejects_any_caller_loss_or_target_surface():
    model = _real_tiny_actionformer().train()
    batch = _batch()
    with pytest.raises(TypeError, match="caller-supplied loss inputs"):
        model(
            inputs=batch[0],
            masks=batch[1],
            metas=batch[2],
            gt_segments=batch[3],
            gt_labels=batch[4],
            return_loss=True,
            chronotransport_per_window_output=True,
            regret_target=torch.zeros(2),
        )
