"""Per-FPN-point decomposition of the unchanged source eval R0 loss.

This is a diagnostic, not an instance-balanced R1 objective or router training.
Points at the same coordinate on different FPN levels remain distinct.
"""
from contextlib import contextmanager

import torch

from geosparse_research.interventions import CounterfactualRunner


@torch.no_grad()
def collect_query_risk(head, cls_pred, reg_pred, mask_list, points, gt_segments,
                       gt_labels, *, source_losses):
    """Decompose the actual head call, retaining background and invalid masks.

    Source target assignment and loss operators are reused. FP32 eval mode is
    explicit: training EMA normalization and AMP attribution are not supported.
    ``source_losses`` must be the unchanged head's result on these exact inputs.
    """
    if head.training or any(x.dtype != torch.float32 for x in cls_pred + reg_pred):
        raise ValueError("query risk requires FP32 source-head eval outputs")
    targets, offsets = head.prepare_targets(points, gt_segments, gt_labels)
    targets = torch.stack(targets)
    valid = torch.cat(mask_list, dim=1)
    positive = (targets.sum(-1) > 0) & valid
    normalizer = max(int(positive.sum()), 1)

    logits = torch.cat([x.permute(0, 2, 1) for x in cls_pred], dim=1)
    smoothed = targets[valid] * (1 - head.label_smoothing)
    smoothed = smoothed + head.label_smoothing / (head.num_classes + 1)
    focal = head.cls_loss(logits[valid], smoothed, reduction="none")
    if focal.shape != logits[valid].shape:
        raise ValueError("source classification loss lacks per-class reduction")
    cls = focal.new_zeros(valid.shape)
    cls[valid] = focal.sum(-1) / normalizer

    lengths = [x.shape[-1] for x in reg_pred]
    offsets = torch.stack(offsets).permute(0, 2, 1).split(lengths, dim=-1)
    predictions = head.get_refined_proposals(points, reg_pred)[positive]
    desired = head.get_refined_proposals(points, offsets)[positive]
    reg = cls.new_zeros(valid.shape)
    elements = head.reg_loss(predictions, desired, reduction="none") if positive.any() else reg.new_empty(0)
    if elements.shape != (int(positive.sum()),):
        raise ValueError("source regression loss lacks per-point reduction")
    raw_reg = elements.sum() / normalizer
    weight = head.loss_weight if head.loss_weight > 0 else source_losses["cls_loss"].detach() / max(float(raw_reg), 0.01)
    reg[positive] = elements / normalizer * weight

    components = {"cls_loss": cls.cpu(), "reg_loss": reg.cpu()}
    residuals = {}
    for name, values in components.items():
        expected = float(source_losses[name].detach())
        reconstructed = float(values.double().sum())
        residuals[name] = reconstructed - expected
        if not torch.isfinite(values).all() or abs(residuals[name]) > 1e-6 + 1e-5 * abs(expected):
            raise ValueError(f"per-point {name} does not reconstruct source loss")
    components["cost"] = components["cls_loss"] + components["reg_loss"]
    return dict(
        risk_name="R0_source_eval", precision="FP32_head", normalizer=normalizer,
        coordinate_system="detector grid; point columns=center, reg_min, reg_max, stride; retain FPN level",
        points=torch.cat(points).detach().cpu(),
        level=torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lengths)]),
        level_index=torch.cat([torch.arange(n) for n in lengths]),
        valid=valid.detach().cpu(), positive=positive.detach().cpu(),
        class_targets=targets.detach().cpu(), components=components,
        source_losses={key: float(value.detach()) for key, value in source_losses.items()},
        reduction_residual=residuals,
    )


@contextmanager
def _capture_head_risk(head):
    """Observe the real scalar-loss call without replacing its return values."""
    original = head.losses
    had_instance_method = "losses" in head.__dict__
    previous_instance_method = head.__dict__.get("losses")
    records = []

    def observed(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
        losses = original(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)
        records.append(collect_query_risk(head, cls_pred, reg_pred, mask_list, points,
                                          gt_segments, gt_labels, source_losses=losses))
        return losses

    head.losses = observed
    try:
        yield records
    finally:
        if had_instance_method:
            head.losses = previous_instance_method
        else:
            del head.losses


def evaluate_query_pair(detector, batch, metas, gt_segments, gt_labels, plan_a,
                        plan_b, *, require_equal_cost=True):
    """Fresh same-weight plan pair with signed risk gain at every FPN point."""
    with _capture_head_risk(detector.rpn_head) as captures:
        result = CounterfactualRunner(detector).evaluate_pair(
            batch, metas, gt_segments, gt_labels, plan_a, plan_b,
            require_equal_cost=require_equal_cost)
    if len(captures) != 2:
        raise ValueError("query attribution needs exactly one source-head loss call per plan")
    left, right = captures
    for key in ("points", "level", "level_index", "valid", "positive", "class_targets"):
        if not torch.equal(left[key], right[key]):
            raise ValueError(f"paired plans changed query geometry or targets: {key}")
    if left["normalizer"] != right["normalizer"]:
        raise ValueError("paired source normalization changed")
    result["left"]["query_risk"], result["right"]["query_risk"] = left, right
    result["signed_query_gain"] = {
        key: left["components"][key] - right["components"][key]
        for key in left["components"]
    }
    result["query_gain_reduction_residual"] = {
        key: float(value.double().sum()) - result["signed_gain"][key]
        for key, value in result["signed_query_gain"].items()
    }
    result["detector_to_input_position_scale"] = batch.frames_hi.shape[2] / len(result["left"]["detector_mask"][0])
    return result
