from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


Reduction = Literal["none", "mean", "sum"]


def _reduce(value: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    if reduction == "sum":
        return value.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


def pinball_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    quantile: float = 0.9,
    reduction: Reduction = "mean",
) -> Tensor:
    quantile = float(quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shape")
    error = target - prediction
    value = torch.maximum(quantile * error, (quantile - 1.0) * error)
    return _reduce(value, reduction)


def nonnegative_detector_regret(
    counterfactual_loss: Tensor,
    dense_reference_loss: Tensor,
    *,
    detach_reference: bool = True,
) -> Tensor:
    reference = dense_reference_loss.detach() if detach_reference else dense_reference_loss
    return (counterfactual_loss - reference).clamp_min(0.0)


def transport_consistency_loss(
    transported: Tensor,
    reference: Tensor,
    *,
    smooth_l1_weight: float = 1.0,
    cosine_weight: float = 0.25,
    reduction: Reduction = "mean",
) -> Tensor:
    if transported.shape != reference.shape:
        raise ValueError("transported and reference tensors must have identical shape")
    if transported.ndim < 2:
        raise ValueError("transport consistency expects a feature dimension")

    smooth = F.smooth_l1_loss(transported, reference, reduction="none").mean(dim=-1)
    cosine = 1.0 - F.cosine_similarity(transported, reference, dim=-1, eps=1e-6)
    value = float(smooth_l1_weight) * smooth + float(cosine_weight) * cosine
    return _reduce(value, reduction)


@dataclass
class R2StageBLosses:
    total: Tensor
    detector: Tensor
    transport: Tensor
    risk: Tensor


def compose_r2_stage_b_loss(
    *,
    counterfactual_task_loss: Tensor,
    counterfactual_features: Tensor,
    dense_features: Tensor,
    predicted_quantile: Tensor,
    regret_target: Tensor,
) -> R2StageBLosses:
    """Frozen CT-P3R-3S-r2 Stage-B objective.

    The coefficients and component definitions are intentionally not exposed
    as arguments: changing them would define a different protocol.
    """

    if counterfactual_features.shape != dense_features.shape:
        raise ValueError("Stage-B feature tensors must have identical shape")
    if predicted_quantile.shape != regret_target.shape:
        raise ValueError("Stage-B risk prediction and regret target must have identical shape")
    detector = counterfactual_task_loss.float().mean()
    transport = F.mse_loss(
        counterfactual_features.float(), dense_features.detach().float(), reduction="mean"
    )
    risk = pinball_loss(
        predicted_quantile.float(),
        regret_target.detach().float(),
        quantile=0.9,
        reduction="mean",
    )
    total = detector + 0.1 * transport + 0.1 * risk
    return R2StageBLosses(
        total=total,
        detector=detector,
        transport=transport,
        risk=risk,
    )
