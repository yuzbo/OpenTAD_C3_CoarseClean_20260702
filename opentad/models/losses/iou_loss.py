import torch
import torch.nn as nn
from ..builder import LOSSES
from ..utils.iou_tools import compute_diou_torch, compute_giou_torch


@LOSSES.register_module()
class DIOULoss(nn.Module):
    def __init__(self):
        super(DIOULoss, self).__init__()

    def forward(
        self,
        input_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        loss = 1 - torch.diag(compute_diou_torch(target_bboxes, input_bboxes))

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()

        return loss


@LOSSES.register_module()
class GIOULoss(nn.Module):
    def __init__(self):
        super(GIOULoss, self).__init__()

    def forward(
        self,
        input_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        loss = 1 - torch.diag(compute_giou_torch(target_bboxes, input_bboxes))

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()
        return loss


@LOSSES.register_module()
class ContinuousPhysicalGIoULoss(nn.Module):
    """Continuous Physical Time Generalized IoU Loss for Temporal Action Detection.

    Directly optimizes continuous physical time segments [start, end] in physical seconds
    or timestamps, penalizing scale-dependent duration errors and centroid shifts.

    Args:
        loss_weight (float): Loss weight factor. Default: 1.0.
        center_weight (float): Centroid distance penalty factor. Default: 1.0.
        eps (float): Small epsilon for numerical stability. Default: 1e-7.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        center_weight: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.center_weight = float(center_weight)
        self.eps = float(eps)

    def forward(
        self,
        input_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if input_bboxes.numel() == 0:
            return input_bboxes.sum() * 0.0

        if input_bboxes.ndim == 1:
            input_bboxes = input_bboxes.unsqueeze(0)
        if target_bboxes.ndim == 1:
            target_bboxes = target_bboxes.unsqueeze(0)

        pred_s, pred_e = input_bboxes[:, 0], input_bboxes[:, 1]
        gt_s, gt_e = target_bboxes[:, 0], target_bboxes[:, 1]

        # Valid intervals
        pred_len = (pred_e - pred_s).clamp_min(self.eps)
        gt_len = (gt_e - gt_s).clamp_min(self.eps)

        # Intersection
        inter_s = torch.max(pred_s, gt_s)
        inter_e = torch.min(pred_e, gt_e)
        inter = (inter_e - inter_s).clamp_min(0.0)

        # Union
        union = pred_len + gt_len - inter
        iou = inter / union.clamp_min(self.eps)

        # Enclosing convex hull
        enclose_s = torch.min(pred_s, gt_s)
        enclose_e = torch.max(pred_e, gt_e)
        enclose_len = (enclose_e - enclose_s).clamp_min(self.eps)

        # Generalized IoU term
        giou = iou - (enclose_len - union) / enclose_len.clamp_min(self.eps)

        # Physical center distance penalty
        c_pred = 0.5 * (pred_s + pred_e)
        c_gt = 0.5 * (gt_s + gt_e)
        c_dist_sq = (c_pred - c_gt) ** 2
        diou_term = self.center_weight * (c_dist_sq / (enclose_len ** 2).clamp_min(self.eps))

        loss = 1.0 - giou + diou_term
        loss = loss * self.loss_weight

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss

