from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS, build_selector
from .base import BaseDetector


def _features_to_bct(inputs):
    if inputs.ndim == 3:
        return inputs
    if inputs.ndim == 5:
        return inputs.mean(dim=(3, 4))
    if inputs.ndim == 6:
        return inputs.mean(dim=(1, 4, 5))
    raise ValueError(f"unsupported smoke detector input shape: {tuple(inputs.shape)}")


@DETECTORS.register_module()
class TrueTimeJointSelectorSmokeDetector(BaseDetector):
    """CPU-safe detector path used only to prove selector gradients through train losses."""

    def __init__(self, frame_selector, in_channels=3, hidden_channels=8):
        super().__init__()
        self.frame_selector = build_selector(frame_selector)
        in_channels = int(in_channels)
        hidden_channels = int(hidden_channels)
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("in_channels and hidden_channels must be positive")
        self.backbone = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.projection = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.cls_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.reg_head = nn.Conv1d(hidden_channels, 2, kernel_size=1)
        self.last_selector_outputs = None
        self.last_selected_inputs = None
        self.last_hard_selected_inputs = None

    def forward_train(self, inputs, masks, metas, gt_segments=None, gt_labels=None, **kwargs):
        selector_result = self.frame_selector.forward_train(
            inputs=inputs,
            masks=masks,
            metas=metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            phase="joint_finetune",
        )
        selected_inputs = selector_result["inputs"]
        selected_masks = selector_result["masks"].to(dtype=selected_inputs.dtype)
        selector_outputs = selector_result["selector_outputs"]
        features = _features_to_bct(selected_inputs)
        x = F.relu(self.backbone(features))
        x = F.relu(self.projection(x))
        cls_logits = self.cls_head(x).squeeze(1)
        reg_pred = self.reg_head(x).transpose(1, 2)

        batch, selected_len = cls_logits.shape
        slot = torch.linspace(0.0, 1.0, selected_len, device=cls_logits.device, dtype=cls_logits.dtype)
        cls_target = slot.unsqueeze(0).expand(batch, -1)
        reg_target = torch.stack((slot, 1.0 - slot), dim=-1).unsqueeze(0).expand(batch, -1, -1)
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction="none")
        loss_cls = (loss_cls * selected_masks).sum() / selected_masks.sum().clamp_min(1.0)
        loss_reg = F.smooth_l1_loss(reg_pred, reg_target, reduction="none").mean(dim=-1)
        loss_reg = (loss_reg * selected_masks).sum() / selected_masks.sum().clamp_min(1.0)

        self.last_selector_outputs = selector_outputs
        self.last_selected_inputs = selected_inputs
        self.last_hard_selected_inputs = selector_outputs.get("hard_selected_inputs")
        losses = {
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
        }
        losses["cost"] = loss_cls + loss_reg
        return losses
