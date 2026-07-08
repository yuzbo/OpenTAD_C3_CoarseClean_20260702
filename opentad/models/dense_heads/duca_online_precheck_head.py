from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS


def _features_to_bct(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim == 3:
        return inputs
    if inputs.ndim == 5:
        return inputs.mean(dim=(3, 4))
    if inputs.ndim == 6:
        return inputs.mean(dim=(1, 4, 5))
    raise ValueError(f"unsupported DUCA precheck head input shape: {tuple(inputs.shape)}")


@HEADS.register_module()
class DucaOnlinePrecheckHead(nn.Module):
    """Tiny registry-built head for DUCA online selector integration checks."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        require_gt_in_train: bool = False,
        require_selected_metas: bool = False,
        require_original_time_positions: bool = False,
        require_selected_axis_remap: bool = False,
        metadata_keys=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.require_gt_in_train = bool(require_gt_in_train)
        self.require_selected_metas = bool(require_selected_metas)
        self.require_original_time_positions = bool(require_original_time_positions)
        self.require_selected_axis_remap = bool(require_selected_axis_remap)
        self.metadata_keys = dict(
            metadata_keys
            or {
                "selected_positions": "duca_online_selected_positions",
                "selected_positions_unit": "duca_online_selected_positions_unit",
                "selected_mask": "duca_online_selected_mask",
                "selected_count": "duca_online_selected_count",
                "remap": "duca_online_selected_axis_remap",
                "source": "duca_online_actionness_source",
            }
        )
        self.extra_config = dict(kwargs)
        self.conv = nn.Conv1d(self.in_channels, self.num_classes, kernel_size=1)
        self.last_gt_segments = None
        self.last_gt_labels = None
        self.last_masks = None
        self.last_metas = None
        self.last_input_shape = None
        self.last_precheck_summary = {}

    def forward_train(self, feat_list, mask_list, gt_segments=None, gt_labels=None, metas=None, **kwargs):
        self._validate_metas(metas)
        if self.require_gt_in_train and (gt_segments is None or gt_labels is None):
            raise ValueError("DucaOnlinePrecheckHead requires gt_segments/gt_labels during train precheck")
        self.last_gt_segments = gt_segments
        self.last_gt_labels = gt_labels
        self.last_masks = mask_list.detach().clone()
        self.last_metas = metas
        self.last_input_shape = tuple(feat_list.shape)
        features = _features_to_bct(feat_list)
        if features.shape[1] != self.in_channels:
            raise ValueError(f"DucaOnlinePrecheckHead expected {self.in_channels} channels, got {features.shape[1]}")
        logits = self.conv(features)
        target = torch.linspace(0.0, 1.0, logits.shape[-1], device=logits.device, dtype=logits.dtype)
        target = target.reshape(1, 1, -1).expand_as(logits)
        valid = mask_list.to(device=logits.device, dtype=logits.dtype).unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        self.last_precheck_summary = self._summary(metas, mask_list)
        return {"loss_detector": (loss * valid).sum() / valid.sum().clamp_min(1.0)}

    def forward_test(self, feat_list, mask_list, metas=None, **kwargs):
        self._validate_metas(metas)
        self.last_masks = mask_list.detach().clone()
        self.last_metas = metas
        self.last_input_shape = tuple(feat_list.shape)
        self.last_precheck_summary = self._summary(metas, mask_list)
        proposals = []
        scores = []
        for mask in mask_list:
            count = int(mask.long().sum().item())
            starts = torch.arange(count, device=mask.device, dtype=torch.float32)
            ends = starts + 1.0
            proposals.append(torch.stack((starts, ends), dim=-1))
            scores.append(torch.ones(count, self.num_classes, device=mask.device, dtype=torch.float32))
        return proposals, scores

    def _validate_metas(self, metas) -> None:
        if metas is None or not self.require_selected_metas:
            return
        if not isinstance(metas, (list, tuple)):
            raise ValueError("DucaOnlinePrecheckHead expects metas as a list/tuple")
        for idx, meta in enumerate(metas):
            if not isinstance(meta, dict):
                raise ValueError(f"metas[{idx}] must be a dict")
            positions_key = self.metadata_keys["selected_positions"]
            unit_key = self.metadata_keys["selected_positions_unit"]
            remap_key = self.metadata_keys["remap"]
            if positions_key not in meta:
                raise ValueError(f"metas[{idx}] missing {positions_key}")
            if self.require_original_time_positions and meta.get(unit_key) != "original_time_index":
                raise ValueError(f"metas[{idx}] must declare original-time selected positions")
            if self.require_selected_axis_remap and remap_key not in meta:
                raise ValueError(f"metas[{idx}] missing selected-axis remap metadata")

    def _summary(self, metas, mask_list):
        if not metas:
            return {}
        first = metas[0]
        positions_key = self.metadata_keys["selected_positions"]
        mask_key = self.metadata_keys["selected_mask"]
        count_key = self.metadata_keys["selected_count"]
        remap_key = self.metadata_keys["remap"]
        unit_key = self.metadata_keys["selected_positions_unit"]
        source_key = self.metadata_keys["source"]
        return {
            positions_key: first.get(positions_key),
            mask_key: first.get(mask_key),
            count_key: first.get(count_key),
            remap_key: first.get(remap_key),
            unit_key: first.get(unit_key),
            source_key: first.get(source_key),
            "detector_mask_count": int(mask_list[0].long().sum().item()) if mask_list.numel() else 0,
        }
