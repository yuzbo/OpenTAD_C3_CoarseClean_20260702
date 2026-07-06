from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from ..builder import SELECTORS
from ..utils.truetime_geometry import SELECTED_AXIS, TRUE_TIME_AXIS


_GT_META_KEYS = ("gt_segments", "gt_labels", "selection_uses_gt", "uses_gt")
_TEACHER_META_KEYS = ("teacher_utility", "teacher_scores", "selection_uses_teacher", "uses_teacher")


def selector_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return total**0.5


def _require_feature_tensor(features):
    if not torch.is_tensor(features):
        raise ValueError("features must be a tensor")
    if torch.is_complex(features) or not bool(torch.isfinite(features).all().item()):
        raise ValueError("features must be finite and real-valued")
    if features.ndim != 3:
        raise ValueError(f"selector feature path expects [B,C,T], got {tuple(features.shape)}")


def _reject_eval_leakage(metas):
    if metas is None:
        return
    holders = [metas] if isinstance(metas, Mapping) else metas
    if not isinstance(holders, (list, tuple)):
        raise ValueError("metas must be a mapping/list/tuple or None")
    for idx, meta in enumerate(holders):
        if not isinstance(meta, Mapping):
            raise ValueError(f"metas[{idx}] must be a mapping")
        gt_present = [key for key in _GT_META_KEYS if meta.get(key) not in (None, False)]
        if gt_present:
            raise ValueError(f"true-time val/test selection forbids GT metadata: {gt_present}")
        teacher_present = [key for key in _TEACHER_META_KEYS if meta.get(key) not in (None, False)]
        if teacher_present:
            raise ValueError(f"true-time val/test selection forbids teacher metadata: {teacher_present}")


def _time_descriptors(inputs):
    if inputs.ndim == 3:
        return inputs
    if inputs.ndim == 5:
        # MMAction can expose frames as [B, C, T, H, W].
        return inputs.mean(dim=(3, 4))
    if inputs.ndim == 6:
        # OpenTAD end-to-end frames are commonly [B, N, C, T, H, W].
        return inputs.mean(dim=(1, 4, 5))
    raise ValueError(f"unsupported true-time selector input shape: {tuple(inputs.shape)}")


def _gather_time(inputs, indices):
    if inputs.ndim == 3:
        gather_index = indices[:, None, :].expand(-1, inputs.shape[1], -1)
        return torch.gather(inputs, dim=2, index=gather_index)
    if inputs.ndim == 5:
        gather_index = indices[:, None, :, None, None].expand(
            -1,
            inputs.shape[1],
            -1,
            inputs.shape[3],
            inputs.shape[4],
        )
        return torch.gather(inputs, dim=2, index=gather_index)
    if inputs.ndim == 6:
        gather_index = indices[:, None, None, :, None, None].expand(
            -1,
            inputs.shape[1],
            inputs.shape[2],
            -1,
            inputs.shape[4],
            inputs.shape[5],
        )
        return torch.gather(inputs, dim=3, index=gather_index)
    raise ValueError(f"unsupported true-time selector input shape: {tuple(inputs.shape)}")


def _soft_select_time(inputs, assignment):
    if inputs.ndim == 3:
        return torch.einsum("bct,bkt->bck", inputs, assignment)
    if inputs.ndim == 5:
        return torch.einsum("bcthw,bkt->bckhw", inputs, assignment)
    if inputs.ndim == 6:
        return torch.einsum("bncthw,bkt->bnckhw", inputs, assignment)
    raise ValueError(f"unsupported true-time selector input shape: {tuple(inputs.shape)}")


@SELECTORS.register_module()
class TrueTimeRelaxedHardTopKSelector(nn.Module):
    """Straight-through hard top-k selector with an explicit true-time contract."""

    def __init__(
        self,
        in_channels,
        selected_count,
        dense_len=None,
        temperature=1.0,
        selector_hidden_channels=0,
        allow_gt_selection=False,
        allow_teacher_utility=False,
        coordinate_space=SELECTED_AXIS,
        true_time_source_axis=TRUE_TIME_AXIS,
        gradient_path="straight_through_relaxed_temporal_surrogate",
        detector_gradient_mode="st_sparse_gather",
        slot_softmax_temperature=1.0,
        slot_distance_penalty=1.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.selected_count = int(selected_count)
        self.dense_len = None if dense_len is None else int(dense_len)
        self.temperature = float(temperature)
        self.allow_gt_selection = bool(allow_gt_selection)
        self.allow_teacher_utility = bool(allow_teacher_utility)
        self.coordinate_space = str(coordinate_space)
        self.true_time_source_axis = str(true_time_source_axis)
        self.gradient_path = str(gradient_path)
        self.detector_gradient_mode = str(detector_gradient_mode)
        self.slot_softmax_temperature = float(slot_softmax_temperature)
        self.slot_distance_penalty = float(slot_distance_penalty)
        if self.selected_count <= 0:
            raise ValueError("selected_count must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.coordinate_space != SELECTED_AXIS:
            raise ValueError("detector outputs must use selected_axis_index coordinates")
        if self.true_time_source_axis != TRUE_TIME_AXIS:
            raise ValueError("true_time_source_axis must be true_time_dense_index")
        if self.detector_gradient_mode != "st_sparse_gather":
            raise ValueError("detector_gradient_mode must be 'st_sparse_gather'")
        if self.slot_softmax_temperature <= 0:
            raise ValueError("slot_softmax_temperature must be positive")
        if self.slot_distance_penalty < 0:
            raise ValueError("slot_distance_penalty must be non-negative")

        hidden = int(selector_hidden_channels)
        if hidden > 0:
            self.scorer = nn.Sequential(
                nn.Conv1d(self.in_channels, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, 1, kernel_size=1),
            )
        else:
            self.scorer = nn.Conv1d(self.in_channels, 1, kernel_size=1)

    def forward_features(self, features, masks=None, phase="joint_finetune"):
        _require_feature_tensor(features)
        batch, channels, time = features.shape
        if channels != self.in_channels:
            raise ValueError(f"selector expected {self.in_channels} channels, got {channels}")
        if self.dense_len is not None and int(time) != int(self.dense_len):
            raise ValueError(f"selector expected dense_len={self.dense_len}, got {time}")
        if self.selected_count > int(time):
            raise ValueError("selected_count must not exceed feature length")

        if masks is None:
            masks = torch.ones((batch, time), dtype=torch.bool, device=features.device)
        else:
            masks = masks.to(device=features.device, dtype=torch.bool)
        if masks.shape != (batch, time):
            raise ValueError(f"masks must be [B,T]={batch,time}, got {tuple(masks.shape)}")
        if bool((masks.long().sum(dim=1) < self.selected_count).any().item()):
            raise ValueError("every sample must have at least selected_count valid positions")

        logits = self.scorer(features).squeeze(1)
        logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        selected_indices = torch.topk(logits, k=self.selected_count, dim=1).indices.sort(dim=1).values
        hard_mask = torch.zeros_like(logits)
        hard_mask.scatter_(1, selected_indices, 1.0)
        hard_mask = hard_mask * masks.to(dtype=hard_mask.dtype)

        probabilities = torch.softmax(logits / self.temperature, dim=1) * masks.to(dtype=logits.dtype)
        probability_mass = probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        probabilities = probabilities / probability_mass
        relaxed_mask = probabilities * float(self.selected_count)
        st_mask = hard_mask + relaxed_mask - relaxed_mask.detach()
        soft_surrogate_features = features * st_mask.unsqueeze(1)

        selected_features = _gather_time(features, selected_indices)
        selected_masks = torch.ones(
            (batch, self.selected_count),
            dtype=torch.bool,
            device=features.device,
        )
        selected_counts = hard_mask.sum(dim=1)
        dense_valid_len = masks.long().sum(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum(dim=1).mean()

        return {
            "features": selected_features,
            "masks": selected_masks,
            "logits": logits,
            "hard_mask": hard_mask,
            "relaxed_mask": relaxed_mask,
            "straight_through_mask": st_mask,
            "soft_surrogate_features": soft_surrogate_features,
            "selected_indices": selected_indices,
            "selected_positions": selected_indices,
            "selected_count_mean": selected_counts.mean(),
            "selected_count_std": selected_counts.float().std(unbiased=False),
            "dense_valid_len": dense_valid_len,
            "entropy": entropy,
            "phase": str(phase),
            "coordinate_space": self.coordinate_space,
            "true_time_source_axis": self.true_time_source_axis,
            "selector_grad_path": self.gradient_path,
            "detector_gradient_mode": self.detector_gradient_mode,
        }

    def forward_train(self, inputs, masks, metas, gt_segments=None, gt_labels=None, phase="joint_finetune", **kwargs):
        if not self.allow_gt_selection:
            self._reject_train_selection_leakage(metas)
        descriptors = _time_descriptors(inputs)
        outputs = self.forward_features(descriptors, masks=masks, phase=phase)
        selected_inputs = self._selected_inputs_for_detector(inputs, outputs, masks)
        selected_masks = outputs["masks"]
        out_metas = self._write_metas(metas, outputs)
        losses = {
            "selector_entropy_loss": outputs["entropy"] * 0.0,
            "selector_selected_count_mean": outputs["selected_count_mean"].detach() * 0.0,
            "selector_selected_count_std": outputs["selected_count_std"].detach() * 0.0,
        }
        return {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": out_metas,
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "losses": losses,
            "selector_outputs": outputs,
        }

    def forward_test(self, inputs, masks, metas=None, **kwargs):
        _reject_eval_leakage(metas)
        descriptors = _time_descriptors(inputs)
        outputs = self.forward_features(descriptors, masks=masks, phase="eval_sparse_selector")
        return {
            "inputs": _gather_time(inputs, outputs["selected_indices"]),
            "masks": outputs["masks"],
            "metas": self._write_metas(metas, outputs),
            "selector_outputs": outputs,
        }

    def _selected_inputs_for_detector(self, inputs, outputs, masks):
        if self.detector_gradient_mode != "st_sparse_gather":
            raise ValueError("detector_gradient_mode must be 'st_sparse_gather'")
        hard_selected = _gather_time(inputs, outputs["selected_indices"])
        assignment = self._slot_soft_assignment(
            outputs["logits"],
            outputs["selected_indices"],
            masks=masks,
        )
        soft_selected = _soft_select_time(inputs, assignment)
        selected_inputs = hard_selected + soft_selected - soft_selected.detach()
        outputs["slot_soft_assignment"] = assignment
        outputs["hard_selected_inputs"] = hard_selected
        outputs["soft_selected_inputs"] = soft_selected
        outputs["selected_input_st_gradient_path"] = self.detector_gradient_mode
        return selected_inputs

    def _slot_soft_assignment(self, logits, selected_indices, masks):
        batch, time = logits.shape
        positions = torch.arange(time, device=logits.device, dtype=logits.dtype).view(1, 1, time)
        centers = selected_indices.to(device=logits.device, dtype=logits.dtype).unsqueeze(-1)
        slot_logits = logits.unsqueeze(1) - self.slot_distance_penalty * torch.abs(positions - centers)
        slot_logits = slot_logits / self.slot_softmax_temperature
        if masks is not None:
            valid = masks.to(device=logits.device, dtype=torch.bool).view(batch, 1, time)
            slot_logits = slot_logits.masked_fill(~valid, torch.finfo(slot_logits.dtype).min)
        return torch.softmax(slot_logits, dim=-1)

    def _reject_train_selection_leakage(self, metas):
        if metas is None:
            return
        holders = [metas] if isinstance(metas, Mapping) else metas
        for idx, meta in enumerate(holders):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{idx}] must be a mapping")
            if meta.get("selection_uses_gt") is True:
                raise ValueError("true-time selector forbids GT-driven selection by default")
            if not self.allow_teacher_utility and meta.get("selection_uses_teacher") is True:
                raise ValueError("true-time selector forbids teacher-driven selection by default")

    def _write_metas(self, metas, outputs):
        batch = int(outputs["selected_indices"].shape[0])
        if metas is None:
            out = [{} for _ in range(batch)]
        else:
            if len(metas) != batch:
                raise ValueError("metas length must match batch size")
            out = [dict(meta) for meta in metas]
        indices = outputs["selected_indices"].detach().cpu().long()
        for idx, meta in enumerate(out):
            positions = [int(item) for item in indices[idx].tolist()]
            if "dense_valid_len" in outputs:
                dense_valid_len = int(outputs["dense_valid_len"][idx].detach().cpu().item())
            else:
                dense_valid_len = int(self.dense_len or max(positions) + 1)
            meta["truetime_selected_positions"] = positions
            meta["truetime_dense_len"] = int(self.dense_len or dense_valid_len)
            meta["truetime_dense_valid_len"] = dense_valid_len
            meta["truetime_selected_count"] = len(positions)
            meta["detector_output_coordinate_space"] = self.coordinate_space
            meta["detector_prediction_inverse_map_required"] = True
            meta["selected_axis_to_true_time_dense_index"] = positions
            meta["irregular_selected_positions"] = positions
            meta["irregular_native_axis"] = False
            meta["irregular_selected_count"] = len(positions)
            meta["irregular_dense_valid_len"] = dense_valid_len
            meta["irregular_selected_valid_len"] = dense_valid_len
        return out
