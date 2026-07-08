from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn

from ..builder import SELECTORS
from ..duca import DucaAcquisitionAdapter, ZeroShotActionnessSource, duca_losses
from ..duca.acquisition import _assert_no_forbidden_payload
from ..utils.truetime_geometry import SELECTED_AXIS, TRUE_TIME_AXIS


_DEFAULT_METADATA_KEYS = {
    "selected_positions": "duca_online_selected_positions",
    "selected_positions_unit": "duca_online_selected_positions_unit",
    "selected_mask": "duca_online_selected_mask",
    "selected_count": "duca_online_selected_count",
    "remap": "duca_online_selected_axis_remap",
    "source": "duca_online_actionness_source",
}

_ACTIONNESS_KWARGS = {
    "feature_dim",
    "hidden_dim",
    "frozen",
    "mode",
    "p_action",
    "uncertainty",
    "video_text_model",
    "tokenizer",
    "action_prompts",
    "background_prompts",
    "temperature",
    "provenance",
    "source_name",
    "checkpoint_hash",
    "thumos_trained",
    "uses_labels",
    "uses_teacher",
    "calibration_split",
    "prompt_hash",
}


def _time_descriptors_btc(inputs: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(inputs):
        raise ValueError("inputs must be a tensor")
    if inputs.ndim == 3:
        return inputs.transpose(1, 2).contiguous()
    if inputs.ndim == 5:
        return inputs.mean(dim=(3, 4)).transpose(1, 2).contiguous()
    if inputs.ndim == 6:
        return inputs.mean(dim=(1, 4, 5)).transpose(1, 2).contiguous()
    raise ValueError(f"unsupported DUCA selector input shape: {tuple(inputs.shape)}")


def _gather_time(inputs: torch.Tensor, selected_positions: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    idx = selected_positions.to(device=inputs.device, dtype=torch.long).clamp_min(0)
    if inputs.ndim == 3:
        gathered = torch.gather(inputs, dim=2, index=idx[:, None, :].expand(-1, inputs.shape[1], -1))
        return gathered * slot_mask[:, None, :].to(dtype=gathered.dtype)
    if inputs.ndim == 5:
        gathered = torch.gather(
            inputs,
            dim=2,
            index=idx[:, None, :, None, None].expand(-1, inputs.shape[1], -1, inputs.shape[3], inputs.shape[4]),
        )
        return gathered * slot_mask[:, None, :, None, None].to(dtype=gathered.dtype)
    if inputs.ndim == 6:
        gathered = torch.gather(
            inputs,
            dim=3,
            index=idx[:, None, None, :, None, None].expand(
                -1, inputs.shape[1], inputs.shape[2], -1, inputs.shape[4], inputs.shape[5]
            ),
        )
        return gathered * slot_mask[:, None, None, :, None, None].to(dtype=gathered.dtype)
    raise ValueError(f"unsupported DUCA selector input shape: {tuple(inputs.shape)}")


def _apply_slot_weights(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if inputs.ndim == 3:
        return inputs * weights[:, None, :].to(dtype=inputs.dtype)
    if inputs.ndim == 5:
        return inputs * weights[:, None, :, None, None].to(dtype=inputs.dtype)
    if inputs.ndim == 6:
        return inputs * weights[:, None, None, :, None, None].to(dtype=inputs.dtype)
    raise ValueError(f"unsupported DUCA selector input shape: {tuple(inputs.shape)}")


@SELECTORS.register_module()
class DucaOnlineFrameSelector(nn.Module):
    """Registry-buildable online DUCA selector for OpenTAD frame_selector hooks."""

    def __init__(
        self,
        in_channels: int,
        budget: int = 384,
        max_radius: int = 16,
        dense_window_size: Optional[int] = None,
        selector_hidden_channels: int = 0,
        actionness_source_cfg: Optional[Mapping[str, Any]] = None,
        detector_gradient_mode: str = "st_sparse_gather",
        coordinate_space: str = SELECTED_AXIS,
        selected_positions_unit: str = "original_time_index",
        true_time_source_axis: str = TRUE_TIME_AXIS,
        loss_weights: Optional[Mapping[str, float]] = None,
        no_ledger_decision: bool = True,
        remap_gt_to_selected_axis: bool = True,
        selected_axis_remap_required: bool = True,
        forbid_ledger: bool = True,
        forbid_raw_prediction_cache: bool = True,
        metadata_keys: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.budget = int(budget)
        self.max_radius = int(max_radius)
        self.dense_window_size = None if dense_window_size is None else int(dense_window_size)
        self.detector_gradient_mode = str(detector_gradient_mode)
        self.selected_positions_coordinate = str(coordinate_space)
        self.coordinate_space = SELECTED_AXIS
        self.selected_positions_unit = str(selected_positions_unit)
        self.true_time_source_axis = str(true_time_source_axis)
        self.loss_weights = dict(loss_weights or {})
        self.no_ledger_decision = bool(no_ledger_decision)
        self.remap_gt_to_selected_axis = bool(remap_gt_to_selected_axis)
        self.selected_axis_remap_required = bool(selected_axis_remap_required)
        self.forbid_ledger = bool(forbid_ledger)
        self.forbid_raw_prediction_cache = bool(forbid_raw_prediction_cache)
        self.metadata_keys = dict(_DEFAULT_METADATA_KEYS)
        if metadata_keys:
            self.metadata_keys.update(dict(metadata_keys))
        self.extra_config = dict(kwargs)
        self.last_forward_summary: dict[str, Any] = {}
        if self.detector_gradient_mode != "st_sparse_gather":
            raise ValueError("detector_gradient_mode must be st_sparse_gather")
        if self.selected_positions_coordinate not in {"original_time", SELECTED_AXIS, TRUE_TIME_AXIS}:
            raise ValueError("coordinate_space must describe original-time selected positions or selected-axis detector output")
        if self.selected_positions_unit != "original_time_index":
            raise ValueError("selected_positions_unit must be original_time_index")
        if self.true_time_source_axis != TRUE_TIME_AXIS:
            raise ValueError("true_time_source_axis must be true_time_dense_index")
        if self.dense_window_size is not None and self.dense_window_size <= 0:
            raise ValueError("dense_window_size must be positive")
        if not self.no_ledger_decision:
            raise ValueError("DUCA online selector requires no_ledger_decision=True")
        if not self.remap_gt_to_selected_axis:
            raise ValueError("DUCA online selector requires remap_gt_to_selected_axis=True")

        actionness_source = None
        self.actionness_source_name = "duca_adapter_internal"
        if actionness_source_cfg:
            cfg = dict(actionness_source_cfg)
            source_type = cfg.pop("type", "ZeroShotActionnessSource")
            self.actionness_source_name = str(cfg.get("source_name") or source_type)
            if source_type in {"DucaOnlineProbeActionnessSource", "ZeroShotMotionActionnessSource"}:
                cfg.setdefault("mode", "motion")
                cfg.setdefault("source_name", self.actionness_source_name)
                cfg.setdefault("thumos_trained", False)
                cfg.setdefault("uses_labels", False)
                cfg.setdefault("uses_teacher", False)
                cfg = {key: value for key, value in cfg.items() if key in _ACTIONNESS_KWARGS}
                actionness_source = ZeroShotActionnessSource(**cfg)
            elif source_type == "ZeroShotActionnessSource":
                cfg = {key: value for key, value in cfg.items() if key in _ACTIONNESS_KWARGS}
                actionness_source = ZeroShotActionnessSource(**cfg)
            else:
                raise ValueError(f"unsupported actionness_source_cfg type {source_type!r}")
        self.adapter = DucaAcquisitionAdapter(
            feature_dim=self.in_channels,
            budget=self.budget,
            max_radius=self.max_radius,
            hidden_dim=int(selector_hidden_channels),
            actionness_source=actionness_source,
        )

    def forward_train(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas,
        gt_segments=None,
        gt_labels=None,
        teacher_utility: Optional[torch.Tensor] = None,
        budget=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._reject_train_decision_payload(metas)
        outputs = self._forward_select(inputs, masks, metas, budget=budget)
        selector_losses = duca_losses(
            outputs["selector_outputs"],
            teacher_utility=teacher_utility,
            loss_weights=self.loss_weights,
        )
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": outputs["metas"],
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "losses": selector_losses,
            "selector_outputs": outputs["selector_outputs"],
        }

    def forward_test(self, inputs: torch.Tensor, masks: torch.Tensor, metas=None, budget=None, **kwargs: Any) -> dict[str, Any]:
        _assert_no_forbidden_payload({"metas": metas, "kwargs": kwargs})
        outputs = self._forward_select(inputs, masks, metas, budget=budget)
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": outputs["metas"],
            "selector_outputs": outputs["selector_outputs"],
        }

    def _forward_select(self, inputs: torch.Tensor, masks: torch.Tensor, metas, budget=None) -> dict[str, Any]:
        descriptors = _time_descriptors_btc(inputs)
        if descriptors.shape[-1] != self.in_channels:
            raise ValueError(f"DUCA selector expected {self.in_channels} channels, got {descriptors.shape[-1]}")
        masks = masks.to(device=inputs.device, dtype=torch.bool)
        grid, scores = self.adapter.acquire(descriptors, budget=budget, valid_mask=masks)
        positions = grid.selected_positions.to(device=inputs.device)
        slot_mask = positions >= 0
        hard_selected = _gather_time(inputs, positions, slot_mask)
        st_weights = torch.gather(scores["selected_mask_st"], 1, positions.clamp_min(0)) * slot_mask.to(
            dtype=scores["selected_mask_st"].dtype
        )
        selected_inputs = _apply_slot_weights(hard_selected, st_weights)
        selected_masks = slot_mask.to(device=inputs.device, dtype=torch.bool)
        scores["grid"] = grid
        scores["hard_selected_inputs"] = hard_selected
        scores["selected_input_st_gradient_path"] = self.detector_gradient_mode
        scores["sparse_grid"] = grid
        selected_counts = selected_masks.long().sum(dim=1).detach().cpu().tolist()
        self.last_forward_summary = {
            self.metadata_keys["selected_count"]: int(selected_counts[0]) if selected_counts else 0,
            "budget": int(grid.budget),
            "budget_unit": grid.budget_unit,
            "coordinate": grid.coordinate,
            self.metadata_keys["source"]: self.actionness_source_name,
        }
        return {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": self._write_metas(metas, grid),
            "selector_outputs": scores,
        }

    @staticmethod
    def _reject_train_decision_payload(metas) -> None:
        if metas is None:
            return
        holders = [metas] if isinstance(metas, Mapping) else metas
        for idx, meta in enumerate(holders):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{idx}] must be a mapping")
            if meta.get("selection_uses_gt") is True:
                raise ValueError("DUCA train selection forbids GT-driven selection")
            if meta.get("selection_uses_teacher") is True:
                raise ValueError("DUCA train selection forbids teacher-driven selection")
            _assert_no_forbidden_payload(
                {"meta": meta},
                forbidden_keys={
                    "teacher_points",
                    "dense_teacher",
                    "dense_teacher_payload",
                    "dense_teacher_points",
                    "oracle_boundary",
                    "prediction_cache",
                    "raw_prediction",
                    "raw_predictions",
                    "ledger",
                    "ledger_path",
                },
            )

    def _write_metas(self, metas, grid) -> list[dict[str, Any]]:
        batch = int(grid.selected_positions.shape[0])
        if metas is None:
            out = [{} for _ in range(batch)]
        else:
            if len(metas) != batch:
                raise ValueError("metas length must match batch size")
            out = [dict(meta) for meta in metas]
        positions_cpu = grid.selected_positions.detach().cpu().long()
        valid_lens = grid.valid_len.detach().cpu().long()
        for idx, meta in enumerate(out):
            positions = [int(item) for item in positions_cpu[idx].tolist() if int(item) >= 0]
            dense_valid_len = int(valid_lens[idx].item())
            remap = {
                "source": SELECTED_AXIS,
                "target": TRUE_TIME_AXIS,
                "selected_to_original": {int(axis): int(pos) for axis, pos in enumerate(positions)},
                "original_to_selected": {int(pos): int(axis) for axis, pos in enumerate(positions)},
                "selected_axis_to_true_time_dense_index": positions,
            }
            meta["duca_online_selected_positions"] = positions
            meta["duca_online_selected_positions_unit"] = self.selected_positions_unit
            meta["duca_online_selected_mask"] = [True] * len(positions)
            meta["duca_online_budget"] = int(grid.budget)
            meta["duca_online_selected_count"] = len(positions)
            meta["duca_online_selected_axis_remap"] = remap
            meta["duca_online_actionness_source"] = self.actionness_source_name
            meta["duca_online_budget_unit"] = grid.budget_unit
            meta["duca_online_coordinate"] = grid.coordinate
            meta["detector_output_coordinate_space"] = self.coordinate_space
            meta["detector_prediction_inverse_map_required"] = True
            meta["selected_axis_to_true_time_dense_index"] = positions
            meta["truetime_selected_positions"] = positions
            meta["truetime_dense_len"] = int(grid.original_length)
            meta["truetime_dense_valid_len"] = dense_valid_len
            meta["irregular_selected_positions"] = positions
            meta["irregular_native_axis"] = True
            meta["irregular_selected_count"] = len(positions)
            meta["irregular_dense_valid_len"] = dense_valid_len
            meta["irregular_selected_valid_len"] = len(positions)
            meta.setdefault(self.metadata_keys["selected_positions"], positions)
            meta.setdefault(self.metadata_keys["selected_positions_unit"], self.selected_positions_unit)
            meta.setdefault(self.metadata_keys["selected_mask"], [True] * len(positions))
            meta.setdefault(self.metadata_keys["selected_count"], len(positions))
            meta.setdefault(self.metadata_keys["remap"], remap)
            meta.setdefault(self.metadata_keys["source"], self.actionness_source_name)
        return out
