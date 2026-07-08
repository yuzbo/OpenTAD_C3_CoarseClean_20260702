from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn

from ..builder import SELECTORS
from ..duca import DucaAcquisitionAdapter, ZeroShotActionnessSource, duca_losses
from ..duca.acquisition import _assert_no_forbidden_payload, validate_actionness_provenance
from ..utils.truetime_geometry import SELECTED_AXIS, TRUE_TIME_AXIS, TrueTimeMap


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
    "uses_gt",
    "uses_prediction_cache",
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


def _add_soft_context_gradient_path(
    hard_selected: torch.Tensor,
    dense_inputs: torch.Tensor,
    soft_coverage: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    weights = soft_coverage.to(device=dense_inputs.device, dtype=dense_inputs.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    if dense_inputs.ndim == 3:
        context = torch.einsum("bct,bt->bc", dense_inputs, weights)
        context = context[:, :, None].expand_as(hard_selected)
        slot = slot_mask[:, None, :]
    elif dense_inputs.ndim == 5:
        context = (dense_inputs * weights[:, None, :, None, None]).sum(dim=2)
        context = context[:, :, None, :, :].expand_as(hard_selected)
        slot = slot_mask[:, None, :, None, None]
    elif dense_inputs.ndim == 6:
        context = (dense_inputs * weights[:, None, None, :, None, None]).sum(dim=3)
        context = context[:, :, :, None, :, :].expand_as(hard_selected)
        slot = slot_mask[:, None, None, :, None, None]
    else:
        raise ValueError(f"unsupported DUCA selector input shape: {tuple(dense_inputs.shape)}")
    return hard_selected + (context - context.detach()) * slot.to(dtype=hard_selected.dtype)


@SELECTORS.register_module()
class DucaOnlineFrameSelector(nn.Module):
    """Registry-buildable online DUCA selector for OpenTAD frame_selector hooks."""

    def __init__(
        self,
        in_channels: int,
        budget: Optional[int] = 384,
        budget_mode: str = "fixed",
        budget_min: int = 64,
        budget_max: Optional[int] = None,
        budget_multiple: int = 16,
        target_budget: Optional[float] = None,
        allow_external_budget_override: Optional[bool] = None,
        max_radius: int = 16,
        dense_window_size: Optional[int] = None,
        selector_hidden_channels: int = 0,
        actionness_source_cfg: Optional[Mapping[str, Any]] = None,
        detector_gradient_mode: str = "st_sparse_gather",
        coordinate_space: str = SELECTED_AXIS,
        detector_output_coordinate_space: str = SELECTED_AXIS,
        selected_positions_unit: str = "original_time_index",
        true_time_source_axis: str = TRUE_TIME_AXIS,
        loss_weights: Optional[Mapping[str, float]] = None,
        no_ledger_decision: bool = True,
        remap_gt_to_selected_axis: bool = True,
        selected_axis_remap_required: bool = True,
        forbid_ledger: bool = True,
        forbid_raw_prediction_cache: bool = True,
        external_actionness_meta_key: Optional[str] = None,
        external_actionness_logits_meta_key: Optional[str] = None,
        external_actionness_provenance_meta_key: Optional[str] = None,
        external_actionness_source_meta_key: Optional[str] = None,
        require_external_actionness: bool = False,
        metadata_keys: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.budget_mode = str(budget_mode)
        if self.budget_mode not in {"fixed", "dynamic_must"}:
            raise ValueError("budget_mode must be fixed or dynamic_must")
        if self.budget_mode == "dynamic_must":
            if budget is not None:
                raise ValueError("dynamic_must selector must set fixed budget=None and use budget_max")
            if budget_max is None:
                raise ValueError("dynamic_must selector requires budget_max")
            if allow_external_budget_override is True:
                raise ValueError("main dynamic_must selector forbids external budget override")
            self.budget = int(budget_max)
            self.allow_external_budget_override = False
        else:
            if budget is None:
                raise ValueError("fixed selector budget cannot be None")
            self.budget = int(budget)
            self.allow_external_budget_override = (
                True if allow_external_budget_override is None else bool(allow_external_budget_override)
            )
        self.budget_min = int(budget_min if self.budget_mode == "dynamic_must" else self.budget)
        self.budget_max = int(self.budget)
        self.budget_multiple = int(budget_multiple)
        self.target_budget = float(self.budget if target_budget is None else target_budget)
        self.max_radius = int(max_radius)
        self.dense_window_size = None if dense_window_size is None else int(dense_window_size)
        self.detector_gradient_mode = str(detector_gradient_mode)
        self.selected_positions_coordinate = str(coordinate_space)
        self.detector_output_coordinate_space = str(detector_output_coordinate_space)
        self.coordinate_space = self.detector_output_coordinate_space
        self.selected_positions_unit = str(selected_positions_unit)
        self.true_time_source_axis = str(true_time_source_axis)
        self.loss_weights = dict(loss_weights or {})
        self.no_ledger_decision = bool(no_ledger_decision)
        self.remap_gt_to_selected_axis = bool(remap_gt_to_selected_axis)
        self.selected_axis_remap_required = bool(selected_axis_remap_required)
        self.forbid_ledger = bool(forbid_ledger)
        self.forbid_raw_prediction_cache = bool(forbid_raw_prediction_cache)
        self.external_actionness_meta_key = external_actionness_meta_key
        self.external_actionness_logits_meta_key = external_actionness_logits_meta_key
        self.external_actionness_provenance_meta_key = external_actionness_provenance_meta_key
        self.external_actionness_source_meta_key = external_actionness_source_meta_key
        self.require_external_actionness = bool(require_external_actionness)
        self.metadata_keys = dict(_DEFAULT_METADATA_KEYS)
        if metadata_keys:
            self.metadata_keys.update(dict(metadata_keys))
        self.extra_config = dict(kwargs)
        self.last_forward_summary: dict[str, Any] = {}
        if self.detector_gradient_mode not in {"st_sparse_gather", "st_sparse_gather_soft_context"}:
            raise ValueError("detector_gradient_mode must be st_sparse_gather or st_sparse_gather_soft_context")
        if self.selected_positions_coordinate not in {"original_time", SELECTED_AXIS, TRUE_TIME_AXIS}:
            raise ValueError("coordinate_space must describe original-time selected positions or selected-axis detector output")
        if self.detector_output_coordinate_space not in {SELECTED_AXIS, TRUE_TIME_AXIS}:
            raise ValueError("detector_output_coordinate_space must be selected-axis or true-time")
        if self.selected_positions_unit != "original_time_index":
            raise ValueError("selected_positions_unit must be original_time_index")
        if self.true_time_source_axis != TRUE_TIME_AXIS:
            raise ValueError("true_time_source_axis must be true_time_dense_index")
        if self.dense_window_size is not None and self.dense_window_size <= 0:
            raise ValueError("dense_window_size must be positive")
        if not self.no_ledger_decision:
            raise ValueError("DUCA online selector requires no_ledger_decision=True")
        if self.detector_output_coordinate_space == SELECTED_AXIS and not self.remap_gt_to_selected_axis:
            raise ValueError("selected-axis detector output requires remap_gt_to_selected_axis=True")

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
            budget=None if self.budget_mode == "dynamic_must" else self.budget,
            budget_mode=self.budget_mode,
            budget_min=self.budget_min,
            budget_max=self.budget_max if self.budget_mode == "dynamic_must" else None,
            budget_multiple=self.budget_multiple,
            target_budget=self.target_budget,
            allow_external_budget_override=self.allow_external_budget_override,
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
        gt_segments, gt_labels, metas = self._remap_train_targets_to_selected_axis(
            gt_segments, gt_labels, outputs["metas"]
        )
        selector_losses = duca_losses(
            outputs["selector_outputs"],
            teacher_utility=teacher_utility,
            loss_weights=self.loss_weights,
        )
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": metas,
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
        external_actionness = self._external_actionness_from_metas(metas, descriptors=descriptors)
        grid, scores = self.adapter.acquire(
            descriptors,
            budget=budget,
            valid_mask=masks,
            actionness_logits=None if external_actionness is None else external_actionness.get("actionness_logits"),
            p_action=None if external_actionness is None else external_actionness.get("p_action"),
        )
        actionness_source_name = self.actionness_source_name
        if external_actionness is not None:
            scores["provenance"] = external_actionness["provenance"]
            scores["external_actionness_provenance"] = external_actionness["provenance"]
            scores["external_actionness_source"] = external_actionness["source_name"]
            actionness_source_name = external_actionness["source_name"]
        validate_actionness_provenance(scores.get("provenance", {}), context="DUCA selector actionness provenance")
        positions = grid.selected_positions.to(device=inputs.device)
        slot_mask = positions >= 0
        hard_selected = _gather_time(inputs, positions, slot_mask)
        if self.detector_gradient_mode == "st_sparse_gather_soft_context":
            hard_selected = _add_soft_context_gradient_path(
                hard_selected,
                inputs,
                scores["soft_coverage"],
                slot_mask,
            )
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
            "requested_budget": [
                int(item) for item in grid.requested_budget.detach().cpu().reshape(-1).tolist()
            ],
            "effective_budget": [
                int(item) for item in grid.effective_budget.detach().cpu().reshape(-1).tolist()
            ],
            "dynamic_budget": bool(grid.metadata.get("budget_is_dynamic", False)),
            "budget_policy": str(grid.metadata.get("budget_policy", "fixed_budget")),
            "budget_unit": grid.budget_unit,
            "coordinate": grid.coordinate,
            self.metadata_keys["source"]: actionness_source_name,
        }
        return {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": self._write_metas(metas, grid, actionness_source_name=actionness_source_name),
            "selector_outputs": scores,
        }

    def _external_actionness_from_metas(self, metas, descriptors: torch.Tensor) -> Optional[dict[str, Any]]:
        if not (
            self.external_actionness_meta_key
            or self.external_actionness_logits_meta_key
            or self.require_external_actionness
        ):
            return None
        if metas is None:
            if self.require_external_actionness:
                raise ValueError("external actionness is required but metas are missing")
            return None
        if len(metas) != descriptors.shape[0]:
            raise ValueError("external actionness metas length must match batch size")

        p_rows = []
        logit_rows = []
        provenances = []
        source_names = []
        need_p = self.external_actionness_meta_key
        need_logits = self.external_actionness_logits_meta_key
        for batch_idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{batch_idx}] must be a mapping for external actionness")
            p_value = self._lookup_meta_value(meta, need_p) if need_p else None
            logits_value = self._lookup_meta_value(meta, need_logits) if need_logits else None
            if p_value is None and logits_value is None:
                if self.require_external_actionness:
                    raise ValueError(f"external actionness is required for metas[{batch_idx}]")
                return None
            if self.external_actionness_provenance_meta_key is None:
                raise ValueError("external actionness provenance meta key is required")
            provenance = self._lookup_meta_value(meta, self.external_actionness_provenance_meta_key)
            validate_actionness_provenance(provenance, context=f"external actionness provenance metas[{batch_idx}]")
            source_name = None
            if self.external_actionness_source_meta_key:
                source_name = self._lookup_meta_value(meta, self.external_actionness_source_meta_key)
            source_name = str(source_name or provenance.get("source_name") or "external_actionness")
            source_names.append(source_name)
            provenances.append(dict(provenance))
            if p_value is not None:
                p_rows.append(self._actionness_row_tensor(p_value, descriptors, batch_idx, name="external p_action"))
            if logits_value is not None:
                logit_rows.append(
                    self._actionness_row_tensor(logits_value, descriptors, batch_idx, name="external actionness logits")
                )
        if len(set(source_names)) != 1:
            raise ValueError(f"external actionness source must be identical within a batch, got {source_names}")
        first_provenance = provenances[0]
        for provenance in provenances[1:]:
            for key in ("thumos_trained", "uses_labels", "uses_teacher", "uses_gt", "uses_prediction_cache"):
                if provenance.get(key) != first_provenance.get(key):
                    raise ValueError(f"external actionness provenance field {key} differs within batch")

        output: dict[str, Any] = {
            "source_name": source_names[0],
            "provenance": dict(first_provenance),
        }
        if p_rows:
            output["p_action"] = torch.stack(p_rows, dim=0)
        if logit_rows:
            output["actionness_logits"] = torch.stack(logit_rows, dim=0)
        return output

    @staticmethod
    def _lookup_meta_value(meta: Mapping[str, Any], key: Optional[str]) -> Any:
        if not key:
            return None
        if key in meta:
            return meta[key]
        current: Any = meta
        for part in str(key).split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _actionness_row_tensor(value: Any, reference: torch.Tensor, batch_idx: int, *, name: str) -> torch.Tensor:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
        tensor = tensor.to(device=reference.device, dtype=torch.float32)
        if tensor.ndim != 1:
            raise ValueError(f"{name} for metas[{batch_idx}] must be a 1-D sequence")
        if tensor.numel() != reference.shape[1]:
            raise ValueError(
                f"{name} for metas[{batch_idx}] length {tensor.numel()} must match dense window {reference.shape[1]}"
            )
        return tensor

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

    def _remap_train_targets_to_selected_axis(self, gt_segments, gt_labels, metas):
        if gt_segments is None:
            return gt_segments, gt_labels, metas
        if not self.remap_gt_to_selected_axis:
            return gt_segments, gt_labels, metas
        if len(gt_segments) != len(metas):
            raise ValueError("gt_segments length must match metas length")
        remapped_segments = []
        updated_metas = [dict(meta) for meta in metas]
        for idx, (segments, meta) in enumerate(zip(gt_segments, updated_metas)):
            if segments is None:
                remapped_segments.append(segments)
                continue
            positions = meta.get("selected_axis_to_true_time_dense_index")
            if not positions:
                raise ValueError("DUCA GT remap requires selected_axis_to_true_time_dense_index metadata")
            segments_tensor = segments if torch.is_tensor(segments) else torch.as_tensor(segments, dtype=torch.float32)
            true_map = TrueTimeMap(
                positions,
                dense_len=int(meta.get("truetime_dense_len", max(positions) + 1)),
                valid_len=int(meta.get("truetime_dense_valid_len", meta.get("truetime_dense_len", max(positions) + 1))),
            )
            remapped = true_map.remap_segments(
                segments_tensor,
                source_coordinate_space=TRUE_TIME_AXIS,
                target_coordinate_space=SELECTED_AXIS,
            ).to(device=segments_tensor.device, dtype=segments_tensor.dtype)
            remapped_segments.append(remapped)
            meta["gt_segments_original_time"] = segments_tensor.detach().cpu().tolist()
            meta["gt_segments_selected_axis"] = remapped.detach().cpu().tolist()
            meta["gt_remapped_to_selected_axis"] = True
            meta["gt_coordinate_space"] = SELECTED_AXIS
            meta["gt_original_coordinate_space"] = TRUE_TIME_AXIS
            updated_metas[idx] = meta
        return remapped_segments, gt_labels, updated_metas

    def _write_metas(self, metas, grid, *, actionness_source_name: str) -> list[dict[str, Any]]:
        batch = int(grid.selected_positions.shape[0])
        if metas is None:
            out = [{} for _ in range(batch)]
        else:
            if len(metas) != batch:
                raise ValueError("metas length must match batch size")
            out = [dict(meta) for meta in metas]
        positions_cpu = grid.selected_positions.detach().cpu().long()
        valid_lens = grid.valid_len.detach().cpu().long()
        requested_budget = grid.requested_budget.detach().cpu().long()
        effective_budget = grid.effective_budget.detach().cpu().long()
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
            meta["duca_online_requested_budget"] = int(requested_budget[idx].item())
            meta["duca_online_effective_budget"] = int(effective_budget[idx].item())
            meta["duca_online_dynamic_budget"] = bool(grid.metadata.get("budget_is_dynamic", False))
            meta["duca_online_budget_policy"] = str(grid.metadata.get("budget_policy", "fixed_budget"))
            meta["duca_online_budget_target"] = float(grid.metadata.get("budget_target", float(grid.budget)))
            meta["duca_online_budget_multiple"] = int(grid.metadata.get("budget_multiple", 1))
            meta["duca_online_selected_count"] = len(positions)
            meta["duca_online_selected_axis_remap"] = remap
            meta["duca_online_actionness_source"] = actionness_source_name
            meta["duca_online_budget_unit"] = grid.budget_unit
            meta["duca_online_coordinate"] = grid.coordinate
            meta["detector_output_coordinate_space"] = self.detector_output_coordinate_space
            meta["detector_prediction_inverse_map_required"] = self.detector_output_coordinate_space == SELECTED_AXIS
            meta["selected_axis_to_true_time_dense_index"] = positions
            meta["truetime_selected_positions"] = positions
            meta["truetime_dense_len"] = int(grid.original_length)
            meta["truetime_dense_valid_len"] = dense_valid_len
            meta["irregular_selected_positions"] = positions
            meta["irregular_native_axis"] = True
            meta["irregular_selected_count"] = len(positions)
            meta["irregular_dense_valid_len"] = dense_valid_len
            meta["irregular_selected_valid_len"] = len(positions)
            meta[self.metadata_keys["selected_positions"]] = positions
            meta[self.metadata_keys["selected_positions_unit"]] = self.selected_positions_unit
            meta[self.metadata_keys["selected_mask"]] = [True] * len(positions)
            meta[self.metadata_keys["selected_count"]] = len(positions)
            meta[self.metadata_keys["remap"]] = remap
            meta[self.metadata_keys["source"]] = actionness_source_name
        return out
