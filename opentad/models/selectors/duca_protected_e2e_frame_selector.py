from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn

from ..builder import SELECTORS
from ..duca.acquisition import (
    C3CoarseProbeActionnessSource,
    _assert_no_forbidden_payload,
)
from ..duca.structured_selection import (
    PhysicalExactKHardOutput,
    PhysicalExactKSelectionOutput,
    exact_uniform_positions,
    physical_exact_k_forward_backward,
    physical_exact_k_homotopy_log_potential,
    physical_exact_k_select,
    physical_exact_k_viterbi,
    physical_exact_uniform_gap_cap,
)
from ..duca.transition_only import (
    ASFORMER_ENCODER_HIDDEN_KIND,
    DucaProtectedTransitionScorer,
    balanced_binary_actionness_loss,
    coverage_floor_distribution,
    local_boundary_mass_coverage_loss,
    transition_distribution_loss,
    transition_utility_paths,
)
from ..utils.truetime_geometry import (
    SELECTED_AXIS,
    TRUE_TIME_AXIS,
    truetime_map_from_metadata,
)


_ARMS = {
    "exact_uniform",
    "probe_uniform",
    "transition_no_bridge",
    "protected_e2e",
    "protected_e2e_bridge025",
    "protected_e2e_homotopy025",
    "protected_e2e_uni_companion",
    "protected_e2e_rho001",
}
_PROTECTED_CONTRACT = "duca_protected_e2e_physical_v1"
_PROTECTED_SELECTED_AXIS_CONTRACT = "duca_protected_e2e_selected_axis_plugin_v2"
_DETECTOR_COORDINATE_MODES = {
    "physical_head_integration",
    "selected_axis_plugin",
}
_DETECTOR_BRIDGE_SCALES = {
    "exact_uniform": 0.0,
    "probe_uniform": 0.0,
    "transition_no_bridge": 0.0,
    "protected_e2e": 1.0,
    "protected_e2e_bridge025": 0.25,
    "protected_e2e_homotopy025": 0.25,
    "protected_e2e_uni_companion": 0.25,
    "protected_e2e_rho001": 1.0,
}
_UNIFORM_COMPANION_FRACTIONS = {
    "exact_uniform": 0.0,
    "probe_uniform": 0.0,
    "transition_no_bridge": 0.0,
    "protected_e2e": 0.0,
    "protected_e2e_bridge025": 0.0,
    "protected_e2e_homotopy025": 0.0,
    "protected_e2e_uni_companion": 0.50,
    "protected_e2e_rho001": 0.0,
}


def _append_jsonl_atomic(path: str, row: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short write while appending the protected inference ledger")
    finally:
        os.close(descriptor)


def _emit_protected_inference_ledger(
    *,
    arm: str,
    budget: int,
    metas,
) -> None:
    ledger_root = os.environ.get("DUCA_RIME_INFERENCE_LEDGER_ROOT", "").strip()
    if not ledger_root:
        return
    budget_protocol_sha256 = os.environ.get(
        "DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256",
        "",
    ).strip().lower()
    if (
        len(budget_protocol_sha256) != 64
        or any(value not in "0123456789abcdef" for value in budget_protocol_sha256)
    ):
        raise ValueError(
            "protected inference ledger requires an exact physical-protocol SHA-256"
        )
    ledger_root = os.path.abspath(
        os.path.expandvars(os.path.expanduser(ledger_root))
    )
    os.makedirs(ledger_root, exist_ok=True)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    ledger_path = os.path.join(
        ledger_root,
        f"inference_ledger.rank{rank:04d}.jsonl",
    )
    for meta in metas:
        video_id = str(meta.get("video_name") or meta.get("video_id") or "")
        window_start = int(meta.get("window_start_frame", 0))
        selected = [int(value) for value in meta.get("selected_dense_indices", ())]
        effective_k = int(meta.get("selected_valid_len", len(selected)))
        dense_valid_len = int(meta.get("irregular_dense_valid_len", -1))
        if (
            not video_id
            or window_start < 0
            or effective_k <= 0
            or dense_valid_len < effective_k
            or len(selected) != effective_k
            or selected != sorted(set(selected))
            or any(value < 0 or value >= dense_valid_len for value in selected)
        ):
            raise ValueError("protected inference ledger has an invalid window or hard path")
        _append_jsonl_atomic(
            ledger_path,
            {
                "schema_version": "duca_rime_inference_ledger_v1",
                "video_id": video_id,
                "window_start_frame": window_start,
                "arm": str(arm),
                "candidate_budgets": [int(budget)],
                "requested_k": int(budget),
                "effective_k": effective_k,
                "raw_budget": int(budget),
                "reachable_budget": effective_k,
                "realized_budget": effective_k,
                "projection_unused_budget": int(budget) - effective_k,
                "solver_unused_budget": 0,
                "budget_scope": "window_fixed_request",
                "unique_k": len(selected),
                "backbone_input_k": effective_k,
                "padded_k": effective_k,
                "risk_fallback": False,
                "cost_unit": "heavy_rgb_frames",
                "dense_valid_len": dense_valid_len,
                "selected_dense_indices": selected,
                "max_gap_seconds_cap": float(meta["duca_max_gap_seconds_cap"]),
                "observed_max_gap_seconds": float(
                    meta["duca_observed_max_gap_seconds"]
                ),
                "budget_protocol_sha256": budget_protocol_sha256,
                "claim_scope": "stage0_engineering_window_execution",
                "provenance": {
                    "task": "offline_temporal_action_detection",
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_prediction_cache": False,
                    "uses_test_batch_composition": False,
                    "raw_predictions_stored": False,
                },
            },
        )


def _time_dim(inputs: torch.Tensor) -> int:
    if inputs.ndim in {3, 5}:
        return 2
    if inputs.ndim == 6:
        return 3
    raise ValueError("protected DUCA expects [B,C,T], [B,C,T,H,W], or [B,N,C,T,H,W]")


def _hard_gather(
    inputs: torch.Tensor,
    positions: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    temporal_dim = _time_dim(inputs)
    positions = positions.to(device=inputs.device, dtype=torch.long)
    slot_mask = slot_mask.to(device=inputs.device, dtype=torch.bool)
    if positions.shape != slot_mask.shape or positions.ndim != 2:
        raise ValueError("hard positions and slot mask must be aligned [B,K]")
    effective_k = slot_mask.sum(dim=1)
    if bool(torch.any(effective_k <= 0).item()):
        raise ValueError("hard gather requires at least one active slot per sample")
    prefix = (
        torch.arange(slot_mask.shape[1], device=slot_mask.device)[None]
        < effective_k[:, None]
    )
    if not torch.equal(slot_mask, prefix):
        raise ValueError("hard gather requires a contiguous active slot prefix")
    unique_effective = torch.unique(effective_k)
    if int(unique_effective.numel()) != 1:
        raise ValueError(
            "hard gather requires one homogeneous effective-K execution bucket"
        )
    execution_k = int(unique_effective[0].item())
    safe = positions[:, :execution_k]
    if bool(torch.any(safe < 0).item()):
        raise ValueError("hard gather received an invalid active position")
    view = [safe.shape[0]] + [1] * (inputs.ndim - 1)
    view[temporal_dim] = safe.shape[1]
    expand = list(inputs.shape)
    expand[temporal_dim] = safe.shape[1]
    return torch.gather(inputs, temporal_dim, safe.view(view).expand(expand))


def _homogeneous_quantized_effective_k(
    valid_mask: torch.Tensor,
    *,
    requested_k: int,
    execution_quantum: int,
) -> int:
    if valid_mask.ndim != 2:
        raise ValueError("effective-K decoding requires a [B,T] valid mask")
    requested = int(requested_k)
    quantum = int(execution_quantum)
    if requested <= 0 or quantum <= 0 or requested % quantum:
        raise ValueError(
            "requested K must be positive and divisible by the execution quantum"
        )
    valid_lens = valid_mask.to(dtype=torch.long).sum(dim=1)
    effective = torch.minimum(
        valid_lens,
        torch.full_like(valid_lens, requested),
    )
    effective = torch.div(effective, quantum, rounding_mode="floor") * quantum
    if bool(torch.any(effective <= 0).item()):
        raise ValueError(
            "no positive quantum-aligned heavy execution is feasible for this window"
        )
    unique = torch.unique(effective)
    if int(unique.numel()) != 1:
        raise ValueError(
            "protected DUCA requires homogeneous effective-K buckets before "
            "heavy execution"
        )
    return int(unique[0].item())


def _soft_resample(inputs: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    work_inputs = inputs if inputs.is_floating_point() else inputs.float()
    weights = assignment.to(device=work_inputs.device, dtype=work_inputs.dtype)
    if work_inputs.ndim == 3:
        return torch.einsum("bct,bkt->bck", work_inputs, weights)
    if work_inputs.ndim == 5:
        return torch.einsum("bcthw,bkt->bckhw", work_inputs, weights)
    if work_inputs.ndim == 6:
        return torch.einsum("bncthw,bkt->bnckhw", work_inputs, weights)
    raise ValueError("protected DUCA expects [B,C,T], [B,C,T,H,W], or [B,N,C,T,H,W]")


def _exact_uniform_hard(
    valid_mask: torch.Tensor,
    *,
    k: int,
    dtype: torch.dtype,
) -> PhysicalExactKHardOutput:
    batch, temporal_len = valid_mask.shape
    hard_rows = []
    slot_rows = []
    position_rows = []
    slot_mask_rows = []
    effective_rows = []
    for batch_idx in range(batch):
        valid_len = int(valid_mask[batch_idx].sum().item())
        if valid_len <= 0:
            raise ValueError("exact-uniform selection requires one valid candidate")
        effective_k = min(int(k), valid_len)
        active = exact_uniform_positions(
            valid_len,
            effective_k,
            device=valid_mask.device,
        )
        positions = torch.full(
            (int(k),),
            -1,
            device=valid_mask.device,
            dtype=torch.long,
        )
        positions[:effective_k] = active
        slot_mask = torch.zeros(
            (int(k),),
            device=valid_mask.device,
            dtype=torch.bool,
        )
        slot_mask[:effective_k] = True
        hard = torch.zeros(
            (temporal_len,),
            device=valid_mask.device,
            dtype=dtype,
        )
        hard.scatter_(0, active, 1.0)
        slots = torch.zeros(
            (int(k), temporal_len),
            device=valid_mask.device,
            dtype=dtype,
        )
        slots[:effective_k].scatter_(1, active[:, None], 1.0)
        hard_rows.append(hard)
        slot_rows.append(slots)
        position_rows.append(positions)
        slot_mask_rows.append(slot_mask)
        effective_rows.append(effective_k)
    zeros = torch.zeros(batch, device=valid_mask.device, dtype=torch.float32)
    return PhysicalExactKHardOutput(
        hard_occupancy=torch.stack(hard_rows),
        hard_slot_assignment=torch.stack(slot_rows),
        hard_positions=torch.stack(position_rows),
        hard_slot_mask=torch.stack(slot_mask_rows),
        edge_count=torch.zeros(batch, device=valid_mask.device, dtype=torch.long),
        effective_k=torch.tensor(
            effective_rows,
            device=valid_mask.device,
            dtype=torch.long,
        ),
        max_gap_seconds=zeros,
    )


def _sample_uniform_companion_mask(
    batch_size: int,
    *,
    fraction: float,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if batch_size <= 1 or fraction <= 0.0:
        return mask
    uniform_count = max(1, int(round(float(batch_size) * float(fraction))))
    uniform_count = min(uniform_count, batch_size - 1)
    permutation = torch.randperm(batch_size, device=device)
    mask[permutation[:uniform_count]] = True
    return mask


def _blend_hard_outputs(
    learned: PhysicalExactKHardOutput,
    uniform: PhysicalExactKHardOutput,
    uniform_mask: torch.Tensor,
) -> PhysicalExactKHardOutput:
    if uniform_mask.ndim != 1:
        raise ValueError("uniform companion mask must be [B]")
    if learned.hard_positions.shape[0] != uniform_mask.shape[0]:
        raise ValueError("uniform companion mask batch does not match hard paths")
    mask_bt = uniform_mask[:, None]
    mask_bkt = uniform_mask[:, None, None]
    return PhysicalExactKHardOutput(
        hard_occupancy=torch.where(
            mask_bt,
            uniform.hard_occupancy,
            learned.hard_occupancy,
        ),
        hard_slot_assignment=torch.where(
            mask_bkt,
            uniform.hard_slot_assignment,
            learned.hard_slot_assignment,
        ),
        hard_positions=torch.where(
            mask_bt,
            uniform.hard_positions,
            learned.hard_positions,
        ),
        hard_slot_mask=torch.where(
            mask_bt,
            uniform.hard_slot_mask,
            learned.hard_slot_mask,
        ),
        edge_count=torch.where(
            uniform_mask,
            uniform.edge_count,
            learned.edge_count,
        ),
        effective_k=torch.where(
            uniform_mask,
            uniform.effective_k,
            learned.effective_k,
        ),
        max_gap_seconds=torch.where(
            uniform_mask,
            uniform.max_gap_seconds,
            learned.max_gap_seconds,
        ),
    )


def _action_target_from_gt_segments(
    gt_segments,
    valid_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    if gt_segments is None:
        return None
    if len(gt_segments) != int(valid_mask.shape[0]):
        raise ValueError("gt_segments batch must match protected DUCA inputs")
    temporal_len = int(valid_mask.shape[1])
    centers = torch.arange(
        temporal_len,
        device=valid_mask.device,
        dtype=torch.float32,
    )
    target = torch.zeros_like(valid_mask, dtype=torch.float32)
    for batch_idx, segments in enumerate(gt_segments):
        if segments is None:
            continue
        row = torch.as_tensor(
            segments,
            device=valid_mask.device,
            dtype=torch.float32,
        ).reshape(-1, 2)
        if row.numel() == 0:
            continue
        starts = torch.minimum(row[:, 0], row[:, 1])[:, None]
        ends = torch.maximum(row[:, 0], row[:, 1])[:, None]
        target[batch_idx] = ((centers[None] >= starts) & (centers[None] < ends)).any(
            dim=0
        )
    return target.masked_fill(~valid_mask, 0.0)


def _transition_target_from_gt_segments(
    gt_segments,
    valid_mask: torch.Tensor,
    *,
    sigma: float,
    radius: int,
    boundary_validity=None,
) -> Optional[torch.Tensor]:
    if gt_segments is None:
        return None
    if len(gt_segments) != int(valid_mask.shape[0]):
        raise ValueError("gt_segments batch must match protected DUCA inputs")
    if boundary_validity is not None and len(boundary_validity) != len(gt_segments):
        raise ValueError("gt_boundary_validity batch must match protected DUCA inputs")
    temporal_len = int(valid_mask.shape[1])
    centers = torch.arange(
        temporal_len,
        device=valid_mask.device,
        dtype=torch.float32,
    )
    target = torch.zeros_like(valid_mask, dtype=torch.float32)
    for batch_idx, segments in enumerate(gt_segments):
        if segments is None:
            continue
        row = torch.as_tensor(
            segments,
            device=valid_mask.device,
            dtype=torch.float32,
        ).reshape(-1, 2)
        if row.numel() == 0:
            continue
        endpoint_matrix = torch.stack(
            (
                torch.minimum(row[:, 0], row[:, 1]),
                torch.maximum(row[:, 0], row[:, 1]),
            ),
            dim=1,
        )
        if boundary_validity is None:
            endpoint_validity = torch.ones_like(
                endpoint_matrix,
                dtype=torch.bool,
            )
        else:
            endpoint_validity = torch.as_tensor(
                boundary_validity[batch_idx],
                device=valid_mask.device,
                dtype=torch.bool,
            ).reshape(-1, 2)
            if endpoint_validity.shape != endpoint_matrix.shape:
                raise ValueError("gt_boundary_validity must align with GT segments")
        endpoints = endpoint_matrix[endpoint_validity]
        if endpoints.numel() == 0:
            continue
        endpoint_mass = 1.0 / float(endpoints.numel())
        row_valid = valid_mask[batch_idx].to(dtype=torch.float32)
        for endpoint in endpoints:
            distance = centers - endpoint
            kernel = torch.exp(-0.5 * (distance / float(sigma)).square())
            kernel *= (distance.abs() <= float(radius)).to(kernel.dtype)
            kernel *= row_valid
            mass = kernel.sum()
            if float(mass.detach().item()) > 0.0:
                target[batch_idx] += endpoint_mass * kernel / mass
    return target.masked_fill(~valid_mask, 0.0)


@SELECTORS.register_module()
class DucaProtectedE2EFrameSelector(nn.Module):
    """Offline pre-backbone exact-K selector with protected detector feedback."""

    def __init__(
        self,
        in_channels: int,
        arm: str,
        budget: int = 384,
        dense_window_size: int = 768,
        coarse_hidden_dim: int = 96,
        selector_hidden_dim: int = 64,
        coverage_floor_weight: float = 0.10,
        score_temperature: float = 0.70,
        path_temperature: float = 1.0,
        transition_target_sigma: float = 2.0,
        transition_target_radius: int = 4,
        transition_boundary_radius: int = 4,
        transition_distribution_temperature: float = 0.70,
        action_loss_weight: float = 1.0,
        transition_loss_weight: float = 0.50,
        transition_boundary_loss_weight: float = 0.25,
        coarse_trunk_lr: float = 2.5e-5,
        action_head_lr: float = 5.0e-5,
        selector_lr: float = 1.0e-4,
        detector_bridge_gradient_scale: Optional[float] = None,
        uniform_companion_fraction: Optional[float] = None,
        homotopy_total_steps: int = 0,
        execution_quantum: int = 1,
        actionness_source_cfg: Optional[Mapping[str, Any]] = None,
        strict_physical_metadata: bool = True,
        forbid_raw_prediction_cache: bool = True,
        detector_coordinate_mode: str = "physical_head_integration",
        **extra_config: Any,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.arm = str(arm)
        self.budget = int(budget)
        self.dense_window_size = int(dense_window_size)
        self.coarse_hidden_dim = int(coarse_hidden_dim)
        self.selector_hidden_dim = int(selector_hidden_dim)
        self.coverage_floor_weight = float(coverage_floor_weight)
        self.score_temperature = float(score_temperature)
        self.path_temperature = float(path_temperature)
        self.transition_target_sigma = float(transition_target_sigma)
        self.transition_target_radius = int(transition_target_radius)
        self.transition_boundary_radius = int(transition_boundary_radius)
        self.transition_distribution_temperature = float(
            transition_distribution_temperature
        )
        self.action_loss_weight = float(action_loss_weight)
        self.transition_loss_weight = float(transition_loss_weight)
        self.transition_boundary_loss_weight = float(transition_boundary_loss_weight)
        self.coarse_trunk_lr = float(coarse_trunk_lr)
        self.action_head_lr = float(action_head_lr)
        self.selector_lr = float(selector_lr)
        self.homotopy_total_steps = int(homotopy_total_steps)
        self.execution_quantum = int(execution_quantum)
        self.homotopy_warmup_steps = round(self.homotopy_total_steps * 0.05)
        self.homotopy_transition_steps = round(self.homotopy_total_steps * 0.30)
        if self.arm not in _ARMS:
            raise ValueError(f"arm must be one of {sorted(_ARMS)}")
        expected_bridge_scale = _DETECTOR_BRIDGE_SCALES.get(self.arm)
        expected_companion_fraction = _UNIFORM_COMPANION_FRACTIONS.get(self.arm)
        self.detector_bridge_gradient_scale = float(
            expected_bridge_scale
            if detector_bridge_gradient_scale is None
            else detector_bridge_gradient_scale
        )
        self.uniform_companion_fraction = float(
            expected_companion_fraction
            if uniform_companion_fraction is None
            else uniform_companion_fraction
        )
        self.strict_physical_metadata = bool(strict_physical_metadata)
        self.forbid_raw_prediction_cache = bool(forbid_raw_prediction_cache)
        self.detector_coordinate_mode = str(detector_coordinate_mode)
        if self.detector_coordinate_mode not in _DETECTOR_COORDINATE_MODES:
            raise ValueError(
                "detector_coordinate_mode must be one of "
                f"{sorted(_DETECTOR_COORDINATE_MODES)}"
            )
        self.extra_config = dict(extra_config)

        if not math.isclose(
            self.detector_bridge_gradient_scale,
            float(_DETECTOR_BRIDGE_SCALES[self.arm]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"{self.arm} requires detector_bridge_gradient_scale="
                f"{_DETECTOR_BRIDGE_SCALES[self.arm]}"
            )
        if not math.isclose(
            self.uniform_companion_fraction,
            float(_UNIFORM_COMPANION_FRACTIONS[self.arm]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"{self.arm} requires uniform_companion_fraction="
                f"{_UNIFORM_COMPANION_FRACTIONS[self.arm]}"
            )
        if self.budget <= 0 or self.dense_window_size <= 0:
            raise ValueError("budget and dense_window_size must be positive")
        if self.budget > self.dense_window_size:
            raise ValueError("budget cannot exceed dense_window_size")
        if self.execution_quantum <= 0 or self.budget % self.execution_quantum:
            raise ValueError(
                "budget must be divisible by the protected execution quantum"
            )
        if self.coarse_hidden_dim != 96 or self.selector_hidden_dim != 64:
            raise ValueError(
                "protected DUCA freezes ASFormer hidden=96 and selector hidden=64"
            )
        if not 0.0 <= self.coverage_floor_weight < 1.0:
            raise ValueError("coverage_floor_weight must lie in [0,1)")
        if (
            min(
                self.score_temperature,
                self.path_temperature,
                self.transition_target_sigma,
                self.transition_distribution_temperature,
            )
            <= 0.0
        ):
            raise ValueError("protected DUCA temperatures and sigma must be positive")
        if self.transition_target_radius < 0 or self.transition_boundary_radius < 0:
            raise ValueError("transition radii must be non-negative")
        if (
            min(
                self.action_loss_weight,
                self.transition_loss_weight,
                self.transition_boundary_loss_weight,
            )
            < 0.0
        ):
            raise ValueError("protected DUCA loss weights must be non-negative")
        if min(self.coarse_trunk_lr, self.action_head_lr, self.selector_lr) <= 0.0:
            raise ValueError("protected DUCA learning rates must be positive")
        if self.arm == "protected_e2e_homotopy025":
            if self.homotopy_total_steps <= 0:
                raise ValueError("homotopy_total_steps must be positive")
            if self.homotopy_warmup_steps <= 0:
                raise ValueError("homotopy warmup must contain at least one step")
            if self.homotopy_transition_steps <= 0:
                raise ValueError("homotopy transition must contain at least one step")
            if (
                self.homotopy_warmup_steps + self.homotopy_transition_steps
                > self.homotopy_total_steps
            ):
                raise ValueError("homotopy phases exceed the total-step contract")
            self.register_buffer(
                "schedule_step",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )

        selected_axis_plugin = (
            self.detector_coordinate_mode == "selected_axis_plugin"
        )
        self.selector_variant = (
            "protected_e2e_selected_axis"
            if selected_axis_plugin
            else "protected_e2e_physical"
        )
        self.detector_coordinate_contract = (
            _PROTECTED_SELECTED_AXIS_CONTRACT
            if selected_axis_plugin
            else _PROTECTED_CONTRACT
        )
        self.require_counterfactual_utility_teacher = False
        self.remap_gt_to_selected_axis = selected_axis_plugin
        self.selected_axis_remap_required = selected_axis_plugin
        self.no_ledger_decision = True
        self.detector_output_coordinate_space = (
            SELECTED_AXIS if selected_axis_plugin else "dense_physical"
        )
        self.detector_prediction_inverse_map_required = selected_axis_plugin
        self.separate_detector_rng = True
        self.last_forward_summary: dict[str, Any] = {}
        self._last_selected_positions: Optional[torch.Tensor] = None
        self._last_physical_metas: Optional[list[dict[str, Any]]] = None
        self.capture_policy_score_gradients = False
        self._last_policy_scores: Optional[torch.Tensor] = None
        self._pending_homotopy_schedule_advance = False

        self.raw_actionness_source: Optional[C3CoarseProbeActionnessSource]
        self.transition_scorer: Optional[DucaProtectedTransitionScorer]
        if self.arm == "exact_uniform":
            if actionness_source_cfg:
                raise ValueError("exact_uniform must skip the coarse network entirely")
            self.raw_actionness_source = None
            self.transition_scorer = None
            self.policy_hidden_gradient_scale = 0.0
        else:
            cfg = dict(actionness_source_cfg or {})
            source_type = cfg.pop("type", "C3CoarseProbeActionnessSource")
            if source_type != "C3CoarseProbeActionnessSource":
                raise ValueError(
                    "protected DUCA requires C3CoarseProbeActionnessSource"
                )
            if str(cfg.get("probe_model", "")) != "official-action-seg":
                raise ValueError(
                    "protected DUCA requires probe_model='official-action-seg'"
                )
            if str(cfg.get("official_action_seg_backend", "")) != "official_asformer":
                raise ValueError(
                    "protected DUCA requires the official ASFormer implementation"
                )
            if bool(cfg.get("frozen", False)) or cfg.get("trainable") is False:
                raise ValueError("protected DUCA requires a trainable coarse ASFormer")
            cfg.setdefault("tcn_hidden_dim", self.coarse_hidden_dim)
            cfg.setdefault("return_hidden_features", True)
            cfg.setdefault("require_hidden_features", True)
            cfg.setdefault("hidden_output_kind", ASFORMER_ENCODER_HIDDEN_KIND)
            cfg.setdefault("forbid_external_actionness", True)
            expected_scope = (
                "asformer_last_encoder_layer"
                if self.arm == "protected_e2e_rho001"
                else "none"
            )
            cfg.setdefault("policy_hidden_gradient_scope", expected_scope)
            if str(cfg["policy_hidden_gradient_scope"]) != expected_scope:
                raise ValueError(
                    f"{self.arm} requires policy_hidden_gradient_scope={expected_scope!r}"
                )
            self.raw_actionness_source = C3CoarseProbeActionnessSource(**cfg)
            self.transition_scorer = (
                None
                if self.arm == "probe_uniform"
                else DucaProtectedTransitionScorer(
                    hidden_dim=self.coarse_hidden_dim,
                    scorer_hidden_dim=self.selector_hidden_dim,
                )
            )
            self.policy_hidden_gradient_scale = (
                0.01 if self.arm == "protected_e2e_rho001" else 0.0
            )

    def capture_amp_replay_state(self) -> dict[str, Any]:
        return {
            "last_forward_summary": copy.deepcopy(self.last_forward_summary),
            "last_selected_positions": (
                None
                if self._last_selected_positions is None
                else self._last_selected_positions.detach().clone()
            ),
            "last_physical_metas": copy.deepcopy(self._last_physical_metas),
            "last_policy_scores": (
                None
                if self._last_policy_scores is None
                else self._last_policy_scores.detach().clone()
            ),
            "pending_homotopy_schedule_advance": bool(
                self._pending_homotopy_schedule_advance
            ),
        }

    def restore_amp_replay_state(self, snapshot: Mapping[str, Any]) -> None:
        self.last_forward_summary = copy.deepcopy(
            snapshot.get("last_forward_summary", {})
        )
        selected = snapshot.get("last_selected_positions")
        self._last_selected_positions = (
            None if selected is None else selected.detach().clone()
        )
        self._last_physical_metas = copy.deepcopy(snapshot.get("last_physical_metas"))
        policy_scores = snapshot.get("last_policy_scores")
        self._last_policy_scores = (
            None if policy_scores is None else policy_scores.detach().clone()
        )
        self._pending_homotopy_schedule_advance = bool(
            snapshot.get("pending_homotopy_schedule_advance", False)
        )

    def _policy_homotopy_state(self, *, training: bool) -> dict[str, Any]:
        if self.arm != "protected_e2e_homotopy025":
            return {"enabled": False, "alpha": 1.0}
        step = int(self.schedule_step.detach().item())
        if not training:
            alpha = 1.0
            phase = "inference_learned_policy"
        elif step < self.homotopy_warmup_steps:
            alpha = 0.0
            phase = "exact_uniform_warmup"
        elif step >= self.homotopy_warmup_steps + self.homotopy_transition_steps:
            alpha = 1.0
            phase = "learned_policy"
        else:
            progress = (
                float(step - self.homotopy_warmup_steps + 1)
                / float(self.homotopy_transition_steps)
            )
            alpha = 0.5 - 0.5 * math.cos(math.pi * progress)
            phase = "physical_exact_k_policy_homotopy"
        return {
            "enabled": True,
            "alpha_zero_contract": "hard_forward_exact_uniform",
            "alpha": float(alpha),
            "step": step,
            "total_steps": self.homotopy_total_steps,
            "warmup_steps": self.homotopy_warmup_steps,
            "transition_steps": self.homotopy_transition_steps,
            "shape": "cosine",
            "phase": phase,
            "inference_alpha": 1.0,
        }

    def after_optimizer_step(self) -> dict[str, Any]:
        if self.arm == "protected_e2e_homotopy025":
            if not self.training:
                return {"updated": False, "reason": "selector_not_training"}
            if not self._pending_homotopy_schedule_advance:
                return {"updated": False, "reason": "no_pending_homotopy_step"}
            before = int(self.schedule_step.detach().item())
            self.schedule_step.add_(1)
            self._pending_homotopy_schedule_advance = False
            summary = {
                "updated": True,
                "source": "successful_optimizer_step",
                "step_before": before,
                "step_after": int(self.schedule_step.detach().item()),
            }
            if isinstance(self.last_forward_summary, dict):
                self.last_forward_summary["schedule_step_update"] = summary
            return summary
        return {"updated": False, "reason": "protected_duca_has_no_selector_schedule"}

    def forward_train(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas,
        gt_segments=None,
        gt_labels=None,
        gt_boundary_validity=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._validate_inputs(inputs, masks, metas)
        self._reject_train_decision_payload(metas, kwargs)
        if self.arm == "probe_uniform":
            raise RuntimeError("probe_uniform is an inference-only Phase-1 cost control")
        action_target = _action_target_from_gt_segments(gt_segments, masks.bool())
        transition_target = _transition_target_from_gt_segments(
            gt_segments,
            masks.bool(),
            sigma=self.transition_target_sigma,
            radius=self.transition_target_radius,
            boundary_validity=gt_boundary_validity,
        )
        selected = self._select(
            inputs,
            masks.bool(),
            metas,
            training=True,
        )
        if self.arm == "protected_e2e_homotopy025":
            self._pending_homotopy_schedule_advance = True
        losses: dict[str, torch.Tensor] = {}
        selector_state = selected["selector_outputs"]
        if self.arm != "exact_uniform":
            if action_target is None or transition_target is None:
                raise ValueError("protected DUCA training requires GT segments")
            if gt_boundary_validity is None:
                raise ValueError(
                    "learned protected DUCA requires true-boundary validity"
                )
            action_loss, positive_weight = balanced_binary_actionness_loss(
                selector_state["actionness_logits"],
                action_target,
                masks,
            )
            transition_loss = transition_distribution_loss(
                selector_state["auxiliary_scores"],
                transition_target,
                masks,
                temperature=self.transition_distribution_temperature,
            )
            transition_boundary_loss = local_boundary_mass_coverage_loss(
                selector_state["auxiliary_soft_occupancy"],
                transition_target,
                masks,
                radius=self.transition_boundary_radius,
            )
            losses = {
                "selector_action_loss": action_loss * self.action_loss_weight,
                "selector_transition_loss": (
                    transition_loss * self.transition_loss_weight
                ),
                "selector_transition_boundary_loss": (
                    transition_boundary_loss * self.transition_boundary_loss_weight
                ),
            }
            selector_state["action_target"] = action_target
            selector_state["transition_target"] = transition_target
            selector_state["action_positive_weight"] = positive_weight.detach()
            selector_state["loss_weights"] = {
                "action": self.action_loss_weight,
                "transition": self.transition_loss_weight,
                "transition_boundary": self.transition_boundary_loss_weight,
            }
        selector_state["training_provenance"] = {
            "task": "offline_tad",
            "gt_scope": "train_only_auxiliary_targets",
            "inference_uses_gt": False,
            "counterfactual_teacher": False,
            "utility_distillation": False,
            "selected_axis_gt_remap": bool(self.remap_gt_to_selected_axis),
        }
        gt_segments, gt_labels, detector_metas = (
            self._remap_train_targets_to_selected_axis(
                gt_segments,
                gt_labels,
                selected["metas"],
            )
        )
        return {
            "inputs": selected["inputs"],
            "masks": selected["masks"],
            "metas": detector_metas,
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "losses": losses,
            "selector_outputs": selector_state,
            "counterfactual_request": None,
        }

    def forward_test(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._validate_inputs(inputs, masks, metas)
        _assert_no_forbidden_payload(
            {"metas": metas, "kwargs": kwargs},
            path="protected_duca_inference",
        )
        selected = self._select(
            inputs,
            masks.bool(),
            metas,
            training=False,
        )
        return {
            "inputs": selected["inputs"],
            "masks": selected["masks"],
            "metas": selected["metas"],
            "selector_outputs": selected["selector_outputs"],
        }

    def _select(
        self,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        metas,
        *,
        training: bool,
    ) -> dict[str, Any]:
        execution_k = _homogeneous_quantized_effective_k(
            valid_mask,
            requested_k=self.budget,
            execution_quantum=self.execution_quantum,
        )
        physical_seconds, source_frames = self._physical_axes(
            metas,
            valid_mask,
            inputs.device,
        )
        caps = physical_exact_uniform_gap_cap(
            physical_seconds,
            valid_mask,
            k=execution_k,
        )
        selector_state: dict[str, Any] = {
            "arm": self.arm,
            "contract": self.detector_coordinate_contract,
            "valid_mask": valid_mask,
            "physical_seconds": physical_seconds,
            "decoded_source_frames": source_frames,
            "max_gap_seconds": caps.to(dtype=torch.float32),
            "selection_scope": "full_window_offline",
            "budget": self.budget,
            "requested_k": self.budget,
            "effective_k": execution_k,
            "execution_quantum": self.execution_quantum,
        }
        homotopy_state = self._policy_homotopy_state(training=training)
        if homotopy_state["enabled"]:
            selector_state["policy_homotopy"] = homotopy_state

        learned_selection: Optional[PhysicalExactKSelectionOutput] = None
        uniform_companion_mask = torch.zeros(
            int(valid_mask.shape[0]),
            device=valid_mask.device,
            dtype=torch.bool,
        )
        if self.arm == "exact_uniform":
            hard = _exact_uniform_hard(
                valid_mask,
                k=execution_k,
                dtype=inputs.dtype,
            )
            hard = PhysicalExactKHardOutput(
                hard_occupancy=hard.hard_occupancy,
                hard_slot_assignment=hard.hard_slot_assignment,
                hard_positions=hard.hard_positions,
                hard_slot_mask=hard.hard_slot_mask,
                edge_count=hard.edge_count,
                effective_k=hard.effective_k,
                max_gap_seconds=caps,
            )
        elif self.arm == "probe_uniform":
            source = self.raw_actionness_source(
                inputs,
                valid_mask=valid_mask,
            )
            hidden = source.get("coarse_hidden_features")
            if (
                hidden is None
                or source.get("hidden_kind") != ASFORMER_ENCODER_HIDDEN_KIND
            ):
                raise RuntimeError(
                    "probe_uniform requires official ASFormer encoder hidden features"
                )
            hard = _exact_uniform_hard(
                valid_mask,
                k=execution_k,
                dtype=inputs.dtype,
            )
            hard = PhysicalExactKHardOutput(
                hard_occupancy=hard.hard_occupancy,
                hard_slot_assignment=hard.hard_slot_assignment,
                hard_positions=hard.hard_positions,
                hard_slot_mask=hard.hard_slot_mask,
                edge_count=hard.edge_count,
                effective_k=hard.effective_k,
                max_gap_seconds=caps,
            )
            selector_state.update(
                {
                    "actionness_logits": source["actionness_logits"],
                    "p_action": source["p_action"],
                    "coarse_hidden_features": hidden,
                    "hidden_kind": source["hidden_kind"],
                    "coarse_provenance": source["provenance"],
                    "coarse_compute_profile": source.get("compute_profile"),
                    "probe_output_used_for_selection": False,
                    "selection_policy": "exact_uniform",
                }
            )
        else:
            source = self.raw_actionness_source(
                inputs,
                valid_mask=valid_mask,
            )
            hidden = source.get("coarse_hidden_features")
            if (
                hidden is None
                or source.get("hidden_kind") != ASFORMER_ENCODER_HIDDEN_KIND
            ):
                raise RuntimeError(
                    "protected DUCA requires official ASFormer encoder hidden features"
                )
            policy_hidden = source.get("policy_hidden_features")
            if self.arm == "protected_e2e_rho001" and policy_hidden is None:
                raise RuntimeError(
                    "rho001 arm requires restricted last-block policy hidden"
                )
            paths = transition_utility_paths(
                self.transition_scorer,
                source["actionness_logits"],
                hidden,
                valid_mask,
                compute_auxiliary=training,
                policy_hidden=policy_hidden,
                policy_hidden_gradient_scale=self.policy_hidden_gradient_scale,
            )
            if training and self.capture_policy_score_gradients:
                paths["policy_scores"].retain_grad()
                self._last_policy_scores = paths["policy_scores"]
            else:
                self._last_policy_scores = None
            auxiliary_prob, auxiliary_log_prob = coverage_floor_distribution(
                paths["auxiliary_scores"],
                valid_mask,
                floor_weight=self.coverage_floor_weight,
                score_temperature=self.score_temperature,
            )
            policy_prob, policy_log_prob = coverage_floor_distribution(
                paths["policy_scores"],
                valid_mask,
                floor_weight=self.coverage_floor_weight,
                score_temperature=self.score_temperature,
            )
            if not torch.equal(
                auxiliary_prob.detach(),
                policy_prob.detach(),
            ):
                raise RuntimeError(
                    "auxiliary and policy coverage distributions must match"
                )
            policy_log_potential = policy_log_prob
            if homotopy_state["enabled"]:
                policy_log_potential = physical_exact_k_homotopy_log_potential(
                    policy_log_prob,
                    valid_mask,
                    k=execution_k,
                    alpha=float(homotopy_state["alpha"]),
                )
            if training:
                learned_selection = physical_exact_k_select(
                    policy_log_potential,
                    physical_seconds,
                    valid_mask,
                    k=execution_k,
                    max_gap_seconds=caps,
                    temperature=self.path_temperature,
                )
                hard = PhysicalExactKHardOutput(
                    hard_occupancy=learned_selection.hard_occupancy,
                    hard_slot_assignment=learned_selection.hard_slot_assignment,
                    hard_positions=learned_selection.hard_positions,
                    hard_slot_mask=learned_selection.hard_slot_mask,
                    edge_count=learned_selection.edge_count,
                    effective_k=learned_selection.effective_k,
                    max_gap_seconds=learned_selection.max_gap_seconds,
                )
                selector_state[
                    "learned_selected_positions"
                ] = learned_selection.hard_positions
                auxiliary_soft = physical_exact_k_forward_backward(
                    auxiliary_log_prob,
                    physical_seconds,
                    valid_mask,
                    k=execution_k,
                    max_gap_seconds=caps,
                    temperature=self.path_temperature,
                )
                selector_state[
                    "auxiliary_soft_occupancy"
                ] = auxiliary_soft.soft_occupancy
                selector_state[
                    "auxiliary_soft_slot_assignment"
                ] = auxiliary_soft.soft_slot_assignment
                if self.uniform_companion_fraction > 0.0:
                    uniform = _exact_uniform_hard(
                        valid_mask,
                        k=execution_k,
                        dtype=inputs.dtype,
                    )
                    uniform = PhysicalExactKHardOutput(
                        hard_occupancy=uniform.hard_occupancy,
                        hard_slot_assignment=uniform.hard_slot_assignment,
                        hard_positions=uniform.hard_positions,
                        hard_slot_mask=uniform.hard_slot_mask,
                        edge_count=uniform.edge_count,
                        effective_k=uniform.effective_k,
                        max_gap_seconds=caps,
                    )
                    uniform_companion_mask = _sample_uniform_companion_mask(
                        int(valid_mask.shape[0]),
                        fraction=self.uniform_companion_fraction,
                        device=valid_mask.device,
                    )
                    hard = _blend_hard_outputs(
                        hard,
                        uniform,
                        uniform_companion_mask,
                    )
                    selector_state[
                        "uniform_companion_positions"
                    ] = uniform.hard_positions
            else:
                hard = physical_exact_k_viterbi(
                    policy_log_potential,
                    physical_seconds,
                    valid_mask,
                    k=execution_k,
                    max_gap_seconds=caps,
                )
            selector_state.update(
                {
                    "actionness_logits": source["actionness_logits"],
                    "p_action": source["p_action"],
                    "coarse_hidden_features": hidden,
                    "hidden_kind": source["hidden_kind"],
                    "transition_descriptors": paths["transition_descriptors"],
                    "policy_descriptors": paths["policy_descriptors"],
                    "auxiliary_scores": paths["auxiliary_scores"],
                    "policy_scores": paths["policy_scores"],
                    "auxiliary_probabilities": auxiliary_prob,
                    "policy_probabilities": policy_prob,
                    "policy_log_probabilities": policy_log_prob,
                    "policy_log_potential": policy_log_potential,
                    "policy_hidden_gradient_scale": self.policy_hidden_gradient_scale,
                    "coarse_provenance": source["provenance"],
                    "coarse_compute_profile": source.get("compute_profile"),
                }
            )

        hard_selected = _hard_gather(
            inputs,
            hard.hard_positions,
            hard.hard_slot_mask,
        )
        detector_bridge = training and self.detector_bridge_gradient_scale > 0.0
        detector_bridge_mask = (
            ~uniform_companion_mask
            if detector_bridge
            else torch.zeros_like(uniform_companion_mask)
        )
        hard_detector_input = hard_selected
        if detector_bridge:
            bridge_assignment = learned_selection.soft_slot_assignment
            if bool(uniform_companion_mask.any().item()):
                bridge_assignment = torch.where(
                    uniform_companion_mask[:, None, None],
                    hard.hard_slot_assignment,
                    bridge_assignment,
                )
            soft_selected = _soft_resample(
                inputs,
                bridge_assignment,
            )
            hard_base = (
                hard_selected
                if hard_selected.is_floating_point()
                else hard_selected.float()
            )
            hard_detector_input = hard_base
            selected_inputs = hard_base + (
                self.detector_bridge_gradient_scale
                * (soft_selected - soft_selected.detach())
            )
            if not torch.equal(
                selected_inputs.detach(),
                hard_base.detach(),
            ):
                raise RuntimeError(
                    "protected DUCA detector input is not exact hard forward"
                )
        else:
            selected_inputs = hard_selected

        output_metas = self._write_detector_metadata(
            metas,
            hard,
            physical_seconds,
            source_frames,
            valid_mask,
        )
        if not training:
            _emit_protected_inference_ledger(
                arm=self.arm,
                budget=self.budget,
                metas=output_metas,
            )
        selector_state.update(
            {
                "hard_occupancy": hard.hard_occupancy,
                "hard_slot_assignment": hard.hard_slot_assignment,
                "selected_positions": hard.hard_positions,
                "selected_mask": hard.hard_slot_mask,
                "selected_count": hard.effective_k,
                "edge_count": hard.edge_count,
                "detector_gradient_bridge": detector_bridge,
                "detector_gradient_bridge_mask": detector_bridge_mask,
                "detector_bridge_gradient_scale": (self.detector_bridge_gradient_scale),
                "uniform_companion_mask": uniform_companion_mask,
                "uniform_companion_fraction": self.uniform_companion_fraction,
                "detector_input": selected_inputs,
                "hard_detector_input": hard_detector_input,
                "backbone_tail_padding_mode": "none_exact_k_bucket",
                "train_inference_hard_decoder": "same_physical_exact_k_viterbi",
            }
        )
        if learned_selection is not None:
            selector_state["soft_occupancy"] = learned_selection.soft_occupancy
            selector_state[
                "soft_slot_assignment"
            ] = learned_selection.soft_slot_assignment
            selector_state["selection_st"] = learned_selection.selection_st
            selector_state["log_partition"] = learned_selection.log_partition

        self.last_forward_summary = {
            "arm": self.arm,
            "training": bool(training),
            "selected_count": [
                int(value) for value in hard.effective_k.detach().cpu().tolist()
            ],
            "max_gap_seconds": [float(value) for value in caps.detach().cpu().tolist()],
            "detector_gradient_bridge": detector_bridge,
            "detector_bridge_gradient_scale": self.detector_bridge_gradient_scale,
            "uniform_companion_count": int(
                uniform_companion_mask.detach().sum().cpu().item()
            ),
            "learned_detector_count": int(
                detector_bridge_mask.detach().sum().cpu().item()
            ),
            "selected_axis_gt_remap": bool(self.remap_gt_to_selected_axis),
            "contract": self.detector_coordinate_contract,
        }
        if homotopy_state["enabled"]:
            self.last_forward_summary["policy_homotopy"] = dict(homotopy_state)
        self._last_selected_positions = hard.hard_positions.detach().clone()
        self._last_physical_metas = copy.deepcopy(output_metas)
        return {
            "inputs": selected_inputs,
            "masks": hard.hard_slot_mask,
            "metas": output_metas,
            "selector_outputs": selector_state,
        }

    def materialize_hard_positions(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas,
        positions: torch.Tensor,
    ) -> dict[str, Any]:
        """Materialize a fixed legal hard path for train-only P3 auditing."""

        self._validate_inputs(inputs, masks, metas)
        valid_mask = masks.to(device=inputs.device, dtype=torch.bool)
        execution_k = _homogeneous_quantized_effective_k(
            valid_mask,
            requested_k=self.budget,
            execution_quantum=self.execution_quantum,
        )
        positions = torch.as_tensor(
            positions,
            device=inputs.device,
            dtype=torch.long,
        )
        if positions.ndim != 2 or int(positions.shape[0]) != int(inputs.shape[0]):
            raise ValueError("fixed hard positions must be [B,K]")
        if int(positions.shape[1]) < execution_k:
            raise ValueError("fixed hard positions do not contain effective K")
        if int(positions.shape[1]) > execution_k and bool(
            torch.any(positions[:, execution_k:] >= 0).item()
        ):
            raise ValueError("fixed hard path activates an infeasible tail slot")
        positions = positions[:, :execution_k]
        physical_seconds, source_frames = self._physical_axes(
            metas,
            valid_mask,
            inputs.device,
        )
        caps = physical_exact_uniform_gap_cap(
            physical_seconds,
            valid_mask,
            k=execution_k,
        )
        batch, temporal_len = valid_mask.shape
        occupancy = torch.zeros(
            (batch, temporal_len),
            device=inputs.device,
            dtype=torch.float32,
        )
        slot_assignment = torch.zeros(
            (batch, execution_k, temporal_len),
            device=inputs.device,
            dtype=torch.float32,
        )
        slot_mask = positions >= 0
        effective_k = slot_mask.sum(dim=1)
        for batch_idx in range(batch):
            valid_len = int(valid_mask[batch_idx].sum().item())
            expected_k = execution_k
            if int(effective_k[batch_idx].item()) != expected_k:
                raise ValueError("fixed hard path violates K_eff")
            active = positions[batch_idx, :expected_k]
            if bool(torch.any(active < 0).item()) or bool(
                torch.any(active >= valid_len).item()
            ):
                raise ValueError("fixed hard path is outside valid candidates")
            if expected_k > 1 and not bool(torch.all(active[1:] > active[:-1]).item()):
                raise ValueError("fixed hard path must be unique and ordered")
            occupancy[batch_idx].scatter_(0, active, 1.0)
            slot_assignment[batch_idx, :expected_k].scatter_(
                1,
                active[:, None],
                1.0,
            )
        hard = PhysicalExactKHardOutput(
            hard_occupancy=occupancy,
            hard_slot_assignment=slot_assignment,
            hard_positions=positions,
            hard_slot_mask=slot_mask,
            edge_count=torch.zeros(
                batch,
                device=inputs.device,
                dtype=torch.long,
            ),
            effective_k=effective_k,
            max_gap_seconds=caps,
        )
        selected_inputs = _hard_gather(inputs, positions, slot_mask)
        output_metas = self._write_detector_metadata(
            metas,
            hard,
            physical_seconds,
            source_frames,
            valid_mask,
        )
        return {
            "inputs": selected_inputs,
            "masks": slot_mask,
            "metas": output_metas,
            "positions": positions,
            "hard_forward_only": True,
            "physical_seconds": physical_seconds,
            "decoded_source_frames": source_frames,
            "max_gap_seconds": caps,
        }

    def _validate_inputs(self, inputs, masks, metas) -> None:
        if not torch.is_tensor(inputs) or not (
            inputs.is_floating_point() or inputs.dtype == torch.uint8
        ):
            raise ValueError(
                "protected DUCA inputs must be floating-point or uint8 RGB tensors"
            )
        temporal_dim = _time_dim(inputs)
        if masks.ndim != 2 or tuple(masks.shape) != (
            int(inputs.shape[0]),
            int(inputs.shape[temporal_dim]),
        ):
            raise ValueError("protected DUCA masks must align with [B,T]")
        if int(inputs.shape[temporal_dim]) != self.dense_window_size:
            raise ValueError(
                "protected DUCA input length must match the frozen dense window"
            )
        valid = masks.to(device=inputs.device, dtype=torch.bool)
        for row in valid:
            valid_len = int(row.sum().item())
            expected = torch.arange(valid_len, device=row.device)
            observed = torch.nonzero(row, as_tuple=False).flatten()
            if valid_len <= 0 or not torch.equal(observed, expected):
                raise ValueError(
                    "protected DUCA requires one nonempty contiguous valid prefix"
                )
        if not isinstance(metas, (list, tuple)) or len(metas) != int(inputs.shape[0]):
            raise ValueError("protected DUCA requires one metadata mapping per sample")
        if not all(isinstance(meta, Mapping) for meta in metas):
            raise ValueError("protected DUCA metadata entries must be mappings")

    @staticmethod
    def _reject_train_decision_payload(metas, kwargs) -> None:
        forbidden = {
            "teacher_utility",
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
        }
        _assert_no_forbidden_payload(
            {"metas": metas, "kwargs": kwargs},
            forbidden_keys=forbidden,
            path="protected_duca_train",
        )

    def _physical_axes(
        self,
        metas,
        valid_mask: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal_len = int(valid_mask.shape[1])
        seconds_rows = []
        frame_rows = []
        for batch_idx, meta in enumerate(metas):
            if "frame_inds" not in meta or "avg_fps" not in meta:
                raise ValueError(
                    "protected DUCA requires frame_inds and avg_fps metadata"
                )
            frame_inds = torch.as_tensor(
                meta["frame_inds"],
                device=device,
                dtype=torch.float64,
            )
            if frame_inds.numel() % temporal_len != 0:
                raise ValueError(
                    "frame_inds must be divisible by dense temporal length"
                )
            frame_inds = frame_inds.reshape(temporal_len, -1)
            if frame_inds.shape[1] <= 0:
                raise ValueError("frame_inds must contain at least one decoded frame")
            source_frames = frame_inds[:, frame_inds.shape[1] // 2]
            fps = float(meta["avg_fps"])
            if not math.isfinite(fps) or fps <= 0.0:
                raise ValueError("avg_fps must be finite and positive")
            valid_len = int(valid_mask[batch_idx].sum().item())
            active = source_frames[:valid_len]
            if not bool(torch.isfinite(active).all().item()):
                raise ValueError("decoded source frames must be finite")
            if valid_len > 1 and not bool(torch.all(active[1:] > active[:-1]).item()):
                raise ValueError(
                    "decoded source frames must be strictly increasing on the valid prefix"
                )
            padded_frames = torch.zeros(
                temporal_len,
                device=device,
                dtype=torch.float64,
            )
            padded_frames[:valid_len] = active
            frame_rows.append(padded_frames)
            seconds_rows.append(padded_frames / fps)
        return torch.stack(seconds_rows), torch.stack(frame_rows)

    def _write_detector_metadata(
        self,
        metas,
        hard: PhysicalExactKHardOutput,
        physical_seconds: torch.Tensor,
        source_frames: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> list[dict[str, Any]]:
        if self.detector_coordinate_mode == "selected_axis_plugin":
            return self._write_selected_axis_metadata(
                metas,
                hard,
                physical_seconds,
                source_frames,
                valid_mask,
            )
        return self._write_physical_metadata(
            metas,
            hard,
            physical_seconds,
            source_frames,
            valid_mask,
        )

    def _write_selected_axis_metadata(
        self,
        metas,
        hard: PhysicalExactKHardOutput,
        physical_seconds: torch.Tensor,
        source_frames: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> list[dict[str, Any]]:
        output = self._write_physical_metadata(
            metas,
            hard,
            physical_seconds,
            source_frames,
            valid_mask,
        )
        for batch_idx, meta in enumerate(output):
            effective_k = int(hard.effective_k[batch_idx].item())
            dense_valid_len = int(valid_mask[batch_idx].sum().item())
            positions = hard.hard_positions[batch_idx, :effective_k]
            dense_positions = [
                int(value) for value in positions.detach().cpu().tolist()
            ]
            meta.update(
                {
                    "duca_contract": self.detector_coordinate_contract,
                    "detector_coordinate_mode": self.detector_coordinate_mode,
                    "selected_axis_to_true_time_dense_index": dense_positions,
                    "truetime_selected_positions": dense_positions,
                    "truetime_dense_len": int(self.dense_window_size),
                    "truetime_dense_valid_len": dense_valid_len,
                    "irregular_selected_positions": dense_positions,
                    "irregular_selected_count": effective_k,
                    "irregular_selected_valid_len": effective_k,
                    "selected_valid_len": effective_k,
                    "irregular_dense_valid_len": dense_valid_len,
                    "remap_gt_to_selected_axis": True,
                    "gt_remapped_to_selected_axis": False,
                    "pc_ot_mras_prebackbone_remap_gt_to_selected_axis": True,
                    "selected_axis_remap_required": True,
                    "detector_prediction_inverse_map_required": True,
                    "detector_output_coordinate_space": SELECTED_AXIS,
                    "proposal_axis": SELECTED_AXIS,
                    "selected_axis_remap": {
                        "source": SELECTED_AXIS,
                        "target": TRUE_TIME_AXIS,
                        "selected_axis_to_true_time_dense_index": dense_positions,
                    },
                }
            )
            meta.pop("physical_grid_contract", None)
            meta.pop("physical_grid_actionformer", None)
        return output

    def _remap_train_targets_to_selected_axis(
        self,
        gt_segments,
        gt_labels,
        metas,
    ):
        if gt_segments is None or not self.remap_gt_to_selected_axis:
            return gt_segments, gt_labels, metas
        if not isinstance(metas, (list, tuple)) or len(gt_segments) != len(metas):
            raise ValueError("GT remap requires one selected-axis metadata row per sample")
        remapped_segments = []
        updated_metas = [dict(meta) for meta in metas]
        for batch_idx, (segments, meta) in enumerate(
            zip(gt_segments, updated_metas)
        ):
            if segments is None:
                remapped_segments.append(None)
                continue
            segment_tensor = (
                segments
                if torch.is_tensor(segments)
                else torch.as_tensor(segments, dtype=torch.float32)
            )
            time_map = truetime_map_from_metadata(
                meta,
                require_inverse_map=True,
            )
            remapped = time_map.remap_segments(
                segment_tensor,
                source_coordinate_space=TRUE_TIME_AXIS,
                target_coordinate_space=SELECTED_AXIS,
            ).to(device=segment_tensor.device, dtype=segment_tensor.dtype)
            remapped_segments.append(remapped)
            meta["gt_segments_original_time"] = (
                segment_tensor.detach().cpu().tolist()
            )
            meta["gt_segments_selected_axis"] = remapped.detach().cpu().tolist()
            meta["gt_remapped_to_selected_axis"] = True
            meta["gt_coordinate_space"] = SELECTED_AXIS
            meta["gt_original_coordinate_space"] = TRUE_TIME_AXIS
            updated_metas[batch_idx] = meta
        return remapped_segments, gt_labels, updated_metas

    def _write_physical_metadata(
        self,
        metas,
        hard: PhysicalExactKHardOutput,
        physical_seconds: torch.Tensor,
        source_frames: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> list[dict[str, Any]]:
        output = []
        for batch_idx, source_meta in enumerate(metas):
            meta = dict(source_meta)
            effective_k = int(hard.effective_k[batch_idx].item())
            dense_valid_len = int(valid_mask[batch_idx].sum().item())
            positions = hard.hard_positions[batch_idx, :effective_k]
            if positions.numel() != effective_k:
                raise RuntimeError("protected DUCA selected-count mismatch")
            if effective_k > 1 and not bool(
                torch.all(positions[1:] > positions[:-1]).item()
            ):
                raise RuntimeError("protected DUCA selected positions are unordered")
            selected_seconds = physical_seconds[batch_idx, positions]
            selected_frames = source_frames[batch_idx, positions]
            intervals = [
                selected_seconds[0] - physical_seconds[batch_idx, 0],
                physical_seconds[batch_idx, dense_valid_len - 1] - selected_seconds[-1],
            ]
            if effective_k > 1:
                intervals.append(selected_seconds[1:] - selected_seconds[:-1])
            observed_max_gap = torch.cat(
                [value.reshape(-1) for value in intervals]
            ).max()
            cap = hard.max_gap_seconds[batch_idx].to(
                device=observed_max_gap.device,
                dtype=observed_max_gap.dtype,
            )
            tolerance = max(
                1.0e-9,
                8.0 * torch.finfo(observed_max_gap.dtype).eps,
            )
            if float(observed_max_gap.item()) > float(cap.item()) + tolerance:
                raise RuntimeError(
                    "protected DUCA hard path violates physical max-gap: "
                    f"observed={float(observed_max_gap.item()):.17g}, "
                    f"cap={float(cap.item()):.17g}, tolerance={tolerance:.3g}, "
                    f"effective_k={effective_k}, dense_valid_len={dense_valid_len}"
                )

            dense_positions = [int(value) for value in positions.cpu().tolist()]
            meta.update(
                {
                    "duca_contract": _PROTECTED_CONTRACT,
                    "detector_coordinate_mode": "physical_head_integration",
                    "duca_arm": self.arm,
                    "irregular_selected_positions": dense_positions,
                    "selected_dense_indices": dense_positions,
                    "selected_valid_len": effective_k,
                    "irregular_selected_count": effective_k,
                    "irregular_dense_valid_len": dense_valid_len,
                    "irregular_native_axis": True,
                    "remap_gt_to_selected_axis": False,
                    "gt_remapped_to_selected_axis": False,
                    "pc_ot_mras_prebackbone_remap_gt_to_selected_axis": False,
                    "selected_axis_remap_required": False,
                    "detector_prediction_inverse_map_required": False,
                    "detector_output_coordinate_space": "dense_physical",
                    "proposal_axis": "dense_physical",
                    "physical_grid_contract": _PROTECTED_CONTRACT,
                    "duca_selected_source_frames": [
                        float(value) for value in selected_frames.cpu().tolist()
                    ],
                    "duca_selected_seconds": [
                        float(value) for value in selected_seconds.cpu().tolist()
                    ],
                    "duca_max_gap_seconds_cap": float(cap.item()),
                    "duca_observed_max_gap_seconds": float(observed_max_gap.item()),
                    "duca_candidate_coordinate_unit": "dense_candidate_ordinal",
                    "duca_source_frame_unit": "decoded_frame_index",
                    "duca_physical_time_unit": "seconds",
                    "duca_requested_k": self.budget,
                    "duca_effective_k": effective_k,
                    "duca_unique_k": effective_k,
                    "duca_backbone_input_k": effective_k,
                    "duca_padded_k": effective_k,
                    "duca_dynamic_compute_realized": True,
                    "duca_execution_quantum": self.execution_quantum,
                    "duca_backbone_tail_padding_mode": "none_exact_k_bucket",
                }
            )
            output.append(meta)
        return output


__all__ = [
    "DucaProtectedE2EFrameSelector",
    "_homogeneous_quantized_effective_k",
]
