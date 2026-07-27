from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..builder import SELECTORS
from ..duca.rime import (
    RIME_CONTRACT,
    RimeBudgetController,
    RimeBudgetDecision,
    decode_rime_exact_k,
    rime_budget_supervision_losses,
    rime_rank_alignment_loss,
)
from ..duca.structured_selection import PhysicalExactKHardOutput
from ..duca.transition_only import (
    ASFORMER_ENCODER_HIDDEN_KIND,
    balanced_binary_actionness_loss,
    coverage_floor_distribution,
    local_boundary_mass_coverage_loss,
    transition_distribution_loss,
    transition_utility_paths,
)
from .duca_protected_e2e_frame_selector import (
    DucaProtectedE2EFrameSelector,
    _action_target_from_gt_segments,
    _hard_gather,
    _soft_resample,
    _transition_target_from_gt_segments,
)


_RIME_ARMS = {
    "fixed_bound",
    "dynamic_no_risk",
    "dynamic_shuffle",
    "rime_full",
    "adaptok_tad",
    "uniform_same_k",
}
_DYNAMIC_ARMS = {"dynamic_no_risk", "rime_full"}
_PROTOCOL_SCHEMA = "duca_rime_budget_protocol_v1"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_protocol(
    path: str,
    expected_sha256: str,
    *,
    candidate_budgets: tuple[int, ...],
) -> dict[str, Any]:
    resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"RIME frozen budget protocol missing: {resolved}")
    actual_hash = _sha256_file(resolved)
    if not expected_sha256 or actual_hash != str(expected_sha256).lower():
        raise ValueError("RIME budget protocol SHA-256 is required and must match")
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != _PROTOCOL_SCHEMA:
        raise ValueError(f"RIME protocol must use schema {_PROTOCOL_SCHEMA}")
    if payload.get("fit_split") not in {"train", "training", "train_only"}:
        raise ValueError("RIME protocol must be frozen on a train-only split")
    if payload.get("uses_validation_or_test_labels") is not False:
        raise ValueError("RIME protocol must explicitly forbid validation/test labels")
    if tuple(int(value) for value in payload.get("candidate_budgets", ())) != candidate_budgets:
        raise ValueError("RIME protocol candidate budgets disagree with the model")
    protocol_costs = tuple(float(value) for value in payload.get("candidate_costs", ()))
    if (
        len(protocol_costs) != len(candidate_budgets)
        or any(not math.isfinite(value) or value <= 0.0 for value in protocol_costs)
        or any(
            right <= left
            for left, right in zip(protocol_costs[:-1], protocol_costs[1:])
        )
    ):
        raise ValueError("RIME protocol costs must be finite, positive, and increasing")
    protocol_risk_weight = float(payload.get("risk_weight", float("nan")))
    if not math.isfinite(protocol_risk_weight) or protocol_risk_weight < 0.0:
        raise ValueError("RIME protocol risk_weight must be finite and non-negative")
    payload = dict(payload)
    payload["path"] = resolved
    payload["sha256"] = actual_hash
    return payload


def _targets_from_batch(
    explicit: Any,
    metas,
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    value = explicit
    if value is None and isinstance(metas, (list, tuple)) and all(
        isinstance(meta, Mapping) and key in meta for meta in metas
    ):
        value = [meta[key] for meta in metas]
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value and all(
        torch.is_tensor(item) for item in value
    ):
        value = torch.stack(
            [item.to(device=device, dtype=dtype) for item in value],
            dim=0,
        )
    return torch.as_tensor(value, device=device, dtype=dtype)


@SELECTORS.register_module()
class DucaRimeFrameSelector(DucaProtectedE2EFrameSelector):
    """Risk-aware, exact-K, physical-time dynamic-budget DUCA selector.

    The cheap ASFormer evidence path runs on the dense window.  The heavy RGB
    backbone receives exactly the homogeneous effective K selected for the
    current bucket; mixed-K batches fail before heavy execution.
    """

    def __init__(
        self,
        in_channels: int,
        rime_arm: str,
        candidate_budgets: Sequence[int] = (192, 256, 384, 512),
        candidate_costs: Sequence[float] | None = None,
        fixed_budget: int | None = None,
        dense_window_size: int = 768,
        frozen_price: float = 0.0,
        target_mean_cost: float | None = None,
        risk_weight: float = 1.0,
        risk_threshold: float = 0.35,
        uncertainty_z: float = 1.645,
        decoder_family: str = "independent",
        weak_overlap_fraction: float = 0.50,
        controller_hidden_dim: int = 128,
        execution_quantum: int = 16,
        budget_protocol_path: str | None = None,
        budget_protocol_sha256: str | None = None,
        require_frozen_protocol: bool = True,
        budget_utility_loss_weight: float = 1.0,
        budget_risk_loss_weight: float = 1.0,
        budget_uncertainty_loss_weight: float = 0.25,
        rank_alignment_loss_weight: float = 0.25,
        **kwargs: Any,
    ) -> None:
        arm = str(rime_arm)
        if arm not in _RIME_ARMS:
            raise ValueError(f"rime_arm must be one of {sorted(_RIME_ARMS)}")
        budgets = tuple(int(value) for value in candidate_budgets)
        if len(budgets) < 2 or tuple(sorted(set(budgets))) != budgets:
            raise ValueError("RIME candidate budgets must be unique and increasing")
        if budgets[0] <= 0 or budgets[-1] > int(dense_window_size):
            raise ValueError("RIME budgets must lie inside the dense window")
        if any(value % 16 != 0 for value in budgets):
            raise ValueError("RIME heavy-frame budgets must be divisible by clip_len=16")
        if int(execution_quantum) != 16:
            raise ValueError("RIME VideoMAE execution_quantum is frozen to 16 frames")
        fixed = budgets[-1] if fixed_budget is None else int(fixed_budget)
        if fixed not in budgets:
            raise ValueError("fixed_budget must belong to candidate_budgets")
        protocol = None
        if budget_protocol_path:
            protocol = _load_frozen_protocol(
                budget_protocol_path,
                str(budget_protocol_sha256 or ""),
                candidate_budgets=budgets,
            )
            frozen_price = float(protocol["frozen_price"])
            target_mean_cost = float(protocol["target_mean_cost"])
            risk_weight = float(protocol["risk_weight"])
            risk_threshold = float(protocol["risk_threshold"])
            decoder_family = str(protocol["decoder_family"])
            weak_overlap_fraction = float(protocol["weak_overlap_fraction"])
            protocol_costs = tuple(float(value) for value in protocol["candidate_costs"])
            if candidate_costs is not None and tuple(float(v) for v in candidate_costs) != protocol_costs:
                raise ValueError("RIME configured costs disagree with the frozen protocol")
            candidate_costs = protocol_costs
        elif require_frozen_protocol and arm in _DYNAMIC_ARMS:
            raise ValueError("dynamic RIME arms require a hashed train-only budget protocol")

        super().__init__(
            in_channels=in_channels,
            arm="protected_e2e",
            budget=budgets[-1],
            dense_window_size=dense_window_size,
            **kwargs,
        )
        self.rime_arm = arm
        self.arm = arm
        self.candidate_budgets = budgets
        self.candidate_costs = (
            tuple(float(value) for value in candidate_costs)
            if candidate_costs is not None
            else tuple(float(value) for value in budgets)
        )
        if len(self.candidate_costs) != len(budgets):
            raise ValueError("candidate_costs must align with candidate_budgets")
        self.fixed_budget = fixed
        self.execution_quantum = int(execution_quantum)
        self.decoder_family = str(decoder_family)
        self.weak_overlap_fraction = float(weak_overlap_fraction)
        self.budget_protocol = protocol
        self.budget_protocol_path = None if protocol is None else protocol["path"]
        self.budget_protocol_sha256 = None if protocol is None else protocol["sha256"]
        self.budget_utility_loss_weight = float(budget_utility_loss_weight)
        self.budget_risk_loss_weight = float(budget_risk_loss_weight)
        self.budget_uncertainty_loss_weight = float(budget_uncertainty_loss_weight)
        self.rank_alignment_loss_weight = float(rank_alignment_loss_weight)
        if min(
            self.budget_utility_loss_weight,
            self.budget_risk_loss_weight,
            self.budget_uncertainty_loss_weight,
            self.rank_alignment_loss_weight,
        ) < 0.0:
            raise ValueError("RIME loss weights must be non-negative")

        self.selector_variant = "duca_rime_physical"
        self.no_ledger_decision = False
        self.require_counterfactual_utility_teacher = False
        self.budget_controller = (
            RimeBudgetController(
                evidence_dim=self.coarse_hidden_dim,
                candidate_budgets=budgets,
                candidate_costs=self.candidate_costs,
                hidden_dim=int(controller_hidden_dim),
                frozen_price=float(frozen_price),
                target_mean_cost=target_mean_cost,
                risk_weight=float(risk_weight),
                risk_threshold=float(risk_threshold),
                uncertainty_z=float(uncertainty_z),
                use_risk=arm == "rime_full",
            )
            if arm in _DYNAMIC_ARMS
            else None
        )

    def _fixed_requested_k(
        self,
        policy_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        metas,
    ) -> torch.Tensor:
        batch = int(policy_scores.shape[0])
        if self.rime_arm == "fixed_bound":
            return torch.full(
                (batch,),
                self.fixed_budget,
                device=policy_scores.device,
                dtype=torch.long,
            )
        replay_roles = {
            "uniform_same_k": "paired_same_realized_cost_control",
            "dynamic_shuffle": "histogram_shuffled_budget_control",
            "adaptok_tad": "adaptok_total_loss_curve_test_batch_ilp",
        }
        if self.rime_arm in replay_roles:
            values = []
            for meta in metas:
                value = int(meta.get("rime_requested_k_replay", -1))
                provenance = meta.get("rime_requested_k_replay_provenance")
                if value not in self.candidate_budgets or not isinstance(provenance, Mapping):
                    raise ValueError(
                        f"{self.rime_arm} requires a candidate K and replay provenance"
                    )
                if provenance.get("role") != replay_roles[self.rime_arm]:
                    raise ValueError(f"{self.rime_arm} replay has the wrong role")
                if any(
                    bool(provenance.get(key, False))
                    for key in ("uses_gt", "uses_teacher", "uses_prediction_cache")
                ):
                    raise ValueError(f"{self.rime_arm} replay provenance is contaminated")
                values.append(value)
            return torch.tensor(values, device=policy_scores.device, dtype=torch.long)
        raise RuntimeError("fixed RIME decision requested for a learned-controller arm")

    def _select(
        self,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        metas,
        *,
        training: bool,
    ) -> dict[str, Any]:
        physical_seconds, source_frames = self._physical_axes(
            metas,
            valid_mask,
            inputs.device,
        )
        source = self.raw_actionness_source(inputs, valid_mask=valid_mask)
        hidden = source.get("coarse_hidden_features")
        if (
            hidden is None
            or source.get("hidden_kind") != ASFORMER_ENCODER_HIDDEN_KIND
        ):
            raise RuntimeError("RIME requires official ASFormer encoder hidden features")
        paths = transition_utility_paths(
            self.transition_scorer,
            source["actionness_logits"],
            hidden,
            valid_mask,
            compute_auxiliary=training,
            policy_hidden=None,
            policy_hidden_gradient_scale=0.0,
        )
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
        if not torch.equal(auxiliary_prob.detach(), policy_prob.detach()):
            raise RuntimeError("RIME auxiliary and policy coverage paths disagree")

        decision: RimeBudgetDecision | None = None
        if self.budget_controller is not None:
            decision = self.budget_controller(hidden, paths["policy_scores"], valid_mask)
            requested_k = decision.requested_k
            risk_fallback = decision.fallback_to_kmax
        else:
            requested_k = self._fixed_requested_k(paths["policy_scores"], valid_mask, metas)
            risk_fallback = torch.zeros_like(requested_k, dtype=torch.bool)
        decoded = decode_rime_exact_k(
            policy_log_prob,
            physical_seconds,
            valid_mask,
            requested_k,
            candidate_budgets=self.candidate_budgets,
            decoder_family=self.decoder_family,
            weak_overlap_fraction=self.weak_overlap_fraction,
            training=training,
            force_uniform=self.rime_arm == "uniform_same_k",
            risk_fallback=risk_fallback,
            require_homogeneous_execution=True,
            execution_quantum=self.execution_quantum,
        )
        auxiliary_decoded = (
            decode_rime_exact_k(
                auxiliary_log_prob,
                physical_seconds,
                valid_mask,
                requested_k,
                candidate_budgets=self.candidate_budgets,
                decoder_family=self.decoder_family,
                weak_overlap_fraction=self.weak_overlap_fraction,
                training=True,
                force_uniform=self.rime_arm == "uniform_same_k",
                risk_fallback=risk_fallback,
                require_homogeneous_execution=True,
                execution_quantum=self.execution_quantum,
            )
            if training
            else None
        )
        decoded.ledger.validate(require_no_padding=True)
        hard_selected = _hard_gather(
            inputs,
            decoded.hard_positions,
            decoded.hard_slot_mask,
        )
        detector_bridge = bool(training and self.detector_bridge_gradient_scale > 0.0)
        hard_detector_input = hard_selected
        if detector_bridge:
            soft_selected = _soft_resample(inputs, decoded.soft_slot_assignment)
            hard_base = hard_selected if hard_selected.is_floating_point() else hard_selected.float()
            selected_inputs = hard_base + self.detector_bridge_gradient_scale * (
                soft_selected - soft_selected.detach()
            )
            if not torch.equal(selected_inputs.detach(), hard_base.detach()):
                raise RuntimeError("RIME detector input is not exact hard forward")
        else:
            selected_inputs = hard_selected

        hard = PhysicalExactKHardOutput(
            hard_occupancy=decoded.hard_occupancy,
            hard_slot_assignment=decoded.hard_slot_assignment,
            hard_positions=decoded.hard_positions,
            hard_slot_mask=decoded.hard_slot_mask,
            edge_count=torch.zeros_like(decoded.effective_k),
            effective_k=decoded.effective_k,
            max_gap_seconds=decoded.max_gap_seconds,
        )
        output_metas = self._write_physical_metadata(
            metas,
            hard,
            physical_seconds,
            source_frames,
            valid_mask,
        )
        ledger = decoded.ledger.to_dict()
        for batch_index, meta in enumerate(output_metas):
            meta.update(
                {
                    "duca_contract": RIME_CONTRACT,
                    "physical_grid_contract": RIME_CONTRACT,
                    "duca_arm": self.rime_arm,
                    "duca_requested_k": ledger["requested_k"][batch_index],
                    "duca_effective_k": ledger["effective_k"][batch_index],
                    "duca_unique_k": ledger["unique_k"][batch_index],
                    "duca_backbone_input_k": ledger["backbone_input_k"][batch_index],
                    "duca_padded_k": ledger["padded_k"][batch_index],
                    "duca_risk_fallback": ledger["risk_fallback"][batch_index],
                    "duca_dynamic_compute_realized": True,
                    "duca_cost_unit": ledger["unit"],
                    "duca_backbone_tail_padding_mode": "none_exact_k_bucket",
                    "duca_execution_quantum": self.execution_quantum,
                    "duca_budget_protocol_sha256": self.budget_protocol_sha256,
                }
            )

        state: dict[str, Any] = {
            "arm": self.rime_arm,
            "contract": RIME_CONTRACT,
            "valid_mask": valid_mask,
            "physical_seconds": physical_seconds,
            "decoded_source_frames": source_frames,
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
            "hard_occupancy": decoded.hard_occupancy,
            "hard_slot_assignment": decoded.hard_slot_assignment,
            "selected_positions": decoded.hard_positions,
            "selected_mask": decoded.hard_slot_mask,
            "selected_count": decoded.effective_k,
            "requested_k": decoded.requested_k,
            "max_gap_seconds": decoded.max_gap_seconds,
            "decoder_family": decoded.decoder_family,
            "weak_overlap_fraction": decoded.overlap_fraction,
            "constant_uniform_identity": decoded.constant_uniform_identity,
            "cost_ledger": ledger,
            "detector_gradient_bridge": detector_bridge,
            "detector_bridge_gradient_scale": self.detector_bridge_gradient_scale,
            "detector_input": selected_inputs,
            "hard_detector_input": hard_detector_input,
            "backbone_tail_padding_mode": "none_exact_k_bucket",
            "train_inference_hard_decoder": "same_rime_physical_exact_k",
            "coarse_provenance": source["provenance"],
            "coarse_compute_profile": source.get("compute_profile"),
        }
        if training:
            state.update(
                {
                    "soft_occupancy": decoded.soft_occupancy,
                    "soft_slot_assignment": decoded.soft_slot_assignment,
                    "selection_st": decoded.selection_st,
                    "auxiliary_soft_occupancy": (
                        auxiliary_decoded.soft_occupancy
                    ),
                    "auxiliary_soft_slot_assignment": (
                        auxiliary_decoded.soft_slot_assignment
                    ),
                }
            )
        if decision is not None:
            state["budget_decision"] = decision
        self.last_forward_summary = {
            "arm": self.rime_arm,
            "training": bool(training),
            "requested_k": list(ledger["requested_k"]),
            "effective_k": list(ledger["effective_k"]),
            "backbone_input_k": list(ledger["backbone_input_k"]),
            "padded_k": list(ledger["padded_k"]),
            "risk_fallback": list(ledger["risk_fallback"]),
            "dynamic_compute_realized": True,
            "contract": RIME_CONTRACT,
        }
        self._last_selected_positions = decoded.hard_positions.detach().clone()
        self._last_physical_metas = copy.deepcopy(output_metas)
        return {
            "inputs": selected_inputs,
            "masks": decoded.hard_slot_mask,
            "metas": output_metas,
            "selector_outputs": state,
        }

    def forward_train(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas,
        gt_segments=None,
        gt_labels=None,
        gt_boundary_validity=None,
        rime_utility_target=None,
        rime_risk_target=None,
        rime_target_mask=None,
        rime_hard_frame_utility=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.rime_arm == "uniform_same_k":
            raise RuntimeError("uniform_same_k is a paired evaluation control, not a train arm")
        self._validate_inputs(inputs, masks, metas)
        self._reject_train_decision_payload(metas, kwargs)
        valid = masks.to(device=inputs.device, dtype=torch.bool)
        action_target = _action_target_from_gt_segments(gt_segments, valid)
        transition_target = _transition_target_from_gt_segments(
            gt_segments,
            valid,
            sigma=self.transition_target_sigma,
            radius=self.transition_target_radius,
            boundary_validity=gt_boundary_validity,
        )
        if action_target is None or transition_target is None or gt_boundary_validity is None:
            raise ValueError("RIME training requires train-only segments and boundary validity")
        selected = self._select(inputs, valid, metas, training=True)
        state = selected["selector_outputs"]
        action_loss, positive_weight = balanced_binary_actionness_loss(
            state["actionness_logits"],
            action_target,
            valid,
        )
        transition_loss = transition_distribution_loss(
            state["auxiliary_scores"],
            transition_target,
            valid,
            temperature=self.transition_distribution_temperature,
        )
        boundary_loss = local_boundary_mass_coverage_loss(
            state["auxiliary_soft_occupancy"],
            transition_target,
            valid,
            radius=self.transition_boundary_radius,
        )
        losses = {
            "selector_action_loss": action_loss * self.action_loss_weight,
            "selector_transition_loss": transition_loss * self.transition_loss_weight,
            "selector_transition_boundary_loss": (
                boundary_loss * self.transition_boundary_loss_weight
            ),
        }

        decision = state.get("budget_decision")
        if decision is not None:
            utility_target = _targets_from_batch(
                rime_utility_target,
                metas,
                "rime_utility_target",
                device=inputs.device,
                dtype=torch.float32,
            )
            risk_target = _targets_from_batch(
                rime_risk_target,
                metas,
                "rime_risk_target",
                device=inputs.device,
                dtype=torch.float32,
            )
            target_mask = _targets_from_batch(
                rime_target_mask,
                metas,
                "rime_target_mask",
                device=inputs.device,
                dtype=torch.bool,
            )
            if utility_target is None or risk_target is None:
                raise ValueError(
                    "dynamic RIME training requires train-only per-K utility and risk labels"
                )
            budget_losses = rime_budget_supervision_losses(
                decision,
                utility_target,
                risk_target,
                target_mask,
            )
            losses.update(
                {
                    "selector_rime_utility_loss": (
                        budget_losses["selector_rime_utility_loss"]
                        * self.budget_utility_loss_weight
                    ),
                    "selector_rime_risk_loss": (
                        budget_losses["selector_rime_risk_loss"]
                        * self.budget_risk_loss_weight
                    ),
                    "selector_rime_uncertainty_loss": (
                        budget_losses["selector_rime_uncertainty_loss"]
                        * self.budget_uncertainty_loss_weight
                    ),
                }
            )
            state["rime_utility_target"] = utility_target
            state["rime_risk_target"] = risk_target

        hard_frame_utility = _targets_from_batch(
            rime_hard_frame_utility,
            metas,
            "rime_hard_frame_utility",
            device=inputs.device,
            dtype=torch.float32,
        )
        if hard_frame_utility is not None and self.rank_alignment_loss_weight > 0.0:
            losses["selector_rime_rank_loss"] = (
                rime_rank_alignment_loss(
                    state["policy_scores"],
                    hard_frame_utility,
                    valid,
                )
                * self.rank_alignment_loss_weight
            )
            state["rime_hard_frame_utility"] = hard_frame_utility
        state["action_target"] = action_target
        state["transition_target"] = transition_target
        state["action_positive_weight"] = positive_weight.detach()
        state["training_provenance"] = {
            "task": "offline_tad",
            "gt_scope": "train_only_auxiliary_targets",
            "budget_target_scope": "train_only_cross_fitted_detector_utility",
            "inference_uses_gt": False,
            "inference_uses_teacher": False,
            "inference_uses_prediction_cache": False,
            "selected_axis_gt_remap": False,
        }
        return {
            "inputs": selected["inputs"],
            "masks": selected["masks"],
            "metas": selected["metas"],
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "losses": losses,
            "selector_outputs": state,
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
        from ..duca.acquisition import _assert_no_forbidden_payload

        _assert_no_forbidden_payload(
            {"metas": metas, "kwargs": kwargs},
            path="duca_rime_inference",
        )
        selected = self._select(
            inputs,
            masks.to(device=inputs.device, dtype=torch.bool),
            metas,
            training=False,
        )
        return {
            "inputs": selected["inputs"],
            "masks": selected["masks"],
            "metas": selected["metas"],
            "selector_outputs": selected["selector_outputs"],
        }


__all__ = ["DucaRimeFrameSelector"]
