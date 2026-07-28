from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from ..builder import SELECTORS
from ..duca.rime import (
    RIME_CONTRACT,
    RimeBudgetController,
    RimeBudgetDecision,
    build_cost_matched_mixed_k_cycle,
    decode_rime_exact_k,
    rime_budget_supervision_losses,
    rime_rank_alignment_loss,
)
from ..duca.structured_selection import (
    PhysicalExactKHardOutput,
    physical_exact_uniform_gap_cap,
)
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
    "uniform_mixed_k",
}
_DYNAMIC_ARMS = {"dynamic_no_risk", "rime_full"}
_PROTOCOL_SCHEMA = "duca_rime_budget_protocol_v1"
_INFERENCE_LEDGER_SCHEMA = "duca_rime_inference_ledger_v1"


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
    allocation_mode = str(payload.get("allocation_mode", ""))
    if not math.isfinite(protocol_risk_weight) or protocol_risk_weight < 0.0:
        raise ValueError("RIME protocol risk_weight must be finite and non-negative")
    if allocation_mode not in {
        "frozen_price_dynamic_budget",
        "fixed_floor_budget_position_only",
    }:
        raise ValueError("RIME protocol has an unsupported allocation_mode")
    target = float(payload.get("target_mean_cost", float("nan")))
    if allocation_mode == "fixed_floor_budget_position_only":
        if (
            target != protocol_costs[0]
            or int(payload.get("forced_budget", -1)) != candidate_budgets[0]
            or payload.get("risk_used_for_allocation") is not False
            or payload.get("dynamic_budget_claim_allowed") is not False
        ):
            raise ValueError("RIME floor-budget protocol is internally inconsistent")
    elif (
        payload.get("forced_budget") is not None
        or payload.get("risk_used_for_allocation") is not True
        or payload.get("dynamic_budget_claim_allowed") is not True
    ):
        raise ValueError("RIME dynamic budget protocol is internally inconsistent")
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


def _append_jsonl_atomic(path: str, row: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short write while appending the RIME inference ledger")
    finally:
        os.close(descriptor)


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
        mixed_k_schedule_counts: Sequence[int] | None = None,
        mixed_k_schedule_seed: int = 3407,
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

        uniform_mixed_k = arm == "uniform_mixed_k"
        super().__init__(
            in_channels=in_channels,
            arm="exact_uniform" if uniform_mixed_k else "protected_e2e",
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
        self.allocation_mode = (
            "unfrozen_control"
            if protocol is None
            else str(protocol["allocation_mode"])
        )
        self.mixed_k_schedule_seed = int(mixed_k_schedule_seed)
        schedule_counts = (
            None
            if mixed_k_schedule_counts is None
            else tuple(int(value) for value in mixed_k_schedule_counts)
        )
        if uniform_mixed_k:
            if schedule_counts is None or target_mean_cost is None:
                raise ValueError(
                    "uniform_mixed_k requires schedule counts and a target mean cost"
                )
            mixed_k_cycle = build_cost_matched_mixed_k_cycle(
                budgets,
                schedule_counts,
                candidate_costs=self.candidate_costs,
                target_mean_cost=float(target_mean_cost),
                schedule_seed=self.mixed_k_schedule_seed,
            )
            self.register_buffer(
                "mixed_k_schedule",
                torch.tensor(mixed_k_cycle, dtype=torch.long),
                persistent=True,
            )
            self.mixed_k_schedule_counts = schedule_counts
            self.mixed_k_schedule_sha256 = hashlib.sha256(
                json.dumps(
                    {
                        "candidate_budgets": budgets,
                        "candidate_costs": self.candidate_costs,
                        "schedule_counts": schedule_counts,
                        "schedule_seed": self.mixed_k_schedule_seed,
                        "cycle": mixed_k_cycle,
                        "target_mean_cost": float(target_mean_cost),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        else:
            if schedule_counts is not None:
                raise ValueError(
                    "mixed_k_schedule_counts are reserved for uniform_mixed_k"
                )
            self.mixed_k_schedule_counts = None
            self.mixed_k_schedule_sha256 = None
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
        self.register_buffer(
            "_loss_weight_schedule_step",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
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

    def after_optimizer_step(self) -> dict[str, Any]:
        if not self.training:
            return {"updated": False, "reason": "selector_not_training"}
        before = int(self._loss_weight_schedule_step.detach().item())
        self._loss_weight_schedule_step.add_(1)
        summary = {
            "updated": True,
            "source": "successful_optimizer_step",
            "step_before": before,
            "step_after": int(self._loss_weight_schedule_step.detach().item()),
        }
        if isinstance(self.last_forward_summary, dict):
            self.last_forward_summary["loss_weight_schedule"] = summary
        return summary

    def _apply_protocol_allocation_mode(
        self,
        decision: RimeBudgetDecision,
    ) -> RimeBudgetDecision:
        if self.allocation_mode != "fixed_floor_budget_position_only":
            return decision
        selected_index = torch.zeros_like(decision.selected_index)
        requested_floor = self.budget_controller.candidate_budgets.to(
            selected_index.device
        )[selected_index]
        return replace(
            decision,
            selected_index=selected_index,
            requested_k=requested_floor,
            fallback_to_kmax=torch.zeros_like(
                decision.fallback_to_kmax,
                dtype=torch.bool,
            ),
            policy_name="rime_fixed_floor_budget_position_only",
        ).validate(batch_size=decision.selected_index.shape[0])

    def _fixed_requested_k(
        self,
        policy_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        metas,
        *,
        training: bool,
    ) -> torch.Tensor:
        batch = int(policy_scores.shape[0])
        if self.rime_arm == "fixed_bound":
            return torch.full(
                (batch,),
                self.fixed_budget,
                device=policy_scores.device,
                dtype=torch.long,
            )
        if self.rime_arm == "uniform_mixed_k":
            if not training:
                self._last_mixed_k_schedule_indices = tuple(
                    -1 for _ in range(batch)
                )
                self._last_mixed_k_schedule_source = "fixed_evaluation_budget"
                requested = torch.full(
                    (batch,),
                    self.fixed_budget,
                    device=policy_scores.device,
                    dtype=torch.long,
                )
            else:
                cycle_len = int(self.mixed_k_schedule.numel())
                cycle_indices = []
                for meta in metas:
                    if not isinstance(meta, Mapping):
                        raise ValueError(
                            "uniform_mixed_k requires per-sample metadata"
                        )
                    epoch = meta.get("duca_stateless_epoch")
                    sample_index = meta.get("duca_stateless_sample_index")
                    if epoch is None or sample_index is None:
                        raise ValueError(
                            "uniform_mixed_k requires stateless epoch and sample index"
                        )
                    epoch = int(epoch)
                    sample_index = int(sample_index)
                    if epoch < 0 or sample_index < 0:
                        raise ValueError(
                            "uniform_mixed_k stateless identities must be non-negative"
                        )
                    cycle_indices.append((epoch + sample_index) % cycle_len)
                index_tensor = torch.tensor(
                    cycle_indices,
                    device=self.mixed_k_schedule.device,
                    dtype=torch.long,
                )
                requested = self.mixed_k_schedule.index_select(
                    0,
                    index_tensor,
                ).to(device=policy_scores.device)
                self._last_mixed_k_schedule_indices = tuple(cycle_indices)
                self._last_mixed_k_schedule_source = (
                    "stateless_epoch_plus_sample_index"
                )
            valid_counts = valid_mask.long().sum(dim=1)
            if bool(torch.any(valid_counts < requested).item()):
                raise ValueError(
                    "uniform_mixed_k forbids effective-K shrinkage on a short window"
                )
            return requested
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
        if self.rime_arm == "uniform_mixed_k":
            zero_scores = torch.zeros(
                valid_mask.shape,
                device=inputs.device,
                dtype=torch.float32,
            )
            uniform_prob = valid_mask.to(dtype=torch.float32)
            uniform_prob = uniform_prob / uniform_prob.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
            uniform_log_prob = uniform_prob.clamp_min(
                torch.finfo(uniform_prob.dtype).tiny
            ).log()
            hidden = torch.zeros(
                (
                    int(valid_mask.shape[0]),
                    int(valid_mask.shape[1]),
                    self.coarse_hidden_dim,
                ),
                device=inputs.device,
                dtype=torch.float32,
            )
            descriptors = zero_scores.unsqueeze(-1)
            source = {
                "actionness_logits": zero_scores,
                "p_action": zero_scores + 0.5,
                "coarse_hidden_features": hidden,
                "hidden_kind": "constant_probe_free_uniform_mixed_k",
                "provenance": {
                    "source": "constant_exact_uniform",
                    "uses_labels": False,
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_prediction_cache": False,
                    "probe_executed": False,
                },
                "compute_profile": {
                    "coarse_probe_executed": False,
                    "learned_selector_executed": False,
                },
            }
            paths = {
                "transition_descriptors": descriptors,
                "policy_descriptors": descriptors,
                "auxiliary_scores": zero_scores,
                "policy_scores": zero_scores,
            }
            auxiliary_prob = uniform_prob
            auxiliary_log_prob = uniform_log_prob
            policy_prob = uniform_prob
            policy_log_prob = uniform_log_prob
        else:
            source = self.raw_actionness_source(inputs, valid_mask=valid_mask)
            hidden = source.get("coarse_hidden_features")
            if (
                hidden is None
                or source.get("hidden_kind") != ASFORMER_ENCODER_HIDDEN_KIND
            ):
                raise RuntimeError(
                    "RIME requires official ASFormer encoder hidden features"
                )
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
            decision = self._apply_protocol_allocation_mode(decision)
            requested_k = decision.requested_k
            risk_fallback = decision.fallback_to_kmax
        else:
            requested_k = self._fixed_requested_k(
                paths["policy_scores"],
                valid_mask,
                metas,
                training=training,
            )
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
            force_uniform=self.rime_arm in {"uniform_same_k", "uniform_mixed_k"},
            risk_fallback=risk_fallback,
            require_homogeneous_execution=True,
            execution_quantum=self.execution_quantum,
        )
        auxiliary_decoded = (
            decoded
            if training and self.rime_arm == "uniform_mixed_k"
            else (
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
                    "duca_rime_allocation_mode": self.allocation_mode,
                }
            )
            if self.rime_arm == "uniform_mixed_k":
                meta.update(
                    {
                        "duca_mixed_k_schedule_sha256": (
                            self.mixed_k_schedule_sha256
                        ),
                        "duca_mixed_k_schedule_seed": self.mixed_k_schedule_seed,
                        "duca_mixed_k_schedule_counts": list(
                            self.mixed_k_schedule_counts
                        ),
                        "duca_mixed_k_schedule_index": int(
                            self._last_mixed_k_schedule_indices[batch_index]
                        ),
                        "duca_mixed_k_schedule_source": (
                            self._last_mixed_k_schedule_source
                        ),
                        "duca_mixed_k_successful_update_step": int(
                            self._loss_weight_schedule_step.detach().item()
                        ),
                        "duca_detector_training_exposure": (
                            "mixed_k_registered_panel"
                        ),
                    }
                )
        if not training:
            ledger_root = os.environ.get("DUCA_RIME_INFERENCE_LEDGER_ROOT", "").strip()
            if ledger_root:
                ledger_root = os.path.abspath(
                    os.path.expandvars(os.path.expanduser(ledger_root))
                )
                os.makedirs(ledger_root, exist_ok=True)
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                ledger_path = os.path.join(
                    ledger_root,
                    f"inference_ledger.rank{rank:04d}.jsonl",
                )
                for batch_index, meta in enumerate(output_metas):
                    video_id = str(
                        meta.get("video_name") or meta.get("video_id") or ""
                    )
                    window_start = int(meta.get("window_start_frame", 0))
                    if not video_id or window_start < 0:
                        raise ValueError(
                            "RIME inference ledger requires video and non-negative window start"
                        )
                    _append_jsonl_atomic(
                        ledger_path,
                        {
                            "schema_version": _INFERENCE_LEDGER_SCHEMA,
                            "video_id": video_id,
                            "window_start_frame": window_start,
                            "arm": self.rime_arm,
                            "candidate_budgets": list(self.candidate_budgets),
                            "requested_k": int(ledger["requested_k"][batch_index]),
                            "effective_k": int(ledger["effective_k"][batch_index]),
                            "unique_k": int(ledger["unique_k"][batch_index]),
                            "backbone_input_k": int(
                                ledger["backbone_input_k"][batch_index]
                            ),
                            "padded_k": int(ledger["padded_k"][batch_index]),
                            "risk_fallback": bool(
                                ledger["risk_fallback"][batch_index]
                            ),
                            "cost_unit": str(ledger["unit"]),
                            "dense_valid_len": int(
                                valid_mask[batch_index].sum().item()
                            ),
                            "selected_dense_indices": [
                                int(value)
                                for value in meta.get("selected_dense_indices", ())
                            ],
                            "max_gap_seconds_cap": float(
                                meta["duca_max_gap_seconds_cap"]
                            ),
                            "observed_max_gap_seconds": float(
                                meta["duca_observed_max_gap_seconds"]
                            ),
                            "budget_protocol_sha256": self.budget_protocol_sha256,
                            "allocation_mode": self.allocation_mode,
                            "mixed_k_schedule_sha256": (
                                self.mixed_k_schedule_sha256
                            ),
                            "detector_training_exposure": (
                                "mixed_k_registered_panel"
                                if self.rime_arm == "uniform_mixed_k"
                                else None
                            ),
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
        if self.rime_arm == "uniform_mixed_k":
            state["mixed_k_schedule"] = {
                "seed": self.mixed_k_schedule_seed,
                "counts": list(self.mixed_k_schedule_counts),
                "cycle": [
                    int(value) for value in self.mixed_k_schedule.cpu().tolist()
                ],
                "sha256": self.mixed_k_schedule_sha256,
                "indices": list(self._last_mixed_k_schedule_indices),
                "source": self._last_mixed_k_schedule_source,
                "target_mean_cost": float(
                    sum(
                        count * cost
                        for count, cost in zip(
                            self.mixed_k_schedule_counts,
                            self.candidate_costs,
                        )
                    )
                    / sum(self.mixed_k_schedule_counts)
                ),
                "probe_executed": False,
                "successful_update_step": int(
                    self._loss_weight_schedule_step.detach().item()
                ),
            }
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
            "mixed_k_schedule_sha256": self.mixed_k_schedule_sha256,
            "allocation_mode": self.allocation_mode,
        }
        self._last_selected_positions = decoded.hard_positions.detach().clone()
        self._last_physical_metas = copy.deepcopy(output_metas)
        return {
            "inputs": selected_inputs,
            "masks": decoded.hard_slot_mask,
            "metas": output_metas,
            "selector_outputs": state,
        }

    def materialize_counterfactual_positions(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas,
        positions: torch.Tensor,
        *,
        requested_k: int,
    ) -> dict[str, Any]:
        """Materialize one externally decoded legal path for train-only measurement.

        This entry point never chooses a position.  It only reuses the production
        physical-axis gather and metadata contract so that train-only
        counterfactual measurements execute the same heavy detector path as RIME.
        """

        self._validate_inputs(inputs, masks, metas)
        requested = int(requested_k)
        if (
            requested not in self.candidate_budgets
            or requested <= 0
            or requested % self.execution_quantum
        ):
            raise ValueError(
                "RIME counterfactual K must be a registered execution bucket"
            )
        valid_mask = masks.to(device=inputs.device, dtype=torch.bool)
        positions = torch.as_tensor(
            positions,
            device=inputs.device,
            dtype=torch.long,
        )
        batch, temporal_len = valid_mask.shape
        if positions.shape != (batch, requested):
            raise ValueError("RIME counterfactual positions must be [B,K]")
        valid_counts = valid_mask.long().sum(dim=1)
        if bool(torch.any(valid_counts < requested).item()):
            raise ValueError(
                "RIME counterfactual path cannot pad a short dense window"
            )
        for batch_index in range(batch):
            active = positions[batch_index]
            if (
                bool(torch.any(active < 0).item())
                or bool(torch.any(active >= valid_counts[batch_index]).item())
                or (
                    requested > 1
                    and not bool(torch.all(active[1:] > active[:-1]).item())
                )
            ):
                raise ValueError(
                    "RIME counterfactual positions must be ordered unique valid indices"
                )

        physical_seconds, source_frames = self._physical_axes(
            metas,
            valid_mask,
            inputs.device,
        )
        caps = physical_exact_uniform_gap_cap(
            physical_seconds,
            valid_mask,
            k=requested,
        )
        occupancy = torch.zeros(
            (batch, temporal_len),
            device=inputs.device,
            dtype=torch.float32,
        )
        slot_assignment = torch.zeros(
            (batch, requested, temporal_len),
            device=inputs.device,
            dtype=torch.float32,
        )
        for batch_index in range(batch):
            active = positions[batch_index]
            occupancy[batch_index].scatter_(0, active, 1.0)
            slot_assignment[batch_index].scatter_(
                1,
                active[:, None],
                1.0,
            )
        slot_mask = torch.ones(
            (batch, requested),
            device=inputs.device,
            dtype=torch.bool,
        )
        effective = torch.full(
            (batch,),
            requested,
            device=inputs.device,
            dtype=torch.long,
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
            effective_k=effective,
            max_gap_seconds=caps,
        )
        selected_inputs = _hard_gather(inputs, positions, slot_mask)
        output_metas = self._write_physical_metadata(
            metas,
            hard,
            physical_seconds,
            source_frames,
            valid_mask,
        )
        ledger = {
            "requested_k": [requested] * batch,
            "effective_k": [requested] * batch,
            "unique_k": [requested] * batch,
            "backbone_input_k": [requested] * batch,
            "padded_k": [requested] * batch,
            "risk_fallback": [False] * batch,
            "dynamic_compute_realized": True,
            "unit": "heavy_rgb_frames",
        }
        for meta in output_metas:
            meta.update(
                {
                    "duca_contract": RIME_CONTRACT,
                    "physical_grid_contract": RIME_CONTRACT,
                    "duca_arm": "train_only_counterfactual_measurement",
                    "duca_requested_k": requested,
                    "duca_effective_k": requested,
                    "duca_unique_k": requested,
                    "duca_backbone_input_k": requested,
                    "duca_padded_k": requested,
                    "duca_risk_fallback": False,
                    "duca_dynamic_compute_realized": True,
                    "duca_cost_unit": "heavy_rgb_frames",
                    "duca_backbone_tail_padding_mode": "none_exact_k_bucket",
                    "duca_execution_quantum": self.execution_quantum,
                }
            )
        return {
            "inputs": selected_inputs,
            "masks": slot_mask,
            "metas": output_metas,
            "positions": positions,
            "physical_seconds": physical_seconds,
            "decoded_source_frames": source_frames,
            "max_gap_seconds": caps,
            "cost_ledger": ledger,
            "hard_forward_only": True,
            "uses_gt_for_selection": False,
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
        if self.rime_arm == "uniform_mixed_k":
            selected = self._select(inputs, valid, metas, training=True)
            state = selected["selector_outputs"]
            state["training_provenance"] = {
                "task": "offline_tad",
                "selection_policy": (
                    "stateless_per_video_cost_matched_exact_uniform_mixed_k"
                ),
                "gt_scope": "detector_loss_only_not_selection",
                "budget_target_scope": "none",
                "inference_uses_gt": False,
                "inference_uses_teacher": False,
                "inference_uses_prediction_cache": False,
                "selected_axis_gt_remap": False,
                "coarse_probe_executed": False,
            }
            return {
                "inputs": selected["inputs"],
                "masks": selected["masks"],
                "metas": selected["metas"],
                "gt_segments": gt_segments,
                "gt_labels": gt_labels,
                "losses": {},
                "selector_outputs": state,
                "counterfactual_request": None,
            }
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
