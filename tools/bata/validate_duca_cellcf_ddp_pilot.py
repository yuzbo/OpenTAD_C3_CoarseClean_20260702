from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mmengine.config import Config

from tools.bata.duca_cellcf_protocol import (
    protocol_from_environment,
    protocol_from_workflow,
)

PILOT_SCHEMA = "duca_cellcf_ddp_pilot_suite_v1"
RUN_SCHEMA = "duca_cellcf_ddp_pilot_run_v1"
CONTEXT_SCHEMA = "duca_cellcf_ddp_pilot_context_v1"
REAL_LOADER_GATE_SCHEMA = "duca_cellcf_real_loader_cuda_gate_v1"
PROBE_SCHEMA = "duca_training_probe_v1"
VARIANT_ORDER = ("uniform", "transition_beta0", "cellcf")
PILOT_CONFIGS = {
    "uniform": "configs/adatad/thumos/duca_cellcf_exact_uniform_fixed384_p0_ddp_pilot.py",
    "transition_beta0": (
        "configs/adatad/thumos/duca_cellcf_transition_beta0_fixed384_p0_ddp_pilot.py"
    ),
    "cellcf": "configs/adatad/thumos/duca_cellcf_fixed384_p0_ddp_pilot.py",
}
FORMAL_CONFIGS = {
    "uniform": (
        "configs/adatad/thumos/"
        "duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train.py"
    ),
    "transition_beta0": (
        "configs/adatad/thumos/"
        "duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py"
    ),
    "cellcf": (
        "configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py"
    ),
}
OFFICIAL_BASE_CONFIG = (
    "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
)
EXPECTED_STEPS = 10
EXPECTED_SUCCESSFUL_UPDATES = EXPECTED_STEPS
EXPECTED_OPTIMIZER_ATTEMPTS = EXPECTED_STEPS + 1
EXPECTED_FORCED_AMP_OVERFLOWS = 1
FIXED_K = 384
DENSE_WINDOW_SIZE = 768
CHECKPOINT_INTERVAL = 5
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SELECTED_K_KEYS = (
    "selected_budget",
    "selected_budgets",
    "selected_count",
    "selected_counts",
    "selected_k",
    "duca_online_selected_count",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _formal_protocol():
    return protocol_from_environment()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must contain a JSON object")
    return resolved, payload


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return output


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value).lower()
    _require(
        bool(_SHA256_PATTERN.fullmatch(normalized)), f"{label} must be a SHA256 digest"
    )
    return normalized


def _require_file_hash(
    payload: Mapping[str, Any], path_key: str, hash_key: str, label: str
) -> Path:
    path = Path(str(payload.get(path_key, ""))).expanduser().resolve()
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(payload.get(hash_key) == _sha256(path), f"{label} hash mismatch")
    return path


def _validate_real_loader_gate(
    path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str,
) -> dict[str, Any]:
    gate_path, payload = _load_json(path, "CellCF real-loader CUDA gate")
    expected_sha256 = _require_sha256(
        expected_sha256, "expected real-loader gate SHA256"
    )
    observed_sha256 = _sha256(gate_path)
    _require(
        observed_sha256 == expected_sha256, "CellCF real-loader gate SHA256 mismatch"
    )
    _require(
        payload.get("schema") == REAL_LOADER_GATE_SCHEMA,
        "CellCF real-loader gate schema mismatch",
    )
    _require(payload.get("ok") is True, "CellCF real-loader gate must declare ok=true")
    _require(
        payload.get("git_commit") == expected_commit,
        "CellCF real-loader gate is bound to another commit",
    )
    from tools.bata.validate_duca_cellcf_real_loader_gate import (
        validate_real_loader_gate_artifact,
    )

    validated = validate_real_loader_gate_artifact(
        gate_path,
        expected_commit=expected_commit,
        expected_sha256=expected_sha256,
        require_clean=False,
    )
    _require(
        validated.get("training_profile") == _formal_protocol().name,
        "CellCF real-loader gate training profile differs from the pilot",
    )
    if "task" in payload:
        _require(
            payload.get("task") == "offline_temporal_action_detection",
            "CellCF real-loader gate must describe offline TAD",
        )
    return {
        "path": str(gate_path),
        "sha256": observed_sha256,
        "schema": REAL_LOADER_GATE_SCHEMA,
        "git_commit": expected_commit,
        "training_profile": str(validated["training_profile"]),
    }


def _validate_config_contract(root: Path, variant: str) -> dict[str, Any]:
    _require(variant in VARIANT_ORDER, f"unknown CellCF pilot variant: {variant}")
    formal_path = (root / FORMAL_CONFIGS[variant]).resolve()
    pilot_path = (root / PILOT_CONFIGS[variant]).resolve()
    official_path = (root / OFFICIAL_BASE_CONFIG).resolve()
    for path, label in (
        (formal_path, f"{variant} formal config"),
        (pilot_path, f"{variant} pilot config"),
        (official_path, "official AdaTAD base config"),
    ):
        _require(path.is_file(), f"{label} is missing: {path}")

    formal = Config.fromfile(str(formal_path))
    pilot = Config.fromfile(str(pilot_path))
    official = Config.fromfile(str(official_path))
    contract = formal.duca_transition_only_contract
    selector = formal.model.frame_selector
    training_protocol = protocol_from_workflow(formal.workflow)
    _require(
        training_protocol == _formal_protocol(),
        f"{variant}: formal training profile differs from the pilot environment",
    )

    _require(
        contract.task == "offline_temporal_action_detection",
        f"{variant}: formal task must be offline TAD",
    )
    _require(
        contract.online_tad is False and contract.streaming is False,
        f"{variant}: online or streaming semantics are forbidden",
    )
    _require(
        contract.full_window_selector is True,
        f"{variant}: selector must see the full window",
    )
    _require(
        formal.model.type == "ActionFormer", f"{variant}: detector wrapper drifted"
    )
    _require(
        formal.model.rpn_head.type == "ActionFormerHead",
        f"{variant}: official ActionFormerHead is required",
    )
    _require(
        _plain(formal.model.rpn_head) == _plain(official.model.rpn_head),
        f"{variant}: official ActionFormer head semantics drifted",
    )
    _require(
        contract.detector_head_changed is False, f"{variant}: detector head changed"
    )
    _require(
        contract.detector_loss_changed is False, f"{variant}: detector loss changed"
    )
    _require(contract.detector_nms_changed is False, f"{variant}: detector NMS changed")
    _require(selector.budget_mode == "fixed", f"{variant}: budget mode must be fixed")
    _require(
        int(selector.budget) == FIXED_K,
        f"{variant}: selected-frame budget must be K=384",
    )
    _require(
        int(selector.dense_window_size) == DENSE_WINDOW_SIZE,
        f"{variant}: dense selector window must be T=768",
    )
    _require(
        int(formal.model.backbone.backbone.total_frames) == FIXED_K,
        f"{variant}: VideoMAE must consume K=384",
    )
    _require(
        int(formal.model.projection.max_seq_len) == FIXED_K,
        f"{variant}: detector projection must remain length 384",
    )
    _require(
        selector.acquisition_policy == "local_cell_deformation",
        f"{variant}: local-cell acquisition is not active",
    )
    _require(
        selector.counterfactual_objective == "local_cell_signed_logistic",
        f"{variant}: local-cell utility objective drifted",
    )
    _require(
        selector.detector_gradient_mode == "none",
        f"{variant}: detector bridge must be off",
    )
    _require(
        selector.forbid_ledger is True, f"{variant}: ledger decisions must be forbidden"
    )
    _require(
        selector.forbid_external_actionness is True,
        f"{variant}: external actionness must be forbidden",
    )
    _require(
        selector.forbid_raw_prediction_cache is True,
        f"{variant}: raw-prediction cache must be forbidden",
    )

    utility_weight = float(selector.counterfactual_utility_distillation_weight)
    if variant == "uniform":
        _require(
            selector.local_cell_force_exact_uniform is True,
            "uniform: exact-uniform flag is off",
        )
        _require(utility_weight == 0.0, "uniform: detector utility must be disabled")
        _require(
            selector.require_counterfactual_utility_teacher is False,
            "uniform: utility teacher must be disabled",
        )
    elif variant == "transition_beta0":
        _require(
            selector.local_cell_force_exact_uniform is False,
            "transition_beta0: selector must remain learned",
        )
        _require(
            utility_weight == 0.0, "transition_beta0: detector utility must be beta=0"
        )
        _require(
            selector.require_counterfactual_utility_teacher is False,
            "transition_beta0: utility teacher must be disabled",
        )
    else:
        _require(
            selector.local_cell_force_exact_uniform is False,
            "cellcf: selector is uniform",
        )
        _require(utility_weight > 0.0, "cellcf: detector utility must be enabled")
        _require(
            selector.require_counterfactual_utility_teacher is True,
            "cellcf: integrated utility teacher must fail closed",
        )

    _require(
        int(formal.workflow.end_epoch) == _formal_protocol().end_epoch,
        f"{variant}: formal training profile drifted",
    )
    _require(
        int(formal.scheduler.max_epoch) == _formal_protocol().end_epoch,
        f"{variant}: scheduler training profile drifted",
    )
    _require(
        int(formal.workflow.checkpoint_interval) == CHECKPOINT_INTERVAL,
        f"{variant}: formal checkpoint interval must remain five",
    )
    _require(
        int(formal.workflow.expected_successful_optimizer_updates)
        == _formal_protocol().expected_successful_optimizer_updates,
        f"{variant}: formal successful-update count drifted",
    )
    _require(
        formal.workflow.primary_checkpoint_state_key == "state_dict_ema",
        f"{variant}: formal primary state must remain EMA",
    )
    _require(
        int(formal.workflow.primary_checkpoint_epoch)
        == _formal_protocol().terminal_epoch,
        f"{variant}: formal primary checkpoint epoch drifted",
    )

    _require(
        _plain(pilot.model) == _plain(formal.model),
        f"{variant}: pilot model differs from formal model",
    )
    _require(
        _plain(pilot.solver) == _plain(formal.solver),
        f"{variant}: pilot solver differs from formal solver",
    )
    _require(
        pilot.model.backbone.backbone.with_cp is False,
        f"{variant}: activation checkpointing must remain disabled",
    )
    _require(pilot.solver.static_graph is False, f"{variant}: static DDP is forbidden")
    _require(
        pilot.solver.find_unused_parameters is True,
        f"{variant}: dynamic DDP unused-parameter discovery is required",
    )
    _require(pilot.solver.ema is True, f"{variant}: pilot must exercise EMA accounting")
    _require(
        pilot.workflow.formal_protocol == "duca_cellcf_pilot_v1",
        f"{variant}: pilot protocol marker drifted",
    )
    _require(
        int(pilot.workflow.max_train_iters) == EXPECTED_STEPS,
        f"{variant}: pilot must consume ten batches",
    )
    _require(
        int(pilot.workflow.force_amp_overflow_attempts)
        == EXPECTED_FORCED_AMP_OVERFLOWS,
        f"{variant}: pilot must inject exactly one AMP overflow",
    )
    _require(
        pilot.workflow.disable_checkpoint is True,
        f"{variant}: pilot checkpoint writes must be off",
    )
    _require(
        int(pilot.workflow.checkpoint_interval) == CHECKPOINT_INTERVAL,
        f"{variant}: pilot must preserve checkpoint_interval=5",
    )
    _require(
        int(pilot.workflow.end_epoch) == 1,
        f"{variant}: pilot must run one truncated epoch",
    )
    _require(
        int(pilot.workflow.val_start_epoch) > int(pilot.workflow.end_epoch),
        f"{variant}: pilot validation must be disabled",
    )
    _require(
        int(pilot.workflow.max_amp_retries_per_batch) > 0,
        f"{variant}: AMP replay must be enabled",
    )
    _require(
        pilot.workflow.fail_on_amp_replay_exhaustion is True,
        f"{variant}: AMP replay must fail closed",
    )
    _require(
        pilot.workflow.require_finite_train_loss is True,
        f"{variant}: finite-loss checks must remain enabled",
    )
    _require(
        pilot.workflow.require_training_probe_context is True,
        f"{variant}: probe context binding must be required",
    )

    return {
        "variant": variant,
        "formal_config": FORMAL_CONFIGS[variant],
        "formal_config_sha256": _sha256(formal_path),
        "pilot_config": PILOT_CONFIGS[variant],
        "pilot_config_sha256": _sha256(pilot_path),
        "task": "offline_temporal_action_detection",
        "fixed_k": FIXED_K,
        "training_profile": _formal_protocol().name,
        "formal_end_epoch": _formal_protocol().end_epoch,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "pilot_checkpoint_disabled": True,
    }


def _as_int_vector(value: Any, label: str) -> list[int]:
    _require(
        isinstance(value, list) and value,
        f"{label} must be a non-empty per-sample list",
    )
    vector: list[int] = []
    for item in value:
        _require(not isinstance(item, bool), f"{label} contains a boolean")
        integer = int(item)
        _require(float(item) == float(integer), f"{label} contains a non-integer value")
        vector.append(integer)
    return vector


def _selected_k_vector(
    step: Mapping[str, Any],
    effective: list[int],
    variant: str,
    *,
    selected_k_proven_by_bound_gate: bool,
) -> tuple[list[int], str]:
    for key in _SELECTED_K_KEYS:
        if key not in step:
            continue
        selected = _as_int_vector(step[key], f"{variant}: {key}")
        _require(
            len(selected) == len(effective),
            f"{variant}: selected-K evidence is not batch-aligned",
        )
        return selected, f"training_probe.{key}"
    positions = step.get("selected_positions")
    if isinstance(positions, list) and len(positions) == len(effective):
        selected = []
        for sample in positions:
            _require(
                isinstance(sample, list),
                f"{variant}: selected_positions must be batched",
            )
            selected.append(sum(int(position) >= 0 for position in sample))
        return selected, "training_probe.selected_positions"
    if selected_k_proven_by_bound_gate:
        # The exact bound real-loader gate checks selected-mask count == effective K
        # on this commit. The arm probe still has to expose each effective-K vector.
        return list(effective), "exact_commit_real_loader_gate"
    raise AssertionError(
        f"{variant}: selector step is missing per-sample selected-K evidence"
    )


def _step_k_vectors(
    step: Mapping[str, Any],
    variant: str,
    *,
    selected_k_proven_by_bound_gate: bool,
) -> tuple[list[int], list[int], list[int], str]:
    requested = _as_int_vector(
        step.get("requested_budget"), f"{variant}: requested_budget"
    )
    effective = _as_int_vector(
        step.get("effective_budget"), f"{variant}: effective_budget"
    )
    _require(
        len(requested) == len(effective),
        f"{variant}: requested/effective K are not aligned",
    )
    selected, selected_source = _selected_k_vector(
        step,
        effective,
        variant,
        selected_k_proven_by_bound_gate=selected_k_proven_by_bound_gate,
    )
    _require(
        all(value == FIXED_K for value in requested),
        f"{variant}: requested K must be exactly 384",
    )
    _require(
        all(0 < value <= FIXED_K for value in effective),
        f"{variant}: effective K must lie in [1,384]",
    )
    _require(selected == effective, f"{variant}: selected K differs from effective K")
    return requested, selected, effective, selected_source


def _k_batch_kind(effective: list[int]) -> str:
    if all(value == FIXED_K for value in effective):
        return "full"
    if any(value == FIXED_K for value in effective) and any(
        value < FIXED_K for value in effective
    ):
        return "mixed"
    if all(value < FIXED_K for value in effective):
        return "all_short"
    raise AssertionError("effective-K vector has an invalid fixed-budget shape")


def _selector_replay_payload(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _plain(value)
        for key, value in step.items()
        if key != "optimizer_step_ran"
    }


def _validate_amp_history(audit: Mapping[str, Any], variant: str) -> None:
    expected_batches = [0, 0, *range(1, EXPECTED_STEPS)]
    observed_batches = audit.get("attempt_batch_indices")
    _require(
        observed_batches == expected_batches,
        f"{variant}: optimizer attempts do not prove a first-batch replay",
    )
    history = audit.get("amp_scale_history")
    _require(
        isinstance(history, list) and len(history) == EXPECTED_OPTIMIZER_ATTEMPTS,
        f"{variant}: AMP scale history is incomplete",
    )
    expected_retries = [0, 1, *([0] * (EXPECTED_STEPS - 1))]
    expected_success = [False, *([True] * EXPECTED_STEPS)]
    for index, (entry, batch_index, retry_index, succeeded) in enumerate(
        zip(history, expected_batches, expected_retries, expected_success)
    ):
        _require(
            isinstance(entry, dict), f"{variant}: AMP history entry {index} is invalid"
        )
        _require(
            int(entry.get("batch_index", -1)) == batch_index,
            f"{variant}: AMP batch index drift",
        )
        _require(
            int(entry.get("retry_index", -1)) == retry_index,
            f"{variant}: AMP retry index drift",
        )
        _require(
            entry.get("optimizer_step_ran") is succeeded,
            f"{variant}: AMP success history is inconsistent",
        )
        before = float(entry.get("before", math.nan))
        after = float(entry.get("after", math.nan))
        _require(
            math.isfinite(before)
            and math.isfinite(after)
            and before > 0.0
            and after > 0.0,
            f"{variant}: AMP scale history is non-finite",
        )
        if succeeded:
            _require(
                after >= before, f"{variant}: successful AMP update reduced the scale"
            )
        else:
            _require(
                after < before,
                f"{variant}: injected AMP overflow did not reduce the scale",
            )


def _validate_counterfactual_steps(
    steps: list[dict[str, Any]], variant: str
) -> dict[str, Any]:
    if variant != "cellcf":
        _require(
            all("counterfactual" not in step for step in steps),
            f"{variant}: beta=0/control arm emitted utility-teacher evidence",
        )
        return {
            "enabled": False,
            "zero_candidate_steps": 0,
            "positive_candidate_steps": 0,
            "distinct_cell_nonzero_utility": False,
        }

    zero_candidate_steps = 0
    positive_candidate_steps = 0
    distinct_cell_nonzero_utility = False
    candidate_counts: list[int] = []
    for index, step in enumerate(steps):
        counterfactual = step.get("counterfactual")
        _require(
            isinstance(counterfactual, dict),
            f"cellcf: step {index} lacks utility evidence",
        )
        _require(
            counterfactual.get("finite") is True,
            f"cellcf: step {index} utility is non-finite",
        )
        _require(
            counterfactual.get("teacher_kind")
            == "detached_distinct_local_cell_hard_flip_official_actionformer_cls_plus_reg",
            f"cellcf: step {index} used the wrong utility teacher",
        )
        _require(
            counterfactual.get("distillation_loss_kind")
            == "distinct_local_cell_weighted_signed_logistic",
            f"cellcf: step {index} used the wrong utility loss",
        )
        count = int(counterfactual.get("candidate_count", -1))
        utilities = counterfactual.get("candidate_utility_values")
        cells = counterfactual.get("candidate_cell_indices")
        _require(count >= 0, f"cellcf: step {index} candidate count is invalid")
        _require(
            isinstance(utilities, list), f"cellcf: step {index} utility list is missing"
        )
        _require(isinstance(cells, list), f"cellcf: step {index} cell list is missing")
        _require(
            len(utilities) == count and len(cells) == count,
            f"cellcf: step {index} candidate evidence is not count-aligned",
        )
        _require(
            all(math.isfinite(float(value)) for value in utilities),
            f"cellcf: step {index} utility list is non-finite",
        )
        _require(
            all(int(cell) >= 0 for cell in cells),
            f"cellcf: step {index} cell index is invalid",
        )
        informative = counterfactual.get("utility_alignment_informative")
        _require(
            isinstance(informative, bool),
            f"cellcf: step {index} utility honesty flag is missing",
        )
        candidate_counts.append(count)
        if count == 0:
            zero_candidate_steps += 1
            _require(
                not utilities and not cells,
                "cellcf: zero-candidate step fabricated utility evidence",
            )
            _require(
                informative is False,
                "cellcf: zero-candidate step claimed informative utility",
            )
            continue
        positive_candidate_steps += 1
        has_distinct_cells = len({int(cell) for cell in cells}) >= 2
        has_nonzero_utility = any(float(value) != 0.0 for value in utilities)
        if has_distinct_cells and has_nonzero_utility and informative:
            distinct_cell_nonzero_utility = True

    _require(
        zero_candidate_steps > 0, "cellcf: pilot missed honest zero-candidate handling"
    )
    _require(
        positive_candidate_steps > 0,
        "cellcf: pilot missed valid counterfactual candidates",
    )
    _require(
        distinct_cell_nonzero_utility,
        "cellcf: pilot lacks distinct-cell, nonzero, informative utility evidence",
    )
    return {
        "enabled": True,
        "candidate_counts": candidate_counts,
        "zero_candidate_steps": zero_candidate_steps,
        "positive_candidate_steps": positive_candidate_steps,
        "distinct_cell_nonzero_utility": True,
    }


def validate_probe(
    payload: dict[str, Any],
    variant: str,
    *,
    selected_k_proven_by_bound_gate: bool = False,
) -> dict[str, Any]:
    _require(variant in VARIANT_ORDER, f"unknown CellCF pilot variant: {variant}")
    _require(
        payload.get("schema_version") == PROBE_SCHEMA,
        f"{variant}: probe schema mismatch",
    )
    attempted = int(payload.get("attempted_steps", -1))
    successful = int(payload.get("successful_optimizer_steps", -1))
    skipped = int(payload.get("skipped_optimizer_steps", -1))
    _require(
        attempted == EXPECTED_OPTIMIZER_ATTEMPTS,
        f"{variant}: pilot must make exactly eleven optimizer attempts",
    )
    _require(
        successful == EXPECTED_SUCCESSFUL_UPDATES,
        f"{variant}: successful updates must equal ten",
    )
    _require(
        skipped == EXPECTED_FORCED_AMP_OVERFLOWS,
        f"{variant}: exactly one update must be skipped",
    )
    _require(
        int(payload.get("finite_loss_steps", -1)) == attempted,
        f"{variant}: pilot produced a non-finite loss",
    )
    _require(
        int(payload.get("finite_gradient_steps", -1)) == successful,
        f"{variant}: finite-gradient accounting does not isolate the injected overflow",
    )
    _require(
        payload.get("static_graph") is False, f"{variant}: static_graph must be false"
    )
    _require(
        payload.get("find_unused_parameters") is True,
        f"{variant}: find_unused_parameters must be true",
    )
    _require(
        int(payload.get("world_size", 0)) == 1, f"{variant}: world_size must be one"
    )

    coverage = payload.get("parameter_group_coverage")
    _require(
        isinstance(coverage, dict), f"{variant}: parameter-group coverage is missing"
    )
    for group in (
        "backbone",
        "coarse_probe",
        "selector",
        "projection",
        "detector_head",
    ):
        counts = coverage.get(group)
        _require(
            isinstance(counts, dict), f"{variant}: missing {group} parameter group"
        )
        trainable = int(counts.get("trainable", 0))
        gradient_seen = int(counts.get("gradient_seen", 0))
        _require(trainable > 0, f"{variant}: {group} has no trainable parameter")
        _require(
            0 < gradient_seen <= trainable,
            f"{variant}: {group} did not exercise a trainable gradient path",
        )
    _require(
        isinstance(payload.get("gradient_never_seen"), list),
        f"{variant}: unused-parameter evidence is missing",
    )

    all_steps = payload.get("selector_steps")
    _require(
        isinstance(all_steps, list) and len(all_steps) == attempted,
        f"{variant}: selector attempt trace is incomplete",
    )
    _require(
        all(isinstance(step, dict) for step in all_steps),
        f"{variant}: invalid selector trace",
    )
    _require(
        all_steps[0].get("optimizer_step_ran") is False,
        f"{variant}: injected overflow must be the first optimizer attempt",
    )
    _require(
        all(step.get("optimizer_step_ran") is True for step in all_steps[1:]),
        f"{variant}: pilot contains an unexpected extra AMP skip",
    )
    _require(
        _selector_replay_payload(all_steps[0])
        == _selector_replay_payload(all_steps[1]),
        f"{variant}: replay did not restore identical selector/RNG state",
    )
    successful_steps = all_steps[1:]

    audit = payload.get("update_audit")
    _require(isinstance(audit, dict), f"{variant}: successful-update audit is missing")
    expected_counters = {
        "attempted_batches": EXPECTED_STEPS,
        "optimizer_attempts": EXPECTED_OPTIMIZER_ATTEMPTS,
        "successful_optimizer_updates": EXPECTED_SUCCESSFUL_UPDATES,
        "amp_skipped_attempts": EXPECTED_FORCED_AMP_OVERFLOWS,
        "replayed_batches": 1,
        "replay_exhaustions": 0,
        "scheduler_updates": EXPECTED_SUCCESSFUL_UPDATES,
        "ema_updates": EXPECTED_SUCCESSFUL_UPDATES,
        "duca_schedule_updates": EXPECTED_SUCCESSFUL_UPDATES,
        "forced_amp_overflow_attempts": EXPECTED_FORCED_AMP_OVERFLOWS,
        "max_amp_retries_observed": 1,
        "replay_state_restorations": 1,
    }
    for key, expected in expected_counters.items():
        _require(
            int(audit.get(key, -1)) == expected,
            f"{variant}: successful-update accounting mismatch for {key}",
        )
    _validate_amp_history(audit, variant)

    schedule_steps: list[int] = []
    k_coverage = {"full": False, "mixed": False, "all_short": False}
    k_evidence: dict[str, dict[str, Any]] = {}
    for index, step in enumerate(successful_steps):
        schedule = step.get("loss_weight_schedule")
        _require(
            isinstance(schedule, dict),
            f"{variant}: selector schedule evidence is missing",
        )
        schedule_steps.append(int(schedule.get("step", -1)))
        _, selected, effective, selected_source = _step_k_vectors(
            step,
            variant,
            selected_k_proven_by_bound_gate=selected_k_proven_by_bound_gate,
        )
        kind = _k_batch_kind(effective)
        k_coverage[kind] = True
        k_evidence.setdefault(
            kind,
            {
                "successful_update_index": index,
                "selected_k": selected,
                "effective_k": effective,
                "selected_k_source": selected_source,
            },
        )
    _require(
        schedule_steps == list(range(EXPECTED_SUCCESSFUL_UPDATES)),
        f"{variant}: selector schedule did not advance only on successful updates",
    )
    _require(
        all(k_coverage.values()),
        f"{variant}: full/mixed/all-short K coverage is incomplete",
    )
    utility_evidence = _validate_counterfactual_steps(successful_steps, variant)

    max_memory = float(payload.get("max_cuda_memory_mb", math.nan))
    _require(
        math.isfinite(max_memory) and max_memory > 0.0,
        f"{variant}: CUDA memory evidence is invalid",
    )
    return {
        "variant": variant,
        "attempted_optimizer_steps": attempted,
        "successful_optimizer_steps": successful,
        "injected_amp_overflow_attempts": EXPECTED_FORCED_AMP_OVERFLOWS,
        "replay_restored": True,
        "successful_update_accounting": {
            "scheduler": EXPECTED_SUCCESSFUL_UPDATES,
            "ema": EXPECTED_SUCCESSFUL_UPDATES,
            "selector": EXPECTED_SUCCESSFUL_UPDATES,
            "selector_schedule_steps": schedule_steps,
        },
        "k_coverage": k_coverage,
        "k_evidence": k_evidence,
        "utility_evidence": utility_evidence,
        "parameter_group_coverage": coverage,
        "gradient_never_seen": payload.get("gradient_never_seen", []),
        "max_cuda_memory_mb": max_memory,
    }


def _resolved_pilot_config_sha256(context: Mapping[str, Any]) -> str:
    previous_probe = os.environ.get("DUCA_TRAINING_PROBE_JSON")
    previous_context = os.environ.get("DUCA_TRAINING_PROBE_CONTEXT_JSON")
    os.environ["DUCA_TRAINING_PROBE_JSON"] = str(context["training_probe_json"])
    os.environ["DUCA_TRAINING_PROBE_CONTEXT_JSON"] = str(context["context_json"])
    try:
        config = Config.fromfile(str(context["source_config_path"]))
        config.merge_from_dict(
            {
                "work_dir": str(context["work_dir"]),
                "model.backbone.custom.pretrain": str(context["checkpoint_path"]),
            }
        )
        return _canonical_sha256(config.to_dict())
    finally:
        if previous_probe is None:
            os.environ.pop("DUCA_TRAINING_PROBE_JSON", None)
        else:
            os.environ["DUCA_TRAINING_PROBE_JSON"] = previous_probe
        if previous_context is None:
            os.environ.pop("DUCA_TRAINING_PROBE_CONTEXT_JSON", None)
        else:
            os.environ["DUCA_TRAINING_PROBE_CONTEXT_JSON"] = previous_context


def _validate_manifest(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    expected_commit: str,
) -> None:
    _require(
        manifest.get("schema") == RUN_SCHEMA, "CellCF DDP pilot run schema mismatch"
    )
    _require(
        manifest.get("git_commit") == expected_commit,
        "CellCF DDP pilot manifest commit mismatch",
    )
    _require(
        manifest.get("real_loader_gate_json") == gate["path"],
        "CellCF DDP pilot manifest gate path mismatch",
    )
    _require(
        manifest.get("real_loader_gate_sha256") == gate["sha256"],
        "CellCF DDP pilot manifest gate SHA256 mismatch",
    )
    _require(
        manifest.get("real_loader_gate_git_commit") == expected_commit,
        "CellCF DDP pilot manifest gate commit mismatch",
    )
    _require(
        str(manifest.get("slurm_job_id", "")).isdigit(),
        "pilot lacks a numeric Slurm job id",
    )
    _require(
        not manifest.get("slurm_array_job_id"),
        "CellCF pilot must not run as a Slurm array",
    )
    _require(bool(manifest.get("pilot_nonce")), "CellCF pilot nonce is missing")
    _require(
        int(manifest.get("world_size", 0)) == 1, "CellCF pilot world_size must be one"
    )
    _require(
        int(manifest.get("nproc_per_node", 0)) == 1,
        "CellCF pilot nproc_per_node must be one",
    )
    _require(
        manifest.get("torch_distributed_entrypoint")
        == "python -m torch.distributed.run",
        "CellCF pilot must use torch.distributed.run",
    )
    _require(
        manifest.get("task") == "offline_temporal_action_detection",
        "CellCF pilot must remain offline TAD",
    )
    _require(int(manifest.get("fixed_k", 0)) == FIXED_K, "CellCF pilot fixed K drifted")
    _require(
        int(manifest.get("dense_window_size", 0)) == DENSE_WINDOW_SIZE,
        "CellCF pilot dense window drifted",
    )
    _require(
        manifest.get("training_profile") == _formal_protocol().name,
        "CellCF formal training profile drifted",
    )
    _require(
        int(manifest.get("formal_end_epoch", 0))
        == _formal_protocol().end_epoch,
        "CellCF formal epoch count drifted",
    )
    _require(
        int(manifest.get("checkpoint_interval", 0)) == CHECKPOINT_INTERVAL,
        "CellCF checkpoint interval drifted",
    )
    _require(
        manifest.get("pilot_checkpoint_disabled") is True,
        "pilot checkpoint writes must be disabled",
    )
    _require(
        int(manifest.get("successful_updates_per_arm", 0))
        == EXPECTED_SUCCESSFUL_UPDATES,
        "CellCF pilot update count drifted",
    )
    _require(
        int(manifest.get("forced_amp_overflow_attempts_per_arm", 0))
        == EXPECTED_FORCED_AMP_OVERFLOWS,
        "CellCF pilot overflow count drifted",
    )
    _require(
        manifest.get("variants") == list(VARIANT_ORDER),
        "CellCF DDP pilot arm order mismatch",
    )
    _require_file_hash(
        manifest, "checkpoint_path", "checkpoint_sha256", "AdaTAD pretrain"
    )
    _require_file_hash(
        manifest,
        "official_asformer_source",
        "official_asformer_source_sha256",
        "official ASFormer source",
    )
    _require_file_hash(
        manifest,
        "canonical_env_path",
        "canonical_env_sha256",
        "CellCF canonical environment",
    )
    _require(
        manifest_path.parent.name != "probes",
        "CellCF manifest must live above the probe directory",
    )


def _validate_probe_bindings(
    payload: Mapping[str, Any],
    *,
    root: Path,
    variant: str,
    probe_path: Path,
    context_path: Path,
    context: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
    config_contract: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> None:
    _require(
        context.get("schema") == CONTEXT_SCHEMA,
        f"{variant}: pilot context schema mismatch",
    )
    expected_context = {
        "git_commit": manifest["git_commit"],
        "variant": variant,
        "seed": int(manifest["seed"]),
        "slurm_job_id": str(manifest["slurm_job_id"]),
        "pilot_nonce": manifest["pilot_nonce"],
        "world_size": 1,
        "source_config_path": str((root / PILOT_CONFIGS[variant]).resolve()),
        "source_config_sha256": config_contract["pilot_config_sha256"],
        "formal_config_path": str((root / FORMAL_CONFIGS[variant]).resolve()),
        "formal_config_sha256": config_contract["formal_config_sha256"],
        "training_probe_json": str(probe_path),
        "context_json": str(context_path),
        "checkpoint_path": manifest["checkpoint_path"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "real_loader_gate_json": gate["path"],
        "real_loader_gate_sha256": gate["sha256"],
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "task": "offline_temporal_action_detection",
        "fixed_k": FIXED_K,
        "training_profile": _formal_protocol().name,
        "formal_end_epoch": _formal_protocol().end_epoch,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "pilot_checkpoint_disabled": True,
    }
    for key, expected in expected_context.items():
        _require(context.get(key) == expected, f"{variant}: context {key} mismatch")

    bindings = payload.get("bindings")
    _require(
        isinstance(bindings, dict), f"{variant}: training probe bindings are missing"
    )
    expected_bindings = {
        "git_commit": manifest["git_commit"],
        "seed": int(manifest["seed"]),
        "slurm_job_id": str(manifest["slurm_job_id"]),
        "source_config_path": context["source_config_path"],
        "source_config_sha256": context["source_config_sha256"],
        "training_probe_json": str(probe_path),
        "context_json": str(context_path),
        "context_json_sha256": _sha256(context_path),
        "context": context,
    }
    for key, expected in expected_bindings.items():
        _require(
            bindings.get(key) == expected, f"{variant}: probe binding {key} mismatch"
        )
    _require(
        bindings.get("resolved_config_sha256")
        == _resolved_pilot_config_sha256(context),
        f"{variant}: resolved pilot config hash mismatch",
    )


def validate_preflight(
    *,
    repo_root: str | Path,
    real_loader_gate_json: str | Path,
    expected_commit: str,
    expected_real_loader_gate_sha256: str,
    require_clean: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    _require(root.is_dir(), f"repository root is missing: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    _require(
        commit == expected_commit, "CellCF pilot HEAD differs from the expected commit"
    )
    if require_clean:
        _require(
            not _git(root, "status", "--porcelain", "--untracked-files=normal"),
            "CellCF pilot requires a clean exact-commit checkout",
        )
    gate = _validate_real_loader_gate(
        real_loader_gate_json,
        expected_commit=commit,
        expected_sha256=expected_real_loader_gate_sha256,
    )
    config_contracts = [
        _validate_config_contract(root, variant) for variant in VARIANT_ORDER
    ]
    return {
        "schema": "duca_cellcf_ddp_pilot_preflight_v1",
        "ok": True,
        "git_commit": commit,
        "real_loader_gate_json": gate["path"],
        "real_loader_gate_sha256": gate["sha256"],
        "task": "offline_temporal_action_detection",
        "fixed_k": FIXED_K,
        "training_profile": _formal_protocol().name,
        "formal_end_epoch": _formal_protocol().end_epoch,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "variants": list(VARIANT_ORDER),
        "config_contracts": config_contracts,
    }


def validate_pilot_suite(
    *,
    repo_root: str | Path,
    probe_dir: str | Path,
    real_loader_gate_json: str | Path,
    expected_commit: str,
    expected_real_loader_gate_sha256: str,
    require_clean: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    preflight = validate_preflight(
        repo_root=root,
        real_loader_gate_json=real_loader_gate_json,
        expected_commit=expected_commit,
        expected_real_loader_gate_sha256=expected_real_loader_gate_sha256,
        require_clean=require_clean,
    )
    gate = {
        "path": preflight["real_loader_gate_json"],
        "sha256": preflight["real_loader_gate_sha256"],
    }
    probes = Path(probe_dir).expanduser().resolve()
    _require(probes.is_dir(), f"CellCF probe directory is missing: {probes}")
    manifest_path, manifest = _load_json(
        probes.parent / "manifest.json", "CellCF pilot manifest"
    )
    _validate_manifest(
        manifest_path,
        manifest,
        gate=gate,
        expected_commit=expected_commit,
    )
    seed = int(manifest.get("seed", -1))
    _require(seed >= 0, "CellCF pilot seed is invalid")

    config_contracts = {item["variant"]: item for item in preflight["config_contracts"]}
    variants = []
    for variant in VARIANT_ORDER:
        contract = config_contracts[variant]
        pilot_config_path = (root / PILOT_CONFIGS[variant]).resolve()
        formal_config_path = (root / FORMAL_CONFIGS[variant]).resolve()
        probe_path, probe = _load_json(
            probes / f"{variant}.training_probe.json",
            f"{variant} CellCF training probe",
        )
        context_path, context = _load_json(
            probes / f"{variant}.context.json", f"{variant} CellCF pilot context"
        )
        _require(probe_path.parent == probes, f"{variant}: probe escaped the run root")
        _require(
            context_path.parent == probes, f"{variant}: context escaped the run root"
        )
        _require(
            Path(str(context.get("source_config_path", ""))).resolve()
            == pilot_config_path,
            f"{variant}: probe did not use the canonical pilot config",
        )
        _require(
            Path(str(context.get("formal_config_path", ""))).resolve()
            == formal_config_path,
            f"{variant}: probe is not bound to the canonical formal config",
        )
        summary = validate_probe(
            probe,
            variant,
            selected_k_proven_by_bound_gate=True,
        )
        _validate_probe_bindings(
            probe,
            root=root,
            variant=variant,
            probe_path=probe_path,
            context_path=context_path,
            context=context,
            manifest_path=manifest_path,
            manifest=manifest,
            config_contract=contract,
            gate=gate,
        )
        variants.append(
            {
                **summary,
                "pilot_config": PILOT_CONFIGS[variant],
                "pilot_config_sha256": contract["pilot_config_sha256"],
                "formal_config": FORMAL_CONFIGS[variant],
                "formal_config_sha256": contract["formal_config_sha256"],
                "probe_json": str(probe_path),
                "probe_json_sha256": _sha256(probe_path),
                "context_json": str(context_path),
                "context_json_sha256": _sha256(context_path),
                "validated_probe_summary_sha256": _canonical_sha256(summary),
            }
        )

    return {
        "schema": PILOT_SCHEMA,
        "ok": True,
        "status": "tested_forced_overflow_ddp_pilot",
        "git_commit": expected_commit,
        "seed": seed,
        "slurm_job_id": str(manifest["slurm_job_id"]),
        "pilot_nonce": manifest["pilot_nonce"],
        "task": "offline_temporal_action_detection",
        "fixed_k": FIXED_K,
        "training_profile": _formal_protocol().name,
        "dense_window_size": DENSE_WINDOW_SIZE,
        "world_size": 1,
        "torch_distributed_entrypoint": "python -m torch.distributed.run",
        "formal_end_epoch": _formal_protocol().end_epoch,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "pilot_checkpoint_disabled": True,
        "expected_successful_updates_per_arm": EXPECTED_SUCCESSFUL_UPDATES,
        "expected_forced_amp_overflows_per_arm": EXPECTED_FORCED_AMP_OVERFLOWS,
        "variant_order": list(VARIANT_ORDER),
        "arm_order": list(VARIANT_ORDER),
        "real_loader_gate_json": gate["path"],
        "real_loader_gate_sha256": gate["sha256"],
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "config_contracts": preflight["config_contracts"],
        "variants": variants,
    }


def validate_pilot_artifact(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_commit: str,
    expected_real_loader_gate_sha256: str,
    require_clean: bool = False,
) -> dict[str, Any]:
    artifact_path, artifact = _load_json(path, "CellCF DDP pilot artifact")
    _require(artifact.get("schema") == PILOT_SCHEMA, "CellCF DDP pilot schema mismatch")
    _require(artifact.get("ok") is True, "CellCF DDP pilot must declare ok=true")
    _require(
        artifact.get("git_commit") == expected_commit,
        "CellCF DDP pilot commit is stale",
    )
    expected_gate_sha = _require_sha256(
        expected_real_loader_gate_sha256, "expected real-loader gate SHA256"
    )
    _require(
        artifact.get("real_loader_gate_sha256") == expected_gate_sha,
        "CellCF DDP pilot real-loader gate binding is stale",
    )
    manifest_path = Path(str(artifact.get("run_manifest_path", ""))).resolve()
    _require(manifest_path.is_file(), "CellCF DDP pilot manifest is missing")
    _require(
        artifact.get("run_manifest_sha256") == _sha256(manifest_path),
        "CellCF DDP pilot manifest hash mismatch",
    )
    variants = artifact.get("variants")
    _require(isinstance(variants, list), "CellCF DDP pilot variant evidence is missing")
    _require(
        [item.get("variant") for item in variants] == list(VARIANT_ORDER),
        "CellCF DDP pilot arm order mismatch",
    )
    for item in variants:
        variant = str(item["variant"])
        probe_path = Path(str(item.get("probe_json", ""))).resolve()
        context_path = Path(str(item.get("context_json", ""))).resolve()
        _require(
            item.get("probe_json_sha256") == _sha256(probe_path),
            f"{variant}: raw probe hash mismatch",
        )
        _require(
            item.get("context_json_sha256") == _sha256(context_path),
            f"{variant}: raw context hash mismatch",
        )

    fresh = validate_pilot_suite(
        repo_root=repo_root,
        probe_dir=manifest_path.parent / "probes",
        real_loader_gate_json=artifact["real_loader_gate_json"],
        expected_commit=expected_commit,
        expected_real_loader_gate_sha256=expected_gate_sha,
        require_clean=require_clean,
    )
    _require(
        artifact == fresh,
        "CellCF DDP pilot artifact does not exactly match revalidated raw evidence",
    )
    return {
        "path": str(artifact_path),
        "sha256": _sha256(artifact_path),
        "git_commit": expected_commit,
        "real_loader_gate_sha256": expected_gate_sha,
        "slurm_job_id": fresh["slurm_job_id"],
        "variants": fresh["variants"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the three-arm fixed-K CellCF forced-overflow DDP pilot"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--probe-dir")
    parser.add_argument("--real-loader-gate-json", required=True)
    parser.add_argument("--expected-real-loader-gate-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise SystemExit("refusing to overwrite CellCF DDP pilot evidence")
    if args.precheck_only:
        payload = validate_preflight(
            repo_root=args.repo_root,
            real_loader_gate_json=args.real_loader_gate_json,
            expected_commit=args.expected_commit,
            expected_real_loader_gate_sha256=args.expected_real_loader_gate_sha256,
            require_clean=args.require_clean,
        )
    else:
        if not args.probe_dir:
            raise SystemExit("--probe-dir is required unless --precheck-only is used")
        payload = validate_pilot_suite(
            repo_root=args.repo_root,
            probe_dir=args.probe_dir,
            real_loader_gate_json=args.real_loader_gate_json,
            expected_commit=args.expected_commit,
            expected_real_loader_gate_sha256=args.expected_real_loader_gate_sha256,
            require_clean=args.require_clean,
        )
    output = _exclusive_write_json(output_path, payload)
    if not args.precheck_only:
        validate_pilot_artifact(
            output,
            repo_root=args.repo_root,
            expected_commit=args.expected_commit,
            expected_real_loader_gate_sha256=args.expected_real_loader_gate_sha256,
            require_clean=args.require_clean,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
