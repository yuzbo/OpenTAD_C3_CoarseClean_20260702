from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from tools.bata.create_duca_rime_splits import TRAIN_ROLES, validate_rime_splits
from tools.bata.duca_rime_phase2 import spearman


_HRIME_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "opentad" / "models" / "duca" / "hrime.py"
)
_HRIME_SPEC = importlib.util.spec_from_file_location(
    "_hrime_stage1_oracle_core",
    _HRIME_MODULE_PATH,
)
if _HRIME_SPEC is None or _HRIME_SPEC.loader is None:
    raise ImportError(f"cannot load the pure H-RIME core: {_HRIME_MODULE_PATH}")
_HRIME_CORE = importlib.util.module_from_spec(_HRIME_SPEC)
sys.modules[_HRIME_SPEC.name] = _HRIME_CORE
_HRIME_SPEC.loader.exec_module(_HRIME_CORE)
HRIME_EXECUTION_QUANTUM = _HRIME_CORE.HRIME_EXECUTION_QUANTUM
MCKPOption = _HRIME_CORE.MCKPOption
MCKPWindow = _HRIME_CORE.MCKPWindow
VideoWindowRef = _HRIME_CORE.VideoWindowRef
canonical_sha256 = _HRIME_CORE.canonical_sha256
canonicalize_effective_k_options = _HRIME_CORE.canonicalize_effective_k_options
group_video_windows = _HRIME_CORE.group_video_windows
solve_exact_mckp = _HRIME_CORE.solve_exact_mckp


PREREGISTRATION_SCHEMA = "hrime_stage1_preregistration_v1"
WINDOW_OPTION_SCHEMA = "hrime_stage1_window_option_measurement_v1"
PLAN_SCHEMA = "hrime_stage1_oracle_plan_v1"
EXECUTION_RECEIPT_SCHEMA = "hrime_stage1_execution_receipt_v1"
ORACLE_RECEIPT_SCHEMA = "hrime_stage1_oracle_receipt_v1"
REPLAY_SCHEMA = "duca_rime_budget_replay_v1"
ORACLE_PLANNER_VERSION = "hrime_stage1_same_total_oracle_v1"
DEFAULT_CANDIDATE_BUDGETS = (192, 256, 384, 512)
STAGE1_ALLOCATION_CONTRACT = {
    "scope": "one_video_across_ordered_768_frame_windows",
    "budget_unit": "heavy_rgb_frames",
    "same_total_reference": (
        "sum_of_anchor_nominal_budget_projected_to_each_window_effective_k"
    ),
    "short_window_projection": (
        "min_nominal_and_valid_prefix_floor_to_execution_quantum"
    ),
    "execution_quantum": HRIME_EXECUTION_QUANTUM,
    "solver_version": _HRIME_CORE.HRIME_SOLVER_VERSION,
    "score_dtype": _HRIME_CORE.HRIME_SCORE_DTYPE,
    "score_scale": _HRIME_CORE.HRIME_SCORE_SCALE,
    "score_rounding": _HRIME_CORE.HRIME_SCORE_ROUNDING,
    "tie_break": (
        "max_objective_then_min_risk_then_lexicographically_smallest_effective_k"
    ),
    "solver_unused_budget_required": 0,
    "tail_padding_mode": "none_exact_k_bucket",
}
STAGE1_EVALUATION_CONTRACT = {
    "pipeline": "full_detector_window_merge_nms",
    "primary_evaluator": "opentad_official_map",
    "paired_resampling_unit": "video",
    "official_final_subset_consumed": False,
    "raw_prediction_cache_used_for_decision": False,
    "gt_scope": "certification_development_oracle_and_evaluation_only",
}
SUPPORTED_METRICS = {
    "avg_map",
    "map_0.6",
    "map_0.7",
    "short_map",
    "medium_map",
    "long_map",
    "pair_support",
    "boundary_error",
}
STRATEGY_CONTRACTS = {
    "uniform_same_total": {
        "role": "hrime_stage1_uniform_same_total",
        "position_policy": "exact_uniform",
        "uses_gt": False,
        "decision_uses_gt": False,
        "surrogate_evaluation_uses_gt": True,
    },
    "independent_exact_total": {
        "role": "hrime_stage1_independent_exact_total",
        "position_policy": "frozen_rime_selector",
        "uses_gt": False,
        "decision_uses_gt": False,
        "surrogate_evaluation_uses_gt": False,
    },
    "joint_oracle": {
        "role": "hrime_stage1_joint_oracle",
        "position_policy": "frozen_rime_selector",
        "uses_gt": True,
        "decision_uses_gt": True,
        "surrogate_evaluation_uses_gt": True,
    },
    "joint_same_k_uniform_positions": {
        "role": "hrime_stage1_joint_same_k_uniform_positions",
        "position_policy": "exact_uniform",
        "uses_gt": True,
        "decision_uses_gt": True,
        "surrogate_evaluation_uses_gt": True,
    },
    "shuffled_null": {
        "role": "hrime_stage1_shuffled_null",
        "position_policy": "frozen_rime_selector",
        "uses_gt": True,
        "decision_uses_gt": True,
        "surrogate_evaluation_uses_gt": True,
    },
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    _require(_SHA256.fullmatch(normalized) is not None, f"{label} must be SHA-256")
    return normalized


def _canonical_without(payload: Mapping[str, Any], field: str) -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != field})


def _read_json(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"JSON artifact is missing: {resolved}")
    if expected_sha256 is not None:
        _require(
            _sha256_file(resolved) == _require_sha256(expected_sha256, "artifact hash"),
            f"JSON artifact SHA-256 drift: {resolved}",
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON artifact must be an object: {resolved}")
    return payload


def _read_jsonl(path: str | Path, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"JSONL artifact is missing: {resolved}")
    if expected_sha256 is not None:
        _require(
            _sha256_file(resolved) == _require_sha256(expected_sha256, "artifact hash"),
            f"JSONL artifact SHA-256 drift: {resolved}",
        )
    rows = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    _require(rows and all(isinstance(row, dict) for row in rows), "JSONL is empty or invalid")
    return rows


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    _require(not target.exists(), f"refusing to overwrite oracle artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target), "sha256": _sha256_file(target)}


def _atomic_write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    _require(not target.exists(), f"refusing to overwrite oracle replay: {target}")
    _require(rows, "oracle replay must be nonempty")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "record_count": len(rows),
    }


def validate_preregistration(
    payload: Mapping[str, Any],
    *,
    expected_git_commit: str | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_split_assignment_sha256: str | None = None,
) -> dict[str, Any]:
    _require(payload.get("schema_version") == PREREGISTRATION_SCHEMA, "wrong preregistration schema")
    _require(payload.get("status") == "frozen", "preregistration must be frozen")
    _require(
        payload.get("task") == "offline_temporal_action_detection",
        "preregistration task drift",
    )
    commit = str(payload.get("git_commit", "")).lower()
    _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid preregistration commit")
    if expected_git_commit is not None:
        _require(commit == str(expected_git_commit).lower(), "preregistration commit drift")
    split_sha = _require_sha256(
        payload.get("split_manifest_sha256"),
        "preregistration split-manifest hash",
    )
    assignment_sha = _require_sha256(
        payload.get("split_assignment_sha256"),
        "preregistration split-assignment hash",
    )
    if expected_split_manifest_sha256 is not None:
        _require(split_sha == str(expected_split_manifest_sha256).lower(), "split hash drift")
    if expected_split_assignment_sha256 is not None:
        _require(
            assignment_sha == str(expected_split_assignment_sha256).lower(),
            "split assignment drift",
        )
    role = str(payload.get("development_role", ""))
    _require(role == "certification_development", "Stage-1 role must be certification development")
    _require(payload.get("uses_official_final") is False, "official-final must remain sealed")
    _require(
        payload.get("official_final_used_for_selection") is False,
        "official-final cannot select the oracle route",
    )
    budgets = tuple(int(value) for value in payload.get("candidate_budgets", ()))
    anchors = tuple(int(value) for value in payload.get("anchor_nominal_budgets", ()))
    _require(
        budgets == DEFAULT_CANDIDATE_BUDGETS
        and anchors
        and tuple(sorted(set(anchors))) == anchors
        and set(anchors).issubset(budgets),
        "candidate or anchor budget protocol drift",
    )
    _require(
        payload.get("allocation_contract") == STAGE1_ALLOCATION_CONTRACT,
        "Stage-1 allocation/numeric contract drift",
    )
    _require(
        payload.get("evaluation_contract") == STAGE1_EVALUATION_CONTRACT,
        "Stage-1 full-detector evaluation contract drift",
    )
    _require(
        payload.get("strategy_contract_sha256")
        == canonical_sha256(STRATEGY_CONTRACTS),
        "Stage-1 strategy matrix contract drift",
    )
    oracle_risk_weight = float(payload.get("oracle_risk_weight", math.nan))
    _require(
        math.isfinite(oracle_risk_weight) and oracle_risk_weight >= 0.0,
        "oracle risk weight is invalid",
    )
    primary = payload.get("primary_endpoint")
    _require(isinstance(primary, Mapping), "one primary endpoint is required")
    _require(primary.get("metric") in SUPPORTED_METRICS, "unsupported primary endpoint")
    _require(primary.get("direction") in {"higher", "lower"}, "invalid endpoint direction")
    for name in (
        "min_mean_delta",
        "min_lcb_delta",
        "noninferiority_margin",
    ):
        value = float(primary.get(name, math.nan))
        _require(math.isfinite(value), f"primary endpoint {name} must be finite")
    alpha = float(primary.get("alpha", math.nan))
    _require(0.0 < alpha < 0.5, "primary endpoint alpha is invalid")
    bootstrap = payload.get("bootstrap")
    _require(
        isinstance(bootstrap, Mapping)
        and bootstrap.get("unit") == "video"
        and int(bootstrap.get("samples", 0)) >= 100
        and int(bootstrap.get("seed", -1)) >= 0,
        "video bootstrap protocol is invalid",
    )
    multiplicity = payload.get("multiplicity")
    expected_primary_family = [
        f"k{anchor}:joint_oracle_vs_{comparator}:{primary['metric']}"
        for anchor in anchors
        for comparator in (
            "uniform_same_total",
            "independent_exact_total",
        )
    ]
    _require(
        isinstance(multiplicity, Mapping)
        and multiplicity.get("method")
        == "intersection_union_single_primary_with_guardrails"
        and multiplicity.get("family") == expected_primary_family,
        "multiplicity protocol is invalid",
    )
    guardrails = payload.get("guardrails")
    _require(isinstance(guardrails, list), "guardrails must be a list")
    for index, guardrail in enumerate(guardrails):
        _require(isinstance(guardrail, Mapping), f"guardrail {index} is invalid")
        _require(guardrail.get("metric") in SUPPORTED_METRICS, f"guardrail {index} metric is invalid")
        _require(
            guardrail.get("direction") in {"higher", "lower"},
            f"guardrail {index} direction is invalid",
        )
        _require(
            str(guardrail.get("comparator", ""))
            in {"uniform_same_total", "independent_exact_total"},
            f"guardrail {index} comparator is invalid",
        )
        _require(
            math.isfinite(float(guardrail.get("min_delta", math.nan))),
            f"guardrail {index} threshold is invalid",
        )
    surrogate = payload.get("surrogate_audit")
    _require(isinstance(surrogate, Mapping), "surrogate audit must be registered")
    min_spearman = float(surrogate.get("min_spearman", math.nan))
    min_sign = float(surrogate.get("min_sign_agreement", math.nan))
    max_rank_error = float(surrogate.get("max_worst_rank_error", math.nan))
    _require(-1.0 <= min_spearman <= 1.0, "surrogate Spearman threshold is invalid")
    _require(0.0 <= min_sign <= 1.0, "surrogate sign threshold is invalid")
    _require(0.0 <= max_rank_error <= 1.0, "surrogate rank-error threshold is invalid")
    _require(
        surrogate.get("error_normalization") == "fractional_midrank",
        "surrogate error normalization drift",
    )
    content_sha = _require_sha256(payload.get("content_sha256"), "preregistration content hash")
    _require(
        content_sha == _canonical_without(payload, "content_sha256"),
        "preregistration content hash drift",
    )
    return {
        "git_commit": commit,
        "split_manifest_sha256": split_sha,
        "split_assignment_sha256": assignment_sha,
        "development_role": role,
        "candidate_budgets": budgets,
        "anchor_nominal_budgets": anchors,
        "allocation_contract": dict(STAGE1_ALLOCATION_CONTRACT),
        "evaluation_contract": dict(STAGE1_EVALUATION_CONTRACT),
        "strategy_contract_sha256": canonical_sha256(STRATEGY_CONTRACTS),
        "oracle_risk_weight": oracle_risk_weight,
        "primary_endpoint": dict(primary),
        "bootstrap": dict(bootstrap),
        "multiplicity": dict(multiplicity),
        "guardrails": [dict(value) for value in guardrails],
        "surrogate_audit": dict(surrogate),
        "content_sha256": content_sha,
    }


def validate_window_option_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_videos: Sequence[str],
    expected_split_role: str,
    expected_split_assignment_sha256: str,
    candidate_budgets: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    budgets = tuple(int(value) for value in candidate_budgets)
    expected_video_set = {str(value) for value in expected_videos}
    assignment_sha = _require_sha256(
        expected_split_assignment_sha256,
        "window-option split assignment hash",
    )
    _require(rows, "window-option measurements are empty")
    validated = []
    identities = set()
    source_hashes = set()
    for index, source in enumerate(rows):
        row = dict(source)
        _require(row.get("schema_version") == WINDOW_OPTION_SCHEMA, f"row {index} schema drift")
        _require(row.get("status") == "measured", f"row {index} is not measured")
        video = str(row.get("video_id", ""))
        start = int(row.get("window_start_frame", -1))
        valid_length = int(row.get("valid_length", 0))
        identity = (video, start)
        _require(
            video in expected_video_set and start >= 0 and valid_length > 0,
            f"row {index} identity is invalid",
        )
        _require(identity not in identities, f"duplicate window option row {identity}")
        identities.add(identity)
        _require(row.get("split_role") == expected_split_role, f"row {index} role drift")
        _require(
            str(row.get("split_assignment_sha256", "")).lower() == assignment_sha,
            f"row {index} split assignment drift",
        )
        _require(
            tuple(int(value) for value in row.get("candidate_budgets", ())) == budgets,
            f"row {index} candidate budgets drift",
        )
        feasible = canonicalize_effective_k_options(valid_length, budgets)
        options = row.get("options")
        _require(isinstance(options, list), f"row {index} options are invalid")
        _require(
            [int(option.get("effective_k", -1)) for option in options]
            == list(feasible.effective_ks),
            f"row {index} effective-K options drift",
        )
        for option, choice in zip(options, feasible.choices):
            _require(
                tuple(int(value) for value in option.get("nominal_budgets", ()))
                == choice.nominal_budgets,
                f"row {index} nominal aliases drift",
            )
            for name in (
                "predicted_utility",
                "predicted_risk",
                "oracle_utility",
                "oracle_risk",
            ):
                value = float(option.get(name, math.nan))
                _require(math.isfinite(value), f"row {index} {name} is non-finite")
                if name.endswith("risk"):
                    _require(value >= 0.0, f"row {index} {name} must be non-negative")
        provenance = row.get("provenance")
        _require(isinstance(provenance, Mapping), f"row {index} provenance is invalid")
        _require(
            provenance.get("uses_official_final") is False
            and provenance.get("uses_gt_for_oracle_utility") is True
            and provenance.get("uses_gt_for_predicted_utility") is False
            and provenance.get("uses_teacher") is False
            and provenance.get("uses_prediction_cache") is False
            and provenance.get("oracle_only") is True
            and provenance.get("deployment_candidate") is False,
            f"row {index} provenance is contaminated or ambiguous",
        )
        source_sha = _require_sha256(
            row.get("source_identity_sha256"),
            f"row {index} source identity hash",
        )
        source_hashes.add(source_sha)
        record_sha = _require_sha256(row.get("record_sha256"), f"row {index} record hash")
        _require(
            record_sha == _canonical_without(row, "record_sha256"),
            f"row {index} record hash drift",
        )
        validated.append(row)
    _require(
        {str(row["video_id"]) for row in validated} == expected_video_set,
        "window-option measurements do not cover the exact development videos",
    )
    _require(len(source_hashes) == 1, "window-option rows mix source identities")
    return tuple(
        sorted(
            validated,
            key=lambda row: (str(row["video_id"]), int(row["window_start_frame"])),
        )
    )


def _mckp_windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    utility_name: str,
    risk_name: str,
    context_sha256: str,
) -> tuple[MCKPWindow, ...]:
    output = []
    for row in rows:
        options = tuple(
            MCKPOption(
                effective_k=int(option["effective_k"]),
                utility=option[utility_name],
                risk=option[risk_name],
                nominal_budgets=tuple(int(value) for value in option["nominal_budgets"]),
            )
            for option in row["options"]
        )
        output.append(
            MCKPWindow(
                window_key=canonical_sha256(
                    {
                        "schema_version": "hrime_window_identity_v1",
                        "video_id": str(row["video_id"]),
                        "window_start_frame": int(row["window_start_frame"]),
                    }
                ),
                options=options,
                option_source_sha256=canonical_sha256(
                    {
                        "context_sha256": context_sha256,
                        "record_sha256": row["record_sha256"],
                        "utility_name": utility_name,
                        "risk_name": risk_name,
                    }
                ),
            )
        )
    return tuple(output)


def _uniform_windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    anchor_nominal_budget: int,
    context_sha256: str,
) -> tuple[MCKPWindow, ...]:
    output = []
    for row in rows:
        feasible = canonicalize_effective_k_options(
            int(row["valid_length"]),
            tuple(int(value) for value in row["candidate_budgets"]),
        )
        anchor_effective = feasible.nominal_to_effective[int(anchor_nominal_budget)]
        options = tuple(
            MCKPOption(
                effective_k=choice.effective_k,
                utility=-(choice.effective_k - anchor_effective) ** 2,
                risk=0,
                nominal_budgets=choice.nominal_budgets,
            )
            for choice in feasible.choices
        )
        output.append(
            MCKPWindow(
                window_key=canonical_sha256(
                    {
                        "schema_version": "hrime_window_identity_v1",
                        "video_id": str(row["video_id"]),
                        "window_start_frame": int(row["window_start_frame"]),
                    }
                ),
                options=options,
                option_source_sha256=canonical_sha256(
                    {
                        "context_sha256": context_sha256,
                        "record_sha256": row["record_sha256"],
                        "anchor_nominal_budget": int(anchor_nominal_budget),
                        "objective": "negative_squared_anchor_deviation",
                    }
                ),
            )
        )
    return tuple(output)


def _assignment_surrogate(
    rows: Sequence[Mapping[str, Any]],
    assignment: Sequence[int],
    *,
    utility_name: str,
    risk_name: str,
    beta: float,
) -> float:
    total = 0.0
    for row, effective_k in zip(rows, assignment):
        option = next(
            value
            for value in row["options"]
            if int(value["effective_k"]) == int(effective_k)
        )
        total += float(option[utility_name]) - float(beta) * float(option[risk_name])
    return total


def _feasibility_preserving_shuffle(
    assignment: Sequence[int],
    feasible_effective_ks: Sequence[Sequence[int]],
    *,
    seed: int,
) -> tuple[tuple[int, ...], bool]:
    shuffled = list(int(value) for value in assignment)
    indices = list(range(len(shuffled)))
    random.Random(int(seed)).shuffle(indices)
    for left_offset, left in enumerate(indices):
        for right in indices[left_offset + 1 :]:
            if (
                shuffled[left] != shuffled[right]
                and shuffled[right] in set(int(value) for value in feasible_effective_ks[left])
                and shuffled[left] in set(int(value) for value in feasible_effective_ks[right])
            ):
                shuffled[left], shuffled[right] = shuffled[right], shuffled[left]
                break
    _require(sorted(shuffled) == sorted(int(value) for value in assignment), "shuffle changed K histogram")
    _require(
        all(value in set(int(candidate) for candidate in feasible) for value, feasible in zip(shuffled, feasible_effective_ks)),
        "shuffle produced an infeasible assignment",
    )
    original = tuple(int(value) for value in assignment)
    return tuple(shuffled), tuple(shuffled) == original


def _assignment_sha(
    *,
    video_id: str,
    anchor_nominal_budget: int,
    strategy: str,
    window_keys: Sequence[str],
    assignment: Sequence[int],
    context_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "hrime_stage1_assignment_v1",
            "planner_version": ORACLE_PLANNER_VERSION,
            "video_id": video_id,
            "anchor_nominal_budget": int(anchor_nominal_budget),
            "strategy": strategy,
            "window_keys": list(window_keys),
            "assignment": [int(value) for value in assignment],
            "context_sha256": context_sha256,
        }
    )


def _replay_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy: str,
    anchor_nominal_budget: int,
    target_total: int,
    assignment: Sequence[int],
    assignment_sha256: str,
    plan_input_sha256: str,
    preregistration_sha256: str,
    budget_protocol_sha256: str,
) -> tuple[dict[str, Any], ...]:
    contract = STRATEGY_CONTRACTS[strategy]
    output = []
    for row, effective_k in zip(rows, assignment):
        feasible = canonicalize_effective_k_options(
            int(row["valid_length"]),
            tuple(int(value) for value in row["candidate_budgets"]),
        )
        choice = feasible.choice_for_effective_k(int(effective_k))
        output.append(
            {
                "schema_version": REPLAY_SCHEMA,
                "video_id": str(row["video_id"]),
                "window_start_frame": int(row["window_start_frame"]),
                "requested_k": int(choice.canonical_nominal_budget),
                "effective_k": int(effective_k),
                "provenance": {
                    "role": contract["role"],
                    "strategy": strategy,
                    "oracle_only": True,
                    "deployment_candidate": False,
                    "uses_official_final": False,
                    "uses_gt": bool(contract["uses_gt"]),
                    "uses_teacher": False,
                    "uses_prediction_cache": False,
                    "uses_test_batch_composition": False,
                    "position_policy": contract["position_policy"],
                    "anchor_nominal_budget": int(anchor_nominal_budget),
                    "reachable_video_budget": int(target_total),
                    "realized_video_budget": int(target_total),
                    "solver_unused_budget": 0,
                    "assignment_sha256": assignment_sha256,
                    "plan_input_sha256": plan_input_sha256,
                    "preregistration_sha256": preregistration_sha256,
                    "budget_protocol_sha256": budget_protocol_sha256,
                    "canonical_nominal_aliases": list(choice.nominal_budgets),
                },
            }
        )
    _require(sum(int(row["effective_k"]) for row in output) == int(target_total), "replay cost drift")
    return tuple(output)


def build_stage1_plan(
    *,
    rows: Sequence[Mapping[str, Any]],
    window_options_sha256: str,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    budget_protocol_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    options_artifact_sha = _require_sha256(
        window_options_sha256,
        "window-options artifact hash",
    )
    prereg_sha = _require_sha256(preregistration_sha256, "preregistration file hash")
    protocol_sha = _require_sha256(budget_protocol_sha256, "budget protocol hash")
    root = Path(output_root).expanduser().resolve()
    _require(not root.exists(), f"fresh Stage-1 plan root is required: {root}")
    validated_prereg = validate_preregistration(preregistration)
    budgets = validated_prereg["candidate_budgets"]
    ordered_rows = tuple(
        sorted(rows, key=lambda row: (str(row["video_id"]), int(row["window_start_frame"])))
    )
    refs = tuple(
        VideoWindowRef(
            video_id=str(row["video_id"]),
            window_start_frame=int(row["window_start_frame"]),
            valid_length=int(row["valid_length"]),
            source_index=index,
            cheap_feature_index=index,
        )
        for index, row in enumerate(ordered_rows)
    )
    groups = group_video_windows(refs)
    row_by_identity = {
        (str(row["video_id"]), int(row["window_start_frame"])): row for row in ordered_rows
    }
    replay_rows_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    video_plans = []
    beta = float(validated_prereg["oracle_risk_weight"])
    _require(math.isfinite(beta) and beta >= 0.0, "oracle risk weight is invalid")
    for group in groups:
        group_rows = tuple(
            row_by_identity[(window.video_id, window.window_start_frame)]
            for window in group.windows
        )
        feasible_sets = tuple(
            canonicalize_effective_k_options(int(row["valid_length"]), budgets)
            for row in group_rows
        )
        for anchor in validated_prereg["anchor_nominal_budgets"]:
            target = sum(feasible.nominal_to_effective[int(anchor)] for feasible in feasible_sets)
            context = canonical_sha256(
                {
                    "schema_version": "hrime_stage1_plan_input_v1",
                    "planner_version": ORACLE_PLANNER_VERSION,
                    "video_id": group.video_id,
                    "group_order_sha256": group.group_order_sha256,
                    "anchor_nominal_budget": int(anchor),
                    "target_total_effective_k": int(target),
                    "preregistration_sha256": prereg_sha,
                    "budget_protocol_sha256": protocol_sha,
                    "window_record_sha256": [row["record_sha256"] for row in group_rows],
                }
            )
            uniform = solve_exact_mckp(
                _uniform_windows(
                    group_rows,
                    anchor_nominal_budget=int(anchor),
                    context_sha256=context,
                ),
                target_total_cost=target,
                beta=0,
                allocation_context_sha256=context,
            )
            expected_uniform = tuple(
                feasible.nominal_to_effective[int(anchor)] for feasible in feasible_sets
            )
            _require(
                uniform.assignment == expected_uniform,
                "variance-minimizing uniform assignment differs from the anchor mapping",
            )
            independent = solve_exact_mckp(
                _mckp_windows(
                    group_rows,
                    utility_name="predicted_utility",
                    risk_name="predicted_risk",
                    context_sha256=context,
                ),
                target_total_cost=target,
                beta=beta,
                allocation_context_sha256=context,
            )
            joint = solve_exact_mckp(
                _mckp_windows(
                    group_rows,
                    utility_name="oracle_utility",
                    risk_name="oracle_risk",
                    context_sha256=context,
                ),
                target_total_cost=target,
                beta=beta,
                allocation_context_sha256=context,
            )
            shuffled, degenerate_null = _feasibility_preserving_shuffle(
                joint.assignment,
                tuple(feasible.effective_ks for feasible in feasible_sets),
                seed=int(
                    canonical_sha256(
                        {
                            "video_id": group.video_id,
                            "anchor": int(anchor),
                            "preregistration_sha256": prereg_sha,
                        }
                    )[:16],
                    16,
                ),
            )
            assignments = {
                "uniform_same_total": uniform.assignment,
                "independent_exact_total": independent.assignment,
                "joint_oracle": joint.assignment,
                "joint_same_k_uniform_positions": joint.assignment,
                "shuffled_null": shuffled,
            }
            receipts = {
                "uniform_same_total": uniform.to_receipt(),
                "independent_exact_total": independent.to_receipt(),
                "joint_oracle": joint.to_receipt(),
            }
            strategies = {}
            for strategy, assignment in assignments.items():
                assignment_sha = _assignment_sha(
                    video_id=group.video_id,
                    anchor_nominal_budget=int(anchor),
                    strategy=strategy,
                    window_keys=tuple(window.window_key for window in group.windows),
                    assignment=assignment,
                    context_sha256=context,
                )
                replay_rows = _replay_rows(
                    group_rows,
                    strategy=strategy,
                    anchor_nominal_budget=int(anchor),
                    target_total=target,
                    assignment=assignment,
                    assignment_sha256=assignment_sha,
                    plan_input_sha256=context,
                    preregistration_sha256=prereg_sha,
                    budget_protocol_sha256=protocol_sha,
                )
                replay_rows_by_cell[(strategy, int(anchor))].extend(replay_rows)
                utility_name = (
                    "oracle_utility"
                    if STRATEGY_CONTRACTS[strategy]["uses_gt"]
                    else (
                        "predicted_utility"
                        if strategy == "independent_exact_total"
                        else "oracle_utility"
                    )
                )
                risk_name = (
                    "oracle_risk"
                    if STRATEGY_CONTRACTS[strategy]["uses_gt"]
                    else (
                        "predicted_risk"
                        if strategy == "independent_exact_total"
                        else "oracle_risk"
                    )
                )
                strategies[strategy] = {
                    "assignment": list(assignment),
                    "assignment_sha256": assignment_sha,
                    "target_total_effective_k": int(target),
                    "realized_total_effective_k": int(sum(assignment)),
                    "position_policy": STRATEGY_CONTRACTS[strategy]["position_policy"],
                    "uses_gt": STRATEGY_CONTRACTS[strategy]["uses_gt"],
                    "decision_uses_gt": STRATEGY_CONTRACTS[strategy][
                        "decision_uses_gt"
                    ],
                    "surrogate_evaluation_uses_gt": STRATEGY_CONTRACTS[strategy][
                        "surrogate_evaluation_uses_gt"
                    ],
                    "surrogate_objective": _assignment_surrogate(
                        group_rows,
                        assignment,
                        utility_name=utility_name,
                        risk_name=risk_name,
                        beta=beta,
                    ),
                    "solver_receipt": receipts.get(strategy),
                    "null_degenerate": bool(
                        strategy == "shuffled_null" and degenerate_null
                    ),
                }
            video_plans.append(
                {
                    "video_id": group.video_id,
                    "group_order_sha256": group.group_order_sha256,
                    "window_keys": [window.window_key for window in group.windows],
                    "window_starts": [window.window_start_frame for window in group.windows],
                    "valid_lengths": [window.valid_length for window in group.windows],
                    "anchor_nominal_budget": int(anchor),
                    "target_total_effective_k": int(target),
                    "plan_input_sha256": context,
                    "strategies": strategies,
                }
            )
    shuffled_null_diagnostics = {}
    for anchor in validated_prereg["anchor_nominal_budgets"]:
        anchor_plans = [
            value
            for value in video_plans
            if int(value["anchor_nominal_budget"]) == int(anchor)
        ]
        degenerate_videos = sorted(
            str(value["video_id"])
            for value in anchor_plans
            if value["strategies"]["shuffled_null"]["null_degenerate"]
        )
        nondegenerate_count = len(anchor_plans) - len(degenerate_videos)
        _require(
            nondegenerate_count > 0,
            f"shuffled-null cell is degenerate for every video at anchor {anchor}",
        )
        shuffled_null_diagnostics[str(int(anchor))] = {
            "video_count": len(anchor_plans),
            "nondegenerate_video_count": nondegenerate_count,
            "degenerate_video_count": len(degenerate_videos),
            "degenerate_video_ids": degenerate_videos,
            "cell_non_degenerate": True,
        }
    root.mkdir(parents=True, exist_ok=False)
    replay_artifacts = {}
    for strategy in STRATEGY_CONTRACTS:
        replay_artifacts[strategy] = {}
        for anchor in validated_prereg["anchor_nominal_budgets"]:
            cell_rows = sorted(
                replay_rows_by_cell[(strategy, int(anchor))],
                key=lambda row: (str(row["video_id"]), int(row["window_start_frame"])),
            )
            artifact = _atomic_write_jsonl(
                root / "replays" / strategy / f"k{int(anchor)}.jsonl",
                cell_rows,
            )
            replay_artifacts[strategy][str(int(anchor))] = artifact
    manifest = {
        "schema_version": PLAN_SCHEMA,
        "status": "planned",
        "planner_version": ORACLE_PLANNER_VERSION,
        "git_commit": validated_prereg["git_commit"],
        "task": "offline_temporal_action_detection",
        "development_role": validated_prereg["development_role"],
        "uses_official_final": False,
        "window_options_sha256": options_artifact_sha,
        "window_options_source_identity_sha256": ordered_rows[0][
            "source_identity_sha256"
        ],
        "window_options_record_set_sha256": canonical_sha256(
            [row["record_sha256"] for row in ordered_rows]
        ),
        "preregistration_sha256": prereg_sha,
        "split_manifest_sha256": validated_prereg["split_manifest_sha256"],
        "split_assignment_sha256": validated_prereg["split_assignment_sha256"],
        "budget_protocol_sha256": protocol_sha,
        "candidate_budgets": list(budgets),
        "anchor_nominal_budgets": list(validated_prereg["anchor_nominal_budgets"]),
        "oracle_risk_weight": beta,
        "allocation_contract": validated_prereg["allocation_contract"],
        "evaluation_contract": validated_prereg["evaluation_contract"],
        "strategy_contract_sha256": validated_prereg[
            "strategy_contract_sha256"
        ],
        "strategies": {
            name: dict(contract) for name, contract in STRATEGY_CONTRACTS.items()
        },
        "shuffled_null_diagnostics": shuffled_null_diagnostics,
        "replay_artifacts": replay_artifacts,
        "video_plans": video_plans,
        "video_count": len(groups),
        "window_count": len(ordered_rows),
        "authorizes_stage2_training": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    artifact = _atomic_write_json(root / "plan_manifest.json", manifest)
    return {**artifact, "payload": manifest}


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(ordered, "percentile input is empty")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(quantile)
    low, high = int(math.floor(rank)), int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def paired_video_bootstrap(
    values: Mapping[str, float],
    *,
    samples: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    videos = sorted(str(video) for video in values)
    _require(videos, "paired bootstrap requires videos")
    vector = [float(values[video]) for video in videos]
    _require(all(math.isfinite(value) for value in vector), "bootstrap values are non-finite")
    rng = random.Random(int(seed))
    draws = [
        mean(rng.choice(vector) for _ in vector) for _ in range(max(1, int(samples)))
    ]
    return {
        "mean": mean(vector),
        "ci_low": _percentile(draws, float(alpha) / 2.0),
        "ci_high": _percentile(draws, 1.0 - float(alpha) / 2.0),
        "video_count": len(videos),
        "bootstrap_samples": max(1, int(samples)),
        "bootstrap_unit": "video",
        "seed": int(seed),
    }


def _fractional_midranks(values: Sequence[float]) -> list[float]:
    _require(values, "midrank input is empty")
    ordered = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    result = [0.0] * len(values)
    cursor = 0
    denominator = max(1, len(values) - 1)
    while cursor < len(ordered):
        stop = cursor + 1
        while stop < len(ordered) and float(values[ordered[stop]]) == float(
            values[ordered[cursor]]
        ):
            stop += 1
        midrank = 0.5 * (cursor + stop - 1) / denominator
        for offset in range(cursor, stop):
            result[ordered[offset]] = midrank
        cursor = stop
    return result


def surrogate_audit(
    surrogate_delta: Sequence[float],
    official_delta: Sequence[float],
) -> dict[str, Any]:
    _require(
        len(surrogate_delta) == len(official_delta) and len(surrogate_delta) >= 2,
        "surrogate audit vectors must align and contain at least two cells",
    )
    surrogate = [float(value) for value in surrogate_delta]
    official = [float(value) for value in official_delta]
    _require(
        all(math.isfinite(value) for value in (*surrogate, *official)),
        "surrogate audit vectors are non-finite",
    )
    surrogate_rank = _fractional_midranks(surrogate)
    official_rank = _fractional_midranks(official)
    def sign(value: float) -> int:
        return 1 if value > 0.0 else (-1 if value < 0.0 else 0)

    sign_agreement = mean(
        float(sign(left) == sign(right))
        for left, right in zip(surrogate, official)
    )
    return {
        "spearman": spearman(surrogate, official),
        "sign_agreement": sign_agreement,
        "worst_rank_error": max(
            abs(left - right) for left, right in zip(surrogate_rank, official_rank)
        ),
        "error_normalization": "fractional_midrank",
        "cell_count": len(surrogate),
    }


def validate_stage1_execution(
    *,
    plan_manifest: str | Path,
    plan_manifest_sha256: str,
    strategy: str,
    anchor_nominal_budget: int,
    replay_jsonl: str | Path,
    replay_sha256: str,
    inference_ledger_jsonl: str | Path,
    terminal_evaluation: str | Path,
    localization_metrics: str | Path,
    output_receipt: str | Path,
) -> dict[str, Any]:
    plan_path = Path(plan_manifest).expanduser().resolve()
    plan = _read_json(plan_path, plan_manifest_sha256)
    _require(plan.get("schema_version") == PLAN_SCHEMA and plan.get("status") == "planned", "plan drift")
    _require(plan.get("uses_official_final") is False, "oracle plan opened official-final")
    _require(strategy in STRATEGY_CONTRACTS, "unknown Stage-1 strategy")
    anchor = int(anchor_nominal_budget)
    _require(anchor in plan.get("anchor_nominal_budgets", ()), "anchor is not registered")
    registered_replay = plan["replay_artifacts"][strategy][str(anchor)]
    replay_path = Path(replay_jsonl).expanduser().resolve()
    replay_rows = _read_jsonl(replay_path, replay_sha256)
    _require(
        Path(registered_replay["path"]).resolve() == replay_path
        and registered_replay["sha256"] == str(replay_sha256).lower()
        and int(registered_replay["record_count"]) == len(replay_rows),
        "execution replay differs from the plan",
    )
    replay_by_key = {
        (str(row["video_id"]), int(row["window_start_frame"])): row for row in replay_rows
    }
    _require(len(replay_by_key) == len(replay_rows), "execution replay contains duplicate windows")
    ledgers = _read_jsonl(inference_ledger_jsonl)
    ledger_by_key = {
        (str(row["video_id"]), int(row["window_start_frame"])): row for row in ledgers
    }
    _require(
        len(ledger_by_key) == len(ledgers) and set(ledger_by_key) == set(replay_by_key),
        "execution ledger window set differs from replay",
    )
    video_costs: dict[str, int] = defaultdict(int)
    for key, replay in replay_by_key.items():
        ledger = ledger_by_key[key]
        requested = int(replay["requested_k"])
        effective = int(replay["effective_k"])
        _require(
            int(ledger.get("requested_k", -1)) == requested
            and int(ledger.get("effective_k", -1)) == effective
            and int(ledger.get("unique_k", -1)) == effective
            and int(ledger.get("backbone_input_k", -1)) == effective
            and int(ledger.get("padded_k", -1)) == effective
            and int(ledger.get("raw_budget", -1)) == requested
            and int(ledger.get("reachable_budget", -1)) == effective
            and int(ledger.get("realized_budget", -1)) == effective
            and int(ledger.get("projection_unused_budget", -1))
            == requested - effective
            and int(ledger.get("solver_unused_budget", -1)) == 0
            and ledger.get("budget_scope")
            == "video_exact_total_window_assignment"
            and ledger.get("claim_scope")
            == "stage1_development_oracle_execution_not_deployable",
            f"execution ledger violates exact-K/no-padding for {key}",
        )
        replay_provenance = replay.get("provenance")
        ledger_provenance = ledger.get("decision_provenance")
        _require(
            isinstance(replay_provenance, Mapping)
            and isinstance(ledger_provenance, Mapping)
            and ledger_provenance.get("role") == replay_provenance.get("role")
            and ledger_provenance.get("assignment_sha256")
            == replay_provenance.get("assignment_sha256")
            and ledger_provenance.get("uses_gt") is replay_provenance.get("uses_gt"),
            f"execution decision provenance drift for {key}",
        )
        video_costs[key[0]] += effective
    expected_video_costs = {}
    for video_plan in plan["video_plans"]:
        if int(video_plan["anchor_nominal_budget"]) == anchor:
            expected_video_costs[str(video_plan["video_id"])] = int(
                video_plan["target_total_effective_k"]
            )
    _require(dict(video_costs) == expected_video_costs, "per-video exact total cost drift")
    terminal_path = Path(terminal_evaluation).expanduser().resolve()
    terminal = _read_json(terminal_path)
    stage1_terminal_contract = terminal.get("stage1_execution_contract")
    _require(
        terminal.get("task") == "offline_temporal_action_detection"
        and terminal.get("git_commit") == plan["git_commit"]
        and terminal.get("runtime_gt_input_to_selector") is False
        and terminal.get("padded_to_kmax") is False
        and terminal.get("evaluation_protocol")
        == "hrime_stage1_oracle_execution_v1"
        and terminal.get("oracle_only") is True
        and terminal.get("evaluation_only") is True
        and terminal.get("deployment_candidate") is False
        and terminal.get("uses_official_final") is False
        and isinstance(stage1_terminal_contract, Mapping)
        and stage1_terminal_contract.get("decision_role")
        == STRATEGY_CONTRACTS[strategy]["role"]
        and stage1_terminal_contract.get("position_policy")
        == STRATEGY_CONTRACTS[strategy]["position_policy"]
        and stage1_terminal_contract.get("uses_gt_at_decision")
        is STRATEGY_CONTRACTS[strategy]["uses_gt"],
        "terminal evaluation contract drift",
    )
    checkpoint_compatibility = terminal.get("checkpoint_compatibility")
    post_processing_execution = terminal.get("post_processing_execution")
    expected_window_counts: dict[str, int] = defaultdict(int)
    for video_id, _ in replay_by_key:
        expected_window_counts[video_id] += 1
    expected_pipeline_events = [
        "model_forward_loop_complete",
        "ddp_result_gather_complete",
        "cross_window_result_aggregation_complete",
        "sliding_window_nms_complete",
        "post_nms_prediction_saved",
        "official_evaluator_evaluate_called",
        "official_evaluator_evaluate_returned",
    ]
    _require(
        checkpoint_compatibility
        == {
            "mode": "strict_exact_v1",
            "missing_keys": [],
            "ignored_unexpected_keys": [],
        },
        "Stage-1 checkpoint was not loaded with strict exact compatibility",
    )
    _require(
        isinstance(post_processing_execution, Mapping)
        and post_processing_execution.get("schema_version")
        == "opentad_window_merge_nms_execution_v1"
        and post_processing_execution.get("world_size") == 1
        and post_processing_execution.get("dataset_class")
        == "opentad.datasets.thumos.ThumosSlidingDataset"
        and post_processing_execution.get("dataset_is_sliding_window") is True
        and int(post_processing_execution.get("dataset_length", -1))
        == len(replay_rows)
        and int(post_processing_execution.get("input_sample_count", -1))
        == len(replay_rows)
        and int(post_processing_execution.get("window_metadata_count", -1))
        == len(replay_rows)
        and post_processing_execution.get("window_counts")
        == dict(sorted(expected_window_counts.items()))
        and int(post_processing_execution.get("model_forward_batch_count", 0)) > 0
        and post_processing_execution.get(
            "cross_window_result_aggregation_executed"
        )
        is True
        and post_processing_execution.get("nms_applied") is True
        and int(post_processing_execution.get("nms_call_count", -1))
        == len(expected_video_costs)
        and int(post_processing_execution.get("pre_nms_video_count", -1))
        == len(expected_video_costs)
        and int(post_processing_execution.get("post_nms_video_count", -1))
        == len(expected_video_costs)
        and set(
            post_processing_execution.get(
                "pre_nms_result_count_by_video", {}
            )
        )
        == set(expected_video_costs)
        and set(
            post_processing_execution.get(
                "post_nms_result_count_by_video", {}
            )
        )
        == set(expected_video_costs)
        and post_processing_execution.get("pipeline_events")
        == expected_pipeline_events
        and post_processing_execution.get("result_saved_after_nms") is True
        and post_processing_execution.get("evaluator_prediction_source")
        == "in_memory_post_nms_result_object"
        and post_processing_execution.get("evaluator_evaluate_called") is True
        and post_processing_execution.get("evaluator_evaluate_succeeded") is True
        and post_processing_execution.get(
            "full_detector_window_merge_nms_evaluation_completed"
        )
        is True
        and post_processing_execution.get("evaluator")
        == terminal.get("evaluator")
        and post_processing_execution.get("content_sha256")
        == _canonical_without(post_processing_execution, "content_sha256")
        and terminal.get("post_processing_execution_sha256")
        == post_processing_execution.get("content_sha256"),
        "terminal does not prove complete detector/window-merge/NMS/evaluator execution",
    )
    prediction_path = Path(str(terminal.get("prediction_path", ""))).resolve()
    _require(
        prediction_path.is_file()
        and _sha256_file(prediction_path) == terminal.get("prediction_sha256"),
        "terminal prediction drift",
    )
    _require(
        Path(str(post_processing_execution.get("result_path", ""))).resolve()
        == prediction_path
        and post_processing_execution.get("result_sha256")
        == terminal.get("prediction_sha256")
        and int(post_processing_execution.get("post_nms_result_count", -1))
        == int(terminal.get("result_count", -2)),
        "saved prediction is not the proved post-NMS result",
    )
    metrics_path = Path(localization_metrics).expanduser().resolve()
    metrics = _read_json(metrics_path)
    _require(
        metrics.get("schema_version") == "duca_rime_localization_metrics_v1"
        and int(metrics.get("phase", -1)) == 3
        and metrics.get("split_role") == plan["development_role"]
        and metrics.get("uses_official_final") is False
        and metrics.get("official_evaluator_used_for_map_metrics") is True
        and metrics.get("terminal_evaluation_sha256") == _sha256_file(terminal_path),
        "localization metrics contract drift",
    )
    _require(
        set(metrics.get("evaluation_video_ids", ())) == set(expected_video_costs),
        "localization metrics video set drift",
    )
    config_path = Path(str(terminal.get("config_path", ""))).resolve()
    _require(config_path.is_file(), "terminal config is missing")
    try:
        from mmengine.config import Config

        cfg = Config.fromfile(str(config_path))
        nms_contract = cfg.get("post_processing", {})
    except Exception as exc:
        raise ValueError("failed to resolve the NMS execution contract") from exc
    pipeline_identity = post_processing_execution.get("pipeline_identity")
    _require(
        cfg.inference.load_from_raw_predictions is False
        and cfg.dataset.test.type == "ThumosSlidingDataset"
        and cfg.post_processing.save_dict is True
        and cfg.post_processing.nms is not None
        and post_processing_execution.get("nms_config")
        == json.loads(json.dumps(cfg.post_processing.nms))
        and post_processing_execution.get("nms_config_sha256")
        == canonical_sha256(cfg.post_processing.nms)
        and post_processing_execution.get("evaluation_config_sha256")
        == canonical_sha256(cfg.evaluation)
        and isinstance(pipeline_identity, Mapping)
        and post_processing_execution.get("pipeline_identity_sha256")
        == canonical_sha256(pipeline_identity),
        "resolved config differs from the proved merge/NMS/evaluator pipeline",
    )
    for identity_name in (
        "engine_source_identity",
        "gather_source_identity",
        "nms_callable_identity",
        "dataset_class_source_identity",
    ):
        identity = pipeline_identity.get(identity_name)
        _require(
            isinstance(identity, Mapping)
            and Path(str(identity.get("source_path", ""))).is_file()
            and _sha256_file(identity["source_path"])
            == _require_sha256(
                identity.get("source_sha256"),
                f"{identity_name} source hash",
            ),
            f"{identity_name} source identity drift",
        )
    payload = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        "status": "passed",
        "task": "offline_temporal_action_detection",
        "git_commit": plan["git_commit"],
        "strategy": strategy,
        "anchor_nominal_budget": anchor,
        "position_policy": STRATEGY_CONTRACTS[strategy]["position_policy"],
        "uses_gt_at_decision": STRATEGY_CONTRACTS[strategy]["uses_gt"],
        "oracle_only": True,
        "deployment_candidate": False,
        "uses_official_final": False,
        "plan_manifest_path": str(plan_path),
        "plan_manifest_sha256": _sha256_file(plan_path),
        "replay_path": str(replay_path),
        "replay_sha256": _sha256_file(replay_path),
        "inference_ledger_path": str(Path(inference_ledger_jsonl).resolve()),
        "inference_ledger_sha256": _sha256_file(inference_ledger_jsonl),
        "terminal_evaluation_path": str(terminal_path),
        "terminal_evaluation_sha256": _sha256_file(terminal_path),
        "localization_metrics_path": str(metrics_path),
        "localization_metrics_sha256": _sha256_file(metrics_path),
        "checkpoint_sha256": _require_sha256(
            terminal.get("checkpoint_sha256"),
            "execution checkpoint hash",
        ),
        "detector_backend": str(terminal.get("detector_backend", "")),
        "evaluation_seed": int(terminal.get("seed", -1)),
        "prediction_sha256": terminal["prediction_sha256"],
        "annotation_sha256": metrics["annotation_sha256"],
        "nms_contract_sha256": canonical_sha256(nms_contract),
        "post_processing_pipeline_identity_sha256": (
            post_processing_execution["pipeline_identity_sha256"]
        ),
        "post_processing_execution_sha256": post_processing_execution[
            "content_sha256"
        ],
        "official_evaluator_source_sha256": _sha256_file(
            Path(__file__).with_name("evaluate_duca_rime_predictions.py")
        ),
        "per_video_realized_effective_k": dict(sorted(video_costs.items())),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    artifact = _atomic_write_json(output_receipt, payload)
    return {**artifact, "payload": payload}


def finalize_stage1_oracle(
    *,
    plan_manifest: str | Path,
    plan_manifest_sha256: str,
    preregistration: str | Path,
    preregistration_sha256: str,
    execution_receipts: Sequence[tuple[str, int, str, str]],
    output_receipt: str | Path,
) -> dict[str, Any]:
    plan_path = Path(plan_manifest).expanduser().resolve()
    plan = _read_json(plan_path, plan_manifest_sha256)
    _require(plan.get("schema_version") == PLAN_SCHEMA and plan.get("status") == "planned", "plan drift")
    prereg_path = Path(preregistration).expanduser().resolve()
    prereg = _read_json(prereg_path, preregistration_sha256)
    validated = validate_preregistration(
        prereg,
        expected_git_commit=plan["git_commit"],
        expected_split_manifest_sha256=plan["split_manifest_sha256"],
        expected_split_assignment_sha256=plan["split_assignment_sha256"],
    )
    _require(
        plan["preregistration_sha256"] == str(preregistration_sha256).lower(),
        "plan/preregistration file hash drift",
    )
    executions = {}
    common_identity = None
    for strategy, anchor, path, expected_sha in execution_receipts:
        key = (str(strategy), int(anchor))
        _require(key not in executions, f"duplicate execution receipt {key}")
        receipt_path = Path(path).expanduser().resolve()
        receipt = _read_json(receipt_path, expected_sha)
        content_sha = receipt.get("content_sha256")
        _require(
            receipt.get("schema_version") == EXECUTION_RECEIPT_SCHEMA
            and receipt.get("status") == "passed"
            and receipt.get("strategy") == key[0]
            and int(receipt.get("anchor_nominal_budget", -1)) == key[1]
            and receipt.get("plan_manifest_sha256") == str(plan_manifest_sha256).lower()
            and receipt.get("uses_official_final") is False
            and receipt.get("oracle_only") is True
            and receipt.get("deployment_candidate") is False
            and receipt.get("position_policy")
            == STRATEGY_CONTRACTS[key[0]]["position_policy"]
            and receipt.get("uses_gt_at_decision")
            is STRATEGY_CONTRACTS[key[0]]["uses_gt"]
            and isinstance(content_sha, str)
            and content_sha == _canonical_without(receipt, "content_sha256"),
            f"execution receipt contract drift for {key}",
        )
        identity = (
            receipt["checkpoint_sha256"],
            receipt["detector_backend"],
            receipt["evaluation_seed"],
            receipt["annotation_sha256"],
            receipt["nms_contract_sha256"],
            receipt["post_processing_pipeline_identity_sha256"],
            receipt["official_evaluator_source_sha256"],
        )
        if common_identity is None:
            common_identity = identity
        else:
            _require(identity == common_identity, "oracle executions do not share detector/evaluator/NMS identity")
        executions[key] = {"path": str(receipt_path), "sha256": _sha256_file(receipt_path), "payload": receipt}
    expected_keys = {
        (strategy, int(anchor))
        for strategy in STRATEGY_CONTRACTS
        for anchor in validated["anchor_nominal_budgets"]
    }
    _require(set(executions) == expected_keys, "oracle execution matrix is incomplete")
    metrics_by_cell = {}
    for key, execution in executions.items():
        receipt = execution["payload"]
        metrics = _read_json(
            receipt["localization_metrics_path"],
            receipt["localization_metrics_sha256"],
        )
        metrics_by_cell[key] = metrics
    primary = validated["primary_endpoint"]
    primary_metric = str(primary["metric"])
    direction = str(primary["direction"])

    def oriented_delta(left: float, right: float, metric_direction: str) -> float:
        return float(left) - float(right) if metric_direction == "higher" else float(right) - float(left)

    primary_results = {}
    surrogate_values = []
    official_values = []
    plan_by_video_anchor = {
        (str(value["video_id"]), int(value["anchor_nominal_budget"])): value
        for value in plan["video_plans"]
    }
    for anchor in validated["anchor_nominal_budgets"]:
        joint_metrics = metrics_by_cell[("joint_oracle", int(anchor))]["video_metrics"][
            primary_metric
        ]
        for comparator in ("uniform_same_total", "independent_exact_total"):
            comparator_metrics = metrics_by_cell[(comparator, int(anchor))]["video_metrics"][
                primary_metric
            ]
            _require(set(joint_metrics) == set(comparator_metrics), "paired metric video set drift")
            deltas = {
                video: oriented_delta(
                    joint_metrics[video],
                    comparator_metrics[video],
                    direction,
                )
                for video in sorted(joint_metrics)
            }
            stats = paired_video_bootstrap(
                deltas,
                samples=int(validated["bootstrap"]["samples"]),
                seed=int(validated["bootstrap"]["seed"]) + int(anchor),
                alpha=float(primary["alpha"]),
            )
            primary_results[f"k{int(anchor)}:{comparator}"] = {
                "metric": primary_metric,
                "direction": direction,
                "paired_delta_by_video": deltas,
                "bootstrap": stats,
                "passes_mean": stats["mean"] >= float(primary["min_mean_delta"]),
                "passes_lcb": stats["ci_low"] >= float(primary["min_lcb_delta"]),
                "passes_noninferiority": stats["ci_low"]
                >= -float(primary["noninferiority_margin"]),
            }
            for video in sorted(deltas):
                video_plan = plan_by_video_anchor[(video, int(anchor))]
                joint_surrogate = float(
                    video_plan["strategies"]["joint_oracle"]["surrogate_objective"]
                )
                comparator_surrogate = float(
                    video_plan["strategies"][comparator]["surrogate_objective"]
                )
                surrogate_values.append(joint_surrogate - comparator_surrogate)
                official_values.append(deltas[video])
    surrogate_result = surrogate_audit(surrogate_values, official_values)
    surrogate_gate = validated["surrogate_audit"]
    surrogate_pass = (
        surrogate_result["spearman"] >= float(surrogate_gate["min_spearman"])
        and surrogate_result["sign_agreement"]
        >= float(surrogate_gate["min_sign_agreement"])
        and surrogate_result["worst_rank_error"]
        <= float(surrogate_gate["max_worst_rank_error"])
    )
    guardrail_results = []
    for guardrail in validated["guardrails"]:
        metric = str(guardrail["metric"])
        metric_direction = str(guardrail["direction"])
        comparator = str(guardrail["comparator"])
        values = {}
        for anchor in validated["anchor_nominal_budgets"]:
            joint = metrics_by_cell[("joint_oracle", int(anchor))]["video_metrics"][metric]
            control = metrics_by_cell[(comparator, int(anchor))]["video_metrics"][metric]
            _require(set(joint) == set(control), "guardrail video set drift")
            for video in sorted(joint):
                values[f"k{int(anchor)}:{video}"] = oriented_delta(
                    joint[video],
                    control[video],
                    metric_direction,
                )
        observed = mean(values.values())
        guardrail_results.append(
            {
                **dict(guardrail),
                "oriented_mean_delta": observed,
                "passes": observed >= float(guardrail["min_delta"]),
            }
        )
    primary_pass = all(
        result["passes_mean"] and result["passes_lcb"] and result["passes_noninferiority"]
        for result in primary_results.values()
    )
    guardrails_pass = all(result["passes"] for result in guardrail_results)
    passed = bool(primary_pass and guardrails_pass and surrogate_pass)
    payload = {
        "schema_version": ORACLE_RECEIPT_SCHEMA,
        "status": "passed" if passed else "failed_closed",
        "task": "offline_temporal_action_detection",
        "git_commit": plan["git_commit"],
        "plan_manifest_path": str(plan_path),
        "plan_manifest_sha256": _sha256_file(plan_path),
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": _sha256_file(prereg_path),
        "development_role": plan["development_role"],
        "uses_official_final": False,
        "execution_matrix_complete": True,
        "common_execution_identity": {
            "checkpoint_sha256": common_identity[0],
            "detector_backend": common_identity[1],
            "evaluation_seed": common_identity[2],
            "annotation_sha256": common_identity[3],
            "nms_contract_sha256": common_identity[4],
            "post_processing_pipeline_identity_sha256": common_identity[5],
            "official_evaluator_source_sha256": common_identity[6],
        },
        "executions": {
            f"{strategy}:k{anchor}": {
                "path": value["path"],
                "sha256": value["sha256"],
            }
            for (strategy, anchor), value in sorted(executions.items())
        },
        "primary_endpoint": primary,
        "multiplicity": {
            **validated["multiplicity"],
            "decision_rule": (
                "all_registered_primary_cells_must_pass_each_preregistered_gate"
            ),
            "observed_family": [
                f"k{anchor}:joint_oracle_vs_{comparator}:{primary_metric}"
                for anchor in validated["anchor_nominal_budgets"]
                for comparator in (
                    "uniform_same_total",
                    "independent_exact_total",
                )
            ],
        },
        "primary_results": primary_results,
        "guardrail_results": guardrail_results,
        "surrogate_audit": {
            **surrogate_result,
            "registered_thresholds": surrogate_gate,
            "passes": surrogate_pass,
        },
        "gate_status": {
            "primary": primary_pass,
            "guardrails": guardrails_pass,
            "surrogate": surrogate_pass,
        },
        "authorizes_stage2_training": passed,
        "claim_scope": "complete_development_oracle_not_paper_result",
    }
    payload["content_sha256"] = canonical_sha256(payload)
    artifact = _atomic_write_json(output_receipt, payload)
    return {**artifact, "payload": payload}


def _execution_binding(value: str) -> tuple[str, int, str, str]:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "execution binding must be strategy:anchor:path:sha256"
        )
    return parts[0], int(parts[1]), parts[2], parts[3]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan, validate, and seal the H-RIME Stage-1 same-total oracle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--preregistration", required=True)
    plan_parser.add_argument("--preregistration-sha256", required=True)
    plan_parser.add_argument("--split-manifest", required=True)
    plan_parser.add_argument("--split-manifest-sha256", required=True)
    plan_parser.add_argument("--window-options-jsonl", required=True)
    plan_parser.add_argument("--window-options-sha256", required=True)
    plan_parser.add_argument("--budget-protocol-sha256", required=True)
    plan_parser.add_argument("--output-root", required=True)
    execution_parser = subparsers.add_parser("validate-execution")
    execution_parser.add_argument("--plan-manifest", required=True)
    execution_parser.add_argument("--plan-manifest-sha256", required=True)
    execution_parser.add_argument("--strategy", choices=tuple(STRATEGY_CONTRACTS), required=True)
    execution_parser.add_argument("--anchor-nominal-budget", type=int, required=True)
    execution_parser.add_argument("--replay-jsonl", required=True)
    execution_parser.add_argument("--replay-sha256", required=True)
    execution_parser.add_argument("--inference-ledger-jsonl", required=True)
    execution_parser.add_argument("--terminal-evaluation", required=True)
    execution_parser.add_argument("--localization-metrics", required=True)
    execution_parser.add_argument("--output-receipt", required=True)
    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--plan-manifest", required=True)
    final_parser.add_argument("--plan-manifest-sha256", required=True)
    final_parser.add_argument("--preregistration", required=True)
    final_parser.add_argument("--preregistration-sha256", required=True)
    final_parser.add_argument(
        "--execution",
        action="append",
        type=_execution_binding,
        required=True,
    )
    final_parser.add_argument("--output-receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        split_validation = validate_rime_splits(
            args.split_manifest,
            expected_sha256=args.split_manifest_sha256,
        )
        split = _read_json(args.split_manifest, args.split_manifest_sha256)
        prereg = _read_json(args.preregistration, args.preregistration_sha256)
        validated_prereg = validate_preregistration(
            prereg,
            expected_split_manifest_sha256=split_validation["manifest_sha256"],
            expected_split_assignment_sha256=split_validation["assignment_sha256"],
        )
        role = validated_prereg["development_role"]
        _require(role in TRAIN_ROLES, "preregistered development role is unknown")
        rows = _read_jsonl(args.window_options_jsonl, args.window_options_sha256)
        validated_rows = validate_window_option_rows(
            rows,
            expected_videos=split["train_roles"][role]["videos"],
            expected_split_role=role,
            expected_split_assignment_sha256=split_validation["assignment_sha256"],
            candidate_budgets=validated_prereg["candidate_budgets"],
        )
        result = build_stage1_plan(
            rows=validated_rows,
            window_options_sha256=args.window_options_sha256,
            preregistration=prereg,
            preregistration_sha256=args.preregistration_sha256,
            budget_protocol_sha256=args.budget_protocol_sha256,
            output_root=args.output_root,
        )
    elif args.command == "validate-execution":
        result = validate_stage1_execution(
            plan_manifest=args.plan_manifest,
            plan_manifest_sha256=args.plan_manifest_sha256,
            strategy=args.strategy,
            anchor_nominal_budget=args.anchor_nominal_budget,
            replay_jsonl=args.replay_jsonl,
            replay_sha256=args.replay_sha256,
            inference_ledger_jsonl=args.inference_ledger_jsonl,
            terminal_evaluation=args.terminal_evaluation,
            localization_metrics=args.localization_metrics,
            output_receipt=args.output_receipt,
        )
    else:
        result = finalize_stage1_oracle(
            plan_manifest=args.plan_manifest,
            plan_manifest_sha256=args.plan_manifest_sha256,
            preregistration=args.preregistration,
            preregistration_sha256=args.preregistration_sha256,
            execution_receipts=args.execution,
            output_receipt=args.output_receipt,
        )
    print(json.dumps({key: value for key, value in result.items() if key != "payload"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
