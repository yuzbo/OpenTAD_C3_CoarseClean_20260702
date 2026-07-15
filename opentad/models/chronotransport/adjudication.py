"""Pure, schema-first adjudicators for the ChronoTransport r2 hard gates."""

from __future__ import annotations

from decimal import Decimal
import math
import random
import re
from typing import Mapping, Sequence

from .full_stack_profiler import (
    validate_full_stack_profile_artifact,
    validate_full_stack_profile_artifact_for_test_only,
)
from .protocol import canonical_sha256
from .registration import (
    EXPECTED_PROFILE_CANDIDATE_ORDER,
    REGISTERED_PROFILE_BACKEND_IDENTITY,
    REGISTERED_PROFILE_BACKEND_SOURCE,
    validate_formal_gate1_context,
    validate_pre_gate1_registration,
)


HOLD_TIME = ("periodic2_hold", "periodic4_hold", "periodic8_hold")
HOLD_LAYER = ("layer_only_early_recompute_hold", "layer_only_late_recompute_hold")
GATE1_HOLD_NAMES = (
    "periodic2_hold",
    "periodic4_hold",
    "periodic8_hold",
    "hold_only",
    "layer_only_early_recompute_hold",
    "layer_only_late_recompute_hold",
    "joint_progressive_hold",
    "joint_reverse_hold",
)
GATE1_CONTROL_NAMES = (
    "motion_topk_p2",
    "motion_topk_p4",
    "motion_topk_p8",
    "random_p2",
    "random_p4",
    "random_p8",
)
GATE1_RECORD_CANDIDATE_ORDER = GATE1_HOLD_NAMES + GATE1_CONTROL_NAMES
GATE1_PAIRED_CANDIDATE_ORDER = EXPECTED_PROFILE_CANDIDATE_ORDER
GATE1_RECORD_SCHEMA = "chronotransport-r2-gate1-record-artifact-formal-v2"
GATE1_RECORD_FIXTURE_SCHEMA = "chronotransport-r2-gate1-record-test-fixture-v2"
GATE1_PAIRED_REPLAY_SCHEMA = "chronotransport-r2-gate1-paired-replay-formal-v2"
GATE1_PAIRED_REPLAY_FIXTURE_SCHEMA = (
    "chronotransport-r2-gate1-paired-replay-test-fixture-v2"
)
_FORMAL_REPLAY_FIELDS = {
    "schema",
    "registration_sha256",
    "split",
    "window_ids",
    "window_order_sha256",
    "candidate_names",
    "candidate_order_sha256",
    "order_probe_candidate_names",
    "order_probe_candidate_order_sha256",
    "rows",
    "artifact_sha256",
}
_FIXTURE_REPLAY_FIELDS = {
    "schema",
    "fixture_registration_sha256",
    "fixture_split",
    "fixture_window_ids",
    "fixture_window_order_sha256",
    "fixture_candidate_names",
    "fixture_candidate_order_sha256",
    "fixture_order_probe_candidate_names",
    "fixture_order_probe_candidate_order_sha256",
    "fixture_rows",
    "fixture_artifact_sha256",
}
_FORMAL_RECORD_FIELDS = {
    "schema",
    "registration_sha256",
    "split",
    "window_ids",
    "window_order_sha256",
    "candidate_names",
    "candidate_order_sha256",
    "paired_replay",
    "paired_replay_artifact_sha256",
    "rows",
    "artifact_sha256",
}
_FIXTURE_RECORD_FIELDS = {
    "schema",
    "fixture_registration_sha256",
    "fixture_split",
    "fixture_window_ids",
    "fixture_window_order_sha256",
    "fixture_candidate_names",
    "fixture_candidate_order_sha256",
    "fixture_paired_replay",
    "fixture_paired_replay_artifact_sha256",
    "fixture_rows",
    "fixture_artifact_sha256",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires non-empty values")
    return float(sum(values) / len(values))


def _percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("percentile CI requires values")
    last = len(ordered) - 1
    return ordered[int(math.floor(0.025 * last))], ordered[int(math.ceil(0.975 * last))]


def _normalize_vectors(
    records: Mapping[str, Mapping[str, float]],
    names: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, list[float]]]:
    windows = tuple(sorted(map(str, records)))
    if not windows:
        raise ValueError("gate records require at least one window")
    vectors = {name: [] for name in names}
    for window in windows:
        row = records[window]
        for name in names:
            if name not in row:
                raise ValueError(f"window {window} is missing candidate {name}")
            vectors[name].append(_finite(row[name], f"regret[{window},{name}]"))
    return windows, vectors


def _argmin_mean(names: Sequence[str], vectors: Mapping[str, Sequence[float]]) -> str:
    if not names:
        raise ValueError("candidate selection requires non-empty names")
    return min(names, key=lambda name: (_mean(vectors[name]), tuple(names).index(name)))


def gate1_oracle_headroom(
    *,
    calibration: Mapping[str, Mapping[str, float]],
    evaluation: Mapping[str, Mapping[str, float]],
    candidate_cost_p50: Mapping[str, float],
    dense_cost_p50: float,
    budget: float,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260711,
) -> dict[str, object]:
    """Adjudicate the frozen equal-cost HOLD-library oracle headroom gate."""

    dense_cost = _finite(dense_cost_p50, "dense cost")
    budget = _finite(budget, "budget")
    if dense_cost <= 0 or budget <= 0:
        raise ValueError("costs and budget must be positive")
    costs = {str(name): _finite(value, f"cost[{name}]") for name, value in candidate_cost_p50.items()}
    hold_names = tuple(
        name
        for name in costs
        if (name.endswith("_hold") or name == "hold_only") and costs[name] <= budget
    )
    for required in (*HOLD_TIME, *HOLD_LAYER):
        if required not in hold_names:
            raise ValueError(f"Gate 1 budget lacks required HOLD candidate: {required}")
    controls = tuple(
        name for name in costs if name.startswith(("motion_topk_p", "random_p")) and costs[name] <= budget
    )
    required_controls = tuple(
        f"{prefix}_p{period}"
        for prefix in ("motion_topk", "random")
        for period in (2, 4, 8)
    )
    missing_controls = [name for name in required_controls if name not in controls]
    if missing_controls:
        raise ValueError(f"Gate 1 requires cost-feasible comparators: {missing_controls}")

    all_names = tuple(dict.fromkeys((*hold_names, *controls)))
    _, calibration_vectors = _normalize_vectors(calibration, hold_names)
    windows, evaluation_vectors = _normalize_vectors(evaluation, all_names)
    calibration_static = _argmin_mean(hold_names, calibration_vectors)

    def replicate(indices: Sequence[int]) -> tuple[float, str, str, dict[str, float]]:
        sampled = {
            name: [evaluation_vectors[name][index] for index in indices]
            for name in all_names
        }
        evaluation_static = _argmin_mean(hold_names, sampled)
        time_oracle = [min(sampled[name][pos] for name in HOLD_TIME) for pos in range(len(indices))]
        layer_oracle = [min(sampled[name][pos] for name in HOLD_LAYER) for pos in range(len(indices))]
        joint_oracle = [min(sampled[name][pos] for name in hold_names) for pos in range(len(indices))]
        comparators: dict[str, list[float]] = {
            f"calibration_static:{calibration_static}": sampled[calibration_static],
            "time_only_oracle": time_oracle,
            "layer_only_oracle": layer_oracle,
            f"evaluation_static:{evaluation_static}": sampled[evaluation_static],
        }
        comparators.update({name: sampled[name] for name in controls})
        strongest = min(comparators, key=lambda name: (_mean(comparators[name]), name))
        means = {name: _mean(vector) for name, vector in comparators.items()}
        return (
            _mean([left - right for left, right in zip(comparators[strongest], joint_oracle)]),
            strongest,
            evaluation_static,
            {**means, "joint_oracle": _mean(joint_oracle)},
        )

    full_indices = list(range(len(windows)))
    improvement, strongest, evaluation_static, means = replicate(full_indices)
    strongest_mean = means[strongest]
    relative = float("nan") if strongest_mean <= 1e-12 else improvement / strongest_mean
    rng = random.Random(int(bootstrap_seed))
    bootstrap = []
    for _ in range(int(bootstrap_samples)):
        indices = [rng.randrange(len(windows)) for _ in windows]
        bootstrap.append(replicate(indices)[0])
    ci = _percentile_ci(bootstrap)
    saving = 1.0 - budget / dense_cost
    exact_budget_saving = Decimal(str(budget)) * 5 <= Decimal(str(dense_cost)) * 4
    hard = {
        "relative_reduction_ge_10pct": strongest_mean > 1e-12 and relative >= 0.10,
        "paired_bootstrap_ci_lower_gt_0": ci[0] > 0.0,
        "budget_saving_ge_20pct": exact_budget_saving,
    }
    return {
        "schema": "chronotransport-r2-gate1-v1",
        "status": "PASS" if all(hard.values()) else "FAIL",
        "oracle_headroom": bool(all(hard.values())),
        "windows": len(windows),
        "candidate_set_size": len(hold_names),
        "feasible_hold_names": list(hold_names),
        "time_oracle_set_size": len(HOLD_TIME),
        "layer_oracle_set_size": len(HOLD_LAYER),
        "joint_oracle_set_size": len(hold_names),
        "calibration_frozen_static": calibration_static,
        "evaluation_best_static": evaluation_static,
        "strongest_comparator": strongest,
        "mean_regret": means,
        "absolute_improvement": improvement,
        "relative_reduction": relative,
        "bootstrap_ci95": list(ci),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "dense_cost_p50": dense_cost,
        "budget": budget,
        "budget_saving": saving,
        "hard_conditions": hard,
    }


def gate1_oracle_headroom_from_profile(
    *,
    registration: Mapping[str, object],
    calibration: Mapping[str, object],
    evaluation: Mapping[str, object],
    full_stack_profile: Mapping[str, object],
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    """Run Gate 1 only from registered, direct full-invocation cost evidence."""

    registered = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    return _gate1_oracle_headroom_from_profile_validated(
        registered=registered,
        calibration=calibration,
        evaluation=evaluation,
        full_stack_profile=full_stack_profile,
        fixture=False,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def gate1_oracle_headroom_from_profile_for_test_only(
    *,
    registration: Mapping[str, object],
    calibration: Mapping[str, object],
    evaluation: Mapping[str, object],
    full_stack_profile: Mapping[str, object],
) -> dict[str, object]:
    """Exercise statistics on synthetic fixtures without minting a formal report."""

    return _gate1_oracle_headroom_from_profile_validated(
        registered=validate_pre_gate1_registration(registration),
        calibration=calibration,
        evaluation=evaluation,
        full_stack_profile=full_stack_profile,
        fixture=True,
        repository_root=None,
        registration_commit=None,
        registration_relpath=None,
    )


def _gate1_oracle_headroom_from_profile_validated(
    *,
    registered: Mapping[str, object],
    calibration: Mapping[str, object],
    evaluation: Mapping[str, object],
    full_stack_profile: Mapping[str, object],
    fixture: bool,
    repository_root: str | None,
    registration_commit: str | None,
    registration_relpath: str | None,
) -> dict[str, object]:
    profile_validator = (
        validate_full_stack_profile_artifact_for_test_only
        if fixture
        else validate_full_stack_profile_artifact
    )
    record_validator = (
        validate_gate1_record_artifact_for_test_only
        if fixture
        else validate_gate1_record_artifact
    )
    if fixture:
        full_stack_profile = profile_validator(
            full_stack_profile, registration=registered
        )
        calibration_artifact = record_validator(
            calibration, registration=registered, expected_split="calibration"
        )
        evaluation_artifact = record_validator(
            evaluation, registration=registered, expected_split="evaluation"
        )
    else:
        context = {
            "repository_root": repository_root,
            "registration_commit": registration_commit,
            "registration_relpath": registration_relpath,
        }
        full_stack_profile = profile_validator(
            full_stack_profile, registration=registered, **context
        )
        calibration_artifact = record_validator(
            calibration,
            registration=registered,
            expected_split="calibration",
            **context,
        )
        evaluation_artifact = record_validator(
            evaluation,
            registration=registered,
            expected_split="evaluation",
            **context,
        )
    calibration_records = _record_artifact_mapping(
        calibration_artifact, fixture=fixture
    )
    evaluation_records = _record_artifact_mapping(
        evaluation_artifact, fixture=fixture
    )
    profiles = {}
    candidate_field = "fixture_candidates" if fixture else "candidates"
    provenance_field = "fixture_provenance" if fixture else "provenance"
    total_field = "fixture_total_ms" if fixture else "total_ms"
    for candidate in full_stack_profile[candidate_field]:
        name = str(candidate[provenance_field]["candidate_name"])
        profiles[name] = candidate
    for required in ("dense", "periodic4_transport"):
        if required not in profiles:
            raise ValueError(f"Gate 1 full-stack profile is missing {required}")
    costs = {
        name: float(candidate[total_field]["p50"])
        for name, candidate in profiles.items()
    }
    budget = costs["periodic4_transport"]
    result = gate1_oracle_headroom(
        calibration=calibration_records,
        evaluation=evaluation_records,
        candidate_cost_p50=costs,
        dense_cost_p50=costs["dense"],
        budget=budget,
        bootstrap_samples=registered["bootstrap"]["gate1_samples"],
        bootstrap_seed=registered["bootstrap"]["seed"],
    )
    result.update(
        {
            "schema": "chronotransport-r2-gate1-v2",
            "budget_source": "measured_p50:periodic4_transport",
            "full_stack_profile_sha256": full_stack_profile[
                "fixture_profile_sha256" if fixture else "profile_sha256"
            ],
            "registration_sha256": full_stack_profile[
                "fixture_registration_sha256" if fixture else "registration_sha256"
            ],
            "calibration_artifact_sha256": calibration_artifact[
                "fixture_artifact_sha256" if fixture else "artifact_sha256"
            ],
            "evaluation_artifact_sha256": evaluation_artifact[
                "fixture_artifact_sha256" if fixture else "artifact_sha256"
            ],
            "candidate_cost_p50": costs,
        }
    )
    return result


def _validate_gate1_record_row(row: object, *, window_id: str) -> dict[str, object]:
    if not isinstance(row, Mapping) or set(row) != {
        "window_id",
        "candidate_names",
        "detector_regret",
    }:
        raise ValueError("Gate 1 record row fields mismatch")
    if row["window_id"] != window_id:
        raise ValueError("Gate 1 record window IDs/order mismatch")
    names = row["candidate_names"]
    if not isinstance(names, list) or tuple(names) != GATE1_RECORD_CANDIDATE_ORDER:
        raise ValueError("Gate 1 record candidate names/order mismatch")
    regrets = row["detector_regret"]
    if not isinstance(regrets, list) or len(regrets) != len(names):
        raise ValueError("Gate 1 record requires one detector regret per candidate")
    for value in regrets:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Gate 1 detector regret must be a numeric scalar")
        if not math.isfinite(float(value)):
            raise ValueError("Gate 1 detector regret must be finite")
    return dict(row)


def _validate_replay_source_values(
    row: object,
    *,
    registration: Mapping[str, object],
    window_id: str,
    invocation_index: int,
) -> tuple[dict[str, object], list[float]]:
    source_fields = {
        "window_id",
        "candidate_names",
        "dense_detector_loss",
        "candidate_detector_loss",
        "order_probe_candidate_names",
        "order_probe_candidate_detector_loss",
        "materialized_window_sha256",
        "augmentation_sha256",
        "deploy_visible_motion_sha256",
        "dense_reference_sha256",
        "dense_checkpoint_sha256",
        "config_sha256",
        "backend_identity",
        "backend_source_sha256",
        "candidate_action_sha256",
    }
    if not isinstance(row, Mapping) or set(row) != source_fields:
        raise ValueError("Gate 1 paired replay row fields mismatch")
    if row["window_id"] != window_id:
        raise ValueError("Gate 1 paired replay window IDs/order mismatch")
    names = row["candidate_names"]
    if not isinstance(names, list) or tuple(names) != GATE1_PAIRED_CANDIDATE_ORDER:
        raise ValueError("Gate 1 paired replay requires the fixed full candidate order")
    dense_loss = _finite(row["dense_detector_loss"], "dense detector loss")
    if dense_loss < 0.0:
        raise ValueError("Gate 1 paired replay detector losses must be non-negative")
    candidate_losses = row["candidate_detector_loss"]
    if not isinstance(candidate_losses, list) or len(candidate_losses) != len(names):
        raise ValueError("Gate 1 paired replay requires one candidate loss per candidate")
    candidate_losses = [
        _finite(value, "candidate detector loss") for value in candidate_losses
    ]
    if any(value < 0.0 for value in candidate_losses):
        raise ValueError("Gate 1 paired replay detector losses must be non-negative")
    probe_names = row["order_probe_candidate_names"]
    probe_losses = row["order_probe_candidate_detector_loss"]
    if (
        not isinstance(probe_names, list)
        or tuple(probe_names) != tuple(reversed(GATE1_PAIRED_CANDIDATE_ORDER))
        or not isinstance(probe_losses, list)
        or len(probe_losses) != len(probe_names)
    ):
        raise ValueError("Gate 1 paired replay order probe must use the fixed reverse order")
    reordered = {
        name: _finite(value, "order probe candidate detector loss")
        for name, value in zip(probe_names, probe_losses)
    }
    if [reordered[name] for name in names] != candidate_losses:
        raise ValueError("Gate 1 paired replay candidate-order probe changed the loss vector")
    for field in (
        "materialized_window_sha256",
        "augmentation_sha256",
        "deploy_visible_motion_sha256",
        "dense_reference_sha256",
        "dense_checkpoint_sha256",
        "config_sha256",
        "backend_source_sha256",
    ):
        if not isinstance(row[field], str) or not _SHA256.fullmatch(row[field]):
            raise ValueError(f"Gate 1 paired replay {field} must be a SHA-256")
    if row["dense_checkpoint_sha256"] != registration["dense_checkpoint"]["sha256"]:
        raise ValueError("Gate 1 paired replay dense checkpoint provenance mismatch")
    if row["config_sha256"] != registration["profiler"]["model_config_sha256"]:
        raise ValueError("Gate 1 paired replay config provenance mismatch")
    if row["backend_identity"] != REGISTERED_PROFILE_BACKEND_IDENTITY:
        raise ValueError("Gate 1 paired replay backend identity mismatch")
    if (
        row["backend_source_sha256"]
        != registration["source_files"][REGISTERED_PROFILE_BACKEND_SOURCE]
    ):
        raise ValueError("Gate 1 paired replay backend source mismatch")
    action_hashes = row["candidate_action_sha256"]
    if not isinstance(action_hashes, list) or len(action_hashes) != len(
        GATE1_PAIRED_CANDIDATE_ORDER
    ):
        raise ValueError("Gate 1 paired replay action provenance vector mismatch")
    plans = {
        plan["candidate_name"]: plan for plan in registration["profiler"]["candidate_plan"]
    }
    expected_actions = [
        plans[name]["requested_action_sha256_by_invocation"][invocation_index]
        for name in GATE1_PAIRED_CANDIDATE_ORDER
    ]
    if action_hashes != expected_actions:
        raise ValueError("Gate 1 paired replay action provenance mismatch")
    rebuilt_source = dict(row)
    rebuilt_source["dense_detector_loss"] = dense_loss
    rebuilt_source["candidate_detector_loss"] = candidate_losses
    rebuilt_source["order_probe_candidate_detector_loss"] = [
        reordered[name] for name in probe_names
    ]
    detector_regret = [
        max(candidate_loss - dense_loss, 0.0)
        for candidate_loss in candidate_losses
    ]
    dense_index = GATE1_PAIRED_CANDIDATE_ORDER.index("dense")
    if (
        candidate_losses[dense_index] != dense_loss
        or detector_regret[dense_index] != 0.0
    ):
        raise ValueError("Gate 1 paired replay dense candidate must equal dense reference")
    return rebuilt_source, detector_regret


def _validate_serialized_formal_replay_row(
    row: object,
    *,
    registration: Mapping[str, object],
    window_id: str,
    invocation_index: int,
) -> dict[str, object]:
    """Validate one already-serialized formal row without minting derived fields."""

    if not isinstance(row, Mapping) or "detector_regret" not in row:
        raise ValueError("formal Gate 1 paired replay row requires serialized regret")
    source = dict(row)
    serialized_regret = source.pop("detector_regret")
    rebuilt_source, expected_regret = _validate_replay_source_values(
        source,
        registration=registration,
        window_id=window_id,
        invocation_index=invocation_index,
    )
    if not isinstance(serialized_regret, list) or serialized_regret != expected_regret:
        raise ValueError("formal Gate 1 paired replay serialized regret mismatch")
    return {**rebuilt_source, "detector_regret": expected_regret}


def _build_fixture_replay_row_from_source(
    row: object,
    *,
    registration: Mapping[str, object],
    window_id: str,
    invocation_index: int,
) -> dict[str, object]:
    """Build the disjoint fixture row directly from fixture source values."""

    rebuilt_source, detector_regret = _validate_replay_source_values(
        row,
        registration=registration,
        window_id=window_id,
        invocation_index=invocation_index,
    )
    return {
        "fixture_window_id": rebuilt_source.pop("window_id"),
        "fixture_source": rebuilt_source,
        "fixture_derived_detector_regret": detector_regret,
    }


def _fixture_replay_common_from_source_rows(
    *,
    registration: Mapping[str, object],
    split: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    registered = validate_pre_gate1_registration(registration)
    if split not in ("calibration", "evaluation"):
        raise ValueError("Gate 1 paired replay split must be calibration or evaluation")
    manifest = registered["window_manifest"]["artifact"]
    expected_ids = manifest["splits"][split]
    if not isinstance(rows, (list, tuple)) or len(rows) != 30:
        raise ValueError("Gate 1 paired replay requires exactly 30 manifested rows")
    global_index = {
        window_id: index
        for index, window_id in enumerate(registered["profiler"]["invocation_ids"])
    }
    fixture_rows = [
        _build_fixture_replay_row_from_source(
            row,
            registration=registered,
            window_id=window_id,
            invocation_index=global_index[window_id],
        )
        for row, window_id in zip(rows, expected_ids)
    ]
    return {
        "fixture_registration_sha256": registered["registration_sha256"],
        "fixture_split": split,
        "fixture_window_ids": list(expected_ids),
        "fixture_window_order_sha256": canonical_sha256(expected_ids),
        "fixture_candidate_names": list(GATE1_PAIRED_CANDIDATE_ORDER),
        "fixture_candidate_order_sha256": canonical_sha256(GATE1_PAIRED_CANDIDATE_ORDER),
        "fixture_order_probe_candidate_names": list(reversed(GATE1_PAIRED_CANDIDATE_ORDER)),
        "fixture_order_probe_candidate_order_sha256": canonical_sha256(
            tuple(reversed(GATE1_PAIRED_CANDIDATE_ORDER))
        ),
        "fixture_rows": fixture_rows,
    }


def build_gate1_paired_replay_artifact(
    *,
    registration: Mapping[str, object],
    split: str,
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    """Run the repository-owned replay session; caller rows are never accepted."""

    from .replay import run_registered_gate1_paired_replay

    validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    return run_registered_gate1_paired_replay(
        registration=registration,
        split=split,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def build_gate1_paired_replay_artifact_for_test_only(
    *,
    registration: Mapping[str, object],
    split: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Test-fixture builder; formal inputs never accept arbitrary rows."""

    common = _fixture_replay_common_from_source_rows(
        registration=registration,
        split=split,
        rows=rows,
    )
    body = {"schema": GATE1_PAIRED_REPLAY_FIXTURE_SCHEMA, **common}
    body["fixture_artifact_sha256"] = canonical_sha256(body)
    return body


def _formal_replay_common_from_serialized(
    artifact: Mapping[str, object],
    *,
    registration: Mapping[str, object],
    expected_split: str,
) -> dict[str, object]:
    if set(artifact) != _FORMAL_REPLAY_FIELDS:
        raise ValueError("formal Gate 1 paired replay artifact fields mismatch")
    if artifact.get("split") != expected_split:
        raise ValueError("formal Gate 1 paired replay split mismatch")
    expected_ids = registration["window_manifest"]["artifact"]["splits"][expected_split]
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError("formal Gate 1 paired replay requires exact serialized rows")
    global_index = {
        window_id: index
        for index, window_id in enumerate(registration["profiler"]["invocation_ids"])
    }
    rebuilt_rows = []
    for row, window_id in zip(rows, expected_ids):
        rebuilt = _validate_serialized_formal_replay_row(
            row,
            registration=registration,
            window_id=window_id,
            invocation_index=global_index[window_id],
        )
        if rebuilt != dict(row):
            raise ValueError("formal Gate 1 paired replay serialized row mismatch")
        rebuilt_rows.append(rebuilt)
    return {
        "registration_sha256": registration["registration_sha256"],
        "split": expected_split,
        "window_ids": list(expected_ids),
        "window_order_sha256": canonical_sha256(expected_ids),
        "candidate_names": list(GATE1_PAIRED_CANDIDATE_ORDER),
        "candidate_order_sha256": canonical_sha256(GATE1_PAIRED_CANDIDATE_ORDER),
        "order_probe_candidate_names": list(reversed(GATE1_PAIRED_CANDIDATE_ORDER)),
        "order_probe_candidate_order_sha256": canonical_sha256(
            tuple(reversed(GATE1_PAIRED_CANDIDATE_ORDER))
        ),
        "rows": rebuilt_rows,
    }


def validate_gate1_paired_replay_artifact(
    artifact: Mapping[str, object],
    *,
    registration: Mapping[str, object],
    expected_split: str,
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    registered = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema") != GATE1_PAIRED_REPLAY_SCHEMA
    ):
        raise ValueError("formal Gate 1 paired replay artifact schema mismatch")
    common = _formal_replay_common_from_serialized(
        artifact,
        registration=registered,
        expected_split=expected_split,
    )
    rebuilt = {"schema": GATE1_PAIRED_REPLAY_SCHEMA, **common}
    rebuilt["artifact_sha256"] = canonical_sha256(rebuilt)
    if rebuilt != dict(artifact):
        raise ValueError("Gate 1 paired replay artifact hash/provenance mismatch")
    return rebuilt


def validate_gate1_paired_replay_artifact_for_test_only(
    artifact: Mapping[str, object],
    *,
    registration: Mapping[str, object],
    expected_split: str,
) -> dict[str, object]:
    if (
        not isinstance(artifact, Mapping)
        or set(artifact) != _FIXTURE_REPLAY_FIELDS
        or artifact.get("schema") != GATE1_PAIRED_REPLAY_FIXTURE_SCHEMA
    ):
        raise ValueError("Gate 1 paired replay test-fixture fields/schema mismatch")
    if artifact.get("fixture_split") != expected_split:
        raise ValueError("Gate 1 paired replay test-fixture split mismatch")
    rows = artifact.get("fixture_rows", ())
    if not isinstance(rows, list):
        raise ValueError("Gate 1 paired replay test rows must be a list")
    source_rows = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "fixture_window_id",
            "fixture_source",
            "fixture_derived_detector_regret",
        }:
            raise ValueError("Gate 1 paired replay test fixture row fields mismatch")
        source = row["fixture_source"]
        if not isinstance(source, Mapping):
            raise ValueError("Gate 1 paired replay test fixture source must be a mapping")
        source = {"window_id": row["fixture_window_id"], **dict(source)}
        source_rows.append(source)
    common = _fixture_replay_common_from_source_rows(
        registration=registration,
        split=expected_split,
        rows=source_rows,
    )
    rebuilt = {"schema": GATE1_PAIRED_REPLAY_FIXTURE_SCHEMA, **common}
    rebuilt["fixture_artifact_sha256"] = canonical_sha256(rebuilt)
    if rebuilt != dict(artifact):
        raise ValueError("Gate 1 paired replay test-fixture hash/provenance mismatch")
    return rebuilt


def _build_formal_gate1_record_from_serialized_replay(
    *,
    registration: Mapping[str, object],
    split: str,
    paired_replay: Mapping[str, object],
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    replay = validate_gate1_paired_replay_artifact(
        paired_replay,
        registration=registration,
        expected_split=split,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    expected_ids = registration["window_manifest"]["artifact"]["splits"][split]
    paired_index = {
        name: index for index, name in enumerate(GATE1_PAIRED_CANDIDATE_ORDER)
    }
    rows = [
        _validate_gate1_record_row(
            {
                "window_id": row["window_id"],
                "candidate_names": list(GATE1_RECORD_CANDIDATE_ORDER),
                "detector_regret": [
                    row["detector_regret"][paired_index[name]]
                    for name in GATE1_RECORD_CANDIDATE_ORDER
                ],
            },
            window_id=window_id,
        )
        for row, window_id in zip(replay["rows"], expected_ids)
    ]
    body: dict[str, object] = {
        "schema": GATE1_RECORD_SCHEMA,
        "registration_sha256": registration["registration_sha256"],
        "split": split,
        "window_ids": list(expected_ids),
        "window_order_sha256": canonical_sha256(expected_ids),
        "candidate_names": list(GATE1_RECORD_CANDIDATE_ORDER),
        "candidate_order_sha256": canonical_sha256(GATE1_RECORD_CANDIDATE_ORDER),
        "paired_replay": replay,
        "paired_replay_artifact_sha256": replay["artifact_sha256"],
        "rows": rows,
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def build_gate1_record_artifact(
    *,
    registration: Mapping[str, object],
    split: str,
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    registered = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    paired_replay = build_gate1_paired_replay_artifact(
        registration=registered,
        split=split,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    return _build_formal_gate1_record_from_serialized_replay(
        registration=registered,
        split=split,
        paired_replay=paired_replay,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def build_gate1_record_artifact_for_test_only(
    *,
    registration: Mapping[str, object],
    split: str,
    paired_replay: Mapping[str, object],
) -> dict[str, object]:
    return _build_gate1_record_fixture_validated(
        registration=validate_pre_gate1_registration(registration),
        split=split,
        paired_replay=paired_replay,
    )


def _build_gate1_record_fixture_validated(
    *,
    registration: Mapping[str, object],
    split: str,
    paired_replay: Mapping[str, object],
) -> dict[str, object]:
    registered = validate_pre_gate1_registration(registration)
    if split not in ("calibration", "evaluation"):
        raise ValueError("Gate 1 record artifact split must be calibration or evaluation")
    expected_ids = registered["window_manifest"]["artifact"]["splits"][split]
    replay = validate_gate1_paired_replay_artifact_for_test_only(
        paired_replay, registration=registered, expected_split=split
    )
    paired_index = {
        name: index for index, name in enumerate(GATE1_PAIRED_CANDIDATE_ORDER)
    }
    rows = []
    for fixture_row in replay["fixture_rows"]:
        row = fixture_row["fixture_source"]
        regrets = fixture_row["fixture_derived_detector_regret"]
        rows.append(
            {
                "window_id": fixture_row["fixture_window_id"],
                "candidate_names": list(GATE1_RECORD_CANDIDATE_ORDER),
                "detector_regret": [
                    regrets[paired_index[name]]
                    for name in GATE1_RECORD_CANDIDATE_ORDER
                ],
            }
        )
    rebuilt_rows = [
        _validate_gate1_record_row(row, window_id=window_id)
        for row, window_id in zip(rows, expected_ids)
    ]
    if [row["window_id"] for row in rebuilt_rows] != expected_ids:
        raise ValueError(f"Gate 1 {split} manifest IDs/order mismatch")
    fixture_rows = [
        {
            "fixture_window_id": row["window_id"],
            "fixture_candidate_names": row["candidate_names"],
            "fixture_detector_regret": row["detector_regret"],
        }
        for row in rebuilt_rows
    ]
    body: dict[str, object] = {
        "schema": GATE1_RECORD_FIXTURE_SCHEMA,
        "fixture_registration_sha256": registered["registration_sha256"],
        "fixture_split": split,
        "fixture_window_ids": list(expected_ids),
        "fixture_window_order_sha256": canonical_sha256(expected_ids),
        "fixture_candidate_names": list(GATE1_RECORD_CANDIDATE_ORDER),
        "fixture_candidate_order_sha256": canonical_sha256(GATE1_RECORD_CANDIDATE_ORDER),
        "fixture_paired_replay": replay,
        "fixture_paired_replay_artifact_sha256": replay["fixture_artifact_sha256"],
        "fixture_rows": fixture_rows,
    }
    body["fixture_artifact_sha256"] = canonical_sha256(body)
    return body


def validate_gate1_record_artifact(
    artifact: Mapping[str, object],
    *,
    registration: Mapping[str, object],
    expected_split: str,
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    registered = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if not isinstance(artifact, Mapping):
        raise TypeError("Gate 1 record artifact must be a mapping")
    if set(artifact) != _FORMAL_RECORD_FIELDS or artifact.get("schema") != GATE1_RECORD_SCHEMA:
        raise ValueError("formal Gate 1 record artifact fields/schema mismatch")
    if artifact.get("split") != expected_split:
        raise ValueError("Gate 1 record artifact split mismatch")
    rebuilt = _build_formal_gate1_record_from_serialized_replay(
        registration=registered,
        split=expected_split,
        paired_replay=artifact.get("paired_replay", {}),
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if rebuilt != dict(artifact):
        raise ValueError(f"Gate 1 {expected_split} artifact hash/manifest identity mismatch")
    return rebuilt


def validate_gate1_record_artifact_for_test_only(
    artifact: Mapping[str, object],
    *,
    registration: Mapping[str, object],
    expected_split: str,
) -> dict[str, object]:
    if not isinstance(artifact, Mapping):
        raise TypeError("Gate 1 record test fixture must be a mapping")
    if (
        set(artifact) != _FIXTURE_RECORD_FIELDS
        or artifact.get("schema") != GATE1_RECORD_FIXTURE_SCHEMA
    ):
        raise ValueError("Gate 1 record test-fixture fields/schema mismatch")
    if artifact.get("fixture_split") != expected_split:
        raise ValueError("Gate 1 record test-fixture split mismatch")
    rebuilt = _build_gate1_record_fixture_validated(
        registration=registration,
        split=expected_split,
        paired_replay=artifact.get("fixture_paired_replay", {}),
    )
    if rebuilt != dict(artifact):
        raise ValueError("Gate 1 record test-fixture hash/manifest identity mismatch")
    return rebuilt


def _record_artifact_mapping(
    artifact: Mapping[str, object],
    *,
    fixture: bool,
) -> dict[str, dict[str, float]]:
    if fixture:
        return {
            row["fixture_window_id"]: {
                name: float(value)
                for name, value in zip(
                    row["fixture_candidate_names"], row["fixture_detector_regret"]
                )
            }
            for row in artifact["fixture_rows"]
        }
    return {
        row["window_id"]: {
            name: float(value)
            for name, value in zip(row["candidate_names"], row["detector_regret"])
        }
        for row in artifact["rows"]
    }


def gate2_matched_transport(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260711,
) -> dict[str, object]:
    """Adjudicate matched TRANSPORT vs HOLD with window/seed hierarchy."""

    normalized = []
    for row in rows:
        seed = int(row["seed"])
        window = str(row["window_id"])
        period = int(row["period"])
        if period not in (2, 4, 8):
            raise ValueError("Gate 2 period must be 2, 4, or 8")
        normalized.append(
            {
                "seed": seed,
                "window": window,
                "period": period,
                "hold_regret": _finite(row["hold_regret"], "hold regret"),
                "transport_regret": _finite(row["transport_regret"], "transport regret"),
                "hold_mse": _finite(row["hold_mse"], "hold mse"),
                "transport_mse": _finite(row["transport_mse"], "transport mse"),
            }
        )
    windows = sorted({row["window"] for row in normalized})
    seeds = sorted({row["seed"] for row in normalized})
    expected = {(window, seed, period) for window in windows for seed in seeds for period in (2, 4, 8)}
    actual = {(row["window"], row["seed"], row["period"]) for row in normalized}
    if actual != expected or len(actual) != len(normalized):
        raise ValueError("Gate 2 requires one complete window×seed×period vector")
    by_key = {(row["window"], row["seed"], row["period"]): row for row in normalized}

    def means(sampled_windows: Sequence[str], sampled_seeds: Sequence[int]) -> tuple[float, float, float]:
        selected = [by_key[(window, seed, period)] for window in sampled_windows for seed in sampled_seeds for period in (2, 4, 8)]
        detector = _mean([row["hold_regret"] - row["transport_regret"] for row in selected])
        feature = _mean([row["hold_mse"] - row["transport_mse"] for row in selected])
        hold = _mean([row["hold_regret"] for row in selected])
        return detector, feature, hold

    detector, feature, hold = means(windows, seeds)
    rng = random.Random(int(bootstrap_seed))
    detector_boot, feature_boot = [], []
    for _ in range(int(bootstrap_samples)):
        sampled_windows = [rng.choice(windows) for _ in windows]
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        det, feat, _ = means(sampled_windows, sampled_seeds)
        detector_boot.append(det)
        feature_boot.append(feat)
    per_seed = {}
    for seed in seeds:
        det, feat, _ = means(windows, [seed])
        per_seed[str(seed)] = {"detector_improvement": det, "feature_improvement": feat}
    relative = float("nan") if hold <= 1e-12 else detector / hold
    detector_ci = _percentile_ci(detector_boot)
    feature_ci = _percentile_ci(feature_boot)
    hard = {
        "relative_reduction_ge_5pct": hold > 1e-12 and relative >= 0.05,
        "detector_ci_lower_gt_0": detector_ci[0] > 0.0,
        "feature_ci_lower_gt_0": feature_ci[0] > 0.0,
        "each_seed_nonnegative": all(
            value["detector_improvement"] >= 0 and value["feature_improvement"] >= 0
            for value in per_seed.values()
        ),
    }
    return {
        "schema": "chronotransport-r2-gate2-v1",
        "status": "PASS" if all(hard.values()) else "FAIL",
        "mechanism": bool(all(hard.values())),
        "detector_improvement": detector,
        "feature_improvement": feature,
        "detector_relative_reduction": relative,
        "detector_ci95": list(detector_ci),
        "feature_ci95": list(feature_ci),
        "per_seed": per_seed,
        "hard_conditions": hard,
    }
