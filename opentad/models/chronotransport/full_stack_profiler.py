"""Registration-driven direct full-invocation cost evidence for r2 Gate 1."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .cost_lookup import CostLookupKey, ScheduleCostLookup, validate_execution_cost_ledger_entry
from .environment import (
    OBSERVED_PROVENANCE_FIELDS,
    build_test_only_observed_environment,
    observed_environment_from_provenance,
    observed_environment_to_provenance,
)
from .profiler import REQUIRED_STAGE_FIELDS
from .protocol import canonical_sha256
from .registration import (
    EXPECTED_PROFILE_CANDIDATE_ORDER,
    REGISTERED_PROFILE_BACKEND_IDENTITY,
    REGISTERED_PROFILE_BACKEND_SOURCE,
    validate_formal_gate1_context,
    validate_pre_gate1_registration,
)


PROFILE_ARTIFACT_SCHEMA = "chronotransport-r2-full-stack-profile-formal-v4"
PROFILE_FIXTURE_ARTIFACT_SCHEMA = (
    "chronotransport-r2-full-stack-profile-test-fixture-v2"
)
_FORMAL_CANDIDATE_SCHEMA = "chronotransport-r2-candidate-full-stack-profile-v3"
_FIXTURE_CANDIDATE_SCHEMA = (
    "chronotransport-r2-candidate-full-stack-profile-test-fixture-v2"
)
_PROVENANCE_FIELDS = {
    "environment_sha256",
    "source_commit",
    "spec_sha256",
    "config_sha256",
    "checkpoint_sha256",
    "library_sha256",
    "registration_sha256",
    "candidate_name",
    "candidate_identity_sha256",
    "factory_identity",
    "factory_config_sha256",
    "backend_identity",
    "backend_source_sha256",
    "invocation_order_sha256",
    "requested_action_sha256",
    "executed_action_sha256",
    "selected_rows_per_group",
} | set(OBSERVED_PROVENANCE_FIELDS)
_INVOCATION_FIELDS = {
    "invocation_index",
    "invocation_id",
    "total_ms",
    "diagnostic_ms",
    "peak_gpu_memory_bytes",
    "cost_ledger",
    "execution_provenance",
}
_FORMAL_CANDIDATE_FIELDS = {
    "schema",
    "provenance",
    "cost_lookup_key",
    "warmup_count",
    "sample_count",
    "invocation_ids",
    "invocation_order_sha256",
    "requested_action_order_sha256",
    "executed_action_order_sha256",
    "execution_provenance_order_sha256",
    "total_ms",
    "diagnostic_ms",
    "peak_gpu_memory_bytes",
    "invocations",
    "candidate_profile_sha256",
}
_FIXTURE_INVOCATION_FIELDS = {
    "fixture_sequence",
    "fixture_measurement",
    "fixture_cost_ledger",
    "fixture_execution_provenance",
}
_FIXTURE_CANDIDATE_FIELDS = {
    "schema",
    "fixture_candidate_name",
    "fixture_provenance",
    "fixture_cost_lookup_key",
    "fixture_warmup_count",
    "fixture_sample_count",
    "fixture_invocation_ids",
    "fixture_invocation_order_sha256",
    "fixture_requested_action_order_sha256",
    "fixture_executed_action_order_sha256",
    "fixture_execution_provenance_order_sha256",
    "fixture_total_ms",
    "fixture_diagnostic_ms",
    "fixture_peak_gpu_memory_bytes",
    "fixture_invocations",
    "fixture_candidate_sha256",
}
_FORMAL_ARTIFACT_FIELDS = {
    "schema",
    "registration_sha256",
    "common_identity",
    "candidate_order",
    "candidate_order_sha256",
    "invocation_ids",
    "invocation_order_sha256",
    "invocation_protocol",
    "cost_lookup",
    "candidates",
    "profile_sha256",
}
_FIXTURE_ARTIFACT_FIELDS = {
    "schema",
    "fixture_registration_sha256",
    "fixture_common_identity",
    "fixture_candidate_order",
    "fixture_candidate_order_sha256",
    "fixture_invocation_ids",
    "fixture_invocation_order_sha256",
    "fixture_invocation_protocol",
    "fixture_cost_lookup",
    "fixture_candidates",
    "fixture_profile_sha256",
}
_FIXED_BACKEND_RESULT_FIELDS = {
    "diagnostic_ms",
    "requested_action_payload",
    "executed_schedule_name",
    "executed_action_payload",
    "repair_count",
    "nan_fallback",
    "whole_window_dense_fallback",
    "safety_override_budget_violation",
    "execution_provenance",
}
_EXECUTION_PROVENANCE_FIELDS = {
    "backend_identity",
    "backend_source_sha256",
    "deploy_visible_signal_sha256",
    "requested_action_sha256",
    "executed_action_sha256",
}


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a measured number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires direct samples")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "samples": [float(value) for value in values],
    }


def _candidate_plan(registration: Mapping[str, object], name: str) -> Mapping[str, object]:
    for plan in registration["profiler"]["candidate_plan"]:
        if plan["candidate_name"] == name:
            return plan
    raise ValueError(f"registered profile candidate is missing: {name}")


def registered_candidate_provenance_for_test_only(
    registration: Mapping[str, object], candidate_name: str
) -> dict[str, object]:
    """Derive deterministic non-formal provenance for isolated test fixtures."""

    validated = validate_pre_gate1_registration(registration)
    return _registered_candidate_provenance_from_validated(
        validated,
        candidate_name,
        observed_environment=build_test_only_observed_environment(
            validated["environment"]
        ),
    )


def _registered_candidate_provenance_from_validated(
    validated: Mapping[str, object],
    candidate_name: str,
    *,
    observed_environment: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(candidate_name, str) or candidate_name not in EXPECTED_PROFILE_CANDIDATE_ORDER:
        raise ValueError("candidate_name is not in the frozen 23-candidate profile order")
    plan = _candidate_plan(validated, candidate_name)
    environment = validated["environment"]
    observed = observed_environment_to_provenance(
        observed_environment,
        required_environment=environment,
    )
    action_order_sha = plan["requested_action_order_sha256"]
    provenance: dict[str, object] = {
        **observed,
        "environment_sha256": environment["environment_sha256"],
        "source_commit": validated["implementation_commit"],
        "spec_sha256": validated["spec"]["sha256"],
        "config_sha256": validated["profiler"]["model_config_sha256"],
        "checkpoint_sha256": validated["dense_checkpoint"]["sha256"],
        "library_sha256": validated["candidate_library"]["library_sha256"],
        "registration_sha256": validated["registration_sha256"],
        "candidate_name": candidate_name,
        "candidate_identity_sha256": plan["candidate_identity_sha256"],
        "factory_identity": plan["factory_identity"],
        "factory_config_sha256": plan["factory_config_sha256"],
        "backend_identity": REGISTERED_PROFILE_BACKEND_IDENTITY,
        "backend_source_sha256": validated["source_files"][REGISTERED_PROFILE_BACKEND_SOURCE],
        "invocation_order_sha256": validated["profiler"]["invocation_order_sha256"],
        "requested_action_sha256": action_order_sha,
        "executed_action_sha256": action_order_sha,
        "selected_rows_per_group": list(plan["selected_rows_per_group"]),
    }
    if set(provenance) != _PROVENANCE_FIELDS:
        raise RuntimeError("internal formal provenance fields mismatch")
    CostLookupKey.from_provenance(provenance).validate_formal()
    return provenance


def _validate_invocation(
    row: Mapping[str, object],
    *,
    index: int,
    expected_id: str,
    expected_action_sha256: str,
    provenance: Mapping[str, object],
) -> None:
    if not isinstance(row, Mapping) or set(row) != _INVOCATION_FIELDS:
        raise ValueError("full-stack invocation fields mismatch")
    if type(row["invocation_index"]) is not int or row["invocation_index"] != index:
        raise ValueError("full-stack invocation indices must be contiguous and canonical")
    if not isinstance(row["invocation_id"], str) or row["invocation_id"] != expected_id:
        raise ValueError("full-stack invocation IDs/order mismatch registered plan")
    _number(row["total_ms"], "total_ms")
    diagnostics = row["diagnostic_ms"]
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(REQUIRED_STAGE_FIELDS):
        missing = sorted(set(REQUIRED_STAGE_FIELDS) - set(diagnostics or {}))
        extra = sorted(set(diagnostics or {}) - set(REQUIRED_STAGE_FIELDS))
        raise ValueError(f"diagnostic stages mismatch; missing={missing}, extra={extra}")
    for name in REQUIRED_STAGE_FIELDS:
        _number(diagnostics[name], name)
    peak = row["peak_gpu_memory_bytes"]
    if type(peak) is not int or peak < 0:
        raise ValueError("peak_gpu_memory_bytes must be a non-negative integer")
    ledger = row["cost_ledger"]
    if not isinstance(ledger, Mapping):
        raise ValueError("each invocation requires exact requested/executed cost ledger")
    validate_execution_cost_ledger_entry(ledger, formal=True)
    if ledger["actual_total_ms"] != row["total_ms"]:
        raise ValueError("cost ledger actual_total_ms must equal direct invocation total_ms")
    if ledger["requested_schedule_name"] != provenance["candidate_name"]:
        raise ValueError("requested schedule name does not match registered candidate")
    if ledger["executed_schedule_name"] != provenance["candidate_name"]:
        raise ValueError("executed schedule name does not match registered candidate")
    if ledger["requested_action_sha256"] != expected_action_sha256:
        raise ValueError("requested action hash does not match registered invocation action")
    if ledger["executed_action_sha256"] != expected_action_sha256:
        raise ValueError("executed action hash does not match registered invocation action")
    execution = row["execution_provenance"]
    if not isinstance(execution, Mapping) or set(execution) != _EXECUTION_PROVENANCE_FIELDS:
        raise ValueError("full-stack execution provenance fields mismatch")
    if execution["backend_identity"] != provenance["backend_identity"]:
        raise ValueError("full-stack execution provenance backend identity mismatch")
    if execution["backend_source_sha256"] != provenance["backend_source_sha256"]:
        raise ValueError("full-stack execution provenance backend source mismatch")
    if (
        execution["requested_action_sha256"] != ledger["requested_action_sha256"]
        or execution["executed_action_sha256"] != ledger["executed_action_sha256"]
    ):
        raise ValueError("full-stack execution provenance action digest mismatch")
    signal_sha = execution["deploy_visible_signal_sha256"]
    if str(provenance["candidate_name"]).startswith("motion_topk_p"):
        if not isinstance(signal_sha, str) or len(signal_sha) != 64 or any(
            character not in "0123456789abcdef" for character in signal_sha
        ):
            raise ValueError("motion profile requires a deploy-visible signal SHA-256")
    elif signal_sha is not None:
        raise ValueError("non-motion profile cannot claim deploy-visible signal provenance")


def _candidate_measurements(
    *,
    validated: Mapping[str, object],
    candidate_name: str,
    warmup_count: object,
    invocations: object,
    serialized_provenance: object | None = None,
    test_only: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    if test_only:
        if serialized_provenance is not None:
            raise ValueError("test-only provenance is repository-derived")
        observed_environment = build_test_only_observed_environment(
            validated["environment"]
        )
    else:
        if not isinstance(serialized_provenance, Mapping):
            raise ValueError("formal candidate lacks observed allocation provenance")
        observed_environment = observed_environment_from_provenance(
            serialized_provenance,
            required_environment=validated["environment"],
        )
    provenance = _registered_candidate_provenance_from_validated(
        validated,
        candidate_name,
        observed_environment=observed_environment,
    )
    if serialized_provenance is not None and provenance != dict(serialized_provenance):
        raise ValueError("formal candidate provenance differs from registered live identity")
    if type(warmup_count) is not int or warmup_count != 50:
        raise ValueError("formal full-stack profile requires exactly 50 warm-up invocations")
    if not isinstance(invocations, (list, tuple)) or len(invocations) != 200:
        raise ValueError("formal full-stack profile requires exactly 200 measured invocations")
    invocation_ids = validated["profiler"]["invocation_ids"]
    action_hashes = _candidate_plan(validated, candidate_name)[
        "requested_action_sha256_by_invocation"
    ]
    rows = [dict(row) for row in invocations]
    for index, row in enumerate(rows):
        _validate_invocation(
            row,
            index=index,
            expected_id=invocation_ids[index],
            expected_action_sha256=action_hashes[index],
            provenance=provenance,
        )
    total_values = [_number(row["total_ms"], "total_ms") for row in rows]
    measured_p50 = _percentile(total_values, 0.50)
    for row in rows:
        ledger = row["cost_ledger"]
        if _number(ledger["requested_cost_p50"], "requested_cost_p50") != measured_p50:
            raise ValueError("requested cost must equal this candidate's direct measured p50")
        if _number(ledger["executed_lookup_cost_p50"], "executed_lookup_cost_p50") != measured_p50:
            raise ValueError("executed lookup cost must equal this candidate's direct measured p50")
    summary = {
        "total_ms": _distribution(total_values),
        "diagnostic_ms": {
            name: _distribution([_number(row["diagnostic_ms"][name], name) for row in rows])
            for name in REQUIRED_STAGE_FIELDS
        },
        "peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in rows),
        "invocation_ids": list(invocation_ids),
        "action_hashes": list(action_hashes),
    }
    return provenance, rows, summary


def _formal_candidate_from_serialized(
    candidate: Mapping[str, object],
    *,
    validated: Mapping[str, object],
    candidate_name: str,
) -> dict[str, object]:
    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != _FORMAL_CANDIDATE_FIELDS
        or candidate.get("schema") != _FORMAL_CANDIDATE_SCHEMA
    ):
        raise ValueError("formal candidate full-stack profile fields/schema mismatch")
    provenance, rows, summary = _candidate_measurements(
        validated=validated,
        candidate_name=candidate_name,
        warmup_count=candidate.get("warmup_count"),
        invocations=candidate.get("invocations"),
        serialized_provenance=candidate.get("provenance"),
    )
    action_hashes = summary.pop("action_hashes")
    invocation_ids = summary.pop("invocation_ids")
    body: dict[str, object] = {
        "schema": _FORMAL_CANDIDATE_SCHEMA,
        "provenance": provenance,
        "cost_lookup_key": CostLookupKey.from_provenance(provenance).encode(),
        "warmup_count": 50,
        "sample_count": len(rows),
        "invocation_ids": invocation_ids,
        "invocation_order_sha256": canonical_sha256(invocation_ids),
        "requested_action_order_sha256": canonical_sha256(action_hashes),
        "executed_action_order_sha256": canonical_sha256(
            [row["cost_ledger"]["executed_action_sha256"] for row in rows]
        ),
        "execution_provenance_order_sha256": canonical_sha256(
            [row["execution_provenance"] for row in rows]
        ),
        **summary,
        "invocations": rows,
    }
    body["candidate_profile_sha256"] = canonical_sha256(body)
    if body != dict(candidate):
        raise ValueError("formal candidate full-stack profile hashes/statistics do not validate")
    return body


def _fixture_invocation_from_source(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "fixture_sequence": {
            "index": row["invocation_index"],
            "id": row["invocation_id"],
        },
        "fixture_measurement": {
            "total_ms": row["total_ms"],
            "diagnostic_ms": dict(row["diagnostic_ms"]),
            "peak_gpu_memory_bytes": row["peak_gpu_memory_bytes"],
        },
        "fixture_cost_ledger": dict(row["cost_ledger"]),
        "fixture_execution_provenance": dict(row["execution_provenance"]),
    }


def _source_invocation_from_fixture(row: object) -> dict[str, object]:
    if not isinstance(row, Mapping) or set(row) != _FIXTURE_INVOCATION_FIELDS:
        raise ValueError("test profile fixture invocation fields mismatch")
    sequence = row["fixture_sequence"]
    measurement = row["fixture_measurement"]
    if not isinstance(sequence, Mapping) or set(sequence) != {"index", "id"}:
        raise ValueError("test profile fixture invocation sequence fields mismatch")
    if not isinstance(measurement, Mapping) or set(measurement) != {
        "total_ms",
        "diagnostic_ms",
        "peak_gpu_memory_bytes",
    }:
        raise ValueError("test profile fixture measurement fields mismatch")
    return {
        "invocation_index": sequence["index"],
        "invocation_id": sequence["id"],
        "total_ms": measurement["total_ms"],
        "diagnostic_ms": measurement["diagnostic_ms"],
        "peak_gpu_memory_bytes": measurement["peak_gpu_memory_bytes"],
        "cost_ledger": row["fixture_cost_ledger"],
        "execution_provenance": row["fixture_execution_provenance"],
    }


def _build_fixture_candidate(
    *,
    validated: Mapping[str, object],
    candidate_name: str,
    warmup_count: object,
    invocations: object,
) -> dict[str, object]:
    provenance, rows, summary = _candidate_measurements(
        validated=validated,
        candidate_name=candidate_name,
        warmup_count=warmup_count,
        invocations=invocations,
        test_only=True,
    )
    action_hashes = summary.pop("action_hashes")
    invocation_ids = summary.pop("invocation_ids")
    fixture_rows = [_fixture_invocation_from_source(row) for row in rows]
    body: dict[str, object] = {
        "schema": _FIXTURE_CANDIDATE_SCHEMA,
        "fixture_candidate_name": candidate_name,
        "fixture_provenance": provenance,
        "fixture_cost_lookup_key": CostLookupKey.from_provenance(provenance).encode(),
        "fixture_warmup_count": 50,
        "fixture_sample_count": len(rows),
        "fixture_invocation_ids": invocation_ids,
        "fixture_invocation_order_sha256": canonical_sha256(invocation_ids),
        "fixture_requested_action_order_sha256": canonical_sha256(action_hashes),
        "fixture_executed_action_order_sha256": canonical_sha256(
            [row["cost_ledger"]["executed_action_sha256"] for row in rows]
        ),
        "fixture_execution_provenance_order_sha256": canonical_sha256(
            [row["execution_provenance"] for row in rows]
        ),
        "fixture_total_ms": summary["total_ms"],
        "fixture_diagnostic_ms": summary["diagnostic_ms"],
        "fixture_peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
        "fixture_invocations": fixture_rows,
    }
    body["fixture_candidate_sha256"] = canonical_sha256(body)
    return body


def _fixture_candidate_from_serialized(
    candidate: Mapping[str, object],
    *,
    validated: Mapping[str, object],
    candidate_name: str,
) -> dict[str, object]:
    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != _FIXTURE_CANDIDATE_FIELDS
        or candidate.get("schema") != _FIXTURE_CANDIDATE_SCHEMA
        or candidate.get("fixture_candidate_name") != candidate_name
    ):
        raise ValueError("test candidate full-stack profile fields/schema mismatch")
    rows = [
        _source_invocation_from_fixture(row)
        for row in candidate.get("fixture_invocations", ())
    ]
    rebuilt = _build_fixture_candidate(
        validated=validated,
        candidate_name=candidate_name,
        warmup_count=candidate.get("fixture_warmup_count"),
        invocations=rows,
    )
    if rebuilt != dict(candidate):
        raise ValueError("test candidate full-stack profile hashes/statistics do not validate")
    return rebuilt


def build_candidate_full_stack_profile_for_test_only(
    *,
    registration: Mapping[str, object],
    candidate_name: str,
    warmup_count: int,
    invocations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return _build_fixture_candidate(
        validated=validate_pre_gate1_registration(registration),
        candidate_name=candidate_name,
        warmup_count=warmup_count,
        invocations=invocations,
    )


def _identity_without_candidate(provenance: Mapping[str, object]) -> dict[str, object]:
    excluded = {
        "candidate_name",
        "candidate_identity_sha256",
        "factory_identity",
        "factory_config_sha256",
        "backend_identity",
        "backend_source_sha256",
        "requested_action_sha256",
        "executed_action_sha256",
        "selected_rows_per_group",
    }
    return {key: value for key, value in provenance.items() if key not in excluded}


def _invocation_protocol() -> dict[str, object]:
    return {
        "timer_boundary": "before_data_decode_to_after_postprocess",
        "cuda_synchronize_before_after": True,
        "warmup_per_candidate": 50,
        "measured_windows_per_candidate": 200,
        "budget_statistic": "direct_total_ms_p50",
    }


def _formal_common_from_serialized_candidates(
    candidates: object, *, validated: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(candidates, list) or len(candidates) != 23:
        raise ValueError("formal full-stack profile requires the exact frozen 23-candidate order")
    actual_names = [
        candidate.get("provenance", {}).get("candidate_name")
        if isinstance(candidate, Mapping)
        else None
        for candidate in candidates
    ]
    if tuple(actual_names) != EXPECTED_PROFILE_CANDIDATE_ORDER:
        raise ValueError("formal full-stack profile requires the exact frozen 23-candidate order")
    rebuilt = [
        _formal_candidate_from_serialized(
            candidate, validated=validated, candidate_name=name
        )
        for name, candidate in zip(EXPECTED_PROFILE_CANDIDATE_ORDER, candidates)
    ]
    common_identity = None
    for profile in rebuilt:
        identity = _identity_without_candidate(profile["provenance"])
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError("all formal candidate profiles must share registered identity")
    lookup_entries = [
        (
            CostLookupKey.from_provenance(profile["provenance"]),
            profile["total_ms"]["p50"],
            profile["total_ms"]["p95"],
        )
        for profile in rebuilt
    ]
    invocation_ids = validated["profiler"]["invocation_ids"]
    return {
        "registration_sha256": validated["registration_sha256"],
        "common_identity": common_identity,
        "candidate_order": list(EXPECTED_PROFILE_CANDIDATE_ORDER),
        "candidate_order_sha256": canonical_sha256(EXPECTED_PROFILE_CANDIDATE_ORDER),
        "invocation_ids": list(invocation_ids),
        "invocation_order_sha256": canonical_sha256(invocation_ids),
        "invocation_protocol": _invocation_protocol(),
        "cost_lookup": ScheduleCostLookup.payload(lookup_entries),
        "candidates": rebuilt,
    }


def _fixture_common_from_serialized_candidates(
    candidates: object, *, validated: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(candidates, (list, tuple)) or len(candidates) != 23:
        raise ValueError("test full-stack profile requires the exact frozen 23-candidate order")
    actual_names = [
        candidate.get("fixture_candidate_name")
        if isinstance(candidate, Mapping)
        else None
        for candidate in candidates
    ]
    if tuple(actual_names) != EXPECTED_PROFILE_CANDIDATE_ORDER:
        raise ValueError("test full-stack profile requires the exact frozen 23-candidate order")
    rebuilt = [
        _fixture_candidate_from_serialized(
            candidate, validated=validated, candidate_name=name
        )
        for name, candidate in zip(EXPECTED_PROFILE_CANDIDATE_ORDER, candidates)
    ]
    common_identity = None
    for profile in rebuilt:
        identity = _identity_without_candidate(profile["fixture_provenance"])
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError("all test candidate profiles must share registered identity")
    lookup_entries = [
        (
            CostLookupKey.from_provenance(profile["fixture_provenance"]),
            profile["fixture_total_ms"]["p50"],
            profile["fixture_total_ms"]["p95"],
        )
        for profile in rebuilt
    ]
    invocation_ids = validated["profiler"]["invocation_ids"]
    return {
        "fixture_registration_sha256": validated["registration_sha256"],
        "fixture_common_identity": common_identity,
        "fixture_candidate_order": list(EXPECTED_PROFILE_CANDIDATE_ORDER),
        "fixture_candidate_order_sha256": canonical_sha256(EXPECTED_PROFILE_CANDIDATE_ORDER),
        "fixture_invocation_ids": list(invocation_ids),
        "fixture_invocation_order_sha256": canonical_sha256(invocation_ids),
        "fixture_invocation_protocol": _invocation_protocol(),
        "fixture_cost_lookup": ScheduleCostLookup.payload(lookup_entries),
        "fixture_candidates": rebuilt,
    }


def build_full_stack_profile_artifact(
    *,
    registration: Mapping[str, object],
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    """Run the sole exact repository-owned OpenTAD profile session."""

    validated = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    from tools.bata.chronotransport_r2_profile_factory import build_registered_profile_session

    artifact = build_registered_profile_session(validated).run_fixed_profile()
    return _validate_serialized_full_stack_profile_artifact(
        artifact, registration=validated
    )


def build_full_stack_profile_artifact_for_test_only(
    candidates: Sequence[Mapping[str, object]],
    *,
    registration: Mapping[str, object],
) -> dict[str, object]:
    common = _fixture_common_from_serialized_candidates(
        candidates, validated=validate_pre_gate1_registration(registration)
    )
    body = {"schema": PROFILE_FIXTURE_ARTIFACT_SCHEMA, **common}
    body["fixture_profile_sha256"] = canonical_sha256(body)
    return body


def build_full_stack_profile_from_invocations_for_test_only(
    *,
    registration: Mapping[str, object],
    invocations_by_candidate: Mapping[str, Sequence[Mapping[str, object]]],
    warmup_count: int = 50,
) -> dict[str, object]:
    validated = validate_pre_gate1_registration(registration)
    if not isinstance(invocations_by_candidate, Mapping) or tuple(
        invocations_by_candidate
    ) != EXPECTED_PROFILE_CANDIDATE_ORDER:
        raise ValueError("full-stack invocation mapping must use the frozen candidate order")
    candidates = [
        _build_fixture_candidate(
            validated=validated,
            candidate_name=name,
            warmup_count=warmup_count,
            invocations=invocations_by_candidate[name],
        )
        for name in EXPECTED_PROFILE_CANDIDATE_ORDER
    ]
    common = _fixture_common_from_serialized_candidates(candidates, validated=validated)
    body = {"schema": PROFILE_FIXTURE_ARTIFACT_SCHEMA, **common}
    body["fixture_profile_sha256"] = canonical_sha256(body)
    return body


def _validate_serialized_full_stack_profile_artifact(
    artifact: Mapping[str, object], *, registration: Mapping[str, object]
) -> dict[str, object]:
    if (
        not isinstance(artifact, Mapping)
        or set(artifact) != _FORMAL_ARTIFACT_FIELDS
        or artifact.get("schema") != PROFILE_ARTIFACT_SCHEMA
    ):
        raise ValueError("formal full-stack profile artifact schema/fields mismatch")
    common = _formal_common_from_serialized_candidates(
        artifact.get("candidates"),
        validated=validate_pre_gate1_registration(registration),
    )
    rebuilt = {"schema": PROFILE_ARTIFACT_SCHEMA, **common}
    rebuilt["profile_sha256"] = canonical_sha256(rebuilt)
    if rebuilt != dict(artifact):
        raise ValueError("formal full-stack profile artifact hash/order/identity mismatch")
    return rebuilt


def validate_full_stack_profile_artifact(
    artifact: Mapping[str, object],
    *,
    registration: Mapping[str, object],
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    validated = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    return _validate_serialized_full_stack_profile_artifact(
        artifact, registration=validated
    )


def validate_full_stack_profile_artifact_for_test_only(
    artifact: Mapping[str, object], *, registration: Mapping[str, object]
) -> dict[str, object]:
    if (
        not isinstance(artifact, Mapping)
        or set(artifact) != _FIXTURE_ARTIFACT_FIELDS
        or artifact.get("schema") != PROFILE_FIXTURE_ARTIFACT_SCHEMA
    ):
        raise ValueError("test full-stack profile artifact fields/schema mismatch")
    common = _fixture_common_from_serialized_candidates(
        artifact.get("fixture_candidates"),
        validated=validate_pre_gate1_registration(registration),
    )
    rebuilt = {"schema": PROFILE_FIXTURE_ARTIFACT_SCHEMA, **common}
    rebuilt["fixture_profile_sha256"] = canonical_sha256(rebuilt)
    if rebuilt != dict(artifact):
        raise ValueError("test full-stack profile artifact hash/order/identity mismatch")
    return rebuilt


def _validate_fixed_backend_result(
    result: object,
    *,
    candidate_name: str,
    expected_action_sha256: str,
    expected_backend_source_sha256: str,
) -> dict[str, object]:
    if not isinstance(result, Mapping) or set(result) != _FIXED_BACKEND_RESULT_FIELDS:
        raise ValueError("fixed backend result fields mismatch")
    if canonical_sha256(result["requested_action_payload"]) != expected_action_sha256:
        raise ValueError("fixed backend requested action payload does not match registration")
    if canonical_sha256(result["executed_action_payload"]) != expected_action_sha256:
        raise ValueError("fixed backend executed action payload does not match registration")
    if result["executed_schedule_name"] != candidate_name:
        raise ValueError("fixed backend executed schedule does not match registration")
    if type(result["repair_count"]) is not int:
        raise TypeError("fixed backend repair_count must be an integer")
    for field in (
        "nan_fallback",
        "whole_window_dense_fallback",
        "safety_override_budget_violation",
    ):
        if type(result[field]) is not bool:
            raise TypeError(f"fixed backend {field} must be boolean")
    if (
        result["repair_count"] != 0
        or result["nan_fallback"]
        or result["whole_window_dense_fallback"]
        or result["safety_override_budget_violation"]
    ):
        raise ValueError("formal fixed-backend execution contains repair or fallback")
    diagnostics = result["diagnostic_ms"]
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(REQUIRED_STAGE_FIELDS):
        raise ValueError("fixed backend diagnostic stages mismatch")
    for field in REQUIRED_STAGE_FIELDS:
        _number(diagnostics[field], field)
    execution = result["execution_provenance"]
    if not isinstance(execution, Mapping) or set(execution) != _EXECUTION_PROVENANCE_FIELDS:
        raise ValueError("fixed backend execution provenance fields mismatch")
    if (
        execution["backend_identity"] != REGISTERED_PROFILE_BACKEND_IDENTITY
        or execution["backend_source_sha256"] != expected_backend_source_sha256
    ):
        raise ValueError("fixed backend execution provenance does not bind registered source")
    requested_sha = canonical_sha256(result["requested_action_payload"])
    executed_sha = canonical_sha256(result["executed_action_payload"])
    if (
        execution["requested_action_sha256"] != requested_sha
        or execution["executed_action_sha256"] != executed_sha
    ):
        raise ValueError("fixed backend execution provenance action digest mismatch")
    signal_sha = execution["deploy_visible_signal_sha256"]
    if candidate_name.startswith("motion_topk_p"):
        if not isinstance(signal_sha, str) or len(signal_sha) != 64:
            raise ValueError("motion fixed backend must expose deploy-visible signal provenance")
    elif signal_sha is not None:
        raise ValueError("non-motion fixed backend cannot expose motion signal provenance")
    return dict(result)
