"""Exact, recomputable Gate-1 unlock consumed by formal Stage B."""

from __future__ import annotations

from typing import Any, Mapping

from .adjudication import (
    gate1_oracle_headroom_from_profile,
    gate1_oracle_headroom_from_profile_for_test_only,
)
from .protocol import canonical_sha256
from .registration import (
    require_formal_gate1_context_arguments,
    validate_formal_gate1_context,
    validate_pre_gate1_registration,
)


GATE1_INPUT_SCHEMA = "chronotransport-r2-gate1-input-v3"
GATE1_UNLOCK_SCHEMA = "chronotransport-r2-gate1-unlock-v2"
GATE1_UNLOCK_FIXTURE_SCHEMA = "chronotransport-r2-gate1-unlock-test-fixture-v1"
_INPUT_FIELDS = {
    "schema",
    "registration",
    "calibration",
    "evaluation",
    "full_stack_profile",
}


def build_gate1_unlock_artifact(
    gate1_input: Mapping[str, Any],
    *,
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    require_formal_gate1_context_arguments(
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    return _build_gate1_unlock_artifact(
        gate1_input,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
        fixture=False,
    )


def build_gate1_unlock_artifact_for_test_only(
    gate1_input: Mapping[str, Any],
) -> dict[str, Any]:
    return _build_gate1_unlock_artifact(
        gate1_input,
        repository_root=None,
        registration_commit=None,
        registration_relpath=None,
        fixture=True,
    )


def _build_gate1_unlock_artifact(
    gate1_input: Mapping[str, Any],
    *,
    repository_root: str | None,
    registration_commit: str | None,
    registration_relpath: str | None,
    fixture: bool,
) -> dict[str, Any]:
    if not isinstance(gate1_input, Mapping) or set(gate1_input) != _INPUT_FIELDS:
        raise ValueError("Gate 1 input artifact fields mismatch")
    if gate1_input["schema"] != GATE1_INPUT_SCHEMA:
        raise ValueError("unsupported Gate 1 input schema")
    if fixture:
        registration = validate_pre_gate1_registration(gate1_input["registration"])
    else:
        registration = validate_formal_gate1_context(
            gate1_input["registration"],
            repository_root=repository_root,
            registration_commit=registration_commit,
            registration_relpath=registration_relpath,
        )
    profile = gate1_input["full_stack_profile"]
    if not isinstance(profile, Mapping):
        raise ValueError("Gate 1 input requires full_stack_profile artifact")
    if fixture:
        result = gate1_oracle_headroom_from_profile_for_test_only(
            registration=registration,
            calibration=gate1_input["calibration"],
            evaluation=gate1_input["evaluation"],
            full_stack_profile=profile,
        )
    else:
        result = gate1_oracle_headroom_from_profile(
            registration=registration,
            calibration=gate1_input["calibration"],
            evaluation=gate1_input["evaluation"],
            full_stack_profile=profile,
            repository_root=repository_root,
            registration_commit=registration_commit,
            registration_relpath=registration_relpath,
        )
    body: dict[str, Any] = {
        "schema": GATE1_UNLOCK_FIXTURE_SCHEMA if fixture else GATE1_UNLOCK_SCHEMA,
        "registration_sha256": registration["registration_sha256"],
        "gate1_input": dict(gate1_input),
        "gate1_input_sha256": canonical_sha256(gate1_input),
        "profile_sha256": result["full_stack_profile_sha256"],
        "calibration_artifact_sha256": result["calibration_artifact_sha256"],
        "evaluation_artifact_sha256": result["evaluation_artifact_sha256"],
        "gate1_result": result,
        "gate1_result_sha256": canonical_sha256(result),
        "status": result["status"],
        "oracle_headroom": result["oracle_headroom"],
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def validate_gate1_unlock_artifact(
    artifact: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    repository_root: str,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    require_formal_gate1_context_arguments(
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if not isinstance(artifact, Mapping) or artifact.get("schema") != GATE1_UNLOCK_SCHEMA:
        raise ValueError("Gate 1 unlock artifact schema mismatch")
    validated_registration = validate_formal_gate1_context(
        registration,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    rebuilt = _build_gate1_unlock_artifact_from_validated_payload(
        artifact.get("gate1_input", {}),
        fixture=False,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if rebuilt != dict(artifact):
        raise ValueError("Gate 1 unlock artifact is not an exact recomputable artifact")
    if rebuilt["registration_sha256"] != validated_registration["registration_sha256"]:
        raise ValueError("Gate 1 unlock registration identity mismatch")
    if rebuilt["status"] != "PASS" or rebuilt["oracle_headroom"] is not True:
        raise ValueError("Stage B requires a PASS Gate 1 unlock")
    return rebuilt


def validate_gate1_unlock_artifact_for_test_only(
    artifact: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema") != GATE1_UNLOCK_FIXTURE_SCHEMA
    ):
        raise ValueError("Gate 1 unlock test-fixture schema mismatch")
    validated_registration = validate_pre_gate1_registration(registration)
    rebuilt = _build_gate1_unlock_artifact_from_validated_payload(
        artifact.get("gate1_input", {}),
        fixture=True,
        repository_root=None,
        registration_commit=None,
        registration_relpath=None,
    )
    if rebuilt != dict(artifact):
        raise ValueError("Gate 1 unlock test fixture is not an exact recomputable artifact")
    if rebuilt["registration_sha256"] != validated_registration["registration_sha256"]:
        raise ValueError("Gate 1 unlock test registration identity mismatch")
    if rebuilt["status"] != "PASS" or rebuilt["oracle_headroom"] is not True:
        raise ValueError("test fixture requires a PASS Gate 1 unlock")
    return rebuilt


def _build_gate1_unlock_artifact_from_validated_payload(
    gate1_input: Mapping[str, Any],
    *,
    fixture: bool,
    repository_root: str | None,
    registration_commit: str | None,
    registration_relpath: str | None,
) -> dict[str, Any]:
    """Recompute a persisted artifact without exposing a formal minting API."""

    return _build_gate1_unlock_artifact(
        gate1_input,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
        fixture=fixture,
    )
