"""Immutable pre-Gate1 registration schema and claim-state helpers."""

from __future__ import annotations

from typing import Any, Mapping

from .protocol import R2_PROTOCOL_ID, canonical_sha256


REGISTRATION_SCHEMA = "chronotransport-r2-pre-gate1-registration-v1"
REQUIRED_FIELDS = (
    "protocol_id",
    "spec",
    "implementation_commit",
    "registration_parent",
    "source_files",
    "upstream_commits",
    "dense_checkpoint",
    "data",
    "window_manifest",
    "candidate_library",
    "exposures",
    "controls",
    "bootstrap",
    "profiler",
    "gates",
    "environment",
    "output_root",
    "attestation",
)
FORBIDDEN_KEY_FRAGMENTS = ("result", "evaluation_output", "replay_output", "gate_report", "profile_output")


def _audit_no_results(value: Any, path: str = "registration") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized != "result_data_unread" and any(
                fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                raise ValueError(f"registration contains forbidden result-derived key: {path}.{key}")
            _audit_no_results(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _audit_no_results(item, f"{path}[{index}]")


def build_pre_gate1_registration(identity: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(identity)
    missing = [field for field in REQUIRED_FIELDS if field not in body]
    if missing:
        raise ValueError(f"registration identity is missing required fields: {missing}")
    if body["protocol_id"] != R2_PROTOCOL_ID:
        raise ValueError("registration protocol_id mismatch")
    attestation = body.get("attestation")
    if not isinstance(attestation, Mapping) or attestation.get("result_data_unread") is not True:
        raise ValueError("registration requires result_data_unread=true attestation")
    _audit_no_results(body)
    registration = {"schema": REGISTRATION_SCHEMA, **body}
    registration["registration_sha256"] = canonical_sha256(registration)
    return registration


def validate_pre_gate1_registration(registration: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(registration)
    digest = payload.pop("registration_sha256", None)
    if payload.pop("schema", None) != REGISTRATION_SCHEMA:
        raise ValueError("unsupported registration schema")
    rebuilt = build_pre_gate1_registration(payload)
    if digest != rebuilt["registration_sha256"]:
        raise ValueError("registration SHA-256 mismatch")
    return rebuilt


def claim_flags(*, gate1: bool = False, gate2: bool = False, gate3: bool = False, gate4: bool = False) -> dict[str, bool]:
    if gate4 and not (gate1 and gate2 and gate3):
        raise ValueError("Gate 4 claim cannot unlock without Gates 1--3")
    if gate3 and not (gate1 and gate2):
        raise ValueError("Gate 3 claim cannot unlock without Gates 1--2")
    if gate2 and not gate1:
        raise ValueError("Gate 2 claim cannot unlock without Gate 1")
    return {
        "oracle_headroom": bool(gate1),
        "mechanism": bool(gate2),
        "calibrated_risk_on_frozen_window_protocol": bool(gate3),
        "metric_adatad_thumos14_official_full_video": bool(gate4),
        "latency_gpu1_fixed_stack": bool(gate4),
        "deploy": False,
        "paper": False,
    }
