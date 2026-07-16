"""Immutable, deeply validated pre-Gate1 registration contracts."""

from __future__ import annotations

import copy
from datetime import datetime
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .controls import (
    R2_RANDOM_CONTROL_SEED,
    random_exact_count_actions,
    validate_r2_control_algorithm_identity,
)
from .environment import validate_required_environment
from .filesystem import (
    BoundDirectory,
    decode_json_bytes,
    ensure_bound_directory,
    load_bound_json,
    open_bound_directory,
    open_bound_regular_file,
    publish_bytes_exclusive,
    read_bound_bytes,
)
from .protocol import (
    R2_PROTOCOL_ID,
    build_stage_b_exposure_artifact,
    canonical_json_bytes,
    canonical_sha256,
    manifest_exact_bytes,
    validate_r2_manifest,
    validate_stage_b_exposure_artifact,
)
from .scheduler import R2_NON_DENSE_NAMES, validate_r2_library_payload


REGISTRATION_SCHEMA = "chronotransport-r2-pre-gate1-registration-v4"
PROFILE_PLAN_SCHEMA = "chronotransport-r2-profiler-plan-v1"
CHECKPOINT_RECEIPT_SCHEMA = "chronotransport-r2-checkpoint-registry-receipt-v2"
CHECKPOINT_RECEIPT_PROVIDER_IDENTITY = "paracloud-registry"
CHECKPOINT_RECEIPT_TOOL_IDENTITY = "paracloud-registry-client/v1"
CHECKPOINT_RECEIPT_PRINCIPAL = "sczc063@BSCC-N16R4"
APPROVED_SPEC_COMMIT = "537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37"
APPROVED_SPEC_SHA256 = "e79dfaab8f9b0093e96cbd6b46bef4ecf8d6433009e2dcb922ad0f4c473b27a6"
FORMAL_OUTPUT_BASE = (
    "/data/run01/sczc063/yuzibo/chronotransport_runs/ct_p3r_3s_r2"
)
REGISTERED_PROFILE_FACTORY_IDENTITY = (
    "tools.bata.chronotransport_r2_profile_factory:"
    "build_registered_profile_session"
)
REGISTERED_PROFILE_BACKEND_IDENTITY = (
    "tools.bata.chronotransport_r2_opentad_profile_backend:"
    "OpenTADRegisteredProfileBackend"
)
REGISTERED_PROFILE_BACKEND_SOURCE = (
    "tools/bata/chronotransport_r2_opentad_profile_backend.py"
)
SOURCE_CLASSIFICATION_PATH = (
    "opentad/models/chronotransport/source_classification.json"
)
SOURCE_CLASSIFICATION_SCHEMA = "chronotransport-r2-source-classification-v1"
SOURCE_CLASSIFICATION_CLASSES = frozenset(
    {"REQUIRED", "TEST_ONLY_NON_FORMAL", "OUT_OF_SCOPE"}
)
SOURCE_CLASSIFICATION_GLOBS = (
    "tests/test_chronotransport*.py",
    "tools/bata/*chronotransport*.py",
    "scripts/*chronotransport*.sh",
)
REQUIRED_REGISTRATION_SOURCE_PATHS = (
    "docs/superpowers/specs/2026-07-12-chronotransport-ct-p3r-3s-r2-design.md",
    SOURCE_CLASSIFICATION_PATH,
    "opentad/datasets/builder.py",
    "opentad/datasets/thumos.py",
    "opentad/datasets/base/sliding_dataset.py",
    "opentad/datasets/transforms/end_to_end.py",
    "opentad/datasets/transforms/formatting.py",
    "opentad/evaluations/mAP.py",
    "opentad/models/__init__.py",
    "opentad/models/builder.py",
    "opentad/models/backbones/vit_adapter.py",
    "opentad/models/detectors/base.py",
    "opentad/models/detectors/actionformer.py",
    "opentad/models/chronotransport/__init__.py",
    "opentad/models/chronotransport/actions.py",
    "opentad/models/chronotransport/adjudication.py",
    "opentad/models/chronotransport/cache.py",
    "opentad/models/chronotransport/controls.py",
    "opentad/models/chronotransport/cost_lookup.py",
    "opentad/models/chronotransport/environment.py",
    "opentad/models/chronotransport/filesystem.py",
    "opentad/models/chronotransport/formal_stage_b.py",
    "opentad/models/chronotransport/full_stack_profiler.py",
    "opentad/models/chronotransport/gate1_unlock.py",
    "opentad/models/chronotransport/gate4.py",
    "opentad/models/chronotransport/gates23.py",
    "opentad/models/chronotransport/losses.py",
    "opentad/models/chronotransport/profiler.py",
    "opentad/models/chronotransport/protocol.py",
    "opentad/models/chronotransport/registration.py",
    "opentad/models/chronotransport/replay.py",
    "opentad/models/chronotransport/risk.py",
    "opentad/models/chronotransport/runtime.py",
    "opentad/models/chronotransport/scheduler.py",
    "opentad/models/chronotransport/stage_c.py",
    "opentad/models/chronotransport/training.py",
    "opentad/models/chronotransport/transport.py",
    "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py",
    "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_b.py",
    "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_c.py",
    "configs/adatad/thumos/c3_chronotransport_r2_stage_b.py",
    "configs/adatad/thumos/c3_chronotransport_r2_stage_c.py",
    "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py",
    "configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",
    "configs/_base_/models/actionformer.py",
    "tools/bata/build_chronotransport_r2_manifest.py",
    "tools/bata/chronotransport_r2_opentad_profile_backend.py",
    "tools/bata/chronotransport_r2_gate1_replay_factory.py",
    "tools/bata/chronotransport_r2_gates23_replay_factory.py",
    "tools/bata/chronotransport_r2_profile_factory.py",
    "tools/bata/profile_chronotransport_r2_full_stack.py",
    "tools/bata/register_chronotransport_r2.py",
    "tools/bata/run_chronotransport_r2_gate1.py",
    "tools/bata/run_chronotransport_r2_gates23.py",
    "tools/bata/chronotransport_r2_stage_b_factory.py",
    "tools/bata/train_chronotransport_r2_stage_b.py",
    "tools/bata/validate_chronotransport_r2_precheck.py",
    "scripts/run_chronotransport_r2_gate1_slurm_single_gpu.sh",
    "tests/test_chronotransport_core.py",
    "tests/test_chronotransport_pipeline.py",
    "tests/test_chronotransport_r2_actions_cache.py",
    "tests/test_chronotransport_r2_adjudication.py",
    "tests/test_chronotransport_r2_environment.py",
    "tests/test_chronotransport_r2_filesystem.py",
    "tests/test_chronotransport_r2_gate1_cost_profile.py",
    "tests/test_chronotransport_r2_gate1_hardening.py",
    "tests/test_chronotransport_r2_gate4.py",
    "tests/test_chronotransport_r2_gates23.py",
    "tests/test_chronotransport_r2_manifest_protocol.py",
    "tests/test_chronotransport_r2_profile_backend.py",
    "tests/test_chronotransport_r2_protocol.py",
    "tests/test_chronotransport_r2_registration.py",
    "tests/test_chronotransport_r2_risk.py",
    "tests/test_chronotransport_r2_runtime.py",
    "tests/test_chronotransport_r2_stage_b.py",
    "tests/test_chronotransport_r2_stage_c.py",
    "tests/test_chronotransport_repository_contract.py",
    "tests/test_chronotransport_vit_adapter_integration.py",
)
PROFILE_CONTROL_NAMES = (
    "motion_topk_p2",
    "motion_topk_p4",
    "motion_topk_p8",
    "random_p2",
    "random_p4",
    "random_p8",
)
EXPECTED_PROFILE_CANDIDATE_ORDER = R2_NON_DENSE_NAMES + ("dense",) + PROFILE_CONTROL_NAMES
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
FORBIDDEN_KEY_FRAGMENTS = (
    "result",
    "evaluation_output",
    "replay_output",
    "gate_report",
    "profile_output",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FACTORY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_UTC_TIMESTAMP = re.compile(
    r"^(?:19|20)\d\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])T"
    r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\dZ$"
)
_CANDIDATE_PLAN_FIELDS = {
    "candidate_name",
    "candidate_identity_sha256",
    "factory_identity",
    "factory_config",
    "factory_config_sha256",
    "requested_action_sha256_by_invocation",
    "requested_action_order_sha256",
    "selected_rows_per_group",
}


def _require_exact_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} fields mismatch")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise ValueError(f"{label} must be a full 40-hex commit")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty NUL-free string")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _validate_checkpoint_registry_receipt_artifact(
    artifact: Any,
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise TypeError("checkpoint registry receipt must be a mapping")
    _require_exact_fields(
        artifact,
        {
            "schema",
            "provider_identity",
            "registry_id",
            "authenticated_uri",
            "retrieval_tool_identity",
            "authenticated_principal",
            "registry_request_id",
            "retrieved_at_utc",
            "content_sha256",
            "content_bytes",
            "provider_receipt_sha256",
            "artifact_sha256",
        },
        "checkpoint registry receipt",
    )
    validated = dict(artifact)
    if validated["schema"] != CHECKPOINT_RECEIPT_SCHEMA:
        raise ValueError("unsupported checkpoint registry receipt schema")
    if validated["provider_identity"] != CHECKPOINT_RECEIPT_PROVIDER_IDENTITY:
        raise ValueError("checkpoint receipt provider identity is not approved")
    if validated["retrieval_tool_identity"] != CHECKPOINT_RECEIPT_TOOL_IDENTITY:
        raise ValueError("checkpoint receipt retrieval tool identity is not approved")
    if validated["authenticated_principal"] != CHECKPOINT_RECEIPT_PRINCIPAL:
        raise ValueError("checkpoint receipt authenticated principal is not approved")
    for field in (
        "registry_id",
        "provider_identity",
        "authenticated_uri",
        "retrieval_tool_identity",
        "authenticated_principal",
        "registry_request_id",
    ):
        _require_nonempty_string(validated[field], f"checkpoint receipt.{field}")
    timestamp = _require_nonempty_string(
        validated["retrieved_at_utc"], "checkpoint receipt.retrieved_at_utc"
    )
    if not _UTC_TIMESTAMP.fullmatch(timestamp):
        raise ValueError("checkpoint receipt retrieved_at_utc must be a UTC second timestamp")
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(
            "checkpoint receipt retrieved_at_utc must be a valid UTC timestamp"
        ) from exc
    _require_sha(validated["content_sha256"], "checkpoint receipt.content_sha256")
    _require_int(validated["content_bytes"], "checkpoint receipt.content_bytes", minimum=1)
    _require_sha(
        validated["provider_receipt_sha256"],
        "checkpoint receipt.provider_receipt_sha256",
    )
    supplied = _require_sha(
        validated["artifact_sha256"], "checkpoint receipt.artifact_sha256"
    )
    unsigned = dict(validated)
    unsigned.pop("artifact_sha256")
    if supplied != canonical_sha256(unsigned):
        raise ValueError("checkpoint registry receipt artifact hash mismatch")
    return validated


def validate_checkpoint_registry_receipt(
    artifact: Any,
    *,
    provider_receipt_path: str | Path,
    registry_id: str,
    authenticated_uri: str,
    content_sha256: str,
    content_bytes: int,
) -> dict[str, Any]:
    """Validate an external registry receipt against the exact checkpoint bytes."""

    validated = _validate_checkpoint_registry_receipt_artifact(artifact)
    expected = {
        "registry_id": _require_nonempty_string(registry_id, "expected registry_id"),
        "authenticated_uri": _require_nonempty_string(
            authenticated_uri, "expected authenticated_uri"
        ),
        "content_sha256": _require_sha(content_sha256, "expected content_sha256"),
        "content_bytes": _require_int(content_bytes, "expected content_bytes", minimum=1),
    }
    for field, value in expected.items():
        if validated[field] != value:
            raise ValueError(f"checkpoint registry receipt {field} mismatch")
    _, provider_bytes, provider_digest = read_bound_bytes(
        provider_receipt_path,
        label="external checkpoint provider receipt",
    )
    if not provider_bytes:
        raise ValueError("external checkpoint provider receipt must not be empty")
    if validated["provider_receipt_sha256"] != provider_digest:
        raise ValueError("external checkpoint provider receipt hash mismatch")
    return validated


def _validate_checkpoint_receipt_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("dense checkpoint registry_receipt must be a mapping")
    _require_exact_fields(
        value,
        {
            "artifact",
            "exact_bytes_sha256",
            "source_path",
            "provider_receipt_path",
        },
        "dense checkpoint registry_receipt",
    )
    artifact = _validate_checkpoint_registry_receipt_artifact(value["artifact"])
    exact_digest = hashlib.sha256(canonical_json_bytes(artifact) + b"\n").hexdigest()
    if value["exact_bytes_sha256"] != exact_digest:
        raise ValueError("checkpoint registry receipt exact bytes hash mismatch")
    _require_nonempty_string(value["source_path"], "checkpoint receipt.source_path")
    _require_nonempty_string(
        value["provider_receipt_path"], "checkpoint receipt.provider_receipt_path"
    )
    return {**dict(value), "artifact": artifact}


def _audit_no_results(value: Any, path: str = "registration") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"registration key must be a string: {path}")
            normalized = key.lower()
            if normalized != "result_data_unread" and any(
                fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                raise ValueError(f"registration contains forbidden result-derived key: {path}.{key}")
            _audit_no_results(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _audit_no_results(item, f"{path}[{index}]")


def _validate_environment(environment: Any) -> dict[str, Any]:
    if not isinstance(environment, Mapping):
        raise TypeError("registration environment must be a mapping")
    return validate_required_environment(environment)


def _validate_window_manifest(value: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(value, Mapping):
        raise TypeError("window_manifest must be a mapping")
    _require_exact_fields(
        value,
        {
            "artifact",
            "exact_bytes_sha256",
            "source_path",
            "registry_path",
            "config_identity_path",
        },
        "window_manifest",
    )
    artifact = validate_r2_manifest(value["artifact"])
    exact_digest = hashlib.sha256(manifest_exact_bytes(artifact)).hexdigest()
    if value["exact_bytes_sha256"] != exact_digest:
        raise ValueError("window manifest exact bytes SHA-256 mismatch")
    for field in ("source_path", "registry_path", "config_identity_path"):
        _require_nonempty_string(value[field], f"window_manifest.{field}")
    ordered = [
        window_id
        for split in ("fit", "calibration", "evaluation")
        for window_id in artifact["splits"][split]
    ]
    return {**dict(value), "artifact": artifact}, ordered


def _validate_sha_mapping(value: Any, label: str, *, commit: bool = False) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    validated: dict[str, str] = {}
    for key, digest in value.items():
        _require_nonempty_string(key, f"{label} key")
        validated[key] = (
            _require_commit(digest, f"{label}.{key}")
            if commit
            else _require_sha(digest, f"{label}.{key}")
        )
    return validated


def _selected_rows(actions: Sequence[Sequence[int]]) -> list[int]:
    return [sum(row[group] == 0 for row in actions) for group in range(3)]


def _validate_profiler_plan(
    value: Any,
    *,
    library: Mapping[str, Any],
    controls: Mapping[str, Any],
    environment: Mapping[str, Any],
    manifest_invocation_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("profiler plan must be a mapping")
    _require_exact_fields(
        value,
        {
            "schema",
            "candidate_order",
            "candidate_order_sha256",
            "invocation_ids",
            "invocation_order_sha256",
            "warmup_count",
            "sample_count",
            "candidate_plan",
            "expected_environment",
            "model_config_sha256",
        },
        "profiler plan",
    )
    if value["schema"] != PROFILE_PLAN_SCHEMA:
        raise ValueError("unsupported profiler plan schema")
    order = value["candidate_order"]
    if not isinstance(order, list) or tuple(order) != EXPECTED_PROFILE_CANDIDATE_ORDER:
        raise ValueError("profiler candidate order must equal the frozen 23-candidate order")
    if value["candidate_order_sha256"] != canonical_sha256(order):
        raise ValueError("profiler candidate order hash mismatch")
    invocation_ids = value["invocation_ids"]
    if not isinstance(invocation_ids, list) or invocation_ids != list(manifest_invocation_ids):
        raise ValueError("profiler invocation IDs/order must equal the exact 200 manifested windows")
    if len(set(invocation_ids)) != 200:
        raise ValueError("profiler invocation IDs must contain 200 unique IDs")
    if value["invocation_order_sha256"] != canonical_sha256(invocation_ids):
        raise ValueError("profiler invocation order hash mismatch")
    if _require_int(value["warmup_count"], "profiler warmup_count") != 50:
        raise ValueError("formal profiler warmup_count must equal 50")
    if _require_int(value["sample_count"], "profiler sample_count") != 200:
        raise ValueError("formal profiler sample_count must equal 200")
    _require_sha(value["model_config_sha256"], "profiler.model_config_sha256")
    if value["expected_environment"] != environment:
        raise ValueError("profiler expected environment does not match registered environment")
    _validate_environment(value["expected_environment"])

    candidates = value["candidate_plan"]
    if not isinstance(candidates, list) or len(candidates) != len(order):
        raise ValueError("profiler candidate plan must contain all 23 candidates")
    library_rows = {row["name"]: row for row in library["candidates"]}
    for index, (name, plan) in enumerate(zip(order, candidates)):
        if not isinstance(plan, Mapping):
            raise TypeError(f"profiler candidate plan {index} must be a mapping")
        _require_exact_fields(plan, _CANDIDATE_PLAN_FIELDS, f"profiler candidate plan {index}")
        if plan["candidate_name"] != name:
            raise ValueError("profiler candidate plan order/name mismatch")
        if name in library_rows:
            expected_identity = library_rows[name]["action_sha256"]
            expected_rows = _selected_rows(library_rows[name]["actions"])
        else:
            algorithm = "motion_topk" if name.startswith("motion_topk_") else "random"
            expected_identity = controls[algorithm]["sha256"]
            period = int(name.rsplit("p", 1)[1])
            expected_rows = [len(range(0, 48, period))] * 3
        if plan["candidate_identity_sha256"] != expected_identity:
            raise ValueError(f"profiler candidate identity mismatch for {name}")
        if plan["factory_identity"] != REGISTERED_PROFILE_FACTORY_IDENTITY:
            raise ValueError(
                f"profiler candidate {name} must use the sole repo-owned profile factory"
            )
        if not isinstance(plan["factory_config"], Mapping):
            raise TypeError(f"profiler factory config must be a mapping for {name}")
        expected_factory_config = {
            "candidate_name": name,
            "mode": "registered_full_stack",
            "backend_identity": REGISTERED_PROFILE_BACKEND_IDENTITY,
            "backend_source_sha256": _require_sha(
                plan["factory_config"].get("backend_source_sha256", ""),
                "profiler.backend_source_sha256",
            ),
        }
        if name.startswith("random_p"):
            expected_factory_config["control_seed"] = R2_RANDOM_CONTROL_SEED
        if plan["factory_config"] != expected_factory_config:
            raise ValueError(
                f"profiler factory config must bind the fixed backend for {name}"
            )
        if plan["factory_config_sha256"] != canonical_sha256(plan["factory_config"]):
            raise ValueError(f"profiler factory config hash mismatch for {name}")
        action_hashes = plan["requested_action_sha256_by_invocation"]
        if not isinstance(action_hashes, list) or len(action_hashes) != 200:
            raise ValueError(f"profiler requires 200 requested action hashes for {name}")
        for digest in action_hashes:
            _require_sha(digest, f"profiler requested action hash for {name}")
        if name in library_rows and any(digest != expected_identity for digest in action_hashes):
            raise ValueError(f"static candidate requested action hash mismatch for {name}")
        if name.startswith("random_p"):
            period = int(name.rsplit("p", 1)[1])
            expected_random_hashes = [
                canonical_sha256(
                    random_exact_count_actions(
                        window_id,
                        seed=R2_RANDOM_CONTROL_SEED,
                        num_groups=3,
                        period=period,
                    ).tolist()
                )
                for window_id in invocation_ids
            ]
            if action_hashes != expected_random_hashes:
                raise ValueError(
                    f"random control generated action hashes mismatch for {name}"
                )
        if plan["requested_action_order_sha256"] != canonical_sha256(action_hashes):
            raise ValueError(f"profiler requested action order hash mismatch for {name}")
        selected_rows = plan["selected_rows_per_group"]
        if (
            not isinstance(selected_rows, list)
            or len(selected_rows) != 3
            or any(type(item) is not int for item in selected_rows)
            or selected_rows != expected_rows
        ):
            raise ValueError(f"profiler selected rows per group mismatch for {name}")
    return dict(value)


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise TypeError("registration identity must be a mapping")
    _audit_no_results(identity)
    _require_exact_fields(identity, set(REQUIRED_FIELDS), "registration identity")
    if identity["protocol_id"] != R2_PROTOCOL_ID:
        raise ValueError("registration protocol_id mismatch")

    spec = identity["spec"]
    if not isinstance(spec, Mapping):
        raise TypeError("registration spec must be a mapping")
    _require_exact_fields(spec, {"commit", "sha256"}, "registration spec")
    if _require_commit(spec["commit"], "spec.commit") != APPROVED_SPEC_COMMIT:
        raise ValueError("registration must bind the approved spec commit")
    if _require_sha(spec["sha256"], "spec.sha256") != APPROVED_SPEC_SHA256:
        raise ValueError("registration must bind the approved spec exact bytes")
    implementation = _require_commit(identity["implementation_commit"], "implementation_commit")
    parent = identity["registration_parent"]
    if not isinstance(parent, Mapping):
        raise TypeError("registration_parent must be a mapping")
    _require_exact_fields(parent, {"commit", "tree"}, "registration_parent")
    if _require_commit(parent["commit"], "registration_parent.commit") != implementation:
        raise ValueError("registration parent commit must equal implementation commit I")
    _require_commit(parent["tree"], "registration_parent.tree")
    source_files = _validate_sha_mapping(identity["source_files"], "source_files")
    if set(source_files) != set(REQUIRED_REGISTRATION_SOURCE_PATHS):
        raise ValueError("registration source files must equal the complete required surface")
    source_files = {
        relative: source_files[relative]
        for relative in REQUIRED_REGISTRATION_SOURCE_PATHS
    }
    backend_sha = source_files[REGISTERED_PROFILE_BACKEND_SOURCE]
    for plan in identity["profiler"]["candidate_plan"]:
        if plan["factory_config"].get("backend_source_sha256") != backend_sha:
            raise ValueError("profiler backend source hash differs from registered source bytes")
    _validate_sha_mapping(identity["upstream_commits"], "upstream_commits", commit=True)

    checkpoint = identity["dense_checkpoint"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError("dense_checkpoint must be a mapping")
    _require_exact_fields(
        checkpoint,
        {
            "sha256",
            "bytes",
            "registry_id",
            "authenticated_uri",
            "content_addressed_path",
            "registry_receipt",
        },
        "dense_checkpoint",
    )
    checkpoint_sha = _require_sha(checkpoint["sha256"], "dense_checkpoint.sha256")
    _require_int(checkpoint["bytes"], "dense_checkpoint.bytes", minimum=1)
    _require_nonempty_string(checkpoint["registry_id"], "dense_checkpoint.registry_id")
    authenticated_uri = _require_nonempty_string(
        checkpoint["authenticated_uri"], "dense_checkpoint.authenticated_uri"
    )
    parsed_uri = urlparse(authenticated_uri)
    registry_identity = f"{parsed_uri.netloc}{parsed_uri.path}".strip("/")
    if (
        parsed_uri.scheme != "registry"
        or not parsed_uri.netloc
        or not parsed_uri.path
        or parsed_uri.params
        or parsed_uri.query
        or parsed_uri.fragment
        or registry_identity != checkpoint["registry_id"]
    ):
        raise ValueError(
            "dense checkpoint requires an authenticated registry URI bound to registry_id"
        )
    content_path = _require_nonempty_string(
        checkpoint["content_addressed_path"], "dense_checkpoint.content_addressed_path"
    )
    normalized_content_path = content_path.replace("\\", "/").rstrip("/")
    if normalized_content_path.split("/")[-1] != checkpoint_sha:
        raise ValueError("dense checkpoint content-addressed path must end in its SHA-256")
    receipt = _validate_checkpoint_receipt_identity(checkpoint["registry_receipt"])
    receipt_artifact = receipt["artifact"]
    for field, expected in (
        ("registry_id", checkpoint["registry_id"]),
        ("authenticated_uri", checkpoint["authenticated_uri"]),
        ("content_sha256", checkpoint_sha),
        ("content_bytes", checkpoint["bytes"]),
    ):
        if receipt_artifact[field] != expected:
            raise ValueError(f"checkpoint registry receipt {field} differs from checkpoint")

    data = identity["data"]
    if not isinstance(data, Mapping):
        raise TypeError("registration data must be a mapping")
    _require_exact_fields(
        data,
        {"root_identity", "root_path", "annotation_sha256", "media_sha256"},
        "data",
    )
    _require_nonempty_string(data["root_identity"], "data.root_identity")
    _require_nonempty_string(data["root_path"], "data.root_path")
    _require_sha(data["annotation_sha256"], "data.annotation_sha256")
    media = _validate_sha_mapping(data["media_sha256"], "data.media_sha256")
    if len(media) != 200:
        raise ValueError("registration requires exactly 200 media SHA-256 identities")

    manifest, invocation_ids = _validate_window_manifest(identity["window_manifest"])
    manifest_artifact = manifest["artifact"]
    if manifest_artifact["data_identity"]["data_sha256"] != data["root_identity"]:
        raise ValueError("manifest data identity differs from registered data root")
    if manifest_artifact["data_identity"]["annotation_sha256"] != data["annotation_sha256"]:
        raise ValueError("manifest annotation identity differs from registered data")
    manifest_media = {
        row["video_id"]: row["media_sha256"] for row in manifest_artifact["windows"]
    }
    if manifest_media != dict(data["media_sha256"]):
        raise ValueError("manifest media hashes differ from registered data")
    library = validate_r2_library_payload(identity["candidate_library"])
    controls = validate_r2_control_algorithm_identity(identity["controls"])

    exposures = identity["exposures"]
    if not isinstance(exposures, Mapping):
        raise TypeError("registration exposures must be a mapping")
    _require_exact_fields(exposures, {"stage_b", "stage_c_formula"}, "exposures")
    validate_stage_b_exposure_artifact(
        exposures["stage_b"], fit_window_ids=manifest_artifact["splits"]["fit"]
    )
    _require_nonempty_string(exposures["stage_c_formula"], "exposures.stage_c_formula")

    bootstrap = identity["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise TypeError("registration bootstrap must be a mapping")
    _require_exact_fields(bootstrap, {"gate1_samples", "seed"}, "bootstrap")
    if _require_int(bootstrap["gate1_samples"], "bootstrap.gate1_samples") != 5000:
        raise ValueError("formal Gate 1 requires exactly 5000 bootstrap samples")
    if _require_int(bootstrap["seed"], "bootstrap.seed") != 20260711:
        raise ValueError("formal Gate 1 requires bootstrap seed 20260711")

    gates = identity["gates"]
    if not isinstance(gates, Mapping):
        raise TypeError("registration gates must be a mapping")
    _require_exact_fields(gates, {"gate1_relative", "budget_saving"}, "gates")
    if type(gates["gate1_relative"]) is not float or gates["gate1_relative"] != 0.1:
        raise ValueError("registered Gate 1 relative threshold must equal 0.1")
    if type(gates["budget_saving"]) is not float or gates["budget_saving"] != 0.2:
        raise ValueError("registered budget saving threshold must equal 0.2")

    environment = _validate_environment(identity["environment"])
    _validate_profiler_plan(
        identity["profiler"],
        library=library,
        controls=controls,
        environment=environment,
        manifest_invocation_ids=invocation_ids,
    )
    output_root = identity["output_root"]
    if not isinstance(output_root, Mapping):
        raise TypeError("output_root must be a base/template mapping")
    _require_exact_fields(output_root, {"base", "template"}, "output_root")
    if output_root["base"] != FORMAL_OUTPUT_BASE:
        raise ValueError("output_root.base must equal the fixed formal output base")
    if output_root["template"] != "{base}/{registration_commit}/shared/gate1":
        raise ValueError("output_root must use the fixed R-derived Gate 1 template")
    attestation = identity["attestation"]
    if not isinstance(attestation, Mapping):
        raise TypeError("registration attestation must be a mapping")
    _require_exact_fields(attestation, {"result_data_unread"}, "attestation")
    if attestation["result_data_unread"] is not True:
        raise ValueError("registration requires result_data_unread=true attestation")

    validated = dict(identity)
    validated["window_manifest"] = manifest
    return validated


def _sha256_file(path: Path) -> tuple[int, str]:
    with open_bound_regular_file(path, label=f"registered input {path}") as bound:
        return bound.size_and_sha256()


def _load_json_file(path: Path) -> object:
    _, value, _, _ = load_bound_json(path, label=f"registered JSON input {path}")
    return value


def validate_source_classification_manifest(
    value: Any,
    *,
    tracked_paths: Sequence[str],
    required_source_paths: Sequence[str],
) -> dict[str, Any]:
    """Require an explicit class for every tracked ChronoTransport test/entrypoint."""

    if not isinstance(value, Mapping):
        raise TypeError("source classification manifest must be a mapping")
    _require_exact_fields(value, {"schema", "files"}, "source classification manifest")
    if value["schema"] != SOURCE_CLASSIFICATION_SCHEMA:
        raise ValueError("unsupported source classification schema")
    files = value["files"]
    if not isinstance(files, Mapping) or not files:
        raise ValueError("source classification files must be a non-empty mapping")

    classified: dict[str, str] = {}
    for path, classification in files.items():
        if not isinstance(path, str) or not path:
            raise TypeError("source classification path must be a non-empty string")
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or "\\" in path
            or any(part in ("", ".", "..") for part in pure.parts)
            or pure.as_posix() != path
        ):
            raise ValueError("source classification paths must be canonical repository paths")
        if classification not in SOURCE_CLASSIFICATION_CLASSES:
            raise ValueError(f"unsupported source classification for {path}")
        if classification == "TEST_ONLY_NON_FORMAL" and not path.startswith("tests/"):
            raise ValueError("TEST_ONLY_NON_FORMAL is restricted to test files")
        classified[path] = classification

    tracked = list(tracked_paths)
    if any(not isinstance(path, str) or not path for path in tracked):
        raise TypeError("tracked source classification paths must be non-empty strings")
    if len(tracked) != len(set(tracked)):
        raise ValueError("tracked source classification inventory contains duplicates")
    if set(classified) != set(tracked):
        raise ValueError("source classification must equal the exact tracked inventory")

    required = set(required_source_paths)
    if SOURCE_CLASSIFICATION_PATH not in required:
        raise ValueError("source classification artifact must be in the source vector")
    required_classified = {
        path for path, classification in classified.items() if classification == "REQUIRED"
    }
    vector_classified = required.intersection(classified)
    if required_classified != vector_classified:
        raise ValueError("REQUIRED classification must exactly equal the source vector")
    nonformal_in_vector = {
        path
        for path, classification in classified.items()
        if classification != "REQUIRED" and path in required
    }
    if nonformal_in_vector:
        raise ValueError("non-formal classified paths must remain outside the source vector")
    return {"schema": value["schema"], "files": classified}


def _tracked_source_classification_paths(
    root: BoundDirectory | Path, revision: str
) -> list[str]:
    tree_paths = _git(root, "ls-tree", "-r", "--name-only", revision).splitlines()
    return [
        path
        for path in tree_paths
        if any(fnmatchcase(path, pattern) for pattern in SOURCE_CLASSIFICATION_GLOBS)
    ]


def _run_git(
    root: BoundDirectory | Path,
    *arguments: str,
    text: bool,
) -> subprocess.CompletedProcess:
    """Run Git against a retained repository fd when one is available."""

    if isinstance(root, BoundDirectory):
        directory = root.proc_path
        descriptor_arguments = {"pass_fds": (root.descriptor,)}
    else:
        directory = str(root)
        descriptor_arguments = {}
    return subprocess.run(
        ["git", "-C", directory, *arguments],
        check=False,
        capture_output=True,
        text=text,
        **descriptor_arguments,
    )


def _git(
    root: BoundDirectory | Path, *arguments: str, check: bool = True
) -> str:
    completed = _run_git(root, *arguments, text=True)
    if check and completed.returncode != 0:
        raise ValueError(
            f"git context command failed: {' '.join(arguments)}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git_blob_bytes(
    root: BoundDirectory | Path, revision: str, relative: str
) -> bytes:
    completed = _run_git(root, "show", f"{revision}:{relative}", text=False)
    if completed.returncode != 0:
        raise ValueError("registered Git blob does not exist")
    return completed.stdout


def _validate_registered_source_file(
    *,
    root: BoundDirectory | str | Path,
    revision: str,
    relative: str,
    registered_sha256: str,
) -> None:
    """Bind one required source to regular worktree and exact Git-blob bytes."""

    if not isinstance(root, BoundDirectory):
        with open_bound_directory(root, label="registered source repository") as bound:
            return _validate_registered_source_file(
                root=bound,
                revision=revision,
                relative=relative,
                registered_sha256=registered_sha256,
            )

    tree_row = _git(root, "ls-tree", revision, "--", relative)
    if "\t" not in tree_row:
        raise ValueError(f"required registration source is not tracked: {relative}")
    identity, tracked_path = tree_row.split("\t", 1)
    parts = identity.split()
    if (
        len(parts) != 3
        or parts[0] not in ("100644", "100755")
        or parts[1] != "blob"
        or tracked_path != relative
    ):
        raise ValueError(
            f"required registration source Git mode must be a regular blob: {relative}"
        )
    try:
        with root.open_regular(
            relative, label=f"required registration source {relative}"
        ) as bound:
            current, current_sha256 = bound.bytes_and_sha256()
    except OSError as error:
        raise ValueError(
            f"required registration source must be a regular non-symlink file: {relative}"
        ) from error
    if _git_blob_bytes(root, revision, relative) != current:
        raise ValueError(f"required source differs from implementation Git blob: {relative}")
    if current_sha256 != registered_sha256:
        raise ValueError(f"required source bytes differ from registration: {relative}")


def validate_registration_commit_shape(
    *,
    repository_root: str | Path,
    registration_commit: str,
    implementation_commit: str,
    registration_relpath: str,
    registration_bytes: bytes,
) -> None:
    """Bind exact canonical registration bytes to a regular file at ``R:path``."""

    with open_bound_directory(
        repository_root, label="registration repository root"
    ) as bound_root:
        _validate_registration_commit_shape_bound(
            root=bound_root,
            registration_commit=registration_commit,
            implementation_commit=implementation_commit,
            registration_relpath=registration_relpath,
            registration_bytes=registration_bytes,
        )


def _validate_registration_commit_shape_bound(
    *,
    root: BoundDirectory,
    registration_commit: str,
    implementation_commit: str,
    registration_relpath: str,
    registration_bytes: bytes,
) -> None:
    """Implementation of commit-shape validation rooted at one directory fd."""

    git_root = root
    _require_commit(registration_commit, "registration commit R")
    _require_commit(implementation_commit, "implementation commit I")
    if not isinstance(registration_relpath, str) or not registration_relpath:
        raise ValueError("formal registration validation requires registration relative path")
    relative = PurePosixPath(registration_relpath)
    if (
        relative.is_absolute()
        or "\\" in registration_relpath
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != registration_relpath
    ):
        raise ValueError("registration relative path must be canonical and repository-local")
    if not isinstance(registration_bytes, bytes):
        raise TypeError("registration bytes must be exact bytes")

    with root.open_regular(
        registration_relpath, label="current registration artifact"
    ) as registration_file:
        current_registration_bytes = registration_file.read_bytes()

    parent_line = _git(git_root, "rev-list", "--parents", "-n", "1", registration_commit)
    if parent_line.split() != [registration_commit, implementation_commit]:
        raise ValueError("registration commit R must have exactly one parent I")
    expected_status = f"A\t{registration_relpath}"
    changed = _git(
        git_root,
        "diff",
        "--name-status",
        "--no-renames",
        implementation_commit,
        registration_commit,
    ).splitlines()
    if changed != [expected_status]:
        raise ValueError("I..R must be exactly one added registration artifact")
    absent = _run_git(
        git_root,
        "cat-file",
        "-e",
        f"{implementation_commit}:{registration_relpath}",
        text=False,
    )
    if absent.returncode == 0:
        raise ValueError("registration artifact must be absent from implementation commit I")
    tree_row = _git(git_root, "ls-tree", registration_commit, "--", registration_relpath)
    if "\t" not in tree_row:
        raise ValueError("registration artifact is absent from registration commit R")
    identity, tracked_path = tree_row.split("\t", 1)
    parts = identity.split()
    if (
        len(parts) != 3
        or parts[0] not in ("100644", "100755")
        or parts[1] != "blob"
        or tracked_path != registration_relpath
    ):
        raise ValueError("registration artifact Git mode must be a regular blob")
    blob = _git_blob_bytes(git_root, registration_commit, registration_relpath)
    if blob != registration_bytes:
        raise ValueError("registration commit blob bytes differ from canonical exact bytes")
    if current_registration_bytes != registration_bytes:
        raise ValueError("current registration artifact differs from canonical exact bytes")


def _validate_repository_context(
    registration: Mapping[str, Any],
    *,
    repository_root: str | Path,
    context_mode: str,
    registration_commit: str | None,
    registration_relpath: str | None,
) -> None:
    with open_bound_directory(
        repository_root, label="registration repository root"
    ) as bound_root:
        _validate_repository_context_bound(
            registration,
            root=bound_root,
            context_mode=context_mode,
            registration_commit=registration_commit,
            registration_relpath=registration_relpath,
        )


def _validate_repository_context_bound(
    registration: Mapping[str, Any],
    *,
    root: BoundDirectory,
    context_mode: str,
    registration_commit: str | None,
    registration_relpath: str | None,
) -> None:
    if context_mode == "formal":
        validate_formal_random_control_lock(registration)
    git_root = root
    if _git(git_root, "status", "--porcelain"):
        raise ValueError("registration repository context must be clean")
    symbolic = _run_git(git_root, "symbolic-ref", "-q", "HEAD", text=True)
    if symbolic.returncode == 0:
        raise ValueError("registration repository context must use detached HEAD")
    head = _git(git_root, "rev-parse", "HEAD")
    implementation = registration["implementation_commit"]
    registered_tree = registration["registration_parent"]["tree"]
    if _git(git_root, "rev-parse", f"{implementation}^{{tree}}") != registered_tree:
        raise ValueError("actual implementation tree differs from registration")
    if context_mode == "generation":
        if registration_commit is not None or head != implementation:
            raise ValueError("registration generation requires clean detached HEAD exactly at I")
    elif context_mode == "formal":
        if not isinstance(registration_commit, str) or not _COMMIT.fullmatch(registration_commit):
            raise ValueError("formal registration validation requires full registration commit R")
        if head != registration_commit:
            raise ValueError("formal registration validation requires HEAD=R")
        if not isinstance(registration_relpath, str) or not registration_relpath:
            raise ValueError("formal registration validation requires registration relative path")
        _validate_registration_commit_shape_bound(
            root=root,
            registration_commit=registration_commit,
            implementation_commit=implementation,
            registration_relpath=registration_relpath,
            registration_bytes=canonical_json_bytes(registration) + b"\n",
        )
    else:
        raise ValueError("registration context_mode must be generation or formal")

    actual_sources: dict[str, str] = {}
    for relative in REQUIRED_REGISTRATION_SOURCE_PATHS:
        registered_sha = registration["source_files"].get(relative)
        if not isinstance(registered_sha, str):
            raise ValueError(f"required registration source hash is missing: {relative}")
        _validate_registered_source_file(
            root=root,
            revision=implementation,
            relative=relative,
            registered_sha256=registered_sha,
        )
        actual_sources[relative] = registered_sha
    if actual_sources != registration["source_files"]:
        raise ValueError("required source file bytes differ from registration")
    classification_bytes, _ = root.read_bytes(
        SOURCE_CLASSIFICATION_PATH,
        label="source classification manifest",
    )
    classification = decode_json_bytes(
        classification_bytes, label="source classification manifest"
    )
    validate_source_classification_manifest(
        classification,
        tracked_paths=_tracked_source_classification_paths(git_root, implementation),
        required_source_paths=REQUIRED_REGISTRATION_SOURCE_PATHS,
    )
    spec_relative = REQUIRED_REGISTRATION_SOURCE_PATHS[0]
    spec_bytes, spec_sha256 = root.read_bytes(
        spec_relative, label="approved ChronoTransport specification"
    )
    if spec_sha256 != APPROVED_SPEC_SHA256:
        raise ValueError("spec file bytes differ from registration")
    if _git_blob_bytes(git_root, APPROVED_SPEC_COMMIT, spec_relative) != spec_bytes:
        raise ValueError("approved spec Git blob differs from current exact bytes")
    ancestry = _run_git(
        git_root,
        "merge-base",
        "--is-ancestor",
        APPROVED_SPEC_COMMIT,
        implementation,
        text=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("approved spec commit must be an ancestor of implementation I")

    manifest_identity = registration["window_manifest"]
    _, manifest, raw_manifest, manifest_digest = load_bound_json(
        manifest_identity["source_path"], label="registered window manifest"
    )
    _, registry, _, _ = load_bound_json(
        manifest_identity["registry_path"], label="registered media registry"
    )
    _, config, _, _ = load_bound_json(
        manifest_identity["config_identity_path"], label="registered config identity"
    )
    if raw_manifest != manifest_exact_bytes(manifest):
        raise ValueError("manifest artifact bytes are not exact canonical bytes")
    if manifest_digest != manifest_identity["exact_bytes_sha256"]:
        raise ValueError("manifest artifact exact bytes hash differs from registration")
    rebuilt_manifest = validate_r2_manifest(
        manifest, registry=registry, config_identity=config
    )
    if rebuilt_manifest != manifest_identity["artifact"]:
        raise ValueError("manifest artifact differs from registered payload")

    with open_bound_regular_file(
        registration["dense_checkpoint"]["content_addressed_path"],
        label="registered dense checkpoint",
    ) as checkpoint_file:
        checkpoint_size, checkpoint_sha = checkpoint_file.size_and_sha256()
    if (
        checkpoint_size != registration["dense_checkpoint"]["bytes"]
        or checkpoint_sha != registration["dense_checkpoint"]["sha256"]
    ):
        raise ValueError("dense checkpoint bytes/hash differ from registration")
    receipt_identity = registration["dense_checkpoint"]["registry_receipt"]
    _, receipt_artifact, raw_receipt, receipt_digest = load_bound_json(
        receipt_identity["source_path"], label="registered checkpoint receipt"
    )
    if raw_receipt != canonical_json_bytes(receipt_artifact) + b"\n":
        raise ValueError("checkpoint receipt artifact bytes are not exact canonical bytes")
    if receipt_digest != receipt_identity["exact_bytes_sha256"]:
        raise ValueError("checkpoint receipt exact bytes differ from registration")
    if receipt_artifact != receipt_identity["artifact"]:
        raise ValueError("checkpoint receipt artifact differs from registration")
    validate_checkpoint_registry_receipt(
        receipt_artifact,
        provider_receipt_path=receipt_identity["provider_receipt_path"],
        registry_id=registration["dense_checkpoint"]["registry_id"],
        authenticated_uri=registration["dense_checkpoint"]["authenticated_uri"],
        content_sha256=checkpoint_sha,
        content_bytes=checkpoint_size,
    )

    media_hashes = registration["data"]["media_sha256"]
    with open_bound_directory(
        registration["data"]["root_path"], label="registered media root"
    ) as data_root:
        for window in manifest["windows"]:
            with data_root.open_regular(
                window["media_path"],
                label=f"registered media {window['video_id']}",
            ) as media:
                _, media_sha256 = media.size_and_sha256()
            if media_sha256 != media_hashes[window["video_id"]]:
                raise ValueError(
                    f"registered media bytes/hash mismatch: {window['video_id']}"
                )


def build_pre_gate1_registration_from_context(
    identity_template: Mapping[str, Any],
    *,
    repository_root: str | Path,
    manifest_path: str | Path,
    registry_path: str | Path,
    config_identity_path: str | Path,
    checkpoint_source: str | Path,
    checkpoint_registry_id: str,
    checkpoint_authenticated_uri: str,
    checkpoint_receipt_path: str | Path,
    checkpoint_provider_receipt_path: str | Path,
    content_store_root: str | Path,
    data_root: str | Path,
) -> dict[str, Any]:
    """Derive every mutable filesystem/Git identity before registration R exists."""

    with open_bound_directory(
        repository_root, label="registration generation repository root"
    ) as bound_root:
        return _build_pre_gate1_registration_from_bound_context(
            identity_template,
            root=bound_root,
            manifest_path=manifest_path,
            registry_path=registry_path,
            config_identity_path=config_identity_path,
            checkpoint_source=checkpoint_source,
            checkpoint_registry_id=checkpoint_registry_id,
            checkpoint_authenticated_uri=checkpoint_authenticated_uri,
            checkpoint_receipt_path=checkpoint_receipt_path,
            checkpoint_provider_receipt_path=checkpoint_provider_receipt_path,
            content_store_root=content_store_root,
            data_root=data_root,
        )


def _build_pre_gate1_registration_from_bound_context(
    identity_template: Mapping[str, Any],
    *,
    root: BoundDirectory,
    manifest_path: str | Path,
    registry_path: str | Path,
    config_identity_path: str | Path,
    checkpoint_source: str | Path,
    checkpoint_registry_id: str,
    checkpoint_authenticated_uri: str,
    checkpoint_receipt_path: str | Path,
    checkpoint_provider_receipt_path: str | Path,
    content_store_root: str | Path,
    data_root: str | Path,
) -> dict[str, Any]:
    """Generate registration while retaining the detached worktree directory fd."""

    git_root = root
    if _git(git_root, "status", "--porcelain"):
        raise ValueError("registration generation requires a clean repository")
    symbolic = _run_git(git_root, "symbolic-ref", "-q", "HEAD", text=True)
    if symbolic.returncode == 0:
        raise ValueError("registration generation requires detached HEAD")
    implementation = _git(git_root, "rev-parse", "HEAD")
    tree = _git(git_root, "rev-parse", "HEAD^{tree}")
    body = copy.deepcopy(dict(identity_template))
    body["implementation_commit"] = implementation
    body["registration_parent"] = {"commit": implementation, "tree": tree}
    source_files: dict[str, str] = {}
    for relative in REQUIRED_REGISTRATION_SOURCE_PATHS:
        _, digest = root.read_bytes(
            relative, label=f"registration generation source {relative}"
        )
        source_files[relative] = digest
    body["source_files"] = source_files
    spec_relative = REQUIRED_REGISTRATION_SOURCE_PATHS[0]
    body["spec"] = {
        "commit": APPROVED_SPEC_COMMIT,
        "sha256": APPROVED_SPEC_SHA256,
    }

    manifest_path, manifest, raw_manifest, manifest_digest = load_bound_json(
        manifest_path, label="registration generation window manifest"
    )
    registry_path, registry, _, _ = load_bound_json(
        registry_path, label="registration generation media registry"
    )
    config_identity_path, config, _, _ = load_bound_json(
        config_identity_path, label="registration generation config identity"
    )
    if raw_manifest != manifest_exact_bytes(manifest):
        raise ValueError("manifest input must use exact canonical bytes")
    manifest = validate_r2_manifest(manifest, registry=registry, config_identity=config)
    body["window_manifest"] = {
        "artifact": manifest,
        "exact_bytes_sha256": manifest_digest,
        "source_path": str(manifest_path),
        "registry_path": str(registry_path),
        "config_identity_path": str(config_identity_path),
    }
    with open_bound_directory(data_root, label="registration data root") as bound_data_root:
        data_root_path = bound_data_root.path
    body["data"] = {
        "root_identity": registry["data_sha256"],
        "root_path": data_root_path,
        "annotation_sha256": registry["annotation_sha256"],
        "media_sha256": {
            record["video_id"]: record["media_sha256"] for record in registry["records"]
        },
    }

    checkpoint_source, checkpoint_payload, checkpoint_sha = read_bound_bytes(
        checkpoint_source, label="registration dense checkpoint source"
    )
    checkpoint_size = len(checkpoint_payload)
    ensured_store = ensure_bound_directory(
        content_store_root, label="registration content store"
    )
    with open_bound_directory(
        ensured_store, label="registration content store"
    ) as bound_store:
        store = Path(bound_store.path)
    destination = store / "sha256" / checkpoint_sha
    publish_bytes_exclusive(
        destination,
        checkpoint_payload,
        label="content-addressed dense checkpoint",
        allow_existing_exact=True,
    )
    body["dense_checkpoint"] = {
        "sha256": checkpoint_sha,
        "bytes": checkpoint_size,
        "registry_id": checkpoint_registry_id,
        "authenticated_uri": checkpoint_authenticated_uri,
        "content_addressed_path": str(destination),
        "registry_receipt": {},
    }
    (
        checkpoint_receipt_path,
        receipt_artifact,
        raw_receipt,
        receipt_digest,
    ) = load_bound_json(
        checkpoint_receipt_path,
        label="registration checkpoint registry receipt",
    )
    checkpoint_provider_receipt_path, _, _ = read_bound_bytes(
        checkpoint_provider_receipt_path,
        label="registration checkpoint provider receipt",
    )
    if raw_receipt != canonical_json_bytes(receipt_artifact) + b"\n":
        raise ValueError("checkpoint registry receipt must use exact canonical bytes")
    receipt_artifact = validate_checkpoint_registry_receipt(
        receipt_artifact,
        provider_receipt_path=checkpoint_provider_receipt_path,
        registry_id=checkpoint_registry_id,
        authenticated_uri=checkpoint_authenticated_uri,
        content_sha256=checkpoint_sha,
        content_bytes=checkpoint_size,
    )
    body["dense_checkpoint"]["registry_receipt"] = {
        "artifact": receipt_artifact,
        "exact_bytes_sha256": receipt_digest,
        "source_path": str(checkpoint_receipt_path),
        "provider_receipt_path": str(checkpoint_provider_receipt_path),
    }
    body["exposures"] = {
        "stage_b": build_stage_b_exposure_artifact(manifest["splits"]["fit"]),
        "stage_c_formula": body["exposures"]["stage_c_formula"],
    }
    body["profiler"]["invocation_ids"] = [
        window_id
        for split in ("fit", "calibration", "evaluation")
        for window_id in manifest["splits"][split]
    ]
    body["profiler"]["invocation_order_sha256"] = canonical_sha256(
        body["profiler"]["invocation_ids"]
    )
    validate_formal_random_control_lock(body)
    for plan in body["profiler"]["candidate_plan"]:
        name = plan["candidate_name"]
        if not name.startswith("random_p"):
            continue
        period = int(name.rsplit("p", 1)[1])
        hashes = [
            canonical_sha256(
                random_exact_count_actions(
                    window_id,
                    seed=R2_RANDOM_CONTROL_SEED,
                    num_groups=3,
                    period=period,
                ).tolist()
            )
            for window_id in body["profiler"]["invocation_ids"]
        ]
        plan["requested_action_sha256_by_invocation"] = hashes
        plan["requested_action_order_sha256"] = canonical_sha256(hashes)
    body["profiler"]["model_config_sha256"] = manifest["config_identity"]["config_sha256"]
    registration = build_pre_gate1_registration(body)
    _validate_repository_context_bound(
        registration,
        root=root,
        context_mode="generation",
        registration_commit=None,
        registration_relpath=None,
    )
    return registration


def build_pre_gate1_registration(identity: Mapping[str, Any]) -> dict[str, Any]:
    body = _validate_identity(identity)
    registration = {"schema": REGISTRATION_SCHEMA, **body}
    registration["registration_sha256"] = canonical_sha256(registration)
    return registration


def validate_formal_random_control_lock(
    registration: Mapping[str, Any],
) -> None:
    """Accept only the three unsuffixed random controls frozen to integer 3407."""

    profiler = registration.get("profiler") if isinstance(registration, Mapping) else None
    plans = profiler.get("candidate_plan") if isinstance(profiler, Mapping) else None
    if not isinstance(plans, list):
        raise ValueError("formal random-control lock requires a profiler candidate plan")
    expected_names = {"random_p2", "random_p4", "random_p8"}
    random_plans = {
        str(plan.get("candidate_name")): plan
        for plan in plans
        if isinstance(plan, Mapping)
        and str(plan.get("candidate_name", "")).startswith("random_p")
    }
    if set(random_plans) != expected_names:
        raise ValueError("formal random controls must equal random_p2/p4/p8")
    for name, plan in random_plans.items():
        factory_config = plan.get("factory_config")
        seed = (
            factory_config.get("control_seed")
            if isinstance(factory_config, Mapping)
            else None
        )
        if isinstance(seed, bool) or type(seed) is not int or seed != R2_RANDOM_CONTROL_SEED:
            raise ValueError(f"formal random control {name} control_seed must equal integer 3407")
    for plan in plans:
        if not isinstance(plan, Mapping):
            raise TypeError("formal profiler candidate plans must be mappings")
        name = str(plan.get("candidate_name", ""))
        factory_config = plan.get("factory_config")
        if (
            name not in expected_names
            and isinstance(factory_config, Mapping)
            and "control_seed" in factory_config
        ):
            raise ValueError("control_seed is forbidden outside random_p2/p4/p8")


def resolve_gate1_output_root(
    registration: Mapping[str, Any],
    registration_commit: str,
) -> Path:
    """Resolve the sole formal Gate-1 directory from registered base and R."""

    _require_commit(registration_commit, "registration commit R")
    output = registration.get("output_root")
    if not isinstance(output, Mapping) or set(output) != {"base", "template"}:
        raise ValueError("registration output_root fields mismatch")
    if output["template"] != "{base}/{registration_commit}/shared/gate1":
        raise ValueError("registration output_root template mismatch")
    base_text = _require_nonempty_string(output["base"], "output_root.base")
    if base_text != FORMAL_OUTPUT_BASE or not base_text.startswith("/"):
        raise ValueError("registered Gate 1 output base is not the frozen absolute path")
    base = Path(base_text)
    derived = base / registration_commit / "shared" / "gate1"
    return derived


def validate_pre_gate1_registration(
    registration: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    context_mode: str | None = None,
    registration_commit: str | None = None,
    registration_relpath: str | None = None,
) -> dict[str, Any]:
    if not isinstance(registration, Mapping):
        raise TypeError("registration must be a mapping")
    expected = {"schema", *REQUIRED_FIELDS, "registration_sha256"}
    _require_exact_fields(registration, expected, "registration")
    payload = dict(registration)
    digest = payload.pop("registration_sha256")
    if payload.pop("schema") != REGISTRATION_SCHEMA:
        raise ValueError("unsupported registration schema")
    rebuilt = build_pre_gate1_registration(payload)
    if digest != rebuilt["registration_sha256"]:
        raise ValueError("registration SHA-256 mismatch")
    if dict(registration) != rebuilt:
        raise ValueError("registration differs from canonical rebuilt artifact")
    if repository_root is not None or context_mode is not None:
        if repository_root is None or context_mode is None:
            raise ValueError("repository_root and context_mode must be supplied together")
        _validate_repository_context(
            rebuilt,
            repository_root=repository_root,
            context_mode=context_mode,
            registration_commit=registration_commit,
            registration_relpath=registration_relpath,
        )
    return rebuilt


def require_formal_gate1_context_arguments(
    *,
    repository_root: str | Path,
    registration_commit: str,
    registration_relpath: str,
) -> None:
    if repository_root is None or not str(repository_root).strip():
        raise ValueError("formal Gate 1 repository_root must be non-empty")
    if registration_commit is None or not str(registration_commit).strip():
        raise ValueError("formal Gate 1 registration_commit must be non-empty")
    if registration_relpath is None or not str(registration_relpath).strip():
        raise ValueError("formal Gate 1 registration_relpath must be non-empty")


def validate_formal_gate1_context(
    registration: Mapping[str, Any],
    *,
    repository_root: str | Path,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    """Deep-validate the complete immutable formal Gate-1 repository context."""

    require_formal_gate1_context_arguments(
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    validated = validate_pre_gate1_registration(
        registration,
        repository_root=repository_root,
        context_mode="formal",
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    # Keep the lock explicit at every formal public boundary even though the
    # formal repository validator also invokes it before touching Git state.
    validate_formal_random_control_lock(validated)
    return validated


def claim_flags(
    *, gate1: bool = False, gate2: bool = False, gate3: bool = False, gate4: bool = False
) -> dict[str, bool]:
    if any(type(value) is not bool for value in (gate1, gate2, gate3, gate4)):
        raise TypeError("claim gate flags must be booleans")
    if gate4 and not (gate1 and gate2 and gate3):
        raise ValueError("Gate 4 claim cannot unlock without Gates 1--3")
    if gate3 and not (gate1 and gate2):
        raise ValueError("Gate 3 claim cannot unlock without Gates 1--2")
    if gate2 and not gate1:
        raise ValueError("Gate 2 claim cannot unlock without Gate 1")
    return {
        "oracle_headroom": gate1,
        "mechanism": gate2,
        "calibrated_risk_on_frozen_window_protocol": gate3,
        "metric_adatad_thumos14_official_full_video": gate4,
        "latency_slurm_single_device_fixed_stack": gate4,
        "deploy": False,
        "paper": False,
    }
