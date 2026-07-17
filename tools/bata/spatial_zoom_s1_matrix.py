from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.bata.spatial_zoom_s1_contract import (
    build_s1_profile_order,
    canonical_sha256,
    sha256_file,
)


S1_PROFILE_MATRIX_START_SCHEMA = "spatial_zoom_s1_profile_matrix_start_v1"
S1_PROFILE_MATRIX_COMPLETION_SCHEMA = "spatial_zoom_s1_profile_matrix_completion_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")

_START_FIELDS = {
    "schema_version",
    "matrix_start_receipt_path",
    "profile_recovery_certificate_path",
    "profile_recovery_certificate_file_sha256",
    "profile_recovery_certificate_sha256",
    "profile_recovery_campaign_id",
    "sidecar_gate_evidence_path",
    "sidecar_gate_evidence_file_sha256",
    "sidecar_gate_sha256",
    "profile_code_commit",
    "frozen_order",
    "profile_order_sha256",
    "slurm_job_id",
    "slurm_step_id",
    "slurm_job_gpus",
    "slurm_step_gpus",
    "scoped_gpu_id",
    "step_gpu_uuid",
    "cuda_visible_devices",
    "slurm_cpus_per_task",
    "effective_step_memory_limit_mb",
    "outer_job_mem_per_node_mb",
    "hardware_identity",
    "hardware_fingerprint",
    "software_fingerprint",
    "matrix_sha256",
}

_COMPLETION_FIELDS = {
    "schema_version",
    "matrix_completion_receipt_path",
    "profile_recovery_certificate_path",
    "profile_recovery_certificate_file_sha256",
    "profile_recovery_certificate_sha256",
    "profile_recovery_campaign_id",
    "matrix_start_receipt_path",
    "matrix_start_receipt_file_sha256",
    "matrix_sha256",
    "slurm_job_id",
    "slurm_step_id",
    "scoped_gpu_id",
    "step_gpu_uuid",
    "frozen_order",
    "profile_order_sha256",
    "descriptor_count",
    "descriptors",
    "descriptor_manifest_sha256",
    "completion_sha256",
}

_DESCRIPTOR_MATRIX_FIELDS = {
    "matrix_start_receipt_path",
    "matrix_start_receipt_file_sha256",
    "matrix_sha256",
    "slurm_job_id",
    "slurm_step_id",
    "step_gpu_uuid",
}


def _json_clone(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not finite JSON") from exc


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    return _json_clone(value, label=label)


def _require_canonical_json(
    path: Path, value: Mapping[str, Any], *, label: str
) -> None:
    expected = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if path.read_text(encoding="utf-8") != expected:
        raise ValueError(f"{label} is not canonical JSON: {path}")


def _require_sha256(value: Any, *, label: str) -> str:
    checked = str(value)
    if not _SHA256_RE.fullmatch(checked):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return checked


def _require_commit(value: Any, *, label: str) -> str:
    checked = str(value).lower()
    if not _COMMIT_RE.fullmatch(checked):
        raise ValueError(f"{label} is not a Git commit")
    return checked


def _require_positive_int(value: Any, *, label: str) -> int:
    try:
        checked = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if checked <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return checked


def _require_nonempty(value: Any, *, label: str) -> str:
    checked = str(value).strip()
    if not checked:
        raise ValueError(f"{label} must be non-empty")
    return checked


def _parse_gpu_list(value: Any, *, label: str) -> tuple[str, ...]:
    raw = _require_nonempty(value, label=label)
    tokens = tuple(field.strip() for field in raw.split(",") if field.strip())
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError(f"{label} must contain unique comma-separated GPU ids")
    return tokens


def _raw_mapping(
    source: Mapping[str, Any] | str | os.PathLike[str], *, label: str
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        return _json_clone(source, label=label), None
    path = Path(source).resolve()
    return _read_json_mapping(path, label=label), path


def _validate_self_hash(
    value: Mapping[str, Any], *, hash_key: str, label: str
) -> dict[str, Any]:
    checked = _json_clone(value, label=label)
    internal_hash = checked.pop(hash_key, None)
    _require_sha256(internal_hash, label=f"{label} {hash_key}")
    if canonical_sha256(checked) != internal_hash:
        raise ValueError(f"{label} self-hash mismatch")
    checked[hash_key] = internal_hash
    return checked


def _recovery_binding(recovery: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "code_commit": "training_code_commit",
        "experiment_namespace": "experiment_namespace",
        "canonical_experiment_root": "canonical_experiment_root",
        "manifest_sha256": "manifest_sha256",
        "protocol_fingerprint": "protocol_fingerprint",
        "precheck_file_sha256": "precheck_file_sha256",
        "precheck_sha256": "precheck_sha256",
        "pretrained_checkpoint_sha256": "pretrained_checkpoint_sha256",
    }
    missing = [source for source in fields.values() if source not in recovery]
    if missing:
        raise ValueError(
            "S1 profile recovery lacks binding fields: " + ", ".join(missing)
        )
    return {target: recovery[source] for target, source in fields.items()}


def _canonical_recovery_path(recovery: Mapping[str, Any]) -> Path:
    campaign_root = _require_nonempty(
        recovery.get("campaign_root"), label="recovery campaign_root"
    )
    return (Path(campaign_root) / "recovery_certificate.json").resolve()


def _validated_recovery(
    recovery: Mapping[str, Any] | str | os.PathLike[str],
) -> tuple[dict[str, Any], Path]:
    raw, supplied_path = _raw_mapping(recovery, label="S1 profile recovery")
    raw = _validate_self_hash(
        raw,
        hash_key="certificate_sha256",
        label="S1 profile recovery",
    )
    canonical_path = _canonical_recovery_path(raw)
    if supplied_path is not None and supplied_path != canonical_path:
        raise ValueError("S1 profile recovery is outside its exact campaign path")

    # Lazy import keeps this evidence module independent from profile entrypoints.
    from tools.bata.spatial_zoom_s1_profile_recovery import (
        load_profile_recovery_certificate,
    )

    checked = load_profile_recovery_certificate(
        canonical_path,
        binding=_recovery_binding(raw),
        verify_checkout=False,
    )
    if checked != raw:
        raise ValueError("S1 profile recovery changed during validation")
    return checked, canonical_path


def _validated_gate(
    gate: Mapping[str, Any] | str | os.PathLike[str],
    *,
    recovery: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    raw, supplied_path = _raw_mapping(gate, label="S1 sidecar Gate")
    raw = _validate_self_hash(
        raw,
        hash_key="gate_sha256",
        label="S1 sidecar Gate",
    )

    # Lazy import avoids a matrix -> profile -> matrix import cycle.
    from tools.bata.spatial_zoom_s1_sidecar_gate import (
        load_sidecar_gate_evidence,
        sidecar_gate_path,
    )

    canonical_path = sidecar_gate_path(recovery)
    if supplied_path is not None and supplied_path != canonical_path:
        raise ValueError("S1 sidecar Gate is outside its exact campaign path")
    checked = load_sidecar_gate_evidence(canonical_path, recovery=recovery)
    if checked != raw:
        raise ValueError("S1 sidecar Gate changed during validation")
    return checked, canonical_path


def _normalized_order(order: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    if isinstance(order, (str, bytes)) or not isinstance(order, Sequence):
        raise TypeError("S1 matrix frozen_order must be a sequence")
    normalized: list[dict[str, int]] = []
    for row in order:
        if not isinstance(row, Mapping) or set(row) != {
            "ordinal",
            "resolution",
            "seed",
        }:
            raise ValueError("S1 matrix frozen_order row is malformed")
        normalized.append(
            {
                "ordinal": int(row["ordinal"]),
                "resolution": int(row["resolution"]),
                "seed": int(row["seed"]),
            }
        )
    expected = build_s1_profile_order()
    if normalized != expected:
        raise ValueError("S1 matrix frozen_order differs from the registered order")
    return normalized


def _hardware_runtime_snapshot(
    hardware_identity: Mapping[str, Any],
    *,
    env: Mapping[str, Any],
    effective_memory_limit_mb: int | None = None,
) -> dict[str, Any]:
    hardware = _json_clone(hardware_identity, label="S1 hardware identity")
    runtime_env = {str(key): str(value) for key, value in env.items()}
    job_id = _require_nonempty(runtime_env.get("SLURM_JOB_ID"), label="SLURM_JOB_ID")
    if not job_id.isdigit():
        raise ValueError("SLURM_JOB_ID must be numeric")
    step_id = _require_nonempty(runtime_env.get("SLURM_STEP_ID"), label="SLURM_STEP_ID")
    job_gpus = _parse_gpu_list(
        runtime_env.get("SLURM_JOB_GPUS"), label="SLURM_JOB_GPUS"
    )
    step_gpus = _parse_gpu_list(
        runtime_env.get("SLURM_STEP_GPUS"), label="SLURM_STEP_GPUS"
    )
    if len(step_gpus) != 1:
        raise ValueError("S1 matrix requires exactly one Slurm step GPU")
    scoped_gpu_id = step_gpus[0]
    if scoped_gpu_id not in job_gpus:
        raise ValueError("S1 matrix step GPU is not a member of SLURM_JOB_GPUS")
    cuda_visible_devices = _require_nonempty(
        runtime_env.get("CUDA_VISIBLE_DEVICES"), label="CUDA_VISIBLE_DEVICES"
    )
    if len(_parse_gpu_list(cuda_visible_devices, label="CUDA_VISIBLE_DEVICES")) != 1:
        raise ValueError("S1 matrix requires exactly one CUDA-visible GPU")
    cpus_per_task = _require_positive_int(
        runtime_env.get("SLURM_CPUS_PER_TASK"), label="SLURM_CPUS_PER_TASK"
    )
    outer_memory = _require_positive_int(
        runtime_env.get("SLURM_MEM_PER_NODE"), label="SLURM_MEM_PER_NODE"
    )

    slurm = hardware.get("slurm_resources")
    nvidia_smi = hardware.get("nvidia_smi")
    if not isinstance(slurm, Mapping) or not isinstance(nvidia_smi, Mapping):
        raise ValueError("S1 hardware identity lacks Slurm or NVIDIA evidence")
    hardware_job_id = _require_nonempty(
        slurm.get("slurm_job_id"), label="hardware slurm_job_id"
    )
    hardware_step_id = _require_nonempty(
        slurm.get("slurm_step_id"), label="hardware slurm_step_id"
    )
    hardware_job_gpus = _parse_gpu_list(
        slurm.get("slurm_job_gpus"), label="hardware slurm_job_gpus"
    )
    hardware_step_gpus = _parse_gpu_list(
        slurm.get("slurm_step_gpus"), label="hardware slurm_step_gpus"
    )
    hardware_identity_checks = {
        "slurm_job_id": (hardware_job_id, job_id),
        "slurm_step_id": (hardware_step_id, step_id),
        "slurm_job_gpus": (hardware_job_gpus, job_gpus),
        "slurm_step_gpus": (hardware_step_gpus, step_gpus),
    }
    for field, (observed, expected) in hardware_identity_checks.items():
        if observed != expected:
            raise ValueError(f"S1 hardware {field} differs from Slurm env")
    if (
        _require_positive_int(
            slurm.get("cpus_per_task"), label="hardware cpus_per_task"
        )
        != cpus_per_task
    ):
        raise ValueError("S1 hardware CPU allocation differs from Slurm step")

    hardware_effective_memory = _require_positive_int(
        slurm.get("effective_step_memory_limit_mb"),
        label="hardware effective_step_memory_limit_mb",
    )
    effective_memory = (
        hardware_effective_memory
        if effective_memory_limit_mb is None
        else _require_positive_int(
            effective_memory_limit_mb,
            label="effective_step_memory_limit_mb",
        )
    )
    if hardware_effective_memory != effective_memory:
        raise ValueError("S1 hardware effective memory differs from step cgroup")
    hardware_outer_memory = _require_positive_int(
        slurm.get("outer_job_mem_per_node_mb"),
        label="hardware outer_job_mem_per_node_mb",
    )
    if hardware_outer_memory != outer_memory:
        raise ValueError("S1 hardware outer memory differs from Slurm job")

    gpu_id_candidates = [
        hardware.get("scoped_gpu_id"),
        hardware.get("physical_gpu_id"),
        slurm.get("scoped_gpu_id"),
        slurm.get("step_gpu_id"),
    ]
    gpu_id_candidates = [str(value) for value in gpu_id_candidates if value is not None]
    if not gpu_id_candidates or any(
        value != scoped_gpu_id for value in gpu_id_candidates
    ):
        raise ValueError("S1 hardware step GPU id differs from Slurm step")

    step_gpu_uuid = _require_nonempty(
        nvidia_smi.get("uuid"), label="hardware NVIDIA GPU UUID"
    )
    if not step_gpu_uuid.startswith("GPU-"):
        raise ValueError("S1 hardware NVIDIA GPU UUID is invalid")
    uuid_candidates = [
        hardware.get("step_gpu_uuid"),
        hardware.get("cuda_visible_device_uuid"),
        slurm.get("step_gpu_uuid"),
    ]
    if any(
        str(value) != step_gpu_uuid for value in uuid_candidates if value is not None
    ):
        raise ValueError("S1 hardware GPU UUID fields disagree")
    hardware_cvd = _require_nonempty(
        slurm.get("cuda_visible_devices"),
        label="hardware cuda_visible_devices",
    )
    top_level_cvd = hardware.get("cuda_visible_devices")
    if hardware_cvd != cuda_visible_devices or (
        top_level_cvd is not None and str(top_level_cvd) != cuda_visible_devices
    ):
        raise ValueError("S1 hardware CUDA visibility differs from Slurm step")

    return {
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "slurm_job_gpus": ",".join(job_gpus),
        "slurm_step_gpus": ",".join(step_gpus),
        "scoped_gpu_id": scoped_gpu_id,
        "step_gpu_uuid": step_gpu_uuid,
        "cuda_visible_devices": cuda_visible_devices,
        "slurm_cpus_per_task": cpus_per_task,
        "effective_step_memory_limit_mb": effective_memory,
        "outer_job_mem_per_node_mb": outer_memory,
    }


def _receipt_env(receipt: Mapping[str, Any]) -> dict[str, str]:
    return {
        "SLURM_JOB_ID": str(receipt["slurm_job_id"]),
        "SLURM_STEP_ID": str(receipt["slurm_step_id"]),
        "SLURM_JOB_GPUS": str(receipt["slurm_job_gpus"]),
        "SLURM_STEP_GPUS": str(receipt["slurm_step_gpus"]),
        "CUDA_VISIBLE_DEVICES": str(receipt["cuda_visible_devices"]),
        "SLURM_CPUS_PER_TASK": str(receipt["slurm_cpus_per_task"]),
        "SLURM_MEM_PER_NODE": str(receipt["outer_job_mem_per_node_mb"]),
    }


def _validate_gate_runtime(
    gate: Mapping[str, Any],
    *,
    hardware_identity: Mapping[str, Any],
    software_fingerprint: str,
) -> None:
    from tools.bata.spatial_zoom_s1_sidecar_gate import (
        validate_sidecar_gate_runtime_identity,
    )

    validate_sidecar_gate_runtime_identity(
        gate,
        hardware_identity=hardware_identity,
        software_fingerprint=software_fingerprint,
    )


def canonical_matrix_start_path(
    recovery: Mapping[str, Any] | str | os.PathLike[str],
) -> Path:
    raw, _ = _raw_mapping(recovery, label="S1 profile recovery")
    campaign_root = _require_nonempty(
        raw.get("campaign_root"), label="recovery campaign_root"
    )
    return (Path(campaign_root) / "matrix.lock" / "matrix.started.json").resolve()


def canonical_matrix_completion_path(
    recovery: Mapping[str, Any] | str | os.PathLike[str],
) -> Path:
    raw, _ = _raw_mapping(recovery, label="S1 profile recovery")
    campaign_root = _require_nonempty(
        raw.get("campaign_root"), label="recovery campaign_root"
    )
    return (Path(campaign_root) / "matrix.lock" / "matrix.completed.json").resolve()


def build_profile_matrix_start_receipt(
    *,
    recovery: Mapping[str, Any] | str | os.PathLike[str],
    sidecar_gate: Mapping[str, Any] | str | os.PathLike[str],
    hardware_identity: Mapping[str, Any],
    software_fingerprint: str,
    profile_code_commit: str,
    frozen_order: Sequence[Mapping[str, Any]],
    env: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked_recovery, recovery_path = _validated_recovery(recovery)
    checked_gate, gate_path = _validated_gate(sidecar_gate, recovery=checked_recovery)
    checked_order = _normalized_order(frozen_order)
    checked_software = _require_sha256(
        software_fingerprint, label="software_fingerprint"
    )
    checked_commit = _require_commit(profile_code_commit, label="profile_code_commit")
    if (
        _require_commit(
            checked_recovery.get("profile_code_commit"),
            label="recovery profile_code_commit",
        )
        != checked_commit
        or _require_commit(
            checked_gate.get("profile_code_commit"),
            label="Gate profile_code_commit",
        )
        != checked_commit
    ):
        raise ValueError("S1 matrix profile code commit differs from recovery or Gate")

    hardware = _json_clone(hardware_identity, label="S1 hardware identity")
    resources = _hardware_runtime_snapshot(
        hardware,
        env=os.environ if env is None else env,
    )
    _validate_gate_runtime(
        checked_gate,
        hardware_identity=hardware,
        software_fingerprint=checked_software,
    )
    receipt = {
        "schema_version": S1_PROFILE_MATRIX_START_SCHEMA,
        "matrix_start_receipt_path": str(canonical_matrix_start_path(checked_recovery)),
        "profile_recovery_certificate_path": str(recovery_path),
        "profile_recovery_certificate_file_sha256": sha256_file(recovery_path),
        "profile_recovery_certificate_sha256": checked_recovery["certificate_sha256"],
        "profile_recovery_campaign_id": checked_recovery["campaign_id"],
        "sidecar_gate_evidence_path": str(gate_path),
        "sidecar_gate_evidence_file_sha256": sha256_file(gate_path),
        "sidecar_gate_sha256": checked_gate["gate_sha256"],
        "profile_code_commit": checked_commit,
        "frozen_order": checked_order,
        "profile_order_sha256": canonical_sha256(checked_order),
        **resources,
        "hardware_identity": hardware,
        "hardware_fingerprint": canonical_sha256(hardware),
        "software_fingerprint": checked_software,
    }
    receipt["matrix_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_profile_matrix_start_receipt(
    receipt_or_path: Mapping[str, Any] | str | os.PathLike[str],
    *,
    recovery: Mapping[str, Any] | str | os.PathLike[str],
    verify_runtime: bool = False,
    hardware_identity: Mapping[str, Any] | None = None,
    software_fingerprint: str | None = None,
    effective_memory_limit_mb: int | None = None,
    env: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt, receipt_path = _raw_mapping(
        receipt_or_path, label="S1 matrix start receipt"
    )
    receipt = _validate_self_hash(
        receipt,
        hash_key="matrix_sha256",
        label="S1 matrix start receipt",
    )
    if set(receipt) != _START_FIELDS:
        raise ValueError("S1 matrix start receipt fields differ from its schema")

    checked_recovery, recovery_path = _validated_recovery(recovery)
    expected_receipt_path = canonical_matrix_start_path(checked_recovery)
    if Path(
        receipt["matrix_start_receipt_path"]
    ).resolve() != expected_receipt_path or (
        receipt_path is not None and receipt_path != expected_receipt_path
    ):
        raise ValueError("S1 matrix start receipt is outside its canonical path")
    if receipt_path is not None:
        _require_canonical_json(receipt_path, receipt, label="S1 matrix start receipt")

    checked_gate, gate_path = _validated_gate(
        sidecar_gate_path_from_recovery(checked_recovery),
        recovery=checked_recovery,
    )
    expected_static = {
        "schema_version": S1_PROFILE_MATRIX_START_SCHEMA,
        "profile_recovery_certificate_path": str(recovery_path),
        "profile_recovery_certificate_file_sha256": sha256_file(recovery_path),
        "profile_recovery_certificate_sha256": checked_recovery["certificate_sha256"],
        "profile_recovery_campaign_id": checked_recovery["campaign_id"],
        "sidecar_gate_evidence_path": str(gate_path),
        "sidecar_gate_evidence_file_sha256": sha256_file(gate_path),
        "sidecar_gate_sha256": checked_gate["gate_sha256"],
        "profile_code_commit": _require_commit(
            checked_recovery["profile_code_commit"],
            label="recovery profile_code_commit",
        ),
    }
    for key, expected in expected_static.items():
        if receipt.get(key) != expected:
            raise ValueError(f"S1 matrix start receipt {key} mismatch")

    checked_order = _normalized_order(receipt["frozen_order"])
    if receipt["profile_order_sha256"] != canonical_sha256(checked_order):
        raise ValueError("S1 matrix profile-order hash mismatch")
    checked_software = _require_sha256(
        receipt["software_fingerprint"], label="software_fingerprint"
    )
    hardware = _json_clone(
        receipt["hardware_identity"], label="S1 receipt hardware identity"
    )
    if receipt["hardware_fingerprint"] != canonical_sha256(hardware):
        raise ValueError("S1 matrix hardware fingerprint mismatch")
    recorded_resources = _hardware_runtime_snapshot(
        hardware,
        env=_receipt_env(receipt),
        effective_memory_limit_mb=receipt["effective_step_memory_limit_mb"],
    )
    for key, expected in recorded_resources.items():
        if receipt.get(key) != expected:
            raise ValueError(f"S1 matrix start receipt {key} mismatch")
    _validate_gate_runtime(
        checked_gate,
        hardware_identity=hardware,
        software_fingerprint=checked_software,
    )

    if verify_runtime:
        if hardware_identity is None or software_fingerprint is None:
            raise ValueError(
                "S1 matrix runtime validation requires hardware and software"
            )
        runtime_hardware = _json_clone(
            hardware_identity, label="S1 runtime hardware identity"
        )
        runtime_software = _require_sha256(
            software_fingerprint, label="runtime software_fingerprint"
        )
        runtime_resources = _hardware_runtime_snapshot(
            runtime_hardware,
            env=os.environ if env is None else env,
            effective_memory_limit_mb=effective_memory_limit_mb,
        )
        for key, expected in runtime_resources.items():
            if receipt.get(key) != expected:
                raise ValueError(f"S1 matrix runtime {key} mismatch")
        if (
            canonical_sha256(runtime_hardware) != receipt["hardware_fingerprint"]
            or runtime_software != checked_software
        ):
            raise ValueError("S1 matrix runtime hardware or software mismatch")
        _validate_gate_runtime(
            checked_gate,
            hardware_identity=runtime_hardware,
            software_fingerprint=runtime_software,
        )
    return receipt


def sidecar_gate_path_from_recovery(recovery: Mapping[str, Any]) -> Path:
    from tools.bata.spatial_zoom_s1_sidecar_gate import sidecar_gate_path

    return sidecar_gate_path(recovery)


def _validated_descriptor(
    path: Path,
    *,
    start: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _read_json_mapping(path, label="S1 run descriptor")
    _require_canonical_json(path, descriptor, label="S1 run descriptor")
    descriptor = _validate_self_hash(
        descriptor,
        hash_key="descriptor_sha256",
        label="S1 run descriptor",
    )
    missing = sorted(_DESCRIPTOR_MATRIX_FIELDS - set(descriptor))
    if missing:
        raise ValueError(
            "S1 descriptor lacks matrix receipt fields: " + ", ".join(missing)
        )
    start_path = Path(start["matrix_start_receipt_path"]).resolve()
    expected = {
        "matrix_start_receipt_path": str(start_path),
        "matrix_start_receipt_file_sha256": sha256_file(start_path),
        "matrix_sha256": start["matrix_sha256"],
        "slurm_job_id": start["slurm_job_id"],
        "slurm_step_id": start["slurm_step_id"],
        "step_gpu_uuid": start["step_gpu_uuid"],
    }
    for key, value in expected.items():
        if descriptor.get(key) != value:
            raise ValueError(f"S1 descriptor {key} does not bind matrix start")
    try:
        ordinal = int(descriptor["profile_order_ordinal"])
        resolution = int(descriptor["resolution"])
        seed = int(descriptor["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("S1 descriptor cell identity is invalid") from exc
    record = {
        "profile_order_ordinal": ordinal,
        "resolution": resolution,
        "seed": seed,
        "descriptor_path": str(path),
        "descriptor_file_sha256": sha256_file(path),
        "descriptor_sha256": descriptor["descriptor_sha256"],
    }
    return descriptor, record


def _collect_descriptors(
    descriptor_paths: Sequence[str | os.PathLike[str]],
    *,
    start: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(descriptor_paths, (str, bytes)) or not isinstance(
        descriptor_paths, Sequence
    ):
        raise TypeError("S1 descriptor_paths must be a sequence")
    paths = [Path(path).resolve() for path in descriptor_paths]
    if len(paths) != 9:
        raise ValueError("S1 matrix completion requires exactly 9 descriptors")
    if len(set(paths)) != len(paths):
        raise ValueError("S1 matrix completion contains duplicate descriptor paths")

    records = [_validated_descriptor(path, start=start)[1] for path in paths]
    cells = [
        (row["profile_order_ordinal"], row["resolution"], row["seed"])
        for row in records
    ]
    if len(set(cells)) != len(cells):
        raise ValueError("S1 matrix completion contains duplicate descriptor cells")
    records.sort(key=lambda row: row["profile_order_ordinal"])
    expected_cells = [
        (row["ordinal"], row["resolution"], row["seed"])
        for row in start["frozen_order"]
    ]
    actual_cells = [
        (row["profile_order_ordinal"], row["resolution"], row["seed"])
        for row in records
    ]
    if actual_cells != expected_cells:
        raise ValueError("S1 matrix descriptors do not cover the frozen 3x3 order")
    return records


def build_profile_matrix_completion_receipt(
    *,
    start_receipt_path: str | os.PathLike[str],
    recovery: Mapping[str, Any] | str | os.PathLike[str],
    descriptor_paths: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    checked_recovery, recovery_path = _validated_recovery(recovery)
    start_path = Path(start_receipt_path).resolve()
    if start_path != canonical_matrix_start_path(checked_recovery):
        raise ValueError("S1 matrix start receipt path is not canonical")
    start = validate_profile_matrix_start_receipt(start_path, recovery=checked_recovery)
    records = _collect_descriptors(descriptor_paths, start=start)
    receipt = {
        "schema_version": S1_PROFILE_MATRIX_COMPLETION_SCHEMA,
        "matrix_completion_receipt_path": str(
            canonical_matrix_completion_path(checked_recovery)
        ),
        "profile_recovery_certificate_path": str(recovery_path),
        "profile_recovery_certificate_file_sha256": sha256_file(recovery_path),
        "profile_recovery_certificate_sha256": checked_recovery["certificate_sha256"],
        "profile_recovery_campaign_id": checked_recovery["campaign_id"],
        "matrix_start_receipt_path": str(start_path),
        "matrix_start_receipt_file_sha256": sha256_file(start_path),
        "matrix_sha256": start["matrix_sha256"],
        "slurm_job_id": start["slurm_job_id"],
        "slurm_step_id": start["slurm_step_id"],
        "scoped_gpu_id": start["scoped_gpu_id"],
        "step_gpu_uuid": start["step_gpu_uuid"],
        "frozen_order": start["frozen_order"],
        "profile_order_sha256": start["profile_order_sha256"],
        "descriptor_count": len(records),
        "descriptors": records,
        "descriptor_manifest_sha256": canonical_sha256(records),
    }
    receipt["completion_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_profile_matrix_completion_receipt(
    receipt_or_path: Mapping[str, Any] | str | os.PathLike[str],
    *,
    recovery: Mapping[str, Any] | str | os.PathLike[str],
    descriptor_paths: Sequence[str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    receipt, receipt_path = _raw_mapping(
        receipt_or_path, label="S1 matrix completion receipt"
    )
    receipt = _validate_self_hash(
        receipt,
        hash_key="completion_sha256",
        label="S1 matrix completion receipt",
    )
    if set(receipt) != _COMPLETION_FIELDS:
        raise ValueError("S1 matrix completion receipt fields differ from its schema")

    checked_recovery, recovery_path = _validated_recovery(recovery)
    expected_receipt_path = canonical_matrix_completion_path(checked_recovery)
    if Path(
        receipt["matrix_completion_receipt_path"]
    ).resolve() != expected_receipt_path or (
        receipt_path is not None and receipt_path != expected_receipt_path
    ):
        raise ValueError("S1 matrix completion receipt is outside its canonical path")
    if receipt_path is not None:
        _require_canonical_json(
            receipt_path, receipt, label="S1 matrix completion receipt"
        )

    start_path = canonical_matrix_start_path(checked_recovery)
    start = validate_profile_matrix_start_receipt(start_path, recovery=checked_recovery)
    expected_static = {
        "schema_version": S1_PROFILE_MATRIX_COMPLETION_SCHEMA,
        "profile_recovery_certificate_path": str(recovery_path),
        "profile_recovery_certificate_file_sha256": sha256_file(recovery_path),
        "profile_recovery_certificate_sha256": checked_recovery["certificate_sha256"],
        "profile_recovery_campaign_id": checked_recovery["campaign_id"],
        "matrix_start_receipt_path": str(start_path),
        "matrix_start_receipt_file_sha256": sha256_file(start_path),
        "matrix_sha256": start["matrix_sha256"],
        "slurm_job_id": start["slurm_job_id"],
        "slurm_step_id": start["slurm_step_id"],
        "scoped_gpu_id": start["scoped_gpu_id"],
        "step_gpu_uuid": start["step_gpu_uuid"],
        "frozen_order": start["frozen_order"],
        "profile_order_sha256": start["profile_order_sha256"],
        "descriptor_count": 9,
    }
    for key, expected in expected_static.items():
        if receipt.get(key) != expected:
            raise ValueError(f"S1 matrix completion receipt {key} mismatch")

    recorded_rows = receipt.get("descriptors")
    if not isinstance(recorded_rows, list) or len(recorded_rows) != 9:
        raise ValueError("S1 matrix completion receipt lacks 9 descriptor rows")
    recorded_paths = []
    for row in recorded_rows:
        if not isinstance(row, Mapping) or "descriptor_path" not in row:
            raise ValueError("S1 matrix completion descriptor row is malformed")
        recorded_paths.append(Path(row["descriptor_path"]).resolve())
    if len(set(recorded_paths)) != len(recorded_paths):
        raise ValueError("S1 matrix completion receipt repeats descriptor paths")
    rebuilt_rows = _collect_descriptors(recorded_paths, start=start)
    if rebuilt_rows != recorded_rows:
        raise ValueError("S1 matrix completion descriptor evidence changed")
    if receipt["descriptor_manifest_sha256"] != canonical_sha256(rebuilt_rows):
        raise ValueError("S1 matrix completion descriptor manifest mismatch")

    if descriptor_paths is not None:
        supplied_paths = [Path(path).resolve() for path in descriptor_paths]
        if (
            len(supplied_paths) != 9
            or len(set(supplied_paths)) != 9
            or set(supplied_paths) != set(recorded_paths)
        ):
            raise ValueError(
                "S1 matrix completion input descriptors do not match the receipt"
            )
    return receipt


__all__ = [
    "S1_PROFILE_MATRIX_COMPLETION_SCHEMA",
    "S1_PROFILE_MATRIX_START_SCHEMA",
    "build_profile_matrix_completion_receipt",
    "build_profile_matrix_start_receipt",
    "canonical_matrix_completion_path",
    "canonical_matrix_start_path",
    "validate_profile_matrix_completion_receipt",
    "validate_profile_matrix_start_receipt",
]
