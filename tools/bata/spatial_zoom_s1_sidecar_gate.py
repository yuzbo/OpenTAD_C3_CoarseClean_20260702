from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tools.bata.spatial_zoom_s1_contract import (
    atomic_publish_json,
    canonical_sha256,
    sha256_file,
)
from tools.bata.spatial_zoom_s1_cost import validate_profile_summary
from tools.bata.spatial_zoom_s1_power import (
    S1_POWER_SIDECAR_BACKEND,
    validate_nvml_sidecar_attempt,
)
from tools.bata.spatial_zoom_s1_profile_recovery import (
    S1_SIDECAR_RECOVERY_REASON,
)


S1_SIDECAR_GATE_SCHEMA = "spatial_zoom_s1_power_sidecar_gate_v1"


def sidecar_gate_path(recovery: Mapping[str, Any]) -> Path:
    if recovery.get("reason") != S1_SIDECAR_RECOVERY_REASON:
        raise ValueError("S1 sidecar Gate requires a v3 recovery certificate")
    return (
        Path(recovery["campaign_root"])
        / str(recovery["sidecar_gate_relative_path"])
    ).resolve()


def sidecar_gate_profile_prefix(recovery: Mapping[str, Any]) -> Path:
    return (
        Path(recovery["campaign_root"])
        / "sidecar_gate"
        / "dense256_seed3408_long_full_path"
    ).resolve()


def sidecar_gate_hardware_class(hardware_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Keep stable hardware/resource class fields while excluding physical IDs."""

    identity = dict(hardware_identity)
    nvidia_smi = dict(identity.get("nvidia_smi", {}))
    slurm = dict(identity.get("slurm_resources", {}))
    return {
        "machine": identity.get("machine"),
        "cpu_model": identity.get("cpu_model"),
        "gpu_name": identity.get("gpu_name"),
        "gpu_total_memory": identity.get("gpu_total_memory"),
        "gpu_compute_capability": identity.get("gpu_compute_capability"),
        "gpu_multi_processor_count": identity.get("gpu_multi_processor_count"),
        "driver_version": nvidia_smi.get("driver_version"),
        "persistence_mode": nvidia_smi.get("persistence_mode"),
        "compute_mode": nvidia_smi.get("compute_mode"),
        "power_limit": nvidia_smi.get("power.limit"),
        "max_sm_clock": nvidia_smi.get("clocks.max.sm"),
        "max_memory_clock": nvidia_smi.get("clocks.max.memory"),
        "cpus_per_task": slurm.get("cpus_per_task"),
        "effective_step_memory_limit_mb": slurm.get(
            "effective_step_memory_limit_mb"
        ),
    }


def validate_sidecar_gate_runtime_identity(
    evidence: Mapping[str, Any],
    *,
    hardware_identity: Mapping[str, Any],
    software_fingerprint: str,
) -> dict[str, Any]:
    """Require the matrix runtime to match the Gate's stable execution class."""

    runtime_hardware_class = sidecar_gate_hardware_class(hardware_identity)
    if (
        runtime_hardware_class != evidence.get("hardware_class")
        or canonical_sha256(runtime_hardware_class)
        != evidence.get("hardware_class_sha256")
    ):
        raise ValueError("S1 matrix runtime differs from the sidecar Gate hardware class")
    if str(software_fingerprint) != evidence.get("software_fingerprint"):
        raise ValueError("S1 matrix runtime differs from the sidecar Gate software")
    return runtime_hardware_class


def _load_self_hashed_marker(path: Path) -> dict[str, Any]:
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker_hash = marker.pop("marker_sha256", None)
    if not marker_hash or canonical_sha256(marker) != marker_hash:
        raise ValueError("S1 sidecar Gate marker self-hash mismatch")
    marker["marker_sha256"] = marker_hash
    return marker


def build_sidecar_gate_evidence(
    *,
    recovery: Mapping[str, Any],
    profile_report: Mapping[str, Any],
    marker_path: str | Path,
    attempt_report_path: str | Path,
    attempt_trace_path: str | Path,
    test_evidence_path: str | Path,
    test_evidence_file_sha256_before: str,
    slurm_job_id: str,
) -> dict[str, Any]:
    """Build compact evidence for a full-path, no-new-test-open cadence Gate."""

    if recovery.get("reason") != S1_SIDECAR_RECOVERY_REASON:
        raise ValueError("S1 sidecar Gate requires a v3 recovery certificate")
    if not str(slurm_job_id).isdigit():
        raise ValueError("S1 sidecar Gate requires a numeric Slurm job id")
    profile = validate_profile_summary(profile_report)
    marker_path = Path(marker_path).resolve()
    attempt_report_path = Path(attempt_report_path).resolve()
    attempt_trace_path = Path(attempt_trace_path).resolve()
    test_evidence_path = Path(test_evidence_path).resolve()
    for path in (
        marker_path,
        attempt_report_path,
        attempt_trace_path,
        test_evidence_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    hardware_identity = profile.get("hardware_identity", {})
    gate_gpu_uuid = str(
        dict(hardware_identity.get("nvidia_smi", {})).get("uuid", "")
    )
    attempt = validate_nvml_sidecar_attempt(
        attempt_report_path,
        attempt_trace_path,
        expected_uuid=gate_gpu_uuid,
        require_pass=True,
    )
    marker = _load_self_hashed_marker(marker_path)
    test_evidence_file_sha256_after = sha256_file(test_evidence_path)
    if test_evidence_file_sha256_after != str(test_evidence_file_sha256_before):
        raise ValueError("S1 sidecar Gate changed the existing test evidence")
    if (
        int(profile.get("resolution", -1)) != 256
        or int(profile.get("seed", -1)) != 3408
        or int(profile.get("sample_count", -1))
        != int(recovery["expected_loader_exposure_count"])
        or int(profile.get("loader_exposure_count", -1))
        != int(recovery["expected_loader_exposure_count"])
        or int(profile.get("physical_window_count", -1))
        != int(recovery["expected_physical_window_count"])
        or profile.get("result_finalizer")
        != "opentad.cores.test_engine.gather_ddp_results"
    ):
        raise ValueError("S1 sidecar Gate did not execute the representative full path")
    if (
        attempt.get("status") != "PASS"
        or attempt.get("cadence", {}).get("formal_cadence_pass") is not True
        or float(attempt.get("cadence", {}).get("max_gap_ms", float("inf")))
        > float(recovery["power_max_gap_limit_ms"])
        or int(attempt.get("interval_ms", -1))
        != int(recovery["power_target_interval_ms"])
        or len(attempt.get("allocated_cpu_ids", ()))
        != int(recovery["allocated_cpu_count"])
        or len(attempt.get("detector_cpu_ids", ()))
        != int(recovery["detector_cpu_count"])
        or attempt.get("sidecar_cpu_id") in attempt.get("detector_cpu_ids", ())
        or set(attempt.get("detector_cpu_ids", ()))
        | {attempt.get("sidecar_cpu_id")}
        != set(attempt.get("allocated_cpu_ids", ()))
        or attempt.get("trace_file_sha256") != sha256_file(attempt_trace_path)
    ):
        raise ValueError("S1 sidecar Gate cadence or CPU isolation failed")
    canonical_prefix = sidecar_gate_profile_prefix(recovery)
    forbidden = (
        canonical_prefix.with_suffix(".summary.json"),
        canonical_prefix.with_suffix(".samples.jsonl"),
        canonical_prefix.with_suffix(".power.jsonl"),
        canonical_prefix.with_suffix(".comparison.json"),
    )
    if any(path.exists() for path in forbidden):
        raise ValueError("S1 sidecar Gate published a formal paper profile artifact")
    evidence = {
        "schema_version": S1_SIDECAR_GATE_SCHEMA,
        "status": "PASS",
        "paper_claim_allowed": False,
        "gate_only": True,
        "published_formal_profile": False,
        "reused_existing_test_evidence": True,
        "opened_new_test_evidence": False,
        "slurm_job_id": str(slurm_job_id),
        "profile_code_commit": recovery["profile_code_commit"],
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
        "resolution": 256,
        "seed": 3408,
        "checkpoint_sha256": profile["checkpoint_sha256"],
        "test_evidence_path": str(test_evidence_path),
        "test_evidence_file_sha256": test_evidence_file_sha256_after,
        "test_evidence_sha256": profile["test_evidence_sha256"],
        "loader_exposure_count": profile["loader_exposure_count"],
        "physical_window_count": profile["physical_window_count"],
        "sample_manifest_sha256": profile["sample_manifest_sha256"],
        "physical_window_manifest_sha256": profile[
            "physical_window_manifest_sha256"
        ],
        "result_finalizer": profile["result_finalizer"],
        "ephemeral_profile_sha256": profile["profile_sha256"],
        "hardware_fingerprint": profile["hardware_fingerprint"],
        "hardware_class": sidecar_gate_hardware_class(hardware_identity),
        "hardware_class_sha256": canonical_sha256(
            sidecar_gate_hardware_class(hardware_identity)
        ),
        "software_fingerprint": profile["software_fingerprint"],
        "gate_gpu_uuid": gate_gpu_uuid,
        "gpu_scope": "gate_uuid_bound_matrix_hardware_class_matched",
        "profile_marker_path": str(marker_path),
        "profile_marker_file_sha256": sha256_file(marker_path),
        "profile_marker_sha256": marker["marker_sha256"],
        "sidecar_attempt_report_path": str(attempt_report_path),
        "sidecar_attempt_report_file_sha256": sha256_file(attempt_report_path),
        "sidecar_attempt_sha256": attempt["attempt_sha256"],
        "sidecar_attempt_trace_path": str(attempt_trace_path),
        "sidecar_attempt_trace_sha256": sha256_file(attempt_trace_path),
        "power_sampler_backend": S1_POWER_SIDECAR_BACKEND,
        "power_cadence": attempt["cadence"],
        "allocated_cpu_ids": attempt["allocated_cpu_ids"],
        "detector_cpu_ids": attempt["detector_cpu_ids"],
        "sidecar_cpu_id": attempt["sidecar_cpu_id"],
    }
    evidence["gate_sha256"] = canonical_sha256(evidence)
    return evidence


def validate_sidecar_gate_evidence(
    evidence: Mapping[str, Any],
    *,
    recovery: Mapping[str, Any],
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    checked = json.loads(json.dumps(dict(evidence)))
    gate_hash = checked.pop("gate_sha256", None)
    if not gate_hash or canonical_sha256(checked) != gate_hash:
        raise ValueError("S1 sidecar Gate evidence self-hash mismatch")
    checked["gate_sha256"] = gate_hash
    expected = {
        "schema_version": S1_SIDECAR_GATE_SCHEMA,
        "status": "PASS",
        "paper_claim_allowed": False,
        "gate_only": True,
        "published_formal_profile": False,
        "reused_existing_test_evidence": True,
        "opened_new_test_evidence": False,
        "profile_code_commit": recovery["profile_code_commit"],
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
        "resolution": 256,
        "seed": 3408,
        "power_sampler_backend": S1_POWER_SIDECAR_BACKEND,
        "gpu_scope": "gate_uuid_bound_matrix_hardware_class_matched",
        "loader_exposure_count": int(recovery["expected_loader_exposure_count"]),
        "physical_window_count": int(recovery["expected_physical_window_count"]),
    }
    for key, value in expected.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 sidecar Gate evidence {key} mismatch")
    if (
        not str(checked.get("slurm_job_id", "")).isdigit()
        or checked.get("power_cadence", {}).get("formal_cadence_pass") is not True
        or float(
            checked.get("power_cadence", {}).get("max_gap_ms", float("inf"))
        )
        > float(recovery["power_max_gap_limit_ms"])
        or len(checked.get("allocated_cpu_ids", ()))
        != int(recovery["allocated_cpu_count"])
        or len(checked.get("detector_cpu_ids", ()))
        != int(recovery["detector_cpu_count"])
        or checked.get("sidecar_cpu_id") in checked.get("detector_cpu_ids", ())
        or set(checked.get("detector_cpu_ids", ()))
        | {checked.get("sidecar_cpu_id")}
        != set(checked.get("allocated_cpu_ids", ()))
    ):
        raise ValueError("S1 sidecar Gate resource or cadence evidence is invalid")
    if (
        not str(checked.get("gate_gpu_uuid", "")).startswith("GPU-")
        or canonical_sha256(checked.get("hardware_class", {}))
        != checked.get("hardware_class_sha256")
    ):
        raise ValueError("S1 sidecar Gate GPU or hardware-class evidence is invalid")
    if verify_artifacts:
        artifact_hashes = (
            ("profile_marker_path", "profile_marker_file_sha256"),
            (
                "sidecar_attempt_report_path",
                "sidecar_attempt_report_file_sha256",
            ),
            (
                "sidecar_attempt_trace_path",
                "sidecar_attempt_trace_sha256",
            ),
            ("test_evidence_path", "test_evidence_file_sha256"),
        )
        for path_key, hash_key in artifact_hashes:
            path = Path(checked[path_key]).resolve()
            if not path.is_file() or sha256_file(path) != checked[hash_key]:
                raise ValueError(f"S1 sidecar Gate artifact mismatch: {path}")
        attempt = validate_nvml_sidecar_attempt(
            Path(checked["sidecar_attempt_report_path"]).resolve(),
            Path(checked["sidecar_attempt_trace_path"]).resolve(),
            expected_uuid=checked["gate_gpu_uuid"],
            require_pass=True,
        )
        if (
            attempt["attempt_sha256"] != checked["sidecar_attempt_sha256"]
            or attempt["trace_file_sha256"]
            != checked["sidecar_attempt_trace_sha256"]
        ):
            raise ValueError("S1 sidecar Gate attempt identity mismatch")
    return checked


def load_sidecar_gate_evidence(
    path: str | Path, *, recovery: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(path).resolve()
    if path != sidecar_gate_path(recovery):
        raise ValueError("S1 sidecar Gate evidence is outside its recovery campaign")
    checked = validate_sidecar_gate_evidence(
        json.loads(path.read_text(encoding="utf-8")),
        recovery=recovery,
        verify_artifacts=True,
    )
    canonical_text = json.dumps(
        json.loads(path.read_text(encoding="utf-8")), indent=2, sort_keys=True
    ) + "\n"
    if path.read_text(encoding="utf-8") != canonical_text:
        raise ValueError("S1 sidecar Gate evidence is not canonical JSON")
    return checked


def write_sidecar_gate_evidence(
    evidence: Mapping[str, Any], *, recovery: Mapping[str, Any]
) -> Path:
    path = sidecar_gate_path(recovery)
    checked = validate_sidecar_gate_evidence(
        evidence, recovery=recovery, verify_artifacts=True
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != checked:
            raise FileExistsError("S1 sidecar Gate evidence already differs")
    else:
        atomic_publish_json(path, checked)
    return path


__all__ = [
    "S1_SIDECAR_GATE_SCHEMA",
    "build_sidecar_gate_evidence",
    "load_sidecar_gate_evidence",
    "sidecar_gate_hardware_class",
    "sidecar_gate_path",
    "sidecar_gate_profile_prefix",
    "validate_sidecar_gate_runtime_identity",
    "validate_sidecar_gate_evidence",
    "write_sidecar_gate_evidence",
]
