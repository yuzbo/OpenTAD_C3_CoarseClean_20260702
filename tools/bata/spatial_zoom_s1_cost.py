from __future__ import annotations

import json
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.spatial_zoom_s1_contract import (
    S1_PROFILE_ORDER_SEED,
    S1_TRAINING_SEEDS,
    atomic_publish_json,
    canonical_sha256,
)

S1_PROFILE_PROTOCOL = "spatial_zoom_s1_offline_full_stack_v6"
S1_PROFILE_SCHEMA = "spatial_zoom_s1_profile_v8"
TOP_LEVEL_STAGES = (
    "input_pipeline_serial_ms",
    "h2d_ms",
    "model_forward_ms",
    "postprocess_ms",
)
FORMAL_TOTAL_STAGES = (
    "decode_to_window_output_wall_ms",
    "final_video_nms_ms",
    "end_to_end_serial_ms",
)
MODEL_STAGES = (
    "backbone_wrapper_ms",
    "projection_ms",
    "neck_ms",
    "head_ms",
)
NESTED_STAGES = MODEL_STAGES + ("heavy_backbone_ms",)


def make_profile_exposure_id(physical_window_id: str, ordinal: int) -> str:
    physical_window_id = str(physical_window_id).strip()
    if not physical_window_id or int(ordinal) < 0:
        raise ValueError("S1 profile exposure identity is invalid")
    return f"{physical_window_id}#exposure={int(ordinal):06d}"


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize empty profile values")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    checked = [_finite_nonnegative(value, "profile sample") for value in values]
    return {
        "count": len(checked),
        "mean": sum(checked) / len(checked),
        "p50": _quantile(checked, 0.50),
        "p95": _quantile(checked, 0.95),
        "min": min(checked),
        "max": max(checked),
    }


def _normalize_power_trace(
    samples: Sequence[Mapping[str, Any]],
    *,
    interval_ms: int,
    sampler: str,
    formal: bool,
) -> tuple[list[dict[str, float]], dict[str, Any] | None, str | None]:
    normalized = []
    for index, sample in enumerate(samples):
        timestamp_ms = _finite_nonnegative(
            sample.get("timestamp_ms"), f"power[{index}].timestamp_ms"
        )
        power_w = _finite_nonnegative(sample.get("power_w"), f"power[{index}].power_w")
        normalized.append({"timestamp_ms": timestamp_ms, "power_w": power_w})
    if not normalized:
        if formal:
            raise ValueError("formal S1 profile requires the raw GPU power trace")
        return [], None, None
    timestamps = [row["timestamp_ms"] for row in normalized]
    if abs(timestamps[0]) > 1e-6 or any(
        right <= left for left, right in zip(timestamps[:-1], timestamps[1:])
    ):
        raise ValueError(
            "S1 power trace timestamps must start at zero and increase strictly"
        )
    gaps = [right - left for left, right in zip(timestamps[:-1], timestamps[1:])]
    max_gap_ms = max(gaps, default=0.0)
    max_gap_limit_ms = max(100.0, float(interval_ms) * 5.0)
    if formal and (len(normalized) < 2 or max_gap_ms > max_gap_limit_ms):
        raise ValueError(
            "formal S1 power trace is too sparse for auditable energy integration: "
            f"sampler={sampler}, samples={len(normalized)}, "
            f"max_gap_ms={max_gap_ms:.6f}, limit_ms={max_gap_limit_ms:.6f}"
        )
    jsonl = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in normalized
    ).encode("utf-8")
    return (
        normalized,
        {
            "sampler": str(sampler),
            "target_interval_ms": int(interval_ms),
            "sample_count": len(normalized),
            "duration_ms": timestamps[-1] - timestamps[0],
            "max_gap_ms": max_gap_ms,
            "max_gap_limit_ms": max_gap_limit_ms,
        },
        hashlib.sha256(jsonl).hexdigest(),
    )


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    checked = dict(metadata)
    required = (
        "method",
        "resolution",
        "protocol",
        "protocol_fingerprint",
        "manifest_sha256",
        "hardware_identity",
        "hardware_fingerprint",
        "software_identity",
        "software_fingerprint",
        "config_commit",
        "experiment_namespace",
        "canonical_experiment_root",
        "checkpoint_sha256",
        "pretrained_checkpoint_sha256",
        "checkpoint_epoch",
        "trained_checkpoint",
        "batch_size",
        "loader_workers",
        "warmup_samples",
        "amp",
        "power_sampling_enabled",
        "power_sampler_backend",
        "trace_publication_mode",
        "trace_io_inside_sampling_loop",
        "formal_profile",
        "split",
        "seed",
        "sample_manifest_sha256",
        "physical_window_manifest_sha256",
        "loader_exposure_count",
        "physical_window_count",
        "duplicate_physical_window_exposure_count",
        "max_physical_window_multiplicity",
        "test_open_certificate_sha256",
        "test_evidence_sha256",
        "test_open_marker_sha256",
        "precheck_file_sha256",
        "precheck_sha256",
        "power_gpu_id",
        "power_interval_ms",
        "video_count",
        "world_size",
        "execution_wrapper",
        "result_finalizer",
        "profile_attempt_marker_path",
        "profile_attempt_marker_file_sha256",
        "profile_attempt_marker_sha256",
        "profile_order_seed",
        "profile_order_sha256",
        "profile_order_ordinal",
        "profile_code_commit",
        "profile_recovery_certificate_path",
        "profile_recovery_certificate_file_sha256",
        "profile_recovery_certificate_sha256",
        "profile_recovery_campaign_id",
        "power_attempt_report_path",
        "power_attempt_report_file_sha256",
        "power_attempt_sha256",
        "power_attempt_trace_path",
        "power_attempt_trace_file_sha256",
        "power_attempt_cadence",
        "allocated_cpu_ids",
        "detector_cpu_ids",
        "sidecar_cpu_id",
        "sidecar_gate_evidence_path",
        "sidecar_gate_evidence_file_sha256",
        "sidecar_gate_sha256",
        "matrix_start_receipt_path",
        "matrix_start_receipt_file_sha256",
        "matrix_sha256",
        "slurm_job_id",
        "slurm_step_id",
        "step_gpu_uuid",
    )
    missing = [key for key in required if key not in checked]
    if missing:
        raise ValueError(f"S1 profile metadata missing {missing}")
    if checked["protocol"] != S1_PROFILE_PROTOCOL:
        raise ValueError("S1 profile uses the wrong protocol")
    if not bool(checked["trained_checkpoint"]):
        raise ValueError("S1 paper cost requires a trained checkpoint")
    if int(checked["resolution"]) not in (160, 224, 256):
        raise ValueError("unexpected S1 resolution")
    if int(checked["batch_size"]) != 1 or int(checked["loader_workers"]) != 0:
        raise ValueError(
            "S1 serial full-stack profile requires batch_size=1 and loader_workers=0"
        )
    if int(checked["profile_order_seed"]) != S1_PROFILE_ORDER_SEED or int(
        checked["profile_order_ordinal"]
    ) not in range(9):
        raise ValueError("S1 profile uses an invalid frozen matrix order")
    for key in ("hardware_identity", "software_identity"):
        if not isinstance(checked[key], Mapping) or not checked[key]:
            raise ValueError(f"S1 profile metadata requires a non-empty {key}")
        fingerprint_key = key.replace("_identity", "_fingerprint")
        if canonical_sha256(checked[key]) != checked[fingerprint_key]:
            raise ValueError(f"S1 profile {key} does not match its fingerprint")
    for key in (
        "method",
        "protocol_fingerprint",
        "manifest_sha256",
        "hardware_fingerprint",
        "software_fingerprint",
        "config_commit",
        "experiment_namespace",
        "canonical_experiment_root",
        "checkpoint_sha256",
        "pretrained_checkpoint_sha256",
        "sample_manifest_sha256",
        "physical_window_manifest_sha256",
        "test_open_certificate_sha256",
        "power_gpu_id",
        "profile_attempt_marker_path",
        "profile_order_sha256",
        "profile_code_commit",
        "profile_recovery_certificate_path",
        "profile_recovery_campaign_id",
        "power_attempt_report_path",
        "power_attempt_trace_path",
        "sidecar_gate_evidence_path",
    ):
        if not str(checked[key]).strip():
            raise ValueError(f"S1 profile metadata requires {key}")
    if bool(checked["formal_profile"]):
        for key in (
            "matrix_start_receipt_path",
            "slurm_job_id",
            "slurm_step_id",
            "step_gpu_uuid",
        ):
            if not str(checked[key]).strip():
                raise ValueError(f"formal S1 profile requires {key}")
        hardware = checked["hardware_identity"]
        gpu_scope = dict(hardware.get("slurm_gpu_scope", {}))
        nvidia_smi = dict(hardware.get("nvidia_smi", {}))
        if (
            checked["step_gpu_uuid"]
            != hardware.get("cuda_visible_device_uuid")
            or checked["step_gpu_uuid"] != nvidia_smi.get("uuid")
            or checked["slurm_job_id"] != gpu_scope.get("job_id")
            or checked["slurm_step_id"] != gpu_scope.get("step_id")
        ):
            raise ValueError(
                "formal S1 profile matrix receipt differs from its CUDA/Slurm identity"
            )
        if (
            int(checked["warmup_samples"]) != 50
            or int(checked["power_interval_ms"]) != 20
        ):
            raise ValueError(
                "formal S1 profile freezes 50 warmup windows and a 20 ms power interval"
            )
        if not bool(checked["amp"]) or not bool(checked["power_sampling_enabled"]):
            raise ValueError("formal S1 profile requires AMP and power sampling")
        if checked["power_sampler_backend"] != "nvml-sidecar-process-v1":
            raise ValueError("formal S1 profile requires the isolated NVML sidecar")
        if (
            checked["trace_publication_mode"]
            != "post_sampling_atomic_jsonl_v1"
            or checked["trace_io_inside_sampling_loop"] is not False
        ):
            raise ValueError(
                "formal S1 profile requires post-sampling atomic trace publication"
            )
        cadence = checked["power_attempt_cadence"]
        if (
            not isinstance(cadence, Mapping)
            or cadence.get("formal_cadence_pass") is not True
            or float(cadence.get("max_gap_ms", math.inf)) > 100.0
            or float(cadence.get("max_gap_limit_ms", -1.0)) != 100.0
            or len(checked["allocated_cpu_ids"]) != 5
            or len(checked["detector_cpu_ids"]) != 4
            or checked["sidecar_cpu_id"] in checked["detector_cpu_ids"]
            or set(checked["detector_cpu_ids"]) | {checked["sidecar_cpu_id"]}
            != set(checked["allocated_cpu_ids"])
        ):
            raise ValueError("formal S1 profile sidecar cadence/CPU contract failed")
        if checked["split"] != "test" or not checked["test_open_certificate_sha256"]:
            raise ValueError(
                "formal S1 profile requires a frozen test-open certificate"
            )
        if int(checked["video_count"]) <= 0:
            raise ValueError(
                "formal S1 profile requires a positive sealed-test video count"
            )
        if (
            int(checked["world_size"]) != 1
            or checked["execution_wrapper"] != "torchrun_ddp_world1"
            or checked["result_finalizer"]
            != "opentad.cores.test_engine.gather_ddp_results"
        ):
            raise ValueError(
                "formal S1 profile must match the official single-process DDP test path"
            )
        if int(checked["seed"]) not in S1_TRAINING_SEEDS:
            raise ValueError("formal S1 profile seed is outside the frozen schema")
        if (
            int(checked["checkpoint_epoch"]) < 0
            or int(checked["power_interval_ms"]) <= 0
        ):
            raise ValueError("formal S1 profile epoch and power interval must be valid")
        hash_fields = (
            "protocol_fingerprint",
            "manifest_sha256",
            "hardware_fingerprint",
            "software_fingerprint",
            "checkpoint_sha256",
            "sample_manifest_sha256",
            "physical_window_manifest_sha256",
            "test_open_certificate_sha256",
            "test_evidence_sha256",
            "test_open_marker_sha256",
            "precheck_file_sha256",
            "precheck_sha256",
            "pretrained_checkpoint_sha256",
            "profile_attempt_marker_file_sha256",
            "profile_attempt_marker_sha256",
            "profile_order_sha256",
            "profile_recovery_certificate_file_sha256",
            "profile_recovery_certificate_sha256",
            "power_attempt_report_file_sha256",
            "power_attempt_sha256",
            "power_attempt_trace_file_sha256",
            "sidecar_gate_evidence_file_sha256",
            "sidecar_gate_sha256",
            "matrix_start_receipt_file_sha256",
            "matrix_sha256",
        )
        if any(
            len(str(checked[key])) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(checked[key]).lower()
            )
            for key in hash_fields
        ):
            raise ValueError("formal S1 profile provenance hashes must be SHA-256")
        commit = str(checked["config_commit"]).lower()
        if len(commit) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError("formal S1 profile requires a concrete Git commit")
        profile_commit = str(checked["profile_code_commit"]).lower()
        if len(profile_commit) != 40 or any(
            character not in "0123456789abcdef" for character in profile_commit
        ):
            raise ValueError("formal S1 profile requires a recovery Git commit")
        exposure_count = int(checked["loader_exposure_count"])
        physical_count = int(checked["physical_window_count"])
        duplicate_count = int(checked["duplicate_physical_window_exposure_count"])
        max_multiplicity = int(checked["max_physical_window_multiplicity"])
        if (
            exposure_count < 200
            or physical_count <= 0
            or duplicate_count != exposure_count - physical_count
            or max_multiplicity < 1
        ):
            raise ValueError("formal S1 profile exposure topology is invalid")
    return checked


def build_profile_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
    power_trace: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if not samples:
        raise ValueError("S1 profile requires measured samples")
    checked_metadata = _validate_metadata(metadata)
    normalized = []
    for index, sample in enumerate(samples):
        row = {
            key: _finite_nonnegative(sample.get(key), f"sample[{index}].{key}")
            for key in TOP_LEVEL_STAGES + NESTED_STAGES + FORMAL_TOTAL_STAGES
        }
        model_children = sum(row[key] for key in MODEL_STAGES)
        tolerance = max(1e-6, row["model_forward_ms"] * 1e-4)
        if model_children > row["model_forward_ms"] + tolerance:
            raise ValueError("S1 model component stages exceed model_forward_ms")
        if row["heavy_backbone_ms"] > row["backbone_wrapper_ms"] + tolerance:
            raise ValueError("heavy_backbone_ms exceeds backbone_wrapper_ms")
        component_total = sum(row[key] for key in TOP_LEVEL_STAGES)
        continuous = row["decode_to_window_output_wall_ms"]
        if component_total > continuous + max(1.0, continuous * 0.05):
            raise ValueError(
                "S1 component timings exceed independent continuous wall time"
            )
        expected_total = continuous + row["final_video_nms_ms"]
        if abs(row["end_to_end_serial_ms"] - expected_total) > max(
            1e-6, expected_total * 1e-6
        ):
            raise ValueError("S1 end-to-end total does not conserve wall time plus NMS")
        row["component_sum_ms"] = component_total
        row["continuous_unattributed_ms"] = max(0.0, continuous - component_total)
        row["model_unattributed_ms"] = max(
            0.0, row["model_forward_ms"] - model_children
        )
        row["backbone_wrapper_overhead_ms"] = max(
            0.0, row["backbone_wrapper_ms"] - row["heavy_backbone_ms"]
        )
        for resource in (
            "peak_gpu_allocated_mb",
            "peak_gpu_reserved_mb",
            "gpu_energy_j",
        ):
            value = sample.get(resource)
            row[resource] = (
                None if value is None else _finite_nonnegative(value, resource)
            )
        row["video_id"] = str(sample.get("video_id", ""))
        row["window_id"] = str(sample.get("window_id", ""))
        row["physical_window_id"] = str(sample.get("physical_window_id", ""))
        try:
            row["loader_ordinal"] = int(sample.get("loader_ordinal"))
        except (TypeError, ValueError) as exc:
            raise ValueError("S1 profile samples require loader ordinals") from exc
        if not row["video_id"] or not row["window_id"] or not row["physical_window_id"]:
            raise ValueError(
                "S1 profile samples require video/exposure/physical identities"
            )
        if row["loader_ordinal"] != index or row[
            "window_id"
        ] != make_profile_exposure_id(row["physical_window_id"], index):
            raise ValueError("S1 profile exposure identity is not loader-order exact")
        normalized.append(row)
    if bool(checked_metadata["formal_profile"]):
        if len(normalized) < 200:
            raise ValueError("formal S1 profile requires at least 200 measured windows")
        if len({row["window_id"] for row in normalized}) != len(normalized):
            raise ValueError("formal S1 profile exposure identities must be unique")
        if any(row["gpu_energy_j"] is None for row in normalized):
            raise ValueError(
                "formal S1 profile requires energy for every measured window"
            )
        if len({row["video_id"] for row in normalized}) != int(
            checked_metadata["video_count"]
        ):
            raise ValueError(
                "formal S1 profile video identities do not match video_count"
            )
    sample_manifest = [row["window_id"] for row in normalized]
    physical_manifest = [row["physical_window_id"] for row in normalized]
    physical_counts = Counter(physical_manifest)
    if canonical_sha256(sample_manifest) != checked_metadata["sample_manifest_sha256"]:
        raise ValueError(
            "S1 profile sample manifest hash does not match measured windows"
        )
    if (
        canonical_sha256(physical_manifest)
        != checked_metadata["physical_window_manifest_sha256"]
    ):
        raise ValueError("S1 profile physical-window manifest hash mismatch")
    expected_topology = {
        "loader_exposure_count": len(normalized),
        "physical_window_count": len(physical_counts),
        "duplicate_physical_window_exposure_count": len(normalized)
        - len(physical_counts),
        "max_physical_window_multiplicity": max(physical_counts.values()),
    }
    for key, value in expected_topology.items():
        if int(checked_metadata[key]) != int(value):
            raise ValueError(f"S1 profile measured {key} differs from metadata")
    stage_names = sorted(
        key
        for key, value in normalized[0].items()
        if key.endswith("_ms") and value is not None
    )
    normalized_power, power_summary, power_trace_file_sha256 = _normalize_power_trace(
        power_trace,
        interval_ms=int(checked_metadata["power_interval_ms"] or 0),
        sampler=str(checked_metadata["power_sampler_backend"]),
        formal=bool(checked_metadata["formal_profile"]),
    )
    raw_samples = [dict(sample) for sample in samples]
    sample_jsonl = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in raw_samples
    ).encode("utf-8")
    sample_trace_file_sha256 = hashlib.sha256(sample_jsonl).hexdigest()
    report = {
        "schema_version": S1_PROFILE_SCHEMA,
        **checked_metadata,
        "sample_count": len(normalized),
        "stages": {
            name: _summary([row[name] for row in normalized]) for name in stage_names
        },
        "resources": {
            name: _summary([row[name] for row in normalized if row[name] is not None])
            if any(row[name] is not None for row in normalized)
            else None
            for name in (
                "peak_gpu_allocated_mb",
                "peak_gpu_reserved_mb",
                "gpu_energy_j",
            )
        },
        "stage_semantics": {
            "input_pipeline_serial_ms": "dataset read, decode, spatial transforms, and collate with zero workers",
            "h2d_ms": "recursive host-to-device transfer",
            "model_forward_ms": "complete detector forward without postprocess",
            "postprocess_ms": "detector conversion and per-window processing",
            "final_video_nms_ms": "official world-size-one result gather, cross-window NMS, and output reconstruction amortized over windows",
            "decode_to_window_output_wall_ms": "independent continuous wall timer from decode start through result-dict accumulation",
            "end_to_end_serial_ms": "continuous window wall plus amortized official final result aggregation",
            "heavy_backbone_ms": "nested inside backbone_wrapper_ms",
        },
        "measurement_scope": {
            "latency": "warm serial per-window gross wall time after 50 warmup windows",
            "energy": "warm serial gross GPU energy per measured window including amortized final aggregation",
            "not_measured": [
                "cold-start latency",
                "whole-video latency distribution",
                "incremental deployment energy",
                "CPU or storage energy",
            ],
        },
        "claims": {
            "trained_checkpoint_only": True,
            "offline_tad": True,
            "decode_preprocess_h2d_included": True,
            "flops_not_substituted_for_latency": True,
            "official_test_result_path_reused": True,
        },
        "raw_samples": raw_samples,
        "sample_trace_file_sha256": sample_trace_file_sha256,
        "power_sampling": power_summary,
        "raw_power_samples": normalized_power,
        "power_trace_file_sha256": power_trace_file_sha256,
    }
    report["profile_sha256"] = canonical_sha256(report)
    return report


def validate_profile_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    checked = json.loads(json.dumps(dict(profile)))
    profile_hash = checked.pop("profile_sha256", None)
    if not profile_hash or canonical_sha256(checked) != profile_hash:
        raise ValueError("S1 profile self-hash mismatch")
    checked["profile_sha256"] = profile_hash
    if checked.get("schema_version") != S1_PROFILE_SCHEMA:
        raise ValueError("unsupported S1 profile schema")
    generated_keys = {
        "schema_version",
        "sample_count",
        "stages",
        "resources",
        "stage_semantics",
        "measurement_scope",
        "claims",
        "raw_samples",
        "sample_trace_file_sha256",
        "power_sampling",
        "raw_power_samples",
        "power_trace_file_sha256",
        "profile_sha256",
    }
    metadata = {
        key: value for key, value in checked.items() if key not in generated_keys
    }
    rebuilt = build_profile_summary(
        checked.get("raw_samples", ()),
        metadata=metadata,
        power_trace=checked.get("raw_power_samples", ()),
    )
    if rebuilt != checked:
        raise ValueError("S1 profile summary does not match its raw samples")
    return checked


def compare_resolution_profiles(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    baseline = validate_profile_summary(baseline)
    candidate = validate_profile_summary(candidate)
    if int(baseline.get("resolution", -1)) != 160:
        raise ValueError("S1 profile baseline must be dense160")
    if int(candidate.get("resolution", -1)) not in (224, 256):
        raise ValueError("S1 candidate profile must be dense224 or dense256")
    for key in (
        "schema_version",
        "protocol",
        "protocol_fingerprint",
        "manifest_sha256",
        "hardware_identity",
        "hardware_fingerprint",
        "software_identity",
        "software_fingerprint",
        "config_commit",
        "experiment_namespace",
        "canonical_experiment_root",
        "pretrained_checkpoint_sha256",
        "batch_size",
        "loader_workers",
        "warmup_samples",
        "amp",
        "power_sampling_enabled",
        "power_sampler_backend",
        "trace_publication_mode",
        "trace_io_inside_sampling_loop",
        "sample_count",
        "formal_profile",
        "split",
        "seed",
        "sample_manifest_sha256",
        "test_open_certificate_sha256",
        "precheck_file_sha256",
        "precheck_sha256",
        "power_gpu_id",
        "power_interval_ms",
        "world_size",
        "execution_wrapper",
        "result_finalizer",
        "profile_order_seed",
        "profile_order_sha256",
        "profile_code_commit",
        "profile_recovery_certificate_path",
        "profile_recovery_certificate_file_sha256",
        "profile_recovery_certificate_sha256",
        "profile_recovery_campaign_id",
        "sidecar_gate_evidence_path",
        "sidecar_gate_evidence_file_sha256",
        "sidecar_gate_sha256",
        "allocated_cpu_ids",
        "detector_cpu_ids",
        "sidecar_cpu_id",
        "physical_window_manifest_sha256",
        "loader_exposure_count",
        "physical_window_count",
        "duplicate_physical_window_exposure_count",
        "max_physical_window_multiplicity",
    ):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"S1 profiles have incompatible {key}")
    baseline_p50 = float(baseline["stages"]["end_to_end_serial_ms"]["p50"])
    candidate_p50 = float(candidate["stages"]["end_to_end_serial_ms"]["p50"])
    baseline_p95 = float(baseline["stages"]["end_to_end_serial_ms"]["p95"])
    candidate_p95 = float(candidate["stages"]["end_to_end_serial_ms"]["p95"])
    return {
        "schema_version": "spatial_zoom_s1_profile_comparison_v1",
        "comparable": True,
        "baseline_resolution": 160,
        "candidate_resolution": int(candidate["resolution"]),
        "end_to_end_p50_ms": {
            "baseline": baseline_p50,
            "candidate": candidate_p50,
            "ratio": candidate_p50 / baseline_p50,
        },
        "end_to_end_p95_ms": {
            "baseline": baseline_p95,
            "candidate": candidate_p95,
            "ratio": candidate_p95 / baseline_p95,
        },
        "peak_gpu_allocated_mb": {
            "baseline": baseline["resources"]["peak_gpu_allocated_mb"],
            "candidate": candidate["resources"]["peak_gpu_allocated_mb"],
        },
        "peak_gpu_reserved_mb": {
            "baseline": baseline["resources"]["peak_gpu_reserved_mb"],
            "candidate": candidate["resources"]["peak_gpu_reserved_mb"],
        },
        "gpu_energy_j": {
            "baseline": baseline["resources"]["gpu_energy_j"],
            "candidate": candidate["resources"]["gpu_energy_j"],
        },
    }


def write_profile_summary(report: Mapping[str, Any], output_path: str | Path) -> Path:
    checked = validate_profile_summary(report)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite a formal S1 profile summary")
    atomic_publish_json(output_path, checked)
    return output_path


__all__ = [
    "S1_PROFILE_PROTOCOL",
    "S1_PROFILE_SCHEMA",
    "build_profile_summary",
    "compare_resolution_profiles",
    "make_profile_exposure_id",
    "validate_profile_summary",
    "write_profile_summary",
]
