#!/usr/bin/env python3
"""Static, model-free runtime attestation for the ZoomToken P1 screen.

The probe runs after allocation/container entry and before any model, CUDA,
checkpoint, dataset, warmup, metric, or cost code.  It imports no framework and
does not initialize CUDA.  A leaf is admitted only when its normalized runtime
class is exactly equal to a separately captured preflight class.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


OBSERVATION_SCHEMA = "georoute_p1_runtime_observations_v001"
ATTESTATION_SCHEMA = "georoute_p1_runtime_attestation_v001"
RUNTIME_CLASS_SCHEMA = "georoute_p1_runtime_class_v001"
STUDY_ID = "ZOOMTOKEN_P1_DNURQ_V001"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CUDA_VERSION_RE = re.compile(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)")
_NVML_VERSION_RE = re.compile(
    r"(?:NVML Version|NVML Library Version)\s*:\s*([^\r\n]+)",
    re.IGNORECASE,
)
_NVIDIA_QUERY_FIELDS = (
    "uuid",
    "name",
    "pci.bus_id",
    "pci.device_id",
    "pci.sub_device_id",
    "memory.total",
    "compute_cap",
    "mig.mode.current",
    "driver_version",
    "persistence_mode",
    "clocks.applications.graphics",
    "clocks.applications.memory",
    "power.limit",
)
_CUDA_LIBRARY_DISTRIBUTIONS = (
    "nvidia-cublas-cu11",
    "nvidia-cuda-runtime-cu11",
    "nvidia-cudnn-cu11",
    "nvidia-nccl-cu11",
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_once(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"runtime probe failed: {' '.join(command)}"
        )
    return completed.stdout.strip()


def _parse_csv_rows(text: str) -> list[dict[str, str]]:
    rows = []
    for raw in csv.reader(text.splitlines()):
        values = [value.strip() for value in raw]
        if len(values) != len(_NVIDIA_QUERY_FIELDS):
            raise ValueError("nvidia-smi runtime row has an unexpected field count")
        rows.append(dict(zip(_NVIDIA_QUERY_FIELDS, values)))
    if not rows:
        raise ValueError("nvidia-smi returned no visible GPU")
    return rows


def _float(value: Any, *, name: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not result > 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _int(value: Any, *, name: str) -> int:
    result = _float(value, name=name)
    if not result.is_integer():
        raise ValueError(f"{name} must be integral")
    return int(result)


def _optional_int(value: Any, *, name: str) -> int | None:
    normalized = str(value).strip()
    if normalized in {"N/A", "[N/A]", "Not Supported"}:
        return None
    return _int(normalized, name=name)


def _literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"torch version metadata lacks {name!r}")


def _framework_metadata() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("torch")
    version_path = Path(distribution.locate_file("torch/version.py")).resolve()
    if not version_path.is_file():
        raise FileNotFoundError("installed torch distribution lacks torch/version.py")
    return {
        "name": "torch",
        "version": distribution.version,
        "build_cuda": str(_literal_assignment(version_path, "cuda")),
        "git_version": str(_literal_assignment(version_path, "git_version")),
        "version_metadata_path": str(version_path),
        "version_metadata_sha256": _sha256_file(version_path),
    }


def _cuda_library_versions() -> dict[str, str]:
    versions = {}
    for name in _CUDA_LIBRARY_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    conda_metadata = Path(sys.prefix) / "conda-meta"
    if conda_metadata.is_dir():
        for path in sorted(conda_metadata.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            name = str(payload.get("name", "")).lower()
            if (
                name in {"cudatoolkit", "pytorch-cuda", "cudnn", "nccl"}
                or name.startswith("cuda-")
            ):
                version = str(payload.get("version", "")).strip()
                if not version:
                    raise ValueError("Conda CUDA-library metadata lacks a version")
                versions[f"conda:{name}"] = version
    if not versions:
        raise RuntimeError(
            "runtime probe found no installed CUDA-library distribution metadata"
        )
    return versions


def collect_runtime_observations(
    *,
    container_image: str | Path,
    dependency_lock: str | Path,
    expected_visible_gpu_count: int,
    nvidia_smi: str = "nvidia-smi",
) -> dict[str, Any]:
    """Collect raw runtime facts once, without importing torch or touching CUDA."""

    container_path = Path(container_image).resolve()
    lock_path = Path(dependency_lock).resolve()
    if not container_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("container image and dependency lock must exist")
    if int(expected_visible_gpu_count) <= 0:
        raise ValueError("expected visible GPU count must be positive")
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible_devices = tuple(
        value.strip()
        for value in str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",")
        if value.strip()
    )
    if (
        not slurm_job_id.isdigit()
        or len(visible_devices) != int(expected_visible_gpu_count)
    ):
        raise RuntimeError("runtime probe requires the frozen Slurm-visible GPU count")
    active_container = str(
        os.environ.get("APPTAINER_CONTAINER")
        or os.environ.get("SINGULARITY_CONTAINER")
        or ""
    ).strip()
    if not active_container or Path(active_container).resolve() != container_path:
        raise RuntimeError("runtime probe is not inside the named immutable container")

    query = _run_once(
        (
            nvidia_smi,
            "--query-gpu=" + ",".join(_NVIDIA_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        )
    )
    gpu_rows = _parse_csv_rows(query)
    if len(gpu_rows) != int(expected_visible_gpu_count):
        raise RuntimeError("visible GPU count differs from the frozen allocation")
    version_output = _run_once((nvidia_smi, "--version"))
    banner_output = _run_once((nvidia_smi,))
    cuda_match = _CUDA_VERSION_RE.search(banner_output)
    nvml_match = _NVML_VERSION_RE.search(version_output)
    if cuda_match is None or nvml_match is None:
        raise RuntimeError("nvidia-smi omitted CUDA or NVML version identity")

    return {
        "schema_version": OBSERVATION_SCHEMA,
        "collector": "georoute_p1_runtime_attestor.collect_runtime_observations",
        "collector_uses_framework_import": False,
        "collector_initializes_cuda": False,
        "gpu_query_fields": list(_NVIDIA_QUERY_FIELDS),
        "gpu_rows": gpu_rows,
        "allocation": {
            "slurm_job_present": True,
            "cuda_visible_device_count": len(visible_devices),
            "cuda_visible_devices_overridden_by_attestor": False,
        },
        "nvidia_smi_cuda_version": cuda_match.group(1),
        "nvml_version": nvml_match.group(1).strip(),
        "container": {
            "path": str(container_path),
            "digest": f"sha256:{_sha256_file(container_path)}",
            "active_runtime_path_verified": True,
        },
        "dependency_lock": {
            "path": str(lock_path),
            "sha256": _sha256_file(lock_path),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "framework": _framework_metadata(),
        "cuda_library_versions": _cuda_library_versions(),
        "kernel_release": platform.release(),
    }


def _gpu_runtime_class(row: Mapping[str, Any]) -> dict[str, Any]:
    device_id = str(row.get("pci.device_id", "")).strip().lower()
    subsystem_id = str(row.get("pci.sub_device_id", "")).strip().lower()
    if not re.fullmatch(r"0x[0-9a-f]{8}", device_id):
        raise ValueError("GPU PCI device/vendor ID is malformed")
    if not re.fullmatch(r"0x[0-9a-f]{8}", subsystem_id):
        raise ValueError("GPU PCI subsystem ID is malformed")
    if device_id[-4:] != "10de":
        raise ValueError("P1 runtime requires an NVIDIA PCI vendor ID")
    compute_capability = str(row.get("compute_cap", "")).strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]+", compute_capability):
        raise ValueError("GPU compute capability is malformed")
    mig_mode = str(row.get("mig.mode.current", "")).strip()
    if mig_mode not in {"Enabled", "Disabled", "N/A"}:
        raise ValueError("GPU MIG mode is not explicit")
    return {
        "vendor": "NVIDIA",
        "sku": str(row.get("name", "")).strip(),
        "pci_device_vendor_id": device_id,
        "pci_subsystem_id": subsystem_id,
        "total_memory_mib": _int(row.get("memory.total"), name="GPU total memory"),
        "compute_capability": compute_capability,
        "mig_mode": mig_mode,
        "driver_version": str(row.get("driver_version", "")).strip(),
        "clock_policy": {
            "persistence_mode": str(row.get("persistence_mode", "")).strip(),
            "application_graphics_clock_mhz": _optional_int(
                row.get("clocks.applications.graphics"),
                name="application graphics clock",
            ),
            "application_memory_clock_mhz": _optional_int(
                row.get("clocks.applications.memory"),
                name="application memory clock",
            ),
            "power_limit_w": _float(row.get("power.limit"), name="power limit"),
        },
    }


def normalize_runtime_class(observations: Mapping[str, Any]) -> dict[str, Any]:
    observations = dict(observations)
    rows = observations.get("gpu_rows")
    framework = observations.get("framework")
    container = observations.get("container")
    dependency_lock = observations.get("dependency_lock")
    python = observations.get("python")
    libraries = observations.get("cuda_library_versions")
    allocation = observations.get("allocation")
    if (
        observations.get("schema_version") != OBSERVATION_SCHEMA
        or observations.get("collector")
        != "georoute_p1_runtime_attestor.collect_runtime_observations"
        or observations.get("collector_uses_framework_import") is not False
        or observations.get("collector_initializes_cuda") is not False
        or tuple(observations.get("gpu_query_fields", ()))
        != _NVIDIA_QUERY_FIELDS
        or not isinstance(rows, list)
        or not rows
        or not isinstance(framework, Mapping)
        or not isinstance(container, Mapping)
        or not isinstance(dependency_lock, Mapping)
        or not isinstance(python, Mapping)
        or not isinstance(libraries, Mapping)
        or not libraries
        or not isinstance(allocation, Mapping)
        or allocation.get("slurm_job_present") is not True
        or int(allocation.get("cuda_visible_device_count", -1)) != len(rows)
        or allocation.get("cuda_visible_devices_overridden_by_attestor")
        is not False
        or container.get("active_runtime_path_verified") is not True
    ):
        raise ValueError("runtime observations are incomplete")
    gpu_uuids = [str(row.get("uuid", "")).strip() for row in rows]
    pci_bus_ids = [str(row.get("pci.bus_id", "")).strip().lower() for row in rows]
    if (
        any(not value for value in gpu_uuids)
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(
            re.fullmatch(r"[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value)
            is None
            for value in pci_bus_ids
        )
        or len(set(pci_bus_ids)) != len(pci_bus_ids)
    ):
        raise ValueError("runtime observations lack unique visible GPU identities")
    gpu_classes = [_gpu_runtime_class(row) for row in rows]
    first_gpu = gpu_classes[0]
    if any(gpu != first_gpu for gpu in gpu_classes[1:]):
        raise ValueError("visible GPUs do not share one runtime class")
    container_digest = str(container.get("digest", ""))
    lock_sha256 = str(dependency_lock.get("sha256", ""))
    metadata_sha256 = str(framework.get("version_metadata_sha256", ""))
    if (
        not _CONTAINER_DIGEST_RE.fullmatch(container_digest)
        or not _SHA256_RE.fullmatch(lock_sha256)
        or not _SHA256_RE.fullmatch(metadata_sha256)
    ):
        raise ValueError("runtime content identity is malformed")
    runtime_class = {
        "schema_version": RUNTIME_CLASS_SCHEMA,
        "visible_gpu_count": len(gpu_classes),
        "slurm_visible_gpu_count": int(
            allocation["cuda_visible_device_count"]
        ),
        "gpu": first_gpu,
        "nvidia_smi_cuda_version": str(
            observations.get("nvidia_smi_cuda_version", "")
        ).strip(),
        "nvml_version": str(observations.get("nvml_version", "")).strip(),
        "container_digest": container_digest,
        "dependency_lock_sha256": lock_sha256,
        "python": {
            "implementation": str(python.get("implementation", "")).strip(),
            "version": str(python.get("version", "")).strip(),
        },
        "framework": {
            "name": str(framework.get("name", "")).strip(),
            "version": str(framework.get("version", "")).strip(),
            "build_cuda": str(framework.get("build_cuda", "")).strip(),
            "git_version": str(framework.get("git_version", "")).strip(),
            "version_metadata_sha256": metadata_sha256,
        },
        "cuda_library_versions": {
            str(key): str(value) for key, value in sorted(libraries.items())
        },
        "kernel_release": str(observations.get("kernel_release", "")).strip(),
    }
    if any(
        not value
        for value in (
            runtime_class["gpu"]["sku"],
            runtime_class["gpu"]["driver_version"],
            runtime_class["nvidia_smi_cuda_version"],
            runtime_class["nvml_version"],
            runtime_class["python"]["implementation"],
            runtime_class["python"]["version"],
            runtime_class["framework"]["name"],
            runtime_class["framework"]["version"],
            runtime_class["framework"]["build_cuda"],
            runtime_class["framework"]["git_version"],
            runtime_class["kernel_release"],
        )
    ):
        raise ValueError("normalized runtime class contains an empty identity")
    return runtime_class


def build_runtime_attestation(
    observations: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    if phase not in {"preflight", "leaf"}:
        raise ValueError("runtime attestation phase must be preflight or leaf")
    runtime_class = normalize_runtime_class(observations)
    return {
        "schema_version": ATTESTATION_SCHEMA,
        "study_id": STUDY_ID,
        "phase": phase,
        "runtime_class": runtime_class,
        "runtime_class_fingerprint": _canonical_sha256(runtime_class),
        "raw_observations": dict(observations),
        "node_name_recorded": False,
        "retry_allowed": False,
        "requeue_allowed": False,
        "fallback_allowed": False,
        "model_imported": False,
        "cuda_initialized": False,
        "checkpoint_accessed": False,
        "dataset_accessed": False,
        "metric_or_cost_work_started": False,
    }


def validate_runtime_attestation(
    attestation: Mapping[str, Any],
    *,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attestation = dict(attestation)
    runtime_class = attestation.get("runtime_class")
    observations = attestation.get("raw_observations")
    if (
        attestation.get("schema_version") != ATTESTATION_SCHEMA
        or attestation.get("study_id") != STUDY_ID
        or attestation.get("phase") not in {"preflight", "leaf"}
        or not isinstance(runtime_class, Mapping)
        or not isinstance(observations, Mapping)
        or dict(runtime_class) != normalize_runtime_class(observations)
        or attestation.get("runtime_class_fingerprint")
        != _canonical_sha256(runtime_class)
        or attestation.get("node_name_recorded") is not False
        or attestation.get("retry_allowed") is not False
        or attestation.get("requeue_allowed") is not False
        or attestation.get("fallback_allowed") is not False
        or attestation.get("model_imported") is not False
        or attestation.get("cuda_initialized") is not False
        or attestation.get("checkpoint_accessed") is not False
        or attestation.get("dataset_accessed") is not False
        or attestation.get("metric_or_cost_work_started") is not False
    ):
        raise ValueError("P1 runtime attestation is invalid")
    if reference is not None:
        reference = validate_runtime_attestation(reference)
        if (
            reference.get("phase") != "preflight"
            or attestation.get("phase") != "leaf"
            or attestation["runtime_class"] != reference["runtime_class"]
            or attestation["runtime_class_fingerprint"]
            != reference["runtime_class_fingerprint"]
        ):
            raise ValueError("P1 leaf runtime class differs from preflight")
    return attestation


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime attestation JSON must be an object")
    return payload


def _exclusive_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "leaf"), required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--expected-visible-gpu-count", type=int, default=2)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (args.phase == "leaf") != (args.reference is not None):
        raise ValueError("leaf requires one preflight reference; preflight forbids it")
    observations = collect_runtime_observations(
        container_image=args.container_image,
        dependency_lock=args.dependency_lock,
        expected_visible_gpu_count=args.expected_visible_gpu_count,
        nvidia_smi=args.nvidia_smi,
    )
    attestation = build_runtime_attestation(observations, phase=args.phase)
    reference = _read_object(args.reference) if args.reference is not None else None
    validate_runtime_attestation(attestation, reference=reference)
    _exclusive_write(args.output.resolve(), attestation)
    print(json.dumps(attestation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
