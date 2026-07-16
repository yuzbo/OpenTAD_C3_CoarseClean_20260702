"""Fail-closed Slurm/CUDA environment identity for formal ChronoTransport r2 runs."""

from __future__ import annotations

import importlib
import os
import re
import subprocess
from typing import Any, Mapping

from .protocol import canonical_sha256


REQUIRED_ENVIRONMENT_SCHEMA = "chronotransport-r2-required-environment-v2"
OBSERVED_ENVIRONMENT_SCHEMA = "chronotransport-r2-observed-slurm-environment-v1"

REQUIRED_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema",
        "gpu_model",
        "driver",
        "cuda",
        "pytorch",
        "cudnn",
        "precision",
        "batch_size",
        "environment_sha256",
    }
)
OBSERVED_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema",
        "cuda_visible_devices_raw",
        "slurm_job_id",
        "slurm_step_id",
        "slurm_job_gpus_raw",
        "slurm_step_gpus_raw",
        "slurm_gpus_on_node_raw",
        "logical_cuda_index",
        "torch_device_count",
        "gpu_model",
        "gpu_uuid",
        "driver",
        "cuda",
        "pytorch",
        "cudnn",
        "precision",
        "batch_size",
        "required_environment_sha256",
        "allocation_identity_sha256",
        "observed_environment_sha256",
    }
)
OBSERVED_PROVENANCE_FIELDS = frozenset(
    (OBSERVED_ENVIRONMENT_FIELDS - {"schema"}) | {"observed_environment_schema"}
)

_FULL_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9][A-Za-z0-9-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class InvalidEnvironmentError(RuntimeError):
    """The live Slurm/CUDA allocation differs from the registered requirement."""


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEnvironmentError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise InvalidEnvironmentError(f"{label} must be a lowercase SHA-256")
    return value


def validate_required_environment(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the pre-allocation model/software contract frozen at registration."""

    if not isinstance(value, Mapping) or set(value) != REQUIRED_ENVIRONMENT_FIELDS:
        raise ValueError("required environment fields mismatch")
    if value["schema"] != REQUIRED_ENVIRONMENT_SCHEMA:
        raise ValueError("unsupported required environment schema")
    for field in (
        "gpu_model",
        "driver",
        "cuda",
        "pytorch",
        "cudnn",
        "precision",
    ):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"required environment {field} must be a non-empty string")
    if value["precision"] != "amp_fp16":
        raise ValueError("formal environment precision must be amp_fp16")
    if type(value["batch_size"]) is not int or value["batch_size"] != 1:
        raise ValueError("formal environment batch_size must equal 1")
    unsigned = dict(value)
    supplied = unsigned.pop("environment_sha256")
    if not isinstance(supplied, str) or not _SHA256.fullmatch(supplied):
        raise ValueError("required environment fingerprint must be a lowercase SHA-256")
    if supplied != canonical_sha256(unsigned):
        raise ValueError("required environment fingerprint mismatch")
    return dict(value)


def _allocation_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cuda_visible_devices_raw": value["cuda_visible_devices_raw"],
        "slurm_job_id": value["slurm_job_id"],
        "slurm_step_id": value["slurm_step_id"],
        "slurm_job_gpus_raw": value["slurm_job_gpus_raw"],
        "slurm_step_gpus_raw": value["slurm_step_gpus_raw"],
        "slurm_gpus_on_node_raw": value["slurm_gpus_on_node_raw"],
        "logical_cuda_index": value["logical_cuda_index"],
        "torch_device_count": value["torch_device_count"],
        "gpu_uuid": value["gpu_uuid"],
    }


def validate_observed_environment(
    value: Mapping[str, Any],
    *,
    required_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one live allocation identity against the registered requirement."""

    required = validate_required_environment(required_environment)
    if not isinstance(value, Mapping) or set(value) != OBSERVED_ENVIRONMENT_FIELDS:
        raise InvalidEnvironmentError("observed environment fields mismatch")
    if value["schema"] != OBSERVED_ENVIRONMENT_SCHEMA:
        raise InvalidEnvironmentError("unsupported observed environment schema")
    for field in (
        "cuda_visible_devices_raw",
        "slurm_job_gpus_raw",
        "slurm_step_gpus_raw",
        "slurm_gpus_on_node_raw",
    ):
        if not isinstance(value[field], str):
            raise InvalidEnvironmentError(f"observed environment {field} must be raw text")
    for field in (
        "slurm_job_id",
        "slurm_step_id",
        "gpu_model",
        "gpu_uuid",
        "driver",
        "cuda",
        "pytorch",
        "cudnn",
        "precision",
    ):
        _require_nonempty_string(value[field], f"observed environment {field}")
    if not _FULL_GPU_UUID.fullmatch(value["gpu_uuid"]):
        raise InvalidEnvironmentError(
            "formal r2 requires a full-GPU UUID; MIG or index-only identity is invalid"
        )
    if type(value["logical_cuda_index"]) is not int or value["logical_cuda_index"] != 0:
        raise InvalidEnvironmentError("formal r2 must use logical cuda:0")
    if type(value["torch_device_count"]) is not int or value["torch_device_count"] != 1:
        raise InvalidEnvironmentError("formal r2 requires exactly one torch-visible CUDA device")
    if type(value["batch_size"]) is not int or value["batch_size"] != 1:
        raise InvalidEnvironmentError("formal r2 requires batch_size=1")
    if value["required_environment_sha256"] != required["environment_sha256"]:
        raise InvalidEnvironmentError("observed environment binds a different requirement")
    for field in (
        "gpu_model",
        "driver",
        "cuda",
        "pytorch",
        "cudnn",
        "precision",
        "batch_size",
    ):
        if value[field] != required[field]:
            raise InvalidEnvironmentError(
                f"observed {field} differs from registered required environment"
            )
    allocation_sha256 = _require_sha256(
        value["allocation_identity_sha256"], "allocation identity"
    )
    if allocation_sha256 != canonical_sha256(_allocation_identity(value)):
        raise InvalidEnvironmentError("allocation identity fingerprint mismatch")
    unsigned = dict(value)
    observed_sha256 = _require_sha256(
        unsigned.pop("observed_environment_sha256"), "observed environment identity"
    )
    if observed_sha256 != canonical_sha256(unsigned):
        raise InvalidEnvironmentError("observed environment fingerprint mismatch")
    return dict(value)


def _torch_module():
    return importlib.import_module("torch")


def _run_nvidia_smi(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _current_process_gpu_uuid(pid: int) -> str:
    output = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    matches = []
    for raw in output.splitlines():
        row = [item.strip() for item in raw.split(",", 1)]
        if len(row) != 2:
            continue
        if row[0] == str(pid):
            matches.append(row[1])
    if len(set(matches)) != 1:
        raise InvalidEnvironmentError(
            "cannot bind the current CUDA process to one exact GPU UUID"
        )
    return matches[0]


def _gpu_inventory_row(gpu_uuid: str) -> tuple[str, str]:
    output = _run_nvidia_smi(
        [
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    matches: list[tuple[str, str]] = []
    for raw in output.splitlines():
        row = [item.strip() for item in raw.split(",", 2)]
        if len(row) == 3 and row[0] == gpu_uuid:
            matches.append((row[1], row[2]))
    if len(matches) != 1:
        raise InvalidEnvironmentError("current process GPU UUID is absent or duplicated in inventory")
    return matches[0]


def observe_formal_slurm_environment(
    required_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Observe the current Slurm allocation; no caller-supplied identity is accepted."""

    required = validate_required_environment(required_environment)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_step_id = os.environ.get("SLURM_STEP_ID", "")
    if not slurm_job_id.strip() or not slurm_step_id.strip():
        raise InvalidEnvironmentError(
            "formal r2 requires both SLURM_JOB_ID and SLURM_STEP_ID"
        )

    torch = _torch_module()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise InvalidEnvironmentError("formal r2 requires exactly one CUDA-visible device")
    torch.cuda.set_device(0)
    if torch.cuda.current_device() != 0:
        raise InvalidEnvironmentError("formal r2 failed to select logical cuda:0")
    probe = torch.empty((1,), device=torch.device("cuda:0"))
    torch.cuda.synchronize()
    del probe

    gpu_uuid = _current_process_gpu_uuid(os.getpid())
    gpu_model, driver = _gpu_inventory_row(gpu_uuid)
    torch_gpu_model = str(torch.cuda.get_device_name(0))
    if torch_gpu_model != gpu_model:
        raise InvalidEnvironmentError("torch and nvidia-smi GPU model identities differ")
    observed: dict[str, Any] = {
        "schema": OBSERVED_ENVIRONMENT_SCHEMA,
        "cuda_visible_devices_raw": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "slurm_job_id": slurm_job_id,
        "slurm_step_id": slurm_step_id,
        "slurm_job_gpus_raw": os.environ.get("SLURM_JOB_GPUS", ""),
        "slurm_step_gpus_raw": os.environ.get("SLURM_STEP_GPUS", ""),
        "slurm_gpus_on_node_raw": os.environ.get("SLURM_GPUS_ON_NODE", ""),
        "logical_cuda_index": 0,
        "torch_device_count": 1,
        "gpu_model": torch_gpu_model,
        "gpu_uuid": gpu_uuid,
        "driver": driver,
        "cuda": str(torch.version.cuda),
        "pytorch": str(torch.__version__),
        "cudnn": str(torch.backends.cudnn.version()),
        "precision": "amp_fp16",
        "batch_size": 1,
        "required_environment_sha256": required["environment_sha256"],
    }
    observed["allocation_identity_sha256"] = canonical_sha256(
        _allocation_identity(observed)
    )
    observed["observed_environment_sha256"] = canonical_sha256(observed)
    return validate_observed_environment(
        observed, required_environment=required
    )


def build_test_only_observed_environment(
    required_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic, unmistakably non-formal allocation fixture."""

    required = validate_required_environment(required_environment)
    observed: dict[str, Any] = {
        "schema": OBSERVED_ENVIRONMENT_SCHEMA,
        "cuda_visible_devices_raw": "TEST_ONLY",
        "slurm_job_id": "TEST_ONLY_JOB",
        "slurm_step_id": "TEST_ONLY_STEP",
        "slurm_job_gpus_raw": "TEST_ONLY_GPU",
        "slurm_step_gpus_raw": "TEST_ONLY_GPU",
        "slurm_gpus_on_node_raw": "1",
        "logical_cuda_index": 0,
        "torch_device_count": 1,
        "gpu_model": required["gpu_model"],
        "gpu_uuid": "GPU-TEST-ONLY",
        "driver": required["driver"],
        "cuda": required["cuda"],
        "pytorch": required["pytorch"],
        "cudnn": required["cudnn"],
        "precision": required["precision"],
        "batch_size": required["batch_size"],
        "required_environment_sha256": required["environment_sha256"],
    }
    observed["allocation_identity_sha256"] = canonical_sha256(
        _allocation_identity(observed)
    )
    observed["observed_environment_sha256"] = canonical_sha256(observed)
    return validate_observed_environment(observed, required_environment=required)


def observed_environment_to_provenance(
    observed_environment: Mapping[str, Any],
    *,
    required_environment: Mapping[str, Any],
) -> dict[str, Any]:
    observed = validate_observed_environment(
        observed_environment, required_environment=required_environment
    )
    return {
        "observed_environment_schema": observed["schema"],
        **{key: observed[key] for key in OBSERVED_ENVIRONMENT_FIELDS if key != "schema"},
    }


def observed_environment_from_provenance(
    provenance: Mapping[str, Any],
    *,
    required_environment: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise InvalidEnvironmentError("formal provenance must be a mapping")
    missing = OBSERVED_PROVENANCE_FIELDS - set(provenance)
    if missing:
        raise InvalidEnvironmentError(
            f"formal provenance lacks observed allocation fields: {sorted(missing)}"
        )
    observed = {
        "schema": provenance["observed_environment_schema"],
        **{
            key: provenance[key]
            for key in OBSERVED_ENVIRONMENT_FIELDS
            if key != "schema"
        },
    }
    return validate_observed_environment(
        observed, required_environment=required_environment
    )
