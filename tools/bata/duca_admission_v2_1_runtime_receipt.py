from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.bata.duca_admission_v2_1_hashing import PROTOCOL_ID, sha256_bytes
from tools.bata.duca_admission_v2_1_incidence import validate_incidence
from tools.bata.duca_safe_publication import (
    read_file_beneath_allowlisted_roots,
    read_file_without_symlinks,
)
from tools.bata.duca_evidence_io import (
    canonical_sha256,
    verify_content_sha256,
    with_content_sha256,
)


CONTROL_REGISTRY_SCHEMA = "duca_admission_v2_1_control_registry_v1"
PLANNED_CELL_SCHEMA = "duca_admission_v2_1_planned_cell_manifest_v1"
RUNTIME_RECEIPT_SCHEMA = "duca_admission_v2_1_runtime_receipt_v1"
CONTROL_EVIDENCE_SCHEMA = "duca_admission_v2_1_control_evidence_v1"
RUNTIME_BINDINGS_SCHEMA = "duca_admission_v2_1_runtime_bindings_v1"
SUPERSEDED_ADMISSION_V2_SCHEMA = "duca_acquisition_admission_v2"
DEFAULT_CONTROL_REGISTRY = Path(
    "configs/protocols/duca_admission_v2_1_control_registry_v1.json"
)

EXPECTED_ENUMS = {
    "status": ["PASSED", "FAILED_CLOSED"],
    "classification": [
        "REPOSITORY_ENFORCED",
        "CLUSTER_ATTESTED",
        "OBSERVED_ONLY",
        "OPTIONAL_STRENGTHENING",
        "UNAVAILABLE_LIMITATION",
    ],
    "claim_state": ["VERIFIED", "ATTESTED", "OBSERVED", "NOT_AVAILABLE", "FAILED"],
    "authorization_scope": ["NONE", "HOLDOUT_ONCE", "PHASE1_V2"],
}
EXPECTED_CONTROL_IDS = {
    "REPOSITORY_ENFORCED": tuple(
        f"R{index:02d}_{name}"
        for index, name in enumerate(
            (
                "REMOTE",
                "BRANCH",
                "HEAD",
                "TREE",
                "WORKTREE_CLEAN",
                "PATH_DIFF_ALLOWLIST",
                "PARENT_RECEIPTS",
                "ROLE_MANIFEST",
                "INCIDENCE",
                "METRIC_REGISTRY",
                "SIMULATION_REGISTRY",
                "PLANNED_CELL_MANIFEST",
                "EXECUTABLE_HASHES",
                "CONFIG_CHECKPOINT_HASHES",
                "ALLOWLISTED_OUTPUT_ROOT",
                "NO_SYMLINK_ANCESTORS",
                "FRESH_OUTPUT_ROOT",
                "ATOMIC_PUBLICATION_SELFTEST",
                "EXCLUSIVE_PUBLICATION",
                "ALL_PLANNED_CELLS_PRESENT",
                "NO_EXTRA_CELLS",
                "RECEIPT_CONTENT_HASH",
            ),
            start=1,
        )
    ),
    "CLUSTER_ATTESTED": (
        "C01_SLURM_JOB",
        "C02_SLURM_STEP",
        "C03_PID_START",
        "C04_CGROUP",
        "C05_HOST_BOOT_ID",
        "C06_CUDA_ALLOCATION",
    ),
    "OBSERVED_ONLY": (
        "O01_MOUNT_TABLE",
        "O02_NETWORK_ROUTES",
        "O03_OPEN_SOCKETS",
        "O04_ENVIRONMENT_FINGERPRINT",
    ),
    "OPTIONAL_STRENGTHENING": (
        "S01_NETWORK_DENY",
        "S02_MOUNT_NAMESPACE_ALLOWLIST",
        "S03_INPUT_MOUNTS_READ_ONLY",
    ),
    "UNAVAILABLE_LIMITATION": (
        "L01_COMPLETE_FILE_ACCESS_AUDIT",
        "L02_OBJECT_STORAGE_LOCK",
    ),
}
EXPECTED_ALLOWED_STATES = {
    "REPOSITORY_ENFORCED": ["VERIFIED", "FAILED"],
    "CLUSTER_ATTESTED": ["ATTESTED", "FAILED"],
    "OBSERVED_ONLY": ["OBSERVED", "NOT_AVAILABLE", "FAILED"],
    "OPTIONAL_STRENGTHENING": [
        "VERIFIED",
        "ATTESTED",
        "OBSERVED",
        "NOT_AVAILABLE",
        "FAILED",
    ],
    "UNAVAILABLE_LIMITATION": ["NOT_AVAILABLE"],
}
EXPECTED_FAILURE_CODES = (
    "IDENTITY_MISMATCH",
    "SCHEMA_INVALID",
    "HASH_MISMATCH",
    "SOURCE_INVENTORY_MISMATCH",
    "ROLE_ASSIGNMENT_INVALID",
    "INCIDENCE_INVALID",
    "PLAN_INCOMPLETE",
    "STRUCTURAL_CATASTROPHE",
    "DEGENERATE_SCALE",
    "INVALID_TYPE1_ORDER_INDEX",
    "NUMERIC_TAIL_ALARM",
    "MC_UNSTABLE",
    "SIMULATION_GATE_FAILED",
    "MC_HALF_WIDTH_CALIBRATION_FAILED",
    "RUNTIME_ATTESTATION_MISSING",
    "FILESYSTEM_POLICY_FAILED",
    "PUBLICATION_FAILED",
    "PARENT_RECEIPT_INVALID",
    "UNEXPECTED_EXCEPTION",
)
EXPECTED_AUTHORIZATION_INVARIANTS = {
    "protocol_implementation_receipt_scope": "NONE",
    "simulation_receipt_scope": "NONE",
    "runtime_preflight_receipt_scope": "NONE",
    "official_final_sealed": True,
    "learned_hrime_authorized": False,
    "full_200_refit_authorized": False,
    "phase4_authorized": False,
}
PROTOCOL_ONLY_RECEIPT_KINDS = {
    "protocol_code",
    "unit_test",
    "simulation",
    "runtime_preflight",
}


def _expected_control_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for classification in EXPECTED_ENUMS["classification"]:
        for control_id in EXPECTED_CONTROL_IDS[classification]:
            output.append(
                {
                    "control_id": control_id,
                    "classification": classification,
                    "mandatory_record": True,
                    "pass_required": classification
                    in {"REPOSITORY_ENFORCED", "CLUSTER_ATTESTED"},
                    "allowed_claim_states": EXPECTED_ALLOWED_STATES[classification],
                }
            )
    return output


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_control_registry(
    path: str | Path = DEFAULT_CONTROL_REGISTRY,
) -> dict[str, Any]:
    registry_path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if os.name == "posix":
        raw, _metadata = read_file_without_symlinks(registry_path)
    else:
        # Non-POSIX loading is for unit tests only. Authoritative runtime proof is
        # deliberately gated on the POSIX no-follow path above.
        raw = registry_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("control registry is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("control registry must be a JSON object")
    validate_control_registry(payload)
    return {
        **payload,
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "semantic_sha256": canonical_sha256(payload),
    }


def validate_control_registry(payload: Mapping[str, Any]) -> None:
    base_keys = {
        "schema",
        "protocol_id",
        "enums",
        "controls",
        "failure_codes",
        "authorization_invariants",
    }
    observed_keys = set(payload)
    if observed_keys not in (
        base_keys,
        base_keys | {"artifact_sha256", "semantic_sha256"},
    ):
        raise ValueError("control registry is not a closed-world object")
    core = {key: payload[key] for key in base_keys}
    if observed_keys != base_keys:
        sha256_bytes(
            payload["artifact_sha256"], field_name="control registry artifact sha256"
        )
        sha256_bytes(
            payload["semantic_sha256"], field_name="control registry semantic sha256"
        )
        if payload["semantic_sha256"] != canonical_sha256(core):
            raise ValueError("control registry semantic SHA-256 drifted")
    if core.get("schema") != CONTROL_REGISTRY_SCHEMA:
        raise ValueError("unsupported control registry schema")
    if core.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("control registry protocol drift")
    enums = core.get("enums")
    if not isinstance(enums, Mapping):
        raise ValueError("control registry enums are missing")
    if dict(enums) != EXPECTED_ENUMS:
        raise ValueError("control registry enums drifted")
    controls = core.get("controls")
    if controls != _expected_control_rows():
        raise ValueError(
            "control registry rows drifted from the frozen 37-row registry"
        )
    if core.get("failure_codes") != list(EXPECTED_FAILURE_CODES):
        raise ValueError("failure code registry drifted")
    if core.get("authorization_invariants") != EXPECTED_AUTHORIZATION_INVARIANTS:
        raise ValueError("control registry authorization invariants drifted")


def build_planned_cell_manifest(
    *,
    incidence: Mapping[str, Any],
    metric_registry_sha256: str,
) -> dict[str, Any]:
    validate_incidence(incidence)
    sha256_bytes(metric_registry_sha256, field_name="metric registry sha256")
    cells = []
    for cell in incidence["cells"]:
        role_id = cell["role_id"]
        process_id = cell["process_id"]
        process_leaf = process_id.split(":", 1)[1]
        cell_id = cell["cell_id"]
        cells.append(
            {
                "cell_id": cell_id,
                "role_id": role_id,
                "video_id": cell["video_id"],
                "canonical_video_rank": cell["canonical_video_rank"],
                "slot": cell["slot"],
                "process_id": process_id,
                "worker_id": process_id,
                "expected_output_relpath": f"cells/{role_id}/{process_leaf}/{cell_id}.json",
                "metric_registry_sha256": metric_registry_sha256,
            }
        )
    payload = {
        "schema": PLANNED_CELL_SCHEMA,
        "status": "PASSED",
        "protocol_id": PROTOCOL_ID,
        "incidence_sha256": incidence["content_sha256"],
        "metric_registry_sha256": metric_registry_sha256,
        "cell_count": len(cells),
        "cells": cells,
        "authorization_scope": "NONE",
        "phase1_v2_authorized": False,
        "holdout_open_authorized": False,
        "paper_claim_allowed": False,
        "official_final_sealed": True,
    }
    return with_content_sha256(payload)


def validate_planned_cell_manifest(
    planned_manifest: Mapping[str, Any],
    *,
    incidence: Mapping[str, Any] | None = None,
) -> None:
    if set(planned_manifest) != {
        "schema",
        "status",
        "protocol_id",
        "incidence_sha256",
        "metric_registry_sha256",
        "cell_count",
        "cells",
        "authorization_scope",
        "phase1_v2_authorized",
        "holdout_open_authorized",
        "paper_claim_allowed",
        "official_final_sealed",
        "content_sha256",
    }:
        raise ValueError("planned-cell manifest is not a closed-world object")
    if (
        planned_manifest.get("schema") != PLANNED_CELL_SCHEMA
        or planned_manifest.get("status") != "PASSED"
        or planned_manifest.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("invalid planned-cell manifest identity")
    verify_content_sha256(planned_manifest)
    incidence_sha = planned_manifest.get("incidence_sha256")
    metric_sha = planned_manifest.get("metric_registry_sha256")
    sha256_bytes(incidence_sha, field_name="planned incidence sha256")
    sha256_bytes(metric_sha, field_name="planned metric registry sha256")
    if (
        planned_manifest.get("authorization_scope") != "NONE"
        or planned_manifest.get("phase1_v2_authorized") is not False
        or planned_manifest.get("holdout_open_authorized") is not False
        or planned_manifest.get("paper_claim_allowed") is not False
        or planned_manifest.get("official_final_sealed") is not True
    ):
        raise ValueError("planned-cell manifest contains forbidden authorization")
    cells = planned_manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 192:
        raise ValueError("planned-cell manifest must contain exactly 192 cells")
    if planned_manifest.get("cell_count") != 192:
        raise ValueError("planned-cell manifest cell_count drifted")
    expected_row_keys = {
        "cell_id",
        "role_id",
        "video_id",
        "canonical_video_rank",
        "slot",
        "process_id",
        "worker_id",
        "expected_output_relpath",
        "metric_registry_sha256",
    }
    identifiers: list[str] = []
    previous_order: tuple[int, int, int] | None = None
    role_rank = {"scale_fit": 0, "calibration": 1, "admission_holdout": 2}
    for row in cells:
        if not isinstance(row, Mapping) or set(row) != expected_row_keys:
            raise ValueError("planned-cell row is not closed-world")
        cell_id = row.get("cell_id")
        role_id = row.get("role_id")
        video_id = row.get("video_id")
        process_id = row.get("process_id")
        if not all(
            isinstance(value, str) for value in (cell_id, role_id, video_id, process_id)
        ):
            raise ValueError("planned-cell textual identities must be strings")
        sha256_bytes(cell_id, field_name="planned cell_id")
        if role_id not in role_rank:
            raise ValueError("planned-cell role_id drifted")
        rank = row.get("canonical_video_rank")
        slot = row.get("slot")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < 32:
            raise ValueError("planned-cell canonical_video_rank is invalid")
        if not isinstance(slot, int) or isinstance(slot, bool) or slot not in (0, 1):
            raise ValueError("planned-cell slot is invalid")
        if (
            not process_id.startswith(f"{role_id}:p")
            or len(process_id) != len(role_id) + 4
        ):
            raise ValueError("planned-cell process_id is invalid")
        process_leaf = process_id.split(":", 1)[1]
        if process_leaf not in {f"p{index:02d}" for index in range(8)}:
            raise ValueError("planned-cell process index is invalid")
        if row.get("worker_id") != process_id:
            raise ValueError("planned-cell worker identity drifted")
        expected_relpath = f"cells/{role_id}/{process_leaf}/{cell_id}.json"
        if row.get("expected_output_relpath") != expected_relpath:
            raise ValueError("planned-cell output path drifted")
        if row.get("metric_registry_sha256") != metric_sha:
            raise ValueError("planned-cell metric registry binding drifted")
        order = (role_rank[role_id], rank, slot)
        if previous_order is not None and order <= previous_order:
            raise ValueError(
                "planned-cell rows are not in canonical role/rank/slot order"
            )
        previous_order = order
        identifiers.append(cell_id)
    if len(set(identifiers)) != 192:
        raise ValueError("planned-cell IDs are not unique")
    if incidence is not None:
        validate_incidence(incidence)
        if incidence.get("content_sha256") != incidence_sha:
            raise ValueError("planned-cell incidence binding drifted")
        expected = build_planned_cell_manifest(
            incidence=incidence,
            metric_registry_sha256=metric_sha,
        )
        if planned_manifest != expected:
            raise ValueError(
                "planned-cell manifest does not match deterministic reconstruction"
            )


def validate_planned_cell_outputs(
    planned_manifest: Mapping[str, Any],
    observed_cell_ids: Sequence[str],
    *,
    incidence: Mapping[str, Any] | None = None,
) -> None:
    validate_planned_cell_manifest(planned_manifest, incidence=incidence)
    if not isinstance(observed_cell_ids, Sequence) or isinstance(
        observed_cell_ids, (str, bytes)
    ):
        raise ValueError("observed cell IDs must be a sequence")
    planned = [row["cell_id"] for row in planned_manifest["cells"]]
    observed = list(observed_cell_ids)
    if any(not isinstance(value, str) for value in observed):
        raise ValueError("observed cell IDs must be strings")
    if len(observed) != len(set(observed)):
        raise ValueError("observed cell IDs are duplicated")
    if set(observed) != set(planned):
        raise ValueError("observed cell IDs do not equal the planned cell IDs")


def validate_parent_receipt(
    binding: Mapping[str, Any],
    *,
    root_registry: Mapping[str, Any],
    expected_schema: str,
    expected_stage: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "stage",
        "status",
        "path",
        "file_sha256",
        "content_sha256",
    }
    if set(binding) != expected_keys:
        raise ValueError("parent receipt binding uses an open or incomplete schema")
    for field in ("schema", "stage", "status", "path", "file_sha256", "content_sha256"):
        if not isinstance(binding[field], str):
            raise ValueError("parent receipt binding fields must be strings")
    sha256_bytes(binding["file_sha256"], field_name="parent receipt file sha256")
    sha256_bytes(binding["content_sha256"], field_name="parent receipt content sha256")
    if binding["schema"] == SUPERSEDED_ADMISSION_V2_SCHEMA:
        raise ValueError("superseded Admission v2 receipt is permanently rejected")
    if binding["schema"] != expected_schema or binding["stage"] != expected_stage:
        raise ValueError("parent receipt binding does not match the expected identity")
    raw, _metadata = read_file_beneath_allowlisted_roots(
        binding["path"], root_registry=root_registry
    )
    if hashlib.sha256(raw).hexdigest() != binding["file_sha256"]:
        raise ValueError("parent receipt file SHA-256 drift")
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("parent receipt is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("parent receipt must be a JSON object")
    verify_content_sha256(payload)
    if payload.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("parent receipt content SHA-256 drift")
    if (
        payload.get("schema") != expected_schema
        or payload.get("stage") != expected_stage
    ):
        raise ValueError("parent receipt identity drift")
    if payload.get("status") != "PASSED" or binding["status"] != "PASSED":
        raise ValueError("parent receipt did not pass")
    return payload


def _gpu_tokens(value: str) -> list[str]:
    tokens = []
    for raw in value.split(","):
        token = raw.strip()
        if token.startswith("gpu:"):
            token = token[4:]
        if not token:
            raise RuntimeError("GPU allocation contains an empty token")
        tokens.append(token)
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("GPU allocation contains duplicate tokens")
    return tokens


def _resolve_gpu_tokens(
    tokens: Sequence[str], inventory: Mapping[str, str]
) -> list[str]:
    resolved = []
    uuids = list(inventory.values())
    for token in tokens:
        if token in inventory:
            resolved.append(inventory[token])
            continue
        exact = [value for value in uuids if value == token]
        prefix = [value for value in uuids if value.startswith(token)]
        matches = exact or prefix
        if len(matches) != 1:
            raise RuntimeError(
                f"GPU allocation token is not uniquely mapped by NVML: {token}"
            )
        resolved.append(matches[0])
    return resolved


def collect_linux_worker_identity(*, worker_id: str) -> dict[str, Any]:
    if os.name != "posix" or not Path("/proc/self/stat").is_file():
        raise RuntimeError(
            "authoritative worker identity collection requires Linux /proc"
        )
    if not isinstance(worker_id, str) or ":p" not in worker_id:
        raise RuntimeError("worker_id must be an explicit role-process identity")
    required_env = (
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
    )
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing Slurm/CUDA worker identity fields: {missing}")
    proc_stat = (
        Path("/proc/self/stat").read_text(encoding="utf-8", errors="strict").split()
    )
    if len(proc_stat) < 22:
        raise RuntimeError("/proc/self/stat is incomplete")
    cgroup_raw = Path("/proc/self/cgroup").read_bytes()
    cgroup_text = cgroup_raw.decode("utf-8", errors="strict")
    job_id = os.environ["SLURM_JOB_ID"]
    job_pattern = re.compile(rf"(?:^|[^0-9]){re.escape(job_id)}(?:[^0-9]|$)")
    if not job_pattern.search(cgroup_text):
        raise RuntimeError("worker cgroup does not attest the declared Slurm job")
    cgroup_paths = [
        line.split(b":", 2)[-1].decode("utf-8", errors="strict")
        for line in cgroup_raw.splitlines()
        if line
    ]
    job_paths = sorted({path for path in cgroup_paths if job_pattern.search(path)})
    if len(job_paths) != 1:
        raise RuntimeError("worker cgroup does not map to one unique Slurm job path")
    cgroup_path = job_paths[0]
    cgroup_inode = None
    for prefix in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/unified")):
        candidate = prefix / cgroup_path.lstrip("/")
        try:
            metadata = os.stat(candidate, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            cgroup_inode = int(metadata.st_ino)
            break
    if cgroup_inode is None:
        raise RuntimeError("unable to bind the worker cgroup inode")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("nvidia-smi is unavailable")
    nvidia_smi_path = Path(nvidia_smi).resolve(strict=True)
    if not nvidia_smi_path.is_absolute() or not nvidia_smi_path.is_file():
        raise RuntimeError("nvidia-smi did not resolve to a regular absolute path")
    nvidia_smi_sha256_before = _sha256_file(nvidia_smi_path)
    gpu_rows = subprocess.run(
        [
            str(nvidia_smi_path),
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    nvidia_smi_sha256_after = _sha256_file(nvidia_smi_path)
    if nvidia_smi_sha256_before != nvidia_smi_sha256_after:
        raise RuntimeError("nvidia-smi binary changed during GPU attestation")
    gpu_inventory: dict[str, str] = {}
    for row in gpu_rows:
        if not row.strip():
            continue
        fields = [value.strip() for value in row.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise RuntimeError("NVML GPU inventory row is malformed")
        if fields[0] in gpu_inventory or fields[1] in gpu_inventory.values():
            raise RuntimeError("NVML GPU inventory is duplicated")
        gpu_inventory[fields[0]] = fields[1]
    if not gpu_inventory:
        raise RuntimeError("NVML GPU UUID list is empty")
    cuda_tokens = _gpu_tokens(os.environ["CUDA_VISIBLE_DEVICES"])
    job_tokens = _gpu_tokens(os.environ["SLURM_JOB_GPUS"])
    step_tokens = _gpu_tokens(os.environ["SLURM_STEP_GPUS"])
    cuda_uuids = _resolve_gpu_tokens(cuda_tokens, gpu_inventory)
    job_uuids = _resolve_gpu_tokens(job_tokens, gpu_inventory)
    step_uuids = _resolve_gpu_tokens(step_tokens, gpu_inventory)
    if (
        not cuda_uuids
        or set(cuda_uuids) != set(job_uuids)
        or set(cuda_uuids) != set(step_uuids)
    ):
        raise RuntimeError("CUDA/NVML/Slurm GPU allocation mismatch")
    return {
        "worker_id": worker_id,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_step_id": os.environ["SLURM_STEP_ID"],
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pid_start_ticks": int(proc_stat[21]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="ascii")
        .strip(),
        "cgroup_raw": cgroup_text,
        "cgroup_raw_sha256": hashlib.sha256(cgroup_raw).hexdigest(),
        "cgroup_path": cgroup_path,
        "cgroup_inode": cgroup_inode,
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "slurm_job_gpus": os.environ["SLURM_JOB_GPUS"],
        "slurm_step_gpus": os.environ["SLURM_STEP_GPUS"],
        "nvml_gpu_inventory": [
            {"index": index, "uuid": gpu_inventory[index]}
            for index in sorted(gpu_inventory, key=int)
        ],
        "nvml_gpu_uuids": list(gpu_inventory.values()),
        "allocated_gpu_uuids": cuda_uuids,
        "gpu_cardinality": len(cuda_uuids),
        "nvidia_smi_path": str(nvidia_smi_path),
        "nvidia_smi_sha256": nvidia_smi_sha256_before,
        "cgroup_job_membership_verified": True,
        "cuda_slurm_mapping_verified": True,
    }


def validate_worker_identities(
    identities: Sequence[Mapping[str, Any]],
    *,
    planned_manifest: Mapping[str, Any],
    planned_allocations: Sequence[Mapping[str, Any]],
    live_identity_collector: Callable[[str], Mapping[str, Any]] | None = None,
    require_live_verification: bool = False,
) -> None:
    """Bind the 24 physical workers to planned process and GPU allocations."""

    validate_planned_cell_manifest(planned_manifest)
    expected_worker_ids = {row["worker_id"] for row in planned_manifest["cells"]}
    if len(expected_worker_ids) != 24:
        raise ValueError("planned-cell manifest does not define exactly 24 workers")
    if len(planned_allocations) != 24 or len(identities) != 24:
        raise ValueError(
            "runtime must provide exactly 24 worker identities and allocations"
        )
    allocation_map: dict[str, Mapping[str, Any]] = {}
    for row in planned_allocations:
        if set(row) != {"worker_id", "slurm_job_id", "slurm_step_id", "gpu_uuids"}:
            raise ValueError("planned worker allocation is not closed-world")
        worker_id = row.get("worker_id")
        gpu_uuids = row.get("gpu_uuids")
        if (
            not isinstance(worker_id, str)
            or not isinstance(row.get("slurm_job_id"), str)
            or not isinstance(row.get("slurm_step_id"), str)
            or not isinstance(gpu_uuids, list)
            or not gpu_uuids
            or any(not isinstance(value, str) for value in gpu_uuids)
            or len(gpu_uuids) != len(set(gpu_uuids))
        ):
            raise ValueError("planned worker allocation fields are invalid")
        if worker_id in allocation_map:
            raise ValueError("planned worker allocation IDs are duplicated")
        allocation_map[worker_id] = row
    if set(allocation_map) != expected_worker_ids:
        raise ValueError("planned worker allocation IDs do not match the cell manifest")

    identity_map: dict[str, Mapping[str, Any]] = {}
    identity_tuples: set[tuple[Any, ...]] = set()
    for row in identities:
        worker_id = row.get("worker_id")
        if not isinstance(worker_id, str) or worker_id in identity_map:
            raise ValueError("runtime worker identity is absent or duplicated")
        if row.get("cgroup_job_membership_verified") is not True:
            raise ValueError("runtime worker cgroup/Slurm binding is not attested")
        if row.get("cuda_slurm_mapping_verified") is not True:
            raise ValueError("runtime worker CUDA/Slurm mapping is not attested")
        identity_tuple = tuple(
            row.get(field)
            for field in (
                "slurm_job_id",
                "slurm_step_id",
                "pid",
                "pid_start_ticks",
                "boot_id",
                "cgroup_inode",
            )
        )
        if (
            any(value is None for value in identity_tuple)
            or identity_tuple in identity_tuples
        ):
            raise ValueError(
                "runtime worker identity tuples are incomplete or duplicated"
            )
        identity_tuples.add(identity_tuple)
        if live_identity_collector is not None:
            live = live_identity_collector(worker_id)
            if not isinstance(live, Mapping) or canonical_sha256(
                live
            ) != canonical_sha256(row):
                raise ValueError(
                    "persisted worker identity does not match live attestation"
                )
        identity_map[worker_id] = row
    if require_live_verification and live_identity_collector is None:
        raise ValueError(
            "authoritative worker validation requires live identity re-attestation"
        )
    if set(identity_map) != expected_worker_ids:
        raise ValueError("runtime worker IDs do not match the planned workers")
    for worker_id, expected in allocation_map.items():
        observed = identity_map[worker_id]
        if observed.get("slurm_job_id") != expected["slurm_job_id"]:
            raise ValueError("runtime worker Slurm job allocation drifted")
        if observed.get("slurm_step_id") != expected["slurm_step_id"]:
            raise ValueError("runtime worker Slurm step allocation drifted")
        observed_gpu_uuids = observed.get("allocated_gpu_uuids")
        if (
            not isinstance(observed_gpu_uuids, list)
            or len(observed_gpu_uuids) != len(set(observed_gpu_uuids))
            or set(observed_gpu_uuids) != set(expected["gpu_uuids"])
        ):
            raise ValueError("runtime worker GPU UUID allocation drifted")
        if observed.get("gpu_cardinality") != len(expected["gpu_uuids"]):
            raise ValueError("runtime worker GPU cardinality drifted")


def build_control_evidence(
    *,
    control_id: str,
    claim_state: str,
    evidence_refs: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    if not isinstance(control_id, str) or not isinstance(claim_state, str):
        raise ValueError("control evidence identity must be textual")
    refs: list[dict[str, str]] = []
    for row in evidence_refs:
        if not isinstance(row, Mapping) or set(row) != {"kind", "sha256"}:
            raise ValueError("control evidence reference is not closed-world")
        if not isinstance(row.get("kind"), str) or not row["kind"]:
            raise ValueError("control evidence reference kind is invalid")
        sha256_bytes(row.get("sha256"), field_name="control evidence reference sha256")
        refs.append({"kind": row["kind"], "sha256": row["sha256"]})
    if not refs:
        raise ValueError("control evidence must bind at least one reference")
    refs.sort(key=lambda row: (row["kind"], row["sha256"]))
    if len({(row["kind"], row["sha256"]) for row in refs}) != len(refs):
        raise ValueError("control evidence references must be unique")
    return with_content_sha256(
        {
            "schema": CONTROL_EVIDENCE_SCHEMA,
            "control_id": control_id,
            "claim_state": claim_state,
            "evidence_refs": refs,
        }
    )


def _validate_control_evidence(
    evidence: Mapping[str, Any], *, control_id: str, claim_state: str
) -> None:
    if set(evidence) != {
        "schema",
        "control_id",
        "claim_state",
        "evidence_refs",
        "content_sha256",
    }:
        raise ValueError("runtime control evidence is not closed-world")
    if (
        evidence.get("schema") != CONTROL_EVIDENCE_SCHEMA
        or evidence.get("control_id") != control_id
        or evidence.get("claim_state") != claim_state
    ):
        raise ValueError("runtime control evidence identity drifted")
    refs = evidence.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("runtime control evidence references are missing")
    for row in refs:
        if not isinstance(row, Mapping) or set(row) != {"kind", "sha256"}:
            raise ValueError("runtime control evidence reference is not closed-world")
        if not isinstance(row.get("kind"), str) or not row["kind"]:
            raise ValueError("runtime control evidence reference kind is invalid")
        sha256_bytes(
            row.get("sha256"), field_name="runtime control evidence reference sha256"
        )
    canonical_refs = sorted(refs, key=lambda row: (row["kind"], row["sha256"]))
    if refs != canonical_refs or len(
        {(row["kind"], row["sha256"]) for row in refs}
    ) != len(refs):
        raise ValueError("runtime control evidence references are not canonical")
    verify_content_sha256(evidence)


def build_runtime_bindings(
    *, artifact_refs: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    refs: list[dict[str, str]] = []
    for row in artifact_refs:
        if not isinstance(row, Mapping) or set(row) != {"kind", "sha256"}:
            raise ValueError("runtime binding reference is not closed-world")
        if not isinstance(row.get("kind"), str) or not row["kind"]:
            raise ValueError("runtime binding reference kind is invalid")
        sha256_bytes(row.get("sha256"), field_name="runtime binding reference sha256")
        refs.append({"kind": row["kind"], "sha256": row["sha256"]})
    if not refs:
        raise ValueError(
            "runtime bindings must contain at least one artifact reference"
        )
    refs.sort(key=lambda row: (row["kind"], row["sha256"]))
    if len({(row["kind"], row["sha256"]) for row in refs}) != len(refs):
        raise ValueError("runtime binding references must be unique")
    return with_content_sha256(
        {"schema": RUNTIME_BINDINGS_SCHEMA, "artifact_refs": refs}
    )


def _validate_runtime_bindings(payload: Any) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "artifact_refs",
        "content_sha256",
    }:
        raise ValueError("runtime receipt bindings are not closed-world")
    if payload.get("schema") != RUNTIME_BINDINGS_SCHEMA:
        raise ValueError("runtime receipt bindings schema drifted")
    refs = payload.get("artifact_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("runtime receipt binding references are missing")
    for row in refs:
        if not isinstance(row, Mapping) or set(row) != {"kind", "sha256"}:
            raise ValueError("runtime receipt binding reference is not closed-world")
        if not isinstance(row.get("kind"), str) or not row["kind"]:
            raise ValueError("runtime receipt binding reference kind is invalid")
        sha256_bytes(
            row.get("sha256"), field_name="runtime receipt binding reference sha256"
        )
    canonical_refs = sorted(refs, key=lambda row: (row["kind"], row["sha256"]))
    if refs != canonical_refs or len(
        {(row["kind"], row["sha256"]) for row in refs}
    ) != len(refs):
        raise ValueError("runtime receipt binding references are not canonical")
    verify_content_sha256(payload)


def validate_runtime_receipt(
    payload: Mapping[str, Any],
    *,
    control_registry: Mapping[str, Any],
    evidence_verifiers: Mapping[
        str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    ]
    | None = None,
) -> None:
    validate_control_registry(control_registry)
    if set(payload) != {
        "schema",
        "stage",
        "receipt_kind",
        "status",
        "protocol_id",
        "controls",
        "failure_codes",
        "bindings",
        "authorization_scope",
        "phase1_v2_authorized",
        "holdout_open_authorized",
        "paper_claim_allowed",
        "learned_hrime_authorized",
        "full_200_refit_authorized",
        "phase4_authorized",
        "official_final_sealed",
        "content_sha256",
    }:
        raise ValueError("runtime receipt is not a closed-world object")
    if payload.get("schema") == SUPERSEDED_ADMISSION_V2_SCHEMA:
        raise ValueError("superseded Admission v2 receipt is permanently rejected")
    if payload.get("schema") != RUNTIME_RECEIPT_SCHEMA:
        raise ValueError("unsupported runtime receipt schema")
    if payload.get("stage") != "admission_v2_1_protocol_implementation":
        raise ValueError("runtime receipt stage drift")
    if payload.get("receipt_kind") not in PROTOCOL_ONLY_RECEIPT_KINDS:
        raise ValueError("unknown runtime receipt kind")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("runtime receipt protocol drift")
    verify_content_sha256(payload)
    status = payload.get("status")
    scope = payload.get("authorization_scope")
    if status not in control_registry["enums"]["status"]:
        raise ValueError("unknown runtime receipt status")
    if scope != "NONE":
        raise ValueError("Admission v2.1 protocol-only receipt cannot authorize")
    controls = payload.get("controls")
    if not isinstance(controls, list):
        raise ValueError("runtime receipt controls are missing")
    expected_rows = control_registry["controls"]
    expected = {row["control_id"]: row for row in expected_rows}
    if any(
        not isinstance(row, Mapping)
        or set(row) != {"control_id", "classification", "claim_state", "evidence"}
        or not isinstance(row.get("evidence"), Mapping)
        for row in controls
    ):
        raise ValueError("runtime control row is not closed-world")
    observed_ids = [row.get("control_id") for row in controls]
    if any(not isinstance(value, str) for value in observed_ids):
        raise ValueError("runtime control IDs must be strings")
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(
        expected
    ):
        raise ValueError("runtime receipt has missing, extra or duplicate control IDs")
    if observed_ids != [row["control_id"] for row in expected_rows]:
        raise ValueError("runtime receipt controls are not in canonical registry order")
    failed_required = False
    for row in controls:
        control_id = row["control_id"]
        registered = expected[control_id]
        if row.get("classification") != registered["classification"]:
            raise ValueError("runtime control classification drift")
        state = row.get("claim_state")
        if state not in registered["allowed_claim_states"]:
            raise ValueError("illegal classification/claim_state pair")
        _validate_control_evidence(
            row["evidence"], control_id=control_id, claim_state=state
        )
        if registered["pass_required"]:
            required_state = (
                "VERIFIED"
                if registered["classification"] == "REPOSITORY_ENFORCED"
                else "ATTESTED"
            )
            failed_required |= state != required_state
            if status == "PASSED":
                verifier = (evidence_verifiers or {}).get(control_id)
                if verifier is None or verifier(row["evidence"], row) is not True:
                    raise ValueError(
                        f"mandatory runtime control evidence was not independently verified: {control_id}"
                    )
    failure_codes = payload.get("failure_codes")
    if (
        not isinstance(failure_codes, list)
        or any(not isinstance(value, str) for value in failure_codes)
        or len(failure_codes) != len(set(failure_codes))
        or any(
            value not in control_registry["failure_codes"] for value in failure_codes
        )
    ):
        raise ValueError("runtime receipt failure codes are invalid")
    _validate_runtime_bindings(payload.get("bindings"))
    if status == "PASSED":
        if failed_required or failure_codes:
            raise ValueError("PASSED receipt contains a failed mandatory control")
    else:
        if not failure_codes:
            raise ValueError("FAILED_CLOSED receipt must contain a failure code")
    if payload.get("official_final_sealed") is not True:
        raise ValueError("official-final must remain sealed")
    if any(
        payload.get(field) is not False
        for field in (
            "phase1_v2_authorized",
            "holdout_open_authorized",
            "paper_claim_allowed",
            "learned_hrime_authorized",
            "full_200_refit_authorized",
            "phase4_authorized",
        )
    ):
        raise ValueError(
            "current v2.1 runtime receipt contains forbidden authorization"
        )


def build_protocol_only_runtime_receipt(
    *,
    receipt_kind: str,
    status: str,
    controls: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    failure_codes: Sequence[str] = (),
) -> dict[str, Any]:
    payload = {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "stage": "admission_v2_1_protocol_implementation",
        "receipt_kind": receipt_kind,
        "status": status,
        "protocol_id": PROTOCOL_ID,
        "controls": [dict(row) for row in controls],
        "failure_codes": list(failure_codes),
        "bindings": dict(bindings),
        "authorization_scope": "NONE",
        "phase1_v2_authorized": False,
        "holdout_open_authorized": False,
        "paper_claim_allowed": False,
        "learned_hrime_authorized": False,
        "full_200_refit_authorized": False,
        "phase4_authorized": False,
        "official_final_sealed": True,
    }
    return with_content_sha256(payload)
