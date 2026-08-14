"""Fail-closed contract for the ZoomToken SCNR steady-state cost falsifier.

This module contains only result-blind protocol mechanics.  It never opens the
held-out test, evaluates mAP, changes a checkpoint, or chooses a scientific
route.  Treatment is determined by the frozen leaf ID and tracked config, never
by a caller-provided arm override.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file


SCHEMA_VERSION = "zoomtoken_scnr_steady_cost_contract_v001"
STUDY_ID = "ZT_SCNR_STEADY_COST_V001"
DECISION_NONCE = "ZT-PRO-REMOTE-20260811-9C7FA321D4E8"
BASE_SHA = "2d26662ec4d124dd906a7a3676f29645684fdf96"
PROJECT_ID = "g-p-6a79701398bc8191a9ef61db6302b24b"
REPOSITORY = "yuzbo/OpenTAD_C3_CoarseClean_20260702"
SCIENTIFIC_ROUTE = "SCNR-Core"
TRAINING_SEED = 3407
WINDOW_BUDGET = 24576
PHYSICAL_WINDOWS = 136
VIDEO_CLUSTERS = 40
WARMUP_WINDOWS_PER_PASS = 136
MEASURED_WINDOWS_PER_PASS = 136
POWER_INTERVAL_MS = 20
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_811
NONINFERIORITY_RATIO = 1.05
SOURCE_POPULATION_SHA256 = (
    "35aaa9192b4dfd4bd03599450fbbceed6c7d60e98d8df7f582bd39df26f40aa8"
)

CONTROL_CONFIG = (
    "configs/adatad/thumos/"
    "georoute_dynamic_scnr_stage1_cost_control_seed3407_v001.py"
)
CENTERED_CONFIG = (
    "configs/adatad/thumos/"
    "georoute_dynamic_scnr_stage1_cost_centered_seed3407_v001.py"
)
POPULATION_MANIFEST = (
    "research-wiki/experiments/zoomtoken_scnr_steady_cost_population_v001.json"
)

ARM_TO_VARIANT = {
    "A": "none_control",
    "B": "residual_window_center",
}
ARM_TO_CONFIG = {
    "A": CONTROL_CONFIG,
    "B": CENTERED_CONFIG,
}
VARIANT_TO_CALIBRATION = {
    "none_control": "none",
    "residual_window_center": "residual_window_center",
}
LEAF_ORDERS = {
    **{f"L{index:02d}": "ABBA" for index in range(1, 5)},
    **{f"L{index:02d}": "BAAB" for index in range(5, 9)},
}
PAIR_INDICES_BY_ORDER = {
    "ABBA": ((0, 1), (3, 2)),
    "BAAB": ((1, 0), (2, 3)),
}  # Each pair is normalized to (control pass index, centered pass index).
FINALIZER_DEPENDENCY = "afterany"

# P1 is a separate, result-blind use of the proven steady-cost mechanics.  The
# historical A/B study above remains immutable and cannot be relabelled as P1.
P1_COST_SCHEMA_VERSION = "zoomtoken_p1_steady_cost_v001"
P1_STUDY_ID = "ZOOMTOKEN_P1_DNURQ_V001"
P1_ARM_ORDER = ("DO", "DN", "U", "R", "Q")
P1_COST_COMPARATORS = ("DO", "DN", "U", "R")
P1_COST_LEAF_SPECS = {
    f"{comparator}_{order}": {
        "comparator": comparator,
        "order": order,
    }
    for comparator in P1_COST_COMPARATORS
    for order in ("ABBA", "BAAB")
}
P1_DENSE_PHYSICAL_TOKENS = 384 * 220
P1_COST_RATIO_LIMIT = 0.85
P1_COST_BOOTSTRAP_SEED = BOOTSTRAP_SEED
P1_COST_BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES
P1_COST_METRICS = {
    "selector_inclusive_decode_to_nms_p50": (
        "end_to_end_serial_ms",
        "p50",
    ),
    "mean_gross_nvml_joules_per_window": (
        "gross_gpu_energy_j_per_sample",
        "mean",
    ),
}

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REMOTE_BOUNDARY = PurePosixPath("/data/run01/sczc063/yuzibo")


def _is_within_remote_boundary(value: Any) -> bool:
    path = PurePosixPath(str(value))
    if not path.is_absolute():
        return False
    try:
        path.relative_to(_REMOTE_BOUNDARY)
    except ValueError:
        return False
    return path != _REMOTE_BOUNDARY


def read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be one JSON object")
    return payload


def read_jsonl_objects(path: str | Path, *, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(row)
    return rows


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


def atomic_write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create one publication exactly once; an existing target is never replaced."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def self_hash(payload: Mapping[str, Any], *, field: str) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop(field, None)
    return canonical_sha256(unsigned)


def add_self_hash(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result[field] = self_hash(result, field=field)
    return result


def require_self_hash(payload: Mapping[str, Any], *, field: str, label: str) -> None:
    if payload.get(field) != self_hash(payload, field=field):
        raise ValueError(f"{label} self hash is invalid")


def leaf_sequence(leaf_id: str, declared_order: str | None = None) -> tuple[str, ...]:
    leaf_id = str(leaf_id)
    if leaf_id not in LEAF_ORDERS:
        raise ValueError("steady-cost leaf ID must be exactly L01--L08")
    frozen = LEAF_ORDERS[leaf_id]
    if declared_order is not None and str(declared_order) != frozen:
        raise ValueError("caller-declared order differs from the frozen leaf order")
    return tuple(ARM_TO_VARIANT[arm] for arm in frozen)


def mirrored_pair_indices(leaf_id: str) -> tuple[tuple[int, int], ...]:
    leaf_sequence(leaf_id)
    return PAIR_INDICES_BY_ORDER[LEAF_ORDERS[str(leaf_id)]]


def validate_tracked_config(root: str | Path, arm: str) -> dict[str, Any]:
    """Materialize and hash one tracked config while checking its sole treatment."""

    from mmengine.config import Config

    arm = str(arm)
    if arm not in ARM_TO_CONFIG:
        raise ValueError("unknown steady-cost arm")
    path = (Path(root) / ARM_TO_CONFIG[arm]).resolve()
    cfg = Config.fromfile(str(path))
    variant = ARM_TO_VARIANT[arm]
    expected = VARIANT_TO_CALIBRATION[variant]
    binding = cfg.get("zoomtoken_scnr_steady_cost")
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema_version") != "zoomtoken_scnr_steady_cost_config_v001"
        or binding.get("study_id") != STUDY_ID
        or binding.get("arm") != ("control" if arm == "A" else "centered")
        or int(binding.get("training_seed", -1)) != TRAINING_SEED
        or binding.get("calibration_mode") != expected
        or int(binding.get("exact_window_budget", -1)) != WINDOW_BUDGET
        or int(binding.get("physical_windows", -1)) != PHYSICAL_WINDOWS
        or binding.get("split") != "Gate/development"
        or binding.get("treatment_from_cli_allowed") is not False
        or binding.get("training_or_resume_allowed") is not False
        or binding.get("metric_evaluation_allowed") is not False
        or binding.get("held_out_test_allowed") is not False
        or cfg.model.backbone.custom.georoute_branch_calibration_mode != expected
        or cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled is not False
        or cfg.model.backbone.custom.georoute_role_calibration_telemetry_enabled
        is not False
    ):
        raise ValueError(f"steady-cost {arm} config violates the frozen treatment")
    return {
        "path": ARM_TO_CONFIG[arm],
        "file_sha256": sha256_file(path),
        "canonical_config_sha256": canonical_sha256(cfg.to_dict()),
        "calibration_mode": expected,
    }


def build_execution_binding(
    root: str | Path, *, arm: str, stage: Mapping[str, Any]
) -> dict[str, Any]:
    """Materialize the existing tracked/legacy identities without a new hash ledger."""

    from tools.bata.georoute_residual_centering_cost_contract import (
        build_residual_centering_cost_config,
        validate_residual_centering_cost_config,
    )

    arm = str(arm)
    if arm not in ARM_TO_VARIANT:
        raise ValueError("unknown steady-cost execution arm")
    variant = ARM_TO_VARIANT[arm]
    calibration = VARIANT_TO_CALIBRATION[variant]
    tracked = validate_tracked_config(root, arm)
    cfg = build_residual_centering_cost_config(stage, variant=variant)
    legacy_binding = validate_residual_centering_cost_config(
        cfg, stage=stage, variant=variant
    )
    checkpoint = stage.get("checkpoint_receipt")
    accuracy = stage.get("config_receipts", {}).get("accuracy_a")
    if not isinstance(checkpoint, Mapping) or not isinstance(accuracy, Mapping):
        raise ValueError("steady-cost legacy stage lacks config/checkpoint receipts")
    if (
        legacy_binding.get("variant") != variant
        or legacy_binding.get("branch_calibration_mode") != calibration
        or cfg.model.backbone.custom.georoute_branch_calibration_mode != calibration
        or int(cfg.model.backbone.custom.georoute_window_token_budget) != WINDOW_BUDGET
    ):
        raise ValueError("steady-cost actual execution binding changed")
    return {
        "schema_version": "zoomtoken_scnr_steady_cost_execution_binding_v001",
        "study_id": STUDY_ID,
        "arm": arm,
        "variant": variant,
        "calibration_mode": calibration,
        "tracked_config": tracked,
        "legacy_calibration_mode": legacy_binding["branch_calibration_mode"],
        "legacy_cost_config_sha256": canonical_sha256(cfg.to_dict()),
        "checkpoint_receipt": copy.deepcopy(dict(checkpoint)),
        "bound_accuracy_config_receipt": copy.deepcopy(dict(accuracy)),
    }


def validate_execution_binding_receipt(
    receipt: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    observed = copy.deepcopy(dict(receipt))
    if observed != dict(expected):
        raise ValueError("steady-cost execution binding receipt changed")
    return observed


def validate_pass_receipts(
    root: str | Path,
    receipts: Sequence[Mapping[str, Any]],
    *,
    sequence: Sequence[str],
    source: Mapping[str, Any],
    expected_accuracy_population_sha256: str,
    measured_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild every pass binding and couple its receipt to measured identities."""

    sequence = tuple(sequence)
    if len(receipts) != len(sequence):
        raise ValueError("steady-cost pass receipt count changed")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in measured_rows:
        grouped[int(row.get("pass_index", -1))].append(row)
    checked = []
    for pass_index, (raw, variant) in enumerate(zip(receipts, sequence)):
        receipt = copy.deepcopy(dict(raw))
        unsigned = copy.deepcopy(receipt)
        observed_hash = unsigned.pop("pass_sha256", None)
        arm = "A" if variant == ARM_TO_VARIANT["A"] else "B"
        stage = source["stages"][variant]
        expected = build_execution_binding(root, arm=arm, stage=stage)
        before = receipt.get("execution_binding_before")
        after = receipt.get("execution_binding_after")
        rows = sorted(
            grouped.get(pass_index, []), key=lambda row: int(row["sample_ordinal"])
        )
        if (
            observed_hash != canonical_sha256(unsigned)
            or int(receipt.get("pass_index", -1)) != pass_index
            or receipt.get("variant") != variant
            or receipt.get("branch_calibration_mode")
            != VARIANT_TO_CALIBRATION[variant]
            or int(receipt.get("sample_count", -1)) != PHYSICAL_WINDOWS
            or receipt.get("accuracy_population_sha256")
            != expected_accuracy_population_sha256
            or receipt.get("checkpoint_sha256")
            != expected["checkpoint_receipt"].get("sha256")
            or receipt.get("bound_accuracy_config_sha256")
            != expected["bound_accuracy_config_receipt"].get("sha256")
            or receipt.get("cost_config_sha256")
            != expected["legacy_cost_config_sha256"]
            or len(rows) != PHYSICAL_WINDOWS
            or receipt.get("sample_manifest_sha256")
            != canonical_sha256([row["window_id"] for row in rows])
            or receipt.get("diagnostic_telemetry_inside_timed_forward") is not False
            or receipt.get("training_or_resume_executed") is not False
        ):
            raise ValueError("steady-cost pass receipt is invalid")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError("steady-cost pass lacks before/after execution identity")
        validate_execution_binding_receipt(before, expected=expected)
        validate_execution_binding_receipt(after, expected=expected)
        checked.append(receipt)
    if set(grouped) != set(range(len(sequence))):
        raise ValueError("steady-cost pass receipts do not cover measured rows")
    return checked


def build_runtime_identity(
    pre_run: Mapping[str, Any],
    *,
    hardware: Mapping[str, Any],
    software: Mapping[str, Any],
    slurm_job_constraints: str,
    active_container_path: str | Path,
) -> dict[str, Any]:
    """Compare observed leaf runtime identities with the frozen PRE_RUN runtime."""

    runtime = pre_run.get("runtime")
    packages = software.get("packages")
    if not isinstance(runtime, Mapping) or not isinstance(packages, Mapping):
        raise ValueError("steady-cost runtime identity is incomplete")
    gpu_name = str(hardware.get("gpu_name", "")).strip()
    constraint = str(slurm_job_constraints).strip()
    expected_gpu = str(runtime.get("gpu_constraint_or_sku", "")).strip()
    constraint_tokens = tuple(token.strip() for token in constraint.split(",") if token.strip())
    container_path = Path(active_container_path).resolve()
    dependency_lock_path = Path(str(runtime.get("dependency_lock_path", ""))).resolve()
    if not container_path.is_file() or not dependency_lock_path.is_file():
        raise ValueError("steady-cost active container or dependency lock is missing")
    container_digest = f"sha256:{sha256_file(container_path)}"
    dependency_lock_sha256 = sha256_file(dependency_lock_path)
    if (
        software.get("python") != runtime.get("python_version")
        or packages.get("numpy") != runtime.get("numpy_version")
        or not gpu_name
        or not expected_gpu
        or (gpu_name != expected_gpu and expected_gpu not in constraint_tokens)
        or container_digest != runtime.get("container_digest")
        or dependency_lock_sha256 != runtime.get("dependency_lock_sha256")
    ):
        raise ValueError("steady-cost observed runtime differs from PRE_RUN")
    return {
        "schema_version": "zoomtoken_scnr_steady_cost_runtime_identity_v001",
        "study_id": STUDY_ID,
        "python_version": software["python"],
        "numpy_version": packages["numpy"],
        "gpu_name": gpu_name,
        "slurm_job_constraints": constraint,
        "gpu_constraint_or_sku": expected_gpu,
        "active_container_path": str(container_path),
        "container_digest": container_digest,
        "dependency_lock_path": str(dependency_lock_path),
        "dependency_lock_sha256": dependency_lock_sha256,
    }


def validate_runtime_identity_receipt(
    receipt: Mapping[str, Any],
    *,
    pre_run: Mapping[str, Any],
    hardware: Mapping[str, Any],
    software: Mapping[str, Any],
) -> dict[str, Any]:
    observed = copy.deepcopy(dict(receipt))
    expected = build_runtime_identity(
        pre_run,
        hardware=hardware,
        software=software,
        slurm_job_constraints=str(observed.get("slurm_job_constraints", "")),
        active_container_path=str(observed.get("active_container_path", "")),
    )
    if observed != expected:
        raise ValueError("steady-cost runtime identity receipt changed")
    return observed


def validate_warmup_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    leaf_id: str,
    sequence: Sequence[str],
    population: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Require one ordered, identity-only full-population warmup before each pass."""

    manifest = validate_population_manifest(population)
    sequence = tuple(sequence)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    allowed_fields = {
        "schema_version",
        "leaf_id",
        "pass_index",
        "arm",
        "measurement_phase",
        "warmup",
        "warmup_ordinal",
        "loader_ordinal",
        "video_id",
        "physical_window_id",
        "window_id",
    }
    for raw in rows:
        row = copy.deepcopy(dict(raw))
        if set(row) != allowed_fields:
            raise ValueError("steady-cost warmup ledger contains non-identity evidence")
        grouped[int(row.get("pass_index", -1))].append(row)
    if set(grouped) != set(range(len(sequence))):
        raise ValueError("steady-cost warmup ledger does not cover every pass")
    normalized = []
    for pass_index, variant in enumerate(sequence):
        pass_rows = sorted(grouped[pass_index], key=lambda row: int(row["warmup_ordinal"]))
        if len(pass_rows) != WARMUP_WINDOWS_PER_PASS:
            raise ValueError("steady-cost warmup did not traverse the full population")
        for ordinal, (row, expected) in enumerate(zip(pass_rows, manifest["windows"])):
            physical_id = str(expected["physical_window_id"])
            if (
                row.get("schema_version")
                != "zoomtoken_scnr_steady_cost_warmup_identity_v001"
                or row.get("leaf_id") != leaf_id
                or int(row.get("pass_index", -1)) != pass_index
                or row.get("arm") != variant
                or row.get("measurement_phase") != "warmup"
                or row.get("warmup") is not True
                or int(row.get("warmup_ordinal", -1)) != ordinal
                or int(row.get("loader_ordinal", -1)) != ordinal
                or row.get("video_id") != expected["video_id"]
                or row.get("physical_window_id") != physical_id
                or row.get("window_id") != f"{physical_id}#{ordinal}"
            ):
                raise ValueError("steady-cost warmup identity/order changed")
            normalized.append(row)
    return normalized


def validate_paired_configs(root: str | Path) -> None:
    """Prove the tracked pair differs only in its explicit calibration binding."""

    from mmengine.config import Config

    control = Config.fromfile(str(Path(root) / CONTROL_CONFIG)).to_dict()
    centered = Config.fromfile(str(Path(root) / CENTERED_CONFIG)).to_dict()
    centered["model"]["backbone"]["custom"]["georoute_branch_calibration_mode"] = "none"
    centered["georoute_protocol"]["branch_calibration"] = "none"
    centered["zoomtoken_scnr_steady_cost"]["arm"] = "control"
    centered["zoomtoken_scnr_steady_cost"]["calibration_mode"] = "none"
    if canonical_sha256(control) != canonical_sha256(centered):
        raise ValueError("steady-cost configs differ outside residual-window calibration")


def validate_population_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(payload))
    windows = manifest.get("windows")
    if (
        manifest.get("schema_version") != "zoomtoken_scnr_steady_cost_population_v001"
        or manifest.get("study_id") != STUDY_ID
        or manifest.get("benchmark") != "THUMOS14"
        or manifest.get("split") != "Gate/development"
        or int(manifest.get("record_count", -1)) != PHYSICAL_WINDOWS
        or int(manifest.get("video_cluster_count", -1)) != VIDEO_CLUSTERS
        or manifest.get("source_population_sha256") != SOURCE_POPULATION_SHA256
        or not isinstance(windows, list)
        or len(windows) != PHYSICAL_WINDOWS
    ):
        raise ValueError("steady-cost population manifest header is invalid")
    expected_ordinals = list(range(PHYSICAL_WINDOWS))
    observed_ordinals = []
    physical_ids = []
    videos = set()
    for row in windows:
        if not isinstance(row, Mapping):
            raise ValueError("steady-cost population row must be an object")
        ordinal = int(row.get("ordinal", -1))
        video_id = str(row.get("video_id", ""))
        first = int(row.get("window_center_first", -1))
        last = int(row.get("window_center_last", -1))
        physical_id = str(row.get("physical_window_id", ""))
        if (
            ordinal < 0
            or not video_id.startswith("video_validation_")
            or first < 0
            or last < first
            or physical_id != f"{video_id}:{first}"
        ):
            raise ValueError("steady-cost population row identity is invalid")
        observed_ordinals.append(ordinal)
        physical_ids.append(physical_id)
        videos.add(video_id)
    if (
        observed_ordinals != expected_ordinals
        or len(set(physical_ids)) != PHYSICAL_WINDOWS
        or len(videos) != VIDEO_CLUSTERS
    ):
        raise ValueError("steady-cost population is duplicate, reordered, or incomplete")
    require_self_hash(manifest, field="manifest_sha256", label="population manifest")
    return manifest


def population_signature(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = validate_population_manifest(payload)
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "source_population_sha256": manifest["source_population_sha256"],
        "physical_window_ids_sha256": canonical_sha256(
            [row["physical_window_id"] for row in manifest["windows"]]
        ),
    }


def validate_preregistration(payload: Mapping[str, Any]) -> dict[str, Any]:
    prereg = copy.deepcopy(dict(payload))
    if (
        prereg.get("schema_version") != "zoomtoken_scnr_steady_cost_prereg_v001"
        or prereg.get("study_id") != STUDY_ID
        or prereg.get("decision_nonce") != DECISION_NONCE
        or prereg.get("base_sha") != BASE_SHA
        or prereg.get("status") != "DESIGNED_NOT_EXECUTED"
        or prereg.get("leaf_orders") != LEAF_ORDERS
        or int(prereg.get("physical_windows", -1)) != PHYSICAL_WINDOWS
        or int(prereg.get("warmup_windows_before_each_pass", -1))
        != WARMUP_WINDOWS_PER_PASS
        or int(prereg.get("measured_windows_per_pass", -1))
        != MEASURED_WINDOWS_PER_PASS
        or int(prereg.get("bootstrap_replicates", -1)) != BOOTSTRAP_REPLICATES
        or int(prereg.get("bootstrap_seed", -1)) != BOOTSTRAP_SEED
        or float(prereg.get("noninferiority_ratio", -1.0))
        != NONINFERIORITY_RATIO
        or prereg.get("finalizer_dependency") != FINALIZER_DEPENDENCY
        or prereg.get("job9_authoritative") is not False
        or prereg.get("admission_requires_job9_completed_0_0") is not True
        or prereg.get("held_out_test_allowed") is not False
        or prereg.get("metric_evaluation_allowed") is not False
        or prereg.get("automatic_retry_allowed") is not False
    ):
        raise ValueError("steady-cost preregistration changed")
    require_self_hash(prereg, field="prereg_sha256", label="preregistration")
    return prereg


def validate_jobgraph(payload: Mapping[str, Any]) -> dict[str, Any]:
    graph = copy.deepcopy(dict(payload))
    leaves = graph.get("leaves")
    finalizer = graph.get("finalizer")
    if (
        graph.get("schema_version") != "zoomtoken_scnr_steady_cost_jobgraph_v001"
        or graph.get("study_id") != STUDY_ID
        or not isinstance(leaves, list)
        or len(leaves) != 8
        or not isinstance(finalizer, Mapping)
        or finalizer.get("job_label") != "J09"
        or finalizer.get("dependency") != FINALIZER_DEPENDENCY
        or finalizer.get("predecessors") != list(LEAF_ORDERS)
        or finalizer.get("gpu_count") != 0
        or finalizer.get("writes_authoritative_decision") is not False
    ):
        raise ValueError("steady-cost jobgraph header is invalid")
    expected = [
        {
            "leaf_id": leaf_id,
            "order": order,
            "gpu_count": 1,
            "measured_passes": 4,
            "warmup_passes": 4,
            "window_executions": 1088,
            "requeue": False,
            "automatic_retry": False,
        }
        for leaf_id, order in LEAF_ORDERS.items()
    ]
    if leaves != expected:
        raise ValueError("steady-cost leaf graph changed")
    require_self_hash(graph, field="jobgraph_sha256", label="jobgraph")
    return graph


def validate_pre_run(payload: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    """Validate a coordinator-populated PRE_RUN record without mutating it."""

    record = copy.deepcopy(dict(payload))
    if phase not in {"contract_gate", "preflight", "full"}:
        raise ValueError("unknown PRE_RUN phase")
    repair_sha = str(record.get("repair_sha", ""))
    configs = record.get("configs")
    checkpoints = record.get("checkpoints")
    population = record.get("population")
    runtime = record.get("runtime")
    budget = record.get("budget")
    repair_state = record.get("repair_state")
    fairness = record.get("fairness")
    statistics = record.get("statistics")
    model_runtime_sha = str(record.get("model_runtime_sha", ""))
    if (
        record.get("schema_version") != "zoomtoken_scnr_steady_cost_pre_run_v001"
        or record.get("study_id") != STUDY_ID
        or record.get("decision_nonce") != DECISION_NONCE
        or record.get("project_id") != PROJECT_ID
        or record.get("repository") != REPOSITORY
        or record.get("scientific_route") != SCIENTIFIC_ROUTE
        or record.get("base_sha") != BASE_SHA
        or record.get("phase") != phase
        or not _SHA_RE.fullmatch(repair_sha)
        or not _SHA_RE.fullmatch(model_runtime_sha)
        or record.get("status") != "PRE_RUN_READY"
        or not isinstance(repair_state, Mapping)
        or repair_state.get("github_resolvable") is not True
        or repair_state.get("clean_checkout") is not True
        or repair_state.get("staged_diff_empty") is not True
        or repair_state.get("unstaged_diff_empty") is not True
        or repair_state.get("untracked_files_empty") is not True
        or not isinstance(configs, Mapping)
        or set(configs) != {"control", "centered"}
        or not isinstance(checkpoints, Mapping)
        or int(checkpoints.get("training_seed", -1)) != TRAINING_SEED
        or not isinstance(population, Mapping)
        or population.get("benchmark") != "THUMOS14"
        or population.get("split") != "Gate/development"
        or int(population.get("physical_windows", -1)) != PHYSICAL_WINDOWS
        or int(population.get("unique_windows", -1)) != PHYSICAL_WINDOWS
        or int(population.get("video_clusters", -1)) != VIDEO_CLUSTERS
        or population.get("canonical_order_required") is not True
        or population.get("source_population_sha256") != SOURCE_POPULATION_SHA256
        or population.get("manifest_path") != POPULATION_MANIFEST
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(population.get("manifest_file_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(population.get("manifest_sha256", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(population.get("physical_window_ids_sha256", ""))
        )
        or not isinstance(runtime, Mapping)
        or runtime.get("scheduler") != "Slurm"
        or runtime.get("login_node_model_or_data_execution") is not False
        or runtime.get("fixed_physical_gpu") is not False
        or runtime.get("cuda_visible_devices_override") is not False
        or runtime.get("process_device") != "cuda:0"
        or int(runtime.get("gpus_per_gpu_job", -1)) != 1
        or runtime.get("requeue") is not False
        or runtime.get("automatic_retry") is not False
        or not re.search(
            r"sha256:[0-9a-f]{64}", str(runtime.get("container_digest", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(runtime.get("dependency_lock_sha256", ""))
        )
        or not _is_within_remote_boundary(runtime.get("dependency_lock_path", ""))
        or not isinstance(runtime.get("python_version"), str)
        or not runtime.get("python_version")
        or not isinstance(runtime.get("numpy_version"), str)
        or not runtime.get("numpy_version")
        or not isinstance(runtime.get("gpu_constraint_or_sku"), str)
        or not runtime.get("gpu_constraint_or_sku")
        or not isinstance(budget, Mapping)
        or not budget.get("existing_account_or_allocation_receipt")
        or not budget.get("walltime_ceiling")
        or not budget.get("quota_or_cost_ceiling")
        or budget.get("new_unbounded_budget") is not False
        or not isinstance(fairness, Mapping)
        or int(fairness.get("native_token_budget_B", -1)) != WINDOW_BUDGET
        or any(
            fairness.get(field) is not True
            for field in (
                "same_preprocessing",
                "same_decode_scope",
                "same_H2D_scope",
                "same_model_scope",
                "same_postprocess_NMS_scope",
                "same_hardware_constraint",
                "same_container_digest",
                "no_GT_or_mAP",
                "no_checkpoint_selection",
                "no_padding_or_dummy_heavy_tokens",
            )
        )
        or not isinstance(statistics, Mapping)
        or int(statistics.get("bootstrap_replicates", -1)) != BOOTSTRAP_REPLICATES
        or int(statistics.get("bootstrap_seed", -1)) != BOOTSTRAP_SEED
        or statistics.get("confidence_interval") != "percentile_95"
        or float(statistics.get("noninferiority_ratio", float("nan")))
        != NONINFERIORITY_RATIO
        or statistics.get("equality_at_1_05") != "PASS"
        or statistics.get("outlier_exclusion") is not False
        or statistics.get("estimator_switch") is not False
        or not _is_within_remote_boundary(record.get("results_root", ""))
        or not _is_within_remote_boundary(record.get("training_run_root", ""))
        or record.get("result_namespace_empty") is not True
        or record.get("overwrite_existing_namespace") is not False
        or record.get("held_out_test_allowed") is not False
        or record.get("metric_evaluation_allowed") is not False
        or record.get("checkpoint_selection_allowed") is not False
    ):
        raise ValueError("steady-cost PRE_RUN record is incomplete or changed")
    for arm, expected_path in (("control", CONTROL_CONFIG), ("centered", CENTERED_CONFIG)):
        receipt = configs[arm]
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("path") != expected_path
            or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("file_sha256", "")))
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(receipt.get("canonical_config_sha256", ""))
            )
        ):
            raise ValueError(f"steady-cost {arm} config receipt is invalid")
    for arm in ("control", "centered"):
        receipt = checkpoints.get(arm)
        if (
            not isinstance(receipt, Mapping)
            or not _is_within_remote_boundary(receipt.get("path", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", "")))
        ):
            raise ValueError(f"steady-cost {arm} checkpoint receipt is invalid")
    required_gates = ()
    if phase == "preflight":
        required_gates = ("contract_gate",)
    elif phase == "full":
        required_gates = ("contract_gate", "preflight")
    gates = record.get("gates", {})
    if not isinstance(gates, Mapping) or set(gates) != set(required_gates):
        raise ValueError("steady-cost PRE_RUN gate set is invalid")
    expected_status = {
        "contract_gate": "CONTRACT_GATE_READY",
        "preflight": "MECHANICAL_READY",
    }
    for name in required_gates:
        receipt = gates[name]
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("status") != expected_status[name]
            or not _is_within_remote_boundary(receipt.get("path", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", "")))
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(receipt.get("pre_run_sha256", ""))
            )
        ):
            raise ValueError(f"steady-cost {name} gate receipt is invalid")
    require_self_hash(record, field="pre_run_sha256", label="PRE_RUN")
    return record


def validate_leaf_rows(
    rows: Sequence[Mapping[str, Any]], *, leaf_id: str
) -> list[dict[str, Any]]:
    """Require four measured traversals and exclude every warmup observation."""

    sequence = leaf_sequence(leaf_id)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = copy.deepcopy(dict(raw))
        observed_sample_hash = row.pop("sample_sha256", None)
        if observed_sample_hash != canonical_sha256(row):
            raise ValueError("steady-cost measured sample self hash is invalid")
        row["sample_sha256"] = observed_sample_hash
        if row.get("measurement_phase") != "measured" or row.get("warmup") is not False:
            raise ValueError("warmup data entered steady-cost statistical rows")
        grouped[int(row.get("pass_index", -1))].append(row)
    if set(grouped) != set(range(4)):
        raise ValueError("steady-cost leaf does not contain four measured passes")
    manifests = []
    normalized = []
    for pass_index, variant in enumerate(sequence):
        pass_rows = sorted(grouped[pass_index], key=lambda row: int(row["sample_ordinal"]))
        if len(pass_rows) != PHYSICAL_WINDOWS:
            raise ValueError("steady-cost measured pass is not the complete population")
        manifest = []
        for ordinal, row in enumerate(pass_rows):
            if (
                int(row.get("sample_ordinal", -1)) != ordinal
                or int(row.get("loader_ordinal", -1)) != ordinal
                or row.get("arm") != variant
                or row.get("leaf_id") != leaf_id
                or not row.get("window_id")
                or int(row.get("exact_window_budget", -1)) != WINDOW_BUDGET
                or int(row.get("selected_physical_tokens", -1)) != WINDOW_BUDGET
                or int(row.get("executed_physical_tokens", -1)) != WINDOW_BUDGET
                or int(row.get("duplicate_selected_physical_tokens", -1)) != 0
                or int(row.get("padded_heavy_tokens", -1)) != 0
                or not row.get("physical_window_id")
                or not row.get("video_id")
                or not isinstance(row.get("route_audit"), Mapping)
                or int(row["route_audit"].get("exact_window_budget", -1))
                != WINDOW_BUDGET
                or int(row["route_audit"].get("padded_heavy_tokens", -1)) != 0
                or row["route_audit"].get("branch_calibration_mode")
                != VARIANT_TO_CALIBRATION[variant]
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(row["route_audit"].get("physical_indices_sha256", "")),
                )
                or int(row["route_audit"].get("k_t_min", -1)) < 0
                or int(row["route_audit"].get("k_t_max", -1)) < int(
                    row["route_audit"].get("k_t_min", -1)
                )
                or int(row["route_audit"].get("k_t_zero_count", -1)) < 0
                or not isinstance(row["route_audit"].get("role_counts"), Mapping)
                or int(row["route_audit"].get("attention_pairs", -1)) <= 0
                or not isinstance(row["route_audit"].get("clip_token_counts"), list)
                or any(
                    int(value) < 0
                    for value in row["route_audit"].get("clip_token_counts", [])
                )
                or sum(
                    int(value)
                    for value in row["route_audit"].get("clip_token_counts", [])
                )
                != WINDOW_BUDGET
                or any(
                    not math.isfinite(float(row.get(field, float("nan"))))
                    or float(row[field]) <= 0.0
                    for field in (
                        "input_pipeline_serial_ms",
                        "h2d_ms",
                        "decode_to_window_output_wall_ms",
                        "model_forward_ms",
                        "postprocess_ms",
                        "final_video_nms_ms",
                        "end_to_end_serial_ms",
                        "peak_gpu_allocated_mb",
                        "peak_gpu_reserved_mb",
                        "gross_gpu_energy_j_per_sample",
                    )
                )
            ):
                raise ValueError("steady-cost measured row violates route or cost invariants")
            manifest.append(str(row["physical_window_id"]))
            normalized.append(row)
        if len(set(manifest)) != PHYSICAL_WINDOWS:
            raise ValueError("steady-cost measured pass contains duplicate windows")
        manifests.append(manifest)
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise ValueError("steady-cost pass order/population changed within a leaf")
    return normalized


def _metric(values: Sequence[float], reducer: str) -> float:
    checked = np.asarray(values, dtype=np.float64)
    if checked.size == 0 or not np.all(np.isfinite(checked)) or np.any(checked <= 0.0):
        raise ValueError("steady-cost metric values must be finite and positive")
    if reducer == "p50":
        return float(np.median(checked))
    if reducer == "mean":
        return float(np.mean(checked))
    raise ValueError("unsupported steady-cost reducer")


def _ratio(
    control_rows: Sequence[Mapping[str, Any]],
    centered_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    reducer: str,
) -> float:
    denominator = _metric([float(row[field]) for row in control_rows], reducer)
    numerator = _metric([float(row[field]) for row in centered_rows], reducer)
    return numerator / denominator


def _analyze_complete_leaves_with_draws(
    leaves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Video-cluster/pass-pair bootstrap over all eight independent leaf Jobs."""

    if set(leaves) != set(LEAF_ORDERS) or int(bootstrap_replicates) <= 0:
        raise ValueError("steady-cost analysis requires all eight leaves")
    pass_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    canonical_manifest: list[tuple[str, str]] | None = None
    for leaf_id in sorted(LEAF_ORDERS):
        normalized = validate_leaf_rows(leaves[leaf_id], leaf_id=leaf_id)
        for pass_index in range(4):
            rows = [row for row in normalized if int(row["pass_index"]) == pass_index]
            pass_rows[(leaf_id, pass_index)] = rows
            manifest = [
                (str(row["video_id"]), str(row["physical_window_id"])) for row in rows
            ]
            canonical_manifest = manifest if canonical_manifest is None else canonical_manifest
            if manifest != canonical_manifest:
                raise ValueError("steady-cost leaves changed canonical window order")
    if canonical_manifest is None:
        raise ValueError("steady-cost bootstrap population is missing")
    ordered_videos = sorted({video_id for video_id, _ in canonical_manifest})
    if len(ordered_videos) != VIDEO_CLUSTERS:
        raise ValueError("steady-cost bootstrap population is incomplete")
    by_pass_video: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    for key, rows in pass_rows.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["video_id"])].append(row)
        if sorted(grouped) != ordered_videos:
            raise ValueError("steady-cost pass changed canonical video clusters")
        by_pass_video[key] = grouped

    metrics = {
        "end_to_end_p50": ("end_to_end_serial_ms", "p50"),
        "gross_gpu_energy_per_sample": ("gross_gpu_energy_j_per_sample", "mean"),
    }
    point_control: list[dict[str, Any]] = []
    point_centered: list[dict[str, Any]] = []
    for leaf_id in sorted(LEAF_ORDERS):
        for control_index, centered_index in mirrored_pair_indices(leaf_id):
            point_control.extend(pass_rows[(leaf_id, control_index)])
            point_centered.extend(pass_rows[(leaf_id, centered_index)])

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = {
        name: np.empty(int(bootstrap_replicates), dtype=np.float64) for name in metrics
    }
    strata = (
        tuple(sorted(leaf_id for leaf_id in LEAF_ORDERS if LEAF_ORDERS[leaf_id] == "ABBA")),
        tuple(sorted(leaf_id for leaf_id in LEAF_ORDERS if LEAF_ORDERS[leaf_id] == "BAAB")),
    )
    for replicate in range(int(bootstrap_replicates)):
        sampled_control: list[dict[str, Any]] = []
        sampled_centered: list[dict[str, Any]] = []
        for stratum in strata:
            job_draw = rng.integers(0, len(stratum), size=4)
            for job_index in job_draw:
                leaf_id = stratum[int(job_index)]
                pairs = mirrored_pair_indices(leaf_id)
                pair_draw = rng.integers(0, len(pairs), size=2)
                for pair_index in pair_draw:
                    control_index, centered_index = pairs[int(pair_index)]
                    video_draw = rng.integers(0, len(ordered_videos), size=VIDEO_CLUSTERS)
                    for video_index in video_draw:
                        video_id = ordered_videos[int(video_index)]
                        sampled_control.extend(
                            by_pass_video[(leaf_id, control_index)][video_id]
                        )
                        sampled_centered.extend(
                            by_pass_video[(leaf_id, centered_index)][video_id]
                        )
        for name, (field, reducer) in metrics.items():
            draws[name][replicate] = _ratio(
                sampled_control, sampled_centered, field=field, reducer=reducer
            )

    results = {}
    for name, (field, reducer) in metrics.items():
        estimate = _ratio(point_control, point_centered, field=field, reducer=reducer)
        interval_values = np.quantile(
            draws[name], [0.025, 0.975], method="linear"
        )
        interval = [float(interval_values[0]), float(interval_values[1])]
        upper_bound_passes = interval[1] <= NONINFERIORITY_RATIO
        results[name] = {
            "field": field,
            "reducer": reducer,
            "centered_over_control_ratio": estimate,
            "percentile_95_ci": interval,
            "upper_bound_le_1_05": upper_bound_passes,
            "record_count_per_arm": len(point_control),
        }
    secondary_diagnostics = {}
    for arm, rows in (("control", point_control), ("centered", point_centered)):
        route_rows = [row["route_audit"] for row in rows]
        clip_counts = [
            int(value) for route in route_rows for value in route["clip_token_counts"]
        ]
        secondary_diagnostics[arm] = {
            "descriptive_only": True,
            "latency_p95_ms": float(
                np.quantile(
                    [float(row["end_to_end_serial_ms"]) for row in rows],
                    0.95,
                    method="linear",
                )
            ),
            "peak_gpu_allocated_mb": max(
                float(row["peak_gpu_allocated_mb"]) for row in rows
            ),
            "peak_gpu_reserved_mb": max(
                float(row["peak_gpu_reserved_mb"]) for row in rows
            ),
            "mean_component_ms": {
                field: float(np.mean([float(row[field]) for row in rows]))
                for field in (
                    "input_pipeline_serial_ms",
                    "h2d_ms",
                    "decode_to_window_output_wall_ms",
                    "model_forward_ms",
                    "postprocess_ms",
                    "final_video_nms_ms",
                )
            },
            "k_t_min": min(int(route["k_t_min"]) for route in route_rows),
            "k_t_max": max(int(route["k_t_max"]) for route in route_rows),
            "k_t_zero_count_mean": float(
                np.mean([int(route["k_t_zero_count"]) for route in route_rows])
            ),
            "clip_token_count_p50": float(np.median(clip_counts)),
            "clip_token_count_p95": float(
                np.quantile(clip_counts, 0.95, method="linear")
            ),
            "attention_pairs_mean": float(
                np.mean([int(route["attention_pairs"]) for route in route_rows])
            ),
        }
    passed = all(result["upper_bound_le_1_05"] for result in results.values())
    analysis = {
        "schema_version": "zoomtoken_scnr_steady_cost_analysis_v001",
        "study_id": STUDY_ID,
        "leaf_orders": dict(LEAF_ORDERS),
        "video_cluster_count": len(ordered_videos),
        "pass_pair_count": 16,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_rng": "numpy.random.Generator(PCG64)",
        "bootstrap_quantile": "numpy.quantile(method=linear)",
        "bootstrap_hierarchy": (
            "four_ABBA_jobs_and_four_BAAB_jobs_then_two_pairs_per_job_"
            "then_40_paired_video_clusters"
        ),
        "noninferiority_ratio": NONINFERIORITY_RATIO,
        "metrics": results,
        "secondary_diagnostics": secondary_diagnostics,
        "cost_noninferior": passed,
        "candidate_decision": (
            "PASS_COST_NONINFERIOR" if passed else "FAIL_COST_NONINFERIOR"
        ),
    }
    draw_rows = [
        {
            "schema_version": "zoomtoken_scnr_steady_cost_bootstrap_draw_v001",
            "replicate": replicate,
            **{name: float(values[replicate]) for name, values in draws.items()},
        }
        for replicate in range(int(bootstrap_replicates))
    ]
    validate_bootstrap_draws(draw_rows, analysis=analysis)
    return analysis, draw_rows


def validate_bootstrap_draws(
    rows: Sequence[Mapping[str, Any]], *, analysis: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected_count = int(analysis.get("bootstrap_replicates", -1))
    metric_names = tuple(analysis.get("metrics", {}))
    expected_fields = {"schema_version", "replicate", *metric_names}
    if expected_count <= 0 or metric_names != (
        "end_to_end_p50",
        "gross_gpu_energy_per_sample",
    ) or len(rows) != expected_count:
        raise ValueError("steady-cost bootstrap draw ledger is incomplete")
    checked = []
    for replicate, raw in enumerate(rows):
        row = copy.deepcopy(dict(raw))
        if (
            set(row) != expected_fields
            or row.get("schema_version")
            != "zoomtoken_scnr_steady_cost_bootstrap_draw_v001"
            or int(row.get("replicate", -1)) != replicate
            or any(
                not math.isfinite(float(row.get(name, float("nan"))))
                or float(row[name]) <= 0.0
                for name in metric_names
            )
        ):
            raise ValueError("steady-cost bootstrap draw ledger is invalid")
        checked.append(row)
    return checked


def analyze_complete_leaves_with_draws(
    leaves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _analyze_complete_leaves_with_draws(
        leaves, bootstrap_replicates=bootstrap_replicates
    )


def analyze_complete_leaves(
    leaves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    analysis, _draws = analyze_complete_leaves_with_draws(
        leaves, bootstrap_replicates=bootstrap_replicates
    )
    return analysis


def p1_cost_leaf_spec(leaf_id: str) -> dict[str, str]:
    try:
        return dict(P1_COST_LEAF_SPECS[str(leaf_id)])
    except KeyError as error:
        raise ValueError("P1 cost leaf ID is outside the frozen eight leaves") from error


def p1_cost_leaf_sequence(leaf_id: str) -> tuple[str, ...]:
    spec = p1_cost_leaf_spec(leaf_id)
    comparator = spec["comparator"]
    return tuple(comparator if symbol == "A" else "Q" for symbol in spec["order"])


def p1_cost_leaf_relative_path(leaf_id: str) -> Path:
    spec = p1_cost_leaf_spec(leaf_id)
    return Path("cost") / spec["comparator"] / spec["order"]


def _p1_expected_physical_tokens(arm: str) -> int:
    if arm in {"DO", "DN"}:
        return P1_DENSE_PHYSICAL_TOKENS
    if arm in {"U", "R", "Q"}:
        return WINDOW_BUDGET
    raise ValueError("unknown P1 cost arm")


def validate_p1_cost_rows(
    rows: Sequence[Mapping[str, Any]], *, leaf_id: str
) -> list[dict[str, Any]]:
    """Validate one P1 ABBA/BAAB leaf without weakening the legacy A/B study."""

    sequence = p1_cost_leaf_sequence(leaf_id)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = copy.deepcopy(dict(raw))
        observed_hash = row.pop("sample_sha256", None)
        if observed_hash != canonical_sha256(row):
            raise ValueError("P1 cost sample self hash is invalid")
        row["sample_sha256"] = observed_hash
        if row.get("measurement_phase") != "measured" or row.get("warmup") is not False:
            raise ValueError("P1 cost statistics contain warmup observations")
        grouped[int(row.get("pass_index", -1))].append(row)
    if set(grouped) != set(range(4)):
        raise ValueError("P1 cost leaf does not contain four measured passes")

    manifests: list[list[tuple[str, str]]] = []
    normalized: list[dict[str, Any]] = []
    for pass_index, arm in enumerate(sequence):
        pass_rows = sorted(
            grouped[pass_index], key=lambda row: int(row.get("sample_ordinal", -1))
        )
        if len(pass_rows) != PHYSICAL_WINDOWS:
            raise ValueError("P1 cost pass is not the complete 136-window population")
        manifest: list[tuple[str, str]] = []
        expected_tokens = _p1_expected_physical_tokens(arm)
        for ordinal, row in enumerate(pass_rows):
            audit = row.get("route_audit")
            if (
                row.get("schema_version") != "zoomtoken_p1_cost_sample_v001"
                or row.get("leaf_id") != leaf_id
                or row.get("arm") != arm
                or int(row.get("pass_index", -1)) != pass_index
                or int(row.get("sample_ordinal", -1)) != ordinal
                or int(row.get("loader_ordinal", -1)) != ordinal
                or int(row.get("exact_window_budget", -1)) != WINDOW_BUDGET
                or int(row.get("selected_physical_tokens", -1)) != expected_tokens
                or int(row.get("executed_physical_tokens", -1)) != expected_tokens
                or int(row.get("duplicate_selected_physical_tokens", -1)) != 0
                or int(row.get("padded_heavy_tokens", -1)) != 0
                or not isinstance(row.get("physical_window_id"), str)
                or not row["physical_window_id"]
                or not isinstance(row.get("video_id"), str)
                or not row["video_id"]
                or not isinstance(audit, Mapping)
                or any(
                    not isinstance(row.get(field), list)
                    or len(row[field]) != 2
                    or any(not math.isfinite(float(value)) for value in row[field])
                    or float(row[field][1]) <= float(row[field][0])
                    for field in (
                        "energy_window_monotonic_s",
                        "nms_energy_window_monotonic_s",
                    )
                )
                or audit.get("arm") != arm
                or audit.get("uses_gt_for_route") is not False
                or audit.get("uses_teacher") is not False
                or audit.get("uses_oracle") is not False
                or audit.get("uses_test_evidence") is not False
                or int(audit.get("selected_physical_tokens", -1)) != expected_tokens
                or int(audit.get("executed_physical_tokens", -1)) != expected_tokens
                or int(audit.get("duplicate_selected_physical_tokens", -1)) != 0
                or int(audit.get("padded_heavy_tokens", -1)) != 0
                or any(
                    not math.isfinite(float(row.get(field, float("nan"))))
                    or float(row[field]) <= 0.0
                    for field in (
                        "input_pipeline_serial_ms",
                        "h2d_ms",
                        "decode_to_window_output_wall_ms",
                        "model_forward_ms",
                        "postprocess_ms",
                        "final_video_nms_ms",
                        "end_to_end_serial_ms",
                        "peak_gpu_allocated_mb",
                        "peak_gpu_reserved_mb",
                        "gross_gpu_energy_j_per_sample",
                    )
                )
            ):
                raise ValueError("P1 cost row violates execution or full-stack scope")
            if arm == "Q":
                clip_counts = audit.get("clip_token_counts")
                if (
                    audit.get("route_mode") != "dynamic_scnr"
                    or audit.get("target_k") is not None
                    or audit.get("dynamic_k_t") is not True
                    or int(audit.get("k_t_min", -1)) < 0
                    or int(audit.get("k_t_max", -1)) < int(audit.get("k_t_min", -1))
                    or int(audit.get("k_t_zero_count", -1)) < 0
                    or not isinstance(clip_counts, list)
                    or any(type(value) is not int or value < 0 for value in clip_counts)
                    or sum(clip_counts) != WINDOW_BUDGET
                    or int(audit.get("attention_pairs", -1))
                    != sum(value**2 for value in clip_counts)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(audit.get("physical_indices_sha256", ""))
                    )
                ):
                    raise ValueError("P1 Q cost row lost dynamic exact-B ragged routing")
            elif arm in {"U", "R"}:
                if (
                    audit.get("route_mode") != ("uniform" if arm == "U" else "random")
                    or int(audit.get("target_k", -1)) != 64
                    or audit.get("dynamic_k_t") is not False
                ):
                    raise ValueError("P1 matched sparse control changed during cost replay")
            elif (
                audit.get("route_mode") != "dense"
                or audit.get("target_k") is not None
                or audit.get("dynamic_k_t") is not False
            ):
                raise ValueError("P1 dense cost comparator changed execution")
            manifest.append((str(row["video_id"]), str(row["physical_window_id"])))
            normalized.append(row)
        if len(set(manifest)) != PHYSICAL_WINDOWS:
            raise ValueError("P1 cost pass contains duplicate physical windows")
        manifests.append(manifest)
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise ValueError("P1 cost pass population/order changed within a leaf")
    if len({video_id for video_id, _ in manifests[0]}) != VIDEO_CLUSTERS:
        raise ValueError("P1 cost leaf does not cover the frozen 40 video clusters")
    return normalized


def validate_p1_cost_warmup_rows(
    rows: Sequence[Mapping[str, Any]], *, leaf_id: str
) -> list[dict[str, Any]]:
    sequence = p1_cost_leaf_sequence(leaf_id)
    allowed_fields = {
        "schema_version",
        "leaf_id",
        "pass_index",
        "arm",
        "measurement_phase",
        "warmup",
        "warmup_ordinal",
        "loader_ordinal",
        "video_id",
        "physical_window_id",
        "window_id",
    }
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = copy.deepcopy(dict(raw))
        if set(row) != allowed_fields:
            raise ValueError("P1 cost warmup ledger contains non-identity evidence")
        grouped[int(row.get("pass_index", -1))].append(row)
    if set(grouped) != set(range(4)):
        raise ValueError("P1 cost warmup ledger does not cover four passes")
    manifests: list[list[tuple[str, str]]] = []
    normalized: list[dict[str, Any]] = []
    for pass_index, arm in enumerate(sequence):
        pass_rows = sorted(
            grouped[pass_index], key=lambda row: int(row.get("warmup_ordinal", -1))
        )
        if len(pass_rows) != WARMUP_WINDOWS_PER_PASS:
            raise ValueError("P1 cost warmup did not traverse all 136 windows")
        manifest: list[tuple[str, str]] = []
        for ordinal, row in enumerate(pass_rows):
            if (
                row.get("schema_version")
                != "zoomtoken_p1_cost_warmup_identity_v001"
                or row.get("leaf_id") != leaf_id
                or int(row.get("pass_index", -1)) != pass_index
                or row.get("arm") != arm
                or row.get("measurement_phase") != "warmup"
                or row.get("warmup") is not True
                or int(row.get("warmup_ordinal", -1)) != ordinal
                or int(row.get("loader_ordinal", -1)) != ordinal
                or not isinstance(row.get("video_id"), str)
                or not row["video_id"]
                or not isinstance(row.get("physical_window_id"), str)
                or not row["physical_window_id"]
                or not isinstance(row.get("window_id"), str)
                or not row["window_id"]
            ):
                raise ValueError("P1 cost warmup identity/order changed")
            manifest.append((str(row["video_id"]), str(row["physical_window_id"])))
            normalized.append(row)
        if len(set(manifest)) != PHYSICAL_WINDOWS:
            raise ValueError("P1 cost warmup contains duplicate physical windows")
        manifests.append(manifest)
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise ValueError("P1 cost warmup population/order changed between passes")
    if len({video_id for video_id, _ in manifests[0]}) != VIDEO_CLUSTERS:
        raise ValueError("P1 cost warmup lacks the frozen 40 video clusters")
    return normalized


def validate_p1_cost_leaf_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_leaf_id: str,
    expected_job_id: str | None = None,
) -> dict[str, Any]:
    checked = copy.deepcopy(dict(receipt))
    require_self_hash(checked, field="receipt_sha256", label="P1 cost leaf")
    spec = p1_cost_leaf_spec(expected_leaf_id)
    artifacts = checked.get("artifacts")
    pass_receipts = checked.get("pass_receipts")
    if (
        checked.get("schema_version") != "zoomtoken_p1_cost_leaf_v001"
        or checked.get("study_id") != P1_STUDY_ID
        or checked.get("status") != "COMPLETE_P1_COST_LEAF"
        or checked.get("leaf_id") != expected_leaf_id
        or checked.get("comparator") != spec["comparator"]
        or checked.get("order") != spec["order"]
        or tuple(checked.get("sequence", ())) != p1_cost_leaf_sequence(expected_leaf_id)
        or not _SHA_RE.fullmatch(str(checked.get("runtime_commit", "")))
        or int(checked.get("seed", -1)) != TRAINING_SEED
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(checked.get("population_sha256", ""))
        )
        or int(checked.get("warmup_windows_before_each_pass", -1))
        != WARMUP_WINDOWS_PER_PASS
        or int(checked.get("measured_windows_per_pass", -1))
        != MEASURED_WINDOWS_PER_PASS
        or int(checked.get("measured_pass_count", -1)) != 4
        or int(checked.get("measured_rows", -1)) != 4 * PHYSICAL_WINDOWS
        or checked.get("training_or_resume_executed") is not False
        or checked.get("metric_evaluation_executed") is not False
        or checked.get("held_out_test_opened") is not False
        or checked.get("authoritative_decision") is not False
        or not isinstance(artifacts, Mapping)
        or set(artifacts)
        != {"measured_samples", "warmup_identities", "power_trace", "sidecar_report"}
        or not isinstance(pass_receipts, list)
        or len(pass_receipts) != 4
        or not isinstance(checked.get("runtime_attestation"), Mapping)
    ):
        raise ValueError("P1 cost leaf receipt is invalid")
    job_id = str(checked.get("slurm_job_id", ""))
    if not job_id.isdigit() or (
        expected_job_id is not None and job_id != str(expected_job_id)
    ):
        raise ValueError("P1 cost leaf Slurm identity changed")
    for name in ("measured_samples", "warmup_identities", "power_trace", "sidecar_report"):
        artifact = artifacts.get(name)
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(artifact.get("path"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", "")))
        ):
            raise ValueError(f"P1 cost leaf lacks the {name} artifact receipt")
    for pass_index, (pass_receipt, arm) in enumerate(
        zip(pass_receipts, p1_cost_leaf_sequence(expected_leaf_id))
    ):
        if not isinstance(pass_receipt, Mapping):
            raise ValueError("P1 cost pass receipt is malformed")
        unsigned = dict(pass_receipt)
        observed_hash = unsigned.pop("pass_sha256", None)
        if (
            observed_hash != canonical_sha256(unsigned)
            or int(pass_receipt.get("pass_index", -1)) != pass_index
            or pass_receipt.get("arm") != arm
            or int(pass_receipt.get("sample_count", -1)) != PHYSICAL_WINDOWS
            or pass_receipt.get("population_sha256")
            != checked.get("population_sha256")
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(pass_receipt.get("checkpoint_sha256", ""))
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(pass_receipt.get("config_sha256", ""))
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(pass_receipt.get("sample_manifest_sha256", "")),
            )
            or pass_receipt.get("diagnostic_telemetry_inside_timed_forward")
            is not False
            or pass_receipt.get("training_or_resume_executed") is not False
        ):
            raise ValueError("P1 cost pass receipt changed")
    return checked


def _p1_ratio(
    denominator_rows: Sequence[Mapping[str, Any]],
    numerator_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    reducer: str,
) -> float:
    return _metric([float(row[field]) for row in numerator_rows], reducer) / _metric(
        [float(row[field]) for row in denominator_rows], reducer
    )


def analyze_p1_cost_leaves(
    leaves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_replicates: int = P1_COST_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Paired 40-video bootstrap; only Q/DN controls the P1 cost gate."""

    if set(leaves) != set(P1_COST_LEAF_SPECS) or int(bootstrap_replicates) <= 0:
        raise ValueError("P1 cost analysis requires all eight frozen leaves")
    normalized = {
        leaf_id: validate_p1_cost_rows(rows, leaf_id=leaf_id)
        for leaf_id, rows in leaves.items()
    }
    canonical_manifest: list[tuple[str, str]] | None = None
    pass_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for leaf_id in P1_COST_LEAF_SPECS:
        for pass_index in range(4):
            rows = [
                row
                for row in normalized[leaf_id]
                if int(row["pass_index"]) == pass_index
            ]
            manifest = [
                (str(row["video_id"]), str(row["physical_window_id"])) for row in rows
            ]
            canonical_manifest = manifest if canonical_manifest is None else canonical_manifest
            if manifest != canonical_manifest:
                raise ValueError("P1 cost leaves changed the canonical population/order")
            pass_rows[(leaf_id, pass_index)] = rows
    if canonical_manifest is None:
        raise ValueError("P1 cost population is missing")
    videos = tuple(sorted({video_id for video_id, _ in canonical_manifest}))
    if len(videos) != VIDEO_CLUSTERS:
        raise ValueError("P1 cost bootstrap requires 40 video clusters")

    by_video: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    for key, rows in pass_rows.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row["video_id"])].append(row)
        if tuple(sorted(groups)) != videos:
            raise ValueError("P1 cost pass changed video clusters")
        by_video[key] = groups

    comparisons: dict[str, Any] = {}
    for comparator in P1_COST_COMPARATORS:
        rng = np.random.Generator(np.random.PCG64(P1_COST_BOOTSTRAP_SEED))
        pair_units: list[tuple[str, int, int]] = []
        pair_units_by_order: dict[str, list[tuple[str, int, int]]] = {}
        point_denominator: list[dict[str, Any]] = []
        point_q: list[dict[str, Any]] = []
        for order in ("ABBA", "BAAB"):
            leaf_id = f"{comparator}_{order}"
            order_units: list[tuple[str, int, int]] = []
            for denominator_index, q_index in PAIR_INDICES_BY_ORDER[order]:
                unit = (leaf_id, denominator_index, q_index)
                order_units.append(unit)
                pair_units.append(unit)
                point_denominator.extend(pass_rows[(leaf_id, denominator_index)])
                point_q.extend(pass_rows[(leaf_id, q_index)])
            pair_units_by_order[order] = order_units
        draws = {
            name: np.empty(int(bootstrap_replicates), dtype=np.float64)
            for name in P1_COST_METRICS
        }
        for replicate in range(int(bootstrap_replicates)):
            sampled_denominator: list[dict[str, Any]] = []
            sampled_q: list[dict[str, Any]] = []
            for order in ("ABBA", "BAAB"):
                order_units = pair_units_by_order[order]
                pair_draw = rng.integers(
                    0, len(order_units), size=len(order_units)
                )
                for unit_index in pair_draw:
                    leaf_id, denominator_index, q_index = order_units[int(unit_index)]
                    video_draw = rng.integers(
                        0, len(videos), size=VIDEO_CLUSTERS
                    )
                    for video_index in video_draw:
                        video_id = videos[int(video_index)]
                        sampled_denominator.extend(
                            by_video[(leaf_id, denominator_index)][video_id]
                        )
                        sampled_q.extend(by_video[(leaf_id, q_index)][video_id])
            for name, (field, reducer) in P1_COST_METRICS.items():
                draws[name][replicate] = _p1_ratio(
                    sampled_denominator,
                    sampled_q,
                    field=field,
                    reducer=reducer,
                )
        metrics: dict[str, Any] = {}
        for name, (field, reducer) in P1_COST_METRICS.items():
            ratio = _p1_ratio(
                point_denominator, point_q, field=field, reducer=reducer
            )
            upper = float(np.quantile(draws[name], 0.95, method="linear"))
            metrics[name] = {
                "field": field,
                "reducer": reducer,
                "q_over_comparator_ratio": ratio,
                "one_sided_95_upper_bound": upper,
                "upper_bound_le_0_85": upper <= P1_COST_RATIO_LIMIT,
                "literal_limit": P1_COST_RATIO_LIMIT,
                "tolerance": 0.0,
            }
        controlling = comparator == "DN"
        passed = all(metric["upper_bound_le_0_85"] for metric in metrics.values())
        comparisons[comparator] = {
            "comparator": comparator,
            "controlling": controlling,
            "report_only": not controlling,
            "paired_pass_units": len(pair_units),
            "metrics": metrics,
            "cost_gate_passed": passed if controlling else None,
        }
    return {
        "schema_version": "zoomtoken_p1_cost_analysis_v001",
        "study_id": P1_STUDY_ID,
        "leaf_specs": copy.deepcopy(P1_COST_LEAF_SPECS),
        "video_cluster_count": len(videos),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": P1_COST_BOOTSTRAP_SEED,
        "bootstrap_rng": "numpy.random.Generator(PCG64)",
        "bootstrap_quantile": "one_sided_95_numpy.quantile(method=linear)",
        "bootstrap_hierarchy": (
            "ABBA_and_BAAB_strata_then_two_mirrored_pairs_per_leaf_"
            "then_40_paired_video_clusters"
        ),
        "ratio_limit": P1_COST_RATIO_LIMIT,
        "dense_denominator": "DN",
        "comparisons": comparisons,
        "q_over_dn_cost_gate_passed": comparisons["DN"]["cost_gate_passed"],
        "do_is_mandatory_report_only": True,
    }


def validate_scheduler_job(
    row: Mapping[str, Any], *, expected_job_id: str, label: str
) -> None:
    job_id = str(row.get("job_id", ""))
    if (
        job_id != str(expected_job_id)
        or not job_id.isdigit()
        or row.get("state") != "COMPLETED"
        or row.get("exit_code") != "0:0"
    ):
        raise ValueError(f"{label} is not scheduler-confirmed COMPLETED 0:0")


def validate_afterany_dependency(
    row: Mapping[str, Any],
    *,
    expected_job_id: str,
    expected_parent_job_ids: Sequence[str],
) -> None:
    job_id = str(row.get("job_id", ""))
    expected_parents = tuple(str(value) for value in expected_parent_job_ids)
    dependency = str(row.get("dependency", ""))
    match = re.fullmatch(r"afterany:(\d+(?::\d+)*)", dependency)
    observed_parents = tuple(match.group(1).split(":")) if match else ()
    if (
        job_id != str(expected_job_id)
        or not job_id.isdigit()
        or len(expected_parents) != 8
        or len(set(expected_parents)) != 8
        or any(not value.isdigit() for value in expected_parents)
        or observed_parents != expected_parents
    ):
        raise ValueError("steady-cost Job 9 lacks the exact native afterany dependency")


def validate_frozen_contract() -> None:
    if (
        list(LEAF_ORDERS) != [f"L{index:02d}" for index in range(1, 9)]
        or list(LEAF_ORDERS.values()) != ["ABBA"] * 4 + ["BAAB"] * 4
        or PAIR_INDICES_BY_ORDER
        != {"ABBA": ((0, 1), (3, 2)), "BAAB": ((1, 0), (2, 3))}
        or WARMUP_WINDOWS_PER_PASS != PHYSICAL_WINDOWS
        or MEASURED_WINDOWS_PER_PASS != PHYSICAL_WINDOWS
        or BOOTSTRAP_REPLICATES != 10_000
        or BOOTSTRAP_SEED != 20_260_811
        or NONINFERIORITY_RATIO != 1.05
        or FINALIZER_DEPENDENCY != "afterany"
        or tuple(P1_COST_LEAF_SPECS)
        != tuple(
            f"{comparator}_{order}"
            for comparator in P1_COST_COMPARATORS
            for order in ("ABBA", "BAAB")
        )
        or P1_COST_RATIO_LIMIT != 0.85
        or P1_DENSE_PHYSICAL_TOKENS != 84_480
    ):
        raise RuntimeError("ZoomToken steady-cost frozen constants changed")


validate_frozen_contract()


def _contract_gate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-run", type=Path, required=True)
    parser.add_argument("--pytest-exit-code", type=int, required=True)
    return parser.parse_args()


def contract_gate_main() -> int:
    """Record R0 only after CPU-only Slurm tests and all static bindings pass."""

    from tools.bata.georoute_dynamic_floor_m2_contract import (
        require_clean_dynamic_floor_m2_checkout,
    )
    from tools.bata.georoute_residual_centering_cost_contract import (
        validate_residual_centering_cost_source,
    )

    args = _contract_gate_args()
    if (
        not os.environ.get("SLURM_JOB_ID", "").isdigit()
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("SLURM_JOB_GPUS")
        or args.pytest_exit_code != 0
    ):
        raise RuntimeError("steady-cost R0 must be successful CPU-only Slurm work")
    pre_run = validate_pre_run(
        read_json_object(args.pre_run, label="steady-cost R0 PRE_RUN"),
        phase="contract_gate",
    )
    root = ROOT
    require_clean_dynamic_floor_m2_checkout(
        expected_commit=pre_run["repair_sha"], root=root
    )
    configs = {
        "control": validate_tracked_config(root, "A"),
        "centered": validate_tracked_config(root, "B"),
    }
    validate_paired_configs(root)
    if configs != pre_run["configs"]:
        raise ValueError("steady-cost R0 config receipts differ from PRE_RUN")
    source = validate_residual_centering_cost_source(
        pre_run["training_run_root"],
        expected_model_runtime_commit=pre_run["model_runtime_sha"],
    )
    for name, variant in (
        ("control", "none_control"),
        ("centered", "residual_window_center"),
    ):
        checkpoint = source["stages"][variant]["checkpoint_receipt"]
        if (
            pre_run["checkpoints"][name].get("path") != checkpoint.get("path")
            or pre_run["checkpoints"][name].get("sha256") != checkpoint.get("sha256")
        ):
            raise ValueError(f"steady-cost R0 {name} checkpoint differs from PRE_RUN")
    manifest_path = root / POPULATION_MANIFEST
    manifest = validate_population_manifest(
        read_json_object(manifest_path, label="steady-cost population")
    )
    signature = population_signature(manifest)
    if (
        pre_run["population"].get("manifest_file_sha256") != sha256_file(manifest_path)
        or pre_run["population"].get("manifest_sha256") != signature["manifest_sha256"]
        or pre_run["population"].get("physical_window_ids_sha256")
        != signature["physical_window_ids_sha256"]
    ):
        raise ValueError("steady-cost R0 population receipt differs from PRE_RUN")
    prereg_path = root / "research-wiki/experiments/zoomtoken_scnr_steady_cost_prereg_v001.json"
    jobgraph_path = root / "research-wiki/experiments/zoomtoken_scnr_steady_cost_jobgraph_v001.json"
    prereg = validate_preregistration(
        read_json_object(prereg_path, label="steady-cost preregistration")
    )
    jobgraph = validate_jobgraph(
        read_json_object(jobgraph_path, label="steady-cost jobgraph")
    )
    output = (
        Path(pre_run["results_root"])
        / STUDY_ID
        / pre_run["repair_sha"]
        / "contract_gate"
        / os.environ["SLURM_JOB_ID"]
    )
    if output.exists():
        raise FileExistsError("steady-cost R0 output path already exists")
    output.mkdir(parents=True, exist_ok=False)
    receipt = add_self_hash(
        {
            "schema_version": "zoomtoken_scnr_steady_cost_contract_gate_v001",
            "study_id": STUDY_ID,
            "status": "CONTRACT_GATE_READY",
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "pre_run_sha256": pre_run["pre_run_sha256"],
            "repair_sha": pre_run["repair_sha"],
            "pytest_exit_code": args.pytest_exit_code,
            "configs": configs,
            "population": signature,
            "prereg_sha256": prereg["prereg_sha256"],
            "jobgraph_sha256": jobgraph["jobgraph_sha256"],
            "model_runtime_sha": pre_run["model_runtime_sha"],
            "checkpoints": pre_run["checkpoints"],
            "gpu_used": False,
            "model_or_data_executed": False,
            "result_bearing_data_created": False,
            "held_out_test_opened": False,
            "metric_evaluation_executed": False,
        },
        field="receipt_sha256",
    )
    atomic_write_json(output / "receipt.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(contract_gate_main())
