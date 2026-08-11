#!/usr/bin/env python3
"""Run one frozen ZoomToken steady-cost leaf or its two-arm preflight."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.bata.georoute_residual_centering_cost_contract as legacy_contract  # noqa: E402
import tools.bata.profile_georoute_residual_centering_cost as legacy_profile  # noqa: E402
from opentad.datasets import build_dataset  # noqa: E402
from tools.bata.georoute_dynamic_floor_m2_contract import (  # noqa: E402
    require_clean_dynamic_floor_m2_checkout,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file  # noqa: E402
from tools.bata.georoute_residual_centering_cost_contract import (  # noqa: E402
    validate_residual_centering_cost_source,
)
from tools.bata.profile_georoute_dynamic_floor_m2 import (  # noqa: E402
    _cpu_ids,
    _population_descriptor,
    _write_jsonl,
)
from tools.bata.profile_georoute_residual_centering_cost import (  # noqa: E402
    _hardware_identity,
    _profile_one_pass,
    _software_identity,
    integrate_energy,
)
from tools.bata.spatial_zoom_s1_power import NvmlSidecarPowerSampler  # noqa: E402
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    require_slurm_memory_limit_mb,
    require_slurm_single_gpu_allocation,
)
from tools.bata.zoomtoken_scnr_steady_cost_contract_v001 import (  # noqa: E402
    ARM_TO_VARIANT,
    PHYSICAL_WINDOWS,
    POWER_INTERVAL_MS,
    STUDY_ID,
    WARMUP_WINDOWS_PER_PASS,
    WINDOW_BUDGET,
    add_self_hash,
    atomic_write_json,
    build_execution_binding,
    build_runtime_identity,
    leaf_sequence,
    population_signature,
    read_json_object,
    require_self_hash,
    validate_leaf_rows,
    validate_pass_receipts,
    validate_population_manifest,
    validate_pre_run,
    validate_tracked_config,
    validate_warmup_ledger,
)


BOUNDARY = Path("/data/run01/sczc063/yuzibo")
_LEGACY_VALIDATE_BRANCH_AUDIT = legacy_profile._validate_branch_audit
_LEGACY_MOVE_TO_DEVICE = legacy_profile._move_to_device
_ACTIVE_WARMUP: dict[str, Any] | None = None


def _move_to_device_with_warmup_identity(value: Any, device: Any) -> Any:
    if _ACTIVE_WARMUP is not None and len(_ACTIVE_WARMUP["rows"]) < WARMUP_WINDOWS_PER_PASS:
        ordinal = len(_ACTIVE_WARMUP["rows"])
        identity = legacy_profile._sample_identity(value, ordinal)
        _ACTIVE_WARMUP["rows"].append(
            {
                "schema_version": "zoomtoken_scnr_steady_cost_warmup_identity_v001",
                "leaf_id": _ACTIVE_WARMUP["leaf_id"],
                "pass_index": _ACTIVE_WARMUP["pass_index"],
                "arm": _ACTIVE_WARMUP["arm"],
                "measurement_phase": "warmup",
                "warmup": True,
                "warmup_ordinal": ordinal,
                **identity,
            }
        )
    return _LEGACY_MOVE_TO_DEVICE(value, device)


def _validate_branch_audit_with_diagnostics(
    audit: Mapping[str, Any], *, variant: str
) -> dict[str, Any]:
    summary = _LEGACY_VALIDATE_BRANCH_AUDIT(audit, variant=variant)
    packed = audit.get("packed")
    clip_rows = packed.get("clip_token_counts") if isinstance(packed, Mapping) else None
    if (
        not isinstance(clip_rows, list)
        or len(clip_rows) != 1
        or not isinstance(clip_rows[0], list)
    ):
        raise RuntimeError("steady-cost route audit lacks per-clip token counts")
    clip_counts = [int(value) for value in clip_rows[0]]
    if any(value < 0 for value in clip_counts) or sum(clip_counts) != WINDOW_BUDGET:
        raise RuntimeError("steady-cost per-clip token counts violate the exact budget")
    summary["clip_token_counts"] = clip_counts
    return summary


def _patch_legacy_pass_contract(sequence: Sequence[str]) -> None:
    """Narrow the proven one-pass primitive to this process's frozen traversal."""

    sequence = tuple(sequence)
    control = [index for index, arm in enumerate(sequence) if arm == ARM_TO_VARIANT["A"]]
    centered = [index for index, arm in enumerate(sequence) if arm == ARM_TO_VARIANT["B"]]
    if len(control) != len(centered) or not control:
        raise ValueError("steady-cost traversal is not treatment-balanced")
    pairs = tuple(zip(control, centered))
    legacy_contract.RESIDUAL_CENTERING_COST_ORDER = sequence
    legacy_contract.RESIDUAL_CENTERING_COST_PAIRS = pairs
    legacy_contract.RESIDUAL_CENTERING_COST_WARMUP_SAMPLES = WARMUP_WINDOWS_PER_PASS
    legacy_profile.RESIDUAL_CENTERING_COST_ORDER = sequence
    legacy_profile.RESIDUAL_CENTERING_COST_WARMUP_SAMPLES = WARMUP_WINDOWS_PER_PASS
    legacy_profile._validate_branch_audit = _validate_branch_audit_with_diagnostics
    legacy_profile._move_to_device = _move_to_device_with_warmup_identity


def _profile_pass_with_warmup(
    *,
    torch: Any,
    variant: str,
    pass_index: int,
    stage: Mapping[str, Any],
    expected_population_sha256: str,
    expected_accuracy_population_sha256: str,
    device: Any,
    leaf_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    global _ACTIVE_WARMUP

    if _ACTIVE_WARMUP is not None:
        raise RuntimeError("steady-cost warmup recorder is already active")
    context: dict[str, Any] = {
        "leaf_id": leaf_id,
        "pass_index": pass_index,
        "arm": variant,
        "rows": [],
    }
    _ACTIVE_WARMUP = context
    try:
        rows, receipt = _profile_one_pass(
            torch=torch,
            variant=variant,
            pass_index=pass_index,
            stage=stage,
            expected_population_sha256=expected_population_sha256,
            expected_accuracy_population_sha256=expected_accuracy_population_sha256,
            device=device,
        )
    finally:
        _ACTIVE_WARMUP = None
    return rows, receipt, list(context["rows"])


def _verify_static_bindings(pre_run: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for name, gate in pre_run.get("gates", {}).items():
        path = Path(str(gate["path"])).resolve()
        if not path.is_file() or gate["sha256"] != sha256_file(path):
            raise ValueError(f"steady-cost {name} gate file receipt is invalid")
        payload = read_json_object(path, label=f"steady-cost {name} gate")
        require_self_hash(payload, field="receipt_sha256", label=f"steady-cost {name} gate")
        if (
            payload.get("status") != gate["status"]
            or payload.get("pre_run_sha256") != gate["pre_run_sha256"]
            or payload.get("repair_sha") != pre_run["repair_sha"]
            or payload.get("held_out_test_opened") is True
            or payload.get("metric_evaluation_executed") is True
        ):
            raise ValueError(f"steady-cost {name} gate payload is invalid")
    for arm, name in (("A", "control"), ("B", "centered")):
        observed = validate_tracked_config(ROOT, arm)
        if observed != dict(pre_run["configs"][name]):
            raise ValueError(f"steady-cost {name} config changed after PRE_RUN")
    manifest_path = (ROOT / pre_run["population"]["manifest_path"]).resolve()
    manifest = validate_population_manifest(
        read_json_object(manifest_path, label="steady-cost population manifest")
    )
    signature = population_signature(manifest)
    if (
        pre_run["population"].get("manifest_file_sha256") != sha256_file(manifest_path)
        or pre_run["population"].get("manifest_sha256") != signature["manifest_sha256"]
        or pre_run["population"].get("physical_window_ids_sha256")
        != signature["physical_window_ids_sha256"]
    ):
        raise ValueError("steady-cost population receipt changed after PRE_RUN")
    source = validate_residual_centering_cost_source(
        pre_run["training_run_root"],
        expected_model_runtime_commit=pre_run["model_runtime_sha"],
    )
    for name, variant in (("control", "none_control"), ("centered", "residual_window_center")):
        stage_receipt = source["stages"][variant]["checkpoint_receipt"]
        if (
            pre_run["checkpoints"][name].get("path") != stage_receipt.get("path")
            or pre_run["checkpoints"][name].get("sha256") != stage_receipt.get("sha256")
        ):
            raise ValueError(f"steady-cost {name} checkpoint changed after PRE_RUN")
    return manifest, source


def _runtime_setup(torch: Any, args: argparse.Namespace) -> tuple[Any, dict[str, Any], dict[str, Any], tuple[int, ...], tuple[int, ...], int]:
    import torch.distributed as dist

    if (
        not os.environ.get("SLURM_JOB_ID", "").isdigit()
        or int(os.environ.get("WORLD_SIZE", -1)) != 1
        or int(os.environ.get("RANK", -1)) != 0
        or int(os.environ.get("LOCAL_RANK", -1)) != 0
        or not torch.cuda.is_available()
        or dist.is_initialized()
    ):
        raise RuntimeError("steady-cost GPU work requires fresh Slurm world1 cuda:0")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Slurm did not provide CUDA_VISIBLE_DEVICES")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    allocated = _cpu_ids(args.allocated_cpus)
    detector = _cpu_ids(args.detector_cpus)
    sidecar_cpu = int(args.sidecar_cpu)
    if (
        len(allocated) != 5
        or len(detector) != 4
        or set(detector) | {sidecar_cpu} != set(allocated)
        or sidecar_cpu in detector
        or tuple(sorted(os.sched_getaffinity(0))) != detector
        or int(os.environ.get("SLURM_CPUS_PER_TASK", -1)) != 5
    ):
        raise RuntimeError("steady-cost detector/sidecar CPU partition changed")
    physical_gpu = require_slurm_single_gpu_allocation()
    memory_limit_mb = require_slurm_memory_limit_mb(minimum_mb=1)
    software = _software_identity(torch)
    hardware = _hardware_identity(
        torch,
        device,
        physical_gpu_id=physical_gpu,
        allocated_cpu_ids=allocated,
        detector_cpu_ids=detector,
        sidecar_cpu_id=sidecar_cpu,
        memory_limit_mb=memory_limit_mb,
    )
    return device, hardware, software, allocated, detector, sidecar_cpu


def execute_sequence(
    args: argparse.Namespace,
    *,
    phase: str,
    sequence: Sequence[str],
    output_root: Path,
    leaf_id: str | None,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    pre_run = validate_pre_run(
        read_json_object(args.pre_run, label="steady-cost PRE_RUN"), phase=phase
    )
    require_clean_dynamic_floor_m2_checkout(
        expected_commit=pre_run["repair_sha"], root=ROOT
    )
    manifest, source = _verify_static_bindings(pre_run)
    expected_ids = [row["physical_window_id"] for row in manifest["windows"]]
    _patch_legacy_pass_contract(sequence)
    if output_root.exists():
        raise FileExistsError("steady-cost output path already exists")
    output_root.mkdir(parents=True, exist_ok=False)

    device, hardware, software, allocated, detector, sidecar_cpu = _runtime_setup(
        torch, args
    )
    runtime_identity = build_runtime_identity(
        pre_run,
        hardware=hardware,
        software=software,
        slurm_job_constraints=os.environ.get("SLURM_JOB_CONSTRAINTS", ""),
    )
    try:
        physical_hashes = set()
        for variant in set(sequence):
            cfg = legacy_contract.build_residual_centering_cost_config(
                source["stages"][variant], variant=variant
            )
            dataset = build_dataset(copy.deepcopy(cfg.dataset.test))
            descriptors, physical_hash, telemetry_hash = _population_descriptor(dataset)
            physical_ids = [
                f"{row['video_id']}:{int(row['window_center_first'])}"
                for row in descriptors
            ]
            if (
                physical_ids != expected_ids
                or telemetry_hash != pre_run["population"]["source_population_sha256"]
            ):
                raise ValueError("steady-cost runtime population differs from manifest")
            physical_hashes.add(physical_hash)
        if len(physical_hashes) != 1:
            raise ValueError("steady-cost arms changed physical population")
        physical_hash = physical_hashes.pop()

        scratch_root = args.power_scratch_root.resolve()
        if not (
            str(scratch_root).startswith("/tmp/")
            or str(scratch_root).startswith("/var/tmp/")
        ):
            raise ValueError("steady-cost power scratch must be node-local")
        sampler = NvmlSidecarPowerSampler(
            expected_uuid=hardware["nvidia_smi"]["uuid"],
            interval_ms=POWER_INTERVAL_MS,
            scratch_dir=(
                scratch_root
                / f"job{os.environ['SLURM_JOB_ID']}_zoomtoken_steady_cost"
            ),
            attempt_prefix=output_root / "power_sidecar",
            sidecar_cpu_id=sidecar_cpu,
            detector_cpu_ids=detector,
            allocated_cpu_ids=allocated,
        )
        all_rows = []
        pass_receipts = []
        warmup_rows = []
        sampler.start()
        time.sleep(sampler.interval_s * 1.5)
        try:
            for pass_index, variant in enumerate(sequence):
                arm = "A" if variant == ARM_TO_VARIANT["A"] else "B"
                binding_before = build_execution_binding(
                    ROOT, arm=arm, stage=source["stages"][variant]
                )
                rows, receipt, pass_warmup = _profile_pass_with_warmup(
                    torch=torch,
                    variant=variant,
                    pass_index=pass_index,
                    stage=source["stages"][variant],
                    expected_population_sha256=physical_hash,
                    expected_accuracy_population_sha256=(
                        pre_run["population"]["source_population_sha256"]
                    ),
                    device=device,
                    leaf_id=leaf_id or "PREFLIGHT",
                )
                binding_after = build_execution_binding(
                    ROOT, arm=arm, stage=source["stages"][variant]
                )
                if binding_before != binding_after:
                    raise ValueError("steady-cost execution identity changed during a pass")
                unsigned_receipt = dict(receipt)
                unsigned_receipt.pop("pass_sha256", None)
                unsigned_receipt["execution_binding_before"] = binding_before
                unsigned_receipt["execution_binding_after"] = binding_after
                unsigned_receipt["pass_sha256"] = canonical_sha256(unsigned_receipt)
                receipt = unsigned_receipt
                for row in rows:
                    audit = row["route_audit"]
                    row.update(
                        {
                            "leaf_id": leaf_id or "PREFLIGHT",
                            "measurement_phase": "measured",
                            "warmup": False,
                            "exact_window_budget": WINDOW_BUDGET,
                            "selected_physical_tokens": WINDOW_BUDGET,
                            "executed_physical_tokens": WINDOW_BUDGET,
                            "duplicate_selected_physical_tokens": 0,
                            "padded_heavy_tokens": int(audit["padded_heavy_tokens"]),
                        }
                    )
                all_rows.extend(rows)
                pass_receipts.append(receipt)
                warmup_rows.extend(pass_warmup)
        finally:
            time.sleep(sampler.interval_s * 1.5)
            sampler.stop()

        pass_counts = {
            index: sum(int(row["pass_index"]) == index for row in all_rows)
            for index in range(len(sequence))
        }
        for row in all_rows:
            start, end = map(float, row["energy_window_monotonic_s"])
            nms_start, nms_end = map(float, row["nms_energy_window_monotonic_s"])
            sample_energy = integrate_energy(sampler.samples, start=start, end=end)
            nms_energy = integrate_energy(sampler.samples, start=nms_start, end=nms_end)
            if sample_energy is None or nms_energy is None:
                raise RuntimeError("steady-cost power trace has incomplete coverage")
            gross_energy = sample_energy + nms_energy / pass_counts[int(row["pass_index"])]
            row["gpu_energy_j"] = gross_energy
            row["gross_gpu_energy_j_per_sample"] = gross_energy
            row["sample_sha256"] = canonical_sha256(row)

        if leaf_id is not None:
            validate_leaf_rows(all_rows, leaf_id=leaf_id)
        else:
            if len(sequence) != 2 or any(count != PHYSICAL_WINDOWS for count in pass_counts.values()):
                raise ValueError("steady-cost preflight did not complete both full traversals")
        validate_warmup_ledger(
            warmup_rows,
            leaf_id=leaf_id or "PREFLIGHT",
            sequence=sequence,
            population=manifest,
        )
        validate_pass_receipts(
            ROOT,
            pass_receipts,
            sequence=sequence,
            source=source,
            expected_accuracy_population_sha256=pre_run["population"][
                "source_population_sha256"
            ],
            measured_rows=all_rows,
        )
        samples_path = output_root / "measured_samples.jsonl"
        power_path = output_root / "power_trace.jsonl"
        warmup_path = output_root / "warmup_identities.jsonl"
        _write_jsonl(samples_path, all_rows)
        _write_jsonl(warmup_path, warmup_rows)
        power_origin = sampler.samples[0][0]
        _write_jsonl(
            power_path,
            [
                {
                    "sequence": index,
                    "monotonic_s": timestamp,
                    "timestamp_ms": (timestamp - power_origin) * 1000.0,
                    "power_w": power,
                }
                for index, (timestamp, power) in enumerate(sampler.samples)
            ],
        )
        receipt = add_self_hash(
            {
                "schema_version": (
                    "zoomtoken_scnr_steady_cost_leaf_v001"
                    if leaf_id is not None
                    else "zoomtoken_scnr_steady_cost_preflight_v001"
                ),
                "status": "COMPLETE_LEAF_CANDIDATE" if leaf_id else "MECHANICAL_READY",
                "study_id": STUDY_ID,
                "phase": phase,
                "leaf_id": leaf_id,
                "order": "".join(
                    "A" if arm == ARM_TO_VARIANT["A"] else "B" for arm in sequence
                ),
                "pre_run_sha256": pre_run["pre_run_sha256"],
                "repair_sha": pre_run["repair_sha"],
                "model_runtime_sha": pre_run["model_runtime_sha"],
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "warmup_windows_before_each_pass": WARMUP_WINDOWS_PER_PASS,
                "warmup_outputs_persisted": False,
                "warmup_identity_ledger_persisted": True,
                "measured_pass_count": len(sequence),
                "measured_rows": len(all_rows),
                "population": population_signature(manifest),
                "physical_descriptor_sha256": physical_hash,
                "pass_receipts": pass_receipts,
                "hardware_identity": hardware,
                "hardware_fingerprint": canonical_sha256(hardware),
                "software_identity": software,
                "software_fingerprint": canonical_sha256(software),
                "runtime_identity": runtime_identity,
                "artifacts": {
                    "measured_samples": {
                        "path": str(samples_path.resolve()),
                        "sha256": sha256_file(samples_path),
                    },
                    "power_trace": {
                        "path": str(power_path.resolve()),
                        "sha256": sha256_file(power_path),
                    },
                    "warmup_identities": {
                        "path": str(warmup_path.resolve()),
                        "row_count": len(warmup_rows),
                    },
                    "sidecar_report": {
                        "path": str(sampler.attempt_report_path.resolve()),
                        "sha256": sha256_file(sampler.attempt_report_path),
                    },
                    "sidecar_trace": {
                        "path": str(sampler.attempt_trace_path.resolve()),
                        "sha256": sha256_file(sampler.attempt_trace_path),
                    },
                },
                "training_or_resume_executed": False,
                "metric_evaluation_executed": False,
                "held_out_test_opened": False,
                "authoritative_decision": False,
            },
            field="receipt_sha256",
        )
        atomic_write_json(output_root / "receipt.json", receipt)
        return receipt
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-run", type=Path, required=True)
    parser.add_argument("--leaf-id")
    parser.add_argument("--declared-order")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--allocated-cpus", required=True)
    parser.add_argument("--detector-cpus", required=True)
    parser.add_argument("--sidecar-cpu", required=True, type=int)
    parser.add_argument("--power-scratch-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pre_run = read_json_object(args.pre_run, label="steady-cost PRE_RUN")
    if args.preflight:
        if args.leaf_id is not None or args.declared_order is not None:
            raise ValueError("preflight accepts no treatment/order override")
        sequence = (ARM_TO_VARIANT["A"], ARM_TO_VARIANT["B"])
        output_root = (
            Path(pre_run["results_root"])
            / STUDY_ID
            / pre_run["repair_sha"]
            / "preflight"
            / os.environ.get("SLURM_JOB_ID", "MISSING")
        )
        execute_sequence(
            args,
            phase="preflight",
            sequence=sequence,
            output_root=output_root,
            leaf_id=None,
        )
    else:
        sequence = leaf_sequence(args.leaf_id, args.declared_order)
        output_root = (
            Path(pre_run["results_root"])
            / STUDY_ID
            / pre_run["repair_sha"]
            / "leaves"
            / args.leaf_id
        )
        execute_sequence(
            args,
            phase="full",
            sequence=sequence,
            output_root=output_root,
            leaf_id=args.leaf_id,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
