#!/usr/bin/env python3
"""Job-9 aggregate-candidate writer for the eight steady-cost leaves."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import sha256_file  # noqa: E402
from tools.bata.georoute_residual_centering_cost_contract import (  # noqa: E402
    validate_residual_centering_cost_source,
)
from tools.bata.zoomtoken_scnr_steady_cost_contract_v001 import (  # noqa: E402
    LEAF_ORDERS,
    STUDY_ID,
    add_self_hash,
    analyze_complete_leaves_with_draws,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    leaf_sequence,
    read_json_object,
    read_jsonl_objects,
    require_self_hash,
    validate_afterany_dependency,
    validate_leaf_rows,
    validate_pass_receipts,
    validate_population_manifest,
    validate_pre_run,
    validate_runtime_identity_receipt,
    validate_warmup_ledger,
)


def _execution_class(
    pre_run: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    hardware = receipt.get("hardware_identity")
    software = receipt.get("software_identity")
    if not isinstance(hardware, Mapping) or not isinstance(software, Mapping):
        raise ValueError("steady-cost receipt lacks runtime identity")
    if (
        receipt.get("hardware_fingerprint") != canonical_sha256(hardware)
        or receipt.get("software_fingerprint") != canonical_sha256(software)
    ):
        raise ValueError("steady-cost stored runtime fingerprint changed")
    runtime_identity = receipt.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        raise ValueError("steady-cost receipt lacks observed PRE_RUN identity")
    validate_runtime_identity_receipt(
        runtime_identity, pre_run=pre_run, hardware=hardware, software=software
    )
    nvidia_smi = hardware.get("nvidia_smi")
    if not isinstance(nvidia_smi, Mapping):
        raise ValueError("steady-cost receipt lacks nvidia-smi identity")
    identity = {
        "gpu_name": hardware.get("gpu_name"),
        "gpu_total_memory": hardware.get("gpu_total_memory"),
        "gpu_compute_capability": hardware.get("gpu_compute_capability"),
        "gpu_multi_processor_count": hardware.get("gpu_multi_processor_count"),
        "driver_version": nvidia_smi.get("driver_version"),
        "persistence_mode": nvidia_smi.get("persistence_mode"),
        "compute_mode": nvidia_smi.get("compute_mode"),
        "power_limit": nvidia_smi.get("power.limit"),
        "max_sm_clock": nvidia_smi.get("clocks.max.sm"),
        "max_memory_clock": nvidia_smi.get("clocks.max.memory"),
    }
    software_fingerprint = receipt.get("software_fingerprint")
    if any(value is None or value == "" for value in identity.values()) or not isinstance(
        software_fingerprint, str
    ) or len(software_fingerprint) != 64:
        raise ValueError("steady-cost execution class is incomplete")
    return {"hardware": identity, "software_fingerprint": software_fingerprint}


def _validate_execution_classes(
    pre_run: Mapping[str, Any], receipts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    preflight_gate = pre_run["gates"]["preflight"]
    preflight_path = Path(str(preflight_gate["path"])).resolve()
    if not preflight_path.is_file() or preflight_gate["sha256"] != sha256_file(preflight_path):
        raise ValueError("steady-cost preflight gate file receipt is invalid")
    preflight = read_json_object(preflight_path, label="steady-cost preflight gate")
    require_self_hash(preflight, field="receipt_sha256", label="steady-cost preflight gate")
    if (
        preflight.get("schema_version") != "zoomtoken_scnr_steady_cost_preflight_v001"
        or preflight.get("status") != "MECHANICAL_READY"
        or preflight.get("pre_run_sha256") != preflight_gate["pre_run_sha256"]
        or preflight.get("repair_sha") != pre_run["repair_sha"]
        or preflight.get("held_out_test_opened") is not False
        or preflight.get("metric_evaluation_executed") is not False
    ):
        raise ValueError("steady-cost preflight gate payload is invalid")
    reference = _execution_class(pre_run, preflight)
    for leaf_id, receipt in receipts.items():
        if _execution_class(pre_run, receipt) != reference:
            raise ValueError(f"steady-cost {leaf_id} execution class differs from preflight")
    return reference


def query_sacct(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids or any(not job_id.isdigit() for job_id in job_ids):
        raise ValueError("steady-cost scheduler query has invalid Job IDs")
    command = [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        ",".join(job_ids),
        "--format=JobIDRaw,State,ExitCode,Dependency",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4:
            continue
        job_id, state, exit_code, dependency = fields[:4]
        if job_id in job_ids and "." not in job_id:
            if job_id in rows:
                raise ValueError("steady-cost scheduler returned a duplicate Job")
            rows[job_id] = {
                "job_id": job_id,
                "state": state,
                "exit_code": exit_code,
                "dependency": dependency,
            }
    if set(rows) != set(job_ids):
        raise ValueError("steady-cost scheduler omitted a Job")
    return rows


def _artifact_path(
    receipt: Mapping[str, Any], name: str, leaf_root: Path, *, require_sha: bool = True
) -> Path:
    artifact = receipt.get("artifacts", {}).get(name)
    if not isinstance(artifact, Mapping):
        raise ValueError(f"steady-cost leaf lacks {name}")
    path = Path(str(artifact.get("path", ""))).resolve()
    try:
        path.relative_to(leaf_root.resolve())
    except ValueError as error:
        raise ValueError("steady-cost leaf artifact escaped its output root") from error
    if not path.is_file() or (
        require_sha and artifact.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"steady-cost leaf {name} receipt is invalid")
    return path


def load_complete_leaf(
    pre_run: Mapping[str, Any], *, leaf_id: str, job_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = (
        Path(pre_run["results_root"])
        / STUDY_ID
        / pre_run["repair_sha"]
        / "leaves"
        / leaf_id
    )
    receipt_path = root / "receipt.json"
    receipt = read_json_object(receipt_path, label=f"steady-cost {leaf_id} receipt")
    require_self_hash(receipt, field="receipt_sha256", label=f"steady-cost {leaf_id}")
    if (
        receipt.get("schema_version") != "zoomtoken_scnr_steady_cost_leaf_v001"
        or receipt.get("status") != "COMPLETE_LEAF_CANDIDATE"
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("leaf_id") != leaf_id
        or receipt.get("order") != LEAF_ORDERS[leaf_id]
        or receipt.get("slurm_job_id") != job_id
        or receipt.get("pre_run_sha256") != pre_run["pre_run_sha256"]
        or receipt.get("repair_sha") != pre_run["repair_sha"]
        or receipt.get("training_or_resume_executed") is not False
        or receipt.get("metric_evaluation_executed") is not False
        or receipt.get("held_out_test_opened") is not False
        or receipt.get("authoritative_decision") is not False
    ):
        raise ValueError(f"steady-cost {leaf_id} receipt header is invalid")
    samples_path = _artifact_path(receipt, "measured_samples", root)
    warmup_path = _artifact_path(
        receipt, "warmup_identities", root, require_sha=False
    )
    _artifact_path(receipt, "power_trace", root)
    _artifact_path(receipt, "sidecar_report", root)
    _artifact_path(receipt, "sidecar_trace", root)
    rows = read_jsonl_objects(samples_path, label=f"steady-cost {leaf_id} samples")
    validate_leaf_rows(rows, leaf_id=leaf_id)
    warmup_rows = read_jsonl_objects(
        warmup_path, label=f"steady-cost {leaf_id} warmup identities"
    )
    warmup_receipt = receipt["artifacts"]["warmup_identities"]
    if int(warmup_receipt.get("row_count", -1)) != len(warmup_rows):
        raise ValueError("steady-cost warmup row count changed")
    manifest = validate_population_manifest(
        read_json_object(
            ROOT / pre_run["population"]["manifest_path"],
            label="steady-cost population manifest",
        )
    )
    validate_warmup_ledger(
        warmup_rows,
        leaf_id=leaf_id,
        sequence=leaf_sequence(leaf_id),
        population=manifest,
    )
    source = validate_residual_centering_cost_source(
        pre_run["training_run_root"],
        expected_model_runtime_commit=pre_run["model_runtime_sha"],
    )
    validate_pass_receipts(
        ROOT,
        receipt.get("pass_receipts", ()),
        sequence=leaf_sequence(leaf_id),
        source=source,
        expected_accuracy_population_sha256=pre_run["population"][
            "source_population_sha256"
        ],
        measured_rows=rows,
    )
    return receipt, rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-run", type=Path, required=True)
    parser.add_argument("--leaf-job-ids", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("steady-cost finalizer requires Slurm")
    if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("SLURM_JOB_GPUS"):
        raise RuntimeError("steady-cost Job 9 must be CPU-only")
    pre_run = validate_pre_run(
        read_json_object(args.pre_run, label="steady-cost full PRE_RUN"), phase="full"
    )
    job_ids = [item.strip() for item in args.leaf_job_ids.split(",") if item.strip()]
    if len(job_ids) != 8 or len(set(job_ids)) != 8 or any(not item.isdigit() for item in job_ids):
        raise ValueError("steady-cost Job 9 requires eight unique numeric leaf Job IDs")
    if os.environ["SLURM_JOB_ID"] in set(job_ids):
        raise ValueError("steady-cost Job 9 cannot reuse a leaf Job ID")
    job_map = dict(zip(LEAF_ORDERS, job_ids))
    finalizer_job_id = os.environ["SLURM_JOB_ID"]
    scheduler = query_sacct(job_ids + [finalizer_job_id])
    validate_afterany_dependency(
        scheduler[finalizer_job_id],
        expected_job_id=finalizer_job_id,
        expected_parent_job_ids=job_ids,
    )
    leaf_scheduler = {job_id: scheduler[job_id] for job_id in job_ids}
    failures = []
    leaves = {}
    receipts = {}
    receipt_payloads = {}
    for leaf_id, job_id in job_map.items():
        state = leaf_scheduler[job_id]
        if state["state"] != "COMPLETED" or state["exit_code"] != "0:0":
            failures.append(
                {"leaf_id": leaf_id, "job_id": job_id, "reason": "scheduler_not_completed_0_0", "scheduler": state}
            )
            continue
        try:
            receipt, rows = load_complete_leaf(pre_run, leaf_id=leaf_id, job_id=job_id)
            receipts[leaf_id] = {
                "path": str(
                    Path(pre_run["results_root"])
                    / STUDY_ID
                    / pre_run["repair_sha"]
                    / "leaves"
                    / leaf_id
                    / "receipt.json"
                ),
                "receipt_sha256": receipt["receipt_sha256"],
            }
            receipt_payloads[leaf_id] = receipt
            leaves[leaf_id] = rows
        except Exception as error:  # Job 9 must record malformed leaves fail-closed.
            failures.append(
                {"leaf_id": leaf_id, "job_id": job_id, "reason": type(error).__name__, "detail": str(error)}
            )
    analysis = None
    bootstrap_draws = None
    bootstrap_receipt = None
    candidate_decision = "INVALID_FAIL_CLOSED"
    execution_class = None
    if not failures:
        try:
            execution_class = _validate_execution_classes(pre_run, receipt_payloads)
            analysis, bootstrap_draws = analyze_complete_leaves_with_draws(leaves)
            candidate_decision = analysis["candidate_decision"]
        except Exception as error:
            failures.append(
                {"reason": type(error).__name__, "detail": str(error)}
            )
    output = (
        Path(pre_run["results_root"])
        / STUDY_ID
        / pre_run["repair_sha"]
        / "aggregate"
    )
    if output.exists():
        raise FileExistsError("steady-cost aggregate path already exists")
    output.mkdir(parents=True, exist_ok=False)
    if bootstrap_draws is not None:
        draws_path = atomic_write_jsonl(output / "bootstrap_draws.jsonl", bootstrap_draws)
        bootstrap_receipt = {
            "path": str(draws_path.resolve()),
            "row_count": len(bootstrap_draws),
        }
    aggregate = add_self_hash(
        {
            "schema_version": "zoomtoken_scnr_steady_cost_aggregate_candidate_v001",
            "study_id": STUDY_ID,
            "status": "AGGREGATE_CANDIDATE_ONLY",
            "candidate_decision": candidate_decision,
            "authoritative": False,
            "pre_run_sha256": pre_run["pre_run_sha256"],
            "repair_sha": pre_run["repair_sha"],
            "finalizer_job_id": finalizer_job_id,
            "finalizer_dependency": "afterany",
            "finalizer_scheduler": scheduler[finalizer_job_id],
            "leaf_job_ids": job_map,
            "leaf_scheduler": leaf_scheduler,
            "leaf_receipts": receipts,
            "execution_class": execution_class,
            "failures": failures,
            "analysis": analysis,
            "bootstrap_draws": bootstrap_receipt,
            "partial_ratios_emitted": False,
            "automatic_rerun_allowed": False,
            "held_out_test_opened": False,
            "paper_claim_allowed": False,
        },
        field="aggregate_sha256",
    )
    atomic_write_json(output / "aggregate_candidate.json", aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
