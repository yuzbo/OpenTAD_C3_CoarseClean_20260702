#!/usr/bin/env python3
"""Aggregate the four frozen PhysTime decode-cross replay conditions."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from tools.bata.validate_phystime_p0_fullprecision_suite import (
    compare_prediction_decisions,
    load_ground_truth,
    proposal_recall_diagnostics,
)


EXPECTED_RUNS = {
    "selected_online": ("selected_axis", "online"),
    "selected_ema": ("selected_axis", "ema"),
    "physical_online": ("physical_metric", "online"),
    "physical_ema": ("physical_metric", "ema"),
}
AXES = ("uniform_rank_seconds", "physical_time_seconds")
METRIC_EPSILON = 1.0e-12
P0_RUNTIME_COMMIT = "c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c"
P0_RUNTIME_TREE = "0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338"
P0_SUITE_SHA256 = (
    "afb3e300424a57eb590a21129217e040677dc875fdede3be344352dc2bd268e7"
)
P0_GATE_SHA256 = (
    "1ca0efcdeb9f6343da076a00660675759358ac467074919a34d01c0d7c7250d9"
)
P0_DATASET_MANIFEST_SHA256 = (
    "1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2"
)
P0_VIDEOMAE_SHA256 = (
    "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
)
FATAL_LOG_PATTERNS = (
    r"\btraceback\b",
    r"\bcuda out of memory\b",
    r"\boutofmemoryerror\b",
    r"\boom(?:[- ]|_)?kill(?:ed)?\b",
    r"\bkilled\b",
    r"\bsegmentation fault\b|\bsegfault\b",
    r"\bbus error\b",
    r"\bnccl\b[^\n]{0,120}\berror\b",
    r"\bloss\s*=\s*(?:nan|[-+]?inf)\b",
    r"\bnan gradient\b",
    r"\bnon[- ]finite\b",
    r"\bamp skipped optimizer step\b",
    r"\bfilenotfounderror\b",
    r"\bpytorchstreamwriter\b",
    r"\bdependencyneversatisfied\b",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the four decode-cross runs."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_dependency_contract(value):
    value = str(value).strip()
    if value == "none":
        return []
    parts = value.split(":")
    require(
        len(parts) >= 2 and parts[0] == "afterok",
        f"invalid expected dependency contract: {value}",
    )
    job_ids = parts[1:]
    require(
        all(job_id.isdigit() for job_id in job_ids),
        f"expected dependency contains a non-numeric job ID: {value}",
    )
    require(
        len(set(job_ids)) == len(job_ids),
        f"expected dependency contains a duplicate job ID: {value}",
    )
    return [
        {
            "dependency_type": "afterok",
            "job_id": job_id,
        }
        for job_id in sorted(job_ids, key=int)
    ]


def validate_scheduler_job_snapshots(
    jobs,
    submission_jobs,
    terminal_jobs,
):
    require(
        set(submission_jobs) == set(jobs)
        and set(terminal_jobs) == set(jobs),
        "scheduler snapshots do not cover the full DAG",
    )
    for variant, job in jobs.items():
        submission_job = submission_jobs[variant]
        terminal_job = terminal_jobs[variant]
        expected_contract = expected_dependency_contract(
            job["dependency"]
        )
        require(
            submission_job.get("job_id") == job["job_id"]
            and submission_job.get("job_name") == job["job_name"]
            and submission_job.get("comment") == job["comment"]
            and submission_job.get("expected_dependency")
            == job["dependency"]
            and submission_job.get("expected_dependency_contract")
            == expected_contract
            and submission_job.get("dependency_contract")
            == expected_contract,
            f"{variant} submission scheduler identity/dependency mismatch",
        )
        require(
            terminal_job.get("job_id") == job["job_id"]
            and terminal_job.get("job_name") == job["job_name"]
            and terminal_job.get("comment") == job["comment"],
            f"{variant} terminal scheduler identity mismatch",
        )


def read_json(path, description):
    path = Path(path)
    require(path.is_file(), f"missing {description}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finite_metrics(payload):
    metrics = {key: float(value) for key, value in payload.items()}
    require(metrics, "metric dictionary is empty")
    require(
        all(math.isfinite(value) for value in metrics.values()),
        "metric dictionary contains a non-finite value",
    )
    return metrics


def metric_delta(lhs, rhs):
    lhs = finite_metrics(lhs)
    rhs = finite_metrics(rhs)
    require(lhs.keys() == rhs.keys(), "metric keys differ")
    return {
        "fraction": {key: lhs[key] - rhs[key] for key in sorted(lhs)},
        "percentage_points": {
            key: 100.0 * (lhs[key] - rhs[key]) for key in sorted(lhs)
        },
    }


def metric_linear_combination(*terms):
    keys = None
    normalized = []
    for coefficient, payload in terms:
        values = finite_metrics(payload)
        if keys is None:
            keys = values.keys()
        require(values.keys() == keys, "interaction metric keys differ")
        normalized.append((float(coefficient), values))
    values = {
        key: sum(coefficient * payload[key] for coefficient, payload in normalized)
        for key in sorted(keys)
    }
    return {
        "fraction": values,
        "percentage_points": {
            key: 100.0 * value for key, value in values.items()
        },
    }


def read_jobs_tsv(path):
    path = Path(path).resolve()
    require(path.is_file(), f"missing jobs TSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_variant = {row["variant"]: row for row in rows}
    require(
        len(by_variant) == len(rows) == 6,
        "jobs TSV must contain six unique DAG rows",
    )
    require(
        set(by_variant)
        == set(EXPECTED_RUNS) | {"decode_cross_gate", "decode_cross_suite"},
        "jobs TSV variant set mismatch",
    )
    for variant, row in by_variant.items():
        sbatch_path = Path(row["sbatch_path"]).resolve()
        require(
            sha256_file(sbatch_path) == row["sbatch_sha256"],
            f"{variant} sbatch hash mismatch",
        )
        require(row["status"] == "submitted", f"{variant} was not submitted")
        require(
            row["job_id"].isdigit()
            and row["dag_token"]
            and row["comment"] == f"{row['dag_token']}:{variant}",
            f"{variant} jobs TSV identity fields are invalid",
        )
        expected_name = (
            "pt_dc_gate"
            if variant == "decode_cross_gate"
            else (
                "pt_dc_suite"
                if variant == "decode_cross_suite"
                else f"pt_dc_{variant}"
            )
        )
        require(
            row["job_name"] == expected_name,
            f"{variant} jobs TSV job name mismatch",
        )
    return by_variant


def scan_logs(paths):
    findings = []
    for path in paths:
        path = Path(path).resolve()
        require(path.is_file(), f"missing log: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FATAL_LOG_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                findings.append({"path": str(path), "pattern": pattern})
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[PhysTime") and " ERROR:" in stripped:
                findings.append(
                    {"path": str(path), "marker": stripped[:200]}
                )
    require(not findings, f"fatal log markers found: {findings[:5]}")
    return {"files_scanned": len(paths), "fatal_findings": findings}


def validate_suite(run_root):
    run_root = Path(run_root).resolve()
    deployment_path = run_root / "deployment_summary.json"
    deployment = read_json(deployment_path, "deployment summary")
    require(
        deployment.get("schema_version")
        == "phystime_decode_cross_deployment_v1",
        "deployment summary schema mismatch",
    )
    require(
        deployment.get("new_training") is False
        and deployment.get("frozen_epoch") == 59,
        "deployment is not frozen epoch-59 replay",
    )
    owner_path = Path(
        deployment["submission_owner_manifest"]
    ).resolve()
    owner = read_json(owner_path, "submission owner manifest")
    global_owner_path = Path(
        deployment["global_submission_owner_manifest"]
    ).resolve()
    global_owner = read_json(
        global_owner_path,
        "global submission owner manifest",
    )
    require(
        owner.get("schema_version")
        == "phystime_decode_cross_submission_owner_v1"
        and owner.get("dag_token") == deployment["dag_token"]
        and owner.get("runtime_commit") == deployment["runtime_commit"]
        and owner.get("runtime_tree") == deployment["runtime_tree"]
        and Path(owner.get("run_root", "")).resolve() == run_root
        and sha256_file(owner_path)
        == deployment["submission_owner_manifest_sha256"],
        "submission owner manifest mismatch",
    )
    require(
        global_owner.get("schema_version")
        == "phystime_decode_cross_submission_owner_v1"
        and global_owner.get("dag_token") == deployment["dag_token"]
        and global_owner.get("runtime_commit") == deployment["runtime_commit"]
        and global_owner.get("runtime_tree") == deployment["runtime_tree"]
        and Path(global_owner.get("run_root", "")).resolve() == run_root
        and global_owner.get("run_uuid") == owner.get("run_uuid")
        and Path(
            global_owner.get("global_owner_manifest", "")
        ).resolve()
        == global_owner_path
        and Path(
            global_owner.get("local_owner_manifest", "")
        ).resolve()
        == owner_path
        and sha256_file(global_owner_path)
        == deployment["global_submission_owner_manifest_sha256"],
        "global submission owner manifest mismatch",
    )
    require(
        Path(owner.get("global_owner_manifest", "")).resolve()
        == global_owner_path
        and Path(owner.get("local_owner_manifest", "")).resolve()
        == owner_path,
        "local submission owner does not bind the global owner",
    )
    jobs_path = Path(deployment["jobs_tsv"]).resolve()
    require(
        sha256_file(jobs_path) == deployment["jobs_tsv_sha256"],
        "jobs TSV differs from deployment summary",
    )
    jobs = read_jobs_tsv(jobs_path)
    submission_attempt_root = run_root / "submission_attempts"
    unresolved_attempts = sorted(
        str(path)
        for pattern in ("*.ambiguous.json", "*.fatal.json")
        for path in submission_attempt_root.glob(pattern)
    )
    require(
        not unresolved_attempts,
        f"unresolved or fatal submission attempts remain: {unresolved_attempts}",
    )
    resolved_paths = {
        path.name.removesuffix(".resolved.json"): path
        for path in submission_attempt_root.glob("*.resolved.json")
    }
    require(
        set(resolved_paths) == set(jobs),
        "resolved submission marker set differs from jobs TSV",
    )
    for variant, row in jobs.items():
        marker = read_json(
            resolved_paths[variant],
            f"{variant} resolved submission marker",
        )
        require(
            marker.get("schema_version")
            == "phystime_decode_cross_submission_attempt_v1"
            and marker.get("state") == "resolved"
            and Path(marker.get("run_root", "")).resolve() == run_root
            and marker.get("dag_token") == row["dag_token"]
            == deployment["dag_token"]
            and marker.get("variant") == variant
            and marker.get("comment") == row["comment"]
            and str(marker.get("resolved_job_id", "")) == row["job_id"]
            and marker.get("runtime_commit")
            == deployment["runtime_commit"]
            and marker.get("runtime_tree") == deployment["runtime_tree"],
            f"{variant} resolved submission marker differs from jobs TSV",
        )
    dag_tokens = {row["dag_token"] for row in jobs.values()}
    require(
        dag_tokens == {deployment["dag_token"]},
        "jobs TSV DAG token differs from deployment",
    )
    require(
        os.environ.get("SLURM_JOB_ID")
        == str(deployment["suite_job"])
        == jobs["decode_cross_suite"]["job_id"],
        "suite Slurm job identity mismatch",
    )
    require(
        jobs["decode_cross_gate"]["job_id"]
        == str(deployment["gate_job"]),
        "gate Slurm job identity mismatch",
    )
    require(
        os.environ.get("SLURM_JOB_NAME")
        == jobs["decode_cross_suite"]["job_name"]
        and os.environ.get("PHYSTIME_EXPECTED_JOB_COMMENT")
        == jobs["decode_cross_suite"]["comment"],
        "suite live Slurm name/comment mismatch",
    )
    expected_suite_dependency = "afterok:" + ":".join(
        jobs[variant]["job_id"]
        for variant in (
            "selected_online",
            "selected_ema",
            "physical_online",
            "physical_ema",
        )
    )
    require(
        deployment["suite_dependency"] == expected_suite_dependency,
        "suite dependency cannot be reconstructed from replay job IDs",
    )
    preflight_path = Path(deployment["preflight_manifest"]).resolve()
    preflight = read_json(preflight_path, "decode-cross preflight")
    require(
        preflight.get("schema_version")
        == "phystime_decode_cross_preflight_v1"
        and preflight.get("validation_pass") is True
        and sha256_file(preflight_path)
        == deployment["preflight_manifest_sha256"],
        "decode-cross preflight artifact mismatch",
    )
    scheduler_submission_path = Path(
        deployment["scheduler_submission"]
    ).resolve()
    scheduler_submission = read_json(
        scheduler_submission_path,
        "submission scheduler snapshot",
    )
    require(
        scheduler_submission.get("schema_version")
        == "phystime_decode_cross_scheduler_snapshot_v2"
        and scheduler_submission.get("validation_pass") is True
        and scheduler_submission.get("mode") == "submission"
        and scheduler_submission.get("dag_token") == deployment["dag_token"]
        and sha256_file(scheduler_submission_path)
        == deployment["scheduler_submission_sha256"],
        "submission scheduler snapshot mismatch",
    )
    scheduler_terminal_path = run_root / "scheduler_terminal.json"
    scheduler_terminal = read_json(
        scheduler_terminal_path,
        "terminal scheduler snapshot",
    )
    require(
        scheduler_terminal.get("schema_version")
        == "phystime_decode_cross_scheduler_snapshot_v2"
        and scheduler_terminal.get("validation_pass") is True
        and scheduler_terminal.get("mode") == "terminal"
        and scheduler_terminal.get("dag_token") == deployment["dag_token"]
        and scheduler_terminal.get("jobs_tsv_sha256")
        == deployment["jobs_tsv_sha256"],
        "terminal scheduler snapshot mismatch",
    )
    validate_scheduler_job_snapshots(
        jobs,
        scheduler_submission["jobs"],
        scheduler_terminal["jobs"],
    )
    decode_gate_path = Path(deployment["gate_output"]).resolve()
    decode_gate = read_json(decode_gate_path, "decode-cross gate")
    gate_slurm = decode_gate.get("runtime", {}).get("slurm", {})
    require(
        decode_gate.get("gate_pass") is True
        and gate_slurm.get("job_id")
        == jobs["decode_cross_gate"]["job_id"]
        and gate_slurm.get("job_variant") == "decode_cross_gate"
        and gate_slurm.get("job_name")
        == jobs["decode_cross_gate"]["job_name"]
        and gate_slurm.get("dag_token")
        == jobs["decode_cross_gate"]["dag_token"]
        and gate_slurm.get("comment")
        == jobs["decode_cross_gate"]["comment"]
        and gate_slurm.get("expected_dependency") == "none"
        and gate_slurm.get("sbatch_sha256")
        == jobs["decode_cross_gate"]["sbatch_sha256"],
        "decode-cross gate Slurm provenance mismatch",
    )
    require(
        decode_gate.get("preflight", {}).get("sha256")
        == deployment["preflight_manifest_sha256"]
        and Path(decode_gate["preflight"]["path"]).resolve()
        == preflight_path,
        "decode-cross gate preflight binding mismatch",
    )
    require(
        jobs["decode_cross_suite"]["dependency"]
        == deployment["suite_dependency"],
        "suite dependency differs from deployment",
    )
    p0_suite_path = Path(deployment["p0_suite_completion"]).resolve()
    p0_suite = read_json(p0_suite_path, "P0 suite completion")
    require(
        p0_suite.get("schema_version")
        == "phystime_p0_fullprecision_suite_completion_v1"
        and p0_suite.get("validation_pass") is True
        and p0_suite.get("runtime_commit") == P0_RUNTIME_COMMIT
        and p0_suite.get("runtime_tree") == P0_RUNTIME_TREE
        and p0_suite.get("source_commit") == deployment["source_commit"]
        and p0_suite.get("source_tree") == deployment["source_tree"],
        "P0 suite provenance/schema did not pass",
    )
    require(
        sha256_file(p0_suite_path) == P0_SUITE_SHA256,
        "P0 suite differs from the reviewed artifact",
    )

    completions = {}
    manifests = {}
    metrics = {}
    predictions = {}
    completion_artifacts = {}
    capture_contracts = {}
    numeric_precision_contracts = {}
    shared_observation_contract = None
    identity = None
    p0_gate_path = None
    for variant, (expected_arm, expected_weights) in EXPECTED_RUNS.items():
        variant_dir = run_root / variant
        completion_path = variant_dir / "DECODE_CROSS_COMPLETE.json"
        completion = read_json(completion_path, f"{variant} completion")
        require(
            completion.get("schema_version")
            == "phystime_decode_cross_completion_v1"
            and completion.get("validation_pass") is True
            and completion.get("status") == "tested",
            f"{variant} recompute completion did not pass",
        )
        require(
            completion.get("arm") == expected_arm
            and completion.get("weights_source") == expected_weights,
            f"{variant} condition mismatch",
        )
        require(
            completion.get("new_training") is False
            and completion.get("evaluation_epoch") == 59
            and completion.get("same_frozen_raw_tensors_for_both_axes")
            is True
            and completion.get("native_direct_exact_equivalence") is True
            and completion.get("reviewed_p0_direct_exact_equivalence") is True,
            f"{variant} frozen/equivalence contract mismatch",
        )
        require(
            Path(completion["run_dir"]).resolve() == variant_dir.resolve(),
            f"{variant} completion run directory mismatch",
        )
        condition_identity = (
            completion["runtime_commit"],
            completion["runtime_tree"],
            completion["source_commit"],
            completion["source_tree"],
        )
        if identity is None:
            identity = condition_identity
        require(
            condition_identity == identity,
            "decode-cross suite snapshot identity mismatch",
        )
        require(
            completion["runtime_commit"] == deployment["runtime_commit"]
            and completion["runtime_tree"] == deployment["runtime_tree"]
            and completion["source_commit"] == deployment["source_commit"]
            and completion["source_tree"] == deployment["source_tree"],
            f"{variant} identity differs from deployment summary",
        )
        for record in completion["artifacts"].values():
            require(
                sha256_file(record["path"]) == record["sha256"],
                f"{variant} artifact hash mismatch",
            )
        for mode_records in completion["mode_artifacts"].values():
            for record in mode_records.values():
                require(
                    sha256_file(record["path"]) == record["sha256"],
                    f"{variant} mode artifact hash mismatch",
                )
        manifest = read_json(
            completion["artifacts"]["run_manifest"]["path"],
            f"{variant} run manifest",
        )
        require(
            manifest.get("arm") == expected_arm
            and manifest.get("weights_source") == expected_weights
            and manifest.get("new_training") is False,
            f"{variant} manifest mismatch",
        )
        job = jobs[variant]
        slurm = manifest.get("slurm", {})
        require(
            slurm.get("job_id") == job["job_id"]
            and slurm.get("job_name") == job["job_name"]
            and slurm.get("job_variant") == variant
            and slurm.get("dag_token") == job["dag_token"]
            and slurm.get("comment") == job["comment"]
            and slurm.get("expected_dependency") == job["dependency"]
            and Path(slurm.get("sbatch_path", "")).resolve()
            == Path(job["sbatch_path"]).resolve()
            and slurm.get("sbatch_sha256") == job["sbatch_sha256"],
            f"{variant} Slurm identity/provenance mismatch",
        )
        require(
            Path(manifest["preflight_manifest"]).resolve()
            == preflight_path
            and manifest["preflight_manifest_sha256"]
            == deployment["preflight_manifest_sha256"],
            f"{variant} preflight provenance mismatch",
        )
        runtime_preflight_path = Path(
            manifest["runtime_preflight_manifest"]
        ).resolve()
        require(
            runtime_preflight_path
            == Path(
                completion["artifacts"]["runtime_preflight_manifest"][
                    "path"
                ]
            ).resolve()
            and manifest["runtime_preflight_manifest_sha256"]
            == completion["artifacts"]["runtime_preflight_manifest"][
                "sha256"
            ]
            == deployment["preflight_manifest_sha256"],
            f"{variant} replay-time preflight provenance mismatch",
        )
        require(
            str(deployment["jobs"][variant]) == job["job_id"],
            f"{variant} job ID differs from deployment summary",
        )
        direct_marker_path = variant_dir / "DIRECT_INFERENCE_COMPLETE"
        validated_marker_path = variant_dir / "DECODE_CROSS_VALIDATED"
        direct_marker = read_json(
            direct_marker_path,
            f"{variant} direct marker",
        )
        validated_marker = read_json(
            validated_marker_path,
            f"{variant} validated marker",
        )
        require(
            direct_marker.get("schema_version")
            == "phystime_decode_cross_direct_marker_v1"
            and direct_marker.get("validation_pass") is True,
            f"{variant} direct marker mismatch",
        )
        require(
            len(direct_marker.get("artifacts", {})) == 6,
            f"{variant} direct marker artifact set is incomplete",
        )
        for record in direct_marker["artifacts"].values():
            require(
                sha256_file(record["path"]) == record["sha256"],
                f"{variant} direct marker artifact hash mismatch",
            )
        runtime_summary_path = variant_dir / "runtime_summary.json"
        runtime_summary = read_json(
            runtime_summary_path,
            f"{variant} runtime summary",
        )
        require(
            runtime_summary.get("schema_version")
            == "phystime_decode_cross_runtime_summary_v1"
            and runtime_summary.get("validation_pass") is True
            and validated_marker.get("schema_version")
            == "phystime_decode_cross_validated_marker_v1"
            and validated_marker.get("validation_pass") is True
            and validated_marker["completion_sha256"]
            == sha256_file(completion_path)
            and validated_marker["runtime_summary_sha256"]
            == sha256_file(runtime_summary_path),
            f"{variant} validated marker/runtime summary mismatch",
        )
        capture_manifest = read_json(
            completion["artifacts"]["capture_manifest"]["path"],
            f"{variant} capture manifest",
        )
        numeric_precision = completion.get("numeric_precision", {})
        require(
            numeric_precision.get("source_amp_enabled")
            == capture_manifest.get("source_amp_enabled")
            and numeric_precision.get("source_tensor_dtypes")
            == capture_manifest.get("source_tensor_dtypes")
            and numeric_precision.get("decode_compute_dtype") == "float32"
            and numeric_precision.get("decode_compute_device") == "cpu"
            and set(
                numeric_precision.get("stored_tensor_dtypes", {}).values()
            )
            == {"float32"},
            f"{variant} numeric precision provenance mismatch",
        )
        numeric_precision_contracts[variant] = numeric_precision
        observation_contract = {
            "observation_sequence_sha256": capture_manifest[
                "observation_sequence_sha256"
            ],
            "uniform_axis_sha256": capture_manifest["array_contract"][
                "uniform_axis_sec"
            ]["canonical_sha256"],
            "physical_axis_sha256": capture_manifest["array_contract"][
                "physical_axis_sec"
            ]["canonical_sha256"],
            "base_mask_sha256": capture_manifest["array_contract"][
                "base_mask"
            ]["canonical_sha256"],
            "native_mask_sha256": capture_manifest["array_contract"][
                "native_mask"
            ]["canonical_sha256"],
            "base_points_sha256": capture_manifest["array_contract"][
                "base_points"
            ]["canonical_sha256"],
            "class_map": capture_manifest["class_map"],
            "window_count": capture_manifest["window_count"],
            "candidate_count": capture_manifest["candidate_count"],
            "native_token_count": capture_manifest["native_token_count"],
        }
        if shared_observation_contract is None:
            shared_observation_contract = observation_contract
        require(
            observation_contract == shared_observation_contract,
            f"{variant} sparse observation/time-axis contract differs",
        )
        capture_contracts[variant] = {
            **observation_contract,
            "native_coordinate_mode": capture_manifest[
                "expected_native_coordinate_mode"
            ],
            "window_sequence_sha256": capture_manifest[
                "window_sequence_sha256"
            ],
        }
        p0_completion = read_json(
            completion["artifacts"]["p0_completion"]["path"],
            f"{variant} P0 completion",
        )
        p0_suite_artifact = p0_suite["completion_artifacts"][variant]
        require(
            Path(p0_suite_artifact["path"]).resolve()
            == Path(
                completion["artifacts"]["p0_completion"]["path"]
            ).resolve()
            and p0_suite_artifact["sha256"]
            == completion["artifacts"]["p0_completion"]["sha256"],
            f"{variant} P0 completion differs from the reviewed suite",
        )
        require(
            p0_completion.get("schema_version")
            == "phystime_p0_fullprecision_completion_v2"
            and p0_completion.get("runtime_commit") == P0_RUNTIME_COMMIT
            and p0_completion.get("runtime_tree") == P0_RUNTIME_TREE
            and p0_completion.get("validation_pass") is True
            and p0_completion.get("arm") == expected_arm
            and p0_completion.get("weights_source") == expected_weights,
            f"{variant} P0 provenance mismatch",
        )
        candidate_gate_path = Path(
            p0_completion["artifacts"]["gate"]["path"]
        ).resolve()
        require(
            p0_completion["artifacts"]["gate"]["sha256"]
            == P0_GATE_SHA256,
            f"{variant} P0 gate hash mismatch",
        )
        if p0_gate_path is None:
            p0_gate_path = candidate_gate_path
        require(
            candidate_gate_path == p0_gate_path,
            "suite P0 gate path differs across conditions",
        )

        variant_metrics = {}
        variant_predictions = {}
        for axis_name in AXES:
            variant_metrics[axis_name] = finite_metrics(
                completion["mode_metrics"][axis_name]
            )
            result_path = completion["mode_artifacts"][axis_name][
                "result"
            ]["path"]
            result = read_json(
                result_path,
                f"{variant}/{axis_name} result",
            )
            require(
                int(result.get("evaluation_epoch", -1)) == 59,
                f"{variant}/{axis_name} result epoch mismatch",
            )
            variant_predictions[axis_name] = result
        require(
            completion["physical_minus_uniform_fraction"]
            == metric_delta(
                variant_metrics["physical_time_seconds"],
                variant_metrics["uniform_rank_seconds"],
            )["fraction"],
            f"{variant} decode delta mismatch",
        )
        completions[variant] = completion
        manifests[variant] = manifest
        metrics[variant] = variant_metrics
        predictions[variant] = variant_predictions
        completion_artifacts[variant] = {
            "path": str(completion_path.resolve()),
            "sha256": sha256_file(completion_path),
            "size_bytes": completion_path.stat().st_size,
        }

    p0_gate = read_json(p0_gate_path, "shared P0 gate")
    require(
        p0_gate.get("schema_version") == "phystime_p0_fullprecision_gate_v1"
        and p0_gate.get("gate_pass") is True
        and sha256_file(p0_gate_path) == P0_GATE_SHA256
        and p0_gate["runtime"]["dataset_manifest_sha256"]
        == P0_DATASET_MANIFEST_SHA256
        and p0_gate["runtime"]["videomae_checkpoint_sha256"]
        == P0_VIDEOMAE_SHA256,
        "shared P0 gate did not match the reviewed artifact",
    )
    ground_truth, evaluation_contract = load_ground_truth(p0_gate)

    within_checkpoint_decode = {
        variant: metric_delta(
            metrics[variant]["physical_time_seconds"],
            metrics[variant]["uniform_rank_seconds"],
        )
        for variant in EXPECTED_RUNS
    }
    cross_checkpoint_descriptive_difference = {}
    for weights in ("online", "ema"):
        cross_checkpoint_descriptive_difference[weights] = {
            axis_name: metric_delta(
                metrics[f"physical_{weights}"][axis_name],
                metrics[f"selected_{weights}"][axis_name],
            )
            for axis_name in AXES
        }
    descriptive_difference_in_differences = {
        weights: metric_linear_combination(
            (
                1.0,
                metrics[f"physical_{weights}"]["physical_time_seconds"],
            ),
            (
                -1.0,
                metrics[f"physical_{weights}"]["uniform_rank_seconds"],
            ),
            (
                -1.0,
                metrics[f"selected_{weights}"]["physical_time_seconds"],
            ),
            (
                1.0,
                metrics[f"selected_{weights}"]["uniform_rank_seconds"],
            ),
        )
        for weights in ("online", "ema")
    }
    weight_source_effect = {
        arm: {
            axis_name: metric_delta(
                metrics[f"{arm}_ema"][axis_name],
                metrics[f"{arm}_online"][axis_name],
            )
            for axis_name in AXES
        }
        for arm in ("selected", "physical")
    }

    within_checkpoint_decisions = {
        variant: compare_prediction_decisions(
            predictions[variant]["physical_time_seconds"],
            predictions[variant]["uniform_rank_seconds"],
        )
        for variant in EXPECTED_RUNS
    }
    cross_checkpoint_decisions = {
        weights: {
            axis_name: compare_prediction_decisions(
                predictions[f"physical_{weights}"][axis_name],
                predictions[f"selected_{weights}"][axis_name],
            )
            for axis_name in AXES
        }
        for weights in ("online", "ema")
    }
    weight_source_decisions = {
        arm: {
            axis_name: compare_prediction_decisions(
                predictions[f"{arm}_ema"][axis_name],
                predictions[f"{arm}_online"][axis_name],
            )
            for axis_name in AXES
        }
        for arm in ("selected", "physical")
    }
    final_detection_oracle_recall = {
        variant: {
            axis_name: proposal_recall_diagnostics(
                predictions[variant][axis_name],
                ground_truth,
            )
            for axis_name in AXES
        }
        for variant in EXPECTED_RUNS
    }

    raw_rows = []
    for variant, (arm, weights) in EXPECTED_RUNS.items():
        for axis_name in AXES:
            raw_rows.append(
                {
                    "variant": variant,
                    "train_axis": (
                        "uniform_rank_seconds"
                        if arm == "selected_axis"
                        else "physical_time_seconds"
                    ),
                    "weights_source": weights,
                    "decode_axis": axis_name,
                    "metrics": metrics[variant][axis_name],
                }
            )
    log_paths = [
        Path(deployment["gate_stdout"]),
        Path(deployment["gate_stderr"]),
    ]
    for variant, manifest in manifests.items():
        variant_dir = run_root / variant
        log_paths.extend(
            [
                variant_dir / "inference.out",
                variant_dir / "replay.out",
                variant_dir / "validator.out",
                Path(manifest["slurm"]["stdout"]),
                Path(manifest["slurm"]["stderr"]),
            ]
        )
    log_scan = scan_logs(log_paths)

    return {
        "schema_version": "phystime_decode_cross_suite_completion_v1",
        "validation_pass": True,
        "status": "tested",
        "completed_at_unix": time.time(),
        "run_root": str(run_root),
        "new_training": False,
        "frozen_epoch": 59,
        "runtime_commit": identity[0],
        "runtime_tree": identity[1],
        "source_commit": identity[2],
        "source_tree": identity[3],
        "deployment_summary": {
            "path": str(deployment_path.resolve()),
            "sha256": sha256_file(deployment_path),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
        },
        "decode_cross_gate": {
            "path": str(decode_gate_path),
            "sha256": sha256_file(decode_gate_path),
        },
        "p0_suite": {
            "path": str(p0_suite_path),
            "sha256": sha256_file(p0_suite_path),
        },
        "p0_gate": {
            "path": str(p0_gate_path),
            "sha256": sha256_file(p0_gate_path),
        },
        "evaluation_contract": evaluation_contract,
        "ground_truth_count": len(ground_truth),
        "completion_artifacts": completion_artifacts,
        "slurm_dag": {
            "jobs_tsv": {
                "path": str(jobs_path),
                "sha256": sha256_file(jobs_path),
            },
            "jobs": jobs,
            "scheduler_submission": {
                "path": str(scheduler_submission_path),
                "sha256": sha256_file(scheduler_submission_path),
            },
            "scheduler_terminal": {
                "path": str(scheduler_terminal_path),
                "sha256": sha256_file(scheduler_terminal_path),
            },
        },
        "log_scan": log_scan,
        "shared_observation_contract": shared_observation_contract,
        "capture_contracts": capture_contracts,
        "numeric_precision_contracts": numeric_precision_contracts,
        "raw_metric_rows": raw_rows,
        "within_checkpoint_physical_decode_minus_uniform_decode": (
            within_checkpoint_decode
        ),
        "fixed_decode_cross_checkpoint_descriptive_difference": (
            cross_checkpoint_descriptive_difference
        ),
        "descriptive_difference_in_differences": (
            descriptive_difference_in_differences
        ),
        "weight_source_ema_minus_online": weight_source_effect,
        "within_checkpoint_decision_diagnostics": (
            within_checkpoint_decisions
        ),
        "cross_checkpoint_decision_diagnostics": (
            cross_checkpoint_decisions
        ),
        "weight_source_decision_diagnostics": weight_source_decisions,
        "final_detection_oracle_recall_by_duration_and_iou": (
            final_detection_oracle_recall
        ),
        "claim_boundary": (
            "This frozen single-seed THUMOS replay isolates inference decode "
            "axis effects from one raw tensor artifact per checkpoint. It "
            "uses a duplicate recomputation of the production post-processing "
            "and evaluator semantics, not an externally independent evaluator. "
            "Its duration-stratified oracle recall is computed on final "
            "detections, not on pre-threshold proposals. Cross-checkpoint "
            "differences and differences-in-differences are descriptive, not "
            "causal training effects. It "
            "does not isolate training assignment, establish multi-seed or "
            "cross-dataset robustness, prove compute savings, or support a "
            "paper-ready claim."
        ),
    }


def main():
    args = parse_args()
    completion = validate_suite(args.run_root)
    atomic_write_json(args.output, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
