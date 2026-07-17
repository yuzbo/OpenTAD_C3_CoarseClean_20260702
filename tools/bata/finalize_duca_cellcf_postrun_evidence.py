from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence


VARIANTS = ("uniform", "transition_beta0", "cellcf")
JOB_KEYS = (
    "convergence_uniform",
    "convergence_transition_beta0",
    "convergence_cellcf",
    "convergence_summary",
    "training_cost",
    "completion",
)
FORMAL_JOB_KEYS = (
    "uniform",
    "transition_beta0",
    "cellcf",
    "aggregate",
    "cost",
    "completion",
)
INTENT_SCHEMA = "duca_cellcf_postrun_submission_intent_v1"
MANIFEST_SCHEMA = "duca_cellcf_postrun_submission_manifest_v1"
RECEIPT_SCHEMA = "duca_cellcf_postrun_slurm_receipt_v1"
CONVERGENCE_SCHEMA = "duca_cellcf_fixed_convergence_trajectory_v1"
TRAINING_COST_SCHEMA = "duca_cellcf_training_cost_summary_v1"
FINAL_SCHEMA = "duca_cellcf_postrun_evidence_completion_v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _validate_embedded_hash(
    payload: Mapping[str, Any], key: str, label: str
) -> None:
    observed = payload.get(key)
    unsigned = dict(payload)
    unsigned.pop(key, None)
    _require(
        isinstance(observed, str)
        and re.fullmatch(r"[0-9a-f]{64}", observed) is not None
        and observed == canonical_sha256(unsigned),
        f"{label} canonical hash mismatch",
    )


def _require_hash(value: Any, label: str) -> str:
    normalized = str(value or "")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", normalized) is not None,
        f"{label} is not a SHA256",
    )
    return normalized


def _require_commit(value: str, label: str) -> str:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", value) is not None,
        f"{label} is not a full git commit",
    )
    return value


def _require_under(path: Path, root: Path, label: str) -> None:
    _require(path != root and root in path.parents, f"{label} escaped {root}")


def _default_aggregate_loader(**kwargs: Any) -> Mapping[str, Any]:
    from tools.bata.duca_cellcf_suite_binding import load_suite_aggregate_binding

    return load_suite_aggregate_binding(**kwargs)


def _default_final_suite_revalidator(**kwargs: Any) -> Mapping[str, Any]:
    from tools.bata.validate_duca_cellcf_suite import validate_suite

    return validate_suite(**kwargs)


def _default_convergence_rebuilder(**kwargs: Any) -> Mapping[str, Any]:
    from tools.bata.summarize_duca_cellcf_convergence import (
        build_convergence_evidence,
    )

    return build_convergence_evidence(**kwargs)


def _default_training_cost_rebuilder(**kwargs: Any) -> Mapping[str, Any]:
    from tools.bata.summarize_duca_cellcf_training_cost import (
        summarize_training_cost,
    )

    return summarize_training_cost(**kwargs)


def _default_scheduler_validator(**kwargs: Any) -> Mapping[str, Any]:
    from tools.bata.validate_duca_cellcf_slurm_receipt import (
        validate_slurm_receipt,
    )

    return validate_slurm_receipt(**kwargs)


def _default_formal_completion_validator(
    *, job_id: int, job_name: str, cluster: str
) -> Mapping[str, Any]:
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-M",
            cluster,
            "-j",
            str(job_id),
            "-n",
            "-P",
            "-o",
            "JobIDRaw,JobName%128,Cluster,State,ExitCode",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    _require(
        result.returncode == 0,
        f"formal completion sacct query failed: {result.stderr.strip()}",
    )
    matches = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 5 and fields[0] == str(job_id):
            matches.append(fields[:5])
    expected = [[str(job_id), job_name, cluster, "COMPLETED", "0:0"]]
    _require(matches == expected, "formal completion is not uniquely COMPLETED/0:0")
    return {
        "ok": True,
        "job_id": job_id,
        "job_name": job_name,
        "cluster": cluster,
        "state": "COMPLETED",
        "exit_code": "0:0",
    }


def _default_repository_validator(root: Path, expected_commit: str) -> None:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(
                f"git validation failed for {root}: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    _require(git("rev-parse", "HEAD") == expected_commit, f"repository commit drift: {root}")
    _require(
        not git("status", "--porcelain", "--untracked-files=normal"),
        f"repository is dirty: {root}",
    )
    ignored = git(
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "*.py",
        "*.pth",
        "sitecustomize.py",
        "usercustomize.py",
    )
    _require(not ignored, f"ignored Python source could shadow repository: {root}")


def _load_ledger(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    _require(
        [row.get("job_key") for row in rows] == list(JOB_KEYS),
        "post-run ledger does not bind the exact ordered six-job DAG",
    )
    return rows


def _load_formal_completion(
    path: Path,
    *,
    expected_commit: str,
    expected_seed: int,
    expected_profile: str,
) -> dict[str, str]:
    _require(path.is_file(), "formal submitted-job ledger is missing")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    _require(
        [row.get("job_key") for row in rows] == list(FORMAL_JOB_KEYS),
        "formal submitted-job ledger does not bind the exact six-job DAG",
    )
    for row in rows:
        _require(
            row.get("commit") == expected_commit,
            f"formal {row.get('job_key')} commit mismatch",
        )
        _require(
            row.get("seed") == str(expected_seed),
            f"formal {row.get('job_key')} seed mismatch",
        )
        _require(
            row.get("training_profile") == expected_profile,
            f"formal {row.get('job_key')} training profile mismatch",
        )
    clusters = {row.get("cluster") for row in rows}
    _require(
        len(clusters) == 1 and None not in clusters and "" not in clusters,
        "formal target cluster is ambiguous",
    )
    ids = {}
    for row in rows:
        job_id = str(row.get("job_id") or "")
        _require(
            re.fullmatch(r"[1-9][0-9]*", job_id) is not None,
            f"formal {row.get('job_key')} job id is invalid",
        )
        ids[str(row["job_key"])] = job_id
    _require(
        len(set(ids.values())) == len(ids),
        "formal submitted-job ledger contains duplicate job ids",
    )
    expected_dependencies = {
        "uniform": "none",
        "transition_beta0": "none",
        "cellcf": "none",
        "aggregate": (
            "afterok:"
            + ":".join(
                ids[key] for key in ("uniform", "transition_beta0", "cellcf")
            )
        ),
        "cost": f"afterok:{ids['aggregate']}",
        "completion": f"afterok:{ids['aggregate']}:{ids['cost']}",
    }
    for row in rows:
        _require(
            row.get("dependency") == expected_dependencies[str(row["job_key"])],
            f"formal {row.get('job_key')} dependency mismatch",
        )
    completion = dict(rows[-1])
    _require(completion.get("job_key") == "completion", "formal completion row is missing")
    _require(bool(completion.get("job_name")), "formal completion job name is missing")
    return completion


def _expected_dependencies(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    ids = {}
    for row in rows:
        job_id = str(row.get("job_id") or "")
        _require(
            re.fullmatch(r"[1-9][0-9]*", job_id) is not None,
            f"{row.get('job_key')} has an invalid job id",
        )
        ids[str(row["job_key"])] = job_id
    _require(len(set(ids.values())) == len(ids), "post-run job ids are not unique")
    return {
        "convergence_uniform": "none",
        "convergence_transition_beta0": "none",
        "convergence_cellcf": "none",
        "convergence_summary": (
            "afterok:"
            + ":".join(
                ids[key]
                for key in (
                    "convergence_uniform",
                    "convergence_transition_beta0",
                    "convergence_cellcf",
                )
            )
        ),
        "training_cost": "none",
        "completion": (
            f"afterok:{ids['convergence_summary']}:{ids['training_cost']}"
        ),
    }


def _validate_receipt(
    path_value: str,
    expected_sha: str,
    *,
    expected_status: str,
    row: Mapping[str, str],
    intent_path: Path,
    intent_sha: str,
    trained_commit: str,
    evidence_commit: str,
    aggregate_sha: str,
    submitted_path: Path | None = None,
    submitted_sha: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    path, payload = _load_json(path_value, f"{row['job_key']} {expected_status} receipt")
    _require(sha256_file(path) == expected_sha, f"{row['job_key']} receipt hash mismatch")
    _validate_embedded_hash(payload, "artifact_sha256", f"{row['job_key']} receipt")
    expected = {
        "schema": RECEIPT_SCHEMA,
        "status": expected_status,
        "task": "offline_temporal_action_detection",
        "job_key": row["job_key"],
        "job_id": int(row["job_id"]),
        "job_name": row["job_name"],
        "cluster": row["cluster"],
        "dependency": None if row["dependency"] == "none" else row["dependency"],
        "submission_token": row["submission_token"],
        "job_file": str(Path(row["job_file"]).resolve()),
        "job_file_sha256": row["job_file_sha256"],
        "trained_git_commit": trained_commit,
        "evidence_git_commit": evidence_commit,
        "aggregate_suite_evidence_sha256": aggregate_sha,
        "submission_intent": str(intent_path),
        "submission_intent_sha256": intent_sha,
    }
    for key, value in expected.items():
        _require(payload.get(key) == value, f"{row['job_key']} receipt mismatch: {key}")
    raw = str(payload.get("raw_sbatch_response") or "").strip()
    _require(
        raw.splitlines()[0] == f"{row['job_id']};{row['cluster']}",
        f"{row['job_key']} receipt raw sbatch response mismatch",
    )
    if expected_status == "SUBMITTED_UNVERIFIED":
        _require(payload.get("scheduler_validation") is None, "unverified receipt has validation")
        _require(payload.get("submitted_receipt") is None, "unverified receipt is self-linked")
    else:
        validation = payload.get("scheduler_validation")
        _require(
            isinstance(validation, Mapping)
            and validation.get("ok") is True
            and int(validation.get("job_id", -1)) == int(row["job_id"]),
            f"{row['job_key']} verified receipt lacks scheduler proof",
        )
        _require(
            submitted_path is not None
            and payload.get("submitted_receipt") == str(submitted_path)
            and payload.get("submitted_receipt_sha256") == submitted_sha,
            f"{row['job_key']} verified receipt lost its submitted receipt",
        )
    return path, payload


def finalize_postrun_evidence(
    *,
    run_root: str | Path,
    control_root: str | Path,
    trained_repo_root: str | Path,
    trained_commit: str,
    evidence_repo_root: str | Path,
    evidence_commit: str,
    aggregate_path: str | Path,
    aggregate_sha256: str,
    final_suite_path: str | Path,
    final_suite_sha256: str,
    aggregate_loader: Callable[..., Mapping[str, Any]] = _default_aggregate_loader,
    final_suite_revalidator: Callable[..., Mapping[str, Any]] = (
        _default_final_suite_revalidator
    ),
    convergence_rebuilder: Callable[..., Mapping[str, Any]] = (
        _default_convergence_rebuilder
    ),
    training_cost_rebuilder: Callable[..., Mapping[str, Any]] = (
        _default_training_cost_rebuilder
    ),
    scheduler_validator: Callable[..., Mapping[str, Any]] = (
        _default_scheduler_validator
    ),
    formal_completion_validator: Callable[..., Mapping[str, Any]] = (
        _default_formal_completion_validator
    ),
    repository_validator: Callable[[Path, str], None] = (
        _default_repository_validator
    ),
) -> dict[str, Any]:
    trained_commit = _require_commit(trained_commit, "trained commit")
    evidence_commit = _require_commit(evidence_commit, "evidence commit")
    aggregate_sha256 = _require_hash(aggregate_sha256, "aggregate evidence hash")
    final_suite_sha256 = _require_hash(final_suite_sha256, "final suite hash")
    run_root_path = Path(run_root).expanduser().resolve()
    control_root_path = Path(control_root).expanduser().resolve()
    trained_root = Path(trained_repo_root).expanduser().resolve()
    evidence_root = Path(evidence_repo_root).expanduser().resolve()
    _require(run_root_path.is_dir(), "formal run root is missing")
    _require(control_root_path.is_dir(), "post-run control root is missing")
    _require_under(control_root_path, run_root_path, "post-run control root")
    _require(trained_root.is_dir(), "trained repository is missing")
    _require(evidence_root.is_dir(), "evidence repository is missing")
    repository_validator(trained_root, trained_commit)
    repository_validator(evidence_root, evidence_commit)

    aggregate_file, aggregate_payload = _load_json(
        aggregate_path, "aggregate suite evidence"
    )
    final_file, final_payload = _load_json(final_suite_path, "final suite evidence")
    _require(
        sha256_file(aggregate_file) == aggregate_sha256,
        "aggregate suite evidence hash mismatch",
    )
    _require(
        sha256_file(final_file) == final_suite_sha256,
        "final suite evidence hash mismatch",
    )
    post_run_paths = {
        variant: run_root_path / "logs" / variant / "post_run_evidence.json"
        for variant in VARIANTS
    }
    aggregate_binding = aggregate_loader(
        path=aggregate_file,
        expected_sha256=aggregate_sha256,
        expected_commit=trained_commit,
        expected_profile="exposure132",
        post_run_paths=post_run_paths,
    )
    _require(aggregate_binding.get("seed") == aggregate_payload.get("seed"), "aggregate seed drift")
    formal_completion = _load_formal_completion(
        run_root_path / "jobs.submitted.tsv",
        expected_commit=trained_commit,
        expected_seed=int(aggregate_binding["seed"]),
        expected_profile="exposure132",
    )
    formal_completion_scheduler = formal_completion_validator(
        job_id=int(formal_completion["job_id"]),
        job_name=formal_completion["job_name"],
        cluster=formal_completion["cluster"],
    )
    _require(
        formal_completion_scheduler.get("ok") is True,
        "formal completion scheduler revalidation failed",
    )

    cost_record = final_payload.get("cost_evidence")
    _require(
        final_payload.get("schema") == "duca_cellcf_suite_manifest_v1"
        and final_payload.get("ok") is True
        and final_payload.get("status") == "complete"
        and final_payload.get("task") == "offline_temporal_action_detection"
        and final_payload.get("git_commit") == trained_commit
        and final_payload.get("training_profile") == "exposure132"
        and final_payload.get("seed") == aggregate_binding["seed"]
        and final_payload.get("cost_evidence_required") is True
        and isinstance(cost_record, Mapping)
        and cost_record.get("validated") is True,
        "final suite evidence has invalid completion semantics",
    )
    regenerated_final = final_suite_revalidator(
        repo_root=trained_root,
        seed=int(aggregate_binding["seed"]),
        expected_commit=trained_commit,
        require_clean=True,
        gate_json=aggregate_binding["real_loader_gate"]["path"],
        pilot_json=aggregate_binding["ddp_pilot"]["path"],
        post_run_evidence=post_run_paths,
        cost_evidence=cost_record["path"],
        require_cost_evidence=True,
    )
    _require(regenerated_final == final_payload, "final suite evidence is not reproducible")

    intent_path, intent = _load_json(
        control_root_path / "submission_intent.json", "post-run submission intent"
    )
    _validate_embedded_hash(intent, "artifact_sha256", "post-run submission intent")
    intent_sha = sha256_file(intent_path)
    postrun_output_root = Path(str(intent.get("postrun_output_root") or "")).resolve()
    _require(
        postrun_output_root == control_root_path / "artifacts",
        "post-run output root is not the versioned control artifact root",
    )
    intent_cluster = str(intent.get("target_cluster") or "")
    _require(
        re.fullmatch(r"[A-Za-z0-9._-]+", intent_cluster) is not None,
        "submission intent target cluster is invalid",
    )
    expected_intent = {
        "schema": INTENT_SCHEMA,
        "status": "INTENT_RECORDED",
        "task": "offline_temporal_action_detection",
        "formal_run_root": str(run_root_path),
        "trained_repository": str(trained_root),
        "trained_git_commit": trained_commit,
        "evidence_repository": str(evidence_root),
        "evidence_git_commit": evidence_commit,
        "target_cluster": intent_cluster,
        "aggregate_suite_evidence_path": str(aggregate_file),
        "aggregate_suite_evidence_sha256": aggregate_sha256,
        "final_suite_evidence_path": str(final_file),
        "final_suite_evidence_sha256": final_suite_sha256,
        "postrun_output_root": str(postrun_output_root),
    }
    for key, value in expected_intent.items():
        _require(intent.get(key) == value, f"submission intent mismatch: {key}")
    intent_jobs = intent.get("jobs")
    _require(
        isinstance(intent_jobs, list)
        and [record.get("job_key") for record in intent_jobs] == list(JOB_KEYS),
        "submission intent job set/order mismatch",
    )

    ledger_path = control_root_path / "jobs.submitted.tsv"
    _require(ledger_path.is_file(), "post-run submitted ledger is missing")
    ledger_sha = sha256_file(ledger_path)
    rows = _load_ledger(ledger_path)
    expected_dependencies = _expected_dependencies(rows)
    clusters = {row.get("cluster") for row in rows}
    _require(len(clusters) == 1 and None not in clusters, "post-run cluster identity is ambiguous")
    target_cluster = next(iter(clusters))
    _require(
        expected_intent["target_cluster"] == target_cluster,
        "submission intent target cluster mismatch",
    )
    _require(
        formal_completion["cluster"] == target_cluster,
        "formal and post-run jobs target different clusters",
    )
    intent_by_key = {str(record["job_key"]): record for record in intent_jobs}
    scheduler_records = []
    receipt_records = []
    for row in rows:
        key = row["job_key"]
        _require(row["dependency"] == expected_dependencies[key], f"{key} dependency mismatch")
        expected_row = {
            "trained_commit": trained_commit,
            "evidence_commit": evidence_commit,
            "aggregate_sha256": aggregate_sha256,
            "submission_intent_sha256": intent_sha,
        }
        for field, value in expected_row.items():
            _require(row.get(field) == value, f"{key} ledger mismatch: {field}")
        intent_job = intent_by_key[key]
        for field in (
            "job_name",
            "submission_token",
            "job_file",
            "job_file_sha256",
        ):
            _require(row.get(field) == str(intent_job.get(field)), f"{key} intent mismatch: {field}")
        job_file = Path(row["job_file"]).resolve()
        _require(
            job_file == control_root_path / "jobs" / f"{key}.sbatch"
            and job_file.is_file()
            and sha256_file(job_file) == row["job_file_sha256"],
            f"{key} bound job file changed",
        )
        _require(
            Path(row["submitted_receipt"]).resolve()
            == control_root_path / "receipts" / f"{key}.submitted.json"
            and Path(row["verified_receipt"]).resolve()
            == control_root_path / "receipts" / f"{key}.verified.json",
            f"{key} receipt path mismatch",
        )
        submitted_path, _ = _validate_receipt(
            row["submitted_receipt"],
            row["submitted_receipt_sha256"],
            expected_status="SUBMITTED_UNVERIFIED",
            row=row,
            intent_path=intent_path,
            intent_sha=intent_sha,
            trained_commit=trained_commit,
            evidence_commit=evidence_commit,
            aggregate_sha=aggregate_sha256,
        )
        verified_path, _ = _validate_receipt(
            row["verified_receipt"],
            row["verified_receipt_sha256"],
            expected_status="VERIFIED",
            row=row,
            intent_path=intent_path,
            intent_sha=intent_sha,
            trained_commit=trained_commit,
            evidence_commit=evidence_commit,
            aggregate_sha=aggregate_sha256,
            submitted_path=submitted_path,
            submitted_sha=row["submitted_receipt_sha256"],
        )
        scheduler = scheduler_validator(
            job_id=int(row["job_id"]),
            job_name=row["job_name"],
            token=row["submission_token"],
            cluster=row["cluster"],
            job_file=job_file,
            job_file_sha256=row["job_file_sha256"],
            dependency="" if row["dependency"] == "none" else row["dependency"],
            require_scheduler_script=False,
        )
        _require(scheduler.get("ok") is True, f"{key} scheduler revalidation failed")
        scheduler_records.append(dict(scheduler))
        receipt_records.append(
            {
                "job_key": key,
                "submitted": {
                    "path": str(submitted_path),
                    "sha256": row["submitted_receipt_sha256"],
                },
                "verified": {
                    "path": str(verified_path),
                    "sha256": row["verified_receipt_sha256"],
                },
            }
        )

    manifest_path, manifest = _load_json(
        control_root_path / "submission_manifest.json",
        "post-run submission manifest",
    )
    _validate_embedded_hash(manifest, "artifact_sha256", "post-run submission manifest")
    expected_manifest = {
        "schema": MANIFEST_SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "training_profile": "exposure132",
        "formal_run_root": str(run_root_path),
        "trained_repository": str(trained_root),
        "trained_git_commit": trained_commit,
        "evidence_repository": str(evidence_root),
        "evidence_git_commit": evidence_commit,
        "target_cluster": target_cluster,
        "aggregate_suite_evidence_path": str(aggregate_file),
        "aggregate_suite_evidence_sha256": aggregate_sha256,
        "final_suite_evidence_path": str(final_file),
        "final_suite_evidence_sha256": final_suite_sha256,
        "postrun_output_root": str(postrun_output_root),
        "submission_intent_path": str(intent_path),
        "submission_intent_sha256": intent_sha,
        "formal_completion_job_id": int(
            formal_completion["job_id"]
        ),
        "jobs_ledger_path": str(ledger_path),
        "jobs_ledger_sha256": ledger_sha,
        "jobs": rows,
    }
    for key, value in expected_manifest.items():
        _require(manifest.get(key) == value, f"submission manifest mismatch: {key}")

    convergence_path, convergence = _load_json(
        postrun_output_root / "convergence" / "fixed_trajectory.json",
        "fixed convergence trajectory",
    )
    training_cost_path, training_cost = _load_json(
        postrun_output_root / "training_cost" / "training_cost_summary.json",
        "training cost summary",
    )
    for payload, schema, label in (
        (convergence, CONVERGENCE_SCHEMA, "fixed convergence trajectory"),
        (training_cost, TRAINING_COST_SCHEMA, "training cost summary"),
    ):
        _require(
            payload.get("schema") == schema
            and payload.get("ok") is True
            and payload.get("task") == "offline_temporal_action_detection"
            and payload.get("git_commit") == trained_commit
            and payload.get("evidence_git_commit") == evidence_commit,
            f"{label} identity/status mismatch",
        )
        _validate_embedded_hash(payload, "artifact_sha256", label)
    _require(
        training_cost.get("training_profile") == "exposure132",
        "training cost profile mismatch",
    )
    _require(
        convergence.get("variants") == list(VARIANTS)
        and convergence.get("fixed_epochs") == [59, 89, 131]
        and convergence.get("primary_epoch") == 131
        and convergence.get("primary_state_key") == "state_dict_ema",
        "convergence trajectory protocol mismatch",
    )

    convergence_root = postrun_output_root / "convergence"
    evaluation_paths = {}
    for variant in VARIANTS:
        evaluation_paths[(variant, 59)] = (
            convergence_root / variant / "epoch_59" / "evaluation.json"
        )
        evaluation_paths[(variant, 89)] = (
            convergence_root / variant / "epoch_89" / "evaluation.json"
        )
        evaluation_paths[(variant, 131)] = (
            run_root_path / "logs" / variant / "terminal_evaluation.json"
        )
    rebuilt_convergence = convergence_rebuilder(
        expected_commit=trained_commit,
        expected_evidence_commit=evidence_commit,
        suite_aggregate_path=aggregate_file,
        suite_aggregate_sha256=aggregate_sha256,
        post_run_paths=post_run_paths,
        variant_receipt_paths={
            variant: convergence_root / variant / "variant_complete.json"
            for variant in VARIANTS
        },
        evaluation_paths=evaluation_paths,
    )
    _require(
        rebuilt_convergence == convergence,
        "fixed convergence trajectory is not reproducible",
    )
    rebuilt_training_cost = training_cost_rebuilder(
        expected_commit=trained_commit,
        expected_evidence_commit=evidence_commit,
        suite_aggregate_path=aggregate_file,
        suite_aggregate_sha256=aggregate_sha256,
        post_run_paths=post_run_paths,
        slurm_cost_paths={
            variant: (
                postrun_output_root
                / "training_cost"
                / f"{variant}.slurm_cost.json"
            )
            for variant in VARIANTS
        },
    )
    _require(
        rebuilt_training_cost == training_cost,
        "training cost summary is not reproducible",
    )
    for row in convergence.get("rows", []):
        metrics = row.get("metrics", row)
        for key, value in metrics.items():
            if key == "variant" or isinstance(value, str):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _require(math.isfinite(float(value)), "convergence contains non-finite metrics")

    payload = {
        "schema": FINAL_SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "training_profile": "exposure132",
        "trained_git_commit": trained_commit,
        "evidence_git_commit": evidence_commit,
        "aggregate_suite_evidence": {
            "path": str(aggregate_file),
            "sha256": aggregate_sha256,
        },
        "final_suite_evidence": {
            "path": str(final_file),
            "sha256": final_suite_sha256,
        },
        "submission_intent": {
            "path": str(intent_path),
            "sha256": intent_sha,
        },
        "submission_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "jobs_ledger": {"path": str(ledger_path), "sha256": ledger_sha},
        "receipts": receipt_records,
        "scheduler_revalidation": scheduler_records,
        "formal_completion_scheduler_revalidation": dict(
            formal_completion_scheduler
        ),
        "artifacts": {
            "convergence": {
                "path": str(convergence_path),
                "sha256": sha256_file(convergence_path),
            },
            "training_cost": {
                "path": str(training_cost_path),
                "sha256": sha256_file(training_cost_path),
            },
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(
        target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--trained-repo-root", required=True)
    parser.add_argument("--trained-commit", required=True)
    parser.add_argument("--evidence-repo-root", required=True)
    parser.add_argument("--evidence-commit", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--aggregate-sha256", required=True)
    parser.add_argument("--final-suite", required=True)
    parser.add_argument("--final-suite-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = finalize_postrun_evidence(
        run_root=args.run_root,
        control_root=args.control_root,
        trained_repo_root=args.trained_repo_root,
        trained_commit=args.trained_commit,
        evidence_repo_root=args.evidence_repo_root,
        evidence_commit=args.evidence_commit,
        aggregate_path=args.aggregate,
        aggregate_sha256=args.aggregate_sha256,
        final_suite_path=args.final_suite,
        final_suite_sha256=args.final_suite_sha256,
    )
    _exclusive_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
