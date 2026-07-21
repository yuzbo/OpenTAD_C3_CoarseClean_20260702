from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_contract import (
    canonical_sha256,
    finalize_self_hash,
)
from tools.bata.continuous_roi_s2_training import (
    S2_FAMILIES,
    S2_TRAINING_SEEDS,
    build_checkpoint_validation_runtime,
    current_git_commit,
    load_pure_data_config,
    require_clean_git_checkout,
    validate_training_completion_with_audit,
)
from tools.bata.spatial_zoom_s1_contract import sha256_file

S2_TRAINING_MATRIX_COMPLETION_SCHEMA = "continuous_roi_s2_training_matrix_completion_v1"
DEPLOYMENT_SCHEMA = "continuous_roi_s2_training_deployment_v2"
INTENT_SCHEMA = "continuous_roi_s2_training_deployment_intent_v2"
JOB_RECEIPT_SCHEMA = "continuous_roi_s2_training_job_submission_v2"
CELL_INTENT_SCHEMA = "continuous_roi_s2_training_cell_intent_v2"
EXPECTED_MATRIX = tuple(
    (family, seed) for family in S2_FAMILIES for seed in S2_TRAINING_SEEDS
)
VALIDATOR_SOURCE_PATHS = (
    Path("tools/bata/continuous_roi_s2_contract.py"),
    Path("tools/bata/continuous_roi_s2_training.py"),
    Path("tools/bata/finalize_continuous_roi_s2_training_matrix.py"),
    Path("tools/bata/spatial_zoom_s1_contract.py"),
)


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return dict(payload)


def _validate_self_hash(
    payload: Mapping[str, Any], hash_key: str, *, schema: str | None = None
) -> dict[str, Any]:
    checked = dict(payload)
    observed = checked.pop(hash_key, None)
    if not observed or canonical_sha256(checked) != observed:
        raise ValueError(f"invalid {hash_key}")
    checked[hash_key] = observed
    if schema is not None and checked.get("schema_version") != schema:
        raise ValueError(
            f"unsupported evidence schema: {checked.get('schema_version')}"
        )
    return checked


def _tracked_source_identity() -> tuple[str, dict[str, str]]:
    commit = current_git_commit(ROOT)
    require_clean_git_checkout(expected_commit=commit, repository_root=ROOT)
    identities: dict[str, str] = {}
    for relative_path in VALIDATOR_SOURCE_PATHS:
        path = (ROOT / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative_path.as_posix()],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"validator source is not tracked: {relative_path}")
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative_path.as_posix()}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if blob.returncode != 0:
            raise RuntimeError(f"cannot read validator Git blob: {relative_path}")
        blob_hash = hashlib.sha256(blob.stdout).hexdigest()
        if sha256_file(path) != blob_hash:
            raise ValueError(f"validator source differs from Git: {relative_path}")
        identities[relative_path.as_posix()] = blob_hash
    return commit, identities


def _load_evidence(path: Path, *, hash_key: str, schema: str) -> dict[str, Any]:
    return _validate_self_hash(_load_json(path), hash_key, schema=schema)


def _validate_deployment_chain(
    deployment_summary_path: Path, *, expected_deployment_sha256: str
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    Path,
]:
    deployment_summary_path = deployment_summary_path.resolve()
    deployment = _load_evidence(
        deployment_summary_path,
        hash_key="deployment_sha256",
        schema=DEPLOYMENT_SCHEMA,
    )
    if deployment["deployment_sha256"] != expected_deployment_sha256:
        raise ValueError("deployment receipt differs from the externally frozen hash")
    if (
        deployment.get("status") != "SUBMITTED"
        or deployment.get("official_test_opened") is not False
        or deployment.get("paper_claim_allowed") is not False
    ):
        raise ValueError("invalid Continuous-RoI S2 training deployment receipt")

    canonical_root = Path(str(deployment["canonical_experiment_root"])).resolve()
    control_dir = canonical_root / "control"
    if deployment_summary_path != control_dir / "deployment_summary.json":
        raise ValueError("deployment summary is not at its canonical campaign path")
    intent_path = Path(str(deployment["intent_path"])).resolve()
    if intent_path != control_dir / "deployment_intent.json":
        raise ValueError("deployment intent is not at its canonical campaign path")
    intent = _load_evidence(intent_path, hash_key="intent_sha256", schema=INTENT_SCHEMA)
    shared_fields = (
        "code_commit",
        "base_experiment_namespace",
        "campaign_namespace",
        "canonical_experiment_root",
        "full_model_gate_sha256",
        "training_runtime_precheck_sha256",
        "runtime_authorization_sha256",
        "submission_environment",
    )
    if (
        intent["intent_sha256"] != deployment["intent_sha256"]
        or any(intent.get(field) != deployment.get(field) for field in shared_fields)
        or intent.get("status") != "SUBMISSION_INTENT"
        or intent["official_test_open_allowed"] is not False
        or intent["paper_claim_allowed"] is not False
    ):
        raise ValueError("deployment intent and summary are inconsistent")

    intent_cells = {
        (str(cell["family"]).upper(), int(cell["seed"])): dict(cell)
        for cell in intent.get("cells", [])
    }
    raw_jobs = deployment.get("jobs")
    if (
        not isinstance(raw_jobs, list)
        or len(raw_jobs) != len(EXPECTED_MATRIX)
        or set(intent_cells) != set(EXPECTED_MATRIX)
    ):
        raise ValueError("deployment does not contain the exact-nine matrix")

    jobs: dict[tuple[str, int], dict[str, Any]] = {}
    cell_intents: dict[tuple[str, int], dict[str, Any]] = {}
    for raw_job in raw_jobs:
        embedded = dict(raw_job)
        key = (str(embedded.get("family", "")).upper(), int(embedded.get("seed", -1)))
        if key in jobs or key not in EXPECTED_MATRIX:
            raise ValueError(f"invalid or duplicate matrix job: {key}")
        family, seed = key
        receipt_path = (
            control_dir / "job_receipts" / f"{family.lower()}_seed{seed}.json"
        )
        receipt = _load_evidence(
            receipt_path,
            hash_key="submission_sha256",
            schema=JOB_RECEIPT_SCHEMA,
        )
        if receipt != embedded:
            raise ValueError(f"embedded job receipt differs from disk: {key}")
        cell_intent_path = Path(str(receipt["cell_intent_path"])).resolve()
        expected_cell_intent_path = receipt_path.with_name(
            f"{family.lower()}_seed{seed}.intent.json"
        )
        if cell_intent_path != expected_cell_intent_path:
            raise ValueError(f"cell intent path is not canonical: {key}")
        cell_intent = _load_evidence(
            cell_intent_path,
            hash_key="cell_intent_sha256",
            schema=CELL_INTENT_SCHEMA,
        )
        token_core = {
            field: value
            for field, value in cell_intent.items()
            if field not in {"job_token", "cell_intent_sha256"}
        }
        expected_token = canonical_sha256(token_core)
        intent_cell = intent_cells[key]
        if (
            cell_intent["job_token"] != expected_token
            or receipt["job_token"] != expected_token
            or receipt["cell_intent_sha256"] != cell_intent["cell_intent_sha256"]
            or receipt["intent_sha256"] != intent["intent_sha256"]
            or receipt["campaign_namespace"] != deployment["campaign_namespace"]
            or receipt["base_experiment_namespace"]
            != deployment["base_experiment_namespace"]
            or receipt.get("status") != "SUBMITTED"
            or receipt["family"] != family
            or int(receipt["seed"]) != seed
            or cell_intent["family"] != family
            or int(cell_intent["seed"]) != seed
            or cell_intent["campaign_namespace"] != deployment["campaign_namespace"]
            or cell_intent["job_name"] != receipt["job_name"]
            or cell_intent["work_dir"] != receipt["work_dir"]
            or cell_intent["deployment_intent_sha256"] != intent["intent_sha256"]
            or intent_cell["job_name"] != receipt["job_name"]
            or intent_cell["work_dir"] != receipt["work_dir"]
            or intent_cell["bound_config_sha256"] != cell_intent["bound_config_sha256"]
        ):
            raise ValueError(f"job/cell/deployment evidence chain differs: {key}")
        jobs[key] = receipt
        cell_intents[key] = cell_intent

    if set(jobs) != set(EXPECTED_MATRIX):
        raise ValueError("deployment job set differs from the exact-nine matrix")
    jobs_path = Path(str(deployment["jobs_tsv"])).resolve()
    if jobs_path != control_dir / "jobs.tsv":
        raise ValueError("jobs.tsv is not at its canonical campaign path")
    expected_lines = ["family\tseed\tjob_id\tjob_name\tjob_token\twork_dir"]
    for key in EXPECTED_MATRIX:
        receipt = jobs[key]
        expected_lines.append(
            "\t".join(
                str(receipt[field])
                for field in (
                    "family",
                    "seed",
                    "job_id",
                    "job_name",
                    "job_token",
                    "work_dir",
                )
            )
        )
    if jobs_path.read_text(encoding="utf-8") != "\n".join(expected_lines) + "\n":
        raise ValueError("jobs.tsv differs from the exact job receipts")
    return deployment, intent, jobs, cell_intents, control_dir


def query_sacct(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if len(job_ids) != len(set(job_ids)) or not job_ids:
        raise ValueError("Slurm job IDs must be unique and non-empty")
    completed = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Comment",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: dict[str, dict[str, str]] = {}
    for raw_line in completed.stdout.splitlines():
        fields = raw_line.strip().split("|")
        if len(fields) < 6 or fields[0] not in job_ids:
            continue
        if fields[0] in rows:
            raise ValueError(f"duplicate primary sacct row for job {fields[0]}")
        rows[fields[0]] = {
            "job_name": fields[1],
            "state": fields[2].split()[0],
            "exit_code": fields[3],
            "elapsed": fields[4],
            "comment": fields[5],
        }
    if set(rows) != set(job_ids):
        raise ValueError("sacct did not return every exact matrix job")
    return rows


def validate_accounting_rows(
    job_ids: list[str],
    accounting: Mapping[str, Mapping[str, str]],
    *,
    expected_jobs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, str]]:
    if set(accounting) != set(job_ids):
        raise ValueError("Slurm accounting set does not match the exact matrix")
    checked: dict[str, dict[str, str]] = {}
    for job_id in job_ids:
        row = dict(accounting[job_id])
        if row.get("state") != "COMPLETED" or row.get("exit_code") != "0:0":
            raise ValueError(f"matrix job {job_id} did not complete successfully")
        elapsed = str(row.get("elapsed", ""))
        if not elapsed:
            raise ValueError(f"matrix job {job_id} has no elapsed time")
        expected = None if expected_jobs is None else dict(expected_jobs[job_id])
        if expected is not None and (
            row.get("job_name") != expected["job_name"]
            or row.get("comment") != f"crs2:{expected['job_token']}"
        ):
            raise ValueError(f"Slurm identity differs from job receipt: {job_id}")
        checked[job_id] = {
            "job_name": str(row.get("job_name", "")),
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed": elapsed,
            "comment": str(row.get("comment", "")),
        }
    return checked


def _checkpoint_inventory(checkpoint_dir: Path) -> list[str]:
    inventory = sorted(path.name for path in checkpoint_dir.iterdir() if path.is_file())
    expected = ["epoch_59.pth", "epoch_59.pth.metadata.json"]
    if inventory != expected:
        raise ValueError(
            f"unexpected final checkpoint inventory in {checkpoint_dir}: {inventory}"
        )
    return inventory


def _validate_job_log_binding(
    *,
    control_dir: Path,
    receipt: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, str]:
    job_id = str(receipt["job_id"])
    job_name = str(receipt["job_name"])
    stdout_path = control_dir / "logs" / f"{job_name}-{job_id}.out"
    stderr_path = control_dir / "logs" / f"{job_name}-{job_id}.err"
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise FileNotFoundError(
            stdout_path if not stdout_path.is_file() else stderr_path
        )
    stdout = stdout_path.read_text(encoding="utf-8")
    expected_pass = (
        "[CONTINUOUS_ROI_S2_TRAIN] PASS "
        f"family={receipt['family']} seed={receipt['seed']} "
        f"completion={Path(str(receipt['work_dir'])) / 'training_completion.json'}"
    )
    required_tokens = (
        expected_pass,
        f'"checkpoint_sha256": "{completion["checkpoint_sha256"]}"',
        f'"completion_sha256": "{completion["completion_sha256"]}"',
        f'"code_commit": "{completion["code_commit"]}"',
    )
    if any(token not in stdout for token in required_tokens):
        raise ValueError(
            f"Slurm stdout does not bind the completion artifact: {job_id}"
        )
    return {
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
    }


def build_training_matrix_completion(
    *,
    deployment_summary_path: str | Path,
    expected_deployment_sha256: str,
) -> dict[str, Any]:
    validation_commit, validation_sources = _tracked_source_identity()
    deployment_summary_path = Path(deployment_summary_path).resolve()
    deployment, intent, jobs, cell_intents, control_dir = _validate_deployment_chain(
        deployment_summary_path,
        expected_deployment_sha256=expected_deployment_sha256,
    )
    job_ids = [str(jobs[key]["job_id"]) for key in EXPECTED_MATRIX]
    expected_jobs = {str(job["job_id"]): job for job in jobs.values()}
    checked_accounting = validate_accounting_rows(
        job_ids,
        query_sacct(job_ids),
        expected_jobs=expected_jobs,
    )

    cells = []
    validation_runtimes: dict[str, tuple[Any, Any]] = {}
    for family, seed in EXPECTED_MATRIX:
        job = jobs[(family, seed)]
        work_dir = Path(str(job["work_dir"])).resolve()
        expected_work_dir = (
            Path(str(deployment["canonical_experiment_root"])).resolve()
            / family.lower()
            / f"seed{seed}"
        )
        if work_dir != expected_work_dir:
            raise ValueError(f"matrix work directory changed for {family}/{seed}")
        config_path = control_dir / f"{family.lower()}_seed{seed}.py"
        checkpoint_path = work_dir / "checkpoint" / "epoch_59.pth"
        completion_path = work_dir / "training_completion.json"
        _checkpoint_inventory(checkpoint_path.parent)
        cfg = load_pure_data_config(config_path)
        bound_config_sha256 = canonical_sha256(cfg.to_dict())
        if bound_config_sha256 != cell_intents[(family, seed)]["bound_config_sha256"]:
            raise ValueError(
                f"bound config differs from the submitted cell intent: {family}/{seed}"
            )
        if family not in validation_runtimes:
            validation_runtimes[family] = build_checkpoint_validation_runtime(cfg)
        strict_model, strict_optimizer = validation_runtimes[family]
        completion, checkpoint_audit = validate_training_completion_with_audit(
            _load_json(completion_path),
            cfg=cfg,
            seed=seed,
            checkpoint_path=checkpoint_path,
            strict_model=strict_model,
            strict_optimizer=strict_optimizer,
        )
        runtime_binding = checkpoint_audit["runtime_binding"]
        if (
            completion["family"] != family
            or int(completion["seed"]) != seed
            or completion["code_commit"] != deployment["code_commit"]
            or completion["experiment_namespace"] != deployment["campaign_namespace"]
            or completion["successful_updates"] != 4800
            or completion["checkpoint_consumer_state_key"] != "state_dict_ema"
            or completion["paper_claim_allowed"] is not False
            or runtime_binding["base_experiment_namespace"]
            != deployment["base_experiment_namespace"]
            or runtime_binding["experiment_namespace"]
            != deployment["campaign_namespace"]
            or runtime_binding["canonical_experiment_root"]
            != deployment["canonical_experiment_root"]
            or runtime_binding["full_model_gate_sha256"]
            != deployment["full_model_gate_sha256"]
            or runtime_binding["training_runtime_precheck_sha256"]
            != deployment["training_runtime_precheck_sha256"]
            or runtime_binding["runtime_authorization_sha256"]
            != deployment["runtime_authorization_sha256"]
        ):
            raise ValueError(f"invalid completion invariants for {family}/{seed}")
        job_id = str(job["job_id"])
        cells.append(
            {
                "family": family,
                "seed": seed,
                "job_id": job_id,
                "job_name": job["job_name"],
                "job_token": job["job_token"],
                "submission_sha256": job["submission_sha256"],
                "slurm": checked_accounting[job_id],
                "job_log_binding": _validate_job_log_binding(
                    control_dir=control_dir,
                    receipt=job,
                    completion=completion,
                ),
                "work_dir": str(work_dir),
                "bound_config_path": str(config_path),
                "bound_config_sha256": bound_config_sha256,
                "bound_config_file_sha256": sha256_file(config_path),
                "completion_path": str(completion_path),
                "completion_file_sha256": sha256_file(completion_path),
                "completion_sha256": completion["completion_sha256"],
                "protocol_sha256": completion["protocol_sha256"],
                "checkpoint_path": completion["checkpoint_path"],
                "checkpoint_sha256": completion["checkpoint_sha256"],
                "checkpoint_sidecar_path": completion["checkpoint_sidecar_path"],
                "checkpoint_sidecar_sha256": completion["checkpoint_sidecar_sha256"],
                "successful_updates": completion["successful_updates"],
                "optimizer_attempts": completion["optimizer_attempts"],
                "amp_skipped_attempts": completion["amp_skipped_attempts"],
                "max_amp_retries_observed": completion["max_amp_retries_observed"],
                "checkpoint_consumer_state_key": "state_dict_ema",
                "checkpoint_state_audit": checkpoint_audit,
            }
        )

    protocol_hashes = {cell["protocol_sha256"] for cell in cells}
    if len(protocol_hashes) != 1 or protocol_hashes != {intent["protocol_sha256"]}:
        raise ValueError("matrix completion receipts bind different protocols")
    report = {
        "schema_version": S2_TRAINING_MATRIX_COMPLETION_SCHEMA,
        "status": "PASS_TRAINING_ONLY",
        "training_code_commit": deployment["code_commit"],
        "validation_code_commit": validation_commit,
        "validation_source_sha256": validation_sources,
        "protocol_sha256": next(iter(protocol_hashes)),
        "base_experiment_namespace": deployment["base_experiment_namespace"],
        "campaign_namespace": deployment["campaign_namespace"],
        "canonical_experiment_root": deployment["canonical_experiment_root"],
        "deployment_summary_path": str(deployment_summary_path),
        "deployment_summary_file_sha256": sha256_file(deployment_summary_path),
        "deployment_sha256": deployment["deployment_sha256"],
        "deployment_intent_path": str(Path(str(deployment["intent_path"])).resolve()),
        "deployment_intent_file_sha256": sha256_file(deployment["intent_path"]),
        "deployment_intent_sha256": intent["intent_sha256"],
        "full_model_gate_sha256": deployment["full_model_gate_sha256"],
        "training_runtime_precheck_sha256": deployment[
            "training_runtime_precheck_sha256"
        ],
        "runtime_authorization_sha256": deployment["runtime_authorization_sha256"],
        "submission_environment": deployment["submission_environment"],
        "outer_allocation": intent["outer_allocation"],
        "inner_training_step": intent["inner_training_step"],
        "matrix_order": [
            {"family": family, "seed": seed} for family, seed in EXPECTED_MATRIX
        ],
        "cells": cells,
        "cell_count": len(cells),
        "all_jobs_completed_zero_exit": True,
        "all_job_names_and_tokens_match_slurm": True,
        "all_job_logs_bind_live_artifacts": True,
        "all_cells_revalidated_from_live_artifacts": True,
        "all_checkpoint_structures_live_validated": True,
        "all_checkpoints_strict_loaded_into_real_models": True,
        "all_cells_final_checkpoint_only": True,
        "all_checkpoint_consumers_use_state_dict_ema": True,
        "all_cells_exact_4800_successful_updates": True,
        "development_only_inputs_bound": True,
        "official_test_open_allowed": False,
        "official_test_runtime_access_audited": False,
        "official_test_opened": None,
        "official_test_evidence_consumed": False,
        "reference_sweep_completed": False,
        "crop_sufficiency_established": False,
        "paper_claim_allowed": False,
    }
    return finalize_self_hash(report, "matrix_completion_sha256")


def validate_training_matrix_completion(
    receipt: Mapping[str, Any],
    *,
    deployment_summary_path: str | Path,
    expected_deployment_sha256: str,
) -> dict[str, Any]:
    checked = _validate_self_hash(
        receipt,
        "matrix_completion_sha256",
        schema=S2_TRAINING_MATRIX_COMPLETION_SCHEMA,
    )
    rebuilt = build_training_matrix_completion(
        deployment_summary_path=deployment_summary_path,
        expected_deployment_sha256=expected_deployment_sha256,
    )
    if checked != rebuilt:
        raise ValueError("training matrix completion no longer matches live evidence")
    return rebuilt


def _publish_once(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate and seal the exact-nine Continuous-RoI S2 training matrix"
        )
    )
    parser.add_argument("--deployment-summary", type=Path, required=True)
    parser.add_argument("--expected-deployment-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        deployment = _load_json(args.deployment_summary)
        canonical_output = (
            Path(str(deployment["canonical_experiment_root"])).resolve()
            / "control"
            / "training_matrix_completion.json"
        )
        if args.output.resolve() != canonical_output:
            raise ValueError("training matrix receipt output path is not canonical")
        if args.output.exists():
            report = validate_training_matrix_completion(
                _load_json(args.output),
                deployment_summary_path=args.deployment_summary,
                expected_deployment_sha256=args.expected_deployment_sha256,
            )
        else:
            report = build_training_matrix_completion(
                deployment_summary_path=args.deployment_summary,
                expected_deployment_sha256=args.expected_deployment_sha256,
            )
            _publish_once(args.output, report)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
