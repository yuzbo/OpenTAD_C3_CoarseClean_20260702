from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_contract import canonical_sha256  # noqa: E402
from tools.bata.continuous_roi_s2_training import (  # noqa: E402
    S2_FAMILIES,
    S2_SOURCE_CONFIGS,
    S2_TRAINING_SEEDS,
    bind_training_config,
    require_clean_git_checkout,
    validate_training_runtime_precheck,
)
from tools.bata.continuous_roi_s2_runtime_gate import (  # noqa: E402
    validate_runtime_authorization,
)
from tools.bata.spatial_zoom_s1_contract import sha256_file  # noqa: E402


DEPLOYMENT_SCHEMA = "continuous_roi_s2_training_deployment_v1"
INTENT_SCHEMA = "continuous_roi_s2_training_deployment_intent_v1"
JOB_RECEIPT_SCHEMA = "continuous_roi_s2_training_job_submission_v1"
CELL_INTENT_SCHEMA = "continuous_roi_s2_training_cell_intent_v1"


def _load_self_hashed(path: Path, *, hash_key: str, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        raise ValueError(f"unsupported evidence schema: {path}")
    expected = payload.pop(hash_key, None)
    if not expected or canonical_sha256(payload) != expected:
        raise ValueError(f"evidence self-hash mismatch: {path}")
    payload[hash_key] = expected
    return payload


def _publish_once(path: Path, payload: dict[str, Any]) -> None:
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


def _run(*arguments: str) -> str:
    completed = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _accounted_jobs(job_names: set[str]) -> dict[str, list[dict[str, str]]]:
    records = {name: [] for name in job_names}
    outputs = [
        _run("squeue", "-h", "-o", "%i|%j|%T|%k"),
        _run(
            "sacct",
            "-X",
            "-S",
            "2026-07-20",
            "-n",
            "-P",
            "-o",
            "JobIDRaw,JobName,State,Comment",
        ),
    ]
    seen = set()
    for output in outputs:
        for raw_line in output.splitlines():
            fields = [field.strip() for field in raw_line.split("|")]
            if len(fields) < 3:
                continue
            job_id, job_name, state = fields[:3]
            comment = fields[3] if len(fields) >= 4 else ""
            key = (job_id, job_name)
            if job_name in records and job_id.isdigit() and key not in seen:
                records[job_name].append(
                    {
                        "job_id": job_id,
                        "job_name": job_name,
                        "state": state,
                        "comment": comment,
                    }
                )
                seen.add(key)
    return records


def _submit_job(
    *,
    launcher: Path,
    job_name: str,
    job_token: str,
    log_dir: Path,
    exports: dict[str, str],
) -> str:
    export_text = "ALL," + ",".join(
        f"{key}={value}" for key, value in sorted(exports.items())
    )
    output = _run(
        "sbatch",
        "--parsable",
        "--partition=gpu",
        "--gpus=2",
        "--cpus-per-task=8",
        f"--job-name={job_name}",
        f"--comment=crs2:{job_token}",
        f"--output={log_dir / (job_name + '-%j.out')}",
        f"--error={log_dir / (job_name + '-%j.err')}",
        f"--export={export_text}",
        str(launcher),
    )
    job_id = output.split(";", 1)[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned an invalid job ID: {output}")
    return job_id


def _write_jobs_tsv(path: Path, receipts: list[dict[str, Any]]) -> None:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(
        ["family", "seed", "job_id", "job_name", "job_token", "work_dir"]
    )
    for receipt in receipts:
        writer.writerow(
            [
                receipt["family"],
                receipt["seed"],
                receipt["job_id"],
                receipt["job_name"],
                receipt["job_token"],
                receipt["work_dir"],
            ]
        )
    expected_text = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != expected_text:
            raise ValueError("existing S2 jobs.tsv differs from job receipts")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(expected_text)
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def deploy(args) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    launcher = (source_root / args.launcher).resolve()
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    require_clean_git_checkout(
        expected_commit=args.expected_commit, repository_root=source_root
    )
    precheck = validate_training_runtime_precheck(
        args.training_runtime_precheck,
        expected_commit=args.expected_commit,
        expected_full_model_gate_sha256=args.full_model_gate_sha256,
    )
    authorization = validate_runtime_authorization(
        args.runtime_authorization,
        expected_commit=args.expected_commit,
        expected_full_model_gate_sha256=args.full_model_gate_sha256,
        expected_precheck_sha256=precheck["precheck_sha256"],
    )
    canonical_root = Path(
        authorization["canonical_experiment_root"]
    ).resolve()
    if str(canonical_root).replace("\\", "/").startswith(
        "/data/run01/sczc063/yuzibo/"
    ) is False:
        raise ValueError("canonical S2 root is outside the remote write boundary")

    cells = []
    for family in S2_FAMILIES:
        for seed in S2_TRAINING_SEEDS:
            work_dir = canonical_root / family.lower() / f"seed{seed}"
            cfg = bind_training_config(
                source_config_path=source_root / S2_SOURCE_CONFIGS[family],
                family=family,
                seed=seed,
                work_dir=work_dir,
                manifest_path=args.manifest,
                development_annotation_path=args.development_annotation,
                class_map_path=args.class_map,
                development_video_root=args.development_video_root,
                pretrained_checkpoint_path=args.pretrained,
                full_model_gate_path=args.full_model_gate,
                training_runtime_precheck_path=args.training_runtime_precheck,
                runtime_authorization_path=args.runtime_authorization,
                repository_root=source_root,
                code_commit=args.expected_commit,
            )
            cells.append(
                {
                    "family": family,
                    "seed": seed,
                    "work_dir": str(work_dir),
                    "bound_config_sha256": canonical_sha256(cfg.to_dict()),
                    "job_name": (
                        f"crs2_{authorization['campaign_namespace'][:8]}_"
                        f"{family.lower()}_{seed}"
                    ),
                }
            )

    control_dir = canonical_root / "control"
    receipt_dir = control_dir / "job_receipts"
    log_dir = control_dir / "logs"
    summary_path = control_dir / "deployment_summary.json"
    intent_path = control_dir / "deployment_intent.json"
    jobs_path = control_dir / "jobs.tsv"
    control_dir.mkdir(parents=True, exist_ok=True)
    lock_path = control_dir / "deployment.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if summary_path.exists():
            summary = _load_self_hashed(
                summary_path,
                hash_key="deployment_sha256",
                schema=DEPLOYMENT_SCHEMA,
            )
            if (
                summary.get("code_commit") != args.expected_commit
                or summary.get("base_experiment_namespace")
                != precheck["experiment_namespace"]
                or summary.get("campaign_namespace")
                != authorization["campaign_namespace"]
                or summary.get("canonical_experiment_root")
                != str(canonical_root)
                or summary.get("full_model_gate_sha256")
                != args.full_model_gate_sha256
                or summary.get("training_runtime_precheck_sha256")
                != precheck["precheck_sha256"]
                or summary.get("runtime_authorization_sha256")
                != authorization["authorization_sha256"]
            ):
                raise ValueError(
                    "existing deployment summary differs from this request"
                )
            return summary

        intent_core = {
            "schema_version": INTENT_SCHEMA,
            "status": "SUBMISSION_INTENT",
            "code_commit": args.expected_commit,
            "protocol_sha256": precheck["protocol_sha256"],
            "base_experiment_namespace": precheck["experiment_namespace"],
            "campaign_namespace": authorization["campaign_namespace"],
            "canonical_experiment_root": str(canonical_root),
            "full_model_gate_sha256": args.full_model_gate_sha256,
            "training_runtime_precheck_sha256": precheck["precheck_sha256"],
            "runtime_authorization_path": str(
                args.runtime_authorization.resolve()
            ),
            "runtime_authorization_file_sha256": sha256_file(
                args.runtime_authorization
            ),
            "runtime_authorization_sha256": authorization[
                "authorization_sha256"
            ],
            "cells": cells,
            "outer_allocation": {
                "partition": "gpu",
                "gpus": 2,
                "cpus": 8,
                "purpose": "N16R4 site memory policy only",
            },
            "inner_training_step": {
                "gpus": 1,
                "cpus": 5,
                "memory_mib": 96000,
                "logical_device": "cuda:0",
                "overrides_cuda_visible_devices": False,
            },
            "official_test_open_allowed": False,
            "paper_claim_allowed": False,
        }
        intent = {**intent_core, "intent_sha256": canonical_sha256(intent_core)}
        if intent_path.exists():
            existing_intent = _load_self_hashed(
                intent_path, hash_key="intent_sha256", schema=INTENT_SCHEMA
            )
            if existing_intent != intent:
                raise ValueError("existing S2 deployment intent differs")
        else:
            _publish_once(intent_path, intent)

        job_names = {cell["job_name"] for cell in cells}
        accounted = _accounted_jobs(job_names)
        receipts = []
        receipt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            cell_intent_path = (
                receipt_dir
                / f"{cell['family'].lower()}_seed{cell['seed']}.intent.json"
            )
            cell_intent_core = {
                "schema_version": CELL_INTENT_SCHEMA,
                "family": cell["family"],
                "seed": cell["seed"],
                "job_name": cell["job_name"],
                "work_dir": cell["work_dir"],
                "bound_config_sha256": cell["bound_config_sha256"],
                "campaign_namespace": authorization["campaign_namespace"],
                "deployment_intent_sha256": intent["intent_sha256"],
            }
            job_token = canonical_sha256(cell_intent_core)
            cell_intent = {
                **cell_intent_core,
                "job_token": job_token,
                "cell_intent_sha256": canonical_sha256(
                    {**cell_intent_core, "job_token": job_token}
                ),
            }
            cell_intent_created = False
            if cell_intent_path.exists():
                existing_cell_intent = _load_self_hashed(
                    cell_intent_path,
                    hash_key="cell_intent_sha256",
                    schema=CELL_INTENT_SCHEMA,
                )
                if existing_cell_intent != cell_intent:
                    raise ValueError("existing S2 cell intent differs")
            else:
                _publish_once(cell_intent_path, cell_intent)
                cell_intent_created = True
            receipt_path = (
                receipt_dir / f"{cell['family'].lower()}_seed{cell['seed']}.json"
            )
            if receipt_path.exists():
                receipt = _load_self_hashed(
                    receipt_path,
                    hash_key="submission_sha256",
                    schema=JOB_RECEIPT_SCHEMA,
                )
                if (
                    receipt["job_name"] != cell["job_name"]
                    or receipt["work_dir"] != cell["work_dir"]
                    or receipt.get("job_token") != job_token
                ):
                    raise ValueError("existing S2 job receipt differs from intent")
                receipts.append(receipt)
                continue

            named = accounted[cell["job_name"]]
            wrong_token = [
                job
                for job in named
                if job.get("comment") != f"crs2:{job_token}"
            ]
            if wrong_token:
                raise RuntimeError(
                    f"Slurm job name collision for {cell['job_name']}: "
                    f"{wrong_token}"
                )
            matching = [
                job
                for job in named
                if job.get("comment") == f"crs2:{job_token}"
            ]
            if len(matching) > 1:
                raise RuntimeError(
                    f"multiple Slurm jobs already use {cell['job_name']}"
                )
            if matching:
                job_id = matching[0]["job_id"]
                recovered_from_accounting = True
                accounting_state = matching[0]["state"]
            else:
                if not cell_intent_created:
                    raise RuntimeError(
                        "cell intent exists but its Slurm job is not visible; "
                        "refusing to risk a duplicate submission"
                    )
                bound_config_path = (
                    control_dir
                    / f"{cell['family'].lower()}_seed{cell['seed']}.py"
                )
                if Path(cell["work_dir"]).exists() or bound_config_path.exists():
                    raise FileExistsError(
                        "S2 work/config artifact exists without a Slurm job receipt"
                    )
                exports = {
                    "CONTINUOUS_ROI_S2_SOURCE_ROOT": str(source_root),
                    "CONTINUOUS_ROI_S2_RUN_ROOT": str(canonical_root),
                    "CONTINUOUS_ROI_S2_MANIFEST": str(args.manifest.resolve()),
                    "CONTINUOUS_ROI_S2_DEVELOPMENT_ANNOTATION": str(
                        args.development_annotation.resolve()
                    ),
                    "CONTINUOUS_ROI_S2_CLASS_MAP": str(args.class_map.resolve()),
                    "CONTINUOUS_ROI_S2_DEVELOPMENT_VIDEO_ROOT": str(
                        args.development_video_root.resolve()
                    ),
                    "CONTINUOUS_ROI_S2_PRETRAINED": str(args.pretrained.resolve()),
                    "CONTINUOUS_ROI_S2_FULL_MODEL_GATE": str(
                        args.full_model_gate.resolve()
                    ),
                    "CONTINUOUS_ROI_S2_TRAINING_RUNTIME_PRECHECK": str(
                        args.training_runtime_precheck.resolve()
                    ),
                    "CONTINUOUS_ROI_S2_RUNTIME_AUTHORIZATION": str(
                        args.runtime_authorization.resolve()
                    ),
                    "CONTINUOUS_ROI_S2_EXPECTED_COMMIT": args.expected_commit,
                    "CONTINUOUS_ROI_S2_FAMILY": cell["family"],
                    "CONTINUOUS_ROI_S2_SEED": str(cell["seed"]),
                    "YUZIBO_ROOT": args.yuzibo_root,
                }
                job_id = _submit_job(
                    launcher=launcher,
                    job_name=cell["job_name"],
                    job_token=job_token,
                    log_dir=log_dir,
                    exports=exports,
                )
                recovered_from_accounting = False
                accounting_state = "SUBMITTED_BY_THIS_PROCESS"
            receipt_core = {
                "schema_version": JOB_RECEIPT_SCHEMA,
                "status": "SUBMITTED",
                "family": cell["family"],
                "seed": cell["seed"],
                "job_id": job_id,
                "job_name": cell["job_name"],
                "job_token": job_token,
                "work_dir": cell["work_dir"],
                "base_experiment_namespace": precheck[
                    "experiment_namespace"
                ],
                "campaign_namespace": authorization["campaign_namespace"],
                "intent_sha256": intent["intent_sha256"],
                "cell_intent_path": str(cell_intent_path),
                "cell_intent_sha256": cell_intent[
                    "cell_intent_sha256"
                ],
                "recovered_from_accounting": recovered_from_accounting,
                "accounting_state_at_receipt": accounting_state,
            }
            receipt = {
                **receipt_core,
                "submission_sha256": canonical_sha256(receipt_core),
            }
            _publish_once(receipt_path, receipt)
            receipts.append(receipt)

        if len(receipts) != 9 or len({item["job_id"] for item in receipts}) != 9:
            raise RuntimeError("S2 deployment did not produce nine unique jobs")
        _write_jobs_tsv(jobs_path, receipts)
        summary_core = {
            "schema_version": DEPLOYMENT_SCHEMA,
            "status": "SUBMITTED",
            "code_commit": args.expected_commit,
            "base_experiment_namespace": precheck["experiment_namespace"],
            "campaign_namespace": authorization["campaign_namespace"],
            "canonical_experiment_root": str(canonical_root),
            "full_model_gate_sha256": args.full_model_gate_sha256,
            "training_runtime_precheck_sha256": precheck["precheck_sha256"],
            "runtime_authorization_sha256": authorization[
                "authorization_sha256"
            ],
            "intent_path": str(intent_path),
            "intent_sha256": intent["intent_sha256"],
            "jobs_tsv": str(jobs_path),
            "jobs": receipts,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        summary = {
            **summary_core,
            "deployment_sha256": canonical_sha256(summary_core),
        }
        _publish_once(summary_path, summary)
        return summary


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Idempotently submit the registered Continuous-RoI S2 3x3 matrix"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--full-model-gate", type=Path, required=True)
    parser.add_argument("--full-model-gate-sha256", required=True)
    parser.add_argument("--training-runtime-precheck", type=Path, required=True)
    parser.add_argument("--runtime-authorization", type=Path, required=True)
    parser.add_argument(
        "--launcher",
        type=Path,
        default=Path("scripts/run_continuous_roi_s2_train_slurm.sh"),
    )
    parser.add_argument(
        "--yuzibo-root",
        default="/data/run01/sczc063/yuzibo",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = deploy(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
