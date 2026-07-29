#!/usr/bin/env python3
"""Run one no-metric GeoRoute real-batch AMP diagnostic arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_amp_diagnostic import (  # noqa: E402
    AMP_DIAGNOSTIC_ARMS,
    AMP_DIAGNOSTIC_DEPLOYMENT_SCHEMA,
    AMP_DIAGNOSTIC_PROFILE,
    AMP_DIAGNOSTIC_STAGE_SCHEMA,
    AMP_DIAGNOSTIC_STUDY_ID,
    AMP_STABILITY_PROFILE,
    amp_protocol_spec,
    bind_amp_diagnostic_config,
    diagnostic_cell_relative_path,
    validate_amp_diagnostic_job_receipt,
    validate_amp_diagnostic_receipt,
)
from tools.bata.georoute_estimator_pilot_contract import (  # noqa: E402
    PILOT_SEED,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _run_logged,
    build_torchrun_prefix,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _validate_deployment(
    path: Path,
    *,
    expected_commit: str,
    expected_job_id: str,
    arm: str,
    protocol_profile: str = AMP_DIAGNOSTIC_PROFILE,
) -> dict[str, Any]:
    spec = amp_protocol_spec(protocol_profile)
    deployment = _read_json(path)
    jobs = validate_amp_diagnostic_job_receipt(deployment.get("jobs"))
    if (
        deployment.get("schema_version") != spec["deployment_schema"]
        or deployment.get("status")
        != spec["deployment_status"]
        or deployment.get("study_id") != spec["study_id"]
        or str(
            deployment.get(
                "protocol_profile",
                AMP_DIAGNOSTIC_PROFILE,
            )
        )
        != spec["profile"]
        or deployment.get("runtime_commit") != expected_commit
        or tuple(deployment.get("arms", [])) != AMP_DIAGNOSTIC_ARMS
        or jobs["stage"][arm] != expected_job_id
        or deployment.get("checkpoint_emitted") is not False
        or deployment.get("evaluator_invoked") is not False
        or deployment.get("official_test_opened") is not False
        or deployment.get("paper_claim_allowed") is not False
        or not _self_hash_matches(
            deployment,
            field="deployment_sha256",
        )
    ):
        raise RuntimeError("AMP diagnostic deployment receipt is invalid")
    if protocol_profile == AMP_STABILITY_PROFILE:
        matched_inputs = deployment.get("matched_diagnostic_inputs")
        parent = deployment.get("parent_diagnostic")
        if (
            not isinstance(matched_inputs, Mapping)
            or matched_inputs.get("all_equal") is not True
            or not isinstance(parent, Mapping)
            or parent.get("decision")
            != "ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED"
        ):
            raise RuntimeError(
                "AMP stability deployment lacks matched diagnostic provenance"
            )
    return deployment


def _validate_train_rendezvous(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    slurm_job_id: str,
    protocol_profile: str = AMP_DIAGNOSTIC_PROFILE,
) -> dict[str, Any]:
    spec = amp_protocol_spec(protocol_profile)
    stage = str(spec["rendezvous_stage"])
    expected_id = (
        f"georoute-{slurm_job_id}-{stage}-{arm}-s{PILOT_SEED}-train"
    )
    if (
        receipt.get("phase") != "train"
        or receipt.get("backend") != "c10d"
        or receipt.get("stage") != stage
        or receipt.get("variant") != arm
        or int(receipt.get("seed", -1)) != PILOT_SEED
        or receipt.get("slurm_job_id") != slurm_job_id
        or receipt.get("rendezvous_id") != expected_id
        or int(receipt.get("nnodes", -1)) != 1
        or int(receipt.get("nproc_per_node", -1)) != 1
        or receipt.get("endpoint_policy")
        != "job_scoped_loopback_and_kernel_assigned_port"
    ):
        raise ValueError("AMP diagnostic rendezvous receipt is invalid")
    return dict(receipt)


def audit_no_performance_artifacts(cell_root: Path) -> dict[str, Any]:
    checkpoint_payloads = sorted(
        {
            *cell_root.rglob("*.pth"),
            *cell_root.rglob("*.pt"),
            *cell_root.rglob("*.ckpt"),
        }
    )
    temporary_payloads = sorted(cell_root.rglob("*.tmp*"))
    forbidden_names = {
        "result_detection.json",
        "georoute_development_profile.json",
        "georoute_diagnostic_telemetry.json",
        "test.out",
    }
    forbidden_outputs = sorted(
        path
        for path in cell_root.rglob("*")
        if path.is_file() and path.name in forbidden_names
    )
    if checkpoint_payloads or temporary_payloads or forbidden_outputs:
        raise RuntimeError(
            "AMP diagnostic emitted a forbidden performance artifact: "
            f"checkpoints={checkpoint_payloads}, "
            f"temporaries={temporary_payloads}, outputs={forbidden_outputs}"
        )
    return {
        "checkpoint_payload_count": 0,
        "temporary_payload_count": 0,
        "prediction_payload_count": 0,
        "evaluator_output_count": 0,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }


def validate_amp_diagnostic_stage_result(
    result: Mapping[str, Any],
    *,
    expected_arm: str | None = None,
    expected_commit: str | None = None,
    expected_job_id: str | None = None,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    result = dict(result)
    if not _self_hash_matches(result, field="stage_result_sha256"):
        raise ValueError("AMP diagnostic stage-result self-hash mismatch")
    arm = str(result.get("arm", ""))
    status = result.get("status")
    profile = str(
        result.get(
            "protocol_profile",
            AMP_DIAGNOSTIC_PROFILE,
        )
    )
    spec = amp_protocol_spec(profile)
    if expected_profile is not None and profile != expected_profile:
        raise ValueError("AMP stage-result protocol profile mismatch")
    if (
        result.get("schema_version") != spec["stage_schema"]
        or result.get("study_id") != spec["study_id"]
        or arm not in AMP_DIAGNOSTIC_ARMS
        or status
        not in {
            spec["stage_pass_status"],
            spec["stage_fail_status"],
        }
        or int(result.get("seed", -1)) != PILOT_SEED
        or result.get("checkpoint_emitted") is not False
        or result.get("prediction_emitted") is not False
        or result.get("evaluator_invoked") is not False
        or result.get("official_test_opened") is not False
        or result.get("performance_inference_allowed") is not False
        or result.get("paper_claim_allowed") is not False
    ):
        raise ValueError("AMP diagnostic stage-result contract is invalid")
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("AMP diagnostic stage-result arm mismatch")
    if (
        expected_commit is not None
        and result.get("runtime_commit") != str(expected_commit).lower()
    ):
        raise ValueError("AMP diagnostic stage-result commit mismatch")
    job_id = str(result.get("slurm_job_id", ""))
    if not job_id.isdigit():
        raise ValueError("AMP diagnostic stage-result lacks a Slurm ID")
    if expected_job_id is not None and job_id != str(expected_job_id):
        raise ValueError("AMP diagnostic stage-result Slurm ID mismatch")
    artifact_audit = result.get("artifact_audit")
    if not isinstance(artifact_audit, Mapping) or any(
        int(artifact_audit.get(key, -1)) != 0
        for key in (
            "checkpoint_payload_count",
            "temporary_payload_count",
            "prediction_payload_count",
            "evaluator_output_count",
        )
    ):
        raise ValueError("AMP diagnostic stage emitted forbidden artifacts")
    diagnostic = result.get("diagnostic_receipt")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("AMP diagnostic stage lacks its observer receipt")
    validated = validate_amp_diagnostic_receipt(
        diagnostic,
        expected_arm=arm,
        expected_commit=result["runtime_commit"],
        expected_slurm_job_id=job_id,
        expected_profile=profile,
    )
    if (
        result.get("binding") != validated["binding"]
        or result.get("binding_sha256")
        != validated["binding"]["binding_sha256"]
    ):
        raise ValueError("AMP diagnostic stage and observer bindings differ")
    expected_status = (
        spec["stage_pass_status"]
        if validated["status"] == spec["receipt_pass_status"]
        and result.get("execution_error") is None
        else spec["stage_fail_status"]
    )
    if status != expected_status:
        raise ValueError("AMP diagnostic stage status differs from observer")
    _validate_train_rendezvous(
        result.get("rendezvous", {}),
        arm=arm,
        slurm_job_id=job_id,
        protocol_profile=profile,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=AMP_DIAGNOSTIC_ARMS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--protocol-profile",
        choices=(AMP_DIAGNOSTIC_PROFILE, AMP_STABILITY_PROFILE),
        default=AMP_DIAGNOSTIC_PROFILE,
    )
    return parser.parse_args()


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    spec = amp_protocol_spec(args.protocol_profile)
    expected_commit = str(args.expected_commit).lower()
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("AMP diagnostic source differs from its bound commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("AMP diagnostic requires an exact clean source")
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not slurm_job_id.isdigit() or not visible or "," in visible:
        raise RuntimeError("AMP diagnostic requires one Slurm-visible GPU")

    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary):
        raise ValueError("AMP diagnostic run root leaves the write boundary")
    deployment_path = run_root / "control" / "deployment.json"
    deployment = _validate_deployment(
        deployment_path,
        expected_commit=expected_commit,
        expected_job_id=slurm_job_id,
        arm=args.arm,
        protocol_profile=args.protocol_profile,
    )
    cell_root = run_root / diagnostic_cell_relative_path(
        arm=args.arm,
        protocol_profile=args.protocol_profile,
    )
    bound_config = (
        run_root
        / "control"
        / "bound_configs"
        / f"{spec['study_id']}_{args.arm}_seed{PILOT_SEED}.py"
    )
    train_log = run_root / "control" / "train_logs" / f"{args.arm}.out"
    if cell_root.exists() or bound_config.exists() or train_log.exists():
        raise FileExistsError(
            "AMP diagnostic cell/config/log exists; refusing resume"
        )

    cfg = bind_amp_diagnostic_config(
        source_config_path=args.source_config,
        arm=args.arm,
        seed=PILOT_SEED,
        work_dir=cell_root,
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        runtime_commit=expected_commit,
        protocol_profile=args.protocol_profile,
    )
    expected_inputs = deployment["input_receipts"]
    if (
        cfg.georoute_amp_diagnostic_binding["source_config_sha256"]
        != expected_inputs["GEOROUTE_SOURCE_CONFIG"]["sha256"]
        or cfg.georoute_amp_diagnostic_binding["manifest_file_sha256"]
        != expected_inputs["GEOROUTE_MANIFEST"]["sha256"]
        or cfg.georoute_amp_diagnostic_binding[
            "development_annotation"
        ]["sha256"]
        != expected_inputs["GEOROUTE_DEVELOPMENT_ANNOTATION"]["sha256"]
        or cfg.georoute_amp_diagnostic_binding["class_map_sha256"]
        != expected_inputs["GEOROUTE_CLASS_MAP"]["sha256"]
        or cfg.georoute_amp_diagnostic_binding[
            "pretrained_checkpoint_sha256"
        ]
        != expected_inputs["GEOROUTE_PRETRAINED"]["sha256"]
        or cfg.georoute_amp_diagnostic_binding["development_video_root"]
        != expected_inputs["GEOROUTE_DEVELOPMENT_VIDEO_ROOT"]["path"]
    ):
        raise RuntimeError("AMP diagnostic immutable input binding changed")
    bound_config.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(bound_config))

    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    train_prefix, rendezvous = build_torchrun_prefix(
        phase="train",
        slurm_job_id=slurm_job_id,
        stage=str(spec["rendezvous_stage"]),
        variant=args.arm,
        seed=PILOT_SEED,
    )
    train_error: BaseException | None = None
    try:
        _run_logged(
            [
                *train_prefix,
                "tools/train.py",
                str(bound_config),
                "--seed",
                str(PILOT_SEED),
                "--id",
                "0",
            ],
            log_path=train_log,
            env=inherited,
        )
    except BaseException as error:
        train_error = error

    diagnostic_path = cell_root / str(spec["receipt_filename"])
    if not diagnostic_path.is_file():
        if train_error is not None:
            raise RuntimeError(
                "AMP diagnostic command failed before publishing its observer "
                "receipt"
            ) from train_error
        raise FileNotFoundError(diagnostic_path)
    diagnostic = validate_amp_diagnostic_receipt(
        _read_json(diagnostic_path),
        expected_arm=args.arm,
        expected_commit=expected_commit,
        expected_slurm_job_id=slurm_job_id,
        expected_profile=args.protocol_profile,
    )
    artifact_audit = audit_no_performance_artifacts(cell_root)
    status = (
        spec["stage_pass_status"]
        if diagnostic["status"] == spec["receipt_pass_status"]
        and train_error is None
        else spec["stage_fail_status"]
    )
    result: dict[str, Any] = {
        "schema_version": spec["stage_schema"],
        "status": status,
        "study_id": spec["study_id"],
        "protocol_profile": spec["profile"],
        "arm": args.arm,
        "seed": PILOT_SEED,
        "runtime_commit": expected_commit,
        "slurm_job_id": slurm_job_id,
        "binding": dict(cfg.georoute_amp_diagnostic_binding),
        "binding_sha256": cfg.georoute_amp_diagnostic_binding[
            "binding_sha256"
        ],
        "diagnostic_receipt": diagnostic,
        "diagnostic_receipt_path": str(diagnostic_path.resolve()),
        "diagnostic_receipt_file_sha256": sha256_file(diagnostic_path),
        "bound_config_path": str(bound_config.resolve()),
        "bound_config_sha256": sha256_file(bound_config),
        "train_log_path": str(train_log.resolve()),
        "train_log_sha256": sha256_file(train_log),
        "rendezvous": _validate_train_rendezvous(
            rendezvous,
            arm=args.arm,
            slurm_job_id=slurm_job_id,
            protocol_profile=args.protocol_profile,
        ),
        "artifact_audit": artifact_audit,
        "execution_error": (
            {
                "exception_type": type(train_error).__name__,
                "exception_message": str(train_error)[:2000],
            }
            if train_error is not None
            else None
        ),
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    result["stage_result_sha256"] = canonical_sha256(result)
    return validate_amp_diagnostic_stage_result(
        result,
        expected_arm=args.arm,
        expected_commit=expected_commit,
        expected_job_id=slurm_job_id,
        expected_profile=args.protocol_profile,
    )


def _write_wrapper_failure(
    *,
    args: argparse.Namespace,
    error: BaseException,
) -> None:
    spec = amp_protocol_spec(args.protocol_profile)
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary):
        return
    path = run_root / "control" / "stage_failures" / f"{args.arm}.json"
    if path.exists():
        return
    trace = traceback.format_exc()
    payload: dict[str, Any] = {
        "schema_version": spec["stage_schema"],
        "status": spec["stage_wrapper_fail_status"],
        "study_id": spec["study_id"],
        "protocol_profile": spec["profile"],
        "arm": args.arm,
        "seed": PILOT_SEED,
        "expected_runtime_commit": str(args.expected_commit).lower(),
        "observed_runtime_commit": (
            _git_output("rev-parse", "HEAD").lower()
            if (ROOT / ".git").exists()
            else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "exception_type": type(error).__name__,
        "exception_message": str(error)[:2000],
        "traceback_sha256": hashlib.sha256(
            trace.encode("utf-8", errors="replace")
        ).hexdigest(),
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    payload["failure_sha256"] = canonical_sha256(payload)
    _atomic_write_json(path, payload)


def main() -> int:
    args = _parse_args()
    try:
        result = _execute(args)
    except BaseException as error:
        try:
            _write_wrapper_failure(args=args, error=error)
        except BaseException:
            pass
        raise
    cell_root = args.run_root.resolve() / diagnostic_cell_relative_path(
        arm=args.arm,
        protocol_profile=args.protocol_profile,
    )
    result_path = cell_root / "stage_result.json"
    if result_path.exists():
        raise FileExistsError("AMP diagnostic stage result already exists")
    _atomic_write_json(result_path, result)
    print(json.dumps(result, sort_keys=True))
    spec = amp_protocol_spec(args.protocol_profile)
    return 0 if result["status"] == spec["stage_pass_status"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
