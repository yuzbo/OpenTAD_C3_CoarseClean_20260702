#!/usr/bin/env python3
"""Run one matched, no-performance GeoRoute gradient-decomposition arm."""

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

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_gradient_decomposition import (  # noqa: E402
    ARMS,
    DEPLOYMENT_SCHEMA,
    DEPLOYMENT_STATUS,
    FAIL_STATUS,
    MAX_NAMESPACE_BYTES,
    PASS_STATUS,
    PROFILE,
    RENDEZVOUS_STAGE,
    SEED,
    STAGE_FAIL_STATUS,
    STAGE_PASS_STATUS,
    STAGE_SCHEMA,
    STAGE_WRAPPER_FAIL_STATUS,
    STUDY_ID,
    bind_gradient_decomposition_config,
    cell_relative_path,
    validate_job_receipt,
    validate_receipt,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _run_logged,
    build_torchrun_prefix,
)


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
            completed.stderr.strip() or f"git {' '.join(arguments)} failed"
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
) -> dict[str, Any]:
    deployment = _read_json(path)
    jobs = validate_job_receipt(deployment.get("jobs"))
    parent = deployment.get("parent_stability_v2")
    inputs = deployment.get("input_receipts")
    if (
        deployment.get("schema_version") != DEPLOYMENT_SCHEMA
        or deployment.get("status") != DEPLOYMENT_STATUS
        or deployment.get("study_id") != STUDY_ID
        or deployment.get("profile") != PROFILE
        or deployment.get("runtime_commit") != expected_commit
        or tuple(deployment.get("arms", ())) != ARMS
        or int(deployment.get("seed", -1)) != SEED
        or jobs["stage"][arm] != expected_job_id
        or not isinstance(parent, Mapping)
        or parent.get("decision")
        != "OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD"
        or not isinstance(inputs, Mapping)
        or deployment.get("checkpoint_emitted") is not False
        or deployment.get("prediction_emitted") is not False
        or deployment.get("evaluator_invoked") is not False
        or deployment.get("official_test_opened") is not False
        or deployment.get("performance_inference_allowed") is not False
        or deployment.get("paper_claim_allowed") is not False
        or not _self_hash_matches(deployment, field="deployment_sha256")
    ):
        raise RuntimeError("gradient-decomposition deployment receipt is invalid")
    return deployment


def _validate_rendezvous(
    receipt: Mapping[str, Any], *, arm: str, slurm_job_id: str
) -> dict[str, Any]:
    expected_id = (
        f"georoute-{slurm_job_id}-{RENDEZVOUS_STAGE}-{arm}-s{SEED}-train"
    )
    if (
        receipt.get("phase") != "train"
        or receipt.get("backend") != "c10d"
        or receipt.get("stage") != RENDEZVOUS_STAGE
        or receipt.get("variant") != arm
        or int(receipt.get("seed", -1)) != SEED
        or receipt.get("slurm_job_id") != slurm_job_id
        or receipt.get("rendezvous_id") != expected_id
        or int(receipt.get("nnodes", -1)) != 1
        or int(receipt.get("nproc_per_node", -1)) != 1
        or receipt.get("endpoint_policy")
        != "job_scoped_loopback_and_kernel_assigned_port"
    ):
        raise ValueError("gradient-decomposition rendezvous receipt is invalid")
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
    forbidden_fragments = (
        "result_detection",
        "prediction",
        "metric",
        "map",
        "evaluator",
        "official_test",
        "latency",
        "energy",
        "profile",
    )
    forbidden_outputs = sorted(
        path
        for path in cell_root.rglob("*")
        if path.is_file()
        and path.name not in {"gradient_decomposition.json", "stage_result.json"}
        and any(fragment in path.name.lower() for fragment in forbidden_fragments)
    )
    namespace_bytes = sum(
        path.stat().st_size
        for path in cell_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if (
        checkpoint_payloads
        or temporary_payloads
        or forbidden_outputs
        or namespace_bytes > MAX_NAMESPACE_BYTES
    ):
        raise RuntimeError(
            "gradient decomposition emitted forbidden or oversized artifacts: "
            f"checkpoints={checkpoint_payloads}, temporaries={temporary_payloads}, "
            f"outputs={forbidden_outputs}, bytes={namespace_bytes}"
        )
    return {
        "checkpoint_payload_count": 0,
        "temporary_payload_count": 0,
        "prediction_payload_count": 0,
        "evaluator_output_count": 0,
        "namespace_bytes": namespace_bytes,
        "namespace_byte_limit": MAX_NAMESPACE_BYTES,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }


def validate_stage_result(
    result: Mapping[str, Any],
    *,
    expected_arm: str | None = None,
    expected_commit: str | None = None,
    expected_job_id: str | None = None,
) -> dict[str, Any]:
    result = dict(result)
    if not _self_hash_matches(result, field="stage_result_sha256"):
        raise ValueError("gradient-decomposition stage-result self-hash mismatch")
    arm = str(result.get("arm", ""))
    status = result.get("status")
    if (
        result.get("schema_version") != STAGE_SCHEMA
        or result.get("study_id") != STUDY_ID
        or result.get("profile") != PROFILE
        or arm not in ARMS
        or status not in {STAGE_PASS_STATUS, STAGE_FAIL_STATUS}
        or int(result.get("seed", -1)) != SEED
        or result.get("checkpoint_emitted") is not False
        or result.get("prediction_emitted") is not False
        or result.get("evaluator_invoked") is not False
        or result.get("official_test_opened") is not False
        or result.get("performance_inference_allowed") is not False
        or result.get("paper_claim_allowed") is not False
    ):
        raise ValueError("gradient-decomposition stage-result contract is invalid")
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("gradient-decomposition stage-result arm mismatch")
    if (
        expected_commit is not None
        and result.get("runtime_commit") != str(expected_commit).lower()
    ):
        raise ValueError("gradient-decomposition stage-result commit mismatch")
    job_id = str(result.get("slurm_job_id", ""))
    if not job_id.isdigit() or (
        expected_job_id is not None and job_id != str(expected_job_id)
    ):
        raise ValueError("gradient-decomposition stage-result Slurm ID mismatch")
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
        raise ValueError("gradient-decomposition stage emitted forbidden artifacts")
    receipt = validate_receipt(
        result.get("diagnostic_receipt", {}),
        expected_arm=arm,
        expected_commit=result["runtime_commit"],
        expected_slurm_job_id=job_id,
    )
    if (
        result.get("binding") != receipt["binding"]
        or result.get("binding_sha256") != receipt["binding"]["binding_sha256"]
    ):
        raise ValueError("gradient-decomposition stage and observer bindings differ")
    expected_status = (
        STAGE_PASS_STATUS
        if receipt["status"] == PASS_STATUS
        and result.get("execution_error") is None
        else STAGE_FAIL_STATUS
    )
    if status != expected_status:
        raise ValueError("gradient-decomposition stage status differs from observer")
    _validate_rendezvous(
        result.get("rendezvous", {}), arm=arm, slurm_job_id=job_id
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--official-reference-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("gradient-decomposition source differs from its commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("gradient decomposition requires an exact clean source")
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not slurm_job_id.isdigit() or not visible or "," in visible:
        raise RuntimeError("gradient decomposition requires one Slurm-visible GPU")

    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("gradient-decomposition run root leaves write boundary")
    deployment = _validate_deployment(
        run_root / "control" / "deployment.json",
        expected_commit=expected_commit,
        expected_job_id=slurm_job_id,
        arm=args.arm,
    )
    cell_root = run_root / cell_relative_path(args.arm)
    bound_config = (
        run_root
        / "control"
        / "bound_configs"
        / f"{STUDY_ID}_{args.arm}_seed{SEED}.py"
    )
    train_log = run_root / "control" / "train_logs" / f"{args.arm}.out"
    if cell_root.exists() or bound_config.exists() or train_log.exists():
        raise FileExistsError(
            "gradient-decomposition cell/config/log exists; refusing resume"
        )

    cfg = bind_gradient_decomposition_config(
        source_config_path=args.source_config,
        arm=args.arm,
        work_dir=cell_root,
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        official_reference_config_path=args.official_reference_config,
        runtime_commit=expected_commit,
        parent_evidence=deployment["parent_evidence"],
    )
    binding = dict(cfg.georoute_gradient_decomposition_binding)
    expected_inputs = deployment["input_receipts"]
    compared = {
        "GEOROUTE_SOURCE_CONFIG": binding["source_config_sha256"],
        "GEOROUTE_MANIFEST": binding["manifest_file_sha256"],
        "GEOROUTE_DEVELOPMENT_ANNOTATION": binding["development_annotation"][
            "sha256"
        ],
        "GEOROUTE_CLASS_MAP": binding["class_map_sha256"],
        "GEOROUTE_PRETRAINED": binding["pretrained_checkpoint_sha256"],
        "GEOROUTE_OFFICIAL_REFERENCE_CONFIG": binding[
            "official_reference_config_sha256"
        ],
    }
    if any(
        compared[name] != expected_inputs[name]["sha256"] for name in compared
    ) or (
        binding["development_video_root"]
        != expected_inputs["GEOROUTE_DEVELOPMENT_VIDEO_ROOT"]["path"]
    ):
        raise RuntimeError("gradient-decomposition immutable input binding changed")
    if binding["parent_evidence_sha256"] != deployment["parent_evidence_sha256"]:
        raise RuntimeError("gradient-decomposition parent evidence changed")
    bound_config.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(bound_config))

    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    train_prefix, rendezvous = build_torchrun_prefix(
        phase="train",
        slurm_job_id=slurm_job_id,
        stage=RENDEZVOUS_STAGE,
        variant=args.arm,
        seed=SEED,
    )
    train_error: BaseException | None = None
    try:
        _run_logged(
            [
                *train_prefix,
                "tools/train.py",
                str(bound_config),
                "--seed",
                str(SEED),
                "--id",
                "0",
            ],
            log_path=train_log,
            env=inherited,
        )
    except BaseException as error:
        train_error = error

    receipt_path = cell_root / "gradient_decomposition.json"
    if not receipt_path.is_file():
        if train_error is not None:
            raise RuntimeError(
                "gradient decomposition failed before publishing its receipt"
            ) from train_error
        raise FileNotFoundError(receipt_path)
    receipt = validate_receipt(
        _read_json(receipt_path),
        expected_arm=args.arm,
        expected_commit=expected_commit,
        expected_slurm_job_id=slurm_job_id,
    )
    artifact_audit = audit_no_performance_artifacts(cell_root)
    status = (
        STAGE_PASS_STATUS
        if receipt["status"] == PASS_STATUS and train_error is None
        else STAGE_FAIL_STATUS
    )
    result: dict[str, Any] = {
        "schema_version": STAGE_SCHEMA,
        "status": status,
        "study_id": STUDY_ID,
        "profile": PROFILE,
        "arm": args.arm,
        "seed": SEED,
        "runtime_commit": expected_commit,
        "slurm_job_id": slurm_job_id,
        "binding": binding,
        "binding_sha256": binding["binding_sha256"],
        "diagnostic_receipt": receipt,
        "diagnostic_receipt_path": str(receipt_path.resolve()),
        "diagnostic_receipt_file_sha256": sha256_file(receipt_path),
        "bound_config_path": str(bound_config.resolve()),
        "bound_config_sha256": sha256_file(bound_config),
        "train_log_path": str(train_log.resolve()),
        "train_log_sha256": sha256_file(train_log),
        "rendezvous": _validate_rendezvous(
            rendezvous, arm=args.arm, slurm_job_id=slurm_job_id
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
    return validate_stage_result(
        result,
        expected_arm=args.arm,
        expected_commit=expected_commit,
        expected_job_id=slurm_job_id,
    )


def _write_wrapper_failure(
    *, args: argparse.Namespace, error: BaseException
) -> None:
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        return
    path = run_root / "control" / "stage_failures" / f"{args.arm}.json"
    if path.exists():
        return
    trace = traceback.format_exc()
    payload: dict[str, Any] = {
        "schema_version": STAGE_SCHEMA,
        "status": STAGE_WRAPPER_FAIL_STATUS,
        "study_id": STUDY_ID,
        "profile": PROFILE,
        "arm": args.arm,
        "seed": SEED,
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
    cell_root = args.run_root.resolve() / cell_relative_path(args.arm)
    result_path = cell_root / "stage_result.json"
    if result_path.exists():
        raise FileExistsError("gradient-decomposition stage result already exists")
    _atomic_write_json(result_path, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == STAGE_PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
