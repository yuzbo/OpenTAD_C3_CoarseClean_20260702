#!/usr/bin/env python3
"""Submit the two-arm, no-metric GeoRoute real-batch AMP diagnostic DAG."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_amp_diagnostic import (  # noqa: E402
    AMP_DIAGNOSTIC_ARMS,
    AMP_DIAGNOSTIC_DEPLOYMENT_SCHEMA,
    AMP_DIAGNOSTIC_FINALIZATION_SCHEMA,
    AMP_DIAGNOSTIC_PROFILE,
    AMP_DIAGNOSTIC_STUDY_ID,
    AMP_REPAIR_INTERVENTION,
    AMP_REPAIR_PROFILE,
    AMP_REPAIR_REGISTERED_CLASS,
    AMP_STABILITY_FINALIZATION_SCHEMA,
    AMP_STABILITY_PROFILE,
    AMP_STABILITY_STUDY_ID,
    AMP_STABILITY_V2_PROFILE,
    amp_protocol_spec,
    validate_amp_diagnostic_job_receipt,
)
from tools.bata.georoute_ddp_fp16_cast_repair import (  # noqa: E402
    KAT_PASS_STATUS as REPAIR_KAT_PASS_STATUS,
    KAT_SCHEMA as REPAIR_KAT_SCHEMA,
    validate_kat_receipt as validate_repair_kat_receipt,
)
from tools.bata.georoute_estimator_pilot_contract import (  # noqa: E402
    PILOT_ARMS,
    PILOT_FINALIZATION_SCHEMA,
    PILOT_STUDY_ID,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402


GPU_OUTER_SLURM_ARGS = ("--gpus", "2", "--cpus-per-task", "8")
CONTROL_SLURM_ARGS = ("--gpus", "1", "--cpus-per-task", "1")
GRADIENT_ARMS = AMP_DIAGNOSTIC_ARMS


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


def _full_hex(value: str, *, length: int, name: str) -> str:
    normalized = str(value).lower()
    if (
        len(normalized) != length
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal digest")
    return normalized


def _clean_export(value: str, *, name: str) -> str:
    if (
        not value
        or value != value.strip()
        or any(character in value for character in (",", "\n", "\r", "\x00"))
    ):
        raise ValueError(f"{name} is unsafe for sbatch --export")
    return value


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _sbatch(
    *,
    name: str,
    script: Path,
    logs: Path,
    exports: Mapping[str, str],
    gpu: bool,
    dependency: Sequence[str] | None = None,
    dependency_type: str = "afterok",
    test_only: bool = False,
    hold: bool = False,
) -> str:
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        name,
        "--output",
        str(logs / f"{name}.%j.out"),
        "--error",
        str(logs / f"{name}.%j.err"),
    ]
    if test_only:
        command.append("--test-only")
    if hold and not test_only:
        command.append("--hold")
    if dependency:
        if dependency_type not in {"afterok", "afterany"}:
            raise ValueError("unsupported AMP diagnostic dependency type")
        command.extend(
            [
                "--dependency",
                f"{dependency_type}:" + ":".join(map(str, dependency)),
            ]
        )
    command.extend(GPU_OUTER_SLURM_ARGS if gpu else CONTROL_SLURM_ARGS)
    command.extend(
        [
            "--export",
            ",".join(
                [
                    "ALL",
                    *(
                        f"{key}={value}"
                        for key, value in sorted(exports.items())
                    ),
                ]
            ),
            str(script),
        ]
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {name}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    if test_only:
        return "TEST_ONLY_PASS"
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch job id for {name}")
    return job_id


def _cancel_jobs(job_ids: Sequence[str]) -> None:
    if not job_ids:
        return
    subprocess.run(
        ["scancel", *map(str, job_ids)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _release_jobs(job_ids: Sequence[str]) -> None:
    completed = subprocess.run(
        ["scontrol", "release", *map(str, job_ids)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to release held AMP diagnostic jobs: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )


def _require_submit_capacity(*, additional_jobs: int) -> dict[str, int]:
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("AMP diagnostic cannot determine Slurm user")
    active = subprocess.run(
        ["squeue", "-h", "-u", user],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if active.returncode != 0:
        raise RuntimeError(
            f"AMP diagnostic cannot query squeue: {active.stderr.strip()}"
        )
    active_count = len(
        [line for line in active.stdout.splitlines() if line.strip()]
    )
    association = subprocess.run(
        [
            "sacctmgr",
            "-n",
            "-P",
            "show",
            "assoc",
            "where",
            f"user={user}",
            "format=MaxSubmitJobs",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if association.returncode != 0:
        raise RuntimeError(
            "AMP diagnostic cannot query MaxSubmitJobs: "
            + association.stderr.strip()
        )
    limits = [
        int(line.split("|", 1)[0])
        for line in association.stdout.splitlines()
        if line.split("|", 1)[0].strip().isdigit()
    ]
    if not limits:
        raise RuntimeError("AMP diagnostic cannot determine MaxSubmitJobs")
    limit = min(limits)
    if active_count + int(additional_jobs) > limit:
        raise RuntimeError(
            "AMP diagnostic refuses partial submission: "
            f"active={active_count}, required={additional_jobs}, limit={limit}"
        )
    return {
        "active_jobs": active_count,
        "additional_jobs": int(additional_jobs),
        "max_submit_jobs": limit,
        "headroom_after_submission": (
            limit - active_count - int(additional_jobs)
        ),
    }


def _validate_parent(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("AMP diagnostic requires its sealed pilot parent")
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("AMP diagnostic parent file hash mismatch")
    parent = _read_json(path)
    if (
        parent.get("schema_version") != PILOT_FINALIZATION_SCHEMA
        or parent.get("study_id") != PILOT_STUDY_ID
        or parent.get("status") != "INCOMPLETE_EXPLORATORY_PILOT"
        or parent.get("decision")
        != "PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE"
        or parent.get("runtime_commit") != expected_runtime_commit
        or "residual_pl_rep_off" not in parent.get("failures", {})
        or parent.get("official_test_opened") is not False
        or parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(parent, field="finalization_sha256")
    ):
        raise ValueError("AMP diagnostic parent is not the sealed failed pilot")
    return parent


def _validate_diagnostic_parent(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "AMP stability gate requires the sealed matched diagnostic"
        )
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("AMP stability diagnostic-parent file hash mismatch")
    parent = _read_json(path)
    if (
        parent.get("schema_version") != AMP_DIAGNOSTIC_FINALIZATION_SCHEMA
        or parent.get("study_id") != AMP_DIAGNOSTIC_STUDY_ID
        or parent.get("status") != "COMPLETE_NUMERICAL_DIAGNOSTIC_ONLY"
        or parent.get("decision")
        != "ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED"
        or parent.get("runtime_commit") != expected_runtime_commit
        or parent.get("all_arms_passed") is not True
        or parent.get("repair_authorized") is not True
        or parent.get("performance_metrics") != {}
        or parent.get("performance_inference_allowed") is not False
        or parent.get("official_test_opened") is not False
        or parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(parent, field="finalization_sha256")
    ):
        raise ValueError(
            "AMP stability parent is not the repair-authorizing diagnostic"
        )
    return parent


def _validate_diagnostic_parent_deployment(
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    deployment_path = Path(str(parent.get("deployment_path", ""))).resolve()
    if (
        not deployment_path.is_file()
        or sha256_file(deployment_path)
        != parent.get("deployment_file_sha256")
    ):
        raise ValueError(
            "repair-authorizing diagnostic deployment artifact changed"
        )
    deployment = _read_json(deployment_path)
    if (
        deployment.get("schema_version")
        != AMP_DIAGNOSTIC_DEPLOYMENT_SCHEMA
        or deployment.get("study_id") != AMP_DIAGNOSTIC_STUDY_ID
        or deployment.get("runtime_commit") != parent.get("runtime_commit")
        or not isinstance(deployment.get("input_receipts"), Mapping)
        or not _self_hash_matches(deployment, field="deployment_sha256")
    ):
        raise ValueError(
            "repair-authorizing diagnostic deployment is invalid"
        )
    return deployment


def _validate_stability_v1_parent(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "official-semantics stability v2 requires sealed stability-v1 HOLD"
        )
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("stability-v1 parent file hash mismatch")
    parent = _read_json(path)
    if (
        parent.get("schema_version") != AMP_STABILITY_FINALIZATION_SCHEMA
        or parent.get("study_id") != AMP_STABILITY_STUDY_ID
        or parent.get("status")
        != "INCOMPLETE_REAL_DATA_AMP_STABILITY_GATE"
        or parent.get("decision") != "STABILITY_GATE_INCOMPLETE_HOLD"
        or parent.get("runtime_commit") != expected_runtime_commit
        or parent.get("stability_gate_passed") is not False
        or parent.get("official_protocol_freeze_authorized") is not False
        or parent.get("performance_metrics") != {}
        or parent.get("performance_inference_allowed") is not False
        or parent.get("official_test_opened") is not False
        or parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(parent, field="finalization_sha256")
    ):
        raise ValueError("stability-v1 parent is not the sealed fail-closed HOLD")
    return parent


def _validate_gradient_parent(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
    expected_arm_receipt_file_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    # Keep the torch-dependent diagnostic module lazy so control-plane parsing
    # and legacy profile validation do not require loading CUDA/PyTorch.
    from tools.bata.georoute_gradient_decomposition import (
        COMPLETE_STATUS as GRADIENT_COMPLETE_STATUS,
        DECISION_REPAIR as GRADIENT_DECISION_REPAIR,
        DEPLOYMENT_SCHEMA as GRADIENT_DEPLOYMENT_SCHEMA,
        DEPLOYMENT_STATUS as GRADIENT_DEPLOYMENT_STATUS,
        FINALIZATION_SCHEMA as GRADIENT_FINALIZATION_SCHEMA,
        PASS_STATUS as GRADIENT_PASS_STATUS,
        PROFILE as GRADIENT_PROFILE,
        STUDY_ID as GRADIENT_STUDY_ID,
        validate_receipt as validate_gradient_receipt,
    )

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "DDP FP16-cast repair gate requires the sealed gradient diagnosis"
        )
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("gradient-decomposition parent finalization changed")
    parent = _read_json(path)
    classification = parent.get("classification")
    if (
        parent.get("schema_version") != GRADIENT_FINALIZATION_SCHEMA
        or parent.get("study_id") != GRADIENT_STUDY_ID
        or parent.get("profile") != GRADIENT_PROFILE
        or parent.get("status") != GRADIENT_COMPLETE_STATUS
        or parent.get("decision") != GRADIENT_DECISION_REPAIR
        or parent.get("runtime_commit") != expected_runtime_commit
        or parent.get("all_arms_passed") is not True
        or parent.get("repair_class_identified") is not True
        or parent.get("repair_class") != AMP_REPAIR_REGISTERED_CLASS
        or parent.get("repair_authorized") is not True
        or not isinstance(classification, Mapping)
        or classification.get("repair_class") != AMP_REPAIR_REGISTERED_CLASS
        or parent.get("performance_metrics") != {}
        or parent.get("performance_inference_allowed") is not False
        or parent.get("official_test_opened") is not False
        or parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(parent, field="finalization_sha256")
    ):
        raise ValueError(
            "gradient parent did not uniquely authorize DDP_FP16_CAST_OVERFLOW "
            "repair"
        )
    deployment_path = Path(str(parent.get("deployment_path", ""))).resolve()
    if (
        not deployment_path.is_file()
        or sha256_file(deployment_path) != parent.get("deployment_file_sha256")
    ):
        raise ValueError("gradient parent deployment changed")
    deployment = _read_json(deployment_path)
    if (
        deployment.get("schema_version") != GRADIENT_DEPLOYMENT_SCHEMA
        or deployment.get("status") != GRADIENT_DEPLOYMENT_STATUS
        or deployment.get("study_id") != GRADIENT_STUDY_ID
        or deployment.get("profile") != GRADIENT_PROFILE
        or deployment.get("runtime_commit") != expected_runtime_commit
        or not isinstance(deployment.get("input_receipts"), Mapping)
        or not _self_hash_matches(deployment, field="deployment_sha256")
    ):
        raise ValueError("gradient parent deployment is invalid")

    receipts: dict[str, dict[str, Any]] = {}
    for arm in GRADIENT_ARMS:
        arm_record = parent.get("arms", {}).get(arm)
        if not isinstance(arm_record, Mapping):
            raise ValueError(f"gradient parent lacks arm {arm}")
        receipt_path = Path(
            str(arm_record.get("diagnostic_receipt_path", ""))
        ).resolve()
        expected_hash = expected_arm_receipt_file_sha256[arm]
        if (
            not receipt_path.is_file()
            or sha256_file(receipt_path) != expected_hash
            or arm_record.get("diagnostic_receipt_file_sha256") != expected_hash
        ):
            raise ValueError(f"gradient parent {arm} receipt changed")
        receipt = validate_gradient_receipt(
            _read_json(receipt_path),
            expected_arm=arm,
            expected_commit=expected_runtime_commit,
        )
        if receipt.get("status") != GRADIENT_PASS_STATUS:
            raise ValueError(f"gradient parent {arm} did not pass execution")
        receipts[arm] = receipt
    return parent, deployment, receipts


def _validate_repair_kat(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "DDP FP16-cast repair gate requires its sealed CUDA KAT"
        )
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("DDP FP16-cast repair CUDA KAT changed")
    kat = validate_repair_kat_receipt(
        _read_json(path),
        expected_commit=expected_runtime_commit,
    )
    if (
        kat.get("schema_version") != REPAIR_KAT_SCHEMA
        or kat.get("status") != REPAIR_KAT_PASS_STATUS
    ):
        raise ValueError("DDP FP16-cast repair CUDA KAT did not pass")
    return kat


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-pilot-finalization", type=Path, required=True)
    parser.add_argument("--expected-parent-file-sha256", required=True)
    parser.add_argument("--expected-parent-runtime-commit", required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--protocol-profile",
        choices=(
            AMP_DIAGNOSTIC_PROFILE,
            AMP_STABILITY_PROFILE,
            AMP_STABILITY_V2_PROFILE,
            AMP_REPAIR_PROFILE,
        ),
        default=AMP_DIAGNOSTIC_PROFILE,
    )
    parser.add_argument("--parent-diagnostic-finalization", type=Path)
    parser.add_argument("--expected-diagnostic-file-sha256")
    parser.add_argument("--expected-diagnostic-runtime-commit")
    parser.add_argument("--parent-stability-v1-finalization", type=Path)
    parser.add_argument("--expected-stability-v1-file-sha256")
    parser.add_argument("--expected-stability-v1-runtime-commit")
    parser.add_argument("--official-reference-config", type=Path)
    parser.add_argument("--expected-origin-ref")
    parser.add_argument("--parent-gradient-finalization", type=Path)
    parser.add_argument("--expected-gradient-file-sha256")
    parser.add_argument("--expected-gradient-runtime-commit")
    parser.add_argument("--expected-gradient-pl-receipt-file-sha256")
    parser.add_argument("--expected-gradient-st-receipt-file-sha256")
    parser.add_argument("--repair-kat-receipt", type=Path)
    parser.add_argument("--expected-repair-kat-file-sha256")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = amp_protocol_spec(args.protocol_profile)
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary):
        raise ValueError("AMP diagnostic root leaves the remote write boundary")
    if run_root.exists():
        raise FileExistsError("AMP diagnostic namespace exists; refusing resume")
    expected_commit = _full_hex(
        args.expected_commit,
        length=40,
        name="--expected-commit",
    )
    expected_parent_file_sha256 = _full_hex(
        args.expected_parent_file_sha256,
        length=64,
        name="--expected-parent-file-sha256",
    )
    expected_parent_runtime_commit = _full_hex(
        args.expected_parent_runtime_commit,
        length=40,
        name="--expected-parent-runtime-commit",
    )
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("AMP diagnostic source differs from --expected-commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("AMP diagnostic deployment requires clean source")
    expected_origin_ref = None
    if args.protocol_profile in {
        AMP_STABILITY_V2_PROFILE,
        AMP_REPAIR_PROFILE,
    }:
        expected_origin_ref = str(args.expected_origin_ref or "")
        if (
            not expected_origin_ref.startswith("refs/remotes/origin/")
            or any(
                character in expected_origin_ref
                for character in (" ", "\t", "\n", "\r", "\x00")
            )
        ):
            raise ValueError(
                "official-prefix AMP gate requires a full origin ref"
            )
        if (
            _git_output("rev-parse", "--verify", expected_origin_ref).lower()
            != expected_commit
        ):
            raise RuntimeError(
                "official-prefix AMP gate origin ref differs from source"
            )
    elif args.expected_origin_ref is not None:
        raise ValueError("expected-origin-ref is official-prefix AMP-gate only")
    parent_path = args.parent_pilot_finalization.resolve()
    parent = _validate_parent(
        parent_path,
        expected_file_sha256=expected_parent_file_sha256,
        expected_runtime_commit=expected_parent_runtime_commit,
    )
    diagnostic_parent = None
    diagnostic_parent_deployment = None
    diagnostic_parent_path = None
    expected_diagnostic_file_sha256 = None
    if args.protocol_profile in {
        AMP_STABILITY_PROFILE,
        AMP_STABILITY_V2_PROFILE,
    }:
        if (
            args.parent_diagnostic_finalization is None
            or args.expected_diagnostic_file_sha256 is None
            or args.expected_diagnostic_runtime_commit is None
        ):
            raise ValueError(
                "AMP stability gate requires exact diagnostic-parent arguments"
            )
        diagnostic_parent_path = (
            args.parent_diagnostic_finalization.resolve()
        )
        expected_diagnostic_file_sha256 = _full_hex(
            args.expected_diagnostic_file_sha256,
            length=64,
            name="--expected-diagnostic-file-sha256",
        )
        expected_diagnostic_runtime_commit = _full_hex(
            args.expected_diagnostic_runtime_commit,
            length=40,
            name="--expected-diagnostic-runtime-commit",
        )
        diagnostic_parent = _validate_diagnostic_parent(
            diagnostic_parent_path,
            expected_file_sha256=expected_diagnostic_file_sha256,
            expected_runtime_commit=expected_diagnostic_runtime_commit,
        )
        diagnostic_parent_deployment = (
            _validate_diagnostic_parent_deployment(diagnostic_parent)
        )
    elif any(
        value is not None
        for value in (
            args.parent_diagnostic_finalization,
            args.expected_diagnostic_file_sha256,
            args.expected_diagnostic_runtime_commit,
        )
    ):
        raise ValueError(
            "diagnostic-parent arguments are stability-gate only"
        )

    stability_v1_parent = None
    stability_v1_parent_path = None
    expected_stability_v1_file_sha256 = None
    if args.protocol_profile == AMP_STABILITY_V2_PROFILE:
        if (
            args.parent_stability_v1_finalization is None
            or args.expected_stability_v1_file_sha256 is None
            or args.expected_stability_v1_runtime_commit is None
            or args.official_reference_config is None
        ):
            raise ValueError(
                "official-semantics stability v2 requires sealed v1 HOLD "
                "and official-reference arguments"
            )
        stability_v1_parent_path = (
            args.parent_stability_v1_finalization.resolve()
        )
        expected_stability_v1_file_sha256 = _full_hex(
            args.expected_stability_v1_file_sha256,
            length=64,
            name="--expected-stability-v1-file-sha256",
        )
        expected_stability_v1_runtime_commit = _full_hex(
            args.expected_stability_v1_runtime_commit,
            length=40,
            name="--expected-stability-v1-runtime-commit",
        )
        stability_v1_parent = _validate_stability_v1_parent(
            stability_v1_parent_path,
            expected_file_sha256=expected_stability_v1_file_sha256,
            expected_runtime_commit=expected_stability_v1_runtime_commit,
        )
    elif any(
        value is not None
        for value in (
            args.parent_stability_v1_finalization,
            args.expected_stability_v1_file_sha256,
            args.expected_stability_v1_runtime_commit,
        )
    ):
        raise ValueError(
            "stability-v1 parent arguments are stability-v2 only"
        )

    gradient_parent = None
    gradient_parent_deployment = None
    gradient_parent_receipts = None
    gradient_parent_path = None
    expected_gradient_file_sha256 = None
    expected_gradient_arm_hashes = None
    repair_kat = None
    repair_kat_path = None
    expected_repair_kat_file_sha256 = None
    if args.protocol_profile == AMP_REPAIR_PROFILE:
        required_repair_arguments = (
            args.parent_gradient_finalization,
            args.expected_gradient_file_sha256,
            args.expected_gradient_runtime_commit,
            args.expected_gradient_pl_receipt_file_sha256,
            args.expected_gradient_st_receipt_file_sha256,
            args.repair_kat_receipt,
            args.expected_repair_kat_file_sha256,
            args.official_reference_config,
        )
        if any(value is None for value in required_repair_arguments):
            raise ValueError(
                "DDP FP16-cast repair gate requires exact gradient-parent, "
                "CUDA KAT, and official-reference arguments"
            )
        gradient_parent_path = args.parent_gradient_finalization.resolve()
        expected_gradient_file_sha256 = _full_hex(
            args.expected_gradient_file_sha256,
            length=64,
            name="--expected-gradient-file-sha256",
        )
        expected_gradient_runtime_commit = _full_hex(
            args.expected_gradient_runtime_commit,
            length=40,
            name="--expected-gradient-runtime-commit",
        )
        expected_gradient_arm_hashes = {
            GRADIENT_ARMS[0]: _full_hex(
                args.expected_gradient_pl_receipt_file_sha256,
                length=64,
                name="--expected-gradient-pl-receipt-file-sha256",
            ),
            GRADIENT_ARMS[1]: _full_hex(
                args.expected_gradient_st_receipt_file_sha256,
                length=64,
                name="--expected-gradient-st-receipt-file-sha256",
            ),
        }
        (
            gradient_parent,
            gradient_parent_deployment,
            gradient_parent_receipts,
        ) = _validate_gradient_parent(
            gradient_parent_path,
            expected_file_sha256=expected_gradient_file_sha256,
            expected_runtime_commit=expected_gradient_runtime_commit,
            expected_arm_receipt_file_sha256=expected_gradient_arm_hashes,
        )
        transitive_chain = gradient_parent_deployment.get(
            "parent_evidence", {}
        ).get("transitive_parent_chain", {})
        transitive_pilot = (
            transitive_chain.get("parent_pilot")
            if isinstance(transitive_chain, Mapping)
            else None
        )
        if (
            not isinstance(transitive_pilot, Mapping)
            or transitive_pilot.get("file_sha256")
            != expected_parent_file_sha256
            or transitive_pilot.get("runtime_commit")
            != expected_parent_runtime_commit
        ):
            raise ValueError(
                "repair gate pilot parent differs from the gradient "
                "diagnosis transitive chain"
            )
        repair_kat_path = args.repair_kat_receipt.resolve()
        expected_repair_kat_file_sha256 = _full_hex(
            args.expected_repair_kat_file_sha256,
            length=64,
            name="--expected-repair-kat-file-sha256",
        )
        repair_kat = _validate_repair_kat(
            repair_kat_path,
            expected_file_sha256=expected_repair_kat_file_sha256,
            expected_runtime_commit=expected_commit,
        )
    elif any(
        value is not None
        for value in (
            args.parent_gradient_finalization,
            args.expected_gradient_file_sha256,
            args.expected_gradient_runtime_commit,
            args.expected_gradient_pl_receipt_file_sha256,
            args.expected_gradient_st_receipt_file_sha256,
            args.repair_kat_receipt,
            args.expected_repair_kat_file_sha256,
        )
    ):
        raise ValueError("gradient-parent and repair-KAT arguments are repair-only")
    if (
        args.protocol_profile
        not in {AMP_STABILITY_V2_PROFILE, AMP_REPAIR_PROFILE}
        and args.official_reference_config is not None
    ):
        raise ValueError(
            "official reference is only allowed for official-prefix AMP gates"
        )

    inputs = {
        "GEOROUTE_SOURCE_CONFIG": args.source_config.resolve(),
        "GEOROUTE_MANIFEST": args.manifest.resolve(),
        "GEOROUTE_DEVELOPMENT_ANNOTATION": (
            args.development_annotation.resolve()
        ),
        "GEOROUTE_CLASS_MAP": args.class_map.resolve(),
        "GEOROUTE_DEVELOPMENT_VIDEO_ROOT": (
            args.development_video_root.resolve()
        ),
        "GEOROUTE_PRETRAINED": args.pretrained.resolve(),
    }
    if args.protocol_profile in {
        AMP_STABILITY_V2_PROFILE,
        AMP_REPAIR_PROFILE,
    }:
        inputs["GEOROUTE_OFFICIAL_REFERENCE_CONFIG"] = (
            args.official_reference_config.resolve()
        )
    for name, input_path in inputs.items():
        if name == "GEOROUTE_DEVELOPMENT_VIDEO_ROOT":
            if not input_path.is_dir() or any(
                part.lower()
                in {"test", "testing", "test_videos", "official_test"}
                for part in input_path.parts
            ):
                raise ValueError(
                    "AMP diagnostic development video root is invalid"
                )
        elif not input_path.is_file():
            raise FileNotFoundError(input_path)
    if (
        args.protocol_profile
        in {
            AMP_STABILITY_PROFILE,
            AMP_STABILITY_V2_PROFILE,
            AMP_REPAIR_PROFILE,
        }
        and not _inside(inputs["GEOROUTE_SOURCE_CONFIG"], ROOT)
    ):
        raise ValueError(
            "AMP stability source config must come from the exact runtime checkout"
        )
    if (
        args.protocol_profile
        in {AMP_STABILITY_V2_PROFILE, AMP_REPAIR_PROFILE}
        and not _inside(inputs["GEOROUTE_OFFICIAL_REFERENCE_CONFIG"], ROOT)
    ):
        raise ValueError(
            "official reference config must come from the exact runtime checkout"
        )
    input_receipts = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for name, path in inputs.items()
    }
    matched_parent_input_names = (
        "GEOROUTE_MANIFEST",
        "GEOROUTE_DEVELOPMENT_ANNOTATION",
        "GEOROUTE_CLASS_MAP",
        "GEOROUTE_DEVELOPMENT_VIDEO_ROOT",
        "GEOROUTE_PRETRAINED",
    )
    if diagnostic_parent_deployment is not None:
        parent_inputs = diagnostic_parent_deployment["input_receipts"]
        if any(
            input_receipts[name] != parent_inputs.get(name)
            for name in matched_parent_input_names
        ):
            raise ValueError(
                "AMP stability immutable inputs differ from matched diagnostic"
            )
    if gradient_parent_deployment is not None:
        gradient_inputs = gradient_parent_deployment["input_receipts"]
        if any(
            input_receipts[name] != gradient_inputs.get(name)
            for name in matched_parent_input_names
        ):
            raise ValueError(
                "DDP FP16-cast repair inputs differ from the gradient diagnosis"
            )
        for name in (
            "GEOROUTE_SOURCE_CONFIG",
            "GEOROUTE_OFFICIAL_REFERENCE_CONFIG",
        ):
            if input_receipts[name]["sha256"] != gradient_inputs.get(
                name, {}
            ).get("sha256"):
                raise ValueError(
                    f"DDP FP16-cast repair {name} content differs from the "
                    "gradient diagnosis"
                )

    # All admission gates precede immutable namespace creation.
    capacity = _require_submit_capacity(additional_jobs=3)
    storage = storage_capacity_receipt(run_root, cell_count=2)
    stage_script = ROOT / "scripts" / "run_georoute_amp_diagnostic_stage_slurm.sh"
    control_script = (
        ROOT / "scripts" / "run_georoute_amp_diagnostic_control_slurm.sh"
    )
    for script in (stage_script, control_script):
        if not script.is_file():
            raise FileNotFoundError(script)

    base_values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_AMP_DIAGNOSTIC_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        "GEOROUTE_AMP_PROTOCOL_PROFILE": args.protocol_profile,
        **{name: str(path) for name, path in inputs.items()},
    }
    base_exports = {
        key: _clean_export(value, name=key)
        for key, value in base_values.items()
    }
    stage_exports = {
        arm: {
            **base_exports,
            "GEOROUTE_AMP_DIAGNOSTIC_ARM": arm,
        }
        for arm in AMP_DIAGNOSTIC_ARMS
    }
    finalizer_exports = {
        **base_exports,
        "GEOROUTE_AMP_DIAGNOSTIC_ACTION": "finalize",
    }
    logs = run_root / "slurm"
    for arm in AMP_DIAGNOSTIC_ARMS:
        _sbatch(
            name=f"{spec['job_prefix']}_{PILOT_ARMS[arm]['slug']}",
            script=stage_script,
            logs=logs,
            exports=stage_exports[arm],
            gpu=True,
            test_only=True,
        )
    _sbatch(
        name=f"{spec['job_prefix']}_finalize",
        script=control_script,
        logs=logs,
        exports=finalizer_exports,
        gpu=False,
        test_only=True,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    for directory in (str(spec["cell_directory"]), "control", "slurm"):
        (run_root / directory).mkdir()
    _atomic_write_json(
        run_root / "control" / "submit_capacity_preflight.json",
        capacity,
    )
    _atomic_write_json(
        run_root / "control" / "deployment_storage_preflight.json",
        storage,
    )

    submitted: list[str] = []
    try:
        stage_jobs: dict[str, str] = {}
        for arm in AMP_DIAGNOSTIC_ARMS:
            job_id = _sbatch(
                name=f"{spec['job_prefix']}_{PILOT_ARMS[arm]['slug']}",
                script=stage_script,
                logs=logs,
                exports=stage_exports[arm],
                gpu=True,
                hold=True,
            )
            stage_jobs[arm] = job_id
            submitted.append(job_id)
        finalizer_job = _sbatch(
            name=f"{spec['job_prefix']}_finalize",
            script=control_script,
            logs=logs,
            exports=finalizer_exports,
            gpu=False,
            dependency=list(stage_jobs.values()),
            dependency_type="afterany",
        )
        submitted.append(finalizer_job)
        jobs = validate_amp_diagnostic_job_receipt(
            {
                "stage": stage_jobs,
                "finalizer": finalizer_job,
            }
        )
        deployment: dict[str, Any] = {
            "schema_version": spec["deployment_schema"],
            "status": spec["deployment_status"],
            "study_id": spec["study_id"],
            "protocol_profile": spec["profile"],
            "runtime_commit": expected_commit,
            "run_root": str(run_root),
            "arms": list(AMP_DIAGNOSTIC_ARMS),
            "seed": int(spec["seed"]),
            "jobs": jobs,
            "input_receipts": input_receipts,
            "expected_origin_ref": expected_origin_ref,
            "origin_ref_parity_verified": (
                True
                if args.protocol_profile
                in {AMP_STABILITY_V2_PROFILE, AMP_REPAIR_PROFILE}
                else None
            ),
            "matched_diagnostic_inputs": (
                {
                    "names": list(matched_parent_input_names),
                    "all_equal": True,
                    "diagnostic_deployment_path": str(
                        diagnostic_parent["deployment_path"]
                    ),
                    "diagnostic_deployment_file_sha256": (
                        diagnostic_parent["deployment_file_sha256"]
                    ),
                }
                if diagnostic_parent is not None
                else None
            ),
            "matched_gradient_inputs": (
                {
                    "exact_path_and_content_names": list(
                        matched_parent_input_names
                    ),
                    "content_matched_runtime_config_names": [
                        "GEOROUTE_SOURCE_CONFIG",
                        "GEOROUTE_OFFICIAL_REFERENCE_CONFIG",
                    ],
                    "all_equal": True,
                    "gradient_deployment_path": str(
                        gradient_parent["deployment_path"]
                    ),
                    "gradient_deployment_file_sha256": gradient_parent[
                        "deployment_file_sha256"
                    ],
                }
                if gradient_parent is not None
                else None
            ),
            "parent_pilot": {
                "path": str(parent_path),
                "file_sha256": expected_parent_file_sha256,
                "finalization_sha256": parent["finalization_sha256"],
                "runtime_commit": parent["runtime_commit"],
                "decision": parent["decision"],
            },
            "parent_diagnostic": (
                {
                    "path": str(diagnostic_parent_path),
                    "file_sha256": expected_diagnostic_file_sha256,
                    "finalization_sha256": diagnostic_parent[
                        "finalization_sha256"
                    ],
                    "runtime_commit": diagnostic_parent["runtime_commit"],
                    "decision": diagnostic_parent["decision"],
                }
                if diagnostic_parent is not None
                else None
            ),
            "parent_stability_v1": (
                {
                    "path": str(stability_v1_parent_path),
                    "file_sha256": expected_stability_v1_file_sha256,
                    "finalization_sha256": stability_v1_parent[
                        "finalization_sha256"
                    ],
                    "runtime_commit": stability_v1_parent["runtime_commit"],
                    "decision": stability_v1_parent["decision"],
                }
                if stability_v1_parent is not None
                else None
            ),
            "parent_gradient_decomposition": (
                {
                    "path": str(gradient_parent_path),
                    "file_sha256": expected_gradient_file_sha256,
                    "finalization_sha256": gradient_parent[
                        "finalization_sha256"
                    ],
                    "runtime_commit": gradient_parent["runtime_commit"],
                    "decision": gradient_parent["decision"],
                    "repair_class": gradient_parent["repair_class"],
                    "arm_receipts": {
                        arm: {
                            "path": gradient_parent["arms"][arm][
                                "diagnostic_receipt_path"
                            ],
                            "file_sha256": expected_gradient_arm_hashes[arm],
                            "receipt_sha256": gradient_parent_receipts[arm][
                                "receipt_sha256"
                            ],
                        }
                        for arm in GRADIENT_ARMS
                    },
                }
                if gradient_parent is not None
                else None
            ),
            "repair_cuda_kat": (
                {
                    "path": str(repair_kat_path),
                    "file_sha256": expected_repair_kat_file_sha256,
                    "kat_sha256": repair_kat["kat_sha256"],
                    "slurm_job_id": repair_kat["slurm_job_id"],
                    "status": repair_kat["status"],
                }
                if repair_kat is not None
                else None
            ),
            "registered_repair_intervention": (
                {
                    "repair_class": AMP_REPAIR_REGISTERED_CLASS,
                    "name": AMP_REPAIR_INTERVENTION,
                    "count": 1,
                    "field": "solver.fp16_compress",
                    "before": True,
                    "after": False,
                    "all_other_registered_factors_frozen": True,
                }
                if args.protocol_profile == AMP_REPAIR_PROFILE
                else None
            ),
            "official_reference_binding": (
                {
                    "bound": True,
                    "path": input_receipts[
                        "GEOROUTE_OFFICIAL_REFERENCE_CONFIG"
                    ]["path"],
                    "sha256": input_receipts[
                        "GEOROUTE_OFFICIAL_REFERENCE_CONFIG"
                    ]["sha256"],
                    "full_official_training_claimed": False,
                }
                if args.protocol_profile
                in {AMP_STABILITY_V2_PROFILE, AMP_REPAIR_PROFILE}
                else None
            ),
            "submit_capacity_preflight": capacity,
            "storage_preflight": storage,
            "dependency_policy": {
                "two_diagnostic_arms_parallel": True,
                "stages_held_until_immutable_receipts": True,
                "finalizer_afterany_both_stages": True,
                "resume_allowed": False,
            },
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "performance_inference_allowed": False,
            "paper_claim_allowed": False,
        }
        deployment["deployment_sha256"] = canonical_sha256(deployment)
        deployment_path = run_root / "control" / "deployment.json"
        _atomic_write_json(deployment_path, deployment)

        submission: dict[str, Any] = {
            "schema_version": spec["deployment_schema"],
            "status": spec["finalizer_submission_status"],
            "protocol_profile": spec["profile"],
            "runtime_commit": expected_commit,
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_job_id": finalizer_job,
            "dependency_type": "afterany",
            "predecessor_job_ids": list(stage_jobs.values()),
        }
        submission["receipt_sha256"] = canonical_sha256(submission)
        submission_path = run_root / "control" / "finalizer_submission.json"
        _atomic_write_json(submission_path, submission)
        _release_jobs(list(stage_jobs.values()))
        release: dict[str, Any] = {
            "schema_version": spec["deployment_schema"],
            "status": spec["stage_release_status"],
            "protocol_profile": spec["profile"],
            "runtime_commit": expected_commit,
            "stage_job_ids": list(stage_jobs.values()),
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_submission_file_sha256": sha256_file(submission_path),
        }
        release["receipt_sha256"] = canonical_sha256(release)
        _atomic_write_json(
            run_root / "control" / "stage_release.json",
            release,
        )
    except BaseException:
        _cancel_jobs(submitted)
        raise
    print(
        json.dumps(
            {
                **deployment,
                "finalizer_job_id": finalizer_job,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
