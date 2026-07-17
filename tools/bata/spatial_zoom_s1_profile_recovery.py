from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    atomic_publish_json,
    build_s1_profile_order,
    canonical_sha256,
    sha256_file,
)
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    current_git_commit,
    validate_bound_s1_training_config,
)

S1_PROFILE_RECOVERY_SCHEMA = "spatial_zoom_s1_profile_recovery_v1"
S1_CHAINED_PROFILE_RECOVERY_SCHEMA = "spatial_zoom_s1_profile_recovery_v2"
S1_SIDECAR_PROFILE_RECOVERY_SCHEMA = "spatial_zoom_s1_profile_recovery_v3"
S1_SUPERSEDED_PROFILE_MARKER_SCHEMA = "spatial_zoom_s1_profile_attempt_v4"
S1_PROFILE_RECOVERY_REASON = "duplicate_physical_window_identity"
S1_PROFILE_FAILURE_SIGNATURE = "formal S1 profile window identities must be unique"
S1_CHAINED_PROFILE_MARKER_SCHEMA = "spatial_zoom_s1_profile_attempt_v5"
S1_SIDECAR_PROFILE_MARKER_SCHEMA = "spatial_zoom_s1_profile_attempt_v6"
S1_CHAINED_RECOVERY_REASON = "duplicate_window_and_power_sampler_cadence"
S1_SIDECAR_RECOVERY_REASON = "out_of_process_power_sidecar"
S1_POWER_FAILURE_SIGNATURE = (
    "formal S1 power trace is too sparse for auditable energy integration"
)
S1_POWER_DIAGNOSTIC_SCHEMA = "spatial_zoom_s1_power_sampler_diagnostic_v1"
S1_SIDECAR_POWER_BACKEND = "nvml-sidecar-process-v1"

_ALLOWED_EXACT_PATHS = {
    "docs/methods/spatial_zoom_s1_contract.md",
    "docs/superpowers/specs/2026-07-17-spatial-zoom-s1-power-sidecar-design.md",
    "scripts/run_spatial_zoom_s1_power_sampler_diag_slurm.sh",
    "scripts/run_spatial_zoom_s1_power_sidecar_gate_slurm.sh",
    "scripts/run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh",
    "scripts/run_spatial_zoom_s1_test_profile_slurm.sh",
    "tests/test_spatial_zoom_s1_infrastructure.py",
    "tools/bata/analyze_spatial_zoom_s1_results.py",
    "tools/bata/build_spatial_zoom_s1_run_descriptor.py",
    "tools/bata/preflight_spatial_zoom_s1_profile.py",
    "tools/bata/profile_spatial_zoom_s1.py",
    "tools/bata/run_spatial_zoom_s1_precheck.py",
    "tools/bata/spatial_zoom_s1_cost.py",
    "tools/bata/spatial_zoom_s1_profile_recovery.py",
    "tools/bata/spatial_zoom_s1_power.py",
    "tools/bata/spatial_zoom_s1_sidecar_gate.py",
    "tools/bata/spatial_zoom_s1_training.py",
}
_ALLOWED_PREFIXES = ("research-wiki/",)
_REQUIRED_REPAIR_PATHS_V1 = {
    "scripts/run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh",
    "scripts/run_spatial_zoom_s1_test_profile_slurm.sh",
    "tests/test_spatial_zoom_s1_infrastructure.py",
    "tools/bata/analyze_spatial_zoom_s1_results.py",
    "tools/bata/build_spatial_zoom_s1_run_descriptor.py",
    "tools/bata/preflight_spatial_zoom_s1_profile.py",
    "tools/bata/profile_spatial_zoom_s1.py",
    "tools/bata/run_spatial_zoom_s1_precheck.py",
    "tools/bata/spatial_zoom_s1_cost.py",
    "tools/bata/spatial_zoom_s1_profile_recovery.py",
    "tools/bata/spatial_zoom_s1_training.py",
}
_REQUIRED_REPAIR_PATHS_CHAINED = _REQUIRED_REPAIR_PATHS_V1 | {
    "tools/bata/spatial_zoom_s1_power.py",
}
_REQUIRED_REPAIR_PATHS_SIDECAR = _REQUIRED_REPAIR_PATHS_CHAINED | {
    "docs/superpowers/specs/2026-07-17-spatial-zoom-s1-power-sidecar-design.md",
    "scripts/run_spatial_zoom_s1_power_sidecar_gate_slurm.sh",
    "tools/bata/spatial_zoom_s1_sidecar_gate.py",
}


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Git command failed")
    return completed.stdout


def require_clean_profile_checkout(*, expected_commit: str) -> None:
    if current_git_commit() != str(expected_commit).lower():
        raise RuntimeError("S1 profile checkout differs from the recovery commit")
    if _run_git("status", "--porcelain", "--untracked-files=all").strip():
        raise RuntimeError("formal S1 profile recovery requires a clean checkout")


def _changed_files(
    base_commit: str,
    profile_commit: str,
    *,
    required_paths: set[str],
) -> list[dict[str, str]]:
    lines = _run_git(
        "diff",
        "--name-status",
        "--find-renames=100%",
        f"{base_commit}..{profile_commit}",
    ).splitlines()
    changed: list[dict[str, str]] = []
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"unsupported S1 recovery Git diff row: {line!r}")
        status, path = fields
        normalized = Path(path).as_posix()
        if status not in {"A", "M"}:
            raise ValueError(
                f"S1 profile recovery forbids Git status {status} for {normalized}"
            )
        if normalized not in _ALLOWED_EXACT_PATHS and not normalized.startswith(
            _ALLOWED_PREFIXES
        ):
            raise ValueError(
                f"S1 profile recovery changed an unauthorized path: {normalized}"
            )
        file_path = ROOT / normalized
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        changed.append(
            {
                "status": status,
                "path": normalized,
                "file_sha256": sha256_file(file_path),
            }
        )
    changed_paths = {row["path"] for row in changed}
    missing = sorted(required_paths - changed_paths)
    if missing:
        raise ValueError(
            f"S1 profile recovery is missing audited repair paths: {missing}"
        )
    return changed


def _validate_superseded_marker(
    path: Path, *, expected_schema: str = S1_SUPERSEDED_PROFILE_MARKER_SCHEMA
) -> dict[str, Any]:
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker_hash = marker.pop("marker_sha256", None)
    if not marker_hash or canonical_sha256(marker) != marker_hash:
        raise ValueError("superseded S1 profile marker self-hash mismatch")
    marker["marker_sha256"] = marker_hash
    if marker.get("schema_version") != expected_schema:
        raise ValueError(
            f"S1 recovery requires the exact failed {expected_schema} marker"
        )
    return marker


def _validate_power_diagnostic(path: Path) -> dict[str, Any]:
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    diagnostic_hash = diagnostic.pop("diagnostic_sha256", None)
    if not diagnostic_hash or canonical_sha256(diagnostic) != diagnostic_hash:
        raise ValueError("S1 power diagnostic self-hash mismatch")
    diagnostic["diagnostic_sha256"] = diagnostic_hash
    code_commit = str(diagnostic.get("code_commit", ""))
    if (
        diagnostic.get("schema_version") != S1_POWER_DIAGNOSTIC_SCHEMA
        or diagnostic.get("reads_test_data") is not False
        or diagnostic.get("paper_claim_allowed") is not False
        or int(diagnostic.get("target_interval_ms", -1)) != 20
        or float(diagnostic.get("duration_seconds_per_backend", 0.0)) < 5.0
        or len(code_commit) != 40
        or any(character not in "0123456789abcdef" for character in code_commit)
        or not str(diagnostic.get("node", "")).strip()
        or not str(diagnostic.get("gpu_uuid", "")).startswith("GPU-")
        or not str(diagnostic.get("slurm_job_id", "")).isdigit()
    ):
        raise ValueError("S1 power diagnostic provenance is invalid")
    backends = {
        str(row.get("backend")): row
        for row in diagnostic.get("backends", ())
        if isinstance(row, Mapping)
    }
    inherited = backends.get("nvidia-smi-persistent-loop-ms")
    candidate = backends.get("nvml-persistent-poll-v1")
    if not inherited or not candidate:
        raise ValueError("S1 power diagnostic is missing a matched backend")
    if (
        inherited.get("status") != "FAIL"
        or inherited.get("cadence", {}).get("formal_cadence_pass") is not False
        or candidate.get("status") != "PASS"
        or candidate.get("cadence", {}).get("formal_cadence_pass") is not True
        or float(candidate.get("cadence", {}).get("max_gap_ms", math.inf)) > 100.0
        or float(candidate.get("cadence", {}).get("max_gap_limit_ms", -1.0))
        != 100.0
    ):
        raise ValueError("S1 power diagnostic does not justify the NVML backend")
    return diagnostic


def profile_campaign_prefix(
    certificate: Mapping[str, Any], *, resolution: int, seed: int
) -> Path:
    if int(resolution) not in (160, 224, 256) or int(seed) not in (
        3407,
        3408,
        3409,
    ):
        raise ValueError("invalid S1 profile campaign cell")
    return (
        Path(certificate["campaign_root"])
        / f"dense{int(resolution)}"
        / f"seed{int(seed)}"
        / f"dense{int(resolution)}_seed{int(seed)}"
    ).resolve()


def _certificate_output_path(certificate: Mapping[str, Any]) -> Path:
    return (Path(certificate["campaign_root"]) / "recovery_certificate.json").resolve()


def build_sidecar_profile_recovery_certificate(
    *,
    binding: Mapping[str, Any],
    failed_marker_path: str | Path,
    failure_log_path: str | Path,
    failed_job_id: str,
    expected_exposure_count: int,
    expected_physical_window_count: int,
    expected_duplicate_physical_window_ids: Sequence[str],
    superseded_recovery_certificate_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Create the v3 recovery that isolates power polling from the detector."""

    profile_commit = current_git_commit()
    require_clean_profile_checkout(expected_commit=profile_commit)
    training_commit = str(binding["code_commit"]).lower()
    changed_files = _changed_files(
        training_commit,
        profile_commit,
        required_paths=_REQUIRED_REPAIR_PATHS_SIDECAR,
    )
    parent_path = Path(superseded_recovery_certificate_path).resolve()
    failed_marker_path = Path(failed_marker_path).resolve()
    failure_log_path = Path(failure_log_path).resolve()
    for path in (parent_path, failed_marker_path, failure_log_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = validate_profile_recovery_certificate(
        json.loads(parent_path.read_text(encoding="utf-8")),
        binding=binding,
        verify_checkout=False,
    )
    if parent.get("reason") != S1_CHAINED_RECOVERY_REASON:
        raise ValueError("S1 sidecar recovery requires the v2 NVML parent")
    failed_marker = _validate_superseded_marker(
        failed_marker_path,
        expected_schema=S1_CHAINED_PROFILE_MARKER_SCHEMA,
    )
    failure_text = failure_log_path.read_text(
        encoding="utf-8", errors="replace"
    )
    if S1_POWER_FAILURE_SIGNATURE not in failure_text:
        raise ValueError("S1 sidecar recovery log lacks the frozen cadence failure")
    if not str(failed_job_id).isdigit():
        raise ValueError("S1 sidecar recovery requires a numeric failed Slurm job id")

    first_profile_cell = build_s1_profile_order()[0]
    marker_expected = {
        "resolution": int(first_profile_cell["resolution"]),
        "seed": int(first_profile_cell["seed"]),
        "code_commit": training_commit,
        "profile_code_commit": parent["profile_code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "profile_order_ordinal": 0,
        "test_open_certificate_sha256": parent[
            "test_open_certificate_sha256"
        ],
        "profile_recovery_certificate_sha256": parent["certificate_sha256"],
        "profile_recovery_campaign_id": parent["campaign_id"],
    }
    for key, expected in marker_expected.items():
        if failed_marker.get(key) != expected:
            raise ValueError(f"S1 sidecar recovery marker {key} mismatch")

    exposure_count = int(expected_exposure_count)
    physical_count = int(expected_physical_window_count)
    duplicates = sorted(set(map(str, expected_duplicate_physical_window_ids)))
    if (
        exposure_count != int(parent["expected_loader_exposure_count"])
        or physical_count != int(parent["expected_physical_window_count"])
        or duplicates != parent["expected_duplicate_physical_window_ids"]
        or exposure_count - physical_count != len(duplicates)
        or not duplicates
    ):
        raise ValueError("S1 sidecar recovery changed the frozen exposure topology")

    basis = {
        "schema_version": S1_SIDECAR_PROFILE_RECOVERY_SCHEMA,
        "reason": S1_SIDECAR_RECOVERY_REASON,
        "failure_signature": parent["failure_signature"],
        "failed_job_id": parent["failed_job_id"],
        "training_code_commit": training_commit,
        "profile_code_commit": profile_commit,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "test_open_certificate_sha256": parent[
            "test_open_certificate_sha256"
        ],
        "superseded_marker_path": parent["superseded_marker_path"],
        "superseded_marker_file_sha256": parent[
            "superseded_marker_file_sha256"
        ],
        "superseded_marker_sha256": parent["superseded_marker_sha256"],
        "failure_log_path": parent["failure_log_path"],
        "failure_log_sha256": parent["failure_log_sha256"],
        "expected_loader_exposure_count": exposure_count,
        "expected_physical_window_count": physical_count,
        "expected_duplicate_physical_window_ids": duplicates,
        "changed_files": changed_files,
        "repair_scope": "out_of_process_power_sidecar_and_failure_evidence_only",
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
        "preserve_recovery_chain": True,
        "superseded_recovery_certificate_path": str(parent_path),
        "superseded_recovery_certificate_file_sha256": sha256_file(parent_path),
        "superseded_recovery_certificate_sha256": parent[
            "certificate_sha256"
        ],
        "superseded_recovery_campaign_id": parent["campaign_id"],
        "superseded_recovery_profile_code_commit": parent[
            "profile_code_commit"
        ],
        "sidecar_power_failure_signature": S1_POWER_FAILURE_SIGNATURE,
        "sidecar_power_failed_job_id": str(failed_job_id),
        "sidecar_power_failure_marker_path": str(failed_marker_path),
        "sidecar_power_failure_marker_file_sha256": sha256_file(
            failed_marker_path
        ),
        "sidecar_power_failure_marker_sha256": failed_marker["marker_sha256"],
        "sidecar_power_failure_log_path": str(failure_log_path),
        "sidecar_power_failure_log_sha256": sha256_file(failure_log_path),
        "power_sampler_backend": S1_SIDECAR_POWER_BACKEND,
        "power_target_interval_ms": 20,
        "power_max_gap_limit_ms": 100.0,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
        "sidecar_gate_relative_path": "sidecar_gate.json",
    }
    campaign_id = canonical_sha256(basis)[:16]
    campaign_root = (
        Path(binding["canonical_experiment_root"])
        / "profile_campaigns"
        / campaign_id
    ).resolve()
    certificate = {
        **basis,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    output_path = _certificate_output_path(certificate)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != certificate:
            raise FileExistsError("S1 sidecar recovery campaign identity collision")
    else:
        atomic_publish_json(output_path, certificate)
    return output_path, certificate


def build_profile_recovery_certificate(
    *,
    binding: Mapping[str, Any],
    failed_marker_path: str | Path,
    failure_log_path: str | Path,
    failed_job_id: str,
    expected_exposure_count: int,
    expected_physical_window_count: int,
    expected_duplicate_physical_window_ids: Sequence[str],
    superseded_recovery_certificate_path: str | Path | None = None,
    power_diagnostic_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if (
        superseded_recovery_certificate_path is not None
        and power_diagnostic_path is None
    ):
        return build_sidecar_profile_recovery_certificate(
            binding=binding,
            failed_marker_path=failed_marker_path,
            failure_log_path=failure_log_path,
            failed_job_id=failed_job_id,
            expected_exposure_count=expected_exposure_count,
            expected_physical_window_count=expected_physical_window_count,
            expected_duplicate_physical_window_ids=(
                expected_duplicate_physical_window_ids
            ),
            superseded_recovery_certificate_path=(
                superseded_recovery_certificate_path
            ),
        )
    profile_commit = current_git_commit()
    require_clean_profile_checkout(expected_commit=profile_commit)
    training_commit = str(binding["code_commit"]).lower()
    chained = superseded_recovery_certificate_path is not None
    if chained != (power_diagnostic_path is not None):
        raise ValueError(
            "S1 chained recovery requires both parent certificate and power diagnostic"
        )
    required_paths = (
        _REQUIRED_REPAIR_PATHS_CHAINED if chained else _REQUIRED_REPAIR_PATHS_V1
    )
    changed_files = _changed_files(
        training_commit, profile_commit, required_paths=required_paths
    )

    failed_marker_path = Path(failed_marker_path).resolve()
    failure_log_path = Path(failure_log_path).resolve()
    if not failed_marker_path.is_file() or not failure_log_path.is_file():
        raise FileNotFoundError(
            "S1 recovery requires the failed marker and failure log"
        )
    parent = None
    diagnostic = None
    marker_schema = (
        S1_CHAINED_PROFILE_MARKER_SCHEMA
        if chained
        else S1_SUPERSEDED_PROFILE_MARKER_SCHEMA
    )
    failure_signature = (
        S1_POWER_FAILURE_SIGNATURE if chained else S1_PROFILE_FAILURE_SIGNATURE
    )
    failed_marker = _validate_superseded_marker(
        failed_marker_path, expected_schema=marker_schema
    )
    if failure_signature not in failure_log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise ValueError(
            "S1 recovery failure log does not contain the frozen signature"
        )
    if chained:
        parent_path = Path(superseded_recovery_certificate_path).resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        parent = validate_profile_recovery_certificate(
            json.loads(parent_path.read_text(encoding="utf-8")),
            binding=binding,
            verify_checkout=False,
        )
        if parent.get("reason") != S1_PROFILE_RECOVERY_REASON:
            raise ValueError("S1 chained recovery requires the identity-repair parent")
        diagnostic_path = Path(power_diagnostic_path).resolve()
        if not diagnostic_path.is_file():
            raise FileNotFoundError(diagnostic_path)
        diagnostic = _validate_power_diagnostic(diagnostic_path)
    first_profile_cell = build_s1_profile_order()[0]
    expected_marker = {
        "resolution": int(first_profile_cell["resolution"]),
        "seed": int(first_profile_cell["seed"]),
        "code_commit": training_commit,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "profile_order_ordinal": 0,
    }
    if chained:
        expected_marker.update(
            {
                "profile_code_commit": parent["profile_code_commit"],
                "profile_recovery_certificate_sha256": parent[
                    "certificate_sha256"
                ],
                "profile_recovery_campaign_id": parent["campaign_id"],
            }
        )
    for key, expected in expected_marker.items():
        if failed_marker.get(key) != expected:
            raise ValueError(f"superseded S1 profile marker {key} mismatch")

    exposure_count = int(expected_exposure_count)
    physical_count = int(expected_physical_window_count)
    duplicates = sorted(set(map(str, expected_duplicate_physical_window_ids)))
    if exposure_count <= 0 or physical_count <= 0 or physical_count >= exposure_count:
        raise ValueError("S1 recovery requires a real duplicate exposure topology")
    if exposure_count - physical_count != len(duplicates) or not duplicates:
        raise ValueError(
            "S1 recovery duplicate identities do not explain the exposure surplus"
        )
    if not str(failed_job_id).isdigit():
        raise ValueError("S1 recovery requires a numeric failed Slurm job id")

    original = parent or {
        "failed_job_id": str(failed_job_id),
        "failure_signature": S1_PROFILE_FAILURE_SIGNATURE,
        "test_open_certificate_sha256": failed_marker[
            "test_open_certificate_sha256"
        ],
        "superseded_marker_path": str(failed_marker_path),
        "superseded_marker_file_sha256": sha256_file(failed_marker_path),
        "superseded_marker_sha256": failed_marker["marker_sha256"],
        "failure_log_path": str(failure_log_path),
        "failure_log_sha256": sha256_file(failure_log_path),
    }
    basis = {
        "schema_version": (
            S1_CHAINED_PROFILE_RECOVERY_SCHEMA
            if chained
            else S1_PROFILE_RECOVERY_SCHEMA
        ),
        "reason": (
            S1_CHAINED_RECOVERY_REASON if chained else S1_PROFILE_RECOVERY_REASON
        ),
        "failure_signature": original["failure_signature"],
        "failed_job_id": original["failed_job_id"],
        "training_code_commit": training_commit,
        "profile_code_commit": profile_commit,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "test_open_certificate_sha256": original[
            "test_open_certificate_sha256"
        ],
        "superseded_marker_path": original["superseded_marker_path"],
        "superseded_marker_file_sha256": original[
            "superseded_marker_file_sha256"
        ],
        "superseded_marker_sha256": original["superseded_marker_sha256"],
        "failure_log_path": original["failure_log_path"],
        "failure_log_sha256": original["failure_log_sha256"],
        "expected_loader_exposure_count": exposure_count,
        "expected_physical_window_count": physical_count,
        "expected_duplicate_physical_window_ids": duplicates,
        "changed_files": changed_files,
        "repair_scope": (
            "profile_identity_power_sampling_and_postprocessing_only"
            if chained
            else "profile_identity_and_postprocessing_only"
        ),
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
    }
    if chained:
        parent_path = Path(superseded_recovery_certificate_path).resolve()
        diagnostic_path = Path(power_diagnostic_path).resolve()
        basis.update(
            {
                "preserve_recovery_chain": True,
                "superseded_recovery_certificate_path": str(parent_path),
                "superseded_recovery_certificate_file_sha256": sha256_file(
                    parent_path
                ),
                "superseded_recovery_certificate_sha256": parent[
                    "certificate_sha256"
                ],
                "superseded_recovery_campaign_id": parent["campaign_id"],
                "superseded_recovery_profile_code_commit": parent[
                    "profile_code_commit"
                ],
                "power_failure_signature": S1_POWER_FAILURE_SIGNATURE,
                "power_failed_job_id": str(failed_job_id),
                "power_failure_marker_path": str(failed_marker_path),
                "power_failure_marker_file_sha256": sha256_file(
                    failed_marker_path
                ),
                "power_failure_marker_sha256": failed_marker["marker_sha256"],
                "power_failure_log_path": str(failure_log_path),
                "power_failure_log_sha256": sha256_file(failure_log_path),
                "power_diagnostic_path": str(diagnostic_path),
                "power_diagnostic_file_sha256": sha256_file(diagnostic_path),
                "power_diagnostic_sha256": diagnostic["diagnostic_sha256"],
                "power_diagnostic_job_id": diagnostic["slurm_job_id"],
                "power_diagnostic_code_commit": diagnostic["code_commit"],
                "power_sampler_backend": "nvml-persistent-poll-v1",
            }
        )
    campaign_id = canonical_sha256(basis)[:16]
    campaign_root = (
        Path(binding["canonical_experiment_root"]) / "profile_campaigns" / campaign_id
    ).resolve()
    certificate = {
        **basis,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    output_path = _certificate_output_path(certificate)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != certificate:
            raise FileExistsError("S1 profile recovery campaign identity collision")
    else:
        atomic_publish_json(output_path, certificate)
    return output_path, certificate


def validate_profile_recovery_certificate(
    certificate: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    verify_checkout: bool = True,
) -> dict[str, Any]:
    checked = json.loads(json.dumps(dict(certificate)))
    certificate_hash = checked.pop("certificate_sha256", None)
    if not certificate_hash or canonical_sha256(checked) != certificate_hash:
        raise ValueError("S1 profile recovery certificate self-hash mismatch")
    checked["certificate_sha256"] = certificate_hash
    chained = checked.get("reason") == S1_CHAINED_RECOVERY_REASON
    sidecar = checked.get("reason") == S1_SIDECAR_RECOVERY_REASON
    has_parent = chained or sidecar
    if checked.get("reason") not in {
        S1_PROFILE_RECOVERY_REASON,
        S1_CHAINED_RECOVERY_REASON,
        S1_SIDECAR_RECOVERY_REASON,
    }:
        raise ValueError("unsupported S1 profile recovery reason")
    expected_schema = (
        S1_SIDECAR_PROFILE_RECOVERY_SCHEMA
        if sidecar
        else (
            S1_CHAINED_PROFILE_RECOVERY_SCHEMA
            if chained
            else S1_PROFILE_RECOVERY_SCHEMA
        )
    )
    if checked.get("schema_version") != expected_schema:
        raise ValueError("S1 profile recovery certificate schema/reason mismatch")
    expected = {
        "failure_signature": S1_PROFILE_FAILURE_SIGNATURE,
        "training_code_commit": str(binding["code_commit"]).lower(),
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "repair_scope": (
            "out_of_process_power_sidecar_and_failure_evidence_only"
            if sidecar
            else (
                "profile_identity_power_sampling_and_postprocessing_only"
                if chained
                else "profile_identity_and_postprocessing_only"
            )
        ),
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
    }
    for key, value in expected.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 profile recovery certificate {key} mismatch")
    if bool(checked.get("preserve_recovery_chain", False)) != has_parent:
        raise ValueError("S1 profile recovery chain-preservation flag mismatch")
    campaign_basis = {
        key: value
        for key, value in checked.items()
        if key not in {"campaign_id", "campaign_root", "certificate_sha256"}
    }
    expected_campaign_id = canonical_sha256(campaign_basis)[:16]
    expected_campaign_root = (
        Path(binding["canonical_experiment_root"])
        / "profile_campaigns"
        / expected_campaign_id
    ).resolve()
    if (
        checked.get("campaign_id") != expected_campaign_id
        or Path(checked.get("campaign_root", "")).resolve() != expected_campaign_root
    ):
        raise ValueError("S1 profile recovery campaign identity mismatch")

    failed_marker_path = Path(checked["superseded_marker_path"]).resolve()
    failure_log_path = Path(checked["failure_log_path"]).resolve()
    for path, key in (
        (failed_marker_path, "superseded_marker_file_sha256"),
        (failure_log_path, "failure_log_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != checked[key]:
            raise ValueError(f"S1 profile recovery artifact mismatch: {path}")
    failed_marker = _validate_superseded_marker(failed_marker_path)
    if failed_marker["marker_sha256"] != checked["superseded_marker_sha256"]:
        raise ValueError("S1 profile recovery superseded-marker identity mismatch")
    first_profile_cell = build_s1_profile_order()[0]
    marker_expected = {
        "resolution": int(first_profile_cell["resolution"]),
        "seed": int(first_profile_cell["seed"]),
        "code_commit": str(binding["code_commit"]).lower(),
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "profile_order_ordinal": 0,
        "test_open_certificate_sha256": checked["test_open_certificate_sha256"],
    }
    for key, value in marker_expected.items():
        if failed_marker.get(key) != value:
            raise ValueError(f"S1 profile recovery superseded marker {key} mismatch")
    if S1_PROFILE_FAILURE_SIGNATURE not in failure_log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise ValueError("S1 profile recovery failure signature disappeared")

    if chained:
        parent_path = Path(
            checked["superseded_recovery_certificate_path"]
        ).resolve()
        power_marker_path = Path(checked["power_failure_marker_path"]).resolve()
        power_log_path = Path(checked["power_failure_log_path"]).resolve()
        diagnostic_path = Path(checked["power_diagnostic_path"]).resolve()
        for path, key in (
            (parent_path, "superseded_recovery_certificate_file_sha256"),
            (power_marker_path, "power_failure_marker_file_sha256"),
            (power_log_path, "power_failure_log_sha256"),
            (diagnostic_path, "power_diagnostic_file_sha256"),
        ):
            if not path.is_file() or sha256_file(path) != checked.get(key):
                raise ValueError(f"S1 chained recovery artifact mismatch: {path}")
        parent = validate_profile_recovery_certificate(
            json.loads(parent_path.read_text(encoding="utf-8")),
            binding=binding,
            verify_checkout=False,
        )
        if (
            parent.get("reason") != S1_PROFILE_RECOVERY_REASON
            or parent["certificate_sha256"]
            != checked.get("superseded_recovery_certificate_sha256")
            or parent["campaign_id"]
            != checked.get("superseded_recovery_campaign_id")
            or parent["profile_code_commit"]
            != checked.get("superseded_recovery_profile_code_commit")
        ):
            raise ValueError("S1 chained recovery parent identity mismatch")
        power_marker = _validate_superseded_marker(
            power_marker_path, expected_schema=S1_CHAINED_PROFILE_MARKER_SCHEMA
        )
        if (
            power_marker["marker_sha256"]
            != checked.get("power_failure_marker_sha256")
            or S1_POWER_FAILURE_SIGNATURE
            not in power_log_path.read_text(encoding="utf-8", errors="replace")
            or checked.get("power_failure_signature")
            != S1_POWER_FAILURE_SIGNATURE
            or str(checked.get("power_failed_job_id", "")).isdigit() is False
        ):
            raise ValueError("S1 chained recovery power-failure evidence mismatch")
        current_marker_expected = {
            "resolution": int(first_profile_cell["resolution"]),
            "seed": int(first_profile_cell["seed"]),
            "code_commit": str(binding["code_commit"]).lower(),
            "profile_code_commit": parent["profile_code_commit"],
            "experiment_namespace": binding["experiment_namespace"],
            "canonical_experiment_root": binding["canonical_experiment_root"],
            "manifest_sha256": binding["manifest_sha256"],
            "precheck_file_sha256": binding["precheck_file_sha256"],
            "precheck_sha256": binding["precheck_sha256"],
            "profile_order_ordinal": 0,
            "test_open_certificate_sha256": checked[
                "test_open_certificate_sha256"
            ],
            "profile_recovery_certificate_sha256": parent[
                "certificate_sha256"
            ],
            "profile_recovery_campaign_id": parent["campaign_id"],
        }
        for key, value in current_marker_expected.items():
            if power_marker.get(key) != value:
                raise ValueError(f"S1 chained recovery marker {key} mismatch")
        diagnostic = _validate_power_diagnostic(diagnostic_path)
        if (
            diagnostic["diagnostic_sha256"]
            != checked.get("power_diagnostic_sha256")
            or diagnostic["slurm_job_id"]
            != checked.get("power_diagnostic_job_id")
            or diagnostic["code_commit"]
            != checked.get("power_diagnostic_code_commit")
            or checked.get("power_sampler_backend")
            != "nvml-persistent-poll-v1"
        ):
            raise ValueError("S1 chained recovery power diagnostic mismatch")
    elif sidecar:
        parent_path = Path(
            checked["superseded_recovery_certificate_path"]
        ).resolve()
        power_marker_path = Path(
            checked["sidecar_power_failure_marker_path"]
        ).resolve()
        power_log_path = Path(
            checked["sidecar_power_failure_log_path"]
        ).resolve()
        for path, key in (
            (parent_path, "superseded_recovery_certificate_file_sha256"),
            (power_marker_path, "sidecar_power_failure_marker_file_sha256"),
            (power_log_path, "sidecar_power_failure_log_sha256"),
        ):
            if not path.is_file() or sha256_file(path) != checked.get(key):
                raise ValueError(f"S1 sidecar recovery artifact mismatch: {path}")
        parent = validate_profile_recovery_certificate(
            json.loads(parent_path.read_text(encoding="utf-8")),
            binding=binding,
            verify_checkout=False,
        )
        if (
            parent.get("reason") != S1_CHAINED_RECOVERY_REASON
            or parent["certificate_sha256"]
            != checked.get("superseded_recovery_certificate_sha256")
            or parent["campaign_id"]
            != checked.get("superseded_recovery_campaign_id")
            or parent["profile_code_commit"]
            != checked.get("superseded_recovery_profile_code_commit")
        ):
            raise ValueError("S1 sidecar recovery parent identity mismatch")
        power_marker = _validate_superseded_marker(
            power_marker_path,
            expected_schema=S1_CHAINED_PROFILE_MARKER_SCHEMA,
        )
        current_marker_expected = {
            "resolution": int(first_profile_cell["resolution"]),
            "seed": int(first_profile_cell["seed"]),
            "code_commit": str(binding["code_commit"]).lower(),
            "profile_code_commit": parent["profile_code_commit"],
            "experiment_namespace": binding["experiment_namespace"],
            "canonical_experiment_root": binding["canonical_experiment_root"],
            "manifest_sha256": binding["manifest_sha256"],
            "precheck_file_sha256": binding["precheck_file_sha256"],
            "precheck_sha256": binding["precheck_sha256"],
            "profile_order_ordinal": 0,
            "test_open_certificate_sha256": checked[
                "test_open_certificate_sha256"
            ],
            "profile_recovery_certificate_sha256": parent[
                "certificate_sha256"
            ],
            "profile_recovery_campaign_id": parent["campaign_id"],
        }
        for key, value in current_marker_expected.items():
            if power_marker.get(key) != value:
                raise ValueError(f"S1 sidecar recovery marker {key} mismatch")
        if (
            power_marker["marker_sha256"]
            != checked.get("sidecar_power_failure_marker_sha256")
            or S1_POWER_FAILURE_SIGNATURE
            not in power_log_path.read_text(encoding="utf-8", errors="replace")
            or checked.get("sidecar_power_failure_signature")
            != S1_POWER_FAILURE_SIGNATURE
            or str(checked.get("sidecar_power_failed_job_id", "")).isdigit()
            is False
            or checked.get("power_sampler_backend")
            != S1_SIDECAR_POWER_BACKEND
            or int(checked.get("power_target_interval_ms", -1)) != 20
            or float(checked.get("power_max_gap_limit_ms", -1.0)) != 100.0
            or int(checked.get("allocated_cpu_count", -1)) != 5
            or int(checked.get("detector_cpu_count", -1)) != 4
            or int(checked.get("sidecar_cpu_count", -1)) != 1
            or checked.get("requires_long_no_open_gate") is not True
            or checked.get("sidecar_gate_relative_path") != "sidecar_gate.json"
        ):
            raise ValueError("S1 sidecar recovery contract mismatch")

    exposure_count = int(checked.get("expected_loader_exposure_count", -1))
    physical_count = int(checked.get("expected_physical_window_count", -1))
    duplicates = checked.get("expected_duplicate_physical_window_ids")
    if (
        not isinstance(duplicates, list)
        or sorted(set(map(str, duplicates))) != duplicates
        or exposure_count - physical_count != len(duplicates)
        or not duplicates
    ):
        raise ValueError("S1 profile recovery exposure topology is invalid")

    changed_files = checked.get("changed_files")
    if not isinstance(changed_files, list):
        raise ValueError("S1 profile recovery has no audited Git diff")
    changed_paths = set()
    for row in changed_files:
        if not isinstance(row, Mapping):
            raise ValueError("S1 profile recovery Git diff row is invalid")
        path = str(row.get("path", ""))
        if (
            row.get("status") not in {"A", "M"}
            or (
                path not in _ALLOWED_EXACT_PATHS
                and not path.startswith(_ALLOWED_PREFIXES)
            )
            or len(str(row.get("file_sha256", ""))) != 64
        ):
            raise ValueError(
                "S1 profile recovery Git diff is outside the audited scope"
            )
        changed_paths.add(path)
    required_paths = (
        _REQUIRED_REPAIR_PATHS_SIDECAR
        if sidecar
        else (
            _REQUIRED_REPAIR_PATHS_CHAINED
            if chained
            else _REQUIRED_REPAIR_PATHS_V1
        )
    )
    if not required_paths.issubset(changed_paths):
        raise ValueError("S1 profile recovery Git diff omits a required repair path")

    if verify_checkout:
        require_clean_profile_checkout(expected_commit=checked["profile_code_commit"])
        if (
            _changed_files(
                checked["training_code_commit"],
                checked["profile_code_commit"],
                required_paths=required_paths,
            )
            != checked["changed_files"]
        ):
            raise ValueError("S1 profile recovery Git diff changed after certification")
    return checked


def load_profile_recovery_certificate(
    path: str | Path,
    *,
    binding: Mapping[str, Any],
    verify_checkout: bool = True,
) -> dict[str, Any]:
    path = Path(path).resolve()
    checked = validate_profile_recovery_certificate(
        json.loads(path.read_text(encoding="utf-8")),
        binding=binding,
        verify_checkout=verify_checkout,
    )
    if path != _certificate_output_path(checked):
        raise ValueError("S1 profile recovery certificate is outside its campaign")
    raw_text = path.read_text(encoding="utf-8")
    canonical_text = json.dumps(json.loads(raw_text), indent=2, sort_keys=True) + "\n"
    if raw_text != canonical_text:
        raise ValueError("S1 profile recovery certificate is not canonical JSON")
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create one immutable S1 post-profile recovery campaign"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--failed-marker", type=Path, required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--failed-job-id", required=True)
    parser.add_argument("--superseded-recovery-certificate", type=Path)
    parser.add_argument("--power-diagnostic", type=Path)
    parser.add_argument("--expected-exposure-count", type=int, required=True)
    parser.add_argument("--expected-physical-window-count", type=int, required=True)
    parser.add_argument(
        "--expected-duplicate-physical-window-id", action="append", required=True
    )
    args = parser.parse_args(argv)
    try:
        cfg = Config.fromfile(str(args.config.resolve()))
        binding = validate_bound_s1_training_config(cfg, seed=int(args.seed))
        path, certificate = build_profile_recovery_certificate(
            binding=binding,
            failed_marker_path=args.failed_marker,
            failure_log_path=args.failure_log,
            failed_job_id=args.failed_job_id,
            expected_exposure_count=args.expected_exposure_count,
            expected_physical_window_count=args.expected_physical_window_count,
            expected_duplicate_physical_window_ids=(
                args.expected_duplicate_physical_window_id
            ),
            superseded_recovery_certificate_path=(
                args.superseded_recovery_certificate
            ),
            power_diagnostic_path=args.power_diagnostic,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "certificate_path": str(path),
                "campaign_id": certificate["campaign_id"],
                "certificate_sha256": certificate["certificate_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "S1_PROFILE_RECOVERY_SCHEMA",
    "S1_CHAINED_PROFILE_RECOVERY_SCHEMA",
    "S1_SIDECAR_PROFILE_RECOVERY_SCHEMA",
    "S1_CHAINED_RECOVERY_REASON",
    "S1_SIDECAR_RECOVERY_REASON",
    "S1_POWER_FAILURE_SIGNATURE",
    "S1_SIDECAR_POWER_BACKEND",
    "build_profile_recovery_certificate",
    "build_sidecar_profile_recovery_certificate",
    "load_profile_recovery_certificate",
    "profile_campaign_prefix",
    "require_clean_profile_checkout",
    "validate_profile_recovery_certificate",
]
