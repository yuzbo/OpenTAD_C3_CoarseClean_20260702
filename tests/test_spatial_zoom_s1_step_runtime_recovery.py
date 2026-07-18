from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.bata.spatial_zoom_s1_contract import (
    build_s1_profile_order,
    canonical_sha256,
    sha256_file,
)
from tools.bata import spatial_zoom_s1_profile_recovery as recovery
from tools.bata import spatial_zoom_s1_sidecar_gate as sidecar_gate


TRAINING_COMMIT = "18139b930bef6ee234f6220a6adc898eb9c23c0c"
PARENT_COMMIT = "43ac70bea6720f0c882ed3208bccfc89b089b5d4"
RUNTIME_COMMIT = "5" * 40
V5_RUNTIME_COMMIT = "3d01d3b7fc956ae17568ac3c8c04f9d6f36c42c5"
V6_RUNTIME_COMMIT = "6" * 40
FORMAL_PYTHON_ENVIRONMENT = {
    "python_no_user_site": "1",
    "site_enable_user_site": False,
    "numpy_version": recovery.S1_FORMAL_NUMPY_VERSION,
    "numpy_path": recovery.S1_FORMAL_NUMPY_PATH,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fake_git_file_sha256(commit: str, path: str) -> str:
    return hashlib.sha256(f"{commit}:{path}".encode("utf-8")).hexdigest()


def _semantic_evidence() -> dict:
    return {
        "tools_test_zero_context_patch_sha256": "1" * 64,
        "tools_test_zero_context_patch_size_bytes": 4096,
        "tools_test_training_normalized_ast_sha256": "2" * 64,
        "tools_test_runtime_normalized_ast_sha256": "2" * 64,
        "tools_test_allowed_ast_counts": dict(recovery._TOOLS_TEST_ALLOWED_AST_COUNTS),
        "tools_test_semantic_scope": recovery._TOOLS_TEST_SEMANTIC_SCOPE,
    }


def _install_git_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    all_rows = [
        {
            "status": "A"
            if path.startswith("docs/superpowers/") or path.startswith("tests/")
            else "M",
            "path": path,
            "file_sha256": _fake_git_file_sha256(RUNTIME_COMMIT, path),
        }
        for path in sorted(recovery._REQUIRED_REPAIR_PATHS_STEP_RUNTIME)
    ]
    parent_rows = [
        {
            "status": "A"
            if path.startswith("docs/superpowers/") or path.startswith("tests/")
            else "M",
            "path": path,
            "file_sha256": _fake_git_file_sha256(RUNTIME_COMMIT, path),
        }
        for path in sorted(recovery._REQUIRED_PARENT_TO_STEP_RUNTIME_PATHS)
    ]
    monkeypatch.setattr(recovery, "current_git_commit", lambda: RUNTIME_COMMIT)
    monkeypatch.setattr(recovery, "require_clean_profile_checkout", lambda **_: None)
    monkeypatch.setattr(
        recovery,
        "_changed_files",
        lambda *_, **__: copy.deepcopy(all_rows),
    )
    monkeypatch.setattr(
        recovery,
        "_changed_files_between_commits",
        lambda *_, **__: copy.deepcopy(parent_rows),
    )
    monkeypatch.setattr(recovery, "_assert_no_model_surface_changes", lambda *_: None)
    monkeypatch.setattr(recovery, "_git_file_sha256", _fake_git_file_sha256)
    monkeypatch.setattr(
        recovery,
        "_formal_python_environment_evidence",
        lambda: copy.deepcopy(FORMAL_PYTHON_ENVIRONMENT),
    )
    monkeypatch.setattr(
        recovery,
        "_tools_test_semantic_evidence_between_commits",
        lambda *_: _semantic_evidence(),
    )


def _install_v6_git_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    all_rows = [
        {
            "status": "A"
            if path.startswith("docs/superpowers/") or path.startswith("tests/")
            else "M",
            "path": path,
            "file_sha256": _fake_git_file_sha256(V6_RUNTIME_COMMIT, path),
        }
        for path in sorted(recovery._REQUIRED_REPAIR_PATHS_STEP_RUNTIME)
    ]
    parent_rows = [
        {
            "status": "A"
            if path.startswith("docs/superpowers/") or path.startswith("tests/")
            else "M",
            "path": path,
            "file_sha256": _fake_git_file_sha256(V6_RUNTIME_COMMIT, path),
        }
        for path in sorted(recovery._REQUIRED_PARENT_TO_SCHEMA_COMPAT_PATHS)
    ]
    monkeypatch.setattr(recovery, "current_git_commit", lambda: V6_RUNTIME_COMMIT)
    monkeypatch.setattr(recovery, "require_clean_profile_checkout", lambda **_: None)
    monkeypatch.setattr(
        recovery,
        "_changed_files",
        lambda *_, **__: copy.deepcopy(all_rows),
    )
    monkeypatch.setattr(
        recovery,
        "_changed_files_between_commits",
        lambda *_, **__: copy.deepcopy(parent_rows),
    )
    monkeypatch.setattr(recovery, "_assert_no_model_surface_changes", lambda *_: None)
    monkeypatch.setattr(recovery, "_git_file_sha256", _fake_git_file_sha256)
    monkeypatch.setattr(
        recovery,
        "_formal_python_environment_evidence",
        lambda: copy.deepcopy(FORMAL_PYTHON_ENVIRONMENT),
    )
    monkeypatch.setattr(
        recovery,
        "_tools_test_semantic_evidence_between_commits",
        lambda *_: _semantic_evidence(),
    )


def _v6_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    canonical_root = (tmp_path / "canonical").resolve()
    parent_root = (canonical_root / "profile_campaigns" / "parent-v5").resolve()
    parent_path = parent_root / "recovery_certificate.json"
    _write_json(parent_path, {"immutable_parent": "v5"})
    test_evidence_path = (
        canonical_root
        / "dense256"
        / "seed3408"
        / "gpu1_id0"
        / "test_evidence"
        / "test.evidence.json"
    ).resolve()
    _write_json(test_evidence_path, {"official_test": "unchanged"})
    binding = {
        "code_commit": TRAINING_COMMIT,
        "experiment_namespace": "s1-test",
        "canonical_experiment_root": str(canonical_root),
        "manifest_sha256": "3" * 64,
        "protocol_fingerprint": "4" * 64,
        "precheck_file_sha256": "5" * 64,
        "precheck_sha256": "6" * 64,
        "pretrained_checkpoint_sha256": "7" * 64,
    }
    parent = {
        "reason": recovery.S1_STEP_RUNTIME_RECOVERY_REASON,
        "campaign_id": "parent-v5",
        "campaign_root": str(parent_root),
        "profile_code_commit": V5_RUNTIME_COMMIT,
        "certificate_sha256": "8" * 64,
        "failure_signature": recovery.S1_PROFILE_FAILURE_SIGNATURE,
        "failed_job_id": recovery.S1_STEP_RUNTIME_PARENT_JOB_ID,
        "test_open_certificate_sha256": "9" * 64,
        "legacy_unbound_test_resolution": 256,
        "legacy_unbound_test_seed": 3408,
        "legacy_unbound_test_evidence_path": str(test_evidence_path),
        "legacy_unbound_test_evidence_file_sha256": sha256_file(test_evidence_path),
        "legacy_unbound_test_evidence_sha256": "b" * 64,
        "superseded_marker_path": str(canonical_root / "attempt.json"),
        "superseded_marker_file_sha256": "c" * 64,
        "superseded_marker_sha256": "d" * 64,
        "failure_log_path": str(canonical_root / "failure.log"),
        "failure_log_sha256": "e" * 64,
        "expected_loader_exposure_count": 792,
        "expected_physical_window_count": 791,
        "expected_duplicate_physical_window_ids": ["window-0"],
        "formal_test_runtime_mode": recovery.S1_STEP_SCOPED_TEST_RUNTIME_MODE,
        "power_sampler_backend": recovery.S1_SIDECAR_POWER_BACKEND,
        "trace_publication_mode": recovery.S1_BUFFERED_TRACE_PUBLICATION_MODE,
        "trace_io_inside_sampling_loop": False,
        "power_target_interval_ms": 20,
        "power_max_gap_limit_ms": 100.0,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
        "sidecar_gate_relative_path": "sidecar_gate.json",
    }
    parent_loads: list[Path] = []

    def _load_parent(path: str | Path, **_: object) -> dict:
        parent_loads.append(Path(path).resolve())
        return copy.deepcopy(parent)

    monkeypatch.setattr(recovery, "_load_parent_recovery_certificate", _load_parent)
    _install_v6_git_stubs(monkeypatch)

    receipt_path = parent_root / "gate_submission.json"
    receipt = {
        "schema_version": "spatial_zoom_s1_sidecar_gate_submission_v2",
        "status": "SUBMITTED",
        "submitted_utc": "2026-07-18T09:40:00+08:00",
        "job_id": recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID,
        "job_name": "s1_3d01_gate",
        "profile_campaign_id": parent["campaign_id"],
        "profile_campaign_root": str(parent_root),
        "profile_code_commit": V5_RUNTIME_COMMIT,
        "profile_source_root": str(tmp_path / "profile-source"),
        "training_code_commit": TRAINING_COMMIT,
        "training_source_root": str(tmp_path / "training-source"),
        "recovery_certificate_path": str(parent_path),
        "recovery_certificate_file_sha256": sha256_file(parent_path),
        "recovery_certificate_sha256": parent["certificate_sha256"],
        "profile_run": False,
        "test_opened": False,
        "physical_gpu_index_overridden": False,
        "outer_resources": {
            "cpus": 8,
            "gpus": 2,
            "memory_request": "site-default",
        },
        "inner_resources": {"cpus": 5, "gpus": 1, "memory_mib": 96000},
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write_json(receipt_path, receipt)

    logs = parent_root / "logs"
    stdout_path = logs / f"gate-{recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID}.out"
    stderr_path = logs / f"gate-{recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID}.err"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(
        "96000\n"
        + json.dumps(
            {
                "status": "FAIL",
                "error_type": "ValueError",
                "error": recovery.S1_SCHEMA_COMPAT_FAILURE_SIGNATURE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text(
        "[SPATIAL_ZOOM_S1_SIDECAR_GATE][FAIL] "
        + recovery.S1_SCHEMA_COMPAT_LAUNCHER_FAILURE_SIGNATURE
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        recovery,
        "S1_SCHEMA_COMPAT_GATE_STDOUT_SHA256",
        sha256_file(stdout_path),
    )
    monkeypatch.setattr(
        recovery,
        "S1_SCHEMA_COMPAT_GATE_STDERR_SHA256",
        sha256_file(stderr_path),
    )
    return {
        "binding": binding,
        "parent": parent,
        "parent_path": parent_path,
        "parent_loads": parent_loads,
        "receipt_path": receipt_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "test_evidence_path": test_evidence_path,
    }


def _build_v6(context: dict) -> tuple[Path, dict]:
    return recovery.build_schema_compat_profile_recovery_certificate(
        binding=context["binding"],
        superseded_recovery_certificate_path=context["parent_path"],
        gate_submission_receipt_path=context["receipt_path"],
        gate_stdout_path=context["stdout_path"],
        gate_stderr_path=context["stderr_path"],
        failed_job_id=recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID,
    )


def _resign_recovery_certificate(certificate: dict, binding: dict) -> dict:
    resigned = copy.deepcopy(certificate)
    basis = {
        key: value
        for key, value in resigned.items()
        if key not in {"campaign_id", "campaign_root", "certificate_sha256"}
    }
    resigned["campaign_id"] = canonical_sha256(basis)[:16]
    resigned["campaign_root"] = str(
        (
            Path(binding["canonical_experiment_root"])
            / "profile_campaigns"
            / resigned["campaign_id"]
        ).resolve()
    )
    resigned.pop("certificate_sha256", None)
    resigned["certificate_sha256"] = canonical_sha256(resigned)
    return resigned


def _v5_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    canonical_root = (tmp_path / "canonical").resolve()
    parent_root = (canonical_root / "profile_campaigns" / "parent-v4").resolve()
    parent_path = parent_root / "recovery_certificate.json"
    _write_json(parent_path, {"immutable_parent": True})
    binding = {
        "code_commit": TRAINING_COMMIT,
        "experiment_namespace": "s1-test",
        "canonical_experiment_root": str(canonical_root),
        "manifest_sha256": "3" * 64,
        "protocol_fingerprint": "4" * 64,
        "precheck_file_sha256": "5" * 64,
        "precheck_sha256": "6" * 64,
        "pretrained_checkpoint_sha256": "7" * 64,
    }
    parent = {
        "reason": recovery.S1_BUFFERED_SIDECAR_RECOVERY_REASON,
        "campaign_id": "parent-v4",
        "campaign_root": str(parent_root),
        "profile_code_commit": PARENT_COMMIT,
        "certificate_sha256": "8" * 64,
        "failure_signature": recovery.S1_PROFILE_FAILURE_SIGNATURE,
        "failed_job_id": "1168823",
        "test_open_certificate_sha256": "9" * 64,
        "legacy_unbound_test_resolution": 256,
        "legacy_unbound_test_seed": 3408,
        "legacy_unbound_test_evidence_path": str(parent_root / "legacy-test.json"),
        "legacy_unbound_test_evidence_file_sha256": "a" * 64,
        "legacy_unbound_test_evidence_sha256": "b" * 64,
        "superseded_marker_path": str(parent_root / "attempt.json"),
        "superseded_marker_file_sha256": "c" * 64,
        "superseded_marker_sha256": "d" * 64,
        "failure_log_path": str(parent_root / "failure.log"),
        "failure_log_sha256": "e" * 64,
        "expected_loader_exposure_count": 792,
        "expected_physical_window_count": 791,
        "expected_duplicate_physical_window_ids": ["window-0"],
        "power_sampler_backend": recovery.S1_SIDECAR_POWER_BACKEND,
        "trace_publication_mode": recovery.S1_BUFFERED_TRACE_PUBLICATION_MODE,
        "trace_io_inside_sampling_loop": False,
        "power_target_interval_ms": 20,
        "power_max_gap_limit_ms": 100.0,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
        "sidecar_gate_relative_path": "sidecar_gate.json",
    }
    parent_loads: list[Path] = []

    def _load_parent(path: str | Path, **_: object) -> dict:
        parent_loads.append(Path(path).resolve())
        return copy.deepcopy(parent)

    monkeypatch.setattr(recovery, "_load_parent_recovery_certificate", _load_parent)
    _install_git_stubs(monkeypatch)

    matrix_start_path = parent_root / "matrix.lock" / "matrix.started.json"
    _write_json(matrix_start_path, {"matrix_started": True})
    sidecar_path = parent_root / "sidecar_gate.json"
    _write_json(sidecar_path, {"sidecar_gate": True})
    matrix_start = {
        "slurm_job_id": recovery.S1_STEP_RUNTIME_PARENT_JOB_ID,
        "slurm_step_id": "0",
        "slurm_job_gpus": "2,4",
        "slurm_step_gpus": "2",
        "step_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
        "matrix_sha256": "f" * 64,
        "profile_code_commit": PARENT_COMMIT,
        "profile_recovery_certificate_sha256": parent["certificate_sha256"],
        "profile_recovery_campaign_id": parent["campaign_id"],
        "frozen_order": build_s1_profile_order(),
        "effective_step_memory_limit_mb": 96000,
        "slurm_cpus_per_task": 5,
        "sidecar_gate_evidence_path": str(sidecar_path),
        "sidecar_gate_evidence_file_sha256": sha256_file(sidecar_path),
        "sidecar_gate_sha256": "0" * 64,
    }
    monkeypatch.setattr(
        recovery,
        "_validate_v3_matrix_start_receipt",
        lambda *_, **__: copy.deepcopy(matrix_start),
    )

    submission_path = parent_root / "matrix_submission.json"
    submission = {
        "schema_version": recovery.S1_MATRIX_SUBMISSION_SCHEMA,
        "status": "SUBMITTED",
        "submitted_utc": "2026-07-17T21:00:00+08:00",
        "job_id": recovery.S1_STEP_RUNTIME_PARENT_JOB_ID,
        "job_name": "s1_43ac_matrix",
        "profile_campaign_id": parent["campaign_id"],
        "profile_campaign_root": str(parent_root),
        "profile_code_commit": PARENT_COMMIT,
        "profile_source_root": str(tmp_path / "profile-source"),
        "training_code_commit": TRAINING_COMMIT,
        "recovery_certificate_path": str(parent_path),
        "recovery_certificate_file_sha256": sha256_file(parent_path),
        "recovery_certificate_sha256": parent["certificate_sha256"],
        "sidecar_gate_path": matrix_start["sidecar_gate_evidence_path"],
        "sidecar_gate_file_sha256": matrix_start["sidecar_gate_evidence_file_sha256"],
        "sidecar_gate_sha256": matrix_start["sidecar_gate_sha256"],
        "frozen_order": build_s1_profile_order(),
        "single_serial_allocation": True,
        "physical_gpu_index_overridden": False,
        "outer_resources": {
            "cpus": 8,
            "gpus": 2,
            "memory_request": "site-default",
        },
        "inner_resources": {"cpus": 5, "gpus": 1, "memory_mib": 96000},
    }
    submission["receipt_sha256"] = canonical_sha256(submission)
    _write_json(submission_path, submission)

    descriptor_path = (
        parent_root / "descriptors" / "dense256_seed3408.run.json"
    ).resolve()
    descriptor = {"descriptor_sha256": "1" * 64}
    _write_json(descriptor_path, descriptor)
    descriptor_record = {
        "profile_order_ordinal": 0,
        "resolution": 256,
        "seed": 3408,
        "descriptor_sha256": descriptor["descriptor_sha256"],
    }
    monkeypatch.setattr(
        recovery,
        "_validate_v5_completed_descriptor",
        lambda *_, **__: (
            copy.deepcopy(descriptor),
            copy.deepcopy(descriptor_record),
        ),
    )

    logs = parent_root / "logs"
    stdout_path = logs / f"matrix-{recovery.S1_STEP_RUNTIME_PARENT_JOB_ID}.out"
    stderr_path = logs / f"matrix-{recovery.S1_STEP_RUNTIME_PARENT_JOB_ID}.err"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(
        "[SPATIAL_ZOOM_S1_TEST_PROFILE] PASS resolution=256 seed=3408 "
        f"descriptor={descriptor_path}\n"
        '{"profile_order_ordinal": 1, "resolution": 224, "seed": 3409}\n',
        encoding="utf-8",
    )
    stderr_path.write_text(
        recovery.S1_STEP_RUNTIME_FAILURE_SIGNATURE + "\n",
        encoding="utf-8",
    )
    return {
        "binding": binding,
        "parent": parent,
        "parent_path": parent_path,
        "parent_loads": parent_loads,
        "matrix_start_path": matrix_start_path,
        "submission_path": submission_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "descriptor_path": descriptor_path,
    }


def _build_v5(context: dict) -> tuple[Path, dict]:
    return recovery.build_step_runtime_profile_recovery_certificate(
        binding=context["binding"],
        superseded_recovery_certificate_path=context["parent_path"],
        matrix_start_receipt_path=context["matrix_start_path"],
        matrix_submission_receipt_path=context["submission_path"],
        matrix_stdout_path=context["stdout_path"],
        matrix_stderr_path=context["stderr_path"],
        completed_descriptor_path=context["descriptor_path"],
        failed_job_id=recovery.S1_STEP_RUNTIME_PARENT_JOB_ID,
    )


def test_v5_parent_delta_uses_a_dedicated_narrow_allowlist() -> None:
    assert recovery._REQUIRED_PARENT_TO_STEP_RUNTIME_PATHS == (
        recovery._PARENT_TO_STEP_RUNTIME_PATHS
    )
    assert "tools/bata/analyze_spatial_zoom_s1_results.py" not in (
        recovery._PARENT_TO_STEP_RUNTIME_PATHS
    )
    assert "tools/bata/profile_spatial_zoom_s1.py" not in (
        recovery._PARENT_TO_STEP_RUNTIME_PATHS
    )
    assert "tools/bata/spatial_zoom_s1_training.py" not in (
        recovery._PARENT_TO_STEP_RUNTIME_PATHS
    )
    assert recovery._REQUIRED_STEP_RUNTIME_CODE_PATHS.issubset(
        recovery._PARENT_TO_STEP_RUNTIME_PATHS
    )


def test_v5_build_validate_and_load_close_recursive_failure_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v5_fixture(tmp_path, monkeypatch)
    certificate_path, certificate = _build_v5(context)

    assert certificate["schema_version"] == (
        recovery.S1_STEP_RUNTIME_PROFILE_RECOVERY_SCHEMA
    )
    assert certificate["step_runtime_failed_profile_order_ordinal"] == 1
    assert certificate["step_runtime_failed_resolution"] == 224
    assert certificate["step_runtime_failed_seed"] == 3409
    assert certificate["formal_python_environment"] == FORMAL_PYTHON_ENVIRONMENT
    assert certificate["step_runtime_completed_descriptor_count"] == 1
    assert certificate["step_runtime_completion_receipt_absent"] is True
    assert certificate["tools_test_zero_context_patch_sha256"] == "1" * 64
    assert set(recovery._REQUIRED_STEP_RUNTIME_CODE_PATHS).issubset(
        {row["path"] for row in certificate["parent_to_current_changed_files"]}
    )

    checked = recovery.validate_profile_recovery_certificate(
        certificate,
        binding=context["binding"],
        verify_checkout=False,
    )
    loaded = recovery.load_profile_recovery_certificate(
        certificate_path,
        binding=context["binding"],
        verify_checkout=False,
    )
    assert checked == certificate
    assert loaded == certificate
    assert context["parent_loads"].count(context["parent_path"].resolve()) >= 3


def test_v5_rejects_tampered_tools_test_zero_context_patch_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v5_fixture(tmp_path, monkeypatch)
    _, certificate = _build_v5(context)
    tampered = copy.deepcopy(certificate)
    tampered["tools_test_zero_context_patch_sha256"] = "0" * 64
    basis = {
        key: value
        for key, value in tampered.items()
        if key not in {"campaign_id", "campaign_root", "certificate_sha256"}
    }
    tampered["campaign_id"] = canonical_sha256(basis)[:16]
    tampered["campaign_root"] = str(
        (
            Path(context["binding"]["canonical_experiment_root"])
            / "profile_campaigns"
            / tampered["campaign_id"]
        ).resolve()
    )
    tampered.pop("certificate_sha256")
    tampered["certificate_sha256"] = canonical_sha256(tampered)

    with pytest.raises(ValueError, match="semantic evidence"):
        recovery.validate_profile_recovery_certificate(
            tampered,
            binding=context["binding"],
            verify_checkout=False,
        )


def test_v5_rejects_tampered_formal_python_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v5_fixture(tmp_path, monkeypatch)
    _, certificate = _build_v5(context)
    tampered = copy.deepcopy(certificate)
    tampered["formal_python_environment"]["numpy_version"] = "2.2.6"
    tampered["certificate_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "certificate_sha256"}
    )
    with pytest.raises(
        ValueError,
        match="formal_python_environment mismatch",
    ):
        recovery.validate_profile_recovery_certificate(
            tampered,
            binding=context["binding"],
            verify_checkout=False,
        )


def test_v5_rejects_parent_matrix_completion_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v5_fixture(tmp_path, monkeypatch)
    completion = context["parent_path"].parent / "matrix.lock" / "matrix.completed.json"
    _write_json(completion, {"status": "COMPLETED"})

    with pytest.raises(ValueError, match="completion receipt"):
        _build_v5(context)


def test_v5_rejects_stdout_that_advances_past_failed_ordinal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v5_fixture(tmp_path, monkeypatch)
    with context["stdout_path"].open("a", encoding="utf-8") as handle:
        handle.write('{"profile_order_ordinal": 2, "resolution": 256, "seed": 3409}\n')

    with pytest.raises(ValueError, match="advances beyond failed ordinal-1"):
        _build_v5(context)


def test_v5_requires_ordinal_resolution_and_seed_in_one_json_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v5_fixture(tmp_path, monkeypatch)
    context["stdout_path"].write_text(
        "[SPATIAL_ZOOM_S1_TEST_PROFILE] PASS resolution=256 seed=3408 "
        f"descriptor={context['descriptor_path']}\n"
        '{"profile_order_ordinal": 1, "resolution": 256, "seed": 3408}\n'
        '{"resolution": 224, "seed": 3409}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ordinal-1 record identity mismatch"):
        _build_v5(context)


@pytest.mark.parametrize(
    "old,new",
    (
        (
            "model = build_detector(cfg.model)",
            "model = build_detector(cfg.model, forbidden_runtime_change=True)",
        ),
        (
            "eval_one_epoch(\n",
            "eval_one_epoch_forbidden(\n",
        ),
    ),
)
def test_tools_test_semantic_scope_rejects_core_inference_changes(
    old: str,
    new: str,
) -> None:
    training_source = recovery._git_file_text(TRAINING_COMMIT, "tools/test.py")
    runtime_source = Path("tools/test.py").read_text(encoding="utf-8")
    assert old in runtime_source
    tampered = runtime_source.replace(old, new, 1)

    with pytest.raises(ValueError, match="protected.*inference path"):
        recovery._build_tools_test_semantic_evidence(
            training_source=training_source,
            runtime_source=tampered,
            zero_context_patch=b"non-empty",
        )


def test_tools_test_semantic_scope_records_exact_allowed_blocks() -> None:
    training_source = recovery._git_file_text(TRAINING_COMMIT, "tools/test.py")
    runtime_source = Path("tools/test.py").read_text(encoding="utf-8")
    evidence = recovery._build_tools_test_semantic_evidence(
        training_source=training_source,
        runtime_source=runtime_source,
        zero_context_patch=b"zero-context-diff",
    )

    assert evidence["tools_test_allowed_ast_counts"] == (
        recovery._TOOLS_TEST_ALLOWED_AST_COUNTS
    )
    assert (
        evidence["tools_test_training_normalized_ast_sha256"]
        == evidence["tools_test_runtime_normalized_ast_sha256"]
    )


def test_tools_test_semantic_scope_rejects_inference_smuggled_into_provenance() -> None:
    training_source = recovery._git_file_text(TRAINING_COMMIT, "tools/test.py")
    runtime_source = Path("tools/test.py").read_text(encoding="utf-8")
    needle = (
        "        if recovery is not None:\n"
        "            cfg.spatial_zoom_s1_test_binding.update(\n"
    )
    assert needle in runtime_source
    tampered = runtime_source.replace(
        needle,
        "        if recovery is not None:\n"
        "            model = build_detector(cfg.model)\n"
        "            cfg.spatial_zoom_s1_test_binding.update(\n",
        1,
    )

    with pytest.raises(ValueError, match="provenance exceeds scope"):
        recovery._build_tools_test_semantic_evidence(
            training_source=training_source,
            runtime_source=tampered,
            zero_context_patch=b"non-empty",
        )


def test_v5_rejects_any_training_to_runtime_model_surface_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery,
        "_run_git",
        lambda *_: "opentad/models/detectors/adatad.py\n",
    )
    with pytest.raises(ValueError, match="model/config surfaces"):
        recovery._assert_no_model_surface_changes(
            TRAINING_COMMIT,
            RUNTIME_COMMIT,
        )


def test_generic_builder_dispatches_v4_parent_to_v5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path = tmp_path / "parent.json"
    _write_json(
        parent_path,
        {"reason": recovery.S1_BUFFERED_SIDECAR_RECOVERY_REASON},
    )
    captured: dict = {}

    def _fake_v5(**kwargs: object) -> tuple[Path, dict]:
        captured.update(kwargs)
        return tmp_path / "certificate.json", {"schema_version": "v5"}

    monkeypatch.setattr(
        recovery,
        "build_step_runtime_profile_recovery_certificate",
        _fake_v5,
    )
    result = recovery.build_profile_recovery_certificate(
        binding={"code_commit": TRAINING_COMMIT},
        failed_marker_path=None,
        failure_log_path=None,
        failed_job_id=recovery.S1_STEP_RUNTIME_PARENT_JOB_ID,
        expected_exposure_count=None,
        expected_physical_window_count=None,
        expected_duplicate_physical_window_ids=None,
        superseded_recovery_certificate_path=parent_path,
        matrix_start_receipt_path=tmp_path / "started.json",
        matrix_submission_receipt_path=tmp_path / "submission.json",
        matrix_stdout_path=tmp_path / "matrix.out",
        matrix_stderr_path=tmp_path / "matrix.err",
        completed_descriptor_path=tmp_path / "descriptor.json",
    )

    assert result[1]["schema_version"] == "v5"
    assert captured["failed_job_id"] == recovery.S1_STEP_RUNTIME_PARENT_JOB_ID
    assert captured["matrix_start_receipt_path"] == tmp_path / "started.json"


def test_cli_forwards_all_v5_evidence_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    monkeypatch.setattr(
        recovery,
        "Config",
        SimpleNamespace(fromfile=lambda _: object()),
    )
    monkeypatch.setattr(
        recovery,
        "validate_bound_s1_training_config",
        lambda *_args, **_kwargs: {"code_commit": TRAINING_COMMIT},
    )

    def _fake_builder(**kwargs: object) -> tuple[Path, dict]:
        captured.update(kwargs)
        return tmp_path / "recovery.json", {
            "campaign_id": "campaign",
            "certificate_sha256": "a" * 64,
        }

    monkeypatch.setattr(recovery, "build_profile_recovery_certificate", _fake_builder)
    paths = {
        "parent": tmp_path / "parent.json",
        "start": tmp_path / "started.json",
        "submission": tmp_path / "submission.json",
        "stdout": tmp_path / "matrix.out",
        "stderr": tmp_path / "matrix.err",
        "descriptor": tmp_path / "descriptor.json",
    }
    status = recovery.main(
        [
            "--config",
            str(tmp_path / "config.py"),
            "--seed",
            "3408",
            "--failed-job-id",
            recovery.S1_STEP_RUNTIME_PARENT_JOB_ID,
            "--superseded-recovery-certificate",
            str(paths["parent"]),
            "--matrix-start-receipt",
            str(paths["start"]),
            "--matrix-submission-receipt",
            str(paths["submission"]),
            "--matrix-stdout",
            str(paths["stdout"]),
            "--matrix-stderr",
            str(paths["stderr"]),
            "--completed-descriptor",
            str(paths["descriptor"]),
        ]
    )

    assert status == 0
    assert captured["matrix_start_receipt_path"] == paths["start"]
    assert captured["matrix_submission_receipt_path"] == paths["submission"]
    assert captured["matrix_stdout_path"] == paths["stdout"]
    assert captured["matrix_stderr_path"] == paths["stderr"]
    assert captured["completed_descriptor_path"] == paths["descriptor"]


def _buffered_recovery_contract(reason: str, tmp_path: Path) -> dict:
    return {
        "reason": reason,
        "campaign_root": str(tmp_path),
        "power_sampler_backend": recovery.S1_SIDECAR_POWER_BACKEND,
        "trace_publication_mode": recovery.S1_BUFFERED_TRACE_PUBLICATION_MODE,
        "trace_io_inside_sampling_loop": False,
        "power_target_interval_ms": 20,
        "power_max_gap_limit_ms": 100.0,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
        "sidecar_gate_relative_path": "sidecar_gate.json",
    }


@pytest.mark.parametrize(
    "reason",
    (
        recovery.S1_BUFFERED_SIDECAR_RECOVERY_REASON,
        recovery.S1_STEP_RUNTIME_RECOVERY_REASON,
        recovery.S1_SCHEMA_COMPAT_RECOVERY_REASON,
    ),
)
def test_buffered_sidecar_capability_is_inherited_across_recovery_versions(
    reason: str,
    tmp_path: Path,
) -> None:
    contract = _buffered_recovery_contract(reason, tmp_path)
    assert recovery.is_buffered_sidecar_recovery(contract) is True
    assert recovery.require_buffered_sidecar_recovery(contract) is contract
    assert (
        sidecar_gate.sidecar_gate_path(contract)
        == (tmp_path / "sidecar_gate.json").resolve()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("trace_publication_mode", "line_buffered_jsonl"),
        ("trace_io_inside_sampling_loop", True),
        ("power_target_interval_ms", 25),
        ("power_target_interval_ms", True),
        ("power_target_interval_ms", "20"),
        ("power_max_gap_limit_ms", 101.0),
        ("power_max_gap_limit_ms", "100"),
        ("allocated_cpu_count", 4),
        ("sidecar_cpu_count", True),
        ("requires_long_no_open_gate", False),
    ),
)
def test_buffered_sidecar_capability_rejects_contract_drift(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    contract = _buffered_recovery_contract(
        recovery.S1_STEP_RUNTIME_RECOVERY_REASON,
        tmp_path,
    )
    contract[field] = value
    assert recovery.is_buffered_sidecar_recovery(contract) is False
    with pytest.raises(ValueError, match="buffered sidecar recovery contract"):
        recovery.require_buffered_sidecar_recovery(contract)
    with pytest.raises(ValueError, match="sidecar recovery certificate"):
        sidecar_gate.sidecar_gate_path(contract)


def test_profiler_checks_buffered_capability_not_exact_recovery_version() -> None:
    source = Path("tools/bata/profile_spatial_zoom_s1.py").read_text(encoding="utf-8")
    assert "require_buffered_sidecar_recovery(recovery)" in source
    assert "formal S1 profile requires the exact v4 buffered recovery" not in source
    assert source.index("require_buffered_sidecar_recovery(recovery)") < source.index(
        "certificate = validate_test_open_certificate("
    )


def test_v6_parent_delta_is_narrow_and_excludes_model_surfaces() -> None:
    assert recovery._REQUIRED_PARENT_TO_SCHEMA_COMPAT_PATHS == (
        recovery._PARENT_TO_SCHEMA_COMPAT_PATHS
    )
    assert recovery._SCHEMA_COMPAT_RUNTIME_PATHS.issubset(
        recovery._PARENT_TO_SCHEMA_COMPAT_PATHS
    )
    assert "tests/test_spatial_zoom_s1_infrastructure.py" in (
        recovery._PARENT_TO_SCHEMA_COMPAT_PATHS
    )
    assert not any(
        path.startswith(("opentad/", "configs/"))
        for path in recovery._PARENT_TO_SCHEMA_COMPAT_PATHS
    )


def test_v6_frozen_failed_gate_log_hashes_match_registered_evidence() -> None:
    assert recovery.S1_SCHEMA_COMPAT_GATE_STDOUT_SHA256 == (
        "8345aff79068bf9d5382a7256c35366711e4fc9fbbc2c7e1fc35bf4af4520282"
    )
    assert recovery.S1_SCHEMA_COMPAT_GATE_STDERR_SHA256 == (
        "c510b037dfd59bb67c0c2312689a6fd143011390b64eac1f9c43f10832e78154"
    )


def test_v6_build_validate_and_load_close_failed_no_open_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    certificate_path, certificate = _build_v6(context)

    assert certificate["schema_version"] == (
        recovery.S1_SCHEMA_COMPAT_PROFILE_RECOVERY_SCHEMA
    )
    assert certificate["reason"] == recovery.S1_SCHEMA_COMPAT_RECOVERY_REASON
    assert certificate["schema_compat_failed_job_id"] == (
        recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID
    )
    assert certificate["schema_compat_sidecar_gate_absent"] is True
    assert certificate["schema_compat_parent_campaign_files"] == [
        "gate_submission.json",
        f"logs/gate-{recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID}.err",
        f"logs/gate-{recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID}.out",
        "recovery_certificate.json",
    ]
    assert certificate["schema_compat_test_evidence_file_sha256"] == sha256_file(
        context["test_evidence_path"]
    )
    assert set(recovery._SCHEMA_COMPAT_RUNTIME_PATHS).issubset(
        {row["path"] for row in certificate["parent_to_current_changed_files"]}
    )

    checked = recovery.validate_profile_recovery_certificate(
        certificate,
        binding=context["binding"],
        verify_checkout=False,
    )
    loaded = recovery.load_profile_recovery_certificate(
        certificate_path,
        binding=context["binding"],
        verify_checkout=False,
    )
    assert checked == certificate
    assert loaded == certificate
    assert context["parent_loads"].count(context["parent_path"].resolve()) >= 3


def test_v6_rejects_any_parent_sidecar_gate_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    _write_json(context["parent_path"].parent / "sidecar_gate.json", {"status": "PASS"})

    with pytest.raises(ValueError, match="unexpected artifacts"):
        _build_v6(context)


def test_v6_rejects_changed_reused_official_test_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    _write_json(context["test_evidence_path"], {"official_test": "tampered"})

    with pytest.raises(ValueError, match="changed the reused official-test evidence"):
        _build_v6(context)


def test_v6_rejects_semantically_tampered_gate_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    receipt = json.loads(context["receipt_path"].read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt["test_opened"] = True
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write_json(context["receipt_path"], receipt)

    with pytest.raises(ValueError, match="test_opened mismatch"):
        _build_v6(context)


def test_v6_rejects_changed_gate_failure_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    context["stdout_path"].write_text(
        '{"status":"FAIL","error":"different failure"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen hashes mismatch"):
        _build_v6(context)


def test_v6_rejects_stdout_stderr_path_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="canonical paths mismatch"):
        recovery.build_schema_compat_profile_recovery_certificate(
            binding=context["binding"],
            superseded_recovery_certificate_path=context["parent_path"],
            gate_submission_receipt_path=context["receipt_path"],
            gate_stdout_path=context["stderr_path"],
            gate_stderr_path=context["stderr_path"],
            failed_job_id=recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID,
        )


def test_v6_rejects_canonical_gate_logs_that_alias_one_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    context["stderr_path"].unlink()
    context["stderr_path"].hardlink_to(context["stdout_path"])

    with pytest.raises(ValueError, match="cannot alias the same file"):
        _build_v6(context)


def test_v6_validator_rejects_resigned_stderr_role_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    _, certificate = _build_v6(context)
    stderr_hash = sha256_file(context["stderr_path"])
    monkeypatch.setattr(
        recovery,
        "S1_SCHEMA_COMPAT_GATE_STDOUT_SHA256",
        stderr_hash,
    )
    tampered = copy.deepcopy(certificate)
    tampered["schema_compat_gate_stdout_path"] = str(context["stderr_path"].resolve())
    tampered["schema_compat_gate_stdout_sha256"] = stderr_hash
    tampered = _resign_recovery_certificate(tampered, context["binding"])

    with pytest.raises(ValueError, match="canonical paths mismatch"):
        recovery.validate_profile_recovery_certificate(
            tampered,
            binding=context["binding"],
            verify_checkout=False,
        )


def test_v6_builder_rejects_copied_parent_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    copied_parent = tmp_path / "copied-parent.json"
    copied_parent.write_bytes(context["parent_path"].read_bytes())

    with pytest.raises(ValueError, match="outside its campaign"):
        recovery.build_schema_compat_profile_recovery_certificate(
            binding=context["binding"],
            superseded_recovery_certificate_path=copied_parent,
            gate_submission_receipt_path=context["receipt_path"],
            gate_stdout_path=context["stdout_path"],
            gate_stderr_path=context["stderr_path"],
            failed_job_id=recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID,
        )


def test_v6_validator_rejects_copied_parent_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    _, certificate = _build_v6(context)
    copied_parent = tmp_path / "copied-parent.json"
    copied_parent.write_bytes(context["parent_path"].read_bytes())
    tampered = copy.deepcopy(certificate)
    tampered["superseded_recovery_certificate_path"] = str(copied_parent.resolve())
    tampered["superseded_recovery_certificate_file_sha256"] = sha256_file(copied_parent)
    tampered = _resign_recovery_certificate(tampered, context["binding"])

    with pytest.raises(ValueError, match="outside its campaign"):
        recovery.validate_profile_recovery_certificate(
            tampered,
            binding=context["binding"],
            verify_checkout=False,
        )


def test_v6_validator_rejects_noncanonical_parent_inventory_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _v6_fixture(tmp_path, monkeypatch)
    _, certificate = _build_v6(context)
    tampered = copy.deepcopy(certificate)
    tampered["schema_compat_parent_campaign_files"].append("unexpected.json")
    basis = {
        key: value
        for key, value in tampered.items()
        if key not in {"campaign_id", "campaign_root", "certificate_sha256"}
    }
    tampered["campaign_id"] = canonical_sha256(basis)[:16]
    tampered["campaign_root"] = str(
        (
            Path(context["binding"]["canonical_experiment_root"])
            / "profile_campaigns"
            / tampered["campaign_id"]
        ).resolve()
    )
    tampered.pop("certificate_sha256")
    tampered["certificate_sha256"] = canonical_sha256(tampered)

    with pytest.raises(ValueError, match="inventory contract mismatch"):
        recovery.validate_profile_recovery_certificate(
            tampered,
            binding=context["binding"],
            verify_checkout=False,
        )


def test_generic_builder_dispatches_v5_parent_to_v6(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path = tmp_path / "parent.json"
    _write_json(parent_path, {"reason": recovery.S1_STEP_RUNTIME_RECOVERY_REASON})
    captured: dict = {}

    def _fake_v6(**kwargs: object) -> tuple[Path, dict]:
        captured.update(kwargs)
        return tmp_path / "certificate.json", {"schema_version": "v6"}

    monkeypatch.setattr(
        recovery,
        "build_schema_compat_profile_recovery_certificate",
        _fake_v6,
    )
    result = recovery.build_profile_recovery_certificate(
        binding={"code_commit": TRAINING_COMMIT},
        failed_marker_path=None,
        failure_log_path=None,
        failed_job_id=recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID,
        expected_exposure_count=None,
        expected_physical_window_count=None,
        expected_duplicate_physical_window_ids=None,
        superseded_recovery_certificate_path=parent_path,
        gate_submission_receipt_path=tmp_path / "gate_submission.json",
        gate_stdout_path=tmp_path / "gate.out",
        gate_stderr_path=tmp_path / "gate.err",
    )

    assert result[1]["schema_version"] == "v6"
    assert captured["failed_job_id"] == recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID
    assert captured["gate_submission_receipt_path"] == (
        tmp_path / "gate_submission.json"
    )


@pytest.mark.parametrize(
    ("extra_kwargs", "expected_message"),
    [
        (
            {"matrix_start_receipt_path": Path("matrix.started.json")},
            "v6 recovery rejects mixed-mode evidence",
        ),
        (
            {"power_diagnostic_path": Path("power-diagnostic.json")},
            "v6 recovery rejects mixed-mode evidence",
        ),
        (
            {"failed_marker_path": Path("legacy-marker.json")},
            "v6 recovery rejects mixed-mode evidence",
        ),
    ],
)
def test_generic_builder_rejects_mixed_evidence_for_v5_parent(
    tmp_path: Path,
    extra_kwargs: dict,
    expected_message: str,
) -> None:
    parent_path = tmp_path / "parent.json"
    _write_json(parent_path, {"reason": recovery.S1_STEP_RUNTIME_RECOVERY_REASON})
    kwargs = {
        "binding": {"code_commit": TRAINING_COMMIT},
        "failed_marker_path": None,
        "failure_log_path": None,
        "failed_job_id": recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID,
        "expected_exposure_count": None,
        "expected_physical_window_count": None,
        "expected_duplicate_physical_window_ids": None,
        "superseded_recovery_certificate_path": parent_path,
        "gate_submission_receipt_path": tmp_path / "gate_submission.json",
        "gate_stdout_path": tmp_path / "gate.out",
        "gate_stderr_path": tmp_path / "gate.err",
    }
    kwargs.update(extra_kwargs)

    with pytest.raises(ValueError, match=expected_message):
        recovery.build_profile_recovery_certificate(**kwargs)


@pytest.mark.parametrize(
    "extra_kwargs",
    (
        {"gate_stdout_path": Path("gate.out")},
        {"failed_marker_path": Path("legacy-marker.json")},
        {"power_diagnostic_path": Path("power-diagnostic.json")},
    ),
)
def test_generic_builder_rejects_mixed_evidence_for_v4_parent(
    tmp_path: Path,
    extra_kwargs: dict,
) -> None:
    parent_path = tmp_path / "parent.json"
    _write_json(
        parent_path,
        {"reason": recovery.S1_BUFFERED_SIDECAR_RECOVERY_REASON},
    )
    kwargs = {
        "binding": {"code_commit": TRAINING_COMMIT},
        "failed_marker_path": None,
        "failure_log_path": None,
        "failed_job_id": recovery.S1_STEP_RUNTIME_PARENT_JOB_ID,
        "expected_exposure_count": None,
        "expected_physical_window_count": None,
        "expected_duplicate_physical_window_ids": None,
        "superseded_recovery_certificate_path": parent_path,
        "matrix_start_receipt_path": tmp_path / "started.json",
        "matrix_submission_receipt_path": tmp_path / "submission.json",
        "matrix_stdout_path": tmp_path / "matrix.out",
        "matrix_stderr_path": tmp_path / "matrix.err",
        "completed_descriptor_path": tmp_path / "descriptor.json",
    }
    kwargs.update(extra_kwargs)

    with pytest.raises(ValueError, match="v5 recovery rejects mixed-mode evidence"):
        recovery.build_profile_recovery_certificate(**kwargs)


def test_generic_builder_rejects_incomplete_v6_gate_evidence(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "parent.json"
    _write_json(parent_path, {"reason": recovery.S1_STEP_RUNTIME_RECOVERY_REASON})

    with pytest.raises(ValueError, match="missing required evidence"):
        recovery.build_profile_recovery_certificate(
            binding={"code_commit": TRAINING_COMMIT},
            failed_marker_path=None,
            failure_log_path=None,
            failed_job_id=recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID,
            expected_exposure_count=None,
            expected_physical_window_count=None,
            expected_duplicate_physical_window_ids=None,
            superseded_recovery_certificate_path=parent_path,
            gate_stdout_path=tmp_path / "gate.out",
        )


def test_generic_builder_rejects_incomplete_v5_matrix_evidence(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "parent.json"
    _write_json(
        parent_path,
        {"reason": recovery.S1_BUFFERED_SIDECAR_RECOVERY_REASON},
    )

    with pytest.raises(ValueError, match="missing required evidence"):
        recovery.build_profile_recovery_certificate(
            binding={"code_commit": TRAINING_COMMIT},
            failed_marker_path=None,
            failure_log_path=None,
            failed_job_id=recovery.S1_STEP_RUNTIME_PARENT_JOB_ID,
            expected_exposure_count=None,
            expected_physical_window_count=None,
            expected_duplicate_physical_window_ids=None,
            superseded_recovery_certificate_path=parent_path,
            matrix_stdout_path=tmp_path / "matrix.out",
        )


def test_generic_builder_rejects_unaudited_v6_supersession(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "parent.json"
    _write_json(parent_path, {"reason": recovery.S1_SCHEMA_COMPAT_RECOVERY_REASON})

    with pytest.raises(ValueError, match="cannot be superseded"):
        recovery.build_profile_recovery_certificate(
            binding={"code_commit": TRAINING_COMMIT},
            failed_marker_path=None,
            failure_log_path=None,
            failed_job_id="future-job",
            expected_exposure_count=None,
            expected_physical_window_count=None,
            expected_duplicate_physical_window_ids=None,
            superseded_recovery_certificate_path=parent_path,
        )


def test_cli_forwards_all_v6_gate_evidence_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        recovery,
        "Config",
        SimpleNamespace(fromfile=lambda _: object()),
    )
    monkeypatch.setattr(
        recovery,
        "validate_bound_s1_training_config",
        lambda *_args, **_kwargs: {"code_commit": TRAINING_COMMIT},
    )

    def _fake_builder(**kwargs: object) -> tuple[Path, dict]:
        captured.update(kwargs)
        return tmp_path / "recovery.json", {
            "campaign_id": "campaign",
            "certificate_sha256": "a" * 64,
        }

    monkeypatch.setattr(recovery, "build_profile_recovery_certificate", _fake_builder)
    paths = {
        "parent": tmp_path / "parent.json",
        "submission": tmp_path / "gate_submission.json",
        "stdout": tmp_path / "gate.out",
        "stderr": tmp_path / "gate.err",
    }
    status = recovery.main(
        [
            "--config",
            str(tmp_path / "config.py"),
            "--seed",
            "3408",
            "--failed-job-id",
            recovery.S1_SCHEMA_COMPAT_PARENT_JOB_ID,
            "--superseded-recovery-certificate",
            str(paths["parent"]),
            "--gate-submission-receipt",
            str(paths["submission"]),
            "--gate-stdout",
            str(paths["stdout"]),
            "--gate-stderr",
            str(paths["stderr"]),
        ]
    )

    assert status == 0
    assert captured["gate_submission_receipt_path"] == paths["submission"]
    assert captured["gate_stdout_path"] == paths["stdout"]
    assert captured["gate_stderr_path"] == paths["stderr"]
