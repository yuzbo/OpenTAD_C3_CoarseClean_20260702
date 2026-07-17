from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.bata.spatial_zoom_s1_contract import (
    atomic_publish_json,
    build_s1_profile_order,
    canonical_sha256,
)
from tools.bata.spatial_zoom_s1_matrix import (
    S1_PROFILE_MATRIX_COMPLETION_SCHEMA,
    S1_PROFILE_MATRIX_START_SCHEMA,
    S1_TEST_MATRIX_BINDING_SCHEMA,
    build_profile_matrix_completion_receipt,
    build_profile_matrix_start_receipt,
    build_test_matrix_binding,
    canonical_matrix_completion_path,
    canonical_matrix_start_path,
    canonical_test_matrix_binding_path,
    validate_profile_matrix_completion_receipt,
    validate_profile_matrix_start_receipt,
    validate_test_matrix_binding,
)
from tools.bata.spatial_zoom_s1_sidecar_gate import (
    sidecar_gate_hardware_class,
)


@pytest.fixture
def matrix_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    from tools.bata import spatial_zoom_s1_profile_recovery as recovery_module
    from tools.bata import spatial_zoom_s1_sidecar_gate as gate_module

    campaign_root = (tmp_path / "campaign").resolve()
    profile_commit = "a" * 40
    software_fingerprint = canonical_sha256({"software": "s1-matrix-test"})
    env = {
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "7",
        "SLURM_JOB_GPUS": "1,2",
        "SLURM_STEP_GPUS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "SLURM_CPUS_PER_TASK": "5",
        "SLURM_MEM_PER_NODE": "124400",
    }
    hardware = {
        "node": "s1-node",
        "machine": "x86_64",
        "cpu_model": "S1 CPU",
        "gpu_name": "S1 GPU",
        "gpu_total_memory": 48 * 1024**3,
        "gpu_compute_capability": [8, 0],
        "gpu_multi_processor_count": 108,
        "physical_gpu_id": "1",
        "scoped_gpu_id": "1",
        "step_gpu_uuid": "GPU-S1-MATRIX",
        "cuda_visible_device_uuid": "GPU-S1-MATRIX",
        "cuda_visible_devices": "0",
        "nvidia_smi": {
            "uuid": "GPU-S1-MATRIX",
            "driver_version": "550.54",
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
            "power.limit": "300.00",
            "clocks.max.sm": "1410",
            "clocks.max.memory": "1215",
        },
        "slurm_resources": {
            "slurm_job_id": "12345",
            "slurm_step_id": "7",
            "slurm_job_gpus": "1,2",
            "slurm_step_gpus": "1",
            "cpus_per_task": 5,
            "effective_step_memory_limit_mb": 96000,
            "outer_job_mem_per_node_mb": 124400,
            "scoped_gpu_id": "1",
            "step_gpu_id": "1",
            "step_gpu_uuid": "GPU-S1-MATRIX",
            "cuda_visible_devices": "0",
            "allocated_cpu_ids": [0, 1, 2, 3, 4],
            "detector_cpu_ids": [0, 1, 2, 3],
            "sidecar_cpu_id": 4,
            "detector_process_affinity": [0, 1, 2, 3],
        },
    }
    recovery = {
        "schema_version": "spatial_zoom_s1_profile_recovery_v3",
        "reason": recovery_module.S1_SIDECAR_RECOVERY_REASON,
        "campaign_id": "matrix-campaign",
        "campaign_root": str(campaign_root),
        "sidecar_gate_relative_path": "sidecar_gate.json",
        "training_code_commit": "b" * 40,
        "profile_code_commit": profile_commit,
        "experiment_namespace": "s1-matrix",
        "canonical_experiment_root": str(tmp_path / "canonical"),
        "manifest_sha256": "c" * 64,
        "protocol_fingerprint": "d" * 64,
        "precheck_file_sha256": "e" * 64,
        "precheck_sha256": "f" * 64,
        "pretrained_checkpoint_sha256": "1" * 64,
    }
    recovery["certificate_sha256"] = canonical_sha256(recovery)
    recovery_path = campaign_root / "recovery_certificate.json"
    atomic_publish_json(recovery_path, recovery)

    hardware_class = sidecar_gate_hardware_class(hardware)
    gate = {
        "schema_version": "spatial_zoom_s1_power_sidecar_gate_v1",
        "status": "PASS",
        "profile_code_commit": profile_commit,
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
        "hardware_class": hardware_class,
        "hardware_class_sha256": canonical_sha256(hardware_class),
        "software_fingerprint": software_fingerprint,
        "gate_gpu_uuid": "GPU-S1-MATRIX",
    }
    gate["gate_sha256"] = canonical_sha256(gate)
    gate_path = campaign_root / "sidecar_gate.json"
    atomic_publish_json(gate_path, gate)

    def fake_load_recovery(path, *, binding, verify_checkout):
        assert Path(path).resolve() == recovery_path.resolve()
        assert binding["code_commit"] == recovery["training_code_commit"]
        assert verify_checkout is False
        return copy.deepcopy(recovery)

    def fake_load_gate(path, *, recovery):
        assert Path(path).resolve() == gate_path.resolve()
        assert (
            recovery["certificate_sha256"]
            == recovery_module_value["certificate_sha256"]
        )
        return copy.deepcopy(gate)

    recovery_module_value = recovery
    monkeypatch.setattr(
        recovery_module,
        "load_profile_recovery_certificate",
        fake_load_recovery,
    )
    monkeypatch.setattr(
        gate_module,
        "load_sidecar_gate_evidence",
        fake_load_gate,
    )

    fixture = {
        "campaign_root": campaign_root,
        "profile_commit": profile_commit,
        "software_fingerprint": software_fingerprint,
        "env": env,
        "hardware": hardware,
        "recovery": recovery,
        "recovery_path": recovery_path,
        "gate": gate,
        "gate_path": gate_path,
        "order": build_s1_profile_order(),
    }
    fixture["start"] = build_profile_matrix_start_receipt(
        recovery=recovery,
        sidecar_gate=gate,
        hardware_identity=hardware,
        software_fingerprint=software_fingerprint,
        profile_code_commit=profile_commit,
        frozen_order=fixture["order"],
        env=env,
    )
    fixture["start_path"] = canonical_matrix_start_path(recovery)
    atomic_publish_json(fixture["start_path"], fixture["start"])
    fixture["descriptor_paths"] = _write_descriptors(fixture)
    return fixture


def _write_descriptor(
    fixture: dict,
    row: dict,
    *,
    suffix: str = "",
    overrides: dict | None = None,
) -> Path:
    start = fixture["start"]
    start_path = fixture["start_path"]
    descriptor = {
        "schema_version": "spatial_zoom_s1_run_v7-test",
        "resolution": row["resolution"],
        "seed": row["seed"],
        "profile_order_ordinal": row["ordinal"],
        "matrix_start_receipt_path": str(start_path),
        "matrix_start_receipt_file_sha256": _sha256_file(start_path),
        "matrix_sha256": start["matrix_sha256"],
        "slurm_job_id": start["slurm_job_id"],
        "slurm_step_id": start["slurm_step_id"],
        "step_gpu_uuid": start["step_gpu_uuid"],
    }
    if overrides:
        descriptor.update(overrides)
    descriptor["descriptor_sha256"] = canonical_sha256(descriptor)
    path = (
        fixture["campaign_root"]
        / "descriptors"
        / f"dense{row['resolution']}_seed{row['seed']}{suffix}.run.json"
    )
    atomic_publish_json(path, descriptor)
    return path.resolve()


def _write_descriptors(fixture: dict) -> list[Path]:
    return [_write_descriptor(fixture, row) for row in fixture["order"]]


def _sha256_file(path: Path) -> str:
    from tools.bata.spatial_zoom_s1_contract import sha256_file

    return sha256_file(path)


def _rehash(value: dict, key: str) -> dict:
    checked = copy.deepcopy(value)
    checked.pop(key, None)
    checked[key] = canonical_sha256(checked)
    return checked


def test_start_and_completion_receipts_round_trip(matrix_fixture: dict) -> None:
    fixture = matrix_fixture
    recovery = fixture["recovery"]
    assert fixture["start"]["schema_version"] == S1_PROFILE_MATRIX_START_SCHEMA
    assert (
        canonical_matrix_start_path(recovery)
        == (fixture["campaign_root"] / "matrix.lock" / "matrix.started.json").resolve()
    )
    assert (
        canonical_matrix_completion_path(recovery)
        == (
            fixture["campaign_root"] / "matrix.lock" / "matrix.completed.json"
        ).resolve()
    )
    assert (
        validate_profile_matrix_start_receipt(
            fixture["start_path"],
            recovery=recovery,
            verify_runtime=True,
            hardware_identity=fixture["hardware"],
            software_fingerprint=fixture["software_fingerprint"],
            effective_memory_limit_mb=96000,
            env=fixture["env"],
        )
        == fixture["start"]
    )

    completion = build_profile_matrix_completion_receipt(
        start_receipt_path=fixture["start_path"],
        recovery=recovery,
        descriptor_paths=list(reversed(fixture["descriptor_paths"])),
    )
    assert completion["schema_version"] == S1_PROFILE_MATRIX_COMPLETION_SCHEMA
    assert [row["profile_order_ordinal"] for row in completion["descriptors"]] == list(
        range(9)
    )
    completion_path = canonical_matrix_completion_path(recovery)
    atomic_publish_json(completion_path, completion)
    assert (
        validate_profile_matrix_completion_receipt(
            completion_path,
            recovery=recovery,
            descriptor_paths=fixture["descriptor_paths"],
        )
        == completion
    )


def test_test_evidence_binding_round_trip_and_tamper(
    matrix_fixture: dict,
) -> None:
    fixture = matrix_fixture
    first = fixture["order"][0]
    evidence_path = fixture["campaign_root"] / "test" / "test.evidence.json"
    evidence = {
        "schema_version": "spatial_zoom_s1_test_evidence_v4",
        "resolution": first["resolution"],
        "seed": first["seed"],
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    atomic_publish_json(evidence_path, evidence)
    binding = build_test_matrix_binding(
        test_evidence_path=evidence_path,
        start_receipt_path=fixture["start_path"],
        recovery=fixture["recovery"],
        resolution=first["resolution"],
        seed=first["seed"],
    )
    assert binding["schema_version"] == S1_TEST_MATRIX_BINDING_SCHEMA
    binding_path = canonical_test_matrix_binding_path(evidence_path)
    atomic_publish_json(binding_path, binding)
    assert (
        validate_test_matrix_binding(
            binding_path,
            test_evidence_path=evidence_path,
            start_receipt_path=fixture["start_path"],
            recovery=fixture["recovery"],
            resolution=first["resolution"],
            seed=first["seed"],
        )
        == binding
    )

    tampered = copy.deepcopy(binding)
    tampered["matrix_sha256"] = "9" * 64
    tampered = _rehash(tampered, "binding_sha256")
    binding_path.unlink()
    atomic_publish_json(binding_path, tampered)
    with pytest.raises(ValueError, match="does not match current evidence"):
        validate_test_matrix_binding(
            binding_path,
            test_evidence_path=evidence_path,
            start_receipt_path=fixture["start_path"],
            recovery=fixture["recovery"],
            resolution=first["resolution"],
            seed=first["seed"],
        )


def test_start_rejects_step_gpu_outside_job_allocation(
    matrix_fixture: dict,
) -> None:
    fixture = matrix_fixture
    forged_env = dict(fixture["env"])
    forged_env["SLURM_STEP_GPUS"] = "3"
    forged_hardware = copy.deepcopy(fixture["hardware"])
    forged_hardware["physical_gpu_id"] = "3"
    forged_hardware["scoped_gpu_id"] = "3"
    forged_hardware["slurm_resources"]["scoped_gpu_id"] = "3"
    forged_hardware["slurm_resources"]["step_gpu_id"] = "3"
    with pytest.raises(ValueError, match="not a member"):
        build_profile_matrix_start_receipt(
            recovery=fixture["recovery"],
            sidecar_gate=fixture["gate"],
            hardware_identity=forged_hardware,
            software_fingerprint=fixture["software_fingerprint"],
            profile_code_commit=fixture["profile_commit"],
            frozen_order=fixture["order"],
            env=forged_env,
        )


def test_start_rejects_missing_step_and_invalid_cpu_partition(
    matrix_fixture: dict,
) -> None:
    fixture = matrix_fixture
    missing_step_env = dict(fixture["env"])
    missing_step_env.pop("SLURM_STEP_ID")
    with pytest.raises(ValueError, match="SLURM_STEP_ID"):
        build_profile_matrix_start_receipt(
            recovery=fixture["recovery"],
            sidecar_gate=fixture["gate"],
            hardware_identity=fixture["hardware"],
            software_fingerprint=fixture["software_fingerprint"],
            profile_code_commit=fixture["profile_commit"],
            frozen_order=fixture["order"],
            env=missing_step_env,
        )

    bad_cpu_hardware = copy.deepcopy(fixture["hardware"])
    bad_cpu_hardware["slurm_resources"]["detector_process_affinity"] = [0, 1, 2]
    with pytest.raises(ValueError, match=r"4\+1 CPU partition"):
        build_profile_matrix_start_receipt(
            recovery=fixture["recovery"],
            sidecar_gate=fixture["gate"],
            hardware_identity=bad_cpu_hardware,
            software_fingerprint=fixture["software_fingerprint"],
            profile_code_commit=fixture["profile_commit"],
            frozen_order=fixture["order"],
            env=fixture["env"],
        )


@pytest.mark.parametrize(
    ("field", "forged_value", "message"),
    [
        ("slurm_job_id", "99999", "slurm_job_id"),
        ("slurm_step_id", "999", "slurm_step_id"),
    ],
)
def test_runtime_rejects_forged_job_or_step(
    matrix_fixture: dict,
    field: str,
    forged_value: str,
    message: str,
) -> None:
    fixture = matrix_fixture
    forged = copy.deepcopy(fixture["start"])
    forged[field] = forged_value
    forged = _rehash(forged, "matrix_sha256")
    with pytest.raises(ValueError, match=message):
        validate_profile_matrix_start_receipt(
            forged,
            recovery=fixture["recovery"],
            verify_runtime=True,
            hardware_identity=fixture["hardware"],
            software_fingerprint=fixture["software_fingerprint"],
            effective_memory_limit_mb=96000,
            env=fixture["env"],
        )


def test_start_rejects_self_hash_and_canonical_path_errors(
    matrix_fixture: dict,
) -> None:
    fixture = matrix_fixture
    bad_hash = copy.deepcopy(fixture["start"])
    bad_hash["slurm_job_id"] = "99999"
    with pytest.raises(ValueError, match="self-hash"):
        validate_profile_matrix_start_receipt(bad_hash, recovery=fixture["recovery"])

    bad_path = copy.deepcopy(fixture["start"])
    bad_path["matrix_start_receipt_path"] = str(
        fixture["campaign_root"] / "elsewhere" / "matrix.started.json"
    )
    bad_path = _rehash(bad_path, "matrix_sha256")
    with pytest.raises(ValueError, match="canonical path"):
        validate_profile_matrix_start_receipt(bad_path, recovery=fixture["recovery"])


def test_completion_rejects_forged_descriptor_matrix_binding(
    matrix_fixture: dict,
) -> None:
    fixture = matrix_fixture
    forged_path = fixture["descriptor_paths"][0]
    forged = _read_json(forged_path)
    forged["matrix_sha256"] = "9" * 64
    forged = _rehash(forged, "descriptor_sha256")
    forged_path.unlink()
    atomic_publish_json(forged_path, forged)
    with pytest.raises(ValueError, match="matrix_sha256"):
        build_profile_matrix_completion_receipt(
            start_receipt_path=fixture["start_path"],
            recovery=fixture["recovery"],
            descriptor_paths=fixture["descriptor_paths"],
        )


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_completion_rejects_missing_and_duplicate_descriptors(
    matrix_fixture: dict,
) -> None:
    fixture = matrix_fixture
    with pytest.raises(ValueError, match="exactly 9"):
        build_profile_matrix_completion_receipt(
            start_receipt_path=fixture["start_path"],
            recovery=fixture["recovery"],
            descriptor_paths=fixture["descriptor_paths"][:-1],
        )
    duplicate_paths = fixture["descriptor_paths"][:-1] + [
        fixture["descriptor_paths"][0]
    ]
    with pytest.raises(ValueError, match="duplicate descriptor paths"):
        build_profile_matrix_completion_receipt(
            start_receipt_path=fixture["start_path"],
            recovery=fixture["recovery"],
            descriptor_paths=duplicate_paths,
        )

    noncanonical_cell_path = _write_descriptor(
        fixture,
        fixture["order"][0],
        suffix="_duplicate-cell",
    )
    noncanonical_cells = fixture["descriptor_paths"][:-1] + [
        noncanonical_cell_path
    ]
    with pytest.raises(ValueError, match="canonical campaign path"):
        build_profile_matrix_completion_receipt(
            start_receipt_path=fixture["start_path"],
            recovery=fixture["recovery"],
            descriptor_paths=noncanonical_cells,
        )


def test_completion_rejects_self_hash_path_and_input_mismatch(
    matrix_fixture: dict,
) -> None:
    fixture = matrix_fixture
    completion = build_profile_matrix_completion_receipt(
        start_receipt_path=fixture["start_path"],
        recovery=fixture["recovery"],
        descriptor_paths=fixture["descriptor_paths"],
    )
    bad_hash = copy.deepcopy(completion)
    bad_hash["slurm_job_id"] = "99999"
    with pytest.raises(ValueError, match="self-hash"):
        validate_profile_matrix_completion_receipt(
            bad_hash, recovery=fixture["recovery"]
        )

    bad_path = copy.deepcopy(completion)
    bad_path["matrix_completion_receipt_path"] = str(
        fixture["campaign_root"] / "elsewhere" / "matrix.completed.json"
    )
    bad_path = _rehash(bad_path, "completion_sha256")
    with pytest.raises(ValueError, match="canonical path"):
        validate_profile_matrix_completion_receipt(
            bad_path, recovery=fixture["recovery"]
        )

    completion_path = canonical_matrix_completion_path(fixture["recovery"])
    atomic_publish_json(completion_path, completion)
    wrong_paths = list(fixture["descriptor_paths"])
    wrong_paths[-1] = fixture["campaign_root"] / "descriptors" / "wrong.json"
    with pytest.raises(ValueError, match="do not match"):
        validate_profile_matrix_completion_receipt(
            completion_path,
            recovery=fixture["recovery"],
            descriptor_paths=wrong_paths,
        )
