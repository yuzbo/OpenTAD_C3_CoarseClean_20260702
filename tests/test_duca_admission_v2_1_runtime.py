from __future__ import annotations

import copy
import hashlib
import os

import pytest

from tools.bata.duca_admission_v2_1_incidence import build_incidence
from tools.bata.duca_admission_v2_1_metrics import build_metric_registry
from tools.bata.duca_admission_v2_1_roles import build_role_manifest
from tools.bata.duca_admission_v2_1_runtime_receipt import (
    build_control_evidence,
    build_runtime_bindings,
    build_planned_cell_manifest,
    build_protocol_only_runtime_receipt,
    load_control_registry,
    validate_control_registry,
    validate_parent_receipt,
    validate_planned_cell_manifest,
    validate_planned_cell_outputs,
    validate_runtime_receipt,
    validate_worker_identities,
)
from tools.bata.duca_evidence_io import with_content_sha256
from tools.bata.duca_safe_publication import (
    RUNTIME_ROOT_REGISTRY_SCHEMA,
    create_fresh_run_root,
    publish_json_under_run_root,
)


def make_inventory():
    return [
        {
            "video_id": f"long_{index:03d}",
            "source_subset": "training",
            "frame_count": (900 + index) * 4,
            "snippet_count": 900 + index,
            "natural_window_valid_lengths": [768],
        }
        for index in range(70)
    ] + [
        {
            "video_id": f"short_{index:03d}",
            "source_subset": "training",
            "frame_count": (100 + 20 * index) * 4,
            "snippet_count": 100 + 20 * index,
            "natural_window_valid_lengths": [100 + 20 * index],
        }
        for index in range(30)
    ]


def passing_controls(registry):
    output = []
    for row in registry["controls"]:
        classification = row["classification"]
        if classification == "REPOSITORY_ENFORCED":
            state = "VERIFIED"
        elif classification == "CLUSTER_ATTESTED":
            state = "ATTESTED"
        elif classification == "OBSERVED_ONLY":
            state = "OBSERVED"
        else:
            state = "NOT_AVAILABLE"
        output.append(
            {
                "control_id": row["control_id"],
                "classification": classification,
                "claim_state": state,
                "evidence": build_control_evidence(
                    control_id=row["control_id"],
                    claim_state=state,
                    evidence_refs=[
                        {
                            "kind": "unit_test_fixture",
                            "sha256": hashlib.sha256(
                                row["control_id"].encode("ascii")
                            ).hexdigest(),
                        }
                    ],
                ),
            }
        )
    return output


def passing_evidence_verifiers(registry):
    return {
        row["control_id"]: (lambda evidence, control: True)
        for row in registry["controls"]
        if row["pass_required"]
    }


def runtime_bindings():
    return build_runtime_bindings(
        artifact_refs=[{"kind": "unit_test_fixture", "sha256": "a" * 64}]
    )


def test_closed_control_registry_and_protocol_receipt_never_authorize():
    registry = load_control_registry()
    assert len(registry["controls"]) == 37
    receipt = build_protocol_only_runtime_receipt(
        receipt_kind="unit_test",
        status="PASSED",
        controls=passing_controls(registry),
        bindings=runtime_bindings(),
    )
    with pytest.raises(ValueError, match="not independently verified"):
        validate_runtime_receipt(receipt, control_registry=registry)
    validate_runtime_receipt(
        receipt,
        control_registry=registry,
        evidence_verifiers=passing_evidence_verifiers(registry),
    )
    assert receipt["authorization_scope"] == "NONE"
    assert receipt["phase1_v2_authorized"] is False
    assert receipt["official_final_sealed"] is True


def test_runtime_receipt_rejects_unknown_duplicate_missing_and_illegal_state():
    registry = load_control_registry()
    controls = passing_controls(registry)
    for mutant in (
        controls[:-1],
        [*controls, copy.deepcopy(controls[0])],
        [*controls[:-1], {**controls[-1], "control_id": "UNKNOWN"}],
    ):
        receipt = build_protocol_only_runtime_receipt(
            receipt_kind="runtime_preflight",
            status="PASSED",
            controls=mutant,
            bindings=runtime_bindings(),
        )
        with pytest.raises(ValueError, match="missing, extra or duplicate"):
            validate_runtime_receipt(receipt, control_registry=registry)
    illegal = copy.deepcopy(controls)
    illegal[0]["claim_state"] = "OBSERVED"
    receipt = build_protocol_only_runtime_receipt(
        receipt_kind="runtime_preflight",
        status="PASSED",
        controls=illegal,
        bindings=runtime_bindings(),
    )
    with pytest.raises(ValueError, match="illegal classification"):
        validate_runtime_receipt(receipt, control_registry=registry)


def test_failed_closed_runtime_receipt_cannot_authorize():
    registry = load_control_registry()
    controls = passing_controls(registry)
    controls[0]["claim_state"] = "FAILED"
    controls[0]["evidence"] = build_control_evidence(
        control_id=controls[0]["control_id"],
        claim_state="FAILED",
        evidence_refs=[{"kind": "failure", "sha256": "f" * 64}],
    )
    receipt = build_protocol_only_runtime_receipt(
        receipt_kind="runtime_preflight",
        status="FAILED_CLOSED",
        controls=controls,
        bindings=runtime_bindings(),
        failure_codes=["IDENTITY_MISMATCH"],
    )
    receipt["authorization_scope"] = "PHASE1_V2"
    receipt = with_content_sha256(receipt)
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_runtime_receipt(receipt, control_registry=registry)


def test_planned_cell_manifest_checks_exact_identity_set():
    roles = build_role_manifest(
        inventory_records=make_inventory(), source_split_artifact_sha256="a" * 64
    )
    incidence = build_incidence(roles)
    metric_registry = build_metric_registry()
    planned = build_planned_cell_manifest(
        incidence=incidence,
        metric_registry_sha256=metric_registry["content_sha256"],
    )
    identifiers = [row["cell_id"] for row in planned["cells"]]
    assert len(identifiers) == 192
    validate_planned_cell_outputs(planned, identifiers, incidence=incidence)
    with pytest.raises(ValueError, match="do not equal"):
        validate_planned_cell_outputs(planned, identifiers[:-1] + ["f" * 64])


def test_control_registry_and_planned_manifest_reject_rehashed_drift():
    registry = load_control_registry()
    core = {
        key: value
        for key, value in registry.items()
        if key not in {"artifact_sha256", "semantic_sha256"}
    }
    control_tamper = copy.deepcopy(core)
    control_tamper["controls"][0]["control_id"] = "R01_ARBITRARY"
    with pytest.raises(ValueError, match="rows drifted"):
        validate_control_registry(control_tamper)
    failure_tamper = copy.deepcopy(core)
    failure_tamper["failure_codes"][0] = "ARBITRARY"
    with pytest.raises(ValueError, match="failure code registry drifted"):
        validate_control_registry(failure_tamper)

    roles = build_role_manifest(
        inventory_records=make_inventory(), source_split_artifact_sha256="a" * 64
    )
    incidence = build_incidence(roles)
    metric_registry = build_metric_registry()
    planned = build_planned_cell_manifest(
        incidence=incidence,
        metric_registry_sha256=metric_registry["content_sha256"],
    )
    tampered = copy.deepcopy(planned)
    tampered["cells"][0]["expected_output_relpath"] = "cells/elsewhere.json"
    tampered = with_content_sha256(tampered)
    with pytest.raises(ValueError, match="output path drifted"):
        validate_planned_cell_manifest(tampered, incidence=incidence)


def test_runtime_receipt_rejects_unknown_kind_non_none_scope_and_duplicate_failure_code():
    registry = load_control_registry()
    controls = passing_controls(registry)
    receipt = build_protocol_only_runtime_receipt(
        receipt_kind="arbitrary",
        status="PASSED",
        controls=controls,
        bindings=runtime_bindings(),
    )
    with pytest.raises(ValueError, match="unknown runtime receipt kind"):
        validate_runtime_receipt(receipt, control_registry=registry)
    receipt = build_protocol_only_runtime_receipt(
        receipt_kind="unit_test",
        status="PASSED",
        controls=controls,
        bindings=runtime_bindings(),
    )
    receipt["authorization_scope"] = "HOLDOUT_ONCE"
    receipt = with_content_sha256(receipt)
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_runtime_receipt(receipt, control_registry=registry)
    failed = copy.deepcopy(controls)
    failed[0]["claim_state"] = "FAILED"
    failed[0]["evidence"] = build_control_evidence(
        control_id=failed[0]["control_id"],
        claim_state="FAILED",
        evidence_refs=[{"kind": "failure", "sha256": "f" * 64}],
    )
    receipt = build_protocol_only_runtime_receipt(
        receipt_kind="runtime_preflight",
        status="FAILED_CLOSED",
        controls=failed,
        bindings=runtime_bindings(),
        failure_codes=["IDENTITY_MISMATCH", "IDENTITY_MISMATCH"],
    )
    with pytest.raises(ValueError, match="failure codes are invalid"):
        validate_runtime_receipt(receipt, control_registry=registry)


def test_worker_identity_validator_binds_all_24_planned_workers():
    roles = build_role_manifest(
        inventory_records=make_inventory(), source_split_artifact_sha256="a" * 64
    )
    incidence = build_incidence(roles)
    metric_registry = build_metric_registry()
    planned = build_planned_cell_manifest(
        incidence=incidence,
        metric_registry_sha256=metric_registry["content_sha256"],
    )
    workers = sorted({row["worker_id"] for row in planned["cells"]})
    allocations = []
    identities = []
    for index, worker_id in enumerate(workers):
        gpu_uuid = f"GPU-{index:02d}"
        allocations.append(
            {
                "worker_id": worker_id,
                "slurm_job_id": f"job-{index}",
                "slurm_step_id": f"step-{index}",
                "gpu_uuids": [gpu_uuid],
            }
        )
        identities.append(
            {
                "worker_id": worker_id,
                "slurm_job_id": f"job-{index}",
                "slurm_step_id": f"step-{index}",
                "pid": 1000 + index,
                "pid_start_ticks": 2000 + index,
                "boot_id": "boot",
                "cgroup_inode": 3000 + index,
                "allocated_gpu_uuids": [gpu_uuid],
                "gpu_cardinality": 1,
                "cgroup_job_membership_verified": True,
                "cuda_slurm_mapping_verified": True,
            }
        )
    validate_worker_identities(
        identities, planned_manifest=planned, planned_allocations=allocations
    )
    with pytest.raises(ValueError, match="requires live identity"):
        validate_worker_identities(
            identities,
            planned_manifest=planned,
            planned_allocations=allocations,
            require_live_verification=True,
        )
    baseline = copy.deepcopy(identities)
    for field, value, match in (
        ("slurm_job_id", "job-drift", "Slurm job allocation drifted"),
        ("slurm_step_id", "step-drift", "Slurm step allocation drifted"),
        ("allocated_gpu_uuids", ["GPU-drift"], "GPU UUID allocation drifted"),
    ):
        mutant = copy.deepcopy(baseline)
        mutant[0][field] = value
        with pytest.raises(ValueError, match=match):
            validate_worker_identities(
                mutant, planned_manifest=planned, planned_allocations=allocations
            )

    live_map = {row["worker_id"]: row for row in baseline}
    validate_worker_identities(
        baseline,
        planned_manifest=planned,
        planned_allocations=allocations,
        live_identity_collector=lambda worker_id: live_map[worker_id],
        require_live_verification=True,
    )
    for field, value in (
        ("pid_start_ticks", -1),
        ("boot_id", "boot-drift"),
        ("cgroup_inode", -1),
    ):
        mutant = copy.deepcopy(baseline)
        mutant[0][field] = value
        with pytest.raises(ValueError, match="live attestation"):
            validate_worker_identities(
                mutant,
                planned_manifest=planned,
                planned_allocations=allocations,
                live_identity_collector=lambda worker_id: live_map[worker_id],
                require_live_verification=True,
            )


@pytest.mark.skipif(
    os.name != "posix", reason="authoritative path policy is POSIX-only"
)
def test_safe_publication_and_parent_receipt_reject_symlink_paths(tmp_path):
    base = tmp_path / "base"
    base.mkdir(mode=0o700)
    registry = with_content_sha256(
        {
            "schema": RUNTIME_ROOT_REGISTRY_SCHEMA,
            "allowlisted_base_roots": [str(base)],
        }
    )
    run_root = create_fresh_run_root(base / "run", root_registry=registry)
    payload = with_content_sha256(
        {"schema": "parent_v1", "stage": "parent", "status": "PASSED"}
    )
    target = publish_json_under_run_root(
        run_root / "receipts" / "parent.json",
        payload,
        run_root=run_root,
        root_registry=registry,
    )
    with pytest.raises(FileExistsError):
        publish_json_under_run_root(
            target, payload, run_root=run_root, root_registry=registry
        )
    raw = target.read_bytes()
    binding = {
        "schema": "parent_v1",
        "stage": "parent",
        "status": "PASSED",
        "path": str(target),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": payload["content_sha256"],
    }
    assert (
        validate_parent_receipt(
            binding,
            root_registry=registry,
            expected_schema="parent_v1",
            expected_stage="parent",
        )
        == payload
    )
    with pytest.raises(ValueError, match="file SHA-256 drift"):
        validate_parent_receipt(
            {**binding, "file_sha256": "0" * 64},
            root_registry=registry,
            expected_schema="parent_v1",
            expected_stage="parent",
        )
    with pytest.raises(ValueError, match="expected identity"):
        validate_parent_receipt(
            binding,
            root_registry=registry,
            expected_schema="different_parent_v1",
            expected_stage="parent",
        )
    old_binding = {**binding, "schema": "duca_acquisition_admission_v2"}
    with pytest.raises(ValueError, match="permanently rejected"):
        validate_parent_receipt(
            old_binding,
            root_registry=registry,
            expected_schema="duca_acquisition_admission_v2",
            expected_stage="parent",
        )
    link = run_root / "receipts" / "parent-link.json"
    link.symlink_to(target)
    link_binding = {**binding, "path": str(link)}
    with pytest.raises((OSError, ValueError)):
        validate_parent_receipt(
            link_binding,
            root_registry=registry,
            expected_schema="parent_v1",
            expected_stage="parent",
        )
    real_directory = run_root / "real-directory"
    real_directory.mkdir()
    symlink_directory = run_root / "symlink-directory"
    symlink_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        publish_json_under_run_root(
            symlink_directory / "receipt.json",
            payload,
            run_root=run_root,
            root_registry=registry,
        )


def test_runtime_receipt_rejects_noncanonical_order_forged_evidence_and_old_v2():
    registry = load_control_registry()
    controls = passing_controls(registry)
    reordered = [controls[1], controls[0], *controls[2:]]
    receipt = build_protocol_only_runtime_receipt(
        receipt_kind="unit_test",
        status="PASSED",
        controls=reordered,
        bindings=runtime_bindings(),
    )
    with pytest.raises(ValueError, match="canonical registry order"):
        validate_runtime_receipt(receipt, control_registry=registry)

    forged = copy.deepcopy(controls)
    forged[0]["evidence"]["evidence_refs"][0]["sha256"] = "0" * 64
    receipt = build_protocol_only_runtime_receipt(
        receipt_kind="unit_test",
        status="PASSED",
        controls=forged,
        bindings=runtime_bindings(),
    )
    with pytest.raises(ValueError, match="content_sha256"):
        validate_runtime_receipt(
            receipt,
            control_registry=registry,
            evidence_verifiers=passing_evidence_verifiers(registry),
        )

    receipt = build_protocol_only_runtime_receipt(
        receipt_kind="unit_test",
        status="PASSED",
        controls=controls,
        bindings=runtime_bindings(),
    )
    receipt["schema"] = "duca_acquisition_admission_v2"
    receipt = with_content_sha256(receipt)
    with pytest.raises(ValueError, match="permanently rejected"):
        validate_runtime_receipt(receipt, control_registry=registry)


def test_runtime_reference_sets_are_canonical_and_duplicate_free():
    evidence = build_control_evidence(
        control_id="CTRL-TEST",
        claim_state="PASSED",
        evidence_refs=[
            {"kind": "zeta", "sha256": "b" * 64},
            {"kind": "alpha", "sha256": "a" * 64},
        ],
    )
    assert [row["kind"] for row in evidence["evidence_refs"]] == ["alpha", "zeta"]
    with pytest.raises(ValueError, match="unique"):
        build_control_evidence(
            control_id="CTRL-TEST",
            claim_state="PASSED",
            evidence_refs=[
                {"kind": "same", "sha256": "a" * 64},
                {"kind": "same", "sha256": "a" * 64},
            ],
        )

    bindings = build_runtime_bindings(
        artifact_refs=[
            {"kind": "zeta", "sha256": "b" * 64},
            {"kind": "alpha", "sha256": "a" * 64},
        ]
    )
    assert [row["kind"] for row in bindings["artifact_refs"]] == ["alpha", "zeta"]
    with pytest.raises(ValueError, match="unique"):
        build_runtime_bindings(
            artifact_refs=[
                {"kind": "same", "sha256": "a" * 64},
                {"kind": "same", "sha256": "a" * 64},
            ]
        )


@pytest.mark.skipif(
    os.name != "posix", reason="authoritative path policy is POSIX-only"
)
def test_safe_publication_rejects_unregistered_or_preexisting_run_root(tmp_path):
    base = tmp_path / "base"
    base.mkdir(mode=0o700)
    registry = with_content_sha256(
        {
            "schema": RUNTIME_ROOT_REGISTRY_SCHEMA,
            "allowlisted_base_roots": [str(base)],
        }
    )
    preexisting = base / "preexisting"
    preexisting.mkdir(mode=0o700)
    with pytest.raises(FileExistsError):
        create_fresh_run_root(preexisting, root_registry=registry)
    run_root = create_fresh_run_root(base / "run", root_registry=registry)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    payload = with_content_sha256(
        {"schema": "parent_v1", "stage": "parent", "status": "PASSED"}
    )
    with pytest.raises(ValueError, match="allowlisted"):
        publish_json_under_run_root(
            outside / "receipt.json",
            payload,
            run_root=outside,
            root_registry=registry,
        )
    with pytest.raises(FileExistsError):
        create_fresh_run_root(run_root, root_registry=registry)
