from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from tools.bata.finalize_duca_cellcf_postrun_evidence import (
    CANDIDATE_SCHEMA,
    JOB_KEYS,
    SUPPORTED_TRAINED_COMMIT,
    _read_snapshot,
    canonical_sha256,
    finalize_postrun_evidence,
    revalidate_trained_suite_exact,
    sha256_file,
    validate_seal_execution_context,
)


TRAINED_COMMIT = SUPPORTED_TRAINED_COMMIT
EVIDENCE_COMMIT = "e" * 40
AGGREGATE_PAYLOAD = {"kind": "aggregate", "seed": 0}
FINAL_PAYLOAD = {
    "schema": "duca_cellcf_suite_manifest_v1",
    "ok": True,
    "status": "complete",
    "task": "offline_temporal_action_detection",
    "git_commit": TRAINED_COMMIT,
    "seed": 0,
    "cost_evidence_required": True,
    "cost_evidence": {"validated": True, "path": "cost.json"},
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hashed(payload: dict) -> dict:
    result = dict(payload)
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def _fixture(tmp_path: Path):
    run_root = tmp_path / "formal"
    control_root = run_root / "postrun_submission_evidence_v1"
    output_root = control_root / "artifacts"
    trained_root = tmp_path / "trained"
    evidence_root = tmp_path / "evidence"
    for path in (run_root, control_root, output_root, trained_root, evidence_root):
        path.mkdir(parents=True, exist_ok=True)
    formal_completion_job_id = 9006
    formal_ids = {
        key: str(9000 + index)
        for index, key in enumerate(
            ("uniform", "transition_beta0", "cellcf", "aggregate", "cost", "completion"),
            start=1,
        )
    }
    formal_dependencies = {
        "uniform": "none",
        "transition_beta0": "none",
        "cellcf": "none",
        "aggregate": (
            f"afterok:{formal_ids['uniform']}:"
            f"{formal_ids['transition_beta0']}:{formal_ids['cellcf']}"
        ),
        "cost": f"afterok:{formal_ids['aggregate']}",
        "completion": (
            f"afterok:{formal_ids['aggregate']}:{formal_ids['cost']}"
        ),
    }
    formal_rows = []
    for index, key in enumerate(
        ("uniform", "transition_beta0", "cellcf", "aggregate", "cost", "completion"),
        start=1,
    ):
        formal_rows.append(
            {
                "job_key": key,
                "seed": "0",
                "commit": TRAINED_COMMIT,
                "manifest_sha256": "f" * 64,
                "sbatch_file": f"/formal/{key}.sbatch",
                "sbatch_sha256": "b" * 64,
                "job_name": f"formal-{key}",
                "dependency": formal_dependencies[key],
                "job_id": formal_ids[key],
                "job_ref": f"{9000 + index};n16r4",
                "cluster": "n16r4",
                "status": "FORMAL_SUBMITTED",
            }
        )
    formal_ledger = run_root / "jobs.submitted.tsv"
    with formal_ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(formal_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(formal_rows)
    aggregate = run_root / "aggregate_suite_evidence.json"
    final_suite = run_root / "final_suite_evidence.json"
    cost = run_root / "cost.json"
    gate = run_root / "gate.json"
    pilot = run_root / "pilot.json"
    _write_json(aggregate, AGGREGATE_PAYLOAD)
    final_payload = json.loads(json.dumps(FINAL_PAYLOAD))
    final_payload["cost_evidence"]["path"] = str(cost.resolve())
    _write_json(cost, {"ok": True})
    _write_json(gate, {"ok": True, "kind": "real-loader-gate"})
    _write_json(pilot, {"ok": True, "kind": "ddp-pilot"})
    _write_json(final_suite, final_payload)
    for variant in ("uniform", "transition_beta0", "cellcf"):
        _write_json(
            run_root / "logs" / variant / "post_run_evidence.json",
            {"variant": variant},
        )

    job_records = []
    for index, key in enumerate(JOB_KEYS, start=1):
        job_file = control_root / "jobs" / f"{key}.sbatch"
        job_file.parent.mkdir(parents=True, exist_ok=True)
        job_name = f"postrun-{key}"
        job_file.write_text(
            f"#!/bin/bash\n#SBATCH --job-name={job_name}\n",
            encoding="utf-8",
        )
        job_sha = sha256_file(job_file)
        job_records.append(
            {
                "job_key": key,
                "job_name": job_name,
                "dependency_role": "fixture",
                "job_file": str(job_file.resolve()),
                "job_file_sha256": job_sha,
                "submission_token": f"token-{key}-{job_sha[:12]}",
                "job_id": str(1000 + index),
            }
        )
    intent = _hashed(
        {
            "schema": "duca_cellcf_postrun_submission_intent_v1",
            "status": "INTENT_RECORDED",
            "task": "offline_temporal_action_detection",
            "formal_run_root": str(run_root.resolve()),
            "trained_repository": str(trained_root.resolve()),
            "trained_git_commit": TRAINED_COMMIT,
            "evidence_repository": str(evidence_root.resolve()),
            "evidence_git_commit": EVIDENCE_COMMIT,
            "target_cluster": "n16r4",
            "aggregate_suite_evidence_path": str(aggregate.resolve()),
            "aggregate_suite_evidence_sha256": sha256_file(aggregate),
            "final_suite_evidence_path": str(final_suite.resolve()),
            "final_suite_evidence_sha256": sha256_file(final_suite),
            "postrun_output_root": str(output_root.resolve()),
            "jobs": [
                {key: value for key, value in record.items() if key != "job_id"}
                for record in job_records
            ],
        }
    )
    intent_path = control_root / "submission_intent.json"
    _write_json(intent_path, intent)
    intent_sha = sha256_file(intent_path)

    ids = {record["job_key"]: record["job_id"] for record in job_records}
    dependencies = {
        "convergence_uniform": "none",
        "convergence_transition_beta0": "none",
        "convergence_cellcf": "none",
        "convergence_summary": (
            f"afterok:{ids['convergence_uniform']}:"
            f"{ids['convergence_transition_beta0']}:"
            f"{ids['convergence_cellcf']}"
        ),
        "training_cost": "none",
        "completion": (
            f"afterok:{ids['convergence_summary']}:{ids['training_cost']}"
        ),
    }
    rows = []
    for record in job_records:
        key = record["job_key"]
        base = {
            "schema": "duca_cellcf_postrun_slurm_receipt_v1",
            "task": "offline_temporal_action_detection",
            "job_key": key,
            "job_id": int(record["job_id"]),
            "job_name": record["job_name"],
            "cluster": "n16r4",
            "dependency": (
                None if dependencies[key] == "none" else dependencies[key]
            ),
            "submission_token": record["submission_token"],
            "job_file": record["job_file"],
            "job_file_sha256": record["job_file_sha256"],
            "raw_sbatch_response": f"{record['job_id']};n16r4",
            "trained_git_commit": TRAINED_COMMIT,
            "evidence_git_commit": EVIDENCE_COMMIT,
            "aggregate_suite_evidence_sha256": sha256_file(aggregate),
            "submission_intent": str(intent_path.resolve()),
            "submission_intent_sha256": intent_sha,
        }
        submitted_path = control_root / "receipts" / f"{key}.submitted.json"
        submitted = _hashed(
            {
                **base,
                "status": "SUBMITTED_UNVERIFIED",
                "scheduler_validation": None,
                "submitted_receipt": None,
                "submitted_receipt_sha256": None,
            }
        )
        _write_json(submitted_path, submitted)
        submitted_sha = sha256_file(submitted_path)
        verified_path = control_root / "receipts" / f"{key}.verified.json"
        verified = _hashed(
            {
                **base,
                "status": "VERIFIED",
                "scheduler_validation": {
                    "ok": True,
                    "job_id": int(record["job_id"]),
                    "scheduler_script_verified": True,
                    "submission_command_held_verified": True,
                    "current_user_hold_verified": True,
                },
                "submitted_receipt": str(submitted_path.resolve()),
                "submitted_receipt_sha256": submitted_sha,
            }
        )
        _write_json(verified_path, verified)
        rows.append(
            {
                "job_key": key,
                "job_id": record["job_id"],
                "job_name": record["job_name"],
                "cluster": "n16r4",
                "dependency": dependencies[key],
                "submission_token": record["submission_token"],
                "job_file": record["job_file"],
                "job_file_sha256": record["job_file_sha256"],
                "submitted_receipt": str(submitted_path.resolve()),
                "submitted_receipt_sha256": submitted_sha,
                "verified_receipt": str(verified_path.resolve()),
                "verified_receipt_sha256": sha256_file(verified_path),
                "trained_commit": TRAINED_COMMIT,
                "evidence_commit": EVIDENCE_COMMIT,
                "aggregate_sha256": sha256_file(aggregate),
                "submission_intent_sha256": intent_sha,
            }
        )
    ledger = control_root / "jobs.submitted.tsv"
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = _hashed(
        {
            "schema": "duca_cellcf_postrun_submission_manifest_v1",
            "ok": True,
            "task": "offline_temporal_action_detection",
            "training_profile": "exposure132",
            "formal_run_root": str(run_root.resolve()),
            "trained_repository": str(trained_root.resolve()),
            "trained_git_commit": TRAINED_COMMIT,
            "evidence_repository": str(evidence_root.resolve()),
            "evidence_git_commit": EVIDENCE_COMMIT,
            "target_cluster": "n16r4",
            "aggregate_suite_evidence_path": str(aggregate.resolve()),
            "aggregate_suite_evidence_sha256": sha256_file(aggregate),
            "final_suite_evidence_path": str(final_suite.resolve()),
            "final_suite_evidence_sha256": sha256_file(final_suite),
            "postrun_output_root": str(output_root.resolve()),
            "submission_intent_path": str(intent_path.resolve()),
            "submission_intent_sha256": intent_sha,
            "formal_completion_job_id": formal_completion_job_id,
            "jobs_ledger_path": str(ledger.resolve()),
            "jobs_ledger_sha256": sha256_file(ledger),
            "jobs": rows,
        }
    )
    _write_json(control_root / "submission_manifest.json", manifest)
    convergence = _hashed(
        {
            "schema": "duca_cellcf_fixed_convergence_trajectory_v1",
            "ok": True,
            "task": "offline_temporal_action_detection",
            "git_commit": TRAINED_COMMIT,
            "evidence_git_commit": EVIDENCE_COMMIT,
            "variants": ["uniform", "transition_beta0", "cellcf"],
            "fixed_epochs": [59, 89, 131],
            "primary_epoch": 131,
            "primary_state_key": "state_dict_ema",
            "rows": [{"variant": "uniform", "average_mAP": 0.5}],
        }
    )
    training_cost = _hashed(
        {
            "schema": "duca_cellcf_training_cost_summary_v1",
            "ok": True,
            "task": "offline_temporal_action_detection",
            "git_commit": TRAINED_COMMIT,
            "evidence_git_commit": EVIDENCE_COMMIT,
            "training_profile": "exposure132",
            "rows": [],
        }
    )
    convergence_path = output_root / "convergence" / "fixed_trajectory.json"
    training_cost_path = output_root / "training_cost" / "training_cost_summary.json"
    _write_json(convergence_path, convergence)
    _write_json(training_cost_path, training_cost)
    for variant in ("uniform", "transition_beta0", "cellcf"):
        _write_json(
            output_root / "convergence" / variant / "variant_complete.json",
            {"ok": True, "variant": variant},
        )
        for epoch in (59, 89):
            _write_json(
                output_root
                / "convergence"
                / variant
                / f"epoch_{epoch}"
                / "evaluation.json",
                {"ok": True, "variant": variant, "epoch": epoch},
            )
        _write_json(
            run_root / "logs" / variant / "terminal_evaluation.json",
            {"ok": True, "variant": variant, "epoch": 131},
        )
        _write_json(
            output_root
            / "training_cost"
            / f"{variant}.slurm_cost.json",
            {"ok": True, "variant": variant},
        )
    kwargs = {
        "run_root": run_root,
        "control_root": control_root,
        "trained_repo_root": trained_root,
        "trained_commit": TRAINED_COMMIT,
        "evidence_repo_root": evidence_root,
        "evidence_commit": EVIDENCE_COMMIT,
        "aggregate_path": aggregate,
        "aggregate_sha256": sha256_file(aggregate),
        "final_suite_path": final_suite,
        "final_suite_sha256": sha256_file(final_suite),
        "aggregate_loader": lambda **_kwargs: {
            "seed": 0,
            "real_loader_gate": {"path": str(gate.resolve())},
            "ddp_pilot": {"path": str(pilot.resolve())},
        },
        "final_suite_revalidator": lambda **_kwargs: final_payload,
        "convergence_rebuilder": lambda **_kwargs: convergence,
        "training_cost_rebuilder": lambda **_kwargs: training_cost,
        "scheduler_validator": lambda **kwargs: {
            "ok": True,
            "job_id": kwargs["job_id"],
            "state": "COMPLETED",
            "scheduler_script_verified": True,
            "submission_command_held_verified": True,
        },
        "formal_completion_validator": lambda **kwargs: {
            "ok": True,
            "job_id": kwargs["job_id"],
            "job_name": kwargs["job_name"],
            "cluster": kwargs["cluster"],
            "state": "COMPLETED",
            "exit_code": "0:0",
        },
        "postrun_terminal_validator": lambda **kwargs: {
            "ok": True,
            "job_id": kwargs["job_id"],
            "job_name": kwargs["job_name"],
            "cluster": kwargs["cluster"],
            "state": "COMPLETED",
            "exit_code": "0:0",
        },
        "repository_validator": lambda *_args: None,
    }
    candidate_kwargs = dict(kwargs)
    candidate_kwargs["require_postrun_completed"] = False
    candidate_kwargs["scheduler_validator"] = lambda **values: {
        "ok": True,
        "job_id": values["job_id"],
        "state": (
            "RUNNING"
            if values["job_name"] == "postrun-completion"
            else "COMPLETED"
        ),
        "scheduler_script_verified": True,
        "submission_command_held_verified": True,
    }
    candidate = finalize_postrun_evidence(**candidate_kwargs)
    _write_json(
        control_root / "postrun_evidence_candidate.json",
        candidate,
    )
    return kwargs, convergence_path, convergence


def test_postrun_finalizer_reopens_full_chain(tmp_path: Path) -> None:
    kwargs, _, _ = _fixture(tmp_path)

    payload = finalize_postrun_evidence(**kwargs)

    assert payload["ok"] is True
    assert payload["trained_git_commit"] == TRAINED_COMMIT
    assert len(payload["receipts"]) == 6
    assert len(payload["scheduler_revalidation"]) == 6
    assert len(payload["postrun_terminal_revalidation"]) == 6
    assert all(
        receipt["submission_time_scheduler_script_verified"] is True
        for receipt in payload["receipts"]
    )
    assert payload["candidate_evidence"] is not None
    assert payload["formal_completion_scheduler_revalidation"]["job_id"] == 9006


def test_postrun_candidate_cannot_claim_completion(tmp_path: Path) -> None:
    kwargs, _, _ = _fixture(tmp_path)
    kwargs["require_postrun_completed"] = False
    kwargs["scheduler_validator"] = lambda **kwargs: {
        "ok": True,
        "job_id": kwargs["job_id"],
        "state": (
            "RUNNING"
            if kwargs["job_name"] == "postrun-completion"
            else "COMPLETED"
        ),
        "scheduler_script_verified": True,
        "submission_command_held_verified": True,
    }

    payload = finalize_postrun_evidence(**kwargs)

    assert payload["schema"] == CANDIDATE_SCHEMA
    assert payload["ok"] is False
    assert payload["status"] == "pending_external_seal"
    assert payload["requires_external_seal"] is True
    assert payload["postrun_terminal_revalidation"] == []


def test_postrun_finalizer_rejects_rehashed_metric_tampering(
    tmp_path: Path,
) -> None:
    kwargs, convergence_path, original = _fixture(tmp_path)
    tampered = json.loads(json.dumps(original))
    tampered["rows"][0]["average_mAP"] = 0.99
    tampered.pop("artifact_sha256")
    tampered = _hashed(tampered)
    _write_json(convergence_path, tampered)

    with pytest.raises(ValueError, match="not reproducible"):
        finalize_postrun_evidence(**kwargs)


def test_postrun_finalizer_rejects_partial_submission(
    tmp_path: Path,
) -> None:
    kwargs, _, _ = _fixture(tmp_path)
    control_root = Path(kwargs["control_root"])
    (control_root / "receipts" / "convergence_uniform.verified.json").unlink()

    with pytest.raises(ValueError, match="VERIFIED receipt is missing"):
        finalize_postrun_evidence(**kwargs)


def test_postrun_finalizer_rejects_postrun_id_as_formal_completion(
    tmp_path: Path,
) -> None:
    kwargs, _, _ = _fixture(tmp_path)
    control_root = Path(kwargs["control_root"])
    manifest_path = control_root / "submission_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_completion_job_id"] = 1006
    manifest.pop("artifact_sha256")
    _write_json(manifest_path, _hashed(manifest))

    with pytest.raises(ValueError, match="formal_completion_job_id"):
        finalize_postrun_evidence(**kwargs)


def test_postrun_finalizer_rejects_candidate_that_self_claims_completion(
    tmp_path: Path,
) -> None:
    kwargs, _, _ = _fixture(tmp_path)
    candidate_path = (
        Path(kwargs["control_root"]) / "postrun_evidence_candidate.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["postrun_completion_job"]["state_at_evidence_write"] = "COMPLETED"
    candidate.pop("artifact_sha256")
    _write_json(candidate_path, _hashed(candidate))

    with pytest.raises(ValueError, match="active completion job"):
        finalize_postrun_evidence(**kwargs)


def test_postrun_finalizer_uses_durable_submission_script_proof(
    tmp_path: Path,
) -> None:
    kwargs, _, _ = _fixture(tmp_path)
    scheduler_calls = []

    def scheduler_validator(**values):
        scheduler_calls.append(values)
        return {
            "ok": True,
            "job_id": values["job_id"],
            "state": "COMPLETED",
            "submission_command_held_verified": True,
        }

    kwargs["scheduler_validator"] = scheduler_validator
    payload = finalize_postrun_evidence(**kwargs)

    assert payload["ok"] is True
    assert len(scheduler_calls) == 6
    assert all(
        call["require_scheduler_script"] is False
        and call["submitted_with_hold"] is True
        and call["require_current_user_hold"] is False
        for call in scheduler_calls
    )
    assert all(
        receipt["submission_time_scheduler_script_verified"] is True
        for receipt in payload["receipts"]
    )


def test_historical_revalidation_executes_the_frozen_repository_module(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "frozen-repo"
    module_root = repo / "tools" / "bata"
    module_root.mkdir(parents=True)
    (repo / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (module_root / "__init__.py").write_text("", encoding="utf-8")
    (module_root / "validate_duca_cellcf_suite.py").write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--expected-commit', required=True)",
                "parser.add_argument('--seed', type=int, required=True)",
                "parser.add_argument('--output-json', required=True)",
                "args, _ = parser.parse_known_args()",
                "payload = {",
                "    'source': 'frozen-repository-validator',",
                "    'git_commit': args.expected_commit,",
                "    'seed': args.seed,",
                "}",
                "Path(args.output_json).write_text(",
                "    json.dumps(payload), encoding='utf-8'",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = revalidate_trained_suite_exact(
        repo_root=repo,
        seed=0,
        expected_commit=TRAINED_COMMIT,
        require_clean=True,
        gate_json=tmp_path / "gate.json",
        pilot_json=tmp_path / "pilot.json",
        post_run_evidence={
            "uniform": tmp_path / "uniform.json",
            "transition_beta0": tmp_path / "transition.json",
            "cellcf": tmp_path / "cellcf.json",
        },
        cost_evidence=tmp_path / "cost.json",
        require_cost_evidence=True,
    )

    assert payload == {
        "source": "frozen-repository-validator",
        "git_commit": TRAINED_COMMIT,
        "seed": 0,
    }


def test_postrun_finalizer_rejects_evidence_drift_during_rebuild(
    tmp_path: Path,
) -> None:
    kwargs, convergence_path, convergence = _fixture(tmp_path)

    def rebuild_with_drift(**_kwargs):
        convergence_path.write_text(
            convergence_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return convergence

    kwargs["convergence_rebuilder"] = rebuild_with_drift
    with pytest.raises(ValueError, match="changed after snapshot"):
        finalize_postrun_evidence(**kwargs)


def test_postrun_finalizer_rejects_transient_swap_then_restore(
    tmp_path: Path,
) -> None:
    kwargs, convergence_path, convergence = _fixture(tmp_path)
    original = convergence_path.read_bytes()

    def rebuild_with_transient_swap(**_kwargs):
        original_path = convergence_path.with_name("original.json")
        os.replace(convergence_path, original_path)
        convergence_path.write_text('{"malicious":true}\n', encoding="utf-8")
        assert convergence_path.read_bytes() != original
        convergence_path.unlink()
        os.replace(original_path, convergence_path)
        assert convergence_path.read_bytes() == original
        return convergence

    kwargs["convergence_rebuilder"] = rebuild_with_transient_swap
    with pytest.raises(
        ValueError,
        match="directory changed after snapshot",
    ):
        finalize_postrun_evidence(**kwargs)


def test_postrun_finalizer_snapshots_transitive_training_audit(
    tmp_path: Path,
) -> None:
    kwargs, _, _ = _fixture(tmp_path)
    run_root = Path(kwargs["run_root"])
    audit_path = tmp_path / "work_dir" / "gpu1_id0" / "training_audit.json"
    _write_json(audit_path, {"status": "complete", "trusted": True})
    post_run_path = (
        run_root / "logs" / "uniform" / "post_run_evidence.json"
    )
    post_run = json.loads(post_run_path.read_text(encoding="utf-8"))
    post_run["training_audit_path"] = str(audit_path.resolve())
    post_run["training_audit_sha256"] = sha256_file(audit_path)
    _write_json(post_run_path, post_run)
    original_loader = kwargs["aggregate_loader"]

    def loader_with_transitive_swap(**loader_kwargs):
        original_path = audit_path.with_name("original_audit.json")
        os.replace(audit_path, original_path)
        _write_json(audit_path, {"status": "complete", "trusted": False})
        assert json.loads(audit_path.read_text(encoding="utf-8"))["trusted"] is False
        audit_path.unlink()
        os.replace(original_path, audit_path)
        return original_loader(**loader_kwargs)

    kwargs["aggregate_loader"] = loader_with_transitive_swap
    with pytest.raises(
        ValueError,
        match="evidence directory changed after snapshot",
    ):
        finalize_postrun_evidence(**kwargs)


def test_candidate_and_final_seal_require_distinct_execution_contexts() -> None:
    assert (
        validate_seal_execution_context(
            candidate=True,
            slurm_job_id="12345",
        )
        == 12345
    )
    assert (
        validate_seal_execution_context(
            candidate=False,
            slurm_job_id=None,
        )
        is None
    )
    with pytest.raises(ValueError, match="must be written by its Slurm"):
        validate_seal_execution_context(candidate=True, slurm_job_id=None)
    with pytest.raises(ValueError, match="outside every Slurm"):
        validate_seal_execution_context(
            candidate=False,
            slurm_job_id="12345",
        )


def test_snapshot_reader_rejects_leaf_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"ok":true}\n', encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="must not be a symlink"):
        _read_snapshot(link, "test evidence")
