from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.bata.duca_cellcf_training import (
    DUCA_P0_CHECKPOINT_METADATA_SCHEMA,
    DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
    DUCA_P0_TRAINING_AUDIT_SCHEMA,
    canonical_sha256,
)
from tools.bata.duca_p0_evaluation import official_evaluator_identity
from tools.bata.summarize_duca_cellcf_convergence import (
    FIXED_EPOCHS,
    VARIANTS,
    _inspect_checkpoint_payload,
    build_convergence_evidence,
)


COMMIT = "a" * 40
SEED = 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hashed(payload: dict, key: str) -> dict:
    output = dict(payload)
    output[key] = canonical_sha256(output)
    return output


def _audit(
    *,
    variant: str,
    epoch: int,
    protocol_sha: str,
    order_sha: str,
    gate_sha: str,
    pilot_sha: str,
    config_sha: str,
    resolved_config_sha: str,
    runtime_config_sha: str,
    annotation_sha: str,
    class_map_sha: str,
    evaluation_config_sha: str,
) -> dict:
    updates = (epoch + 1) * 100
    complete = epoch == 131
    payload = {
        "schema_version": DUCA_P0_TRAINING_AUDIT_SCHEMA,
        "status": "complete" if complete else "in_progress",
        "training_profile": "exposure132",
        "git_commit": COMMIT,
        "variant": variant,
        "seed": SEED,
        "slurm_job_id": str(1000 + list(VARIANTS).index(variant)),
        "protocol_sha256": protocol_sha,
        "ordered_exposure_sha256": order_sha,
        "real_loader_gate_sha256": gate_sha,
        "ddp_pilot_sha256": pilot_sha,
        "source_config_sha256": config_sha,
        "resolved_config_sha256": resolved_config_sha,
        "runtime_config_sha256": runtime_config_sha,
        "evaluation_annotation_sha256": annotation_sha,
        "evaluation_class_map_sha256": class_map_sha,
        "evaluation_config_sha256": evaluation_config_sha,
        "expected_successful_optimizer_updates": 13200,
        "last_completed_epoch": epoch,
        "epochs_completed": epoch + 1,
        "scheduler_last_epoch": updates,
        "selector_schedule_step": updates,
        "epoch_records": [{"epoch": item} for item in range(epoch + 1)],
        "update_audit": {
            "attempted_batches": updates,
            "successful_optimizer_updates": updates,
            "scheduler_updates": updates,
            "ema_updates": updates,
            "duca_schedule_updates": updates,
            "optimizer_attempts": updates,
            "amp_skipped_attempts": 0,
            "replay_exhaustions": 0,
            "forced_amp_overflow_attempts": 0,
        },
    }
    return _hashed(payload, "audit_sha256")


def _fixture_tree(tmp_path: Path):
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    gate = tmp_path / "gate.json"
    pilot = tmp_path / "pilot.json"
    annotation.write_text('{"database": {}}\n', encoding="utf-8")
    class_map.write_text("0 action\n", encoding="utf-8")
    gate.write_text('{"ok": true}\n', encoding="utf-8")
    pilot.write_text('{"ok": true}\n', encoding="utf-8")
    gate_sha = _sha(gate)
    pilot_sha = _sha(pilot)
    shared_protocol = {"fixed_k": 384, "checkpoint_interval": 5}
    protocol_sha = canonical_sha256(shared_protocol)
    order_sha = "c" * 64
    evaluation_config = {
        "type": "mAP",
        "ground_truth_filename": str(annotation.resolve()),
        "subset": "validation",
        "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "top_k": None,
        "blocked_videos": None,
        "thread": 16,
    }
    evaluation_config_sha = canonical_sha256(evaluation_config)
    evaluator = official_evaluator_identity()
    metrics = {
        "average_mAP": 0.5,
        "mAP@0.3": 0.7,
        "mAP@0.4": 0.6,
        "mAP@0.5": 0.5,
        "mAP@0.6": 0.4,
        "mAP@0.7": 0.3,
    }
    post_runs: dict[str, Path] = {}
    variant_receipts: dict[str, Path] = {}
    evaluations: dict[tuple[str, int], Path] = {}
    completed_runs = {}
    variant_records = []

    for variant_index, variant in enumerate(VARIANTS):
        checkpoint_dir = tmp_path / variant / "checkpoint"
        terminal_evaluation = tmp_path / variant / "terminal_evaluation.json"
        post_run_path = tmp_path / variant / "post_run_evidence.json"
        terminal_audit_path = tmp_path / variant / "training_audit.json"
        config_sha = f"{variant_index + 1:064x}"
        resolved_config_sha = f"{variant_index + 11:064x}"
        runtime_config_sha = f"{variant_index + 31:064x}"
        terminal_audit = _audit(
            variant=variant,
            epoch=131,
            protocol_sha=protocol_sha,
            order_sha=order_sha,
            gate_sha=gate_sha,
            pilot_sha=pilot_sha,
            config_sha=config_sha,
            resolved_config_sha=resolved_config_sha,
            runtime_config_sha=runtime_config_sha,
            annotation_sha=_sha(annotation),
            class_map_sha=_sha(class_map),
            evaluation_config_sha=evaluation_config_sha,
        )
        _write_json(terminal_audit_path, terminal_audit)

        post_run = {
            "schema": "duca_cellcf_post_run_evidence_v1",
            "ok": True,
            "variant": variant,
            "git_commit": COMMIT,
            "seed": SEED,
            "training_profile": "exposure132",
            "protocol_sha256": protocol_sha,
            "ordered_exposure_sha256": order_sha,
            "real_loader_gate_sha256": gate_sha,
            "ddp_pilot_sha256": pilot_sha,
            "successful_optimizer_updates": 13200,
            "config_sha256": config_sha,
            "resolved_config_sha256": resolved_config_sha,
            "runtime_config_sha256": runtime_config_sha,
            "evaluation_annotation_sha256": _sha(annotation),
            "evaluation_class_map_sha256": _sha(class_map),
            "evaluation_config_sha256": evaluation_config_sha,
            "checkpoint_epoch": 131,
            "checkpoint_state_key": "state_dict_ema",
            "checkpoint_path": str(
                (checkpoint_dir / "epoch_131.pth").resolve()
            ),
            "training_audit_path": str(terminal_audit_path.resolve()),
            "training_audit_sha256": _sha(terminal_audit_path),
            "terminal_evaluation_path": str(terminal_evaluation.resolve()),
            "evaluator": evaluator,
            "metrics": metrics,
        }

        for epoch in FIXED_EPOCHS:
            checkpoint = checkpoint_dir / f"epoch_{epoch}.pth"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{variant}-{epoch}".encode())
            audit = (
                terminal_audit
                if epoch == 131
                else _audit(
                    variant=variant,
                    epoch=epoch,
                    protocol_sha=protocol_sha,
                    order_sha=order_sha,
                    gate_sha=gate_sha,
                    pilot_sha=pilot_sha,
                    config_sha=config_sha,
                    resolved_config_sha=resolved_config_sha,
                    runtime_config_sha=runtime_config_sha,
                    annotation_sha=_sha(annotation),
                    class_map_sha=_sha(class_map),
                    evaluation_config_sha=evaluation_config_sha,
                )
            )
            metadata = _hashed(
                {
                    "schema_version": DUCA_P0_CHECKPOINT_METADATA_SCHEMA,
                    "training_audit": audit,
                },
                "metadata_sha256",
            )
            sidecar = _hashed(
                {
                    "schema_version": DUCA_P0_CHECKPOINT_SIDECAR_SCHEMA,
                    "checkpoint_sha256": _sha(checkpoint),
                    "experiment_metadata": metadata,
                },
                "sidecar_sha256",
            )
            _write_json(Path(f"{checkpoint}.metadata.json"), sidecar)
            prediction = tmp_path / variant / f"prediction_{epoch}.json"
            prediction.write_text(
                '{"results":{"video":[{}]}}\n',
                encoding="utf-8",
            )
            evaluation_path = (
                terminal_evaluation
                if epoch == 131
                else tmp_path / variant / f"evaluation_{epoch}.json"
            )
            evaluation = _hashed(
                {
                    "schema_version": "duca_cellcf_terminal_evaluation_v1",
                    "git_commit": COMMIT,
                    "config_sha256": config_sha,
                    "resolved_config_sha256": resolved_config_sha,
                    "runtime_config_sha256": f"{epoch + variant_index + 20:064x}",
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": _sha(checkpoint),
                    "checkpoint_epoch": epoch,
                    "checkpoint_state_key": "state_dict_ema",
                    "prediction_path": str(prediction.resolve()),
                    "prediction_sha256": _sha(prediction),
                    "metrics": metrics,
                    "result_count": 1,
                    "video_count": 1,
                    "evaluator": evaluator,
                    "evaluation_config": evaluation_config,
                    "evaluation_annotation_path": str(annotation.resolve()),
                    "evaluation_annotation_sha256": _sha(annotation),
                    "evaluation_class_map_path": str(class_map.resolve()),
                    "evaluation_class_map_sha256": _sha(class_map),
                    "evaluation_config_sha256": evaluation_config_sha,
                },
                "evaluation_sha256",
            )
            _write_json(evaluation_path, evaluation)
            evaluations[(variant, epoch)] = evaluation_path

        terminal_checkpoint = checkpoint_dir / "epoch_131.pth"
        post_run.update(
            checkpoint_sha256=_sha(terminal_checkpoint),
            terminal_evaluation_sha256=_sha(terminal_evaluation),
        )
        post_run["artifact_chain_sha256"] = canonical_sha256(post_run)
        _write_json(post_run_path, post_run)
        post_runs[variant] = post_run_path
        receipt_path = tmp_path / variant / "variant_complete.json"
        artifact_paths = [
            post_run_path,
            *(evaluations[(variant, epoch)] for epoch in FIXED_EPOCHS),
        ]
        receipt = _hashed(
            {
                "schema": "duca_cellcf_convergence_variant_receipt_v1",
                "ok": True,
                "task": "offline_temporal_action_detection",
                "git_commit": COMMIT,
                "training_profile": "exposure132",
                "variant": variant,
                "seed": SEED,
                "evaluation_runtime_config_sha256": {
                    str(epoch): json.loads(
                        evaluations[(variant, epoch)].read_text(
                            encoding="utf-8"
                        )
                    )["runtime_config_sha256"]
                    for epoch in FIXED_EPOCHS
                },
                "artifacts": [
                    {"path": str(path.resolve()), "sha256": _sha(path)}
                    for path in artifact_paths
                ],
            },
            "receipt_sha256",
        )
        _write_json(receipt_path, receipt)
        variant_receipts[variant] = receipt_path
        completed_runs[variant] = {
            "path": str(post_run_path.resolve()),
            "sha256": _sha(post_run_path),
            "metrics": metrics,
            "checkpoint_path": str(terminal_checkpoint.resolve()),
            "checkpoint_sha256": _sha(terminal_checkpoint),
        }
        variant_records.append(
            {
                "name": variant,
                "config_sha256": config_sha,
                "resolved_config_sha256": resolved_config_sha,
            }
        )

    aggregate = {
        "schema": "duca_cellcf_suite_manifest_v1",
        "ok": True,
        "status": "runs_complete_cost_pending",
        "task": "offline_temporal_action_detection",
        "git_commit": COMMIT,
        "git_tree_clean": True,
        "seed": SEED,
        "training_profile": "exposure132",
        "variant_order": list(VARIANTS),
        "ordered_exposure_sha256": order_sha,
        "shared_protocol": shared_protocol,
        "shared_protocol_sha256": protocol_sha,
        "variants": variant_records,
        "real_loader_gate": {"path": str(gate.resolve()), "sha256": gate_sha},
        "ddp_pilot": {"path": str(pilot.resolve()), "sha256": pilot_sha},
        "completed_runs": completed_runs,
    }
    aggregate_path = tmp_path / "aggregate_suite_evidence.json"
    _write_json(aggregate_path, aggregate)
    return (
        aggregate_path,
        _sha(aggregate_path),
        post_runs,
        variant_receipts,
        evaluations,
    )


def _recompute(_prediction, _config):
    return {
        "metrics": {
            "average_mAP": 0.5,
            "mAP@0.3": 0.7,
            "mAP@0.4": 0.6,
            "mAP@0.5": 0.5,
            "mAP@0.6": 0.4,
            "mAP@0.7": 0.3,
        },
        "result_count": 1,
        "video_count": 1,
    }


def _inspect_checkpoint(_path, _metadata, epoch, updates):
    return {
        "payload_reopened": True,
        "epoch": epoch,
        "scheduler_last_epoch": updates,
        "embedded_metadata_exact": True,
    }


def _build(tmp_path: Path):
    (
        aggregate,
        aggregate_sha,
        post_runs,
        variant_receipts,
        evaluations,
    ) = _fixture_tree(tmp_path)
    return build_convergence_evidence(
        expected_commit=COMMIT,
        suite_aggregate_path=aggregate,
        suite_aggregate_sha256=aggregate_sha,
        post_run_paths=post_runs,
        variant_receipt_paths=variant_receipts,
        evaluation_paths=evaluations,
        recompute=_recompute,
        checkpoint_inspector=_inspect_checkpoint,
    )


def test_convergence_evidence_freezes_trajectory_without_checkpoint_selection(
    tmp_path: Path,
) -> None:
    payload = _build(tmp_path)

    assert payload["ok"] is True
    assert payload["fixed_epochs"] == [59, 89, 131]
    assert payload["primary_epoch"] == 131
    assert payload["checkpoint_selection"]["allowed"] is False
    assert payload["suite_aggregate_binding"]["seed"] == SEED
    assert len(payload["rows"]) == 9
    assert all(
        row["official_60_epoch_run"] is False for row in payload["rows"]
    )
    assert all(
        row["checkpoint_payload_contract"]["payload_reopened"] is True
        for row in payload["rows"]
    )


def test_default_checkpoint_inspector_reads_real_selector_schedule_key(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("the audited local Windows torch runtime cannot load c10.dll")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"local torch runtime is unavailable: {exc}")

    metadata = {"bound": True}
    checkpoint = tmp_path / "epoch_59.pth"
    schedule_key = (
        "module.backbone.frame_selector._loss_weight_schedule_step"
    )
    torch.save(
        {
            "epoch": 59,
            "state_dict": {schedule_key: torch.tensor(6000)},
            "state_dict_ema": {schedule_key: torch.tensor(6000)},
            "scheduler": {"last_epoch": 6000},
            "experiment_metadata": metadata,
            "rng_state": {},
        },
        checkpoint,
    )

    payload = _inspect_checkpoint_payload(
        checkpoint,
        metadata,
        59,
        6000,
    )

    assert payload["payload_reopened"] is True
    assert payload["selector_schedule_steps"] == {
        "state_dict": 6000,
        "state_dict_ema": 6000,
    }


def test_convergence_evidence_rejects_metric_based_artifact_tampering(
    tmp_path: Path,
) -> None:
    (
        aggregate,
        aggregate_sha,
        post_runs,
        variant_receipts,
        evaluations,
    ) = _fixture_tree(tmp_path)
    target = evaluations[("cellcf", 89)]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["metrics"]["average_mAP"] = 0.9
    payload.pop("evaluation_sha256")
    payload = _hashed(payload, "evaluation_sha256")
    _write_json(target, payload)

    with pytest.raises(ValueError, match="differs from official recomputation"):
        build_convergence_evidence(
            expected_commit=COMMIT,
            suite_aggregate_path=aggregate,
            suite_aggregate_sha256=aggregate_sha,
            post_run_paths=post_runs,
            variant_receipt_paths=variant_receipts,
            evaluation_paths=evaluations,
            recompute=_recompute,
            checkpoint_inspector=_inspect_checkpoint,
        )


def test_convergence_evidence_rejects_mixed_suite_post_run(
    tmp_path: Path,
) -> None:
    (
        aggregate,
        aggregate_sha,
        post_runs,
        variant_receipts,
        evaluations,
    ) = _fixture_tree(tmp_path)
    replacement = tmp_path / "replacement_post_run.json"
    replacement.write_bytes(post_runs["uniform"].read_bytes())
    post_runs["uniform"] = replacement

    with pytest.raises(ValueError, match="binds another post-run path"):
        build_convergence_evidence(
            expected_commit=COMMIT,
            suite_aggregate_path=aggregate,
            suite_aggregate_sha256=aggregate_sha,
            post_run_paths=post_runs,
            variant_receipt_paths=variant_receipts,
            evaluation_paths=evaluations,
            recompute=_recompute,
            checkpoint_inspector=_inspect_checkpoint,
        )


def test_convergence_evidence_rejects_variant_receipt_runtime_tampering(
    tmp_path: Path,
) -> None:
    (
        aggregate,
        aggregate_sha,
        post_runs,
        variant_receipts,
        evaluations,
    ) = _fixture_tree(tmp_path)
    target = variant_receipts["cellcf"]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["evaluation_runtime_config_sha256"]["89"] = "f" * 64
    payload.pop("receipt_sha256")
    payload = _hashed(payload, "receipt_sha256")
    _write_json(target, payload)

    with pytest.raises(ValueError, match="runtime config hashes mismatch"):
        build_convergence_evidence(
            expected_commit=COMMIT,
            suite_aggregate_path=aggregate,
            suite_aggregate_sha256=aggregate_sha,
            post_run_paths=post_runs,
            variant_receipt_paths=variant_receipts,
            evaluation_paths=evaluations,
            recompute=_recompute,
            checkpoint_inspector=_inspect_checkpoint,
        )


def test_convergence_evidence_rejects_unscoped_missing_profile_compatibility(
    tmp_path: Path,
) -> None:
    (
        aggregate,
        _,
        post_runs,
        variant_receipts,
        evaluations,
    ) = _fixture_tree(tmp_path)
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    payload.pop("training_profile")
    _write_json(aggregate, payload)

    with pytest.raises(ValueError, match="one audited legacy"):
        build_convergence_evidence(
            expected_commit=COMMIT,
            suite_aggregate_path=aggregate,
            suite_aggregate_sha256=_sha(aggregate),
            post_run_paths=post_runs,
            variant_receipt_paths=variant_receipts,
            evaluation_paths=evaluations,
            recompute=_recompute,
            checkpoint_inspector=_inspect_checkpoint,
        )


def test_convergence_evidence_rejects_missing_fixed_point(
    tmp_path: Path,
) -> None:
    (
        aggregate,
        aggregate_sha,
        post_runs,
        variant_receipts,
        evaluations,
    ) = _fixture_tree(tmp_path)
    evaluations.pop(("uniform", 59))

    with pytest.raises(ValueError, match="exactly variants x fixed epochs"):
        build_convergence_evidence(
            expected_commit=COMMIT,
            suite_aggregate_path=aggregate,
            suite_aggregate_sha256=aggregate_sha,
            post_run_paths=post_runs,
            variant_receipt_paths=variant_receipts,
            evaluation_paths=evaluations,
            recompute=_recompute,
            checkpoint_inspector=_inspect_checkpoint,
        )
