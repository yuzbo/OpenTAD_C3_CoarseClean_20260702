from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.bata.aggregate_duca_r5_paper_matrix as aggregate_module
from tools.bata.aggregate_duca_r5_paper_matrix import aggregate_matrix
from tools.bata.profile_duca_full_stack_cost import (
    load_r5_terminal_cost_binding,
)
from tools.bata.duca_trained_checkpoint_binding import (
    build_trained_checkpoint_binding,
    load_trained_checkpoint_binding,
    write_trained_checkpoint_binding,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sealed(payload: dict, key: str) -> dict:
    result = dict(payload)
    result[key] = aggregate_module._canonical_sha256(result)
    return result


def _build_matrix(tmp_path: Path) -> tuple[Path, str, list[dict]]:
    commit = "a" * 40
    dense_commit = "b" * 40
    pretrain = tmp_path / "pretrain.pth"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    frontend = tmp_path / "frontend.pth"
    for path, content in (
        (pretrain, b"pretrain"),
        (annotation, b"{}\n"),
        (class_map, b"action\n"),
        (frontend, b"frontend"),
    ):
        path.write_bytes(content)
    alignment = tmp_path / "alignment.json"
    _write_json(alignment, _sealed({"schema": "alignment-test"}, "alignment_sha256"))

    gate = tmp_path / "temporalmaxer_one_step.json"
    _write_json(
        gate,
        {
            "ok": True,
            "task": "offline_temporal_action_detection",
            "git_commit": commit,
            "forward_backward_optimizer_step_completed": True,
            "pretrain_path": str(pretrain.resolve()),
            "pretrain_sha256": _sha(pretrain),
        },
    )
    gate_sha_file = tmp_path / "temporalmaxer_one_step.json.sha256"
    gate_sha_file.write_text(_sha(gate) + "\n", encoding="utf-8")

    dense_config = tmp_path / "dense" / "dense_adatad.py"
    dense_checkpoint = tmp_path / "dense" / "checkpoint" / "epoch_59.pth"
    dense_training = tmp_path / "dense" / "training.json"
    dense_evaluation = tmp_path / "dense" / "evaluation.json"
    dense_config.parent.mkdir(parents=True, exist_ok=True)
    dense_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    dense_config.write_text("model = dict(type='ActionFormer')\n", encoding="utf-8")
    dense_checkpoint.write_bytes(b"dense-trained-ema")
    _write_json(dense_training, {"ok": True, "epoch": 59})
    _write_json(dense_evaluation, {"ok": True, "average_mAP": 0.6829})
    dense_resolved_sha = "f" * 64
    dense_evidence = write_trained_checkpoint_binding(
        tmp_path / "dense" / "checkpoint_binding.json",
        build_trained_checkpoint_binding(
            role="dense_adatad_baseline",
            git_commit=dense_commit,
            config_path=dense_config,
            resolved_config_sha256=dense_resolved_sha,
            checkpoint_path=dense_checkpoint,
            checkpoint_epoch=59,
            checkpoint_state_key="state_dict_ema",
            training_evidence_path=dense_training,
            evaluation_evidence_path=dense_evaluation,
        ),
    )
    dense_binding = load_trained_checkpoint_binding(
        dense_evidence,
        _sha(dense_evidence),
        expected_role="dense_adatad_baseline",
        expected_commit=dense_commit,
        expected_config_path=dense_config,
        expected_config_sha256=_sha(dense_config),
        expected_resolved_config_sha256=dense_resolved_sha,
        expected_checkpoint_path=dense_checkpoint,
    )
    dense = {
        "config": str(dense_config.resolve()),
        "config_sha256": _sha(dense_config),
        "checkpoint": str(dense_checkpoint.resolve()),
        "checkpoint_sha256": _sha(dense_checkpoint),
        "checkpoint_evidence": str(dense_evidence.resolve()),
        "checkpoint_evidence_sha256": _sha(dense_evidence),
        "trained_commit": dense_commit,
    }

    cells = []
    costs = []
    for backend in ("actionformer", "temporalmaxer"):
        for arm in ("uniform", "learned"):
            for budget in (384, 256):
                for seed in (3407, 5801, 8123):
                    cell_id = f"{backend}_{arm}_k{budget}_s{seed}"
                    config = tmp_path / "configs" / f"{cell_id}.py"
                    config.parent.mkdir(parents=True, exist_ok=True)
                    config.write_text(f"r5_cell = {cell_id!r}\n", encoding="utf-8")
                    cells.append(
                        {
                            "id": cell_id,
                            "backend": backend,
                            "arm": arm,
                            "budget": budget,
                            "max_unselected_hole": 2 if budget == 384 else 3,
                            "seed": seed,
                            "config": str(config.resolve()),
                            "config_sha256": _sha(config),
                        }
                    )
                    if seed == 3407:
                        costs.append(
                            {
                                "id": f"cost_{cell_id}",
                                "kind": "r5_cell",
                                "source_cell": cell_id,
                                "summary": str(
                                    (tmp_path / "cost" / f"{cell_id}.summary.json").resolve()
                                ),
                            }
                        )
    costs.append(
        {
            "id": "cost_dense_adatad_k768",
            "kind": "dense_baseline",
            "source_cell": "temporalmaxer_one_step",
            "summary": str(
                (tmp_path / "cost" / "dense_adatad_k768.summary.json").resolve()
            ),
            **dense,
        }
    )
    summary_path = tmp_path / "matrix_summary.json"
    summary_sha_file = tmp_path / "matrix_summary.json.sha256"
    summary = {
        "schema": "duca_r5_paper_matrix_v1",
        "task": "offline_temporal_action_detection",
        "git_commit": commit,
        "cells": cells,
        "costs": costs,
        "dense_cost_baseline": dense,
        "matrix_summary_sha256_file": str(summary_sha_file.resolve()),
        "mechanism_gate_output": str(gate.resolve()),
        "mechanism_gate_sha256_file": str(gate_sha_file.resolve()),
    }
    _write_json(summary_path, summary)
    summary_sha_file.write_text(_sha(summary_path) + "\n", encoding="utf-8")

    for cell in cells:
        cell_id = cell["id"]
        checkpoint = (
            tmp_path / "runs" / cell_id / "gpu1_id0/checkpoint/epoch_59.pth"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(cell_id.encode("ascii"))
        resolved_sha = hashlib.sha256(f"resolved:{cell_id}".encode()).hexdigest()
        runtime_sha = hashlib.sha256(f"runtime:{cell_id}".encode()).hexdigest()
        update_audit = {
            "attempted_batches": 6000,
            "successful_optimizer_updates": 6000,
            "optimizer_attempts": 6000,
            "amp_skipped_attempts": 0,
            "scheduler_updates": 6000,
            "ema_updates": 6000,
            "duca_schedule_updates": 6000,
            "replay_exhaustions": 0,
            "forced_amp_overflow_attempts": 0,
            "max_amp_retries_observed": 0,
        }
        records = []
        for epoch in range(60):
            records.append(
                {
                    "epoch": epoch,
                    "counter_delta": {
                        **{key: 100 for key in (
                            "attempted_batches",
                            "successful_optimizer_updates",
                            "optimizer_attempts",
                            "scheduler_updates",
                            "ema_updates",
                            "duca_schedule_updates",
                        )},
                        "amp_skipped_attempts": 0,
                        "replay_exhaustions": 0,
                        "forced_amp_overflow_attempts": 0,
                        "max_amp_retries_observed": 0,
                    },
                    "scheduler_last_epoch": (epoch + 1) * 100,
                    "selector_schedule_step": (epoch + 1) * 100,
                }
            )
        r5_cell = {
            key: cell[key]
            for key in (
                "backend",
                "arm",
                "budget",
                "max_unselected_hole",
                "seed",
            )
        }
        r5_cell.update(
            {
                "max_selected_interval_source_frames": (
                    int(cell["max_unselected_hole"]) + 1
                )
                * 4,
                "sampling_regime": "boundary_burst_with_global_coverage",
            }
        )
        audit = {
            "schema_version": "duca_p0_training_audit_v2",
            "status": "complete",
            "git_commit": commit,
            "variant": cell_id,
            "seed": cell["seed"],
            "formal_protocol": "duca_r5_mechanism_matrix_v1",
            "training_profile": "official60",
            "checkpoint_criterion": "terminal_epoch_59_state_dict_ema",
            "primary_checkpoint_epoch": 59,
            "primary_checkpoint_state_key": "state_dict_ema",
            "expected_train_batches_per_epoch": 100,
            "expected_successful_optimizer_updates": 6000,
            "last_completed_epoch": 59,
            "epochs_completed": 60,
            "train_batches_per_epoch": 100,
            "scheduler_last_epoch": 6000,
            "selector_schedule_step": 6000,
            "r5_cell": r5_cell,
            "source_config_path": cell["config"],
            "source_config_sha256": cell["config_sha256"],
            "resolved_config_sha256": resolved_sha,
            "runtime_config_sha256": runtime_sha,
            "matrix_summary_path": str(summary_path.resolve()),
            "matrix_summary_sha256": _sha(summary_path),
            "mechanism_gate_path": str(gate.resolve()),
            "mechanism_gate_sha256": _sha(gate),
            "pretrain_path": str(pretrain.resolve()),
            "pretrain_sha256": _sha(pretrain),
            "evaluation_annotation_path": str(annotation.resolve()),
            "evaluation_annotation_sha256": _sha(annotation),
            "evaluation_class_map_path": str(class_map.resolve()),
            "evaluation_class_map_sha256": _sha(class_map),
            "evaluation_config_sha256": "e" * 64,
            "update_audit": update_audit,
            "epoch_records": records,
        }
        if cell["arm"] == "learned":
            audit["selector_initialization_contract"] = {
                "checkpoint_path": str(frontend.resolve()),
                "checkpoint_sha256": _sha(frontend),
                "checkpoint_epoch": 4,
                "checkpoint_state_key": "state_dict_ema",
                "selected_p0_variant": "boundary_burst_r2q3",
                "learned_variant": "boundary_burst_r2q3_g1",
            }
            audit["hard_swap_alignment"] = {
                "path": str(alignment.resolve()),
                "sha256": _sha(alignment),
                "self_sha256": json.loads(alignment.read_text())["alignment_sha256"],
                "context_sha256": "c" * 64,
                "terminal_suite_sha256": "d" * 64,
            }
        audit = _sealed(audit, "audit_sha256")
        audit_path = checkpoint.parent.parent / "duca_selected_axis_training_audit.json"
        _write_json(audit_path, audit)
        metadata = _sealed(
            {
                "schema_version": "duca_p0_checkpoint_metadata_v2",
                "training_audit": audit,
            },
            "metadata_sha256",
        )
        sidecar = _sealed(
            {
                "schema_version": "duca_p0_checkpoint_sidecar_v2",
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
                "experiment_metadata": metadata,
            },
            "sidecar_sha256",
        )
        sidecar_path = Path(f"{checkpoint}.metadata.json")
        _write_json(sidecar_path, sidecar)
        binding = load_r5_terminal_cost_binding(
            method_name=cell_id,
            config_path=cell["config"],
            checkpoint_path=checkpoint,
            expected_commit=commit,
            matrix_summary_path=summary_path,
            matrix_summary_sha256=_sha(summary_path),
            mechanism_gate_path=gate,
            mechanism_gate_sha256=_sha(gate),
            expected_resolved_config_sha256=resolved_sha,
        )
        metrics = {
            "average_mAP": 0.65 + (0.01 if cell["arm"] == "learned" else 0.0),
            **{
                f"mAP@{iou:.1f}": 0.70 + 0.01 * index
                for index, iou in enumerate((0.3, 0.4, 0.5, 0.6, 0.7))
            },
        }
        identity = {
            "variant": cell_id,
            "seed": cell["seed"],
            "successful_optimizer_updates": 6000,
            "checkpoint_sidecar_path": binding["checkpoint_sidecar_path"],
            "checkpoint_sidecar_sha256": binding["checkpoint_sidecar_sha256"],
            "training_audit_path": binding["training_audit_path"],
            "training_audit_sha256": binding["training_audit_sha256"],
            "training_audit_self_sha256": binding["training_audit_self_sha256"],
            "matrix_summary_sha256": binding["matrix_summary_sha256"],
            "mechanism_gate_sha256": binding["mechanism_gate_sha256"],
            "pretrain_path": binding["pretrain"]["path"],
            "pretrain_sha256": binding["pretrain"]["sha256"],
            "frontend_initialization": binding["frontend_initialization"],
        }
        evaluation = {
            "schema_version": "duca_r5_terminal_evaluation_v1",
            "git_commit": commit,
            "task": "offline_temporal_action_detection",
            "config_path": cell["config"],
            "config_sha256": cell["config_sha256"],
            "resolved_config_sha256": resolved_sha,
            "runtime_config_sha256": runtime_sha,
            "evaluation_annotation_path": str(annotation.resolve()),
            "evaluation_annotation_sha256": _sha(annotation),
            "evaluation_class_map_path": str(class_map.resolve()),
            "evaluation_class_map_sha256": _sha(class_map),
            "evaluation_config_sha256": "e" * 64,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha(checkpoint),
            "checkpoint_epoch": 59,
            "checkpoint_state_key": "state_dict_ema",
            "result_count": 10,
            "video_count": 5,
            "metrics": metrics,
            "r5_cell": r5_cell,
            "training_identity": identity,
        }
        evaluation = _sealed(evaluation, "evaluation_sha256")
        _write_json(tmp_path / "results" / f"{cell_id}.terminal_evaluation.json", evaluation)

        if cell["seed"] == 3407:
            cost_path = tmp_path / "cost" / f"{cell_id}.summary.json"
            cost = {
                "method": cell_id,
                "config_commit": commit,
                "random_init": False,
                "uses_ema": True,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
                "checkpoint_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "r5_cost_binding": binding,
                "r5_cost_binding_sha256": aggregate_module._canonical_sha256(binding),
                "stages": {"end_to_end_serial_ms": {"p50": 100.0}},
                "selected_count": {"p50": float(cell["budget"])},
                "resources": {"peak_gpu_memory_mb": {"p50": 1024.0}},
            }
            _write_json(cost_path, cost)
    dense_cost = {
        "method": "dense-adatad",
        "config_commit": dense_commit,
        "trained_commit": dense_commit,
        "evidence_git_commit": commit,
        "random_init": False,
        "uses_ema": True,
        "inference_code_tree_binding": {
            "profile_model_loaded_from_trained_repository": True,
            "profile_configs_loaded_from_trained_repository": True,
            "execution_repository": str((tmp_path / "dense").resolve()),
            "loaded_opentad_root": str(
                (tmp_path / "dense" / "opentad").resolve()
            ),
        },
        "profile_config_git_binding": {
            "trained_repository": str((tmp_path / "dense").resolve()),
            "trained_commit": dense_commit,
        },
        "config_path": str(dense_config.resolve()),
        "profile_config_sha256": _sha(dense_config),
        "profile_resolved_config_sha256": dense_resolved_sha,
        "checkpoint_path": str(dense_checkpoint.resolve()),
        "checkpoint_sha256": _sha(dense_checkpoint),
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "trained_checkpoint_binding": dense_binding,
        "trained_checkpoint_binding_sha256": aggregate_module._canonical_sha256(
            dense_binding
        ),
        "stages": {"end_to_end_serial_ms": {"p50": 200.0}},
        "selected_count": {"p50": 768.0},
        "resources": {"peak_gpu_memory_mb": {"p50": 2048.0}},
    }
    _write_json(
        tmp_path / "cost" / "dense_adatad_k768.summary.json", dense_cost
    )
    return summary_path, commit, cells


def test_r5_aggregate_reopens_training_chain_and_binds_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path, commit, _ = _build_matrix(tmp_path)
    monkeypatch.setattr(
        aggregate_module, "validate_and_rebuild_profile_summary", lambda payload: {}
    )

    result = aggregate_matrix(matrix_summary=summary_path, expected_commit=commit)

    assert result["ok"] is True
    assert result["cell_count"] == 24
    assert result["cost_count"] == 9
    assert len(result["three_seed_aggregates"]) == 8
    assert len(result["paired_deltas"]) == 12
    assert all(row["training_binding"]["epoch_record_count"] == 60 for row in result["rows"])
    assert all(row["source_checkpoint_sha256"] for row in result["costs"])
    assert result["dense_baseline_cost"]["selected_count_p50"] == 768.0
    assert len(result["cost_comparisons"]) == 8
    assert all(row["speedup_vs_dense"] == 2.0 for row in result["cost_comparisons"])


def test_r5_aggregate_rejects_persisted_audit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path, commit, cells = _build_matrix(tmp_path)
    monkeypatch.setattr(
        aggregate_module, "validate_and_rebuild_profile_summary", lambda payload: {}
    )
    cell_id = cells[0]["id"]
    audit_path = (
        tmp_path / "runs" / cell_id / "gpu1_id0/duca_selected_axis_training_audit.json"
    )
    audit = json.loads(audit_path.read_text())
    audit["scheduler_last_epoch"] = 5999
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="persisted and checkpoint-embedded"):
        aggregate_matrix(matrix_summary=summary_path, expected_commit=commit)


def test_r5_aggregate_rejects_cost_bound_to_another_source_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path, commit, _ = _build_matrix(tmp_path)
    monkeypatch.setattr(
        aggregate_module, "validate_and_rebuild_profile_summary", lambda payload: {}
    )
    cost_path = tmp_path / "cost/actionformer_uniform_k384_s3407.summary.json"
    cost = json.loads(cost_path.read_text())
    cost["r5_cost_binding"]["method"] = "temporalmaxer_uniform_k384_s3407"
    cost["r5_cost_binding"].pop("binding_sha256", None)
    cost["r5_cost_binding"]["binding_sha256"] = aggregate_module._canonical_sha256(
        cost["r5_cost_binding"]
    )
    cost["r5_cost_binding_sha256"] = aggregate_module._canonical_sha256(
        cost["r5_cost_binding"]
    )
    _write_json(cost_path, cost)

    with pytest.raises(RuntimeError, match="not bound to its terminal source cell"):
        aggregate_matrix(matrix_summary=summary_path, expected_commit=commit)
