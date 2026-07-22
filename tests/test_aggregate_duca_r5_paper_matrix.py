from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tools.bata.aggregate_duca_r5_paper_matrix as aggregate_module
from tools.bata.aggregate_duca_r5_paper_matrix import aggregate_matrix


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r5_aggregate_requires_all_cells_and_reports_paired_deltas(
    tmp_path: Path, monkeypatch
) -> None:
    commit = "a" * 40
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
                    checkpoint = (
                        tmp_path
                        / "runs"
                        / cell_id
                        / "gpu1_id0/checkpoint/epoch_59.pth"
                    )
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(cell_id.encode("ascii"))
                    metrics = {
                        "average_mAP": 0.65 + (0.01 if arm == "learned" else 0.0),
                        **{
                            f"mAP@{iou:.1f}": 0.70 + 0.01 * index
                            for index, iou in enumerate((0.3, 0.4, 0.5, 0.6, 0.7))
                        },
                    }
                    evaluation = {
                        "schema_version": "duca_r5_terminal_evaluation_v1",
                        "git_commit": commit,
                        "task": "offline_temporal_action_detection",
                        "config_path": str(config.resolve()),
                        "config_sha256": _sha(config),
                        "checkpoint_path": str(checkpoint.resolve()),
                        "checkpoint_sha256": _sha(checkpoint),
                        "checkpoint_epoch": 59,
                        "checkpoint_state_key": "state_dict_ema",
                        "result_count": 10,
                        "video_count": 5,
                        "metrics": metrics,
                        "r5_cell": {
                            "backend": backend,
                            "arm": arm,
                            "budget": budget,
                            "seed": seed,
                        },
                        "training_identity": {
                            "variant": cell_id,
                            "seed": seed,
                            "successful_optimizer_updates": 6000,
                        },
                    }
                    evaluation["evaluation_sha256"] = aggregate_module._canonical_sha256(
                        evaluation
                    )
                    evaluation_path = (
                        tmp_path / "results" / f"{cell_id}.terminal_evaluation.json"
                    )
                    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
                    evaluation_path.write_text(
                        json.dumps(evaluation), encoding="utf-8"
                    )
                    cells.append(
                        {
                            "id": cell_id,
                            "backend": backend,
                            "arm": arm,
                            "budget": budget,
                            "seed": seed,
                            "config": str(config.resolve()),
                            "config_sha256": _sha(config),
                        }
                    )
                    if budget == 384 and seed == 3407:
                        cost_path = tmp_path / "cost" / f"{cell_id}.summary.json"
                        cost_path.parent.mkdir(parents=True, exist_ok=True)
                        cost_path.write_text(
                            json.dumps(
                                {
                                    "config_commit": commit,
                                    "random_init": False,
                                    "stages": {
                                        "end_to_end_serial_ms": {"p50": 100.0}
                                    },
                                    "selected_count": {"p50": 384.0},
                                    "resources": {
                                        "peak_gpu_memory_mb": {"p50": 1024.0}
                                    },
                                }
                            ),
                            encoding="utf-8",
                        )
                        costs.append(
                            {
                                "id": f"cost_{cell_id}",
                                "source_cell": cell_id,
                                "summary": str(cost_path.resolve()),
                            }
                        )
    summary = {
        "schema": "duca_r5_paper_matrix_v1",
        "task": "offline_temporal_action_detection",
        "git_commit": commit,
        "cells": cells,
        "costs": costs,
    }
    summary_path = tmp_path / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(
        aggregate_module, "validate_and_rebuild_profile_summary", lambda payload: {}
    )

    result = aggregate_matrix(
        matrix_summary=summary_path,
        expected_commit=commit,
    )

    assert result["ok"] is True
    assert result["cell_count"] == 24
    assert result["cost_count"] == 4
    assert len(result["three_seed_aggregates"]) == 8
    assert len(result["paired_deltas"]) == 12
    assert all(
        abs(row["learned_minus_uniform_average_mAP"] - 0.01) < 1e-12
        for row in result["paired_deltas"]
    )
