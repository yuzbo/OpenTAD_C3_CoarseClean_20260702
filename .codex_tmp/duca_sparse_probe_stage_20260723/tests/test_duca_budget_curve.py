from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.bata.aggregate_duca_budget_curve import export_budget_curve
from tools.bata.aggregate_duca_r5_paper_matrix import _canonical_sha256
from tools.bata.duca_p0_evaluation import official_evaluator_identity


def _aggregate(path: Path, budgets: tuple[int, ...], commit: str) -> Path:
    rows = []
    for backend in ("actionformer", "temporalmaxer"):
        for arm in ("uniform", "learned"):
            for budget in budgets:
                for seed in (3407, 5801, 8123):
                    rows.append(
                        {
                            "id": f"{backend}_{arm}_k{budget}_s{seed}",
                            "backend": backend,
                            "arm": arm,
                            "budget": budget,
                            "seed": seed,
                            "average_mAP": 60.0 + budget / 100.0 + (arm == "learned"),
                            "iou_mAP": {
                                f"mAP@{iou:.1f}": 50.0 + budget / 100.0 + iou
                                for iou in (0.3, 0.4, 0.5, 0.6, 0.7)
                            },
                            "evaluator": official_evaluator_identity(),
                            "evaluation_config": {
                                "subset": "validation",
                                "blocked_videos": None,
                            },
                        }
                    )
    payload = {
        "schema": "duca_r5_paper_matrix_results_v1",
        "ok": True,
        "status": "r5_raw_evidence_complete_pending_claim_adjudication",
        "task": "offline_temporal_action_detection",
        "git_commit": commit,
        "matrix_axes": {
            "backends": ["actionformer", "temporalmaxer"],
            "arms": ["uniform", "learned"],
            "budgets": list(budgets),
            "seeds": [3407, 5801, 8123],
        },
        "rows": rows,
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_merges_disjoint_official_r5_aggregates_into_five_point_curve(
    tmp_path: Path,
) -> None:
    old = _aggregate(tmp_path / "old.json", (384, 256), "a" * 40)
    extension = _aggregate(tmp_path / "extension.json", (320, 192, 128), "b" * 40)
    result = export_budget_curve(
        aggregate_jsons=(old, extension),
        output_dir=tmp_path / "curve",
    )
    assert result["expected_budgets"] == [384, 320, 256, 192, 128]
    assert len(result["raw_rows"]) == 60
    assert len(result["summary_rows"]) == 20
    assert (tmp_path / "curve/duca_official_budget_curve_summary.csv").is_file()
    assert (tmp_path / "curve/duca_official_budget_curve.json").is_file()


def test_budget_curve_rejects_missing_budget(tmp_path: Path) -> None:
    incomplete = _aggregate(tmp_path / "incomplete.json", (384, 256), "a" * 40)
    with pytest.raises(RuntimeError, match="budget curve mismatch"):
        export_budget_curve(
            aggregate_jsons=(incomplete,),
            output_dir=tmp_path / "curve",
        )


@pytest.mark.skipif(os.name == "nt", reason="selection analysis imports Windows torch")
def test_selection_curve_discovers_disjoint_matrix_cells_and_extracts_metrics(
    tmp_path: Path,
) -> None:
    from tools.bata.analyze_duca_budget_selection_curve import _load_cells, _metric_row

    summaries = []
    for name, budgets in (("old", (384, 256)), ("new", (320, 192, 128))):
        root = tmp_path / name
        cells = []
        for budget in budgets:
            config = root / "configs" / f"actionformer_learned_k{budget}_s3407.py"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text("model = {}\n", encoding="utf-8")
            cells.append(
                {
                    "id": f"actionformer_learned_k{budget}_s3407",
                    "backend": "actionformer",
                    "arm": "learned",
                    "budget": budget,
                    "seed": 3407,
                    "config": str(config.resolve()),
                }
            )
        summary_path = root / "matrix_summary.json"
        sha_path = root / "matrix_summary.json.sha256"
        payload = {
            "schema": "duca_r5_paper_matrix_v1",
            "git_commit": "a" * 40,
            "output_dir": str(root.resolve()),
            "budgets": list(budgets),
            "cells": cells,
            "matrix_summary_sha256_file": str(sha_path.resolve()),
        }
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        sha_path.write_text(
            hashlib.sha256(summary_path.read_bytes()).hexdigest() + "\n",
            encoding="utf-8",
        )
        summaries.append(summary_path)
    cells, sources = _load_cells(
        summaries,
        expected_budgets=(384, 320, 256, 192, 128),
        backend="actionformer",
        seed=3407,
    )
    assert [cell["budget"] for cell in cells] == [384, 320, 256, 192, 128]
    assert len(sources) == 2
    metrics = _metric_row(
        budget=128,
        cell_id="cell",
        summary={
            "coarse": {"pooled": {"auroc": 0.8, "auprc": 0.7}},
            "transition": {"r0": {"policy": {"auprc": 0.6}}},
            "selection": {
                "learned": {
                    "selected_count": {"mean": 128},
                    "mean_endpoint_distance": {"mean": 1.5},
                    "max_unselected_hole": {"mean": 6},
                    "action_enrichment": {"mean": 1.2},
                    "pooled": {"boundary_recall": {"r0": 0.4, "r2": 0.8, "r4": 0.9}},
                    "boundary_burst": {
                        "r2q3": {
                            "endpoint_quota_recall": {"mean": 0.5},
                            "endpoint_bilateral_recall": {"mean": 0.7},
                            "both_endpoints_quota_recall": {"mean": 0.3},
                        }
                    },
                }
            },
            "comparison": {
                "paired_learned_minus_uniform_boundary_recall_r0": {"mean": 0.1},
                "paired_uniform_minus_learned_endpoint_distance": {"mean": 0.2},
            },
        },
    )
    assert metrics["boundary_recall_r2"] == pytest.approx(0.8)
    assert metrics["r2q3_endpoint_bilateral_recall"] == pytest.approx(0.7)
