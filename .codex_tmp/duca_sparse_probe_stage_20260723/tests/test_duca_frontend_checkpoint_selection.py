from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.bata.aggregate_duca_frontend_candidates import (
    EXPECTED_VARIANTS,
    aggregate,
)
from tools.bata.select_duca_frontend_checkpoint import select_checkpoint


def _write_json(path: Path, payload) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(*, action_auroc: float, boundary_gain: float) -> dict:
    uniform_r0 = 0.50
    uniform_r1 = 0.9998
    return {
        "schema_version": "duca_selection_quality_summary_v2",
        "sample_count": 2,
        "coarse": {
            "pooled": {
                "auroc": action_auroc,
                "auprc": 0.40,
                "auprc_lift": 1.60,
            }
        },
        "transition": {
            "r0": {
                "policy": {"auroc": 0.70},
                "pure_abs_delta_p_action": {"auroc": 0.65},
            },
            "r1": {
                "policy": {"auroc": 0.70},
                "pure_abs_delta_p_action": {"auroc": 0.65},
            }
        },
        "selection": {
            "learned": {
                "boundary_recall": {
                    "r0": {"mean": uniform_r0 + boundary_gain},
                    "r1": {"mean": uniform_r1 - 0.05},
                },
                "mean_endpoint_distance": {"mean": 0.20},
                "max_unselected_hole": {"mean": 2.0},
            },
            "uniform": {
                "boundary_recall": {
                    "r0": {"mean": uniform_r0},
                    "r1": {"mean": uniform_r1},
                },
                "mean_endpoint_distance": {"mean": 0.30},
            },
        },
    }


def _records(path: Path) -> str:
    rows = [
        {
            "valid_len": 8,
            "budget": 4,
            "gt_segments": [[1.0, 2.0], [4.0, 6.0]],
            "selected_positions": [0, 1, 2, 6],
        },
        {
            "valid_len": 8,
            "budget": 4,
            "gt_segments": [[1.0, 2.0]],
            "selected_positions": [0, 1, 2, 7],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frontend_selection_uses_train_holdout_and_prefers_boundary_gain(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    split_sha = _write_json(split, {"test_subset_consumed": False})
    candidates = []
    for variant_idx, variant in enumerate(("a", "b", "c")):
        for epoch in (5, 10, 15, 20):
            checkpoint = tmp_path / f"{variant}_{epoch}.pth"
            checkpoint.write_bytes(f"{variant}-{epoch}".encode("ascii"))
            summary = tmp_path / f"{variant}_{epoch}.json"
            gain = 0.01 + (0.05 if variant == "b" and epoch == 10 else 0.0)
            summary_sha = _write_json(summary, _summary(action_auroc=0.60, boundary_gain=gain))
            records = tmp_path / f"{variant}_{epoch}.jsonl"
            records_sha = _records(records)
            candidates.append(
                {
                    "variant": variant,
                    "epoch_one_based": epoch,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "summary_path": str(summary),
                    "summary_sha256": summary_sha,
                    "records_path": str(records),
                    "records_sha256": records_sha,
                    "loss_weights": {"actionness": 1.0, "transition": 0.1, "transition_boundary": 16.0},
                    "component_lrs": {
                        "coarse_trunk": 5.0e-5,
                        "action_head": 1.0e-4,
                        "transition_scorer": 2.5e-5,
                    },
                }
            )
    manifest = tmp_path / "candidate_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "duca_frontend_candidate_manifest_v1",
            "source_subset": "training",
            "test_subset_consumed": False,
            "split_manifest_path": str(split),
            "split_manifest_sha256": split_sha,
            "candidates": candidates,
        },
    )
    result = select_checkpoint(manifest, tmp_path / "decision.json")
    assert result["ok"] is True
    assert result["winner"]["variant"] == "b"
    assert result["winner"]["epoch_one_based"] == 10
    assert result["test_subset_consumed"] is False
    assert result["winner"]["metrics"]["learned_boundary_recall_r1"] < result["winner"]["metrics"]["uniform_boundary_recall_r1"]
    assert "learned_boundary_recall_r1_not_below_uniform" not in result["winner"]["gates"]
    assert any("radius-1 coverage is diagnostic only" in item for item in result["selection_rule"])


def test_frontend_aggregate_binds_the_fixed_loss_and_component_lr_grid(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    split = tmp_path / "split.json"
    split_sha = _write_json(split, {"test_subset_consumed": False})
    receipts = []
    for variant, spec in EXPECTED_VARIANTS.items():
        contract = tmp_path / f"{variant}_contract.json"
        contract_sha = _write_json(
            contract,
            {
                "schema_version": "duca_frontend_p0_contract_v1",
                "ok": True,
                "git_commit": commit,
                "detector_executed": False,
                "test_subset_consumed": False,
                "loss_weights": spec["loss_weights"],
                "component_lrs": spec["component_lrs"],
            },
        )
        candidates = []
        for epoch in (5, 10, 15, 20):
            checkpoint = tmp_path / f"{variant}_{epoch}.pth"
            checkpoint.write_bytes(f"{variant}-{epoch}".encode("ascii"))
            summary = tmp_path / f"{variant}_{epoch}.json"
            boundary_gain = 0.05 if "coarse50" in variant else 0.01
            summary_sha = _write_json(
                summary,
                _summary(action_auroc=0.60, boundary_gain=boundary_gain),
            )
            records = tmp_path / f"{variant}_{epoch}.jsonl"
            records_sha = _records(records)
            candidates.append(
                {
                    "variant": variant,
                    "epoch_one_based": epoch,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "summary_path": str(summary),
                    "summary_sha256": summary_sha,
                    "records_path": str(records),
                    "records_sha256": records_sha,
                    "loss_weights": spec["loss_weights"],
                    "component_lrs": spec["component_lrs"],
                }
            )
        receipt = tmp_path / f"{variant}_completion.json"
        _write_json(
            receipt,
            {
                "schema": "duca_frontend_variant_completion_v1",
                "ok": True,
                "git_commit": commit,
                "variant": variant,
                "split_manifest_sha256": split_sha,
                "test_subset_consumed": False,
                "loss_weights": spec["loss_weights"],
                "component_lrs": spec["component_lrs"],
                "p0_contract_path": str(contract),
                "p0_contract_sha256": contract_sha,
                "candidates": candidates,
            },
        )
        receipts.append(receipt)

    candidate_manifest = tmp_path / "candidate_manifest.json"
    result = aggregate(
        expected_commit=commit,
        split_manifest_path=split,
        split_manifest_sha256=split_sha,
        receipt_paths=receipts,
        candidate_manifest_path=candidate_manifest,
        decision_path=tmp_path / "decision.json",
    )
    manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["winner"]["variant"] == "lr_coarse50_action100_scorer25"
    assert manifest["loss_grid"] == {
        "actionness": 1.0,
        "transition": 0.10,
        "transition_boundary": 16.0,
    }
    assert manifest["component_lr_grid"] == {
        name: spec["component_lrs"] for name, spec in EXPECTED_VARIANTS.items()
    }
