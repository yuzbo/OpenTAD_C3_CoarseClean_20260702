from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.bata.select_duca_frontend_checkpoint import select_checkpoint


def _write_json(path: Path, payload) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(*, action_auroc: float, boundary_gain: float) -> dict:
    uniform_r1 = 0.70
    return {
        "schema_version": "duca_selection_quality_summary_v2",
        "sample_count": 2,
        "coarse": {"pooled": {"auroc": action_auroc}},
        "transition": {
            "r1": {
                "policy": {"auroc": 0.70},
                "pure_abs_delta_p_action": {"auroc": 0.65},
            }
        },
        "selection": {
            "learned": {
                "boundary_recall": {"r1": {"mean": uniform_r1 + boundary_gain}},
                "mean_endpoint_distance": {"mean": 0.20},
                "max_unselected_hole": {"mean": 2.0},
            },
            "uniform": {
                "boundary_recall": {"r1": {"mean": uniform_r1}},
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
