import hashlib
import json

from tools.bata.create_duca_rime_splits import create_rime_splits
from tools.bata.duca_rime_training import (
    PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE,
    PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS,
)
from tools.bata.evaluate_duca_rime_predictions import evaluate_predictions


def _json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prediction_metrics_are_split_bound_and_perfect_for_perfect_segments(tmp_path):
    database = {
        f"train_{index:03d}": {
            "subset": "training",
            "annotations": [{"label": "action", "segment": [1.0, 2.0]}],
        }
        for index in range(30)
    }
    database.update(
        {
            f"test_{index:03d}": {
                "subset": "validation",
                "annotations": [{"label": "action", "segment": [1.0, 2.0]}],
            }
            for index in range(5)
        }
    )
    annotation = _json(tmp_path / "annotation.json", {"database": database})
    split = create_rime_splits(annotation, tmp_path / "split")
    manifest = json.loads(
        (tmp_path / "split" / "duca_rime_split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    development = manifest["train_roles"]["certification_development"]
    predictions = _json(
        tmp_path / "predictions.json",
        {
            "results": {
                video: [
                    {
                        "label": "action",
                        "segment": [1.0, 2.0],
                        "score": 1.0,
                    }
                ]
                for video in development["videos"]
            }
        },
    )
    terminal = _json(
        tmp_path / "terminal.json",
        {
            "schema_version": "duca_rime_terminal_evaluation_v1",
            "task": "offline_temporal_action_detection",
            "runtime_gt_input_to_selector": False,
            "padded_to_kmax": False,
            "detector_backend": "ActionFormer",
            "target_mean_cost": 384.0,
            "git_commit": "a" * 40,
            "variant": "RIME-full",
            "seed": 3407,
            "prediction_path": str(predictions.resolve()),
            "prediction_sha256": _sha(predictions),
            "evaluation_config": {
                "type": "mAP",
                "ground_truth_filename": str(annotation.resolve()),
                "subset": "training",
                "tiou_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
                "top_k": None,
                "blocked_videos": development["block_list_path"],
                "thread": 1,
            },
        },
    )
    result = evaluate_predictions(
        terminal_evaluation=terminal,
        split_manifest=split["manifest_path"],
        split_manifest_sha256=split["manifest_sha256"],
        phase=3,
        short_max_seconds=2.0,
        medium_max_seconds=5.0,
        output=tmp_path / "metrics.json",
    )
    metrics = result["payload"]["video_metrics"]
    assert set(metrics["avg_map"].values()) == {1.0}
    assert set(metrics["map_0.7"].values()) == {1.0}
    assert set(metrics["short_map"].values()) == {1.0}
    assert set(metrics["pair_support"].values()) == {1.0}
    assert set(metrics["boundary_error"].values()) == {0.0}

    baseline_payload = json.loads(terminal.read_text(encoding="utf-8"))
    baseline_payload.update(
        {
            "schema_version": (
                "duca_rime_phase2_baseline_terminal_evaluation_v1"
            ),
            "variant": "U-fixed",
            "training_identity": None,
            "baseline_contract": {
                "phase": 2,
                "uses_official_final": False,
                "padded_to_kmax": False,
            },
            "checkpoint_compatibility": {
                "mode": PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE,
                "missing_keys": [],
                "ignored_unexpected_keys": sorted(
                    PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS
                ),
            },
        }
    )
    baseline_terminal = _json(
        tmp_path / "baseline_terminal.json",
        baseline_payload,
    )
    baseline = evaluate_predictions(
        terminal_evaluation=baseline_terminal,
        split_manifest=split["manifest_path"],
        split_manifest_sha256=split["manifest_sha256"],
        phase=2,
        split_role="certification_development",
        short_max_seconds=2.0,
        medium_max_seconds=5.0,
        output=tmp_path / "baseline_metrics.json",
    )
    assert baseline["payload"]["terminal_schema_version"] == (
        "duca_rime_phase2_baseline_terminal_evaluation_v1"
    )
    assert baseline["payload"]["uses_official_final"] is False
