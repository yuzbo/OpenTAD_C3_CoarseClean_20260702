from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.bata.build_duca_rime_training_targets import build_training_targets
from tools.bata.create_duca_rime_splits import create_rime_splits


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path):
    database = {}
    for index in range(10):
        database[f"train_{index:02d}"] = {"subset": "training", "annotations": []}
    for index in range(3):
        database[f"test_{index:02d}"] = {"subset": "validation", "annotations": []}
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"database": database}), encoding="utf-8")
    manifest = create_rime_splits(annotation, tmp_path / "split")
    payload = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
    target_video = payload["train_roles"]["detector_selector_train"]["videos"][0]
    fit_video = payload["train_roles"]["hard_label_generation"]["videos"][0]
    provenance = {
        "fit_split": "train_only",
        "cross_fitted": True,
        "uses_validation_or_test": False,
        "fit_video_ids": [fit_video],
        "eval_video_ids": [target_video],
    }
    observations = []
    for budget, utility, failure in ((2, 0.1, 1), (4, 0.4, 0)):
        observations.append(
            {
                "schema_version": "duca_rime_budget_target_observation_v1",
                "video_id": target_video,
                "window_start_frame": 0,
                "budget": budget,
                "utility": utility,
                "observed_pair_failure": failure,
                "provenance": provenance,
            }
        )
    hard = [
        {
            "schema_version": "duca_rime_hard_frame_target_v1",
            "video_id": target_video,
            "window_start_frame": 0,
            "hard_frame_utility": [0.0, 1.0, 0.5, 0.25],
            "provenance": provenance,
        }
    ]
    return (
        manifest,
        target_video,
        _write_jsonl(tmp_path / "observations.jsonl", observations),
        _write_jsonl(tmp_path / "hard.jsonl", hard),
    )


def test_build_rime_targets_is_cross_fit_hash_bound_and_immutable(tmp_path):
    manifest, target_video, observations, hard = _fixture(tmp_path)
    output = tmp_path / "targets.jsonl"
    result = build_training_targets(
        split_manifest=manifest["manifest_path"],
        split_manifest_sha256=manifest["manifest_sha256"],
        observations_jsonl=observations,
        hard_utility_jsonl=hard,
        output_jsonl=output,
        candidate_budgets=(2, 4),
    )
    assert result["official_final_subset_consumed"] is False
    assert result["target_artifact"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["video_id"] == target_video
    assert row["utility_target"] == pytest.approx([0.1, 0.4])
    assert row["risk_target"] == pytest.approx([1.0, 0.0])
    assert row["provenance"]["cross_fitted"] is True
    assert row["provenance"]["uses_validation_or_test"] is False
    repeated = build_training_targets(
        split_manifest=manifest["manifest_path"],
        split_manifest_sha256=manifest["manifest_sha256"],
        observations_jsonl=observations,
        hard_utility_jsonl=hard,
        output_jsonl=output,
        candidate_budgets=(2, 4),
    )
    assert repeated["target_artifact"]["sha256"] == result["target_artifact"]["sha256"]


def test_build_rime_targets_rejects_non_cross_fitted_source(tmp_path):
    manifest, _target_video, observations, hard = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in observations.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["provenance"]["cross_fitted"] = False
    _write_jsonl(observations, rows)
    with pytest.raises(ValueError, match="cross-fit"):
        build_training_targets(
            split_manifest=manifest["manifest_path"],
            split_manifest_sha256=manifest["manifest_sha256"],
            observations_jsonl=observations,
            hard_utility_jsonl=hard,
            output_jsonl=tmp_path / "targets.jsonl",
            candidate_budgets=(2, 4),
        )


def test_build_rime_targets_rejects_incomplete_budget_panel(tmp_path):
    manifest, _target_video, observations, hard = _fixture(tmp_path)
    observations.write_text(
        observations.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        build_training_targets(
            split_manifest=manifest["manifest_path"],
            split_manifest_sha256=manifest["manifest_sha256"],
            observations_jsonl=observations,
            hard_utility_jsonl=hard,
            output_jsonl=tmp_path / "targets.jsonl",
            candidate_budgets=(2, 4),
        )
