from __future__ import annotations

import hashlib
import json

import pytest

from tools.bata.build_duca_rime_gate_records import supervised_records
from tools.bata.build_duca_rime_training_targets import build_training_targets
from tools.bata.create_duca_rime_splits import create_rime_splits
from tools.bata.produce_duca_rime_crossfit_records import (
    MEASUREMENT_SCHEMA,
    produce_crossfit_records,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    annotation = tmp_path / "annotation.json"
    database = {
        f"train_{index:03d}": {"subset": "training", "annotations": []}
        for index in range(40)
    }
    database.update(
        {
            f"final_{index:03d}": {"subset": "validation", "annotations": []}
            for index in range(5)
        }
    )
    annotation.write_text(json.dumps({"database": database}), encoding="utf-8")
    split = create_rime_splits(annotation, tmp_path / "split")
    manifest = json.loads(
        open(split["manifest_path"], encoding="utf-8").read()
    )
    role_by_video = {
        video: role
        for role, payload in manifest["train_roles"].items()
        for video in payload["videos"]
    }
    budgets = (2, 4, 6)
    pattern = (0.2, -0.4, 0.8)
    rows = []
    for video_index, video in enumerate(sorted(role_by_video)):
        scalar = (video_index + 1) / 50.0
        budget_features = [
            [scalar, pattern[index], scalar * pattern[index]]
            for index in range(len(budgets))
        ]
        frame_features = [
            [scalar, frame / 8.0] for frame in range(8)
        ]
        frame_fit_positions = (1, 3, 5)
        row = {
                "schema_version": MEASUREMENT_SCHEMA,
                "video_id": video,
                "window_start_frame": 0,
                "split_role": role_by_video[video],
                "candidate_budgets": list(budgets),
                "budget_features": budget_features,
                "actual_utility": [
                    0.1 * scalar + 2.0 * value for value in pattern
                ],
                "observed_pair_failure": [
                    int(value < 0.0) for value in pattern
                ],
                "frame_features": frame_features,
                "frame_fit_features": [
                    frame_features[frame] for frame in frame_fit_positions
                ],
                "actual_frame_utility": [
                    scalar + frame / 8.0 for frame in frame_fit_positions
                ],
                "frame_counterfactuals": [
                    {
                        "added_position": frame,
                        "removed_position": frame - 1,
                        "utility": scalar + frame / 8.0,
                    }
                    for frame in frame_fit_positions
                ],
                "cost_ledger": [
                    {
                        "requested_k": budget,
                        "effective_k": budget,
                        "unique_k": budget,
                        "backbone_input_k": budget,
                        "padded_k": budget,
                        "max_gap_violation": False,
                    }
                    for budget in budgets
                ],
                "split_assignment_sha256": split["assignment_sha256"],
                "provenance": {
                    "measurement_kind": "measured_detector_counterfactual",
                    "fit_split": "train_only",
                    "uses_official_final": False,
                    "uses_gt_for_supervision": True,
                    "uses_gt_at_deployment": False,
                    "uses_teacher_at_deployment": False,
                    "uses_prediction_cache_at_deployment": False,
                    "cheap_features_only_at_deployment": True,
                    "counterfactual_utility": True,
                    "proposal_score_surrogate_utility": False,
                    "pad_to_kmax": False,
                    "detector_checkpoint_sha256": "a" * 64,
                    "source_artifact_sha256": "b" * 64,
                },
            }
        row["record_sha256"] = hashlib.sha256(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        rows.append(row)
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return split, budgets, measurements


def test_crossfit_producer_emits_real_gate_and_training_sources(tmp_path):
    split, budgets, measurements = _fixture(tmp_path)
    root = tmp_path / "producer"
    result = produce_crossfit_records(
        split_manifest=split["manifest_path"],
        split_manifest_sha256=split["manifest_sha256"],
        measurements_jsonl=measurements,
        output_root=root,
        candidate_budgets=budgets,
        risk_threshold=0.5,
    )
    assert result["official_final_subset_consumed"] is False
    assert (
        result["source_measurements"]["proposal_score_surrogate_utility"]
        is False
    )
    for kind in ("o3", "o4", "price"):
        sealed = supervised_records(
            source_jsonl=root / f"{kind}_source.jsonl",
            split_manifest=split["manifest_path"],
            split_manifest_sha256=split["manifest_sha256"],
            output=tmp_path / f"{kind}_records.jsonl",
            kind=kind,
        )
        assert sealed["official_final_subset_consumed"] is False
        assert sealed["output"]["record_count"] > 0
    targets = build_training_targets(
        split_manifest=split["manifest_path"],
        split_manifest_sha256=split["manifest_sha256"],
        observations_jsonl=root / "budget_observations.jsonl",
        hard_utility_jsonl=root / "hard_frame_utility.jsonl",
        output_jsonl=tmp_path / "targets.jsonl",
        candidate_budgets=budgets,
    )
    assert targets["target_video_count"] == 20
    assert targets["official_final_subset_consumed"] is False


def test_crossfit_producer_rejects_proposal_score_surrogate(tmp_path):
    split, budgets, measurements = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in measurements.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["provenance"]["counterfactual_utility"] = False
    rows[0]["provenance"]["proposal_score_surrogate_utility"] = True
    rows[0]["record_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in rows[0].items() if key != "record_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    measurements.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="real train-only"):
        produce_crossfit_records(
            split_manifest=split["manifest_path"],
            split_manifest_sha256=split["manifest_sha256"],
            measurements_jsonl=measurements,
            output_root=tmp_path / "producer",
            candidate_budgets=budgets,
        )
