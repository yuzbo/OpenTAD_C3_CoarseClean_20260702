from __future__ import annotations

import json

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.datasets.transforms.end_to_end import DucaExternalActionnessFromJsonl


def _write_x3d_jsonl(path, *, video_id: str = "video_test", scores=None) -> None:
    if scores is None:
        scores = [0.1, 0.2, 0.8, 0.9]
    provenance = {
        "source_name": "frozen_kinetics_x3d_xs_actionness",
        "thumos_trained": False,
        "uses_labels": False,
        "uses_teacher": False,
        "uses_gt": False,
        "uses_prediction_cache": False,
        "calibration_split": "none",
        "checkpoint_hash": "pytorch_provider:x3d_xs:pretrained=True",
    }
    with path.open("w", encoding="utf-8") as handle:
        for idx, score in enumerate(scores):
            prob = float(score)
            handle.write(
                json.dumps(
                    {
                        "video_id": video_id,
                        "window_id": f"{video_id}_{idx:04d}",
                        "time_index": idx,
                        "original_time": float(idx),
                        "p_action": prob,
                        "logit": float(torch.logit(torch.tensor(prob).clamp(1e-6, 1 - 1e-6)).item()),
                        "valid": True,
                        "source_name": provenance["source_name"],
                        "source_provenance": provenance,
                        "thumos_trained": False,
                        "uses_labels": False,
                        "uses_teacher": False,
                        "calibration_split": "none",
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def test_duca_x3d_jsonl_transform_writes_external_actionness_meta(tmp_path) -> None:
    jsonl = tmp_path / "x3d.jsonl"
    _write_x3d_jsonl(jsonl)
    transform = DucaExternalActionnessFromJsonl(actionness_jsonl=str(jsonl))

    out = transform(
        {
            "video_name": "video_test",
            "avg_fps": 1.0,
            "frame_inds": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "masks": torch.ones(4, dtype=torch.bool),
        }
    )

    assert out["duca_external_p_action"] == pytest.approx([0.1, 0.2, 0.8, 0.9])
    assert len(out["duca_external_actionness_logits"]) == 4
    assert out["duca_external_actionness_source"] == "frozen_kinetics_x3d_xs_actionness"
    assert out["duca_external_actionness_provenance"]["uses_gt"] is False
    assert out["duca_external_actionness_valid"] == [True, True, True, True]


def test_duca_x3d_jsonl_transform_aligns_grouped_frame_indices_to_observations(tmp_path) -> None:
    jsonl = tmp_path / "x3d.jsonl"
    _write_x3d_jsonl(jsonl, scores=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    transform = DucaExternalActionnessFromJsonl(actionness_jsonl=str(jsonl))

    out = transform(
        {
            "video_name": "video_test",
            "avg_fps": 1.0,
            "frame_inds": np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
            "masks": torch.ones(3, dtype=torch.bool),
        }
    )

    assert out["duca_external_p_action"] == pytest.approx([0.05, 0.25, 0.45])
    assert out["duca_external_actionness_observation_times"] == pytest.approx([0.5, 2.5, 4.5])


def test_duca_x3d_jsonl_transform_fails_closed_for_missing_video(tmp_path) -> None:
    jsonl = tmp_path / "x3d.jsonl"
    _write_x3d_jsonl(jsonl, video_id="different_video")
    transform = DucaExternalActionnessFromJsonl(actionness_jsonl=str(jsonl))

    with pytest.raises(ValueError, match="missing external actionness"):
        transform(
            {
                "video_name": "video_test",
                "avg_fps": 1.0,
                "frame_inds": np.asarray([0, 1], dtype=np.int64),
                "masks": torch.ones(2, dtype=torch.bool),
            }
        )
