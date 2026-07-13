from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import diagnose_duca_coarse_boundary_ceiling as ceiling


def _row(video: str) -> dict:
    return {"sample_id": f"{video}|0", "video_id": video, "valid_len": 8, "budget": 4, "gt_segments": [[2, 6]], "p_action": [0, 0, .8, .9, .9, .8, 0, 0], "abs_delta_p_action": [0, 0, .8, .1, 0, .1, .8, 0], "uncertainty": [.1, .1, .2, .1, .1, .2, .1, .1]}


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_ceiling_uses_held_out_deploy_visible_features_and_writes_outputs(tmp_path: Path) -> None:
    train, evaluate = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _dump(train, [_row("train")]); _dump(evaluate, [_row("eval")])
    result = ceiling.run(train, evaluate, tmp_path / "out", radius=1, steps=3)
    assert result["video_overlap_count"] == 0
    assert result["leakage_contract"]["gt_used_as_model_input"] is False
    assert result["diagnostic_role"].endswith("not_ceiling_not_oracle")
    assert result["probability_semantics"] == "unweighted_logistic_posterior_fit_on_train_only_labels"
    assert "ece" in result["metrics"]
    assert (tmp_path / "out" / "coarse_boundary_probe.json").is_file()
    assert (tmp_path / "out" / "coarse_boundary_probe_per_sample.csv").is_file()


def test_ceiling_rejects_video_overlap(tmp_path: Path) -> None:
    train, evaluate = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    _dump(train, [_row("same")]); _dump(evaluate, [_row("same")])
    with pytest.raises(ValueError, match="leakage"):
        ceiling.run(train, evaluate, tmp_path / "out", steps=1)
