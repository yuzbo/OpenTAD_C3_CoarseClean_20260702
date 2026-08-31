from __future__ import annotations

from inspect import getsource
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.bata import run_duca_whole_video_consistent_budget_falsifier as runner


def _row(video_id: str, window: int, valid_observations: int) -> dict:
    def accounting(budget: int) -> dict:
        return {
            "actual_cost": min(valid_observations, budget),
            "effective_tier": (
                384
                if budget != 384 and min(valid_observations, budget) == min(valid_observations, 384)
                else budget
            ),
        }

    return {
        "video_id": video_id,
        "sample_id": f"{video_id}|{window:04d}",
        "valid_observations": valid_observations,
        "budget_accounting": {
            str(budget): accounting(budget) for budget in runner.BUDGETS
        },
        "predictions": {str(budget): [] for budget in runner.BUDGETS},
    }


def _frozen_shape_rows() -> tuple[list[dict], list[str]]:
    videos = [f"video_{index:03d}" for index in range(40)]
    rows = []
    # Four videos have four windows and the remaining 36 have three: 124 total.
    for video_index, video_id in enumerate(videos):
        window_count = 4 if video_index < 4 else 3
        for window in range(window_count):
            valid = 600
            if video_index == 0 and window == 0:
                valid = 67
            elif video_index == 1 and window == 0:
                valid = 195
            rows.append(_row(video_id, window, valid))
    rows.sort(key=lambda value: (value["video_id"], value["sample_id"]))
    return rows, videos


def test_complete_whole_video_candidate_space_uses_real_cost_and_uniform_tiers() -> None:
    rows, videos = _frozen_shape_rows()

    manifest = runner._build_candidate_manifest(rows, videos)

    assert len(rows) == 124
    assert manifest["fixed_actual_observation_cost"] == 47_110
    assert manifest["ordered_pair_count"] == 40 * 39
    assert len(manifest["candidates"]) == 40 * 39
    assert len({value["candidate_id"] for value in manifest["candidates"]}) == 40 * 39
    assert manifest["legal_candidate_count"] > 0
    assert manifest["labels_or_ground_truth_read"] is False
    assert manifest["metric_evaluated"] is False

    candidate = next(value for value in manifest["candidates"] if value["legal"])
    budgets = runner._requested_budgets_for_pair(
        rows, candidate["donor_video_id"], candidate["recipient_video_id"]
    )
    by_video: dict[str, set[int]] = {}
    for row, budget in zip(rows, budgets):
        by_video.setdefault(row["video_id"], set()).add(budget)
    assert by_video[candidate["donor_video_id"]] == {256}
    assert by_video[candidate["recipient_video_id"]] == {512}
    assert all(
        tiers == {384}
        for video_id, tiers in by_video.items()
        if video_id not in {candidate["donor_video_id"], candidate["recipient_video_id"]}
    )
    assert runner._actual_cost(rows, budgets) == candidate["actual_observation_cost"]
    assert candidate["actual_observation_cost"] <= 47_110

    source = getsource(runner)
    assert "allocate_equal_budget_marginal_reallocation" not in source
    assert "_allocate_rows_by_video" not in source


def test_input_bundle_preserves_sealed_prediction_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, videos = _frozen_shape_rows()
    sealed_rows = [rows[3], rows[1], rows[2], rows[0], *rows[4:]]
    split_dir = tmp_path / "controller_split"
    split_dir.mkdir()
    (split_dir / "frontend_split_manifest.json").write_text(
        json.dumps({"holdout_videos": videos}), encoding="utf-8"
    )

    selection = [
        {**row, "prediction_k384": [{"sample_id": row["sample_id"]}]}
        for row in sealed_rows
    ]
    counterfactual = {
        budget: [
            {
                "sample_id": row["sample_id"],
                "prediction": [{"sample_id": row["sample_id"], "budget": budget}],
            }
            for row in sealed_rows
        ]
        for budget in (256, 512)
    }

    def read_rows(path: Path):
        if path.name == "selection_k384.jsonl.gz":
            return selection
        if path.name == "counterfactual_k256.jsonl.gz":
            return counterfactual[256]
        if path.name == "counterfactual_k512.jsonl.gz":
            return counterfactual[512]
        raise AssertionError(path)

    monkeypatch.setattr(runner, "_read_jsonl_gz", read_rows)

    bundle = runner._prepare_unlabeled_bundle(tmp_path)

    assert [row["sample_id"] for row in bundle["rows"]] == [
        row["sample_id"] for row in sealed_rows
    ]


def test_short_windows_collapse_by_actual_observations_not_requested_budget() -> None:
    rows = [
        _row("short", 0, 67),
        _row("middle", 0, 300),
        _row("partial_packet", 0, 401),
        _row("long", 0, 600),
    ]

    assert [runner._actual_cost(rows, [budget] * len(rows)) for budget in (256, 384, 512)] == [
        67 + 256 + 256 + 256,
        67 + 300 + 384 + 384,
        67 + 300 + 401 + 512,
    ]
    assert runner._actual_observation_cost(rows[0], 256) == runner._actual_observation_cost(
        rows[0], 384
    )
    assert runner._actual_observation_cost(rows[1], 512) == runner._actual_observation_cost(
        rows[1], 384
    )


def test_pre_run_writes_complete_candidates_before_labeled_context_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []
    rows = [_row("donor", 0, 600), _row("recipient", 0, 600)]
    candidate_manifest = {
        "schema": "test",
        "fixed_actual_observation_cost": 47_110,
        "ordered_pair_count": 2,
        "legal_candidate_count": 1,
        "candidates": [],
    }

    monkeypatch.setattr(runner, "_verify_current_git", lambda _head: {"head": "head", "dirty": False})
    monkeypatch.setattr(
        runner,
        "_prepare_unlabeled_bundle",
        lambda _path: {"rows": rows, "holdout_videos": ["donor", "recipient"]},
    )

    def build_manifest(_rows, _videos):
        events.append("candidates_generated")
        return candidate_manifest

    monkeypatch.setattr(runner, "_build_candidate_manifest", build_manifest)
    monkeypatch.setattr(runner, "_load_terminal_evidence", lambda bundle: bundle)

    def load_labeled(_args, _bundle):
        assert events == ["candidates_generated"]
        assert (tmp_path / "whole_video_candidate_manifest.json").is_file()
        events.append("labeled_context_loaded")
        return {
            "current_source": {},
            "holdout_block_list": tmp_path / "source_block_list.txt",
            "provenance": {},
        }

    monkeypatch.setattr(runner, "_load_labeled_context", load_labeled)
    monkeypatch.setattr(
        runner,
        "_copy_holdout_block_list",
        lambda _source, _output: tmp_path / "copied_block_list.txt",
    )

    def reproduce(**_kwargs):
        assert events == ["candidates_generated", "labeled_context_loaded"]
        events.append("metrics_evaluated")
        return {"metrics": {}, "reproduction_error_pp": {}, "maximum_reproduction_error_pp": 0.0}

    monkeypatch.setattr(runner, "_reproduce_anchors", reproduce)
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        input_dir=str(tmp_path / "inputs"),
        expected_head="head",
        evaluator_threads=1,
    )

    receipt = runner.run_pre_run_stage(args)

    assert receipt["status"] == "PRE_RUN_PASS"
    assert events == [
        "candidates_generated",
        "labeled_context_loaded",
        "metrics_evaluated",
    ]
    assert receipt["candidate_manifest"]["generated_before_label_or_metric_access"] is True


def test_joint_gate_ranking_uses_margin_then_cost_then_video_ids() -> None:
    records = [
        {
            "candidate_id": "b=>c",
            "donor_video_id": "b",
            "recipient_video_id": "c",
            "joint_gate_margin_pp": 0.1,
            "actual_observation_cost": 47_000,
        },
        {
            "candidate_id": "a=>c",
            "donor_video_id": "a",
            "recipient_video_id": "c",
            "joint_gate_margin_pp": 0.1,
            "actual_observation_cost": 46_900,
        },
        {
            "candidate_id": "a=>b",
            "donor_video_id": "a",
            "recipient_video_id": "b",
            "joint_gate_margin_pp": 0.1,
            "actual_observation_cost": 46_900,
        },
        {
            "candidate_id": "z=>a",
            "donor_video_id": "z",
            "recipient_video_id": "a",
            "joint_gate_margin_pp": 0.2,
            "actual_observation_cost": 47_110,
        },
    ]

    first = sorted(records, key=runner._candidate_rank)
    second = sorted(list(reversed(records)), key=runner._candidate_rank)

    assert [value["candidate_id"] for value in first] == [
        "z=>a",
        "a=>b",
        "a=>c",
        "b=>c",
    ]
    assert first == second
