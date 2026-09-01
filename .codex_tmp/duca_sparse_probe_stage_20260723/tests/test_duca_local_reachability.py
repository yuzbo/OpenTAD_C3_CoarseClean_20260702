from __future__ import annotations

import json

import pytest

from tools.bata.analyze_duca_selection_quality import RECORD_SCHEMA_VERSION
from tools.bata.duca_exact_physical_solver import GroundTruthObjectiveSpec
from tools.bata.diagnose_duca_local_reachability import (
    FAMILY_KEYS,
    diagnose_record,
    run_diagnostic,
)


def _record(*, video_id: str = "video_validation_0000001") -> dict:
    valid_len = 8
    budget = 4
    delta = [0.0, 0.9, 0.2, 0.1, 0.1, 0.3, 0.8, 0.0]
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "sample_id": f"{video_id}|0",
        "video_id": video_id,
        "window_start_frame": 0,
        "snippet_stride": 4,
        "valid_len": valid_len,
        "budget": budget,
        "gt_segments": [[1.0, 2.0], [5.0, 6.0]],
        "p_action": [0.1, 0.8, 0.9, 0.2, 0.1, 0.7, 0.8, 0.1],
        "actionness_logits": [-2.0, 1.4, 2.0, -1.0, -2.0, 0.8, 1.4, -2.0],
        "transition_policy_scores": delta,
        "raw_transition_scores": delta,
        "abs_delta_p_action": delta,
        "uncertainty": [0.1] * valid_len,
        "selected_positions": [0, 2, 5, 7],
        "decode_diagnostics": {
            "repair_count": 0,
            "repair_enabled": True,
            "repair_feasible": True,
            "repair_satisfied": True,
            "max_hole_before": 2,
            "max_hole_after": 2,
        },
        "selection_path": "test",
        "policy_mix_alpha": 1.0,
        "source": {
            "config": "fixture.py",
            "checkpoint": "fixture.pth",
            "checkpoint_sha256": "0" * 64,
            "checkpoint_state_key": "state_dict_ema",
            "checkpoint_epoch": 19,
            "git_commit": "0" * 40,
            "split": "val",
            "selector_only_inference": True,
            "detector_backbone_executed": False,
            "uses_gt_for_selection": False,
            "seed": 3407,
        },
        "gt_role": "evaluation_only_not_selector_input",
    }


def _spec() -> GroundTruthObjectiveSpec:
    return GroundTruthObjectiveSpec(
        boundary_radii=(0, 1, 2),
        short_action_max_length=3.0,
        distance_scale=10,
        lex_block_size=8,
    )


def test_local_reachability_has_matched_exact_k_and_privileged_oracles() -> None:
    pytest.importorskip("scipy.optimize")
    row = _record()
    result = diagnose_record(
        row,
        holdout_videos={row["video_id"]},
        objective_spec=_spec(),
        gt_time_limit_seconds=30.0,
    )
    assert tuple(result["families"]) == FAMILY_KEYS
    for family in result["families"].values():
        assert family["budget"] == row["budget"]
        assert len(family["positions"]) == row["budget"]
        assert family["physical_cap_compliant"]
    assert result["families"]["L_privileged_local_gt_oracle"]["privileged"]
    assert not result["families"]["L_privileged_local_gt_oracle"]["deployable"]
    assert result["families"]["G_privileged_global_gt_oracle"]["privileged"]
    assert not result["families"]["G_privileged_global_gt_oracle"]["deployable"]
    local = result["families"]["L_privileged_local_gt_oracle"]["solver"]
    assert "_one_per_cell_" in local["solver_identity"]


def test_deployable_local_selections_do_not_change_with_gt() -> None:
    pytest.importorskip("scipy.optimize")
    row = _record()
    changed = _record()
    changed["gt_segments"] = [[0.0, 7.0]]
    first = diagnose_record(
        row,
        holdout_videos={row["video_id"]},
        objective_spec=_spec(),
        gt_time_limit_seconds=30.0,
    )
    second = diagnose_record(
        changed,
        holdout_videos={changed["video_id"]},
        objective_spec=_spec(),
        gt_time_limit_seconds=30.0,
    )
    for key in ("U_exact_uniform", "D_pure_delta_one_per_cell", "C_current_checkpoint"):
        assert first["families"][key]["positions"] == second["families"][key]["positions"]


def test_run_diagnostic_binds_records_to_training_holdout(tmp_path) -> None:
    pytest.importorskip("scipy.optimize")
    row = _record()
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "schema": "duca_frontend_train_holdout_split_v1",
        "source_subset": "training",
        "test_subset_consumed": False,
        "holdout_videos": [row["video_id"]],
        "train_videos": ["video_validation_0000002"],
    }
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary = run_diagnostic(
        input_jsonl=input_path,
        split_manifest=manifest_path,
        output_jsonl=tmp_path / "diagnostic.jsonl",
        summary_json=tmp_path / "summary.json",
        objective_spec=_spec(),
        gt_time_limit_seconds=30.0,
    )
    assert summary["sample_count"] == 1
    assert summary["contract"]["training_holdout_only"]
    assert not summary["contract"]["detector_map_evaluated"]
    assert set(summary["families"]) == set(FAMILY_KEYS)


def test_reachability_rejects_records_outside_holdout() -> None:
    row = _record()
    with pytest.raises(ValueError, match="outside the training holdout"):
        diagnose_record(
            row,
            holdout_videos={"another_video"},
            objective_spec=_spec(),
            gt_time_limit_seconds=30.0,
        )
