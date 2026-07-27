from __future__ import annotations

import json

import pytest

from tools.bata.build_duca_rime_budget_replay import (
    histogram_shuffle,
    paired_same_k,
)
from tools.bata.create_duca_rime_splits import (
    TRAIN_ROLES,
    create_rime_splits,
    validate_rime_splits,
)
from tools.bata.duca_rime_phase2 import (
    analyze_o1,
    analyze_o2,
    analyze_o3,
    analyze_o4,
    phase0_variance,
)


def test_six_role_split_is_deterministic_disjoint_and_hashed(tmp_path):
    database = {
        **{
            f"train_{index:02d}": {"subset": "training"}
            for index in range(20)
        },
        **{
            f"test_{index:02d}": {"subset": "testing"}
            for index in range(5)
        },
    }
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"database": database}), encoding="utf-8")

    manifest = create_rime_splits(annotation, tmp_path / "split", seed=11)
    validation = validate_rime_splits(
        manifest["manifest_path"],
        expected_sha256=manifest["manifest_sha256"],
    )

    assert validation["ok"] is True
    role_sets = [
        set(manifest["train_roles"][role]["videos"]) for role in TRAIN_ROLES
    ]
    assert sum(len(values) for values in role_sets) == 20
    assert all(
        not role_sets[left] & role_sets[right]
        for left in range(len(role_sets))
        for right in range(left + 1, len(role_sets))
    )
    assert manifest["official_final_evaluation"][
        "consumed_during_method_development"
    ] is False

    final_path = tmp_path / "split" / "official_final_evaluation_videos.txt"
    final_path.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="official final artifact drifted"):
        validate_rime_splits(manifest["manifest_path"])


def test_budget_replay_rejects_requested_k_outside_frozen_grid():
    rows = [{"video_id": "v0", "window_start_frame": 0, "requested_k": 320}]

    with pytest.raises(ValueError, match="outside candidate_budgets"):
        paired_same_k(rows, candidate_budgets=(192, 384, 512))
    with pytest.raises(ValueError, match="outside candidate_budgets"):
        histogram_shuffle(
            rows,
            seed=1,
            candidate_budgets=(192, 384, 512),
        )


def test_phase0_reports_video_icc_mde_and_rule_derived_thresholds():
    rows = [
        {
            "schema_version": "duca_rime_phase0_measurement_v1",
            "video_id": f"v{video}",
            "replicate_id": replicate,
            "replicate_kind": "deterministic_reexecution",
            "metric_name": "avg_map",
            "value": video + 0.1 * replicate,
        }
        for video in range(4)
        for replicate in range(3)
    ]

    result = phase0_variance(
        rows,
        primary_metric="avg_map",
        alpha=0.05,
        power=0.8,
    )

    assert result["video_count"] == 4
    assert result["replicate_count"] == 12
    assert result["paired_video_mde"] > 0.0
    assert result["rule_derived_thresholds"]["min_o1_headroom"] > 0.0
    assert result["deterministic_reexecution_only"] is True
    assert result["independent_training_seed_variance_included"] is False


def test_o1_exact_mean_cost_oracle_beats_fixed_and_shuffled_assignment():
    gains = {
        "v0": (0.0, 5.0),
        "v1": (0.0, 4.0),
        "v2": (0.0, 0.1),
        "v3": (0.0, 0.0),
    }
    rows = []
    for video, scores in gains.items():
        for budget, score in zip((1, 2), scores):
            rows.append(
                {
                    "schema_version": "duca_rime_o1_budget_panel_v1",
                    "video_id": video,
                    "budget": budget,
                    "cost": float(budget),
                    "score": score,
                }
            )

    result = analyze_o1(
        rows,
        target_mean_cost=1.5,
        min_headroom=0.0,
        bootstrap_samples=200,
        shuffles=200,
        seed=3,
    )

    assert result["realized_oracle_mean_cost"] == 1.5
    assert result["oracle_score"] > result["best_fixed_score"]
    assert result["oracle_minus_best_fixed"]["mean"] > 0.0


def test_o2_decoder_panel_reports_nested_regret_and_overlap():
    rows = []
    for video in ("v0", "v1", "v2"):
        for budget, independent, nested in (
            (2, [1, 4], [1, 4]),
            (3, [1, 3, 5], [1, 4, 5]),
        ):
            for family, positions, score in (
                ("independent", independent, 1.0),
                ("strict_nested", nested, 0.99),
            ):
                rows.append(
                    {
                        "schema_version": "duca_rime_o2_decoder_panel_v1",
                        "video_id": video,
                        "budget": budget,
                        "family": family,
                        "score": score,
                        "selected_positions": positions,
                        "max_gap_violation": False,
                    }
                )

    result = analyze_o2(
        rows,
        selected_family="strict_nested",
        max_regret=0.02,
        bootstrap_samples=100,
        seed=5,
    )

    assert result["gate_pass"] is True
    assert result["independent_minus_selected_regret"]["mean"] < 0.02
    assert result["consecutive_budget_overlap"]["mean"] >= 0.5


def test_o3_cross_fit_rank_separates_learned_scores_from_reverse_null():
    rows = []
    videos = ["v0", "v1", "v2", "v3"]
    for video in videos:
        fit = [value for value in videos if value != video]
        provenance = {
            "cross_fitted": True,
            "uses_validation_or_test": False,
            "fit_video_ids": fit,
            "eval_video_ids": [video],
        }
        for index, actual in enumerate((-1.0, 0.0, 1.0, 2.0)):
            rows.extend(
                [
                    {
                        "schema_version": "duca_rime_o3_rank_record_v1",
                        "video_id": video,
                        "score_family": "learned",
                        "predicted_gain": actual,
                        "actual_gain": actual,
                        "scale": index,
                        "provenance": provenance,
                    },
                    {
                        "schema_version": "duca_rime_o3_rank_record_v1",
                        "video_id": video,
                        "score_family": "score_reverse",
                        "predicted_gain": -actual,
                        "actual_gain": actual,
                        "scale": index,
                        "provenance": provenance,
                    },
                ]
            )

    result = analyze_o3(
        rows,
        min_spearman=0.5,
        null_margin=0.0,
        bootstrap_samples=100,
        seed=9,
    )

    assert result["gate_pass"] is True
    assert result["families"]["learned"]["spearman"]["mean"] == 1.0
    assert result["families"]["score_reverse"]["spearman"]["mean"] == -1.0


def test_o4_calibration_validates_no_padding_and_fallback_ledger():
    rows = []
    for index, (score, label) in enumerate(
        ((0.05, 0), (0.10, 0), (0.85, 1), (0.90, 1))
    ):
        rows.append(
            {
                "schema_version": "duca_rime_o4_risk_record_v1",
                "video_id": f"v{index}",
                "predicted_risk": score,
                "observed_pair_failure": label,
                "requested_k": 2,
                "effective_k": 2,
                "unique_k": 2,
                "backbone_input_k": 2,
                "padded_k": 2,
                "risk_fallback": score > 0.8,
                "provenance": {
                    "fit_split": "train_only",
                    "uses_validation_or_test": False,
                },
            }
        )

    result = analyze_o4(
        rows,
        risk_threshold=0.2,
        max_brier=0.05,
        max_ece=0.2,
        min_coverage=0.4,
        max_low_risk_failure=0.0,
        calibration_bins=5,
    )

    assert result["gate_pass"] is True
    assert result["no_padding_ledger"] is True
    assert result["low_risk_observed_failure"] == 0.0

    with pytest.raises(ValueError, match="bins"):
        analyze_o4(
            rows,
            risk_threshold=0.2,
            max_brier=0.05,
            max_ece=0.2,
            min_coverage=0.4,
            max_low_risk_failure=0.0,
            calibration_bins=0,
        )
