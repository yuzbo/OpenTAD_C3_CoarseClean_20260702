from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import analyze_duca_selection_quality as quality
from tools.bata import export_duca_selection_quality as exporter


def _record(sample_id: str, learned_positions: list[int]) -> dict:
    return {
        "schema_version": quality.RECORD_SCHEMA_VERSION,
        "sample_id": sample_id,
        "video_id": sample_id.split("|")[0],
        "valid_len": 8,
        "requested_budget": 4,
        "effective_budget": 4,
        "budget": 4,
        "max_hole_contract": {"requested_max_unselected_hole": 2},
        "gt_segments": [[2.0, 6.0]],
        "gt_boundary_validity": [[True, True]],
        "p_action": [0.05, 0.10, 0.80, 0.90, 0.85, 0.75, 0.10, 0.05],
        "transition_policy_scores": [0.0, 0.2, 1.0, 0.1, 0.1, 0.9, 0.2, 0.0],
        "abs_delta_p_action": [0.0, 0.05, 0.70, 0.10, 0.05, 0.65, 0.10, 0.05],
        "raw_transition_scores": [0.0, 0.1, 0.8, 0.2, 0.1, 0.7, 0.1, 0.0],
        "selected_positions": learned_positions,
    }


def test_exact_uniform_reference_matches_round_linspace_endpoints() -> None:
    assert quality.exact_uniform_positions(valid_len=8, budget=4) == [0, 2, 5, 7]
    assert quality.exact_uniform_positions(valid_len=3, budget=8) == [0, 1, 2]


def test_exporter_extracts_only_frame_selector_state_from_ddp_checkpoint() -> None:
    state = {
        "module.frame_selector.probe.weight": 1,
        "module.frame_selector.adapter.bias": 2,
        "module.backbone.weight": 3,
    }

    assert exporter.selector_state_dict(state) == {"probe.weight": 1, "adapter.bias": 2}


def test_exporter_finds_repository_from_config_tree_not_deployment_script(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    config_dir = repo / "configs" / "method"
    config_dir.mkdir(parents=True)
    (repo / ".git").write_text("gitdir: somewhere\n", encoding="utf-8")

    assert exporter._find_git_root(config_dir) == repo


def test_export_cli_can_separate_dataset_and_selector_configs() -> None:
    source = Path(exporter.__file__).read_text(encoding="utf-8")
    assert '"--selector-config"' in source
    assert "build_selector(selector_cfg.model.frame_selector)" in source


def test_exporter_records_existing_decoder_repair_metadata() -> None:
    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def long(self):
            return self

        def float(self):
            return self

        def sum(self):
            def flatten(value):
                return sum((flatten(item) for item in value), 0) if isinstance(value, list) else value
            return FakeTensor(flatten(self.value))

        def item(self):
            return self.value

        def tolist(self):
            return self.value

        def __getitem__(self, index):
            return FakeTensor(self.value[index])

    class Grid:
        selected_positions = FakeTensor([[0, 2, 5, 7]])
        metadata = {"max_gap_repair": [{"enabled": True, "repair_count": 2, "feasible": True, "satisfied": True}]}

    output = {
        "selector_outputs": {
            "grid": Grid(),
            "p_action": FakeTensor([[0.0] * 8]),
            "actionness_logits": FakeTensor([[0.0] * 8]),
            "transition_policy_scores": FakeTensor([[0.0] * 8]),
            "transition_score": FakeTensor([[0.0] * 8]),
            "abs_delta_p_action": FakeTensor([[0.0] * 8]),
            "uncertainty": FakeTensor([[0.0] * 8]),
        }
    }
    records = exporter._records_from_batch(
        selector_output=output,
        masks=FakeTensor([[1] * 8]),
        gt_segments=[[[2.0, 6.0]]],
        metas=[{"video_name": "v"}],
        source={},
        seen_count=0,
        requested_budget=4,
        requested_max_unselected_hole=2,
    )
    assert records[0]["decode_diagnostics"]["repair_count"] == 2
    assert records[0]["decode_diagnostics"]["repair_enabled"] is True
    assert records[0]["requested_budget"] == 4
    assert records[0]["effective_budget"] == 4
    assert records[0]["max_hole_contract"]["requested_max_unselected_hole"] == 2


def test_exporter_reads_budget_and_max_hole_independently_from_selector_config() -> None:
    class Selector:
        budget = 384
        max_unselected_hole = 2

    assert exporter._selector_sampling_contract(
        Selector(), {"budget": 384, "max_unselected_hole": 2}
    ) == (384, 2)
    with pytest.raises(ValueError, match="budget drift"):
        exporter._selector_sampling_contract(
            Selector(), {"budget": 256, "max_unselected_hole": 2}
        )


def test_binary_metrics_cover_discrimination_calibration_and_prevalence() -> None:
    metrics = quality.binary_metrics(
        labels=[0, 0, 1, 1],
        scores=[0.1, 0.2, 0.8, 0.9],
        calibration_bins=2,
    )

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["prevalence"] == pytest.approx(0.5)
    assert metrics["auprc_lift"] == pytest.approx(2.0)
    assert metrics["brier"] == pytest.approx(0.025)
    assert metrics["ece"] == pytest.approx(0.15)


def test_uncalibrated_utility_keeps_raw_ranking_and_omits_probability_metrics() -> None:
    metrics = quality.binary_metrics(
        labels=[1, 0, 1],
        scores=[10.0, 9.0, 8.0],
        calibrated=False,
    )

    assert metrics["auprc"] == pytest.approx(5.0 / 6.0)
    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["brier"] is None
    assert metrics["ece"] is None


def test_half_open_segment_end_equal_to_valid_len_keeps_last_frame_positive() -> None:
    segments = quality._validated_segments({"gt_segments": [[6.0, 8.0]]}, valid_len=8)

    assert quality._action_labels(8, segments) == [0, 0, 0, 0, 0, 0, 1, 1]
    assert quality._boundaries(8, segments) == [6.0, 7.0]


def test_endpoint_events_match_training_floor_start_ceil_end_minus_one() -> None:
    integer = quality._validated_segments({"gt_segments": [[2.0, 6.0]]}, valid_len=8)
    fractional = quality._validated_segments(
        {"gt_segments": [[2.2, 5.2], [2.8, 5.01]]}, valid_len=8
    )

    assert quality._segment_endpoints(8, integer) == [(2, 5)]
    assert quality._boundaries(8, integer) == [2, 5]
    assert quality._segment_endpoints(8, fractional) == [(2, 5), (2, 5)]
    assert quality._boundaries(8, fractional) == [2, 5]


def test_cropped_endpoint_validity_excludes_window_cut_from_boundary_metrics() -> None:
    record = _record("video_crop|0", [0, 2, 5, 7])
    record["gt_boundary_validity"] = [[True, False]]
    row = quality.analyze_record(record)

    assert row["gt_boundary_count"] == 1
    assert row["selection"]["learned"]["both_endpoint_coverage"]["r0"] is None


def test_selection_coverage_clamps_half_open_end_to_last_observable_position() -> None:
    metrics = quality._selection_metrics(
        valid_len=8,
        positions=[0, 2, 6, 7],
        segments=[(6.0, 8.0)],
        boundaries=[6.0, 7.0],
    )

    assert metrics["both_endpoint_coverage"]["r0"] == pytest.approx(1.0)


def test_boundary_burst_metrics_distinguish_microclusters_from_uniform_hits() -> None:
    clustered = quality._selection_metrics(
        valid_len=20,
        positions=[2, 3, 4, 5, 6, 12, 13, 14, 15, 16],
        segments=[(4.0, 15.2)],
        boundaries=[4.0, 15.0],
    )
    sparse = quality._selection_metrics(
        valid_len=20,
        positions=[0, 2, 4, 7, 10, 13, 15, 17, 18, 19],
        segments=[(4.0, 15.2)],
        boundaries=[4.0, 15.0],
    )

    assert clustered["boundary_burst"]["r2q3"]["endpoint_quota_recall"] == pytest.approx(1.0)
    assert clustered["boundary_burst"]["r2q3"]["both_endpoints_quota_recall"] == pytest.approx(1.0)
    assert sparse["boundary_recall"]["r0"] == pytest.approx(1.0)
    assert sparse["boundary_burst"]["r2q3"]["both_endpoints_quota_recall"] == pytest.approx(0.0)


def test_sample_analysis_separates_coarse_transition_and_selection_quality() -> None:
    row = quality.analyze_record(_record("video_a|0", [1, 2, 5, 7]), random_seed=17)

    assert row["coarse"]["auroc"] == pytest.approx(1.0)
    assert row["transition"]["r0"]["policy"]["auprc"] == pytest.approx(1.0)
    assert row["transition"]["r0"]["pure_abs_delta_p_action"]["auprc"] == pytest.approx(1.0)
    assert row["transition"]["r0"]["raw_actionness_transition"]["auprc"] == pytest.approx(1.0)
    assert row["selection"]["learned"]["selected_count"] == 4
    assert row["selection"]["uniform"]["selected_positions"] == [0, 2, 5, 7]
    assert row["selection"]["learned"]["boundary_recall"]["r0"] == pytest.approx(1.0)
    assert row["selection"]["learned"]["max_unselected_hole"] == 2
    assert row["selection"]["utility_topk_diagnostic"]["boundary_recall"]["r0"] == pytest.approx(1.0)
    assert row["selection"]["pure_delta_topk_diagnostic"]["selected_positions"] != row["selection"]["raw_transition_topk_diagnostic"]["selected_positions"]


def test_simple_delta_structured_dp_enforces_exact_k_order_and_max_hole() -> None:
    cases = (
        ([12.0 - index for index in range(12)], 6, 2),
        ([0.0, 9.0, 8.0, 0.0, 0.0, 7.0, 6.0, 0.0], 4, 1),
    )
    decoded = []
    for scores, budget, max_hole in cases:
        positions = quality._global_structured_positions(
            scores,
            budget=budget,
            max_unselected_hole=max_hole,
        )
        decoded.append(positions)
        assert len(positions) == budget
        assert positions == sorted(positions)
        assert len(set(positions)) == budget
        assert quality._max_unselected_hole(len(scores), positions) <= max_hole

    assert decoded[0] != list(range(6))


def test_simple_delta_quality_can_truthfully_exceed_learned_policy() -> None:
    row = quality.analyze_record(_record("video_delta_wins|0", [0, 3, 4, 7]))
    learned = row["selection"]["learned"]
    simple_delta = row["selection"]["pure_delta_same_feasible_dp"]
    assert simple_delta == row["selection"]["pure_delta_topk_diagnostic"]

    assert simple_delta["selected_count"] == row["sampling_contract"]["effective_budget"]
    assert simple_delta["selected_positions"] == sorted(set(simple_delta["selected_positions"]))
    assert simple_delta["max_unselected_hole"] <= row["sampling_contract"]["requested_max_unselected_hole"]
    assert simple_delta["boundary_recall"]["r0"] > learned["boundary_recall"]["r0"]
    assert simple_delta["mean_endpoint_distance"] < learned["mean_endpoint_distance"]


def test_sample_analysis_fails_closed_on_budget_or_max_hole_violation() -> None:
    wrong_budget = _record("video_a|0", [1, 2, 5])
    with pytest.raises(ValueError, match="effective_budget"):
        quality.analyze_record(wrong_budget)

    excessive_hole = _record("video_b|0", [0, 1, 2, 7])
    with pytest.raises(ValueError, match="exceeds requested G"):
        quality.analyze_record(excessive_hole)


def test_short_window_uses_min_requested_budget_and_valid_length() -> None:
    record = _record("video_short|0", [0, 1, 2])
    record["valid_len"] = 3
    record["requested_budget"] = 4
    record["effective_budget"] = 3
    for key in (
        "p_action",
        "transition_policy_scores",
        "abs_delta_p_action",
        "raw_transition_scores",
    ):
        record[key] = record[key][:3]
    record["gt_segments"] = [[0.0, 3.0]]

    row = quality.analyze_record(record)

    assert row["sampling_contract"]["requested_budget"] == 4
    assert row["sampling_contract"]["effective_budget"] == 3
    assert row["sampling_contract"]["selected_count"] == 3


def test_representative_samples_are_best_median_and_worst_without_manual_choice() -> None:
    rows = []
    for idx, gain in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0]):
        rows.append({"sample_id": f"v{idx}|0", "selection_gain_vs_uniform": gain})

    chosen = quality.choose_representative_samples(rows, per_stratum=1)

    assert [row["sample_id"] for row in chosen] == ["v4|0", "v2|0", "v0|0"]
    assert [row["sample_stratum"] for row in chosen] == ["best", "median", "worst"]


def test_bootstrap_resamples_video_clusters_instead_of_treating_windows_as_independent() -> None:
    interval = quality._bootstrap_ci(
        [0.0, 0.0, 1.0],
        clusters=["video_a", "video_a", "video_b"],
        samples=64,
        seed=3,
    )

    assert interval["n"] == 3
    assert interval["cluster_n"] == 2
    assert interval["mean"] == pytest.approx(1.0 / 3.0)


def test_analyze_jsonl_writes_machine_readable_outputs_and_vector_figure(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    with records.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_record("video_a|0", [1, 2, 5, 7])) + "\n")
        handle.write(json.dumps(_record("video_b|0", [0, 2, 4, 6])) + "\n")

    summary = quality.analyze_jsonl(
        records_jsonl=records,
        output_dir=tmp_path / "out",
        bootstrap_samples=32,
        random_seed=5,
        representative_per_stratum=1,
    )

    assert summary["sample_count"] == 2
    evidence = summary["protocol"]["sampling_contract_evidence"]
    assert evidence["requested_budget_min"] == 4
    assert evidence["requested_budget_max"] == 4
    assert evidence["effective_budget_min"] == 4
    assert evidence["selected_count_min"] == 4
    assert evidence["budget_violation_count"] == 0
    assert evidence["observed_max_unselected_hole_max"] <= 2
    assert evidence["max_hole_violation_count"] == 0
    assert "pure_abs_delta_p_action" in summary["transition"]["r0"]
    assert (tmp_path / "out" / "selection_quality_summary.json").is_file()
    assert (tmp_path / "out" / "selection_quality_per_sample.csv").is_file()
    assert (tmp_path / "out" / "selection_quality_overview.pdf").is_file()
    assert (tmp_path / "out" / "selection_quality_samples.pdf").is_file()


def test_analyze_jsonl_pooled_metrics_exclude_crop_cut_boundary(tmp_path: Path) -> None:
    record = _record("video_crop|0", [0, 2, 5, 7])
    record["gt_segments"] = [[0.0, 2.0]]
    record["gt_boundary_validity"] = [[False, True]]
    record["transition_policy_scores"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sample = quality.analyze_record(record)
    records = tmp_path / "crop_records.jsonl"
    records.write_text(json.dumps(record) + "\n", encoding="utf-8")

    summary = quality.analyze_jsonl(
        records_jsonl=records,
        output_dir=tmp_path / "crop_out",
        bootstrap_samples=8,
        random_seed=7,
        representative_per_stratum=0,
    )

    sample_transition = sample["transition"]["r0"]["policy"]
    pooled_transition = summary["transition"]["r0"]["policy"]
    assert sample["gt_boundary_count"] == 1
    assert sample_transition["positive_count"] == 1
    assert sample_transition["prevalence"] == pytest.approx(1.0 / 8.0)
    assert sample_transition["auprc"] == pytest.approx(1.0)
    assert pooled_transition == sample_transition
    assert summary["selection"]["learned"]["pooled"]["boundary_recall"]["r0"] == pytest.approx(
        sample["selection"]["learned"]["boundary_recall"]["r0"]
    )
    for field in (
        "mean_endpoint_selected_count",
        "endpoint_quota_recall",
        "endpoint_bilateral_recall",
        "both_endpoints_quota_recall",
    ):
        sample_value = sample["selection"]["learned"]["boundary_burst"]["r2q3"][field]
        pooled_value = summary["selection"]["learned"]["boundary_burst"]["r2q3"][field]["mean"]
        if sample_value is None:
            assert pooled_value is None
        else:
            assert pooled_value == pytest.approx(sample_value)
