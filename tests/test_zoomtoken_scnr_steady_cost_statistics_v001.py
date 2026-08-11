import copy
import json
from pathlib import Path

import pytest

from tools.bata.zoomtoken_scnr_steady_cost_contract_v001 import (
    LEAF_ORDERS,
    NONINFERIORITY_RATIO,
    WINDOW_BUDGET,
    analyze_complete_leaves,
    analyze_complete_leaves_with_draws,
    canonical_sha256,
    leaf_sequence,
    read_json_object,
    validate_warmup_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = read_json_object(
    ROOT / "research-wiki/experiments/zoomtoken_scnr_steady_cost_population_v001.json",
    label="population",
)


def _complete_leaves(centered_latency_ratio, centered_energy_ratio=None):
    if centered_energy_ratio is None:
        centered_energy_ratio = centered_latency_ratio
    leaves = {}
    for leaf_id in LEAF_ORDERS:
        rows = []
        for pass_index, arm in enumerate(leaf_sequence(leaf_id)):
            latency_ratio = (
                centered_latency_ratio if arm == "residual_window_center" else 1.0
            )
            energy_ratio = (
                centered_energy_ratio if arm == "residual_window_center" else 1.0
            )
            for window in MANIFEST["windows"]:
                ordinal = window["ordinal"]
                row = {
                    "leaf_id": leaf_id,
                    "pass_index": pass_index,
                    "sample_ordinal": ordinal,
                    "loader_ordinal": ordinal,
                    "arm": arm,
                    "video_id": window["video_id"],
                    "physical_window_id": window["physical_window_id"],
                    "window_id": f"{window['physical_window_id']}#{ordinal}",
                    "measurement_phase": "measured",
                    "warmup": False,
                    "exact_window_budget": WINDOW_BUDGET,
                    "selected_physical_tokens": WINDOW_BUDGET,
                    "executed_physical_tokens": WINDOW_BUDGET,
                    "duplicate_selected_physical_tokens": 0,
                    "padded_heavy_tokens": 0,
                    "route_audit": {
                        "physical_indices_sha256": "0" * 64,
                        "k_t_min": 0,
                        "k_t_max": 128,
                        "k_t_zero_count": 1,
                        "role_counts": {"heavy": WINDOW_BUDGET},
                        "attention_pairs": WINDOW_BUDGET,
                        "clip_token_counts": [WINDOW_BUDGET],
                        "exact_window_budget": WINDOW_BUDGET,
                        "padded_heavy_tokens": 0,
                        "branch_calibration_mode": (
                            "none" if arm == "none_control" else "residual_window_center"
                        ),
                    },
                    "input_pipeline_serial_ms": 0.1,
                    "h2d_ms": 0.1,
                    "decode_to_window_output_wall_ms": 100.0 * latency_ratio,
                    "model_forward_ms": 60.0 * latency_ratio,
                    "postprocess_ms": 0.1,
                    "final_video_nms_ms": 0.1,
                    "end_to_end_serial_ms": 100.0 * latency_ratio,
                    "peak_gpu_allocated_mb": 1.0,
                    "peak_gpu_reserved_mb": 1.0,
                    "gross_gpu_energy_j_per_sample": 10.0 * energy_ratio,
                }
                row["sample_sha256"] = canonical_sha256(row)
                rows.append(row)
        leaves[leaf_id] = rows
    return leaves


def test_known_answer_cost_noninferiority_passes():
    leaves = _complete_leaves(1.04)
    result = analyze_complete_leaves(leaves)
    repeated = analyze_complete_leaves(leaves)
    assert canonical_sha256(result) == canonical_sha256(repeated)
    assert result["candidate_decision"] == "PASS_COST_NONINFERIOR"
    assert result["pass_pair_count"] == 16
    assert result["video_cluster_count"] == 40
    assert result["bootstrap_replicates"] == 10_000
    assert result["metrics"]["end_to_end_p50"]["centered_over_control_ratio"] == pytest.approx(1.04)
    assert result["metrics"]["gross_gpu_energy_per_sample"][
        "centered_over_control_ratio"
    ] == pytest.approx(1.04)
    assert all(
        metric["percentile_95_ci"] == pytest.approx([1.04, 1.04])
        for metric in result["metrics"].values()
    )
    assert all(
        metric["upper_bound_le_1_05"] for metric in result["metrics"].values()
    )


def test_known_answer_cost_noninferiority_fails_without_changing_estimator():
    result = analyze_complete_leaves(_complete_leaves(1.06, 1.04))
    assert result["candidate_decision"] == "FAIL_COST_NONINFERIOR"
    assert result["metrics"]["end_to_end_p50"]["percentile_95_ci"] == pytest.approx(
        [1.06, 1.06]
    )
    assert result["metrics"]["gross_gpu_energy_per_sample"][
        "percentile_95_ci"
    ] == pytest.approx([1.04, 1.04])


def test_exact_threshold_equality_passes():
    result = analyze_complete_leaves(
        _complete_leaves(NONINFERIORITY_RATIO)
    )
    assert result["candidate_decision"] == "PASS_COST_NONINFERIOR"
    assert all(
        metric["percentile_95_ci"][1] == pytest.approx(NONINFERIORITY_RATIO)
        for metric in result["metrics"].values()
    )


def test_missing_leaf_and_warmup_contamination_fail_closed():
    leaves = _complete_leaves(1.0)
    leaves.pop("L08")
    with pytest.raises(ValueError):
        analyze_complete_leaves(leaves, bootstrap_replicates=8)
    leaves = _complete_leaves(1.0)
    contaminated = copy.deepcopy(leaves)
    contaminated["L01"][0]["measurement_phase"] = "warmup"
    contaminated["L01"][0]["warmup"] = True
    with pytest.raises(ValueError):
        analyze_complete_leaves(contaminated, bootstrap_replicates=8)


def test_duplicate_canonical_window_fails_before_bootstrap():
    leaves = _complete_leaves(1.0)
    duplicated = copy.deepcopy(leaves)
    for pass_index in range(4):
        row = duplicated["L08"][pass_index * 136 + 1]
        source = duplicated["L08"][pass_index * 136]
        row["physical_window_id"] = source["physical_window_id"]
        row["video_id"] = source["video_id"]
        row["sample_sha256"] = canonical_sha256(
            {key: value for key, value in row.items() if key != "sample_sha256"}
        )
    with pytest.raises(ValueError):
        analyze_complete_leaves(duplicated, bootstrap_replicates=8)


def _warmup_rows(leaf_id="L01"):
    rows = []
    for pass_index, arm in enumerate(leaf_sequence(leaf_id)):
        for window in MANIFEST["windows"]:
            ordinal = window["ordinal"]
            physical_id = window["physical_window_id"]
            rows.append(
                {
                    "schema_version": "zoomtoken_scnr_steady_cost_warmup_identity_v001",
                    "leaf_id": leaf_id,
                    "pass_index": pass_index,
                    "arm": arm,
                    "measurement_phase": "warmup",
                    "warmup": True,
                    "warmup_ordinal": ordinal,
                    "loader_ordinal": ordinal,
                    "video_id": window["video_id"],
                    "physical_window_id": physical_id,
                    "window_id": f"{physical_id}#{ordinal}",
                }
            )
    return rows


def test_warmup_identity_ledger_is_complete_ordered_and_identity_only():
    rows = _warmup_rows()
    assert len(
        validate_warmup_ledger(
            rows,
            leaf_id="L01",
            sequence=leaf_sequence("L01"),
            population=MANIFEST,
        )
    ) == 4 * 136
    reordered = copy.deepcopy(rows)
    reordered[0]["physical_window_id"] = reordered[1]["physical_window_id"]
    with pytest.raises(ValueError):
        validate_warmup_ledger(
            reordered,
            leaf_id="L01",
            sequence=leaf_sequence("L01"),
            population=MANIFEST,
        )
    contaminated = copy.deepcopy(rows)
    contaminated[0]["model_forward_ms"] = 1.0
    with pytest.raises(ValueError):
        validate_warmup_ledger(
            contaminated,
            leaf_id="L01",
            sequence=leaf_sequence("L01"),
            population=MANIFEST,
        )


def test_all_ten_thousand_bootstrap_draws_recompute_byte_identically():
    leaves = _complete_leaves(1.04)
    analysis, draws = analyze_complete_leaves_with_draws(leaves)
    repeated_analysis, repeated_draws = analyze_complete_leaves_with_draws(leaves)
    assert analysis == repeated_analysis
    assert len(draws) == 10_000
    assert json.dumps(draws, sort_keys=True, separators=(",", ":")) == json.dumps(
        repeated_draws, sort_keys=True, separators=(",", ":")
    )


def test_literal_threshold_rejects_positive_epsilon():
    result = analyze_complete_leaves(
        _complete_leaves(NONINFERIORITY_RATIO + 1e-10), bootstrap_replicates=8
    )
    assert result["candidate_decision"] == "FAIL_COST_NONINFERIOR"
    assert all(
        metric["upper_bound_le_1_05"] is False
        for metric in result["metrics"].values()
    )
