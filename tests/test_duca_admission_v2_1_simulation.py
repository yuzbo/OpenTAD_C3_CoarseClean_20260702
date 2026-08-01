from __future__ import annotations

import copy

import pytest

from tools.bata.duca_admission_v2_1_incidence import build_incidence
from tools.bata.duca_admission_v2_1_roles import build_role_manifest
from tools.bata.duca_admission_v2_1_simulation import (
    MC_PARAMETER_IDS,
    build_outer_reference_summary,
    clopper_pearson_onesided_99,
    evaluate_mc_calibration_registry,
    evaluate_mc_calibration_scenario,
    evaluate_simulation_scenario,
    generate_outer_cells,
    load_simulation_registry,
    shift_truth,
    true_contrast_variance,
    validate_mc_calibration_registry_receipt,
    validate_mc_calibration_scenario_receipt,
    validate_simulation_scenario_receipt,
    validate_simulation_registry,
    verify_reference_environment,
)
from tools.bata.duca_evidence_io import with_content_sha256


def make_inventory():
    return [
        {
            "video_id": f"long_{index:03d}",
            "source_subset": "training",
            "frame_count": (900 + index) * 4,
            "snippet_count": 900 + index,
            "natural_window_valid_lengths": [768],
        }
        for index in range(70)
    ] + [
        {
            "video_id": f"short_{index:03d}",
            "source_subset": "training",
            "frame_count": (100 + 20 * index) * 4,
            "snippet_count": 100 + 20 * index,
            "natural_window_valid_lengths": [100 + 20 * index],
        }
        for index in range(30)
    ]


def test_registry_is_fully_enumerated_and_reference_rng_is_bound():
    registry = load_simulation_registry()
    assert len(registry["scenarios"]) == 52
    assert len(registry["mc_calibration_scenarios"]) == 24
    assert registry["execution"]["outer_datasets_per_scenario"] == 500
    assert registry["execution"]["initial_inner_replicates"] == 100_000
    observed = verify_reference_environment(registry)
    assert observed["golden_raw_byte_count"] == 4096


def test_outer_generation_is_deterministic_and_exact_zero_is_positive_zero():
    registry = load_simulation_registry()
    roles = build_role_manifest(
        inventory_records=make_inventory(), source_split_artifact_sha256="a" * 64
    )
    incidence = build_incidence(roles)
    scenario = registry["scenarios"][50]
    first = generate_outer_cells(
        registry=registry, scenario=scenario, incidence=incidence, outer_index=0
    )
    second = generate_outer_cells(
        registry=registry, scenario=scenario, incidence=incidence, outer_index=0
    )
    assert first == second
    assert len(first) == 192
    assert all(
        value == 0.0 and str(value) != "-0.0"
        for row in first
        for key, value in row.items()
        if key.startswith("M")
    )
    summary = build_outer_reference_summary(
        registry=registry, scenario=scenario, incidence=incidence, outer_index=0
    )
    assert summary["status"] == "PASSED"
    assert summary["authorization_scope"] == "NONE"
    assert summary["paper_claim_allowed"] is False


def test_simulation_registry_rejects_nested_or_scenario_drift():
    registry = load_simulation_registry()
    nested = copy.deepcopy(registry)
    nested["effect_mixes"]["ROW"]["unknown"] = 1
    nested.pop("artifact_sha256")
    nested.pop("semantic_sha256")
    with pytest.raises(ValueError, match="effect-mix registry drift"):
        validate_simulation_registry(nested)
    scenario = copy.deepcopy(registry)
    scenario["scenarios"][0]["rho"] = 0.36
    scenario.pop("artifact_sha256")
    scenario.pop("semantic_sha256")
    with pytest.raises(ValueError, match="scenario drift"):
        validate_simulation_registry(scenario)


def _endpoint(observed, scales, *, numeric_tail_passed):
    q_plus = 2.0
    q_minus = 2.0
    return {
        "numeric_tail_passed": numeric_tail_passed,
        "q_plus": q_plus,
        "q_minus": q_minus,
        "scales": dict(scales),
        "lower": {
            metric_id: float(observed[metric_id] - q_plus * scales[metric_id])
            for metric_id in scales
        },
        "upper": {
            metric_id: float(observed[metric_id] + q_minus * scales[metric_id])
            for metric_id in scales
        },
        "mc_status": "PASSED",
        "replicate_count": 100_000,
        "secondary_diagnostic": {
            "replicate_count": 100_000,
            "passed": True,
            "binary_pass_vector": [True],
        },
    }


def test_full_scenario_gate_applies_coverage_width_power_and_integer_counts():
    registry = load_simulation_registry()
    roles = build_role_manifest(
        inventory_records=make_inventory(), source_split_artifact_sha256="a" * 64
    )
    incidence = build_incidence(roles)
    scenario = registry["scenarios"][0]
    variance = true_contrast_variance(
        registry=registry, scenario=scenario, incidence=incidence
    )
    scales = {metric_id: float(value**0.5) for metric_id, value in variance.items()}
    truths = {
        profile_id: shift_truth(
            profile_id=profile_id, true_variance=variance, registry=registry
        )
        for profile_id in registry["shift_profiles"]
    }
    endpoints = {
        profile_id: _endpoint(
            truth,
            scales,
            numeric_tail_passed=profile_id in {"NULL", "SAFE_ALL_M6"},
        )
        for profile_id, truth in truths.items()
    }
    receipt = evaluate_simulation_scenario(
        registry=registry,
        scenario=scenario,
        incidence=incidence,
        outer_records=[
            {"outer_index": index, "endpoints": copy.deepcopy(endpoints)}
            for index in range(500)
        ],
    )
    assert receipt["status"] == "PASSED"
    assert receipt["authorization_scope"] == "NONE"
    assert all(row["passed"] for row in receipt["checks"])
    validate_simulation_scenario_receipt(receipt, registry=registry)
    tampered = copy.deepcopy(receipt)
    tampered["checks"][0]["limit"] -= 1
    tampered = with_content_sha256(tampered)
    with pytest.raises(ValueError, match="check drifted"):
        validate_simulation_scenario_receipt(tampered, registry=registry)
    intervals = clopper_pearson_onesided_99(500, 500)
    assert 0.0 < intervals["lower_99"] < 1.0
    assert intervals["upper_99"] == 1.0


def test_exact_zero_and_mc_half_width_scenario_gates():
    registry = load_simulation_registry()
    roles = build_role_manifest(
        inventory_records=make_inventory(), source_split_artifact_sha256="a" * 64
    )
    incidence = build_incidence(roles)
    zero_map = {metric_id: 0.0 for metric_id in registry["metric_ids"]}
    exact_endpoint = {
        "numeric_tail_passed": True,
        "q_plus": 0.0,
        "q_minus": 0.0,
        "scales": dict(zero_map),
        "lower": dict(zero_map),
        "upper": dict(zero_map),
        "mc_status": "PASSED_EXACT_ZERO",
        "replicate_count": 0,
        "secondary_diagnostic": None,
    }
    exact = evaluate_simulation_scenario(
        registry=registry,
        scenario=registry["scenarios"][50],
        incidence=incidence,
        outer_records=[
            {"outer_index": index, "endpoints": {"NULL": dict(exact_endpoint)}}
            for index in range(500)
        ],
    )
    assert exact["status"] == "PASSED"
    validate_simulation_scenario_receipt(exact, registry=registry)

    estimates = {parameter_id: 0.0 for parameter_id in MC_PARAMETER_IDS}
    h100 = {parameter_id: 0.2 for parameter_id in MC_PARAMETER_IDS}
    h200 = {parameter_id: 0.1 for parameter_id in MC_PARAMETER_IDS}
    mc_receipt = evaluate_mc_calibration_scenario(
        registry=registry,
        scenario_id=registry["mc_calibration_scenarios"][0],
        operational_streams=[
            {
                "stream_index": index,
                "estimate_100k": dict(estimates),
                "half_width_100k": dict(h100),
                "estimate_200k": dict(estimates),
                "half_width_200k": dict(h200),
            }
            for index in range(200)
        ],
        reference={
            "combined_4m": dict(estimates),
            "half0_2m": dict(estimates),
            "half1_2m": dict(estimates),
        },
    )
    assert mc_receipt["status"] == "PASSED"
    assert all(value == 200 for value in mc_receipt["coverage_counts"].values())
    validate_mc_calibration_scenario_receipt(mc_receipt, registry=registry)

    scenario_receipts = []
    for scenario_id in registry["mc_calibration_scenarios"]:
        scenario_receipts.append(
            evaluate_mc_calibration_scenario(
                registry=registry,
                scenario_id=scenario_id,
                operational_streams=[
                    {
                        "stream_index": index,
                        "estimate_100k": dict(estimates),
                        "half_width_100k": dict(h100),
                        "estimate_200k": dict(estimates),
                        "half_width_200k": dict(h200),
                    }
                    for index in range(200)
                ],
                reference={
                    "combined_4m": dict(estimates),
                    "half0_2m": dict(estimates),
                    "half1_2m": dict(estimates),
                },
            )
        )
    aggregate = evaluate_mc_calibration_registry(
        registry=registry, scenario_receipts=scenario_receipts
    )
    validate_mc_calibration_registry_receipt(aggregate, registry=registry)
    assert aggregate["status"] == "PASSED"
