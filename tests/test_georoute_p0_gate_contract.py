from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata.run_georoute_p0_gate import (
    GEOROUTE_P0_GATE_SCHEMA,
    build_p0_gate_report,
    validate_p0_gate_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_payload() -> dict:
    return {
        "schema_version": GEOROUTE_P0_GATE_SCHEMA,
        "status": "PASS",
        "official_test_opened": False,
        "heavy_backbone_forward_count": 1,
        "shared_backbone_instances": 1,
        "uses_grid_sample": False,
        "uses_resized_local_crop": False,
        "exact_k": {"target_k": 16, "observed_min": 16, "observed_max": 16, "duplicates": 0},
        "estimator": {"name": "straight_through", "claim": "biased_straight_through"},
        "memory": {"peak_allocated_bytes": 4096, "peak_reserved_bytes": 8192},
        "losses": {"cost": 1.0},
        "gradient": {"all_trainable_gradients_finite": True, "nonzero_components": ["scout", "rpn_head"]},
        "detector": {"training_forward": True, "backward_completed": True, "output_length": 768},
        "p0_scope": {"synthetic_inputs_only": True, "full_training": False, "official_evaluation": False},
    }


def test_p0_report_builder_and_validator_preserve_the_gate_contract():
    report = build_p0_gate_report(_valid_payload())

    validate_p0_gate_report(report)
    assert report["schema_version"] == GEOROUTE_P0_GATE_SCHEMA
    assert report["status"] == "PASS"
    assert len(report["report_sha256"]) == 64
    assert report["p0_scope"]["official_evaluation"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("heavy_backbone_forward_count",), 2, "one heavy"),
        (("uses_grid_sample",), True, "grid_sample"),
        (("exact_k", "duplicates"), 1, "duplicate"),
        (("p0_scope", "official_evaluation"), True, "official"),
        (("estimator", "claim"), "unbiased_st", "estimator"),
    ],
)
def test_p0_report_validator_fails_closed_for_invalid_core_claims(path, value, message):
    report = build_p0_gate_report(_valid_payload())
    holder = report
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    report = build_p0_gate_report(report)

    with pytest.raises(ValueError, match=message):
        validate_p0_gate_report(report)


def test_p0_tool_is_statically_bound_to_georoute_and_never_to_official_test_or_second_heavy_model():
    source = (ROOT / "tools" / "bata" / "run_georoute_p0_gate.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "opentad" / "models" / "backbones" / "georoute_wrapper.py").read_text(encoding="utf-8")
    detector = (ROOT / "opentad" / "models" / "detectors" / "actionformer.py").read_text(encoding="utf-8")

    assert "georoute_native_packed_v1" in source
    assert "official_test_opened" in source
    assert source.count("forward_train(") == 1
    assert "forward_test(" not in source
    assert "F.grid_sample(" not in wrapper
    assert wrapper.count("forward_native_packed(") == 1
    assert "uses_resized_local_crop\": False" in wrapper
    assert "consume_detector_policy_loss" in detector
    assert "detector_losses=loc_losses" in detector


def test_p0_report_is_json_round_trip_stable():
    report = build_p0_gate_report(_valid_payload())
    restored = json.loads(json.dumps(report, sort_keys=True))
    validate_p0_gate_report(restored)
