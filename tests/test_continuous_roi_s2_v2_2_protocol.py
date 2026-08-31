from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs" / "methods" / "continuous_roi_s2_v2_2_protocol.json"


def load_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def resign(payload):
    from tools.validate_continuous_roi_s2_v2_2_protocol import protocol_core_sha256

    payload["declared_protocol_sha256"] = protocol_core_sha256(payload)
    return payload


def require_frozen_torch():
    if sys.platform == "win32":
        pytest.skip("frozen Sobol known answer runs in the required Linux closure")
    try:
        import torch
    except Exception as error:  # Windows user-site Torch may not load c10.dll.
        pytest.skip(f"frozen Torch runtime unavailable on this host: {error}")
    if torch.__version__ != "2.0.1":
        pytest.skip(f"known answer requires frozen Torch 2.0.1, got {torch.__version__}")


def test_protocol_self_hash_scope_and_exact_nine_are_frozen():
    from tools.validate_continuous_roi_s2_v2_2_protocol import validate_protocol

    payload = load_protocol()
    audit = validate_protocol(payload)
    assert audit["static_protocol_valid"] is True
    assert audit["training_authorized"] is False
    assert audit["raw_inference_authorized"] is False
    assert audit["official_test_open_allowed"] is False
    assert audit["protocol_sha256"] == payload["declared_protocol_sha256"]
    cells = payload["frozen_training_identities"]["cells"]
    assert [(cell["family"], cell["seed"]) for cell in cells] == [
        (family, seed)
        for family in ("D160", "G96", "U128")
        for seed in (3407, 3408, 3409)
    ]
    assert [cell["job_id"] for cell in cells] == [str(job) for job in range(1177668, 1177677)]


def test_size_one_inverse_is_explicit_and_not_a_clamp():
    from tools.validate_continuous_roi_s2_v2_2_protocol import inverse_center_logit

    assert inverse_center_logit(0.5, 1.0) == 0.0
    with pytest.raises(ValueError, match="unique center"):
        inverse_center_logit(0.5001, 1.0)
    with pytest.raises(ValueError, match="strictly inside"):
        inverse_center_logit(0.2, 0.4)


def test_candidate_manifest_is_byte_reproducible_and_centers_match():
    require_frozen_torch()
    from tools.validate_continuous_roi_s2_v2_2_protocol import (
        build_candidate_manifest,
        canonical_json_bytes,
        canonical_sha256,
    )

    protocol = load_protocol()
    first = build_candidate_manifest(protocol)
    second = build_candidate_manifest(protocol)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_sha256(first) == protocol["candidate_generator"]["known_answer_manifest_sha256"]
    assert len(first["candidates"]) == 17
    assert [item["candidate_id"] for item in first["candidates"]] == [
        f"candidate-{index:03d}" for index in range(17)
    ]
    for candidate in first["candidates"]:
        assert len(candidate["tubelets"]) == 48
        for tubelet in candidate["tubelets"]:
            fixed = tuple(map(float, tubelet["fixed_size"]["box"]))
            variable = tuple(map(float, tubelet["variable_size"]["box"]))
            assert fixed[:2] == variable[:2] == tuple(map(float, tubelet["common_center"]))
            assert 0.0 < fixed[2] <= 1.0 and 0.0 < fixed[3] <= 1.0
            assert 0.0 < variable[2] <= 1.0 and 0.0 < variable[3] <= 1.0


def test_candidate_manifest_is_valid_for_the_raw_object_graph():
    require_frozen_torch()
    from tools.validate_continuous_roi_s2_v2_2_protocol import (
        assert_raw_object_graph_clean,
        build_candidate_manifest,
    )

    protocol = load_protocol()
    assert_raw_object_graph_clean(build_candidate_manifest(protocol), protocol)


def test_result_blind_candidate_id_is_allowed_but_preferred_id_is_forbidden():
    from tools.validate_continuous_roi_s2_v2_2_protocol import assert_raw_object_graph_clean

    protocol = load_protocol()
    assert_raw_object_graph_clean(
        {
            "candidate_id": "candidate-003",
            "video_id": "video_validation_0000056",
            "prediction_rows": [],
        },
        protocol,
    )
    with pytest.raises(ValueError, match="preferred_candidate"):
        assert_raw_object_graph_clean(
            {"preferred_candidate_id": "candidate-003"}, protocol
        )
    with pytest.raises(ValueError, match="annotation"):
        assert_raw_object_graph_clean({"annotation_path": "hidden.json"}, protocol)
    with pytest.raises(ValueError, match="target_cache"):
        assert_raw_object_graph_clean({"input": "target_cache.bin"}, protocol)


def test_privileged_join_rejects_post_seal_mutation():
    from tools.validate_continuous_roi_s2_v2_2_protocol import (
        privileged_join,
        seal_raw_receipt,
    )

    raw = {"candidate_id": "candidate-001", "prediction_rows": [[1, 2, 3]]}
    seal = seal_raw_receipt(raw)
    joined = privileged_join(raw, seal, {"window-000": "candidate-001"})
    assert joined["bound_raw_payload_sha256"] == seal["raw_payload_sha256"]
    assert joined["preferred_candidate_ids"] == {"window-000": "candidate-001"}
    mutated = copy.deepcopy(raw)
    mutated["prediction_rows"].append([4, 5, 6])
    with pytest.raises(ValueError, match="changed after"):
        privileged_join(mutated, seal, {"window-000": "candidate-001"})


def test_raw_population_builder_emits_only_sanitized_window_identity():
    from tools.validate_continuous_roi_s2_v2_2_protocol import (
        assert_raw_object_graph_clean,
        build_raw_population_manifest,
    )

    protocol = load_protocol()
    protocol["raw_reference_population"]["gate_video_ids"] = ["video-a"]
    development = {
        "database": {
            "video-a": {
                "subset": "training",
                "frame": 4000,
                "duration": 10.0,
                "annotations": [
                    {"label": "A", "segment": [0.0, 10.0]},
                    {"label": "Ambiguous", "segment": [0.0, 10.0]},
                ],
            }
        }
    }
    manifest = build_raw_population_manifest(protocol, development)
    assert manifest["entries"]
    assert set(manifest["entries"][0]) == {
        "ordinal",
        "video_id",
        "dataset_window_index",
        "feature_start_index",
        "feature_end_index_inclusive",
        "window_start_frame",
        "window_end_frame_inclusive",
        "valid_snippet_count",
    }
    assert_raw_object_graph_clean(manifest, protocol)


def test_protocol_rejects_preferred_id_in_raw_schema_even_when_resigned():
    from tools.validate_continuous_roi_s2_v2_2_protocol import validate_protocol

    payload = load_protocol()
    payload["privilege_boundary"]["raw_output_fields"].append("preferred_candidate_id")
    resign(payload)
    with pytest.raises(ValueError, match="preferred ID leaked"):
        validate_protocol(payload)


def test_protocol_rejects_generator_version_drift_even_when_resigned():
    from tools.validate_continuous_roi_s2_v2_2_protocol import validate_protocol

    payload = load_protocol()
    payload["candidate_generator"]["implementation"]["version"] = "2.0.0"
    resign(payload)
    with pytest.raises(ValueError, match="implementation changed"):
        validate_protocol(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (("job_id", "9999999"), ("job_name", "replacement-job")),
)
def test_protocol_rejects_exact_job_identity_drift_even_when_resigned(field, value):
    from tools.validate_continuous_roi_s2_v2_2_protocol import validate_protocol

    payload = load_protocol()
    payload["frozen_training_identities"]["cells"][0][field] = value
    resign(payload)
    with pytest.raises(ValueError, match="job identity changed"):
        validate_protocol(payload)


def test_result_blind_statistics_are_machine_frozen():
    protocol = load_protocol()
    statistics = protocol["statistics"]
    assert statistics["d0"]["x0"] == [0, 32, 64, 96, 128, 160, 192]
    assert statistics["d0"]["y0"] == [0, 26, 52]
    assert statistics["short_q1"]["duration_seconds_lte"] == "1.5"
    assert statistics["bootstrap"] == {
        "class_support": "fixed",
        "replicates": 20000,
        "seed": 20260720,
        "unit": "paired_two_level_seed_then_gate_video",
        "video_draws_per_selected_seed": 40,
        "window_proposal_and_instance_units_forbidden": True,
    }
    assert statistics["max_t"]["zero_standard_error_outcome"] == "NO_DECISION_INVALID_EVIDENCE"
    assert statistics["missing_or_invalid_outcome"] == "NO_DECISION_INVALID_EVIDENCE"
