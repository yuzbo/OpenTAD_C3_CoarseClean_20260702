from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "docs" / "methods" / "continuous_roi_s2_v2_1_protocol.json"
)


def load_and_resign(mutator=None):
    from tools.bata.continuous_roi_s2_contract import protocol_core_sha256

    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if mutator is not None:
        mutator(payload)
    payload["declared_protocol_sha256"] = protocol_core_sha256(payload)
    return payload


def test_frozen_protocol_hash_and_static_audit():
    from tools.bata.continuous_roi_s2_contract import (
        CONTINUOUS_ROI_S2_AUDIT_SCHEMA,
        load_protocol,
        validate_protocol,
    )

    audit = validate_protocol(load_protocol(PROTOCOL_PATH))
    assert audit["schema_version"] == CONTINUOUS_ROI_S2_AUDIT_SCHEMA
    assert audit["static_protocol_valid"] is True
    assert audit["implementation_authorized"] is True
    assert audit["training_authorized"] is False
    assert audit["official_test_open_allowed"] is False
    assert audit["state_assignments_checked"] == 128
    assert audit["check_count"] == 8


def test_protocol_rejects_s2_s3_conflation_even_when_resigned():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    payload = load_and_resign(
        lambda item: item["models"].update(u128_contains_selector=True)
    )
    with pytest.raises(ValueError, match="selector"):
        validate_protocol(payload)


def test_protocol_rejects_unbalanced_training_support():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    def mutate(payload):
        payload["training"]["family_counts"]["anchor"] += 1
        payload["training"]["family_counts"]["fixed_size"] -= 1

    with pytest.raises(ValueError, match="not balanced"):
        validate_protocol(load_and_resign(mutate))


def test_protocol_rejects_inference_only_control_outside_training_support():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    def mutate(payload):
        payload["models"]["cell_contracts"]["FS-PREF"][
            "geometry_family"
        ] = "untrained_fixed"

    with pytest.raises(ValueError, match="outside"):
        validate_protocol(load_and_resign(mutate))


def test_protocol_rejects_unmatched_reference_privilege():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    def mutate(payload):
        payload["models"]["cell_contracts"]["FS-PREF"]["gt_privilege"] = "none"

    with pytest.raises(ValueError, match="unequal GT privilege"):
        validate_protocol(load_and_resign(mutate))


def test_protocol_rejects_confidence_convergence_as_search_evidence():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    def mutate(payload):
        payload["candidate_reference"]["search_adequacy"][
            "confidence_convergence_is_evidence"
        ] = True

    with pytest.raises(ValueError, match="confidence convergence"):
        validate_protocol(load_and_resign(mutate))


def test_protocol_rejects_selector_cost_double_counting():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    def mutate(payload):
        payload["cost"]["measured_roi_policy_head"] = True

    with pytest.raises(ValueError, match="measured policy head"):
        validate_protocol(load_and_resign(mutate))


def test_protocol_rejects_incorrect_abba_population():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    def mutate(payload):
        payload["abba"]["invocations_per_arm"] = 1548

    with pytest.raises(ValueError, match="per-arm"):
        validate_protocol(load_and_resign(mutate))


def test_protocol_rejects_old_cluster_contract():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    def mutate(payload):
        payload["resources"]["inner_step"]["memory_mib"] = 128000

    with pytest.raises(ValueError, match="inner Slurm step"):
        validate_protocol(load_and_resign(mutate))


def test_protocol_rejects_deleting_failed_namespace():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    def mutate(payload):
        payload["resources"]["delete_failed_namespace"] = True
        payload["resources"]["preserve_failed_namespace"] = False

    with pytest.raises(ValueError, match="immutable"):
        validate_protocol(load_and_resign(mutate))


def test_outcome_state_machine_is_total_and_reaches_every_state():
    from tools.bata.continuous_roi_s2_contract import (
        EXPECTED_OUTCOMES,
        resolve_outcome,
    )

    observed = set()
    for state in range(128):
        flags = [bool(state & (1 << bit)) for bit in range(7)]
        observed.add(
            resolve_outcome(
                evidence_valid=flags[0],
                geometry_valid=flags[1],
                reference_adequate=flags[2],
                variable_sufficient=flags[3],
                fixed_sufficient=flags[4],
                variable_headroom=flags[5],
                cost_viable=flags[6],
            )
        )
    assert observed == set(EXPECTED_OUTCOMES)


def test_validator_does_not_mutate_input():
    from tools.bata.continuous_roi_s2_contract import validate_protocol

    payload = load_and_resign()
    before = copy.deepcopy(payload)
    validate_protocol(payload)
    assert payload == before
