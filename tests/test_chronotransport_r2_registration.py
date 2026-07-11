import copy

import pytest

from opentad.models.chronotransport.registration import (
    build_pre_gate1_registration,
    claim_flags,
    validate_pre_gate1_registration,
)


def _identity():
    return {
        "protocol_id": "CT-P3R-3S-r2",
        "spec": {"commit": "e4422f5", "sha256": "87FA"},
        "implementation_commit": "I" * 40,
        "registration_parent": {"commit": "I" * 40, "tree": "T" * 40},
        "source_files": {"runtime.py": "a" * 64},
        "upstream_commits": {"opentad": "u" * 40},
        "dense_checkpoint": {"sha256": "c" * 64, "bytes": 1, "registry_id": "dense"},
        "data": {"root_identity": "thumos14", "annotation_sha256": "d" * 64, "media_sha256": {}},
        "window_manifest": {"sha256": "w" * 64, "windows": 200},
        "candidate_library": {"sha256": "l" * 64, "candidates": 16},
        "exposures": {"stage_b_sha256": "b" * 64, "stage_c_formula": "fixed"},
        "controls": {"motion_sha256": "m" * 64, "random_sha256": "r" * 64},
        "bootstrap": {"gate1_samples": 5000, "seed": 20260711},
        "profiler": {"order_sha256": "p" * 64},
        "gates": {"gate1_relative": 0.1, "budget_saving": 0.2},
        "environment": {"gpu": "GPU1", "precision": "AMP FP16"},
        "output_root": "/data/run01/sczc063/yuzibo/chronotransport_runs/r2",
        "attestation": {"result_data_unread": True},
    }


def test_registration_is_canonical_hash_bound_and_validatable():
    registration = build_pre_gate1_registration(_identity())
    assert len(registration["registration_sha256"]) == 64
    assert validate_pre_gate1_registration(registration) == registration
    damaged = copy.deepcopy(registration)
    damaged["gates"]["gate1_relative"] = 0.2
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_pre_gate1_registration(damaged)


def test_registration_rejects_result_derived_fields():
    identity = _identity()
    identity["gate_report_path"] = "/tmp/result.json"
    with pytest.raises(ValueError, match="forbidden result-derived"):
        build_pre_gate1_registration(identity)


def test_claims_follow_gate_chain_and_never_unlock_deploy_or_paper():
    flags = claim_flags(gate1=True, gate2=True, gate3=True, gate4=True)
    assert flags["oracle_headroom"] and flags["mechanism"]
    assert flags["metric_adatad_thumos14_official_full_video"]
    assert flags["deploy"] is False and flags["paper"] is False
    with pytest.raises(ValueError, match="Gate 2"):
        claim_flags(gate2=True)


def test_gpu1_launcher_contains_registration_and_slurm_hard_guards():
    text = open("scripts/run_chronotransport_r2_gate1_gpu1.sh", encoding="utf-8").read()
    assert 'CUDA_VISIBLE_DEVICES:-}" == "1"' in text
    assert "CHRONOTRANSPORT_REGISTRATION_COMMIT" in text
    assert "SLURM_JOB_ID" in text and "SLURM_STEP_ID" in text
    assert "PRECHECK_ONLY" in text
