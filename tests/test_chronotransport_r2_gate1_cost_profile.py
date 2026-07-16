import copy
import hashlib
import inspect
from pathlib import Path

import pytest

from opentad.models.chronotransport.adjudication import (
    GATE1_RECORD_CANDIDATE_ORDER,
    build_gate1_paired_replay_artifact,
    build_gate1_paired_replay_artifact_for_test_only,
    build_gate1_record_artifact,
    build_gate1_record_artifact_for_test_only,
    gate1_oracle_headroom,
    gate1_oracle_headroom_from_profile,
    gate1_oracle_headroom_from_profile_for_test_only,
)
from opentad.models.chronotransport.cost_lookup import (
    CostLookupKey,
    build_execution_cost_ledger_entry,
    validate_execution_cost_ledger_entry,
)
from opentad.models.chronotransport.full_stack_profiler import (
    build_candidate_full_stack_profile_for_test_only,
    build_full_stack_profile_artifact,
    build_full_stack_profile_artifact_for_test_only,
    build_full_stack_profile_from_invocations_for_test_only,
    registered_candidate_provenance_for_test_only,
    validate_full_stack_profile_artifact,
    validate_full_stack_profile_artifact_for_test_only,
)
from opentad.models.chronotransport.profiler import REQUIRED_STAGE_FIELDS
from opentad.models.chronotransport.protocol import canonical_sha256
from opentad.models.chronotransport.registration import (
    EXPECTED_PROFILE_CANDIDATE_ORDER,
    FORMAL_OUTPUT_BASE,
    REGISTERED_PROFILE_BACKEND_IDENTITY,
    REGISTERED_PROFILE_BACKEND_SOURCE,
    build_pre_gate1_registration,
    resolve_gate1_output_root,
)
from test_chronotransport_r2_registration import _identity
from tools.bata.profile_chronotransport_r2_full_stack import validate_profile_request
from tools.bata.chronotransport_r2_profile_factory import build_registered_profile_session
from tools.bata.run_chronotransport_r2_gate1 import (
    run_gate1_payload,
    run_gate1_payload_for_test_only,
)


HOLD = (
    "periodic2_hold",
    "periodic4_hold",
    "periodic8_hold",
    "hold_only",
    "layer_only_early_recompute_hold",
    "layer_only_late_recompute_hold",
    "joint_progressive_hold",
    "joint_reverse_hold",
)
CONTROLS = (
    "motion_topk_p2",
    "motion_topk_p4",
    "motion_topk_p8",
    "random_p2",
    "random_p4",
    "random_p8",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _registration() -> dict:
    return build_pre_gate1_registration(_identity())


def _invocations(registration: dict, candidate: str, p50: float) -> list[dict]:
    candidate_plan = next(
        row
        for row in registration["profiler"]["candidate_plan"]
        if row["candidate_name"] == candidate
    )
    rows = []
    for index, (invocation_id, action_hash) in enumerate(
        zip(
            registration["profiler"]["invocation_ids"],
            candidate_plan["requested_action_sha256_by_invocation"],
        )
    ):
        total = p50 - 0.01 if index < 100 else p50 + 0.01
        ledger = build_execution_cost_ledger_entry(
            requested_schedule_name=candidate,
            requested_action_sha256=action_hash,
            requested_cost_p50=p50,
            executed_schedule_name=candidate,
            executed_action_sha256=action_hash,
            repair_count=0,
            nan_fallback=False,
            whole_window_dense_fallback=False,
            executed_lookup_cost_p50=p50,
            actual_total_ms=total,
            safety_override_budget_violation=False,
        )
        rows.append(
            {
                "invocation_index": index,
                "invocation_id": invocation_id,
                "total_ms": total,
                "diagnostic_ms": {name: 1000.0 + index for name in REQUIRED_STAGE_FIELDS},
                "peak_gpu_memory_bytes": 1024 + index,
                "cost_ledger": ledger,
                "execution_provenance": {
                    "backend_identity": REGISTERED_PROFILE_BACKEND_IDENTITY,
                    "backend_source_sha256": registration["source_files"][
                        REGISTERED_PROFILE_BACKEND_SOURCE
                    ],
                    "deploy_visible_signal_sha256": (
                        _sha(f"motion-signal:{index}")
                        if candidate.startswith("motion_topk_p")
                        else None
                    ),
                    "requested_action_sha256": action_hash,
                    "executed_action_sha256": action_hash,
                },
            }
        )
    return rows


def _candidate_profile(registration: dict, name: str, p50: float) -> dict:
    return build_candidate_full_stack_profile_for_test_only(
        registration=registration,
        candidate_name=name,
        warmup_count=50,
        invocations=_invocations(registration, name, p50),
    )


def _profile_artifact(registration: dict) -> dict:
    rows = {
        name: _invocations(
            registration,
            name,
            10.0 if name == "dense" else (7.0 if name == "periodic4_transport" else 6.9),
        )
        for name in EXPECTED_PROFILE_CANDIDATE_ORDER
    }
    return build_full_stack_profile_from_invocations_for_test_only(
        registration=registration,
        invocations_by_candidate=rows,
    )


def _record_rows(window_ids: list[str], joint_gain: float) -> list[dict]:
    rows = []
    for index, window_id in enumerate(window_ids):
        regrets = {name: 1.0 + 0.05 * position for position, name in enumerate(HOLD)}
        regrets["joint_progressive_hold"] = 1.0 - joint_gain if index % 2 == 0 else 1.2
        regrets["joint_reverse_hold"] = 1.2 if index % 2 == 0 else 1.0 - joint_gain
        regrets.update({name: 1.05 for name in CONTROLS})
        rows.append(
            {
                "window_id": window_id,
                "candidate_names": list(GATE1_RECORD_CANDIDATE_ORDER),
                "detector_regret": [regrets[name] for name in GATE1_RECORD_CANDIDATE_ORDER],
            }
        )
    return rows


def _records(registration: dict, split: str, joint_gain: float) -> dict:
    ids = registration["window_manifest"]["artifact"]["splits"][split]
    raw_rows = _record_rows(ids, joint_gain)
    plans = {
        row["candidate_name"]: row for row in registration["profiler"]["candidate_plan"]
    }
    invocation_index = {
        window_id: index
        for index, window_id in enumerate(registration["profiler"]["invocation_ids"])
    }
    replay_rows = []
    for row in raw_rows:
        index = invocation_index[row["window_id"]]
        regret_by_name = {name: 1.05 for name in EXPECTED_PROFILE_CANDIDATE_ORDER}
        regret_by_name.update(dict(zip(row["candidate_names"], row["detector_regret"])))
        regret_by_name["dense"] = 0.0
        dense_loss = 1.0
        candidate_losses = [
            dense_loss + regret_by_name[name]
            for name in EXPECTED_PROFILE_CANDIDATE_ORDER
        ]
        replay_rows.append(
            {
                "window_id": row["window_id"],
                "candidate_names": list(EXPECTED_PROFILE_CANDIDATE_ORDER),
                "dense_detector_loss": dense_loss,
                "candidate_detector_loss": candidate_losses,
                "order_probe_candidate_names": list(
                    reversed(EXPECTED_PROFILE_CANDIDATE_ORDER)
                ),
                "order_probe_candidate_detector_loss": list(reversed(candidate_losses)),
                "materialized_window_sha256": _sha(f"materialized:{row['window_id']}"),
                "augmentation_sha256": _sha(f"augmentation:{row['window_id']}"),
                "deploy_visible_motion_sha256": _sha(f"motion:{row['window_id']}"),
                "dense_reference_sha256": _sha(f"dense-reference:{row['window_id']}"),
                "dense_checkpoint_sha256": registration["dense_checkpoint"]["sha256"],
                "config_sha256": registration["profiler"]["model_config_sha256"],
                "backend_identity": REGISTERED_PROFILE_BACKEND_IDENTITY,
                "backend_source_sha256": registration["source_files"][
                    REGISTERED_PROFILE_BACKEND_SOURCE
                ],
                "candidate_action_sha256": [
                    plans[name]["requested_action_sha256_by_invocation"][index]
                    for name in EXPECTED_PROFILE_CANDIDATE_ORDER
                ],
            }
        )
    replay = build_gate1_paired_replay_artifact_for_test_only(
        registration=registration,
        split=split,
        rows=replay_rows,
    )
    return build_gate1_record_artifact_for_test_only(
        registration=registration,
        split=split,
        paired_replay=replay,
    )


def test_cost_lookup_key_requires_exact_formal_provenance_without_casting():
    registration = _registration()
    provenance = registered_candidate_provenance_for_test_only(
        registration, "periodic4_transport"
    )
    key = CostLookupKey.from_provenance(provenance)
    key.validate_formal()
    assert key.environment_sha256 == registration["environment"]["environment_sha256"]
    assert key.registration_sha256 == registration["registration_sha256"]
    assert key.factory_config_sha256 == provenance["factory_config_sha256"]
    incomplete = dict(provenance, gpu_uuid="")
    with pytest.raises(ValueError, match="gpu_uuid"):
        CostLookupKey.from_provenance(incomplete).validate_formal()
    impersonated = dict(provenance, batch_size=True)
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        CostLookupKey.from_provenance(impersonated)
    for field in (
        "environment_sha256",
        "registration_sha256",
        "factory_config_sha256",
    ):
        incomplete = dict(provenance)
        incomplete.pop(field)
        with pytest.raises(ValueError, match=field):
            CostLookupKey.from_provenance(incomplete)


def test_gate1_output_root_is_derived_only_from_registered_base_and_r():
    registration = _registration()
    assert registration["output_root"]["base"] == FORMAL_OUTPUT_BASE
    registration_commit = "a" * 40
    resolved = resolve_gate1_output_root(registration, registration_commit)
    assert resolved == (
        Path(registration["output_root"]["base"])
        / registration_commit
        / "shared"
        / "gate1"
    ).resolve()
    damaged = copy.deepcopy(registration)
    damaged["output_root"]["template"] = "{base}/../../escape"
    damaged.pop("registration_sha256")
    damaged.pop("schema")
    with pytest.raises(ValueError, match="fixed R-derived"):
        build_pre_gate1_registration(damaged)


def test_requested_and_executed_cost_ledger_is_exact_typed_and_invalidates_safety_changes():
    valid = build_execution_cost_ledger_entry(
        requested_schedule_name="periodic4_transport",
        requested_action_sha256=_sha("requested"),
        requested_cost_p50=7.0,
        executed_schedule_name="periodic4_transport",
        executed_action_sha256=_sha("requested"),
        repair_count=0,
        nan_fallback=False,
        whole_window_dense_fallback=False,
        executed_lookup_cost_p50=7.0,
        actual_total_ms=7.2,
        safety_override_budget_violation=False,
    )
    validate_execution_cost_ledger_entry(valid, formal=True)
    for damaged, match in (
        (dict(valid, executed_action_sha256=_sha("executed")), "action hash"),
        (dict(valid, repair_count=1), "repair"),
        (dict(valid, repair_count=0.0), "repair_count"),
        (dict(valid, nan_fallback=0), "boolean"),
        (dict(valid, safety_override_budget_violation=True), "safety override"),
        ({**valid, "extra": "cheat"}, "fields mismatch"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_execution_cost_ledger_entry(damaged, formal=True)
    with pytest.raises((TypeError, ValueError), match="repair_count"):
        build_execution_cost_ledger_entry(
            requested_schedule_name="periodic4_transport",
            requested_action_sha256=_sha("requested"),
            requested_cost_p50=7.0,
            executed_schedule_name="periodic4_transport",
            executed_action_sha256=_sha("requested"),
            repair_count=True,
            nan_fallback=False,
            whole_window_dense_fallback=False,
            executed_lookup_cost_p50=7.0,
            actual_total_ms=7.2,
            safety_override_budget_violation=False,
        )


def test_full_stack_profile_is_exactly_registration_bound_and_rejects_placeholders():
    registration = _registration()
    profile = _candidate_profile(registration, "periodic4_transport", 7.0)
    assert profile["fixture_warmup_count"] == 50
    assert profile["fixture_sample_count"] == 200
    assert profile["fixture_total_ms"]["p50"] == pytest.approx(7.0)
    assert profile["fixture_diagnostic_ms"]["data_decode"]["p50"] > 1000
    assert profile["fixture_provenance"] == registered_candidate_provenance_for_test_only(
        registration, "periodic4_transport"
    )

    too_short = _invocations(registration, "periodic4_transport", 7.0)[:199]
    with pytest.raises(ValueError, match="exactly 200"):
        build_candidate_full_stack_profile_for_test_only(
            registration=registration,
            candidate_name="periodic4_transport",
            warmup_count=50,
            invocations=too_short,
        )
    broken = copy.deepcopy(profile)
    broken["fixture_invocations"][0]["fixture_measurement"]["diagnostic_ms"][
        "transport"
    ] = None
    profiles = [
        broken if name == "periodic4_transport" else _candidate_profile(registration, name, 6.9)
        for name in EXPECTED_PROFILE_CANDIDATE_ORDER
    ]
    with pytest.raises(ValueError, match="transport"):
        build_full_stack_profile_artifact_for_test_only(
            profiles, registration=registration
        )


def test_profile_artifact_requires_exact_23_candidate_and_200_invocation_order():
    registration = _registration()
    artifact = _profile_artifact(registration)
    validate_full_stack_profile_artifact_for_test_only(artifact, registration=registration)
    assert artifact["fixture_candidate_order"] == list(EXPECTED_PROFILE_CANDIDATE_ORDER)
    assert artifact["fixture_invocation_ids"] == registration["profiler"]["invocation_ids"]

    missing = copy.deepcopy(artifact)
    missing["fixture_candidates"].pop()
    with pytest.raises(ValueError, match="23-candidate order"):
        validate_full_stack_profile_artifact_for_test_only(missing, registration=registration)
    reordered = copy.deepcopy(artifact)
    reordered["fixture_candidates"][0], reordered["fixture_candidates"][1] = (
        reordered["fixture_candidates"][1],
        reordered["fixture_candidates"][0],
    )
    with pytest.raises(ValueError, match="23-candidate order"):
        validate_full_stack_profile_artifact_for_test_only(reordered, registration=registration)
    wrong_id = copy.deepcopy(artifact)
    wrong_id["fixture_candidates"][0]["fixture_invocations"][0][
        "fixture_sequence"
    ]["id"] = "fake-window"
    with pytest.raises(ValueError, match="invocation IDs/order"):
        validate_full_stack_profile_artifact_for_test_only(wrong_id, registration=registration)


def test_motion_profile_persists_and_cross_checks_signal_and_action_digests():
    registration = _registration()
    artifact = _profile_artifact(registration)
    motion_profile = next(
        profile
        for profile in artifact["fixture_candidates"]
        if profile["fixture_provenance"]["candidate_name"] == "motion_topk_p4"
    )
    first = motion_profile["fixture_invocations"][0]["fixture_execution_provenance"]
    assert len(first["deploy_visible_signal_sha256"]) == 64
    assert first["requested_action_sha256"] == first["executed_action_sha256"]
    damaged = copy.deepcopy(artifact)
    damaged_motion = next(
        profile
        for profile in damaged["fixture_candidates"]
        if profile["fixture_provenance"]["candidate_name"] == "motion_topk_p4"
    )
    damaged_motion["fixture_invocations"][0]["fixture_execution_provenance"][
        "deploy_visible_signal_sha256"
    ] = _sha(
        "forged-motion-signal"
    )
    with pytest.raises(ValueError, match="hashes|statistics"):
        validate_full_stack_profile_artifact_for_test_only(damaged, registration=registration)


def test_gate1_requires_hash_bound_manifest_artifacts_and_exact_formal_bootstrap():
    registration = _registration()
    artifact = _profile_artifact(registration)
    calibration = _records(registration, "calibration", 0.2)
    evaluation = _records(registration, "evaluation", 0.2)
    result = gate1_oracle_headroom_from_profile_for_test_only(
        registration=registration,
        calibration=calibration,
        evaluation=evaluation,
        full_stack_profile=artifact,
    )
    assert result["status"] == "PASS"
    assert result["budget"] == pytest.approx(7.0)
    assert result["budget_source"] == "measured_p50:periodic4_transport"
    assert result["budget_saving"] >= 0.20
    assert result["bootstrap_samples"] == 5000
    assert result["bootstrap_seed"] == 20260711

    with pytest.raises((TypeError, ValueError), match="paired replay"):
        build_gate1_record_artifact_for_test_only(
            registration=registration,
            split="evaluation",
            paired_replay={"raw": _record_rows(
                registration["window_manifest"]["artifact"]["splits"]["calibration"],
                0.2,
            )},
        )
    corrupted = copy.deepcopy(artifact)
    corrupted["fixture_candidates"][0]["fixture_invocations"][0][
        "fixture_cost_ledger"
    ]["repair_count"] = 1
    with pytest.raises(ValueError, match="repair"):
        gate1_oracle_headroom_from_profile_for_test_only(
            registration=registration,
            calibration=calibration,
            evaluation=evaluation,
            full_stack_profile=corrupted,
        )


def test_gate1_unlock_is_exact_recomputable_and_stage_b_rejects_forgery():
    from opentad.models.chronotransport.gate1_unlock import (
        validate_gate1_unlock_artifact_for_test_only,
    )

    registration = _registration()
    payload = {
        "schema": "chronotransport-r2-gate1-input-v3",
        "registration": registration,
        "calibration": _records(registration, "calibration", 0.2),
        "evaluation": _records(registration, "evaluation", 0.2),
        "full_stack_profile": _profile_artifact(registration),
    }
    unlock = run_gate1_payload_for_test_only(payload)
    assert unlock["schema"] == "chronotransport-r2-gate1-unlock-test-fixture-v1"
    assert unlock["status"] == "PASS"
    validate_gate1_unlock_artifact_for_test_only(unlock, registration=registration)
    forged = copy.deepcopy(unlock)
    forged["gate1_result"]["oracle_headroom"] = False
    with pytest.raises(ValueError, match="exact recomputable"):
        validate_gate1_unlock_artifact_for_test_only(
            forged, registration=registration
        )


def test_gate1_replay_derives_regret_and_rejects_caller_regret_reorder_and_fake_action():
    registration = _registration()
    ids = registration["window_manifest"]["artifact"]["splits"]["calibration"]
    raw = _record_rows(ids, 0.2)
    plans = {
        row["candidate_name"]: row for row in registration["profiler"]["candidate_plan"]
    }
    start = 140
    rows = []
    for offset, row in enumerate(raw):
        regret_by_name = {name: 1.05 for name in EXPECTED_PROFILE_CANDIDATE_ORDER}
        regret_by_name.update(dict(zip(row["candidate_names"], row["detector_regret"])))
        regret_by_name["dense"] = 0.0
        dense_loss = 2.0
        candidate_losses = [
            dense_loss + regret_by_name[name]
            for name in EXPECTED_PROFILE_CANDIDATE_ORDER
        ]
        rows.append(
            {
                "window_id": row["window_id"],
                "candidate_names": list(EXPECTED_PROFILE_CANDIDATE_ORDER),
                "dense_detector_loss": dense_loss,
                "candidate_detector_loss": candidate_losses,
                "order_probe_candidate_names": list(
                    reversed(EXPECTED_PROFILE_CANDIDATE_ORDER)
                ),
                "order_probe_candidate_detector_loss": list(reversed(candidate_losses)),
                "materialized_window_sha256": _sha(f"materialized:{offset}"),
                "augmentation_sha256": _sha(f"augmentation:{offset}"),
                "deploy_visible_motion_sha256": _sha(f"motion:{offset}"),
                "dense_reference_sha256": _sha(f"dense:{offset}"),
                "dense_checkpoint_sha256": registration["dense_checkpoint"]["sha256"],
                "config_sha256": registration["profiler"]["model_config_sha256"],
                "backend_identity": REGISTERED_PROFILE_BACKEND_IDENTITY,
                "backend_source_sha256": registration["source_files"][
                    REGISTERED_PROFILE_BACKEND_SOURCE
                ],
                "candidate_action_sha256": [
                    plans[name]["requested_action_sha256_by_invocation"][start + offset]
                    for name in EXPECTED_PROFILE_CANDIDATE_ORDER
                ],
            }
        )
    caller_regret = copy.deepcopy(rows)
    caller_regret[0]["detector_regret"] = [0.0] * len(EXPECTED_PROFILE_CANDIDATE_ORDER)
    with pytest.raises(ValueError, match="fields mismatch"):
        build_gate1_paired_replay_artifact_for_test_only(
            registration=registration, split="calibration", rows=caller_regret
        )
    artifact = build_gate1_paired_replay_artifact_for_test_only(
        registration=registration, split="calibration", rows=rows
    )
    assert artifact["fixture_rows"][0]["fixture_derived_detector_regret"] == pytest.approx(
        [
            max(value - rows[0]["dense_detector_loss"], 0.0)
            for value in rows[0]["candidate_detector_loss"]
        ]
    )
    reordered = copy.deepcopy(rows)
    reordered[0]["order_probe_candidate_detector_loss"][0] += 1.0
    with pytest.raises(ValueError, match="candidate-order probe"):
        build_gate1_paired_replay_artifact_for_test_only(
            registration=registration, split="calibration", rows=reordered
        )
    fake_action = copy.deepcopy(rows)
    fake_action[0]["candidate_action_sha256"][0] = _sha("fake-action")
    with pytest.raises(ValueError, match="action provenance"):
        build_gate1_paired_replay_artifact_for_test_only(
            registration=registration, split="calibration", rows=fake_action
        )


def test_formal_clis_reject_caller_supplied_factory_provenance_and_raw_gate_records():
    registration = _registration()
    request = {
        "schema": "chronotransport-r2-full-stack-profile-request-v2",
        "registration": registration,
    }
    assert validate_profile_request(request) == registration
    for forbidden in ("candidates", "factory", "factory_config", "provenance"):
        with pytest.raises(ValueError, match="fields mismatch"):
            validate_profile_request({**request, forbidden: "cheat"})

    with pytest.raises((TypeError, ValueError), match="artifact"):
        run_gate1_payload_for_test_only(
            {
                "schema": "chronotransport-r2-gate1-input-v3",
                "registration": registration,
                "calibration": {"raw": "mapping"},
                "evaluation": {"raw": "mapping"},
                "full_stack_profile": {"raw": "profile"},
            }
        )


def test_registered_profile_factory_has_one_fixed_backend_boundary():
    assert tuple(inspect.signature(build_registered_profile_session).parameters) == (
        "registration",
    )


def test_formal_profile_artifact_requires_fixed_session_measurements_only():
    import opentad.models.chronotransport as chronotransport
    from tools.bata.profile_chronotransport_r2_full_stack import profile_request

    assert "build_full_stack_profile_from_invocations" not in chronotransport.__all__
    assert tuple(inspect.signature(profile_request).parameters) == (
        "payload",
        "repository_root",
        "registration_commit",
        "registration_relpath",
    )
    registration = _registration()
    parameters = inspect.signature(build_full_stack_profile_artifact).parameters
    assert "session_result" not in parameters
    assert "candidates" not in parameters
    assert tuple(parameters) == (
        "registration",
        "repository_root",
        "registration_commit",
        "registration_relpath",
    )


def test_test_only_profile_schema_cannot_enter_formal_validator(monkeypatch):
    import opentad.models.chronotransport.full_stack_profiler as profile_module

    registration = _registration()
    fixture = _profile_artifact(registration)
    assert validate_full_stack_profile_artifact_for_test_only(
        fixture, registration=registration
    ) == fixture
    monkeypatch.setattr(
        profile_module,
        "validate_formal_gate1_context",
        lambda registration, **kwargs: registration,
    )
    with pytest.raises(ValueError, match="formal full-stack profile artifact schema"):
        validate_full_stack_profile_artifact(
            fixture,
            registration=registration,
            repository_root="/repo",
            registration_commit="a" * 40,
            registration_relpath="artifacts/registration.json",
        )


def test_gate1_paired_artifact_accepts_only_repo_runner_output():
    signature = inspect.signature(build_gate1_paired_replay_artifact)
    assert "runner_output" not in signature.parameters
    assert "rows" not in signature.parameters
    assert tuple(signature.parameters) == (
        "registration",
        "split",
        "repository_root",
        "registration_commit",
        "registration_relpath",
    )
    from opentad.models.chronotransport.replay import (
        run_registered_gate1_paired_replay,
    )

    parameters = tuple(inspect.signature(run_registered_gate1_paired_replay).parameters)
    assert parameters == (
        "registration",
        "split",
        "repository_root",
        "registration_commit",
        "registration_relpath",
    )
    assert not ({"detector", "materialized_batches", "deploy_visible_motion"} & set(parameters))


def test_formal_gate1_minting_apis_require_repository_context():
    from opentad.models.chronotransport.gate1_unlock import build_gate1_unlock_artifact

    required = ("repository_root", "registration_commit", "registration_relpath")
    for callback in (
        gate1_oracle_headroom_from_profile,
        build_gate1_record_artifact,
        build_gate1_unlock_artifact,
        run_gate1_payload,
    ):
        parameters = inspect.signature(callback).parameters
        assert all(name in parameters for name in required)


def test_gate1_budget_saving_uses_exact_20_percent_cross_multiply_boundary():
    ids = [f"w{index:02d}" for index in range(30)]
    raw_rows = _record_rows(ids, 0.2)
    records = {
        row["window_id"]: dict(zip(row["candidate_names"], row["detector_regret"]))
        for row in raw_rows
    }
    costs = {name: 7.0 for name in GATE1_RECORD_CANDIDATE_ORDER}
    exact = gate1_oracle_headroom(
        calibration=records,
        evaluation=records,
        candidate_cost_p50=costs,
        dense_cost_p50=10.0,
        budget=8.0,
        bootstrap_samples=10,
    )
    assert exact["hard_conditions"]["budget_saving_ge_20pct"] is True
    over = gate1_oracle_headroom(
        calibration=records,
        evaluation=records,
        candidate_cost_p50=costs,
        dense_cost_p50=10.0,
        budget=8.000000000000001,
        bootstrap_samples=10,
    )
    assert over["hard_conditions"]["budget_saving_ge_20pct"] is False
