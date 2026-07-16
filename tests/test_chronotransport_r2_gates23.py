from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import threading

import pytest
import torch

from opentad.models.chronotransport.gates23 import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    GATES23_REPLAY_FORMAL_SCHEMA,
    R2_SEEDS,
    _hierarchical_bootstrap,
    _percentile_interval,
    adjudicate_gates23,
    build_gates23_terminal_marker,
    build_gates23_replay_artifact_for_test_only,
    calibrate_simultaneous_window_offset,
    adjudicate_gate2,
    adjudicate_gates23_for_test_only,
    load_exact_canonical_json,
    select_risk_constrained_schedule,
    validate_gates23_report_for_test_only,
    validate_gates23_report,
    validate_gate3_unlock_artifact,
    validate_stage_b_phase_markers_static,
    validate_gates23_replay_artifact,
    validate_gates23_replay_artifact_for_test_only,
)
from opentad.models.chronotransport.formal_stage_b import (
    build_fit_schedule_constant_artifact,
)
from opentad.models.chronotransport.protocol import canonical_json_bytes, canonical_sha256
from opentad.models.chronotransport.post_stage_c import (
    adjudicate_post_stage_c_gate3_for_test_only,
    build_post_stage_c_gate3_unlock,
    build_post_stage_c_replay_artifact_for_test_only,
    validate_post_stage_c_gate3_report,
    validate_post_stage_c_gate3_unlock,
    validate_post_stage_c_replay_artifact,
)
from opentad.models.chronotransport.scheduler import R2_NON_DENSE_NAMES
from tools.bata.run_chronotransport_r2_gates23 import (
    GATES23_REGISTERED_SOURCE_PATHS,
    _assert_gates23_sources_registered,
    _atomic_write,
    _validate_gate1,
)


SHA = "a" * 64
COMMIT = "b" * 40


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture_inputs():
    calibration = [f"cal-{index:02d}" for index in range(30)]
    evaluation = [f"eval-{index:02d}" for index in range(30)]
    windows = calibration + evaluation
    videos = {window: f"video-{window}" for window in windows}
    actions = {name: _sha(f"action:{name}") for name in R2_NON_DENSE_NAMES}
    phases = {
        str(seed): {
            "phase_marker_sha256": _sha(f"phase:{seed}"),
            "trained_checkpoint_sha256": _sha(f"checkpoint:{seed}"),
            "predictor_canonical_sha256": _sha(f"predictor:{seed}"),
            "fit_baseline_payload_sha256": _sha(f"baseline:{seed}"),
        }
        for seed in R2_SEEDS
    }
    base_regret = [
        0.20,
        0.36,
        0.23,
        0.41,
        0.27,
        0.46,
        0.50,
        0.61,
        0.29,
        0.39,
        0.31,
        0.43,
        0.18,
        0.34,
        0.15,
        0.37,
    ]
    rows = []
    for split, split_windows in (("calibration", calibration), ("evaluation", evaluation)):
        for seed in R2_SEEDS:
            for window_index, window in enumerate(split_windows):
                shift = window_index * 0.0002 + (seed - R2_SEEDS[0]) * 0.0001
                regret = [value + shift for value in base_regret]
                q_hat = [max(0.0, value - 0.01) for value in regret]
                feature = [value * 0.5 for value in regret]
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "window_id": window,
                        "video_id": videos[window],
                        "trained_checkpoint_sha256": phases[str(seed)][
                            "trained_checkpoint_sha256"
                        ],
                        "predictor_canonical_sha256": phases[str(seed)][
                            "predictor_canonical_sha256"
                        ],
                        "materialized_window_sha256": _sha(f"materialized:{seed}:{window}"),
                        "augmentation_sha256": _sha(f"augmentation:{seed}:{window}"),
                        "candidate_order": list(R2_NON_DENSE_NAMES),
                        "q_hat": q_hat,
                        "detector_regret": regret,
                        "feature_mse": feature,
                        "requested_action_sha256": [actions[name] for name in R2_NON_DENSE_NAMES],
                        "executed_action_sha256": [actions[name] for name in R2_NON_DENSE_NAMES],
                        "execution": {
                            "repair_count": 0,
                            "nan_fallback": False,
                            "whole_window_dense_fallback": False,
                            "safety_override_budget_violation": False,
                            "window_cache_reset": True,
                        },
                        "no_leak": {
                            "gt_used_for_scheduler": False,
                            "teacher_used_for_scheduler": False,
                            "dense_reference_used_for_scheduler": False,
                            "raw_prediction_cache_used_for_scheduler": False,
                            "counterfactual_ledger_used_for_scheduler": False,
                            "evaluation_oracle_used_for_scheduler": False,
                            "scheduler_target_access": False,
                            "targets_evaluation_only": True,
                        },
                    }
                )
    artifact = build_gates23_replay_artifact_for_test_only(
        rows,
        registration_sha256=SHA,
        registration_commit=COMMIT,
        gate1_unlock_artifact_sha256=_sha("gate1"),
        manifest_sha256=_sha("manifest"),
        library_sha256=_sha("library"),
        split_window_ids={"calibration": calibration, "evaluation": evaluation},
        video_id_by_window=videos,
        candidate_action_sha256_by_name=actions,
        phase_bindings=phases,
    )
    costs = {name: 0.50 + 0.01 * index for index, name in enumerate(R2_NON_DENSE_NAMES)}
    costs["joint_reverse_transport"] = 0.25
    costs["dense"] = 1.20
    baselines = {
        str(seed): {name: 0.80 for name in R2_NON_DENSE_NAMES}
        for seed in R2_SEEDS
    }
    return artifact, costs, baselines


def test_post_stage_c_recalibration_requires_new_exact_gate3_unlock():
    original, costs, baselines = _fixture_inputs()
    bindings = {
        str(seed): {
            "completion_artifact_sha256": _sha(f"stage-c-terminal:{seed}"),
            "checkpoint_file_sha256": _sha(f"stage-c-checkpoint:{seed}"),
            "checkpoint_provenance_sha256": _sha(f"stage-c-provenance:{seed}"),
            "predictor_canonical_sha256": _sha(f"stage-c-predictor:{seed}"),
            "fit_baseline_payload_sha256": original["phase_bindings"][str(seed)][
                "fit_baseline_payload_sha256"
            ],
        }
        for seed in R2_SEEDS
    }
    rows = []
    for raw in original["rows"]:
        row = {key: copy.deepcopy(value) for key, value in raw.items() if key != "row_sha256"}
        binding = bindings[str(row["seed"])]
        row["trained_checkpoint_sha256"] = binding["checkpoint_file_sha256"]
        row["predictor_canonical_sha256"] = binding[
            "predictor_canonical_sha256"
        ]
        rows.append(row)
    replay = build_post_stage_c_replay_artifact_for_test_only(
        rows,
        registration_sha256=original["registration_sha256"],
        registration_commit=original["registration_commit"],
        gate1_unlock_artifact_sha256=original[
            "gate1_unlock_artifact_sha256"
        ],
        pre_stage_c_gates23_report_sha256=_sha("pre-stage-c-gate3"),
        manifest_sha256=original["manifest_sha256"],
        library_sha256=original["library_sha256"],
        split_window_ids=original["split_window_ids"],
        video_id_by_window=original["video_id_by_window"],
        candidate_action_sha256_by_name=original[
            "candidate_action_sha256_by_name"
        ],
        stage_c_bindings=bindings,
    )
    validate_post_stage_c_replay_artifact(replay, fixture=True)
    report = adjudicate_post_stage_c_gate3_for_test_only(
        replay,
        candidate_cost_p50=costs,
        budget=costs["periodic4_transport"],
        fit_baseline_constants_by_seed=baselines,
    )
    assert report["status"] == "PASS"
    validate_post_stage_c_gate3_report(
        report,
        replay=replay,
        candidate_cost_p50=costs,
        budget=costs["periodic4_transport"],
        fit_baseline_constants_by_seed=baselines,
        fixture=True,
    )
    unlock = build_post_stage_c_gate3_unlock(report, replay)
    validate_post_stage_c_gate3_unlock(unlock, report=report, replay=replay)
    assert unlock["q_conf_by_seed"] == report["gate3"]["calibration"][
        "q_conf_by_seed"
    ]

    tampered = copy.deepcopy(unlock)
    tampered["q_conf_by_seed"][str(R2_SEEDS[0])] += 0.01
    with pytest.raises(ValueError, match="unlock"):
        validate_post_stage_c_gate3_unlock(
            tampered,
            report=report,
            replay=replay,
        )


def test_exact_json_loader_rejects_duplicate_keys_and_noncanonical_bytes(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_exact_canonical_json(duplicate, label="fixture")

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b'{"b":2, "a":1}\n')
    with pytest.raises(ValueError, match="exact canonical JSON"):
        load_exact_canonical_json(noncanonical, label="fixture")

    canonical = tmp_path / "canonical.json"
    payload = {"b": 2, "a": 1}
    canonical.write_bytes(canonical_json_bytes(payload) + b"\n")
    assert load_exact_canonical_json(canonical, label="fixture") == payload


def test_exact_json_loader_rejects_symlink_leaf(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_bytes(canonical_json_bytes({"safe": True}) + b"\n")
    link = tmp_path / "formal-input.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink|regular file"):
        load_exact_canonical_json(link, label="formal input")


def test_exact_json_loader_rejects_symlink_in_parent_component(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    target = real_parent / "formal-input.json"
    target.write_bytes(canonical_json_bytes({"safe": True}) + b"\n")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_exact_canonical_json(linked_parent / target.name, label="formal input")


def test_repository_replay_factory_rejects_symlinked_registered_input_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import tools.bata.chronotransport_r2_gates23_replay_factory as factory

    real_parent = tmp_path / "real-inputs"
    real_parent.mkdir()
    for name in ("manifest.json", "registry.json", "config.json"):
        (real_parent / name).write_bytes(b"{}\n")
    linked_parent = tmp_path / "linked-inputs"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setattr(
        factory,
        "build_repository_stage_b_components",
        lambda **kwargs: pytest.fail("symlinked formal input reached Stage-B factory"),
    )
    registration = {
        "window_manifest": {
            "source_path": str(linked_parent / "manifest.json"),
            "registry_path": str(real_parent / "registry.json"),
            "config_identity_path": str(real_parent / "config.json"),
        },
        "exposures": {"stage_b": {}},
    }
    with pytest.raises(ValueError, match="symlink"):
        factory.build_repository_gates23_seed_context(
            registration=registration,
            seed=3407,
            registration_commit=COMMIT,
            registration_relpath="registration.json",
        )


def test_formal_context_rejects_symlinked_repository_root(tmp_path: Path):
    import opentad.models.chronotransport.gates23 as gates23

    real_root = tmp_path / "real-repository"
    real_root.mkdir()
    (real_root / "registration.json").write_bytes(b"{}\n")
    linked_root = tmp_path / "linked-repository"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        gates23._validate_formal_gate_context(
            registration={},
            gate1_unlock={},
            gate1_unlock_path=real_root / "gate1.json",
            repository_root=linked_root,
            registration_commit=COMMIT,
            registration_relpath="registration.json",
        )


def test_fixture_schema_is_exact_and_hash_bound():
    import opentad.models.chronotransport.gates23 as gates23

    artifact, _, _ = _fixture_inputs()
    assert validate_gates23_replay_artifact_for_test_only(artifact) == artifact

    extra = copy.deepcopy(artifact)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_gates23_replay_artifact_for_test_only(extra)

    reordered = copy.deepcopy(artifact)
    reordered["rows"][0], reordered["rows"][1] = reordered["rows"][1], reordered["rows"][0]
    unsigned = dict(reordered)
    unsigned.pop("artifact_sha256")
    reordered["artifact_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="canonical order"):
        validate_gates23_replay_artifact_for_test_only(reordered)

    leaked = copy.deepcopy(artifact)
    leaked["rows"][0]["no_leak"]["gt_used_for_scheduler"] = True
    row = dict(leaked["rows"][0])
    row.pop("row_sha256")
    leaked["rows"][0]["row_sha256"] = canonical_sha256(row)
    unsigned = dict(leaked)
    unsigned.pop("artifact_sha256")
    leaked["artifact_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="no-leak"):
        validate_gates23_replay_artifact_for_test_only(leaked)

    raw_rows = []
    for row in artifact["rows"]:
        raw = dict(row)
        raw.pop("row_sha256")
        raw_rows.append(raw)
    with pytest.raises(ValueError, match="only the disjoint test-fixture"):
        gates23._build_replay_artifact(
            raw_rows,
            schema=GATES23_REPLAY_FORMAL_SCHEMA,
            registration_sha256=artifact["registration_sha256"],
            registration_commit=artifact["registration_commit"],
            gate1_unlock_artifact_sha256=artifact[
                "gate1_unlock_artifact_sha256"
            ],
            manifest_sha256=artifact["manifest_sha256"],
            library_sha256=artifact["library_sha256"],
            split_window_ids=artifact["split_window_ids"],
            video_id_by_window=artifact["video_id_by_window"],
            candidate_action_sha256_by_name=artifact[
                "candidate_action_sha256_by_name"
            ],
            phase_bindings=artifact["phase_bindings"],
        )


def test_raw_rows_and_self_reported_no_leak_cannot_mint_formal_replay(
    monkeypatch: pytest.MonkeyPatch,
):
    """Formal validation must compare against a fresh repository-owned execution."""

    import opentad.models.chronotransport.gates23 as gates23

    fixture, _, _ = _fixture_inputs()
    repository_result = copy.deepcopy(fixture)
    repository_result["schema"] = GATES23_REPLAY_FORMAL_SCHEMA
    unsigned = dict(repository_result)
    unsigned.pop("artifact_sha256")
    repository_result["artifact_sha256"] = canonical_sha256(unsigned)

    forged = copy.deepcopy(repository_result)
    forged["rows"][0]["no_leak"]["gt_used_for_scheduler"] = True
    row = dict(forged["rows"][0])
    row.pop("row_sha256")
    forged["rows"][0]["row_sha256"] = canonical_sha256(row)
    unsigned = dict(forged)
    unsigned.pop("artifact_sha256")
    forged["artifact_sha256"] = canonical_sha256(unsigned)

    registration = {"registration_sha256": SHA}
    gate1 = {"artifact_sha256": _sha("gate1")}
    monkeypatch.setattr(
        gates23,
        "_validate_formal_gate_context",
        lambda **kwargs: (registration, gate1),
        raising=False,
    )
    monkeypatch.setattr(
        gates23,
        "run_registered_gates23_replay",
        lambda **kwargs: repository_result,
        raising=False,
    )
    with pytest.raises(ValueError, match="repository-owned replay|exact recomputation"):
        validate_gates23_replay_artifact(
            forged,
            registration=registration,
            gate1_unlock=gate1,
            phase_marker_paths={seed: Path(f"phase-{seed}.json") for seed in R2_SEEDS},
            gate1_unlock_path=Path("gate1.json"),
            repository_root=Path("repo"),
            registration_commit=COMMIT,
            registration_relpath="registration.json",
        )


def test_gate2_exact_complete_vector_and_hierarchical_bootstrap():
    artifact, _, _ = _fixture_inputs()
    validated = validate_gates23_replay_artifact_for_test_only(artifact)
    report = adjudicate_gate2(validated["rows"])
    assert report["status"] == "PASS"
    assert report["bootstrap"] == {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "outer_unit": "unique_manifested_window",
        "inner_unit": "seed",
        "complete_vector_unit": "three_periods",
    }
    assert report["detector_relative_reduction"] >= 0.05
    assert report["detector_improvement_ci95"][0] > 0.0
    assert report["feature_improvement_ci95"][0] > 0.0
    assert all(item["detector_improvement"] >= 0 for item in report["per_seed"].values())
    assert report["hold_only_transport_only_diagnostics"]["gate_membership"] is False

    incomplete = [row for row in validated["rows"] if not (
        row["split"] == "evaluation" and row["seed"] == 3407 and row["window_id"] == "eval-00"
    )]
    with pytest.raises(ValueError, match="30 evaluation windows"):
        adjudicate_gate2(incomplete)


def test_hierarchical_bootstrap_keeps_each_seed_run_global_across_windows():
    windows = [f"eval-{index:02d}" for index in range(30)]
    values = {
        (window, seed): (1.0 if seed in (3407, 3408) else -1.2)
        for window in windows
        for seed in R2_SEEDS
    }
    interval = _percentile_interval(
        _hierarchical_bootstrap(windows, value_by_key=values)
    )
    assert interval[0] <= 0.0
    assert interval[1] >= 0.0


def test_calibration_rank_is_window_max_then_28_of_30():
    artifact, _, _ = _fixture_inputs()
    rows = validate_gates23_replay_artifact_for_test_only(artifact)["rows"]
    calibration = calibrate_simultaneous_window_offset(rows)
    assert calibration["order_statistic_rank"] == 28
    assert calibration["candidate_count"] == 16
    assert calibration["calibration_windows_per_seed"] == 30
    assert calibration["residual_reduction"] == "candidate_max_per_window_before_rank"
    assert calibration["q_conf_by_seed"] == {
        str(seed): pytest.approx(0.01) for seed in R2_SEEDS
    }


def test_scheduler_uses_only_non_dense_cost_then_canonical_tie_and_dense_fallback():
    costs = {name: 0.5 for name in R2_NON_DENSE_NAMES}
    costs["dense"] = 1.2
    selected = select_risk_constrained_schedule(
        q_hat=[0.2] * 16,
        q_conf=0.1,
        candidate_cost_p50=costs,
        budget=0.5,
    )
    assert selected["selected_schedule"] == R2_NON_DENSE_NAMES[0]
    assert selected["selected_candidate_index"] == 0

    fallback = select_risk_constrained_schedule(
        q_hat=[1.1] * 16,
        q_conf=0.1,
        candidate_cost_p50=costs,
        budget=0.5,
    )
    assert fallback["selected_schedule"] == "dense"
    assert fallback["dense_safety_fallback"] is True

    metadata_fallback = select_risk_constrained_schedule(
        q_hat=[0.2] * 16,
        q_conf=0.1,
        candidate_cost_p50=costs,
        budget=0.5,
        metadata_valid=[False] * 16,
    )
    assert metadata_fallback["selected_schedule"] == "dense"


def test_full_gate3_passes_support_coverage_ranking_and_pinball_contracts():
    artifact, costs, baselines = _fixture_inputs()
    report = adjudicate_gates23_for_test_only(
        artifact,
        candidate_cost_p50=costs,
        budget=costs["periodic4_transport"],
        fit_baseline_constants_by_seed=baselines,
    )
    assert report["status"] == "PASS"
    assert report["gate2"]["status"] == "PASS"
    gate3 = report["gate3"]
    assert gate3["status"] == "PASS"
    assert gate3["selected_support"]["per_seed"] == {
        str(seed): 30 for seed in R2_SEEDS
    }
    assert gate3["selected_support"]["pooled"] == 90
    assert gate3["selected_support"]["distinct_windows"] == 30
    assert gate3["coverage"]["pooled_selected"] >= 0.85
    assert gate3["coverage"]["all_candidate_simultaneous"]["pooled"] >= 0.85
    assert gate3["coverage"]["all_selected_window"]["rate"] >= 0.85
    assert gate3["ranking"]["median_seed_mean_rho"] >= 0.2
    assert gate3["ranking"]["pooled_hierarchical_ci95"][0] > 0.0
    assert gate3["pinball"]["relative_improvement"] >= 0.10
    assert set(gate3["diagnostics"]) == {"calibrated", "uncalibrated", "dense"}
    assert report["claim_flags"] == {
        "oracle_headroom": True,
        "mechanism": True,
        "calibrated_risk_on_frozen_window_protocol": True,
        "metric_adatad_thumos14_official_full_video": False,
        "latency_slurm_single_device_fixed_stack": False,
        "deploy": False,
        "paper": False,
    }
    assert validate_gates23_report_for_test_only(
        report,
        replay_artifact=artifact,
        candidate_cost_p50=costs,
        budget=costs["periodic4_transport"],
        fit_baseline_constants_by_seed=baselines,
    ) == report
    tampered = copy.deepcopy(report)
    tampered["gate3"]["status"] = "FAIL"
    with pytest.raises(ValueError, match="exact recomputation"):
        validate_gates23_report_for_test_only(
            tampered,
            replay_artifact=artifact,
            candidate_cost_p50=costs,
            budget=costs["periodic4_transport"],
            fit_baseline_constants_by_seed=baselines,
        )


def test_gate3_degenerate_rank_and_zero_baseline_fail_without_zero_fill():
    artifact, costs, baselines = _fixture_inputs()
    degenerate = copy.deepcopy(artifact)
    for row in degenerate["rows"]:
        if row["split"] == "evaluation" and row["seed"] == 3407 and row["window_id"] == "eval-00":
            row["q_hat"] = [0.2] * 16
            unsigned_row = dict(row)
            unsigned_row.pop("row_sha256")
            row["row_sha256"] = canonical_sha256(unsigned_row)
    unsigned = dict(degenerate)
    unsigned.pop("artifact_sha256")
    degenerate["artifact_sha256"] = canonical_sha256(unsigned)
    report = adjudicate_gates23_for_test_only(
        degenerate,
        candidate_cost_p50=costs,
        budget=costs["periodic4_transport"],
        fit_baseline_constants_by_seed=baselines,
    )
    assert report["status"] == "FAIL"
    assert report["gate3"]["ranking"]["invalid_seed_windows"] == [
        {"seed": 3407, "window_id": "eval-00", "reason": "fewer_than_3_distinct_ranks"}
    ]
    assert report["gate3"]["ranking"]["pooled_rho"] is None

    zero_baseline_artifact = copy.deepcopy(artifact)
    constant_targets = list(zero_baseline_artifact["rows"][90]["detector_regret"])
    for row in zero_baseline_artifact["rows"]:
        if row["split"] == "evaluation":
            row["detector_regret"] = list(constant_targets)
            unsigned_row = dict(row)
            unsigned_row.pop("row_sha256")
            row["row_sha256"] = canonical_sha256(unsigned_row)
    unsigned = dict(zero_baseline_artifact)
    unsigned.pop("artifact_sha256")
    zero_baseline_artifact["artifact_sha256"] = canonical_sha256(unsigned)
    exact = {
        str(seed): {
            name: constant_targets[index]
            for index, name in enumerate(R2_NON_DENSE_NAMES)
        }
        for seed in R2_SEEDS
    }
    report = adjudicate_gates23_for_test_only(
        zero_baseline_artifact,
        candidate_cost_p50=costs,
        budget=costs["periodic4_transport"],
        fit_baseline_constants_by_seed=exact,
    )
    assert report["status"] == "FAIL"
    assert report["gate3"]["pinball"]["baseline_mean"] <= 1e-12
    assert report["gate3"]["pinball"]["relative_improvement"] is None


def test_cli_rejects_unregistered_gate23_source_surface(tmp_path: Path):
    registration = {"source_files": {}}
    with pytest.raises(ValueError, match="Gate2/3 source is absent from registration R"):
        _assert_gates23_sources_registered(registration, repository_root=tmp_path)
    assert set(GATES23_REGISTERED_SOURCE_PATHS) == {
        "opentad/models/chronotransport/gates23.py",
        "tools/bata/chronotransport_r2_gates23_replay_factory.py",
        "tools/bata/run_chronotransport_r2_gates23.py",
        "tests/test_chronotransport_r2_gates23.py",
    }


def test_every_public_formal_mint_requires_full_repository_context():
    required = {
        "repository_root",
        "registration_commit",
        "registration_relpath",
        "gate1_unlock_path",
        "phase_marker_paths",
    }
    for function in (
        validate_gates23_replay_artifact,
        adjudicate_gates23,
        validate_gates23_report,
        validate_gate3_unlock_artifact,
        build_gates23_terminal_marker,
    ):
        signature = inspect.signature(function)
        assert required <= set(signature.parameters)
        for name in required:
            assert signature.parameters[name].default is inspect.Parameter.empty


def test_cli_gate1_recompute_passes_full_repository_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import opentad.models.chronotransport.gate1_unlock as gate1_module

    registration = {"registration_sha256": SHA}
    artifact = {
        "gate1_input": {"fixed": True},
        "registration_sha256": SHA,
        "status": "PASS",
        "oracle_headroom": True,
    }
    path = tmp_path / "gate1.json"
    path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    calls = []

    def rebuild(gate1_input, **context):
        calls.append((gate1_input, context))
        return artifact

    monkeypatch.setattr(gate1_module, "build_gate1_unlock_artifact", rebuild)
    monkeypatch.setattr(
        gate1_module,
        "validate_gate1_unlock_artifact",
        lambda value, **context: value,
    )
    assert _validate_gate1(
        path,
        registration=registration,
        repository_root=tmp_path,
        registration_commit=COMMIT,
        registration_relpath="registration.json",
    ) == artifact
    assert calls == [
        (
            artifact["gate1_input"],
            {
                "repository_root": str(tmp_path),
                "registration_commit": COMMIT,
                "registration_relpath": "registration.json",
            },
        )
    ]


def test_formal_terminal_requires_full_context_and_derives_pass_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import opentad.models.chronotransport.gates23 as gates23

    signature = inspect.signature(build_gates23_terminal_marker)
    required = {
        "report",
        "replay_artifact",
        "registration",
        "gate1_unlock",
        "phase_marker_paths",
        "gate1_unlock_path",
        "repository_root",
        "registration_commit",
        "registration_relpath",
        "report_path",
    }
    assert required <= set(signature.parameters)
    assert {"terminal_state", "report_sha256", "reason"}.isdisjoint(
        signature.parameters
    )

    base = tmp_path / "formal"
    monkeypatch.setattr(gates23, "FORMAL_OUTPUT_BASE", str(base))
    report_path = base / COMMIT / "shared" / "gates23" / "gates23_report.json"
    report_path.parent.mkdir(parents=True)
    report = {"status": "PASS", "artifact_sha256": SHA}
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    replay = {"artifact_sha256": _sha("replay")}
    registration = {"registration_sha256": _sha("registration")}
    gate1 = {"artifact_sha256": _sha("gate1")}
    marker_paths = {seed: tmp_path / f"phase-{seed}.json" for seed in R2_SEEDS}
    calls = []

    def validate_context(**context):
        calls.append(("context", None, context))
        return registration, gate1

    def validate_report(value, **context):
        calls.append(("report", value, context))
        return value

    def validate_unlock(value, **context):
        calls.append(("unlock", value, context))
        return value

    monkeypatch.setattr(gates23, "_validate_formal_gate_context", validate_context)
    monkeypatch.setattr(gates23, "validate_gates23_report", validate_report)
    monkeypatch.setattr(gates23, "validate_gate3_unlock_artifact", validate_unlock)
    marker = build_gates23_terminal_marker(
        report=report,
        replay_artifact=replay,
        registration=registration,
        gate1_unlock=gate1,
        phase_marker_paths=marker_paths,
        gate1_unlock_path=tmp_path / "gate1.json",
        repository_root=tmp_path,
        registration_commit=COMMIT,
        registration_relpath="registration.json",
        report_path=report_path,
    )
    assert marker["terminal_state"] == "SUCCESS"
    assert marker["report_sha256"] == SHA
    assert marker["report_file_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert [item[0] for item in calls] == ["context", "report", "unlock"]
    unsigned = dict(marker)
    assert unsigned.pop("artifact_sha256") == canonical_sha256(unsigned)

    failed_report = {"status": "FAIL", "artifact_sha256": _sha("failed-report")}
    report_path.write_bytes(canonical_json_bytes(failed_report) + b"\n")
    calls.clear()
    failed_marker = build_gates23_terminal_marker(
        report=failed_report,
        replay_artifact=replay,
        registration=registration,
        gate1_unlock=gate1,
        phase_marker_paths=marker_paths,
        gate1_unlock_path=tmp_path / "gate1.json",
        repository_root=tmp_path,
        registration_commit=COMMIT,
        registration_relpath="registration.json",
        report_path=report_path,
    )
    assert failed_marker["terminal_state"] == "FAIL"
    assert failed_marker["report_sha256"] == failed_report["artifact_sha256"]
    assert [item[0] for item in calls] == ["context", "report"]

    report_path.write_bytes(canonical_json_bytes({"status": "FAIL"}) + b"\n")
    with pytest.raises(ValueError, match="report.*mapping|exact file bytes"):
        build_gates23_terminal_marker(
            report=report,
            replay_artifact=replay,
            registration=registration,
            gate1_unlock=gate1,
            phase_marker_paths=marker_paths,
            gate1_unlock_path=tmp_path / "gate1.json",
            repository_root=tmp_path,
            registration_commit=COMMIT,
            registration_relpath="registration.json",
            report_path=report_path,
        )


def test_formal_publication_is_no_clobber_and_run_lock_is_exclusive(tmp_path: Path):
    import tools.bata.run_chronotransport_r2_gates23 as cli

    output = tmp_path / "artifact.json"
    _atomic_write(output, b"first\n")
    with pytest.raises(FileExistsError):
        _atomic_write(output, b"second\n")
    assert output.read_bytes() == b"first\n"

    race_output = tmp_path / "race.json"
    barrier = threading.Barrier(2)
    outcomes = []
    outcomes_lock = threading.Lock()

    def publish(payload: bytes):
        barrier.wait()
        try:
            _atomic_write(race_output, payload)
        except FileExistsError:
            outcome = "exists"
        else:
            outcome = "published"
        with outcomes_lock:
            outcomes.append(outcome)

    writers = [
        threading.Thread(target=publish, args=(payload,))
        for payload in (b"alpha\n", b"beta\n")
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=5)
    assert sorted(outcomes) == ["exists", "published"]
    assert race_output.read_bytes() in {b"alpha\n", b"beta\n"}

    root = tmp_path / "run"
    with cli._exclusive_run_lock(root):
        with pytest.raises(FileExistsError, match="lock|active"):
            with cli._exclusive_run_lock(root):
                pass


def test_formal_publication_recovers_only_exact_partial_artifacts(tmp_path: Path):
    import tools.bata.run_chronotransport_r2_gates23 as cli

    replay = tmp_path / "gates23_replay.json"
    report = tmp_path / "gates23_report.json"
    terminal = tmp_path / "terminal_marker.json"
    outputs = {
        "root": tmp_path,
        "replay": replay,
        "report": report,
        "terminal": terminal,
    }
    replay_payload = canonical_json_bytes({"artifact": "replay"}) + b"\n"
    report_payload = canonical_json_bytes({"artifact": "report"}) + b"\n"

    cli._validate_recoverable_publication_state(outputs)
    assert cli._publish_or_validate_exact(
        replay, replay_payload, label="Gate2/3 replay"
    ) is True
    cli._validate_recoverable_publication_state(outputs)
    assert cli._publish_or_validate_exact(
        replay, replay_payload, label="Gate2/3 replay"
    ) is False
    assert cli._publish_or_validate_exact(
        report, report_payload, label="Gate2/3 report"
    ) is True
    cli._validate_recoverable_publication_state(outputs)

    with pytest.raises(ValueError, match="existing Gate2/3 replay bytes differ"):
        cli._publish_or_validate_exact(
            replay,
            canonical_json_bytes({"artifact": "different"}) + b"\n",
            label="Gate2/3 replay",
        )

    terminal.write_bytes(canonical_json_bytes({"terminal": "done"}) + b"\n")
    with pytest.raises(FileExistsError, match="terminal"):
        cli._validate_recoverable_publication_state(outputs)


def test_formal_output_root_rejects_symlink_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import tools.bata.run_chronotransport_r2_gates23 as cli

    real_base = tmp_path / "real-formal"
    real_base.mkdir()
    linked_base = tmp_path / "linked-formal"
    linked_base.symlink_to(real_base, target_is_directory=True)
    monkeypatch.setattr(cli, "FORMAL_OUTPUT_BASE", str(linked_base))
    with pytest.raises(ValueError, match="symlink"):
        cli._resolve_outputs(COMMIT)


@pytest.mark.parametrize("checkpoint_payload", ["plaintext", "arbitrary_ledger"])
def test_three_phase_markers_reject_unvalidated_checkpoint_or_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_payload: str,
):
    import opentad.models.chronotransport.gates23 as gates23

    base = tmp_path / "formal"
    monkeypatch.setattr(gates23, "FORMAL_OUTPUT_BASE", str(base))
    manifest_sha = _sha("phase-manifest")
    library_sha = _sha("phase-library")
    monkeypatch.setattr(
        gates23,
        "validate_pre_gate1_registration",
        lambda value: value,
    )
    actions = {name: _sha(f"phase-action:{name}") for name in R2_NON_DENSE_NAMES}
    fit_windows = [f"fit-{index:03d}" for index in range(140)]
    config_sha = _sha("phase-config")
    registration = {
        "registration_sha256": SHA,
        "window_manifest": {
            "artifact": {
                "manifest_sha256": manifest_sha,
                "splits": {"fit": fit_windows},
            }
        },
        "candidate_library": {
            "library_sha256": library_sha,
            "candidates": [
                {"name": name, "action_sha256": actions[name]}
                for name in R2_NON_DENSE_NAMES
            ],
        },
        "source_files": {
            "configs/adatad/thumos/c3_chronotransport_r2_stage_b.py": config_sha
        },
    }
    marker_paths = {}
    for seed in R2_SEEDS:
        seed_root = base / COMMIT / str(seed)
        seed_root.mkdir(parents=True)
        checkpoint = seed_root / "stage_b.pth"
        if checkpoint_payload == "plaintext":
            checkpoint.write_bytes(f"checkpoint-{seed}".encode("ascii"))
        else:
            torch.save(
                {
                    "schema": "forged-stage-b",
                    "ledger_rows": [{"seed": seed, "row": index} for index in range(140)],
                },
                checkpoint,
            )
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

        ledger = seed_root / "stage_b.jsonl"
        ledger_rows = [{"seed": seed, "row": index} for index in range(140)]
        ledger_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in ledger_rows)
        ledger.write_bytes(ledger_bytes)

        baseline_rows = []
        for window_index, window in enumerate(fit_windows):
            for candidate_index, name in enumerate(R2_NON_DENSE_NAMES):
                baseline_rows.append(
                    {
                        "seed": seed,
                        "window_id": window,
                        "candidate_index": candidate_index,
                        "schedule": name,
                        "regret": 0.1 + candidate_index * 0.01 + window_index * 0.0001,
                        "materialized_window_sha256": _sha(
                            f"phase-materialized:{seed}:{window}"
                        ),
                        "augmentation_sha256": _sha(
                            f"phase-augmentation:{seed}:{window}"
                        ),
                        "requested_action_sha256": actions[name],
                        "executed_action_sha256": actions[name],
                    }
                )
        predictor_sha = _sha(f"phase-predictor:{seed}")
        baseline = build_fit_schedule_constant_artifact(
            baseline_rows,
            seed=seed,
            fit_window_ids=fit_windows,
            candidate_action_sha256_by_name=actions,
            provenance={
                "registration_sha256": SHA,
                "manifest_sha256": manifest_sha,
                "library_sha256": library_sha,
                "trained_checkpoint_sha256": checkpoint_sha,
                "predictor_state_sha256": predictor_sha,
            },
        )
        baseline_path = seed_root / "fit_baseline.json"
        baseline_bytes = canonical_json_bytes(baseline) + b"\n"
        baseline_path.write_bytes(baseline_bytes)
        marker = {
            "schema": "chronotransport-r2-stage-b-phase-completion-v1",
            "protocol": "CT-P3R-3S-r2",
            "status": "PHASE_COMPLETE",
            "registration_sha256": SHA,
            "registration_commit": COMMIT,
            "seed": seed,
            "manifest_sha256": manifest_sha,
            "library_sha256": library_sha,
            "config_sha256": config_sha,
            "candidate_order": list(R2_NON_DENSE_NAMES),
            "trained_checkpoint": {
                "path": str(checkpoint.resolve()),
                "bytes": checkpoint.stat().st_size,
                "exact_bytes_sha256": checkpoint_sha,
                "state_dict_ema_sha256": _sha(f"phase-ema:{seed}"),
                "predictor_canonical_sha256": predictor_sha,
            },
            "ledger": {
                "path": str(ledger.resolve()),
                "bytes": len(ledger_bytes),
                "exact_bytes_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
                "canonical_rows_sha256": canonical_sha256(ledger_rows),
                "row_count": 140,
            },
            "fit_baseline": {
                "path": str(baseline_path.resolve()),
                "bytes": len(baseline_bytes),
                "exact_bytes_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
                "payload_sha256": baseline["artifact_sha256"],
                "row_count": baseline["row_count"],
                "fit_window_order_sha256": baseline["fit_window_order_sha256"],
                "fit_replay_key_sha256": baseline["fit_replay_key_sha256"],
            },
        }
        marker["artifact_sha256"] = canonical_sha256(marker)
        marker_path = seed_root / "phase_marker.json"
        marker_path.write_bytes(canonical_json_bytes(marker) + b"\n")
        marker_paths[seed] = marker_path
    gate1_path = tmp_path / "gate1.json"
    gate1_path.write_bytes(canonical_json_bytes({"artifact_sha256": _sha("gate1")}) + b"\n")
    monkeypatch.setattr(
        gates23,
        "_validate_formal_gate_context",
        lambda **kwargs: (registration, {"artifact_sha256": _sha("gate1")}),
        raising=False,
    )
    with pytest.raises(ValueError, match="loadable|checkpoint fields|frozen key set"):
        validate_stage_b_phase_markers_static(
            marker_paths,
            registration=registration,
            gate1_unlock={"artifact_sha256": _sha("gate1")},
            gate1_unlock_path=gate1_path,
            repository_root=tmp_path,
            registration_commit=COMMIT,
            registration_relpath="registration.json",
        )
