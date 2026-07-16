"""Post-Stage-C recalibration and Gate-3 re-adjudication contracts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .gates23 import (
    R2_SEEDS,
    _normalize_row,
    _validate_candidate_actions,
    _validate_split_windows,
    adjudicate_gate3,
)
from .protocol import R2_PROTOCOL_ID, canonical_sha256
from .scheduler import R2_NON_DENSE_NAMES


POST_STAGE_C_REPLAY_SCHEMA = "chronotransport-r2-post-stage-c-replay-v1"
POST_STAGE_C_REPLAY_FIXTURE_SCHEMA = (
    "chronotransport-r2-post-stage-c-replay-test-only-v1"
)
POST_STAGE_C_GATE3_REPORT_SCHEMA = "chronotransport-r2-post-stage-c-gate3-report-v1"
POST_STAGE_C_GATE3_UNLOCK_SCHEMA = "chronotransport-r2-post-stage-c-gate3-unlock-v1"
POST_STAGE_C_GATE3_TERMINAL_SCHEMA = (
    "chronotransport-r2-post-stage-c-gate3-terminal-v1"
)
_SHA_FIELDS = {
    "completion_artifact_sha256",
    "checkpoint_file_sha256",
    "checkpoint_provenance_sha256",
    "predictor_canonical_sha256",
    "fit_baseline_payload_sha256",
}
_ARTIFACT_FIELDS = {
    "schema",
    "protocol",
    "registration_sha256",
    "registration_commit",
    "gate1_unlock_artifact_sha256",
    "pre_stage_c_gates23_report_sha256",
    "manifest_sha256",
    "library_sha256",
    "seed_order",
    "candidate_order",
    "candidate_action_sha256_by_name",
    "split_window_ids",
    "video_id_by_window",
    "stage_c_bindings",
    "row_count",
    "rows",
    "artifact_sha256",
}


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("post-Stage-C registration commit must be full SHA-1")
    return value


def _stage_c_bindings(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {
        str(seed) for seed in R2_SEEDS
    }:
        raise ValueError("post-Stage-C replay requires exactly three Stage-C bindings")
    result = {}
    for seed in R2_SEEDS:
        raw = value[str(seed)]
        if not isinstance(raw, Mapping) or set(raw) != _SHA_FIELDS:
            raise ValueError(f"post-Stage-C binding {seed} fields mismatch")
        result[str(seed)] = {
            field: _sha(raw[field], f"post-Stage-C binding {seed}/{field}")
            for field in sorted(_SHA_FIELDS)
        }
    return result


def _phase_view(bindings: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    return {
        str(seed): {
            "phase_marker_sha256": bindings[str(seed)][
                "completion_artifact_sha256"
            ],
            "trained_checkpoint_sha256": bindings[str(seed)][
                "checkpoint_file_sha256"
            ],
            "predictor_canonical_sha256": bindings[str(seed)][
                "predictor_canonical_sha256"
            ],
            "fit_baseline_payload_sha256": bindings[str(seed)][
                "fit_baseline_payload_sha256"
            ],
        }
        for seed in R2_SEEDS
    }


def _build_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: str,
    registration_sha256: str,
    registration_commit: str,
    gate1_unlock_artifact_sha256: str,
    pre_stage_c_gates23_report_sha256: str,
    manifest_sha256: str,
    library_sha256: str,
    split_window_ids: Mapping[str, Sequence[str]],
    video_id_by_window: Mapping[str, str],
    candidate_action_sha256_by_name: Mapping[str, str],
    stage_c_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if schema not in {POST_STAGE_C_REPLAY_SCHEMA, POST_STAGE_C_REPLAY_FIXTURE_SCHEMA}:
        raise ValueError("unsupported post-Stage-C replay schema")
    identities = {
        "registration_sha256": _sha(registration_sha256, "registration SHA-256"),
        "gate1_unlock_artifact_sha256": _sha(
            gate1_unlock_artifact_sha256, "Gate1 unlock SHA-256"
        ),
        "pre_stage_c_gates23_report_sha256": _sha(
            pre_stage_c_gates23_report_sha256, "pre-Stage-C Gate3 report SHA-256"
        ),
        "manifest_sha256": _sha(manifest_sha256, "manifest SHA-256"),
        "library_sha256": _sha(library_sha256, "library SHA-256"),
    }
    commit = _commit(registration_commit)
    splits = _validate_split_windows(
        {name: list(split_window_ids[name]) for name in ("calibration", "evaluation")}
    )
    all_windows = splits["calibration"] + splits["evaluation"]
    if not isinstance(video_id_by_window, Mapping) or set(video_id_by_window) != set(
        all_windows
    ):
        raise ValueError("post-Stage-C video/window mapping fields mismatch")
    videos = {window: str(video_id_by_window[window]) for window in all_windows}
    if any(not value for value in videos.values()) or len(set(videos.values())) != len(
        videos
    ):
        raise ValueError("post-Stage-C windows require distinct non-empty video IDs")
    actions = _validate_candidate_actions(candidate_action_sha256_by_name)
    bindings = _stage_c_bindings(stage_c_bindings)
    phases = _phase_view(bindings)
    if not isinstance(rows, Sequence) or len(rows) != 180:
        raise ValueError("post-Stage-C replay requires 180 seed-window vectors")
    normalized = []
    ordinal = 0
    for split in ("calibration", "evaluation"):
        for seed in R2_SEEDS:
            for window in splits[split]:
                normalized.append(
                    _normalize_row(
                        rows[ordinal],
                        expected_seed=seed,
                        expected_split=split,
                        expected_window=window,
                        expected_video=videos[window],
                        actions=actions,
                        phase_bindings=phases,
                    )
                )
                ordinal += 1
    artifact = {
        "schema": schema,
        "protocol": R2_PROTOCOL_ID,
        **identities,
        "registration_commit": commit,
        "seed_order": list(R2_SEEDS),
        "candidate_order": list(R2_NON_DENSE_NAMES),
        "candidate_action_sha256_by_name": actions,
        "split_window_ids": splits,
        "video_id_by_window": videos,
        "stage_c_bindings": bindings,
        "row_count": len(normalized),
        "rows": normalized,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def build_post_stage_c_replay_artifact_for_test_only(
    rows: Sequence[Mapping[str, Any]], **identity: Any
) -> dict[str, Any]:
    return _build_replay(
        rows,
        schema=POST_STAGE_C_REPLAY_FIXTURE_SCHEMA,
        **identity,
    )


def validate_post_stage_c_replay_artifact(
    artifact: Mapping[str, Any], *, fixture: bool = False
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_FIELDS:
        raise ValueError("post-Stage-C replay artifact fields mismatch")
    expected_schema = (
        POST_STAGE_C_REPLAY_FIXTURE_SCHEMA if fixture else POST_STAGE_C_REPLAY_SCHEMA
    )
    if artifact["schema"] != expected_schema or artifact["protocol"] != R2_PROTOCOL_ID:
        raise ValueError("post-Stage-C replay schema/protocol mismatch")
    rebuilt = _build_replay(
        [
            {key: value for key, value in row.items() if key != "row_sha256"}
            for row in artifact["rows"]
        ],
        schema=expected_schema,
        registration_sha256=artifact["registration_sha256"],
        registration_commit=artifact["registration_commit"],
        gate1_unlock_artifact_sha256=artifact["gate1_unlock_artifact_sha256"],
        pre_stage_c_gates23_report_sha256=artifact[
            "pre_stage_c_gates23_report_sha256"
        ],
        manifest_sha256=artifact["manifest_sha256"],
        library_sha256=artifact["library_sha256"],
        split_window_ids=artifact["split_window_ids"],
        video_id_by_window=artifact["video_id_by_window"],
        candidate_action_sha256_by_name=artifact[
            "candidate_action_sha256_by_name"
        ],
        stage_c_bindings=artifact["stage_c_bindings"],
    )
    if rebuilt != dict(artifact):
        raise ValueError("post-Stage-C replay differs from exact recomputation")
    return rebuilt


def _post_gate3_report(
    replay: Mapping[str, Any],
    *,
    candidate_cost_p50: Mapping[str, Any],
    budget: float,
    fit_baseline_constants_by_seed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gate3 = adjudicate_gate3(
        replay["rows"],
        candidate_cost_p50=candidate_cost_p50,
        budget=budget,
        fit_baseline_constants_by_seed=fit_baseline_constants_by_seed,
    )
    passed = gate3["status"] == "PASS"
    report = {
        "schema": POST_STAGE_C_GATE3_REPORT_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "status": "PASS" if passed else "FAIL",
        "registration_sha256": replay["registration_sha256"],
        "registration_commit": replay["registration_commit"],
        "post_stage_c_replay_sha256": replay["artifact_sha256"],
        "pre_stage_c_gates23_report_sha256": replay[
            "pre_stage_c_gates23_report_sha256"
        ],
        "stage_c_bindings_sha256": canonical_sha256(replay["stage_c_bindings"]),
        "gate3": gate3,
        "q_conf_by_seed": dict(gate3["calibration"]["q_conf_by_seed"]),
        "claim_flags": {
            "stage_c_complete": True,
            "post_stage_c_gate3_pass": passed,
            "metric_adatad_thumos14_official_full_video": False,
            "latency_slurm_single_device_fixed_stack": False,
            "deploy": False,
            "paper": False,
        },
    }
    report["artifact_sha256"] = canonical_sha256(report)
    return report


def adjudicate_post_stage_c_gate3_for_test_only(
    replay: Mapping[str, Any],
    *,
    candidate_cost_p50: Mapping[str, Any],
    budget: float,
    fit_baseline_constants_by_seed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validated = validate_post_stage_c_replay_artifact(replay, fixture=True)
    return _post_gate3_report(
        validated,
        candidate_cost_p50=candidate_cost_p50,
        budget=budget,
        fit_baseline_constants_by_seed=fit_baseline_constants_by_seed,
    )


def validate_post_stage_c_gate3_report(
    report: Mapping[str, Any],
    *,
    replay: Mapping[str, Any],
    candidate_cost_p50: Mapping[str, Any],
    budget: float,
    fit_baseline_constants_by_seed: Mapping[str, Mapping[str, Any]],
    fixture: bool = False,
) -> dict[str, Any]:
    validated = validate_post_stage_c_replay_artifact(replay, fixture=fixture)
    expected = _post_gate3_report(
        validated,
        candidate_cost_p50=candidate_cost_p50,
        budget=budget,
        fit_baseline_constants_by_seed=fit_baseline_constants_by_seed,
    )
    if not isinstance(report, Mapping) or dict(report) != expected:
        raise ValueError("post-Stage-C Gate3 report differs from exact recomputation")
    return expected


def build_post_stage_c_gate3_unlock(
    report: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        report.get("schema") != POST_STAGE_C_GATE3_REPORT_SCHEMA
        or report.get("status") != "PASS"
        or report.get("post_stage_c_replay_sha256") != replay.get("artifact_sha256")
        or report.get("claim_flags", {}).get("post_stage_c_gate3_pass") is not True
    ):
        raise ValueError("Gate4 requires an exact PASS post-Stage-C Gate3 report")
    unlock = {
        "schema": POST_STAGE_C_GATE3_UNLOCK_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "status": "PASS",
        "registration_sha256": report["registration_sha256"],
        "registration_commit": report["registration_commit"],
        "post_stage_c_replay_sha256": replay["artifact_sha256"],
        "post_stage_c_gate3_report_sha256": report["artifact_sha256"],
        "stage_c_bindings": replay["stage_c_bindings"],
        "q_conf_by_seed": report["q_conf_by_seed"],
        "claim_flags": dict(report["claim_flags"]),
    }
    unlock["artifact_sha256"] = canonical_sha256(unlock)
    return unlock


def validate_post_stage_c_gate3_unlock(
    unlock: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    expected = build_post_stage_c_gate3_unlock(report, replay)
    if not isinstance(unlock, Mapping) or dict(unlock) != expected:
        raise ValueError("post-Stage-C Gate3 unlock differs from exact recomputation")
    return expected


__all__ = [
    "POST_STAGE_C_GATE3_REPORT_SCHEMA",
    "POST_STAGE_C_GATE3_TERMINAL_SCHEMA",
    "POST_STAGE_C_GATE3_UNLOCK_SCHEMA",
    "POST_STAGE_C_REPLAY_SCHEMA",
    "adjudicate_post_stage_c_gate3_for_test_only",
    "build_post_stage_c_gate3_unlock",
    "build_post_stage_c_replay_artifact_for_test_only",
    "validate_post_stage_c_gate3_report",
    "validate_post_stage_c_gate3_unlock",
    "validate_post_stage_c_replay_artifact",
]
