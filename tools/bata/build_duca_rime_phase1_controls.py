from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from tools.bata.create_duca_rime_splits import validate_rime_splits
from tools.bata.duca_full_stack_cost import (
    PROFILE_SCHEMA_VERSION,
    validate_and_rebuild_profile_summary,
)
from tools.bata.duca_rime_stage_contract import PHASE1_CONTROL_SCHEMA
from tools.bata.finalize_duca_rime_inference_ledger import (
    exact_uniform_positions,
)
from tools.bata.duca_rime_training import (
    PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE,
    PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS,
    STRICT_EXACT_CHECKPOINT_COMPATIBILITY_MODE,
)


METRICS_SCHEMA = "duca_rime_localization_metrics_v1"
LEDGER_SUMMARY_SCHEMA = "duca_rime_inference_ledger_summary_v1"
LEDGER_SCHEMA = "duca_rime_inference_ledger_v1"
GEOMETRY_SCHEMA = "duca_rime_phase1_geometry_audit_v1"
FULL_MODEL_GATE_SCHEMA = "duca_protected_physical_full_model_gate_v1"
_DENSE_VARIANTS = {"released_dense", "local_dense"}
_UNIFORM_VARIANTS = {"uniform_k384": 384, "uniform_k192": 192}
_PROFILE_CONTRACTS = {
    "no_probe_uniform_cost": (
        "phase1-no-probe-uniform",
        "duca_rime_phase1_no_probe_uniform_cost_v1",
        False,
        (
            "frame_selector._loss_weight_schedule_step",
            "frame_selector.adapter.transition_scorer.",
            "frame_selector.raw_actionness_source.",
        ),
    ),
    "probe_uniform_cost": (
        "phase1-probe-uniform",
        "duca_rime_phase1_probe_uniform_cost_v1",
        True,
        (
            "frame_selector._loss_weight_schedule_step",
            "frame_selector.adapter.transition_scorer.",
        ),
    ),
}


@lru_cache(maxsize=None)
def _sha256_resolved(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return _sha256_resolved(str(resolved))


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {resolved}")
    return resolved, payload


def _binding(path: str | Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _verify_binding(binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    expected = str(binding.get("sha256", ""))
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or not path.is_file()
        or _sha256_file(path) != expected
    ):
        raise ValueError(f"{label} artifact binding drift")
    return path


def _verify_content_hash(payload: Mapping[str, Any], label: str) -> None:
    unsigned = dict(payload)
    embedded = unsigned.pop("content_sha256", None)
    if embedded != _canonical_sha256(unsigned):
        raise ValueError(f"{label} content hash is invalid")


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    output = dict(payload)
    output["content_sha256"] = _canonical_sha256(output)
    text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different control: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _binding(path)


def _mean_video_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    output = {}
    video_metrics = metrics.get("video_metrics")
    if not isinstance(video_metrics, Mapping):
        raise ValueError("localization metrics lack per-video measurements")
    for name, values in video_metrics.items():
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"localization metric {name} is empty")
        numeric = [float(value) for value in values.values()]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"localization metric {name} is non-finite")
        output[str(name)] = mean(numeric)
    return output


def _validate_metrics(
    path: str | Path,
    *,
    expected_commit: str,
    split_validation: Mapping[str, Any],
    split_role: str,
    expected_videos: Sequence[str],
    variant: str,
    target_cost: float,
) -> dict[str, Any]:
    metrics_path, metrics = _load_json(path)
    _verify_content_hash(metrics, f"{variant} localization metrics")
    if (
        metrics.get("schema_version") != METRICS_SCHEMA
        or int(metrics.get("phase", -1)) != 1
        or metrics.get("git_commit") != expected_commit
        or metrics.get("variant") != variant
        or float(metrics.get("target_mean_cost", math.nan)) != float(target_cost)
        or metrics.get("padded_to_kmax") is not False
        or metrics.get("uses_official_final") is not False
        or metrics.get("official_final_used_for_training_or_selection") is not False
        or metrics.get("split_role") != split_role
        or metrics.get("split_assignment_sha256")
        != split_validation["assignment_sha256"]
        or metrics.get("evaluation_video_ids") != list(expected_videos)
        or metrics.get("official_evaluator_used_for_map_metrics") is not True
    ):
        raise ValueError(f"invalid or contaminated Phase-1 metrics: {metrics_path}")

    terminal_path = _verify_binding(
        {
            "path": metrics.get("terminal_evaluation_path"),
            "sha256": metrics.get("terminal_evaluation_sha256"),
        },
        f"{variant} terminal evaluation",
    )
    _, terminal = _load_json(terminal_path)
    expected_schema = (
        "duca_rime_phase1_dense_terminal_evaluation_v1"
        if variant in _DENSE_VARIANTS
        else "duca_rime_phase1_uniform_terminal_evaluation_v1"
    )
    baseline = terminal.get("baseline_contract")
    compatibility = terminal.get("checkpoint_compatibility")
    expected_mode = (
        STRICT_EXACT_CHECKPOINT_COMPATIBILITY_MODE
        if variant in _DENSE_VARIANTS
        else PHASE2_BASELINE_CHECKPOINT_COMPATIBILITY_MODE
    )
    expected_ignored = (
        []
        if variant in _DENSE_VARIANTS
        else sorted(PHASE2_BASELINE_IGNORED_UNEXPECTED_KEYS)
    )
    if (
        terminal.get("schema_version") != expected_schema
        or terminal.get("git_commit") != expected_commit
        or terminal.get("task") != "offline_temporal_action_detection"
        or terminal.get("variant") != variant
        or terminal.get("runtime_gt_input_to_selector") is not False
        or terminal.get("padded_to_kmax") is not False
        or terminal.get("training_identity") is not None
        or int(terminal.get("checkpoint_epoch", -1)) != 59
        or terminal.get("checkpoint_state_key") != "state_dict_ema"
        or not isinstance(baseline, Mapping)
        or int(baseline.get("phase", -1)) != 1
        or baseline.get("variant") != variant
        or baseline.get("uses_official_final") is not False
        or baseline.get("padded_to_kmax") is not False
        or not isinstance(compatibility, Mapping)
        or compatibility.get("mode") != expected_mode
        or compatibility.get("missing_keys") != []
        or compatibility.get("ignored_unexpected_keys") != expected_ignored
    ):
        raise ValueError(f"invalid Phase-1 terminal evaluation: {terminal_path}")

    checkpoint_path = _verify_binding(
        {
            "path": terminal.get("checkpoint_path"),
            "sha256": terminal.get("checkpoint_sha256"),
        },
        f"{variant} checkpoint",
    )
    config_path = _verify_binding(
        {
            "path": terminal.get("config_path"),
            "sha256": terminal.get("config_sha256"),
        },
        f"{variant} config",
    )
    prediction_path = _verify_binding(
        {
            "path": metrics.get("prediction_path"),
            "sha256": metrics.get("prediction_sha256"),
        },
        f"{variant} predictions",
    )
    return {
        "path": metrics_path,
        "payload": metrics,
        "terminal_path": terminal_path,
        "terminal": terminal,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "prediction_path": prediction_path,
        "aggregate_metrics": _mean_video_metrics(metrics),
        "source_artifacts": [
            _binding(metrics_path),
            _binding(terminal_path),
            _binding(config_path),
            _binding(checkpoint_path),
            _binding(prediction_path),
        ],
    }


def _validate_ledger(
    path: str | Path,
    *,
    expected_k: int,
    expected_videos: Sequence[str],
) -> dict[str, Any]:
    summary_path, summary = _load_json(path)
    ledger_path = _verify_binding(
        {"path": summary.get("path"), "sha256": summary.get("sha256")},
        f"K={expected_k} inference ledger",
    )
    if (
        summary.get("schema_version") != LEDGER_SUMMARY_SCHEMA
        or summary.get("status") != "sealed"
        or summary.get("arm") != "exact_uniform"
        or summary.get("no_padding_ledger") is not True
        or summary.get("all_observed_gaps_within_cap") is not True
        or summary.get("official_final_labels_used_for_decision") is not False
        or float(summary.get("requested_mean_k", math.nan)) != float(expected_k)
        or float(summary.get("effective_mean_k", math.nan)) != float(expected_k)
    ):
        raise ValueError(f"invalid K={expected_k} ledger summary: {summary_path}")

    rows = []
    identities = set()
    videos = set()
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        prefix = f"{ledger_path}:{line_number}"
        provenance = row.get("provenance")
        positions = [int(value) for value in row.get("selected_dense_indices", ())]
        dense_valid_len = int(row.get("dense_valid_len", -1))
        video = str(row.get("video_id", ""))
        start = int(row.get("window_start_frame", -1))
        observed_gap = float(row.get("observed_max_gap_seconds", math.nan))
        gap_cap = float(row.get("max_gap_seconds_cap", math.nan))
        if (
            not isinstance(row, Mapping)
            or row.get("schema_version") != LEDGER_SCHEMA
            or row.get("arm") != "exact_uniform"
            or not video
            or start < 0
            or (video, start) in identities
            or dense_valid_len < expected_k
            or any(
                int(row.get(key, -1)) != expected_k
                for key in (
                    "requested_k",
                    "effective_k",
                    "unique_k",
                    "backbone_input_k",
                    "padded_k",
                )
            )
            or positions != sorted(set(positions))
            or len(positions) != expected_k
            or positions != exact_uniform_positions(dense_valid_len, expected_k)
            or not math.isfinite(observed_gap)
            or not math.isfinite(gap_cap)
            or observed_gap < 0.0
            or gap_cap < 0.0
            or observed_gap > gap_cap + 1.0e-8
            or abs(observed_gap - gap_cap) > 1.0e-8
            or not isinstance(provenance, Mapping)
            or any(
                bool(provenance.get(key, False))
                for key in (
                    "uses_gt",
                    "uses_teacher",
                    "uses_prediction_cache",
                    "uses_test_batch_composition",
                    "raw_predictions_stored",
                )
            )
        ):
            raise ValueError(f"{prefix}: exact native-K ledger violation")
        identities.add((video, start))
        videos.add(video)
        rows.append(row)
    if (
        not rows
        or len(rows) != int(summary.get("record_count", -1))
        or videos != set(expected_videos)
        or int(summary.get("video_count", -1)) != len(expected_videos)
        or summary.get("requested_k_histogram")
        != {str(expected_k): len(rows)}
        or abs(
            float(summary.get("max_observed_gap_seconds", math.nan))
            - max(float(row["observed_max_gap_seconds"]) for row in rows)
        )
        > 1.0e-8
        or abs(
            float(summary.get("max_gap_seconds_cap", math.nan))
            - max(float(row["max_gap_seconds_cap"]) for row in rows)
        )
        > 1.0e-8
    ):
        raise ValueError(f"K={expected_k} ledger coverage or histogram drift")
    return {
        "summary_path": summary_path,
        "summary": summary,
        "ledger_path": ledger_path,
        "record_count": len(rows),
        "video_count": len(videos),
        "max_observed_gap_seconds": max(
            float(row["observed_max_gap_seconds"]) for row in rows
        ),
        "source_artifacts": [_binding(summary_path), _binding(ledger_path)],
    }


def _validate_geometry(
    path: str | Path,
    *,
    expected_commit: str,
    assignment_sha256: str,
) -> dict[str, Any]:
    geometry_path, geometry = _load_json(path)
    _verify_content_hash(geometry, "Phase-1 geometry audit")
    checks = geometry.get("checks")
    if (
        geometry.get("schema_version") != GEOMETRY_SCHEMA
        or geometry.get("status") != "passed"
        or geometry.get("gate_pass") is not True
        or geometry.get("git_commit") != expected_commit
        or geometry.get("split_assignment_sha256") != assignment_sha256
        or geometry.get("uses_official_final") is not False
        or not isinstance(checks, Mapping)
        or checks.get("remap_before_official_nms") is not True
        or int(checks.get("official_nms_call_count", -1)) != 1
        or float(checks.get("pre_nms_remap_max_abs", math.inf)) > 1.0e-6
        or float(checks.get("coordinate_roundtrip_max_abs", math.inf))
        > 1.0e-6
        or int(checks.get("roundtrip_violation_count", -1)) != 0
        or float(checks.get("physical_head_passthrough_max_abs", math.inf))
        > 1.0e-6
        or checks.get("physical_head_output_remapped_twice") is not False
        or int(checks.get("max_gap_violation_count", -1)) != 0
    ):
        raise ValueError("Phase-1 q-to-time geometry audit failed validation")
    sources = geometry.get("source_artifacts")
    if not isinstance(sources, Mapping) or set(sources) != {
        "single_stage_detector",
        "true_time_geometry",
        "official_nms",
    }:
        raise ValueError("Phase-1 geometry source bindings are incomplete")
    source_artifacts = [_binding(geometry_path)]
    for name, binding in sorted(sources.items()):
        source_artifacts.append(_binding(_verify_binding(binding, name)))
    return {
        "path": geometry_path,
        "payload": geometry,
        "checks": dict(checks),
        "source_artifacts": source_artifacts,
    }


def _validate_full_model_gate(
    path: str | Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    gate_path, gate = _load_json(path)
    runtime = gate.get("runtime")
    parity = gate.get("exact_uniform_physical_legacy_parity")
    if (
        gate.get("schema") != FULL_MODEL_GATE_SCHEMA
        or gate.get("ok") is not True
        or gate.get("status") != "p1_p2_full_model_gate_passed"
        or not isinstance(runtime, Mapping)
        or runtime.get("git_commit") != expected_commit
        or gate.get("hard_forward_equals_real_backbone_input") is not True
        or gate.get("paper_claim_allowed") is not False
        or not isinstance(parity, Mapping)
        or parity.get("target_assignment_parity") is not True
        or parity.get("decode_parity") is not True
        or parity.get("target_and_decode_parity") is not True
    ):
        raise ValueError("protected physical full-model parity gate is invalid")
    proposal_errors = []
    score_errors = []
    target_errors = []
    for name in ("full_window", "short_padded_window"):
        row = parity.get(name)
        target = row.get("target_assignment") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or row.get("target_and_decode_parity") is not True
            or row.get("target_assignment_parity") is not True
            or row.get("decode_parity") is not True
            or not isinstance(target, Mapping)
            or target.get("classification_targets_equal") is not True
            or target.get("positive_masks_equal") is not True
            or target.get("physical_regression_targets_equal") is not True
        ):
            raise ValueError(f"full-model parity failed on {name}")
        proposal_errors.append(float(row.get("proposal_max_abs_error", math.inf)))
        score_errors.append(float(row.get("score_max_abs_error", math.inf)))
        target_errors.append(
            float(target.get("physical_regression_target_max_abs_error", math.inf))
        )
    if (
        max(proposal_errors) > 1.0e-4
        or max(score_errors) > 1.0e-6
        or max(target_errors) > 1.0e-4
    ):
        raise ValueError("full-model exact-uniform parity exceeds tolerance")
    perturbation = gate.get("unselected_perturbation_audit")
    if (
        not isinstance(perturbation, Mapping)
        or perturbation.get("hard_gather_equal") is not True
        or float(perturbation.get("max_abs_error", math.inf)) > 1.0e-6
    ):
        raise ValueError("full-model hard gather parity failed")
    return {
        "path": gate_path,
        "payload": gate,
        "checks": {
            "mask_equal": True,
            "tensor_max_abs": float(perturbation["max_abs_error"]),
            "raw_proposal_max_abs": max(proposal_errors),
            "raw_score_max_abs": max(score_errors),
            "physical_target_max_abs": max(target_errors),
            "target_assignment_parity": True,
            "decode_parity": True,
            "full_and_short_padded_windows_covered": True,
        },
        "source_artifacts": [_binding(gate_path)],
    }


def _validate_profile(
    path: str | Path,
    *,
    control: str,
    expected_commit: str,
) -> dict[str, Any]:
    profile_path, profile = _load_json(path)
    rebuilt = validate_and_rebuild_profile_summary(profile)
    method, contract_name, probe_executed, drop_prefixes = _PROFILE_CONTRACTS[
        control
    ]
    contract = profile.get("phase1_cost_contract")
    stages = profile.get("stages")
    selected = profile.get("selected_count")
    if (
        profile.get("schema_version") != PROFILE_SCHEMA_VERSION
        or profile.get("method") != method
        or profile.get("config_commit") != expected_commit
        or profile.get("evidence_git_commit") != expected_commit
        or profile.get("tracked_tree_clean") is not True
        or int(profile.get("research_phase", -1)) != 1
        or profile.get("uses_official_final") is not False
        or profile.get("accuracy_claim_allowed") is not False
        or profile.get("random_init") is not False
        or profile.get("uses_ema") is not True
        or profile.get("amp") is not True
        or int(profile.get("loader_workers", -1)) != 0
        or int(profile.get("batch_size", -1)) != 1
        or int(profile.get("sample_count", -1)) < 30
        or int(profile.get("warmup_samples", -1)) < 5
        or int(profile.get("checkpoint_epoch", -1)) != 59
        or profile.get("checkpoint_state_key") != "state_dict_ema"
        or tuple(profile.get("checkpoint_dropped_prefixes", ()))
        != drop_prefixes
        or int(profile.get("checkpoint_dropped_key_count", 0)) <= 0
        or not isinstance(contract, Mapping)
        or contract.get("contract") != contract_name
        or contract.get("coarse_probe_executed") is not probe_executed
        or contract.get("selection_policy") != "exact_uniform"
        or contract.get("paired_checkpoint_identity_required") is not True
        or contract.get("accuracy_claim_allowed") is not False
        or not isinstance(stages, Mapping)
        or not isinstance(selected, Mapping)
        or abs(float(selected.get("mean", math.inf)) - 384.0) > 1.0e-6
    ):
        raise ValueError(f"invalid {control} full-stack cost profile")
    for stage in (
        "end_to_end_serial_ms",
        "model_forward_ms",
        "frame_selector_total_ms",
        "coarse_probe_ms",
        "heavy_backbone_ms",
        "postprocess_ms",
    ):
        if (
            stage not in stages
            or int(stages[stage].get("count", -1)) != int(profile["sample_count"])
            or not math.isfinite(float(stages[stage].get("p50", math.nan)))
        ):
            raise ValueError(f"{control} lacks reconstructable stage {stage}")
    coarse_p50 = float(stages["coarse_probe_ms"]["p50"])
    if (probe_executed and coarse_p50 <= 0.0) or (
        not probe_executed and coarse_p50 != 0.0
    ):
        raise ValueError(f"{control} coarse-probe execution evidence is invalid")
    checkpoint_path = _verify_binding(
        {
            "path": profile.get("checkpoint_path"),
            "sha256": profile.get("checkpoint_sha256"),
        },
        f"{control} checkpoint",
    )
    config_path = _verify_binding(
        {
            "path": profile.get("config_path"),
            "sha256": profile.get("profile_config_sha256"),
        },
        f"{control} config",
    )
    return {
        "path": profile_path,
        "payload": profile,
        "rebuild_hashes": rebuilt,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "source_artifacts": [
            _binding(profile_path),
            _binding(checkpoint_path),
            _binding(config_path),
        ],
    }


def build_phase1_controls(
    *,
    expected_commit: str,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    split_role: str,
    released_dense_metrics: str | Path,
    local_dense_metrics: str | Path,
    uniform_k384_metrics: str | Path,
    uniform_k384_ledger_summary: str | Path,
    uniform_k192_metrics: str | Path,
    uniform_k192_ledger_summary: str | Path,
    wrapper_gate: str | Path,
    geometry_audit: str | Path,
    no_probe_profile: str | Path,
    probe_profile: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)) is None:
        raise ValueError("Phase-1 controls require an exact Git commit")
    split_validation = validate_rime_splits(
        split_manifest,
        expected_sha256=split_manifest_sha256,
    )
    split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    role = split.get("train_roles", {}).get(str(split_role))
    if not isinstance(role, Mapping):
        raise ValueError("Phase-1 control split role is not registered")
    expected_videos = [str(value) for value in role["videos"]]
    common = {
        "schema_version": PHASE1_CONTROL_SCHEMA,
        "gate_pass": True,
        "git_commit": str(expected_commit),
        "split_manifest_sha256": split_validation["manifest_sha256"],
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "split_role": str(split_role),
        "evaluation_video_ids": expected_videos,
        "uses_official_final": False,
    }

    metric_specs = {
        "released_dense": (released_dense_metrics, 768.0),
        "local_dense": (local_dense_metrics, 768.0),
        "uniform_k384": (uniform_k384_metrics, 384.0),
        "uniform_k192": (uniform_k192_metrics, 192.0),
    }
    measurements = {
        name: _validate_metrics(
            path,
            expected_commit=expected_commit,
            split_validation=split_validation,
            split_role=split_role,
            expected_videos=expected_videos,
            variant=name,
            target_cost=target,
        )
        for name, (path, target) in metric_specs.items()
    }
    ledgers = {
        "uniform_k384": _validate_ledger(
            uniform_k384_ledger_summary,
            expected_k=384,
            expected_videos=expected_videos,
        ),
        "uniform_k192": _validate_ledger(
            uniform_k192_ledger_summary,
            expected_k=192,
            expected_videos=expected_videos,
        ),
    }
    geometry = _validate_geometry(
        geometry_audit,
        expected_commit=expected_commit,
        assignment_sha256=split_validation["assignment_sha256"],
    )
    parity = _validate_full_model_gate(
        wrapper_gate,
        expected_commit=expected_commit,
    )
    profiles = {
        "no_probe_uniform_cost": _validate_profile(
            no_probe_profile,
            control="no_probe_uniform_cost",
            expected_commit=expected_commit,
        ),
        "probe_uniform_cost": _validate_profile(
            probe_profile,
            control="probe_uniform_cost",
            expected_commit=expected_commit,
        ),
    }
    left = profiles["no_probe_uniform_cost"]["payload"]
    right = profiles["probe_uniform_cost"]["payload"]
    for key in (
        "protocol",
        "hardware_fingerprint",
        "host_fingerprint",
        "software_fingerprint",
        "profile_session_id",
        "profile_pair_id",
        "profile_repeat_index",
        "sample_count",
        "loader_workers",
        "amp",
        "uses_ema",
    ):
        if left.get(key) != right.get(key):
            raise ValueError(f"Phase-1 cost profiles are not paired on {key}")
    if {
        int(left.get("profile_order_position", -1)),
        int(right.get("profile_order_position", -1)),
    } != {1, 2}:
        raise ValueError("Phase-1 cost profiles do not occupy paired order slots")

    payloads: dict[str, dict[str, Any]] = {}
    for name in ("released_dense", "local_dense"):
        measurement = measurements[name]
        terminal = measurement["terminal"]
        payloads[name] = {
            **common,
            "control": name,
            "measurement": {
                "kind": "dense_sanity_control",
                "native_heavy_rgb_frames": 768,
                "checkpoint_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "checkpoint_compatibility_mode": (
                    STRICT_EXACT_CHECKPOINT_COMPATIBILITY_MODE
                ),
                "checkpoint_sha256": terminal["checkpoint_sha256"],
                "aggregate_metrics": measurement["aggregate_metrics"],
            },
            "source_artifacts": measurement["source_artifacts"],
            "claim_scope": "phase1_dense_sanity_control_only",
        }
    for name, expected_k in _UNIFORM_VARIANTS.items():
        measurement = measurements[name]
        ledger = ledgers[name]
        payloads[name] = {
            **common,
            "control": name,
            "measurement": {
                "kind": "exact_uniform_native_k_control",
                "native_heavy_rgb_frames": expected_k,
                "checkpoint_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "checkpoint_sha256": measurement["terminal"][
                    "checkpoint_sha256"
                ],
                "aggregate_metrics": measurement["aggregate_metrics"],
            },
            "cost_ledger": {
                "requested_k": expected_k,
                "effective_k": expected_k,
                "unique_k": expected_k,
                "backbone_input_k": expected_k,
                "padded_k": expected_k,
                "record_count": ledger["record_count"],
                "video_count": ledger["video_count"],
                "max_observed_gap_seconds": ledger[
                    "max_observed_gap_seconds"
                ],
                "constant_evidence_exact_uniform_identity": True,
            },
            "source_artifacts": [
                *measurement["source_artifacts"],
                *ledger["source_artifacts"],
            ],
            "claim_scope": "phase1_exact_uniform_execution_and_localization_control",
        }
    payloads["wrapper_parity"] = {
        **common,
        "control": "wrapper_parity",
        "checks": {
            **parity["checks"],
            "coordinate_roundtrip_max_abs": float(
                geometry["checks"]["coordinate_roundtrip_max_abs"]
            ),
            "remap_before_official_nms": True,
        },
        "source_artifacts": [
            *parity["source_artifacts"],
            *geometry["source_artifacts"],
        ],
        "claim_scope": "phase1_full_model_wrapper_infrastructure_parity_only",
    }
    payloads["q_to_t_before_nms"] = {
        **common,
        "control": "q_to_t_before_nms",
        "checks": geometry["checks"],
        "gap_audit": geometry["payload"]["gap_audit"],
        "source_artifacts": geometry["source_artifacts"],
        "claim_scope": "phase1_coordinate_order_and_gap_infrastructure_only",
    }
    for name, profile in profiles.items():
        report = profile["payload"]
        stages = report["stages"]
        payloads[name] = {
            **common,
            "control": name,
            "measurement": {
                "kind": "real_paired_full_stack_cost",
                "method": report["method"],
                "protocol": report["protocol"],
                "sample_count": int(report["sample_count"]),
                "warmup_samples": int(report["warmup_samples"]),
                "profile_session_id": report["profile_session_id"],
                "profile_pair_id": report["profile_pair_id"],
                "profile_repeat_index": int(report["profile_repeat_index"]),
                "profile_order_position": int(report["profile_order_position"]),
                "hardware_fingerprint": report["hardware_fingerprint"],
                "coarse_probe_executed": report[
                    "phase1_cost_contract"
                ]["coarse_probe_executed"],
                "selection_policy": "exact_uniform",
                "selected_count_mean": float(
                    report["selected_count"]["mean"]
                ),
                "end_to_end_p50_ms": float(
                    stages["end_to_end_serial_ms"]["p50"]
                ),
                "frame_selector_p50_ms": float(
                    stages["frame_selector_total_ms"]["p50"]
                ),
                "coarse_probe_p50_ms": float(
                    stages["coarse_probe_ms"]["p50"]
                ),
                "heavy_backbone_p50_ms": float(
                    stages["heavy_backbone_ms"]["p50"]
                ),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "checkpoint_dropped_prefixes": report[
                    "checkpoint_dropped_prefixes"
                ],
                "checkpoint_dropped_key_count": int(
                    report["checkpoint_dropped_key_count"]
                ),
                "summary_rebuild_hashes": profile["rebuild_hashes"],
            },
            "source_artifacts": profile["source_artifacts"],
            "claim_scope": "phase1_real_cost_decomposition_no_accuracy_claim",
        }

    target_dir = Path(output_dir).expanduser().resolve()
    outputs = {
        name: _write_immutable(target_dir / f"{name}.json", payload)
        for name, payload in payloads.items()
    }
    manifest = {
        "schema_version": "duca_rime_phase1_control_build_v1",
        "status": "passed",
        "git_commit": expected_commit,
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "control_artifacts": outputs,
        "control_names": sorted(outputs),
        "uses_official_final": False,
        "claim_scope": "phase1_control_evidence_build_only",
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    manifest_path = target_dir / "control_build_manifest.json"
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"refusing to overwrite a different control manifest: {manifest_path}"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(text, encoding="utf-8")
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "payload": manifest,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build source-bound DUCA-RIME Phase-1 controls."
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--split-role", required=True)
    parser.add_argument("--released-dense-metrics", required=True)
    parser.add_argument("--local-dense-metrics", required=True)
    parser.add_argument("--uniform-k384-metrics", required=True)
    parser.add_argument("--uniform-k384-ledger-summary", required=True)
    parser.add_argument("--uniform-k192-metrics", required=True)
    parser.add_argument("--uniform-k192-ledger-summary", required=True)
    parser.add_argument("--wrapper-gate", required=True)
    parser.add_argument("--geometry-audit", required=True)
    parser.add_argument("--no-probe-profile", required=True)
    parser.add_argument("--probe-profile", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = build_phase1_controls(
        expected_commit=args.expected_commit,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        split_role=args.split_role,
        released_dense_metrics=args.released_dense_metrics,
        local_dense_metrics=args.local_dense_metrics,
        uniform_k384_metrics=args.uniform_k384_metrics,
        uniform_k384_ledger_summary=args.uniform_k384_ledger_summary,
        uniform_k192_metrics=args.uniform_k192_metrics,
        uniform_k192_ledger_summary=args.uniform_k192_ledger_summary,
        wrapper_gate=args.wrapper_gate,
        geometry_audit=args.geometry_audit,
        no_probe_profile=args.no_probe_profile,
        probe_profile=args.probe_profile,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
