from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from tools.bata.create_duca_rime_splits import validate_rime_splits


RECEIPT_SCHEMA = "duca_rime_stage_receipt_v1"
PHASE1_CONTROL_SCHEMA = "duca_rime_phase1_control_v1"
PHASE3_RESULT_SCHEMA = "duca_rime_phase3_arm_result_v1"
PHASE4_RESULT_SCHEMA = "duca_rime_phase4_result_v1"
REQUIRED_PHASE1_CONTROLS = (
    "released_dense",
    "local_dense",
    "uniform_k384",
    "uniform_k192",
    "wrapper_parity",
    "q_to_t_before_nms",
    "no_probe_uniform_cost",
    "probe_uniform_cost",
)
PHASE3_ARMS = (
    "U-fixed",
    "U-same-K",
    "F-bound",
    "D-shuffle",
    "D-no-risk",
    "AdapTok-TAD",
    "RIME-full",
)
PHASE3_TRAIN_ARMS = tuple(arm for arm in PHASE3_ARMS if arm != "U-same-K")
PHASE3_METRICS = ("avg_map", "map_0.7", "short_map", "pair_support")
PHASE4_METRICS = (
    "avg_map",
    "map_0.6",
    "map_0.7",
    "short_map",
    "medium_map",
    "long_map",
    "boundary_error",
    "pair_support",
    "max_gap_seconds",
)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


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
        raise ValueError(f"JSON artifact must be an object: {resolved}")
    return resolved, payload


def _load_jsonl(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain nonempty object records: {resolved}")
    return resolved, rows


def _write_immutable(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different RIME receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "payload": dict(payload),
    }


def _parse_key_value_receipt(path: str | Path) -> tuple[Path, dict[str, str]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    values = {}
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, separator, value = line.partition("=")
            if not separator or not key or key in values:
                raise ValueError("invalid key=value RIME code-gate receipt")
            values[key] = value
    return resolved, values


def _artifact(path: str | Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _verify_content_sha256(payload: Mapping[str, Any], label: str) -> None:
    unsigned = dict(payload)
    embedded = unsigned.pop("content_sha256", None)
    if embedded != _canonical_sha256(unsigned):
        raise ValueError(f"{label} content hash is invalid")


def _load_bound_artifact(
    artifacts: Mapping[str, Any],
    name: str,
) -> tuple[Path, dict[str, Any]]:
    binding = artifacts.get(name)
    if not isinstance(binding, Mapping):
        raise ValueError(f"Phase-4 {name} artifact binding is missing")
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    expected_sha256 = str(binding.get("sha256", ""))
    if (
        not path.is_file()
        or len(expected_sha256) != 64
        or _sha256_file(path) != expected_sha256
    ):
        raise ValueError(f"Phase-4 {name} artifact binding drifted")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase-4 {name} artifact must be a JSON object")
    return path, payload


def seal_phase1(
    *,
    expected_commit: str,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    phase0_summary: str | Path,
    code_gate_receipt: str | Path,
    controls: Sequence[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    if len(str(expected_commit)) != 40:
        raise ValueError("Phase-1 requires an exact Git commit")
    split_validation = validate_rime_splits(
        split_manifest,
        expected_sha256=split_manifest_sha256,
    )
    phase0_path, phase0 = _load_json(phase0_summary)
    if (
        phase0.get("schema_version") != "duca_rime_causal_gate_summary_v1"
        or phase0.get("stage") != "phase0_variance_power"
        or phase0.get("gate_pass") is not True
    ):
        raise ValueError("Phase-0 variance/power summary is not sealed")
    code_path, code_gate = _parse_key_value_receipt(code_gate_receipt)
    if (
        code_gate.get("schema") != "duca_rime_code_gate_v1"
        or code_gate.get("status") != "passed"
        or code_gate.get("commit") != str(expected_commit)
    ):
        raise ValueError("RIME code gate did not pass at the expected commit")

    by_name = {}
    artifacts = []
    for control_path in controls:
        path, row = _load_json(control_path)
        name = str(row.get("control"))
        if (
            row.get("schema_version") != PHASE1_CONTROL_SCHEMA
            or not name
            or name in by_name
            or row.get("gate_pass") is not True
            or row.get("git_commit") != str(expected_commit)
            or row.get("split_assignment_sha256")
            != split_validation["assignment_sha256"]
            or row.get("uses_official_final") is not False
        ):
            raise ValueError(f"invalid or contaminated Phase-1 control: {path}")
        by_name[name] = row
        artifacts.append(_artifact(path))
    if set(by_name) != set(REQUIRED_PHASE1_CONTROLS):
        raise ValueError("Phase-1 controls are incomplete or contain unregistered roles")

    for name, expected_k in (("uniform_k384", 384), ("uniform_k192", 192)):
        ledger = by_name[name].get("cost_ledger")
        if (
            not isinstance(ledger, Mapping)
            or int(ledger.get("requested_k", -1)) != expected_k
            or int(ledger.get("effective_k", -1)) != expected_k
            or int(ledger.get("unique_k", -1)) != expected_k
            or int(ledger.get("backbone_input_k", -1)) != expected_k
            or int(ledger.get("padded_k", -1)) != expected_k
            or ledger.get("constant_evidence_exact_uniform_identity") is not True
        ):
            raise ValueError(f"{name} violates exact native-K execution")
    parity = by_name["wrapper_parity"].get("checks")
    if (
        not isinstance(parity, Mapping)
        or parity.get("mask_equal") is not True
        or float(parity.get("tensor_max_abs", math.inf)) > 1.0e-6
        or float(parity.get("raw_proposal_max_abs", math.inf)) > 1.0e-5
        or float(parity.get("coordinate_roundtrip_max_abs", math.inf)) > 1.0e-6
        or float(parity.get("map_abs_delta", math.inf)) > 1.0e-6
    ):
        raise ValueError("clean/wrapper parity is outside the frozen tolerance")
    geometry = by_name["q_to_t_before_nms"].get("checks")
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("remap_before_official_nms") is not True
        or int(geometry.get("roundtrip_violation_count", -1)) != 0
        or int(geometry.get("max_gap_violation_count", -1)) != 0
    ):
        raise ValueError("q -> physical time -> official NMS contract failed")

    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "phase": "phase1",
        "status": "passed",
        "gate_pass": True,
        "git_commit": str(expected_commit),
        "split_manifest": _artifact(split_manifest),
        "split_assignment_sha256": split_validation["assignment_sha256"],
        "phase0_summary": _artifact(phase0_path),
        "phase0_thresholds": dict(phase0["rule_derived_thresholds"]),
        "code_gate_receipt": _artifact(code_path),
        "control_artifacts": artifacts,
        "control_names": list(REQUIRED_PHASE1_CONTROLS),
        "official_final_subset_consumed": False,
        "phase2_authorized": True,
        "phase3_training_authorized": False,
        "claim_scope": "infrastructure_and_clean_controls_only",
    }
    return _write_immutable(output, payload)


def seal_phase2(
    *,
    phase1_receipt: str | Path,
    summaries: Sequence[str | Path],
    budget_protocol: str | Path | None = None,
    budget_protocols: Sequence[str | Path] | None = None,
    output: str | Path,
) -> dict[str, Any]:
    phase1_path, phase1 = _load_json(phase1_receipt)
    if (
        phase1.get("schema_version") != RECEIPT_SCHEMA
        or phase1.get("phase") != "phase1"
        or phase1.get("gate_pass") is not True
        or phase1.get("phase2_authorized") is not True
        or phase1.get("official_final_subset_consumed") is not False
    ):
        raise ValueError("Phase-2 is blocked by its Phase-1 receipt")
    evidence = {}
    evidence_hashes = {}
    evidence_artifacts = []
    for summary_path in summaries:
        path, row = _load_json(summary_path)
        stage = str(row.get("stage"))
        if (
            row.get("schema_version") != "duca_rime_causal_gate_summary_v1"
            or row.get("gate_pass") is not True
            or stage in evidence
        ):
            raise ValueError(f"invalid or failed Phase-2 gate: {path}")
        evidence[stage] = row
        artifact = _artifact(path)
        evidence_hashes[stage] = artifact["sha256"]
        evidence_artifacts.append(artifact)
    required = {
        "o1_dynamic_budget_headroom",
        "o2_decoder_family_regret",
        "o3_cross_fitted_hard_utility_rank",
        "o4_pair_risk_calibration",
    }
    if set(evidence) != required:
        raise ValueError("Phase-2 requires exactly O1, O2, O3, and O4")
    supplied_protocols = list(budget_protocols or ())
    if budget_protocol is not None:
        supplied_protocols.append(budget_protocol)
    if not supplied_protocols:
        raise ValueError("Phase-2 requires frozen budget protocols")
    protocol_rows = []
    protocol_by_target = {}
    common_grid = None
    for supplied in supplied_protocols:
        protocol_path, protocol = _load_json(supplied)
        if (
            protocol.get("schema_version") != "duca_rime_budget_protocol_v1"
            or protocol.get("gate_pass") is not True
            or protocol.get("fit_split") != "train_only"
            or protocol.get("uses_validation_or_test_labels") is not False
            or protocol.get("decoder_family")
            != evidence["o2_decoder_family_regret"].get("selected_family")
        ):
            raise ValueError("frozen RIME protocol is invalid or disagrees with O2")
        protocol_evidence = {
            str(row.get("stage")): str(row.get("sha256"))
            for row in protocol.get("evidence_summaries", ())
        }
        if set(protocol_evidence) != required or evidence_hashes != protocol_evidence:
            raise ValueError(
                "frozen RIME protocol is not hash-bound to the supplied O1-O4 evidence"
            )
        target = float(protocol.get("target_mean_cost", math.nan))
        if target in protocol_by_target or not math.isfinite(target):
            raise ValueError("RIME protocol targets are duplicated or nonfinite")
        grid = (
            tuple(int(value) for value in protocol.get("candidate_budgets", ())),
            tuple(float(value) for value in protocol.get("candidate_costs", ())),
        )
        if common_grid is None:
            common_grid = grid
        elif grid != common_grid:
            raise ValueError("RIME formal budget protocols use different candidate grids")
        artifact = _artifact(protocol_path)
        protocol_rows.append(
            {
                "target_mean_cost": target,
                **artifact,
            }
        )
        protocol_by_target[target] = (protocol_path, protocol, artifact)
    if set(protocol_by_target) != {384.0, 192.0}:
        raise ValueError("Phase-2 must freeze the registered 384 and 192 budget panels")
    protocol_path, protocol, development_artifact = protocol_by_target[384.0]

    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "phase": "phase2",
        "status": "passed",
        "gate_pass": True,
        "git_commit": phase1["git_commit"],
        "split_manifest": dict(phase1["split_manifest"]),
        "split_assignment_sha256": phase1["split_assignment_sha256"],
        "phase1_receipt": _artifact(phase1_path),
        "phase0_thresholds": dict(phase1["phase0_thresholds"]),
        "gate_artifacts": evidence_artifacts,
        "budget_protocol": development_artifact,
        "formal_budget_protocols": sorted(
            protocol_rows,
            key=lambda row: -float(row["target_mean_cost"]),
        ),
        "candidate_budgets": list(protocol["candidate_budgets"]),
        "candidate_costs": list(protocol["candidate_costs"]),
        "target_mean_cost": float(protocol["target_mean_cost"]),
        "decoder_family": str(protocol["decoder_family"]),
        "official_final_subset_consumed": False,
        "phase3_training_authorized": True,
        "phase4_authorized": False,
        "claim_scope": "oracle_and_calibration_evidence_only",
    }
    return _write_immutable(output, payload)


def _finite_metric_map(
    row: Mapping[str, Any],
    metric: str,
    expected_videos: set[str],
) -> dict[str, float]:
    metrics = row.get("video_metrics")
    values = None if not isinstance(metrics, Mapping) else metrics.get(metric)
    if not isinstance(values, Mapping) or set(str(key) for key in values) != expected_videos:
        raise ValueError(f"Phase-3 metric {metric} is missing or has video drift")
    output = {str(key): float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError(f"Phase-3 metric {metric} contains nonfinite values")
    return output


def _paired_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if set(left) != set(right) or len(left) < 3:
        raise ValueError("paired video bootstrap requires >=3 aligned videos")
    videos = sorted(left)
    differences = [float(left[video]) - float(right[video]) for video in videos]
    rng = random.Random(int(seed))
    draws = sorted(
        mean(rng.choice(differences) for _ in differences)
        for _ in range(max(1, int(samples)))
    )

    def percentile(q: float) -> float:
        rank = (len(draws) - 1) * q
        low, high = int(math.floor(rank)), int(math.ceil(rank))
        if low == high:
            return draws[low]
        return draws[low] + (draws[high] - draws[low]) * (rank - low)

    return {
        "mean": mean(differences),
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
        "video_count": len(videos),
        "bootstrap_samples": max(1, int(samples)),
    }


def seal_phase3(
    *,
    phase2_receipt: str | Path,
    results_jsonl: str | Path,
    output: str | Path,
    expected_seed: int,
    bootstrap_samples: int = 5000,
    cost_tolerance: float = 1.0,
) -> dict[str, Any]:
    phase2_path, phase2 = _load_json(phase2_receipt)
    if (
        phase2.get("schema_version") != RECEIPT_SCHEMA
        or phase2.get("phase") != "phase2"
        or phase2.get("gate_pass") is not True
        or phase2.get("phase3_training_authorized") is not True
        or phase2.get("official_final_subset_consumed") is not False
    ):
        raise ValueError("Phase-3 is blocked by its Phase-2 receipt")
    split_path = Path(phase2["split_manifest"]["path"])
    validate_rime_splits(
        split_path,
        expected_sha256=phase2["split_manifest"]["sha256"],
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development_videos = set(
        str(value)
        for value in split["train_roles"]["certification_development"]["videos"]
    )
    results_path, rows = _load_jsonl(results_jsonl)
    by_arm = {}
    initialization = set()
    exposure = set()
    for row in rows:
        arm = str(row.get("arm"))
        common_invalid = (
            row.get("schema_version") != PHASE3_RESULT_SCHEMA
            or arm in by_arm
            or arm not in PHASE3_ARMS
            or int(row.get("seed", -1)) != int(expected_seed)
            or row.get("uses_official_final") is not False
            or row.get("split_assignment_sha256")
            != phase2["split_assignment_sha256"]
            or row.get("padded_to_kmax") is not False
        )
        if arm == "U-same-K":
            execution_invalid = (
                row.get("evaluation_only") is not True
                or row.get("source_training_arm") != "RIME-full"
                or row.get("independent_training_run") is not False
                or int(row.get("successful_detector_updates", -1)) != 0
                or int(row.get("source_successful_detector_updates", -1)) != 6000
                or row.get("source_formal_update_audit_passed") is not True
                or not str(row.get("source_training_receipt_sha256", ""))
            )
        else:
            execution_invalid = (
                row.get("evaluation_only") is True
                or int(row.get("successful_detector_updates", -1)) != 6000
                or row.get("formal_update_audit_passed") is not True
            )
        if common_invalid or execution_invalid:
            raise ValueError("invalid, incomplete, or contaminated Phase-3 arm result")
        result_videos = set(str(value) for value in row.get("evaluation_video_ids", ()))
        if result_videos != development_videos:
            raise ValueError("Phase-3 must evaluate exactly the certification/development role")
        for metric in PHASE3_METRICS:
            _finite_metric_map(row, metric, development_videos)
        initialization.add(str(row.get("initialization_sha256")))
        exposure.add(str(row.get("training_exposure_sha256")))
        by_arm[arm] = row
    if set(by_arm) != set(PHASE3_ARMS):
        raise ValueError("Phase-3 seven-arm matrix is incomplete")
    same_k = by_arm["U-same-K"]
    full = by_arm["RIME-full"]
    if (
        same_k.get("source_training_receipt_sha256")
        != full.get("training_receipt_sha256")
        or same_k.get("initialization_sha256") != full.get("initialization_sha256")
        or same_k.get("training_exposure_sha256")
        != full.get("training_exposure_sha256")
    ):
        raise ValueError("U-same-K is not bound to the sealed RIME-full training run")
    if len(initialization) != 1 or "" in initialization or len(exposure) != 1 or "" in exposure:
        raise ValueError("Phase-3 arms do not share initialization and exposure")
    for arm in ("U-same-K", "D-shuffle"):
        if by_arm[arm].get("k_histogram") != by_arm["RIME-full"].get("k_histogram"):
            raise ValueError(f"{arm} does not preserve the RIME-full K histogram")
    for arm, row in by_arm.items():
        if arm != "RIME-full" and row.get("cost") is not None:
            raise ValueError("only RIME-full may carry the shared paired cost artifact")
    full_cost_evidence = by_arm["RIME-full"].get("cost")
    if not isinstance(full_cost_evidence, Mapping):
        raise ValueError("RIME-full lacks paired full-stack cost evidence")
    artifact_path = Path(str(full_cost_evidence.get("artifact_path", ""))).resolve()
    artifact_sha = str(full_cost_evidence.get("artifact_sha256", ""))
    if (
        full_cost_evidence.get("schema_version")
        != "duca_rime_paired_full_stack_cost_v1"
        or int(full_cost_evidence.get("research_phase", -1)) != 3
        or full_cost_evidence.get("arm") != "RIME-full"
        or int(full_cost_evidence.get("seed", -1)) != int(expected_seed)
        or full_cost_evidence.get("detector_backend") != "ActionFormer"
        or float(full_cost_evidence.get("target_mean_cost", math.nan)) != 384.0
        or full_cost_evidence.get("real_full_stack_measurement") is not True
        or full_cost_evidence.get("includes_probe_decoder_solver") is not True
        or full_cost_evidence.get("matched_realized_cost") is not True
        or float(full_cost_evidence.get("matched_k_tolerance", math.inf))
        > float(cost_tolerance)
        or not artifact_path.is_file()
        or _sha256_file(artifact_path) != artifact_sha
    ):
        raise ValueError("Phase-3 paired full-stack cost artifact is invalid")
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    embedded_cost = {
        key: value
        for key, value in full_cost_evidence.items()
        if key not in {"artifact_path", "artifact_sha256"}
    }
    if artifact_payload != embedded_cost:
        raise ValueError("Phase-3 embedded cost evidence differs from its artifact")

    metrics = {
        arm: {
            metric: _finite_metric_map(row, metric, development_videos)
            for metric in PHASE3_METRICS
        }
        for arm, row in by_arm.items()
    }
    comparisons = {}

    def compare(left: str, right: str, metric: str, seed_offset: int) -> dict[str, float]:
        result = _paired_bootstrap(
            metrics[left][metric],
            metrics[right][metric],
            samples=bootstrap_samples,
            seed=int(expected_seed) + seed_offset,
        )
        comparisons[f"{left}_minus_{right}_{metric}"] = result
        return result

    minimum_gain = float(phase2["phase0_thresholds"]["min_o1_headroom"])
    full_fixed = compare("RIME-full", "U-fixed", "avg_map", 1)
    inner_fixed = compare("F-bound", "U-fixed", "avg_map", 2)
    positions = compare("RIME-full", "U-same-K", "avg_map", 3)
    assignment = compare("RIME-full", "D-shuffle", "avg_map", 4)
    risk_high = compare("RIME-full", "D-no-risk", "map_0.7", 5)
    risk_short = compare("RIME-full", "D-no-risk", "short_map", 6)
    risk_pair = compare("RIME-full", "D-no-risk", "pair_support", 7)
    full_cost = float(full_cost_evidence.get("latency_p50_ms", math.nan))
    fixed_cost = float(full_cost_evidence.get("fixed_latency_p50_ms", math.nan))
    dense_cost = float(full_cost_evidence.get("dense_latency_p50_ms", math.nan))
    cost_gate = (
        all(
            math.isfinite(value) and value > 0.0
            for value in (full_cost, fixed_cost, dense_cost)
        )
        and full_cost < dense_cost
        and full_cost_evidence.get("matched_realized_cost") is True
    )
    contribution_gates = {
        "full_over_best_fixed": full_fixed["ci95_low"] > minimum_gain,
        "fixed_inner_over_uniform": inner_fixed["ci95_low"] > minimum_gain,
        "learned_positions": positions["ci95_low"] > minimum_gain,
        "content_conditioned_budget": assignment["ci95_low"] > minimum_gain,
        "pair_risk_high_iou": risk_high["ci95_low"] > 0.0,
        "pair_risk_short_non_degrade": risk_short["ci95_low"] >= 0.0,
        "pair_risk_support": risk_pair["ci95_low"] > 0.0,
        "real_full_stack_cost": cost_gate,
    }
    gate = all(contribution_gates.values())
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "phase": "phase3",
        "status": "go" if gate else "no_go",
        "gate_pass": gate,
        "git_commit": phase2["git_commit"],
        "phase2_receipt": _artifact(phase2_path),
        "results_jsonl": _artifact(results_path),
        "development_seed": int(expected_seed),
        "development_seed_excluded_from_formal_statistics": True,
        "successful_detector_updates_per_train_arm": 6000,
        "train_arms": list(PHASE3_TRAIN_ARMS),
        "evaluation_only_arms": ["U-same-K"],
        "shared_initialization_sha256": next(iter(initialization)),
        "shared_training_exposure_sha256": next(iter(exposure)),
        "phase0_minimum_primary_gain": minimum_gain,
        "comparisons": comparisons,
        "contribution_gates": contribution_gates,
        "realized_mean_cost": {
            "RIME-full": full_cost,
            "U-fixed": fixed_cost,
            "dense_reference": dense_cost,
            "matched_effective_k_tolerance": float(
                full_cost_evidence["matched_k_tolerance"]
            ),
        },
        "official_final_subset_consumed": False,
        "phase4_authorized": gate,
        "claim_scope": (
            "development_go_only_not_formal_evidence"
            if gate
            else "development_no_go_stop_before_formal"
        ),
    }
    return _write_immutable(output, payload)


def authorize_phase4(
    *,
    phase3_receipt: str | Path,
    output: str | Path,
    formal_seeds: Sequence[int],
) -> dict[str, Any]:
    phase3_path, phase3 = _load_json(phase3_receipt)
    seeds = tuple(int(value) for value in formal_seeds)
    phase2_path, phase2 = _load_json(phase3.get("phase2_receipt", {}).get("path", ""))
    formal_protocols = phase2.get("formal_budget_protocols")
    protocol_targets = (
        {
            float(row.get("target_mean_cost"))
            for row in formal_protocols
            if isinstance(row, Mapping)
        }
        if isinstance(formal_protocols, list)
        else set()
    )
    if (
        phase3.get("schema_version") != RECEIPT_SCHEMA
        or phase3.get("phase") != "phase3"
        or phase3.get("gate_pass") is not True
        or phase3.get("phase4_authorized") is not True
        or phase3.get("official_final_subset_consumed") is not False
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or int(phase3["development_seed"]) in seeds
        or phase3.get("phase2_receipt", {}).get("sha256") != _sha256_file(phase2_path)
        or phase2.get("schema_version") != RECEIPT_SCHEMA
        or phase2.get("phase") != "phase2"
        or protocol_targets != {384.0, 192.0}
    ):
        raise RuntimeError("Phase-4 submission is blocked by Phase-3 or invalid formal seeds")
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "phase": "phase4_authorization",
        "status": "authorized",
        "gate_pass": True,
        "git_commit": phase3["git_commit"],
        "phase3_receipt": _artifact(phase3_path),
        "phase2_receipt": _artifact(phase2_path),
        "formal_budget_protocols": formal_protocols,
        "formal_seeds": list(seeds),
        "required_detectors": ["ActionFormer", "TriDet"],
        "required_budget_panels": [384, 192],
        "official_final_subset_consumed": False,
        "paper_claim_allowed": False,
        "claim_scope": "formal_submission_authorization_only",
    }
    return _write_immutable(output, payload)


def _finite_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def seal_phase4(
    *,
    authorization_receipt: str | Path,
    results_jsonl: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    authorization_path, authorization = _load_json(authorization_receipt)
    if (
        authorization.get("schema_version") != RECEIPT_SCHEMA
        or authorization.get("phase") != "phase4_authorization"
        or authorization.get("status") != "authorized"
        or authorization.get("gate_pass") is not True
        or authorization.get("official_final_subset_consumed") is not False
        or authorization.get("paper_claim_allowed") is not False
    ):
        raise ValueError("Phase-4 results are blocked by their authorization")
    seeds = tuple(int(value) for value in authorization["formal_seeds"])
    detectors = tuple(str(value) for value in authorization["required_detectors"])
    budgets = tuple(float(value) for value in authorization["required_budget_panels"])
    expected_cells = {
        (detector, budget, seed)
        for detector in detectors
        for budget in budgets
        for seed in seeds
    }
    result_path, rows = _load_jsonl(results_jsonl)
    by_cell = {}
    final_video_ids = None
    authorization_sha256 = _sha256_file(authorization_path)
    for row in rows:
        detector = str(row.get("detector_backend"))
        budget = float(row.get("target_mean_cost", math.nan))
        seed = int(row.get("seed", -1))
        cell = (detector, budget, seed)
        if (
            row.get("schema_version") != PHASE4_RESULT_SCHEMA
            or cell not in expected_cells
            or cell in by_cell
            or row.get("git_commit") != authorization["git_commit"]
            or row.get("method_frozen_before_final_evaluation") is not True
            or row.get("development_seed_excluded") is not True
            or row.get("uses_official_final") is not True
            or row.get("official_final_used_for_training_or_selection") is not False
            or int(row.get("rime_successful_detector_updates", -1)) != 6000
            or int(row.get("fixed_successful_detector_updates", -1)) != 6000
            or int(row.get("same_k_successful_detector_updates", -1)) != 0
            or row.get("same_k_source_training_arm") != "RIME-full"
            or row.get("padded_to_kmax") is not False
        ):
            raise ValueError("invalid, incomplete, or contaminated Phase-4 cell")
        videos = tuple(sorted(str(value) for value in row.get("evaluation_video_ids", ())))
        if not videos or len(videos) != len(set(videos)):
            raise ValueError("Phase-4 cell has an invalid final-evaluation video set")
        if final_video_ids is None:
            final_video_ids = videos
        elif videos != final_video_ids:
            raise ValueError("Phase-4 cells do not share the official final video set")
        artifacts = row.get("artifacts")
        required_artifacts = {
            "authorization",
            "rime_metrics",
            "fixed_metrics",
            "same_k_metrics",
            "comparisons",
            "cost",
            "rime_ledger_summary",
        }
        if (
            not isinstance(artifacts, Mapping)
            or not required_artifacts.issubset(artifacts)
        ):
            raise ValueError("Phase-4 cell evidence artifacts are incomplete")
        artifact_payloads = {
            name: _load_bound_artifact(artifacts, name)
            for name in required_artifacts
        }
        if (
            str(artifacts["authorization"].get("sha256", ""))
            != authorization_sha256
            or artifact_payloads["authorization"][1] != authorization
        ):
            raise ValueError("Phase-4 cell is not bound to its authorization receipt")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("Phase-4 cell metrics are missing")
        for metric in PHASE4_METRICS:
            value = _finite_number(metrics.get(metric), f"Phase-4 {metric}")
            if metric in {"avg_map", "map_0.6", "map_0.7", "short_map", "medium_map", "long_map", "pair_support"} and not 0.0 <= value <= 1.0:
                raise ValueError(f"Phase-4 {metric} lies outside [0,1]")
            if metric in {"boundary_error", "max_gap_seconds"} and value < 0.0:
                raise ValueError(f"Phase-4 {metric} must be non-negative")
        comparisons = row.get("comparisons")
        if not isinstance(comparisons, Mapping):
            raise ValueError("Phase-4 paired comparisons are missing")
        comparison_artifact = artifact_payloads["comparisons"][1]
        _verify_content_sha256(comparison_artifact, "Phase-4 comparisons")
        if (
            comparison_artifact.get("schema_version")
            != "duca_rime_phase4_comparisons_v1"
            or comparison_artifact.get("git_commit") != authorization["git_commit"]
            or comparison_artifact.get("detector_backend") != detector
            or float(comparison_artifact.get("target_mean_cost", math.nan)) != budget
            or int(comparison_artifact.get("seed", -1)) != seed
            or tuple(
                sorted(
                    str(value)
                    for value in comparison_artifact.get(
                        "evaluation_video_ids", ()
                    )
                )
            )
            != videos
            or comparison_artifact.get(
                "official_final_used_for_training_or_selection"
            )
            is not False
            or comparison_artifact.get("comparisons") != comparisons
        ):
            raise ValueError(
                "Phase-4 embedded comparisons differ from their hash-bound artifact"
            )
        for name in ("rime_minus_best_fixed", "rime_minus_uniform_same_k"):
            comparison = comparisons.get(name)
            official_bootstrap = (
                None
                if not isinstance(comparison, Mapping)
                else comparison.get("official_map_bootstrap")
            )
            auxiliary_bootstrap = (
                None
                if not isinstance(comparison, Mapping)
                else comparison.get("auxiliary_video_bootstrap")
            )
            if (
                not isinstance(comparison, Mapping)
                or not isinstance(official_bootstrap, Mapping)
                or official_bootstrap.get(
                    "official_evaluator_reexecuted_per_resample"
                )
                is not True
                or official_bootstrap.get("paired_video_cluster_bootstrap")
                is not True
                or int(official_bootstrap.get("bootstrap_samples", 0)) < 1000
                or not isinstance(auxiliary_bootstrap, Mapping)
                or auxiliary_bootstrap.get(
                    "official_evaluator_reexecuted_per_resample"
                )
                is not False
                or auxiliary_bootstrap.get("paired_video_cluster_bootstrap")
                is not True
                or int(auxiliary_bootstrap.get("bootstrap_samples", 0)) < 1000
            ):
                raise ValueError(
                    f"Phase-4 {name} lacks separated official and auxiliary bootstraps"
                )
            for metric in ("avg_map", "map_0.7", "short_map", "pair_support"):
                interval = comparison.get(metric)
                if not isinstance(interval, Mapping):
                    raise ValueError(f"Phase-4 {name}/{metric} interval is missing")
                low = _finite_number(interval.get("ci95_low"), f"{name}/{metric} low")
                high = _finite_number(interval.get("ci95_high"), f"{name}/{metric} high")
                _finite_number(interval.get("mean"), f"{name}/{metric} mean")
                if low > high:
                    raise ValueError(f"Phase-4 {name}/{metric} interval is reversed")
        cost = row.get("cost")
        cost_artifact = artifact_payloads["cost"][1]
        _verify_content_sha256(cost_artifact, "Phase-4 cost evidence")
        expected_arm = "RIME-full-TriDet" if detector == "TriDet" else "RIME-full"
        matched_k_tolerance = _finite_number(
            cost_artifact.get("matched_k_tolerance"),
            "matched K tolerance",
        )
        candidate_k = _finite_number(
            cost_artifact.get("candidate_effective_mean_k"),
            "candidate effective mean K",
        )
        fixed_k = _finite_number(
            cost_artifact.get("fixed_effective_mean_k"),
            "fixed effective mean K",
        )
        if (
            not isinstance(cost, Mapping)
            or dict(cost) != cost_artifact
            or cost_artifact.get("schema_version")
            != "duca_rime_paired_full_stack_cost_v1"
            or int(cost_artifact.get("research_phase", -1)) != 4
            or cost_artifact.get("arm") != expected_arm
            or int(cost_artifact.get("seed", -1)) != seed
            or cost_artifact.get("detector_backend") != detector
            or float(cost_artifact.get("target_mean_cost", math.nan)) != budget
            or cost.get("real_full_stack_measurement") is not True
            or cost.get("matched_realized_cost") is not True
            or cost.get("includes_probe_decoder_solver") is not True
            or cost.get("official_final_labels_used_for_cost_decision") is not False
            or matched_k_tolerance < 0.0
            or matched_k_tolerance > 1.0
            or candidate_k <= 0.0
            or fixed_k <= 0.0
            or abs(candidate_k - fixed_k) > matched_k_tolerance
            or abs(candidate_k - budget) > matched_k_tolerance
            or abs(fixed_k - budget) > matched_k_tolerance
            or _finite_number(cost.get("latency_p50_ms"), "latency p50") <= 0.0
            or _finite_number(cost.get("latency_p95_ms"), "latency p95") <= 0.0
            or _finite_number(cost.get("throughput_videos_per_second"), "throughput")
            <= 0.0
            or _finite_number(cost.get("energy_joules_per_video"), "energy") <= 0.0
            or _finite_number(cost.get("peak_gpu_memory_mb"), "memory") <= 0.0
            or _finite_number(cost.get("fixed_latency_p50_ms"), "fixed latency")
            <= 0.0
            or _finite_number(cost.get("dense_latency_p50_ms"), "dense latency") <= 0.0
            or _finite_number(cost.get("dense_latency_p95_ms"), "dense latency p95")
            <= 0.0
            or cost.get("candidate_below_dense")
            is not (
                float(cost.get("latency_p50_ms"))
                < float(cost.get("dense_latency_p50_ms"))
            )
        ):
            raise ValueError("Phase-4 real full-stack cost evidence is incomplete")
        by_cell[cell] = row
    if set(by_cell) != expected_cells:
        raise ValueError("Phase-4 detector/budget/seed matrix is incomplete")

    cell_gates = {}
    direction = {}
    for cell, row in by_cell.items():
        comparisons = row["comparisons"]
        fixed = comparisons["rime_minus_best_fixed"]
        same_k = comparisons["rime_minus_uniform_same_k"]
        cost = row["cost"]
        gate = {
            "avg_map_over_best_fixed": float(fixed["avg_map"]["ci95_low"]) > 0.0,
            "avg_map_over_uniform_same_k": float(same_k["avg_map"]["ci95_low"]) > 0.0,
            "high_iou_non_degrade": float(fixed["map_0.7"]["ci95_low"]) >= 0.0,
            "short_non_degrade": float(fixed["short_map"]["ci95_low"]) >= 0.0,
            "pair_support_non_degrade": float(fixed["pair_support"]["ci95_low"]) >= 0.0,
            "real_cost_below_dense": float(cost["latency_p50_ms"])
            < float(cost["dense_latency_p50_ms"]),
        }
        cell_id = f"{cell[0]}|K{int(cell[1])}|seed{cell[2]}"
        cell_gates[cell_id] = gate
        direction.setdefault((cell[0], cell[1]), []).append(
            float(fixed["avg_map"]["mean"])
        )
    direction_summary = {
        f"{detector}|K{int(budget)}": {
            "mean_rime_minus_best_fixed_avg_map": mean(values),
            "all_seed_directions_positive": all(value > 0.0 for value in values),
        }
        for (detector, budget), values in direction.items()
    }
    gate_pass = (
        all(all(values.values()) for values in cell_gates.values())
        and all(
            row["all_seed_directions_positive"]
            for row in direction_summary.values()
        )
    )
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "phase": "phase4",
        "status": "paper_ready" if gate_pass else "formal_no_go",
        "gate_pass": gate_pass,
        "git_commit": authorization["git_commit"],
        "authorization_receipt": _artifact(authorization_path),
        "results_jsonl": _artifact(result_path),
        "formal_seeds": list(seeds),
        "detectors": list(detectors),
        "budget_panels": list(budgets),
        "cell_count": len(by_cell),
        "official_final_video_ids": list(final_video_ids or ()),
        "cell_gates": cell_gates,
        "cross_seed_detector_budget_direction": direction_summary,
        "official_final_subset_consumed": True,
        "paper_claim_allowed": gate_pass,
        "claim_scope": (
            "offline_tad_pre_backbone_acquisition_plugin_formal_evidence"
            if gate_pass
            else "formal_no_go_no_general_plugin_claim"
        ),
    }
    return _write_immutable(output, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal fail-closed DUCA-RIME stage receipts")
    sub = parser.add_subparsers(dest="command", required=True)

    phase1 = sub.add_parser("phase1")
    phase1.add_argument("--expected-commit", required=True)
    phase1.add_argument("--split-manifest", required=True)
    phase1.add_argument("--split-manifest-sha256", required=True)
    phase1.add_argument("--phase0-summary", required=True)
    phase1.add_argument("--code-gate-receipt", required=True)
    phase1.add_argument("--control", action="append", required=True)
    phase1.add_argument("--output", required=True)

    phase2 = sub.add_parser("phase2")
    phase2.add_argument("--phase1-receipt", required=True)
    phase2.add_argument("--summary", action="append", required=True)
    phase2.add_argument("--budget-protocol", action="append", required=True)
    phase2.add_argument("--output", required=True)

    phase3 = sub.add_parser("phase3")
    phase3.add_argument("--phase2-receipt", required=True)
    phase3.add_argument("--results-jsonl", required=True)
    phase3.add_argument("--output", required=True)
    phase3.add_argument("--expected-seed", type=int, required=True)
    phase3.add_argument("--bootstrap-samples", type=int, default=5000)
    phase3.add_argument("--cost-tolerance", type=float, default=1.0)

    phase4 = sub.add_parser("authorize-phase4")
    phase4.add_argument("--phase3-receipt", required=True)
    phase4.add_argument("--output", required=True)
    phase4.add_argument("--formal-seeds", nargs=3, type=int, required=True)

    phase4_results = sub.add_parser("phase4")
    phase4_results.add_argument("--authorization-receipt", required=True)
    phase4_results.add_argument("--results-jsonl", required=True)
    phase4_results.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "phase1":
        result = seal_phase1(
            expected_commit=args.expected_commit,
            split_manifest=args.split_manifest,
            split_manifest_sha256=args.split_manifest_sha256,
            phase0_summary=args.phase0_summary,
            code_gate_receipt=args.code_gate_receipt,
            controls=args.control,
            output=args.output,
        )
    elif args.command == "phase2":
        result = seal_phase2(
            phase1_receipt=args.phase1_receipt,
            summaries=args.summary,
            budget_protocols=args.budget_protocol,
            output=args.output,
        )
    elif args.command == "phase3":
        result = seal_phase3(
            phase2_receipt=args.phase2_receipt,
            results_jsonl=args.results_jsonl,
            output=args.output,
            expected_seed=args.expected_seed,
            bootstrap_samples=args.bootstrap_samples,
            cost_tolerance=args.cost_tolerance,
        )
    elif args.command == "authorize-phase4":
        result = authorize_phase4(
            phase3_receipt=args.phase3_receipt,
            output=args.output,
            formal_seeds=args.formal_seeds,
        )
    else:
        result = seal_phase4(
            authorization_receipt=args.authorization_receipt,
            results_jsonl=args.results_jsonl,
            output=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
