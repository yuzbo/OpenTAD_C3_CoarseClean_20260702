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
PHASE3_METRICS = ("avg_map", "map_0.7", "short_map", "pair_support")


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


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
    budget_protocol: str | Path,
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
    protocol_path, protocol = _load_json(budget_protocol)
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
        raise ValueError("frozen RIME protocol is not hash-bound to the supplied O1-O4 evidence")

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
        "budget_protocol": _artifact(protocol_path),
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
        if (
            row.get("schema_version") != PHASE3_RESULT_SCHEMA
            or arm in by_arm
            or arm not in PHASE3_ARMS
            or int(row.get("seed", -1)) != int(expected_seed)
            or int(row.get("successful_detector_updates", -1)) != 6000
            or row.get("formal_update_audit_passed") is not True
            or row.get("uses_official_final") is not False
            or row.get("split_assignment_sha256")
            != phase2["split_assignment_sha256"]
            or row.get("padded_to_kmax") is not False
        ):
            raise ValueError("invalid, incomplete, or contaminated Phase-3 arm result")
        result_videos = set(str(value) for value in row.get("evaluation_video_ids", ()))
        if result_videos != development_videos:
            raise ValueError("Phase-3 must evaluate exactly the certification/development role")
        for metric in PHASE3_METRICS:
            _finite_metric_map(row, metric, development_videos)
        costs = {
            str(key): float(value)
            for key, value in row.get("realized_total_cost", {}).items()
        }
        if (
            set(costs) != development_videos
            or not all(math.isfinite(value) and value > 0.0 for value in costs.values())
        ):
            raise ValueError("Phase-3 realized full-stack cost ledger is incomplete")
        initialization.add(str(row.get("initialization_sha256")))
        exposure.add(str(row.get("training_exposure_sha256")))
        by_arm[arm] = row
    if set(by_arm) != set(PHASE3_ARMS):
        raise ValueError("Phase-3 seven-arm matrix is incomplete")
    if len(initialization) != 1 or "" in initialization or len(exposure) != 1 or "" in exposure:
        raise ValueError("Phase-3 arms do not share initialization and exposure")
    for arm in ("U-same-K", "D-shuffle"):
        if by_arm[arm].get("k_histogram") != by_arm["RIME-full"].get("k_histogram"):
            raise ValueError(f"{arm} does not preserve the RIME-full K histogram")

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
    full_cost = mean(
        float(value) for value in by_arm["RIME-full"]["realized_total_cost"].values()
    )
    fixed_cost = mean(
        float(value) for value in by_arm["U-fixed"]["realized_total_cost"].values()
    )
    dense_cost = float(by_arm["RIME-full"].get("dense_reference_mean_cost", math.nan))
    cost_gate = (
        math.isfinite(dense_cost)
        and full_cost < dense_cost
        and abs(full_cost - fixed_cost) <= float(cost_tolerance)
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
        "successful_detector_updates_per_arm": 6000,
        "shared_initialization_sha256": next(iter(initialization)),
        "shared_training_exposure_sha256": next(iter(exposure)),
        "phase0_minimum_primary_gain": minimum_gain,
        "comparisons": comparisons,
        "contribution_gates": contribution_gates,
        "realized_mean_cost": {
            "RIME-full": full_cost,
            "U-fixed": fixed_cost,
            "dense_reference": dense_cost,
            "match_tolerance": float(cost_tolerance),
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
    if (
        phase3.get("schema_version") != RECEIPT_SCHEMA
        or phase3.get("phase") != "phase3"
        or phase3.get("gate_pass") is not True
        or phase3.get("phase4_authorized") is not True
        or phase3.get("official_final_subset_consumed") is not False
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or int(phase3["development_seed"]) in seeds
    ):
        raise RuntimeError("Phase-4 submission is blocked by Phase-3 or invalid formal seeds")
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "phase": "phase4_authorization",
        "status": "authorized",
        "gate_pass": True,
        "git_commit": phase3["git_commit"],
        "phase3_receipt": _artifact(phase3_path),
        "formal_seeds": list(seeds),
        "required_detectors": ["ActionFormer", "TriDet"],
        "required_budget_panels": [384, 192],
        "official_final_subset_consumed": False,
        "paper_claim_allowed": False,
        "claim_scope": "formal_submission_authorization_only",
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
    phase2.add_argument("--budget-protocol", required=True)
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
            budget_protocol=args.budget_protocol,
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
    else:
        result = authorize_phase4(
            phase3_receipt=args.phase3_receipt,
            output=args.output,
            formal_seeds=args.formal_seeds,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
