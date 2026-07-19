from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.diagnose_duca_allocation_family_ceiling import read_input_records
from tools.bata.export_duca_allocation_ceiling_inputs import sha256, write_json_exclusive
from tools.bata.finalize_duca_allocation_ceiling_gate import (
    _revalidate_dataset_files,
    _validate_suite_manifest,
)
from tools.bata.subset_duca_allocation_inputs import _hash_video_round_robin
from tools.bata.validate_duca_allocation_ceiling_artifact import (
    validate_artifact_receipt,
)
from tools.bata.validate_duca_allocation_candidate_loss_artifact import (
    validate_candidate_artifact,
)
from tools.bata.validate_duca_allocation_solver_cost_artifact import (
    validate_solver_cost_artifact,
)


def finalize_suite(
    *,
    gate_json: str | Path,
    full_input_jsonl: str | Path,
    full_ceiling_jsonl: str | Path,
    full_ceiling_summary_json: str | Path,
    full_ceiling_validation_json: str | Path,
    gt_input_jsonl: str | Path,
    gt_ceiling_jsonl: str | Path,
    gt_ceiling_summary_json: str | Path,
    gt_ceiling_validation_json: str | Path,
    candidate_jsonl: str | Path,
    candidate_summary_json: str | Path,
    solver_cost_samples_jsonl: str | Path,
    solver_cost_summary_json: str | Path,
    suite_manifest_json: str | Path,
    suite_manifest_sha256: str,
    output_json: str | Path,
) -> dict[str, Any]:
    paths = {
        "gate": Path(gate_json).resolve(),
        "full_input": Path(full_input_jsonl).resolve(),
        "full_ceiling_jsonl": Path(full_ceiling_jsonl).resolve(),
        "full_ceiling": Path(full_ceiling_summary_json).resolve(),
        "full_ceiling_validation": Path(full_ceiling_validation_json).resolve(),
        "gt_input": Path(gt_input_jsonl).resolve(),
        "gt_ceiling_jsonl": Path(gt_ceiling_jsonl).resolve(),
        "gt_ceiling": Path(gt_ceiling_summary_json).resolve(),
        "gt_ceiling_validation": Path(gt_ceiling_validation_json).resolve(),
        "candidate_jsonl": Path(candidate_jsonl).resolve(),
        "candidate": Path(candidate_summary_json).resolve(),
        "solver_cost_samples": Path(solver_cost_samples_jsonl).resolve(),
        "solver_cost": Path(solver_cost_summary_json).resolve(),
        "suite_manifest": Path(suite_manifest_json).resolve(),
    }
    output_path = Path(output_json).resolve()
    if output_path.exists():
        raise FileExistsError("allocation training suite never overwrites evidence")
    payloads = {
        key: _load(paths[key])
        for key in ("gate", "full_ceiling", "gt_ceiling", "candidate", "solver_cost")
    }
    full_validation = validate_artifact_receipt(
        validation_json=paths["full_ceiling_validation"],
        input_jsonl=paths["full_input"],
        output_jsonl=paths["full_ceiling_jsonl"],
        summary_json=paths["full_ceiling"],
        require_gt_solver_replay=False,
    )
    gt_validation = validate_artifact_receipt(
        validation_json=paths["gt_ceiling_validation"],
        input_jsonl=paths["gt_input"],
        output_jsonl=paths["gt_ceiling_jsonl"],
        summary_json=paths["gt_ceiling"],
        require_gt_solver_replay=True,
    )
    candidate_validation = validate_candidate_artifact(
        ceiling_jsonl=paths["gt_ceiling_jsonl"],
        candidate_jsonl=paths["candidate_jsonl"],
        summary_json=paths["candidate"],
    )
    cost_validation = validate_solver_cost_artifact(
        input_jsonl=paths["full_input"],
        samples_jsonl=paths["solver_cost_samples"],
        summary_json=paths["solver_cost"],
    )
    if payloads["gate"].get("gate_passed") is not True:
        raise ValueError("real allocation gate did not pass")
    if payloads["gate"].get("execution_cluster") != "n16r4":
        raise ValueError("real allocation gate did not execute on n16r4")
    for key in ("full_ceiling", "gt_ceiling"):
        if payloads[key].get("schema_version") != "duca_allocation_family_ceiling_summary_v1":
            raise ValueError(f"{key} ceiling summary schema mismatch")
    if payloads["candidate"].get("schema_version") != "duca_allocation_candidate_detector_loss_summary_v1":
        raise ValueError("candidate detector summary schema mismatch")
    if payloads["solver_cost"].get("schema_version") != "duca_allocation_solver_cost_summary_v1":
        raise ValueError("solver cost summary schema mismatch")
    if payloads["full_ceiling"].get("gt_families") != "none":
        raise ValueError("full recoverability pass must not solve privileged GT families")
    if payloads["gt_ceiling"].get("gt_families") != "both":
        raise ValueError("bounded privileged ceiling must contain both D-GT and E-GT")

    expected_commit = str(payloads["gate"].get("git_commit"))
    expected_checkpoint_sha = str(payloads["gate"].get("checkpoint_sha256"))
    expected_checkpoint_epoch = int(payloads["gate"].get("checkpoint_epoch", -1))
    checkpoint_path = Path(str(payloads["gate"].get("checkpoint", ""))).resolve()
    pretrain_path = Path(str(payloads["gate"].get("pretrain", ""))).resolve()
    expected_pretrain_sha = str(payloads["gate"].get("pretrain_sha256"))
    manifest = _validate_suite_manifest(
        paths["suite_manifest"],
        expected_sha256=suite_manifest_sha256,
        expected_commit=expected_commit,
        expected_checkpoint=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha,
        expected_checkpoint_epoch=expected_checkpoint_epoch,
        expected_pretrain=pretrain_path,
        expected_pretrain_sha256=expected_pretrain_sha,
    )
    if sha256(checkpoint_path) != expected_checkpoint_sha:
        raise ValueError("terminal checkpoint bytes changed after the real gate")
    if sha256(pretrain_path) != expected_pretrain_sha:
        raise ValueError("backbone pretrain bytes changed after the real gate")
    input_records: dict[str, list[dict[str, Any]]] = {}
    for input_key in ("full_input", "gt_input"):
        records = read_input_records(paths[input_key])
        input_records[input_key] = records
        if any(row.get("split") != "train" for row in records):
            raise ValueError(f"{input_key} consumed a non-training split")
        for row in records:
            source = row["source"]
            if source.get("git_commit") != expected_commit:
                raise ValueError(f"{input_key} source commit mismatch")
            if source.get("checkpoint_sha256") != expected_checkpoint_sha:
                raise ValueError(f"{input_key} checkpoint hash mismatch")
            if source.get("checkpoint_state_key") != "state_dict_ema":
                raise ValueError(f"{input_key} checkpoint state mismatch")
            if int(source.get("checkpoint_epoch", -1)) != expected_checkpoint_epoch:
                raise ValueError(f"{input_key} checkpoint epoch mismatch")
    candidate_source = payloads["candidate"].get("source")
    if not isinstance(candidate_source, Mapping):
        raise ValueError("candidate source is missing")
    if candidate_source.get("git_commit") != expected_commit:
        raise ValueError("candidate source commit mismatch")
    if candidate_source.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise ValueError("candidate checkpoint hash mismatch")
    if candidate_source.get("checkpoint_state_key") != "state_dict_ema":
        raise ValueError("candidate checkpoint state mismatch")
    if int(candidate_source.get("checkpoint_epoch", -1)) != expected_checkpoint_epoch:
        raise ValueError("candidate checkpoint epoch mismatch")
    if candidate_source.get("backbone_pretrain_sha256") != expected_pretrain_sha:
        raise ValueError("candidate pretrain hash mismatch")
    if (
        Path(str(candidate_source.get("ceiling_validation_json", ""))).resolve()
        != paths["gt_ceiling_validation"]
        or candidate_source.get("ceiling_validation_json_sha256")
        != sha256(paths["gt_ceiling_validation"])
        or candidate_source.get("ceiling_validation") != gt_validation
    ):
        raise ValueError("candidate ceiling-validation receipt binding mismatch")
    full_source = input_records["full_input"][0]["source"]
    _revalidate_dataset_files(full_source)
    for key, manifest_key in (
        ("annotation_path", "annotation"),
        ("annotation_sha256", "annotation_sha256"),
        ("class_map_path", "class_map"),
        ("class_map_sha256", "class_map_sha256"),
        ("data_path", "train_data_path"),
        ("config", "training_config"),
        ("config_sha256", "training_config_sha256"),
    ):
        if full_source.get(key) != manifest.get(manifest_key):
            raise ValueError(f"training source differs from suite manifest: {key}")
    provenance_keys = (
        "annotation_sha256",
        "class_map_sha256",
        "data_path",
        "config",
        "config_sha256",
        "dataset_data_manifest_sha256",
        "dataset_data_file_count",
        "dataset_data_total_bytes",
        "dataset_data_hash_algorithm",
        "dataset_subset_name",
        "dataset_test_mode",
        "dataset_filter_gt",
        "dataset_ioa_thresh",
        "dataset_feature_stride",
        "dataset_sample_stride",
        "dataset_window_size",
        "dataset_window_overlap_ratio",
        "dataset_offset_frames",
        "dataset_config_sha256",
        "dataset_window_manifest_sha256",
        "dataset_window_count",
        "dataset_window_deduplication",
        "dataset_duplicate_window_count_removed",
    )
    for row in input_records["full_input"] + input_records["gt_input"]:
        source = row["source"]
        for key in provenance_keys:
            if source.get(key) != full_source.get(key):
                raise ValueError(f"input dataset provenance mismatch: {key}")
    for key in provenance_keys:
        if candidate_source.get(key) != full_source.get(key):
            raise ValueError(f"candidate dataset provenance mismatch: {key}")
    expected_gt_records = _hash_video_round_robin(
        input_records["full_input"],
        first_n=32,
        seed="duca-allocation-ceiling-v1",
    )
    expected_gt_binding = [
        (row["sample_id"], row["record_sha256"])
        for row in expected_gt_records
    ]
    actual_gt_binding = [
        (row["sample_id"], row["record_sha256"])
        for row in input_records["gt_input"]
    ]
    if actual_gt_binding != expected_gt_binding:
        raise ValueError(
            "privileged GT subset does not reproduce the preregistered "
            "hash-video-round-robin selection"
        )

    gate_input_path = Path(str(payloads["gate"].get("input_jsonl", ""))).resolve()
    if (
        not gate_input_path.is_file()
        or sha256(gate_input_path) != payloads["gate"].get("input_jsonl_sha256")
    ):
        raise ValueError("real gate input artifact binding is invalid")
    gate_input_records = read_input_records(gate_input_path)
    if len(gate_input_records) != 1:
        raise ValueError("real gate must contain exactly one input sample")
    gate_source = gate_input_records[0]["source"]
    for key in provenance_keys:
        if gate_source.get(key) != full_source.get(key):
            raise ValueError(f"real gate dataset provenance mismatch: {key}")

    full_families = payloads["full_ceiling"]["families"]
    gt_families = payloads["gt_ceiling"]["families"]
    candidate_families = payloads["candidate"]["families"]
    required_full = {
        "A_exact_uniform",
        "B_one_per_uniform_cell",
        "C_uniform_scaffold_residual",
        "D_deploy_score",
    }
    required_gt = required_full | {
        "D_privileged_gt_ceiling",
        "E_privileged_unrestricted_gt",
    }
    if not required_full.issubset(full_families):
        raise ValueError("full ceiling summary is incomplete")
    if not required_gt.issubset(gt_families):
        raise ValueError("GT ceiling summary is incomplete")
    required_candidate = {
        "A_exact_uniform",
        "D_deploy_score",
        "D_privileged_gt_ceiling",
        "E_privileged_unrestricted_gt",
    }
    if not required_candidate.issubset(candidate_families):
        raise ValueError("candidate detector summary is incomplete")
    if int(payloads["gt_ceiling"].get("sample_count", -1)) != 32:
        raise ValueError("privileged ceiling must use the preregistered 32-sample subset")
    if int(payloads["candidate"].get("sample_count", -1)) != 32:
        raise ValueError("candidate detector loss must cover the same 32 samples")
    if int(payloads["solver_cost"].get("sample_count", -1)) != 100:
        raise ValueError("solver cost must contain the preregistered 100 timed replays")

    radii = (0, 1, 2, 4)
    geometry_deltas: dict[str, Any] = {}
    for family_key in (
        "D_deploy_score",
        "D_privileged_gt_ceiling",
        "E_privileged_unrestricted_gt",
    ):
        source = full_families if family_key == "D_deploy_score" else gt_families
        reference = (
            full_families["A_exact_uniform"]
            if family_key == "D_deploy_score"
            else gt_families["A_exact_uniform"]
        )
        geometry_deltas[family_key] = {
            f"both_boundary_recall_r{radius}_gain": _metric(
                source[family_key],
                f"both_boundary_recall_r{radius}",
            )
            - _metric(reference, f"both_boundary_recall_r{radius}")
            for radius in radii
        }
        geometry_deltas[family_key]["mean_endpoint_distance_reduction"] = (
            _metric(reference, "mean_endpoint_distance")
            - _metric(source[family_key], "mean_endpoint_distance")
        )

    detector_deltas = {
        family_key: {
            "detector_loss_gain_vs_uniform": float(
                candidate_families[family_key]["gain_vs_uniform"]
            ),
            "mean_detector_loss": float(
                candidate_families[family_key]["mean_detector_loss"]
            ),
        }
        for family_key in required_candidate
    }
    result = {
        "schema_version": "duca_allocation_training_suite_evidence_v1",
        "status": "training_side_ceiling_complete_human_go_kill_required",
        "git_commit": expected_commit,
        "checkpoint_sha256": expected_checkpoint_sha,
        "pretrain_sha256": expected_pretrain_sha,
        "sample_counts": {
            "full_recoverability": int(payloads["full_ceiling"]["sample_count"]),
            "privileged_gt_subset": int(payloads["gt_ceiling"]["sample_count"]),
            "candidate_detector_subset": int(payloads["candidate"]["sample_count"]),
            "solver_cost": int(payloads["solver_cost"]["sample_count"]),
        },
        "geometry_deltas_vs_uniform": geometry_deltas,
        "frozen_detector_deltas_vs_uniform": detector_deltas,
        "coarse_signal_metrics": payloads["full_ceiling"].get(
            "mean_coarse_signal_metrics",
            {},
        ),
        "solver_latency_ms": payloads["solver_cost"]["latency_ms"],
        "suite_manifest": manifest,
        "full_ceiling_validation": full_validation,
        "gt_ceiling_validation": gt_validation,
        "candidate_validation": candidate_validation,
        "solver_cost_validation": cost_validation,
        "artifacts": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "decision_contract": {
            "validation_subset_consumed": False,
            "selector_training_authorized": False,
            "paper_claim_allowed": False,
            "full_stack_cost_measured_in_this_stage": False,
            "next_decision": "GEOMETRY_GO_HOLD_OR_KILL_GLOBAL_ALLOCATION_ROUTE",
            "geometry_go_requires_privileged_headroom": True,
            "geometry_go_requires_deploy_score_recoverability": True,
            "geometry_go_only_authorizes_validation_and_full_stack_cost": True,
            "paper_go_still_requires_positive_full_stack_savings": True,
        },
    }
    write_json_exclusive(output_path, result)
    return result


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected an object")
    return payload


def _metric(family: Mapping[str, Any], key: str) -> float:
    value = family.get("mean_metrics", {}).get(key)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"family metric is unavailable or non-finite: {key}")
    return float(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize bounded training-side DUCA allocation evidence."
    )
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--full-input-jsonl", required=True)
    parser.add_argument("--full-ceiling-jsonl", required=True)
    parser.add_argument("--full-ceiling-summary-json", required=True)
    parser.add_argument("--full-ceiling-validation-json", required=True)
    parser.add_argument("--gt-input-jsonl", required=True)
    parser.add_argument("--gt-ceiling-jsonl", required=True)
    parser.add_argument("--gt-ceiling-summary-json", required=True)
    parser.add_argument("--gt-ceiling-validation-json", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--candidate-summary-json", required=True)
    parser.add_argument("--solver-cost-samples-jsonl", required=True)
    parser.add_argument("--solver-cost-summary-json", required=True)
    parser.add_argument("--suite-manifest-json", required=True)
    parser.add_argument("--suite-manifest-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    result = finalize_suite(
        gate_json=args.gate_json,
        full_input_jsonl=args.full_input_jsonl,
        full_ceiling_jsonl=args.full_ceiling_jsonl,
        full_ceiling_summary_json=args.full_ceiling_summary_json,
        full_ceiling_validation_json=args.full_ceiling_validation_json,
        gt_input_jsonl=args.gt_input_jsonl,
        gt_ceiling_jsonl=args.gt_ceiling_jsonl,
        gt_ceiling_summary_json=args.gt_ceiling_summary_json,
        gt_ceiling_validation_json=args.gt_ceiling_validation_json,
        candidate_jsonl=args.candidate_jsonl,
        candidate_summary_json=args.candidate_summary_json,
        solver_cost_samples_jsonl=args.solver_cost_samples_jsonl,
        solver_cost_summary_json=args.solver_cost_summary_json,
        suite_manifest_json=args.suite_manifest_json,
        suite_manifest_sha256=args.suite_manifest_sha256,
        output_json=args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
