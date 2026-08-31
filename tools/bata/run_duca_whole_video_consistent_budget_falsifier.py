from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bata.run_duca_marginal_frozen_h65_probe import (  # noqa: E402
    BASELINE_BUDGET,
    BUDGETS,
    CAP_RELEASE_BASE_REVISION,
    CAP_RELEASE_TERMINAL_REVISION,
    METRIC_KEYS,
    SEALED_PRODUCER_REVISION,
    _actual_observation_cost,
    _canonical_sha256,
    _git_identity,
    _load_config_and_paths,
    _metric_delta_pp,
    _metric_reproduction_error_pp,
    _official_holdout_metrics,
    _raw_results_for_budgets,
    _read_jsonl_gz,
    _sha256,
    _stage_paths,
    _stage_source,
    _write_json,
)


EVIDENCE_REVISION = "46812facc8773d9b4a9c21833cbe397c8aaa5a2d"
EXPECTED_PROBE_SHA256 = (
    "8d6df7240c8b81b4d6d9aa8ff98bae530d6823ddd1d411bed47ce983ebd94925"
)
EXPECTED_CAP_RELEASE_SHA256 = (
    "fb3c122e233952a4165c2ca9a6ff3d2839b8e0d108c977443786714ec0cf6ed4"
)
EXPECTED_NEIGHBORHOOD_SHA256 = (
    "a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b"
)
EXPECTED_STAGE_SHA256 = {
    "selection": "1d668d4e5eb4b5ef3c1057c97ec63cc2c1eed3c0e62297290520063b4e1ec38f",
    "k256": "6dc8893a41b5c8132b176f32133ffc2f48a5491146385c147b8227167608a309",
    "k512": "c7fa06258c07163d0906b512a78e367c27607c64fc41b28bce9fe51fbd0815d7",
}
EXPECTED_HOLDOUT_VIDEOS = 40
EXPECTED_HOLDOUT_WINDOWS = 124
EXPECTED_ORDERED_PAIRS = EXPECTED_HOLDOUT_VIDEOS * (EXPECTED_HOLDOUT_VIDEOS - 1)
BASELINE_ACTUAL_COST = 47_110
AVERAGE_MAP_GATE_PP = 0.8
MAP_07_GATE_PP = 1.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "candidate_manifest": output_dir / "whole_video_candidate_manifest.json",
        "pre_run": output_dir / "whole_video_pre_run_receipt.json",
        "result": output_dir / "whole_video_consistent_budget_result.json",
    }


def _verify_current_git(expected_head: str) -> dict[str, Any]:
    identity = _git_identity(REPO_ROOT)
    if identity["dirty"]:
        raise RuntimeError("whole-video falsifier requires a clean Git worktree")
    if identity["head"] != str(expected_head):
        raise RuntimeError(
            f"Git HEAD mismatch: expected {expected_head}, got {identity['head']}"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EVIDENCE_REVISION, identity["head"]],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("whole-video implementation is not descended from the sealed evidence revision")
    return identity


def _prepare_unlabeled_bundle(input_dir: Path) -> dict[str, Any]:
    input_paths = _stage_paths(input_dir)
    manifest_path = input_paths["split_dir"] / "frontend_split_manifest.json"
    split_manifest = _read_json(manifest_path)
    holdout_videos = sorted(str(value) for value in split_manifest.get("holdout_videos", []))
    if len(holdout_videos) != EXPECTED_HOLDOUT_VIDEOS or len(holdout_videos) != len(
        set(holdout_videos)
    ):
        raise RuntimeError("frozen split must contain exactly 40 unique holdout videos")

    selection_rows = _read_jsonl_gz(input_paths["selection"])
    k256_rows = {
        str(row["sample_id"]): row for row in _read_jsonl_gz(input_paths["k256"])
    }
    k512_rows = {
        str(row["sample_id"]): row for row in _read_jsonl_gz(input_paths["k512"])
    }
    selection_ids = {str(row["sample_id"]) for row in selection_rows}
    if set(k256_rows) != selection_ids or set(k512_rows) != selection_ids:
        raise RuntimeError("sealed prediction artifacts have different sample sets")
    merged_rows = []
    for selected in selection_rows:
        sample_id = str(selected["sample_id"])
        row = dict(selected)
        row["predictions"] = {
            "256": k256_rows[sample_id]["prediction"],
            "384": selected["prediction_k384"],
            "512": k512_rows[sample_id]["prediction"],
        }
        merged_rows.append(row)
    holdout_set = set(holdout_videos)
    rows = sorted(
        (row for row in merged_rows if str(row["video_id"]) in holdout_set),
        key=lambda row: (str(row["video_id"]), str(row["sample_id"])),
    )
    sample_ids = [str(row["sample_id"]) for row in rows]
    observed_videos = {str(row["video_id"]) for row in rows}
    if len(rows) != EXPECTED_HOLDOUT_WINDOWS:
        raise RuntimeError(
            f"frozen holdout must contain 124 windows, observed {len(rows)}"
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("frozen holdout contains duplicate sample IDs")
    if observed_videos != holdout_set:
        raise RuntimeError("sealed prediction rows do not cover the frozen holdout video set")
    for row in rows:
        predictions = row.get("predictions")
        if not isinstance(predictions, Mapping) or any(
            str(budget) not in predictions for budget in BUDGETS
        ):
            raise RuntimeError(f"{row['sample_id']}: one or more sealed budget predictions are missing")
        for budget in BUDGETS:
            _actual_observation_cost(row, budget)

    return {
        "input_dir": input_dir,
        "input_paths": input_paths,
        "manifest_path": manifest_path,
        "split_manifest": split_manifest,
        "holdout_videos": holdout_videos,
        "rows": rows,
    }


def _load_terminal_evidence(bundle: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(bundle)
    paths = bundle["input_paths"]
    enriched.update(
        {
            "probe_result": _read_json(paths["result"]),
            "cap_release_result": _read_json(paths["cap_release_result"]),
            "neighborhood_result": _read_json(paths["cap_release_neighborhood_result"]),
        }
    )
    return enriched


def _requested_budgets_for_pair(
    rows: Sequence[Mapping[str, Any]], donor_video_id: str, recipient_video_id: str
) -> list[int]:
    if donor_video_id == recipient_video_id:
        raise ValueError("donor and recipient videos must differ")
    return [
        256
        if str(row["video_id"]) == donor_video_id
        else 512
        if str(row["video_id"]) == recipient_video_id
        else BASELINE_BUDGET
        for row in rows
    ]


def _actual_cost(rows: Sequence[Mapping[str, Any]], budgets: Sequence[int]) -> int:
    if len(rows) != len(budgets):
        raise ValueError("row and budget vectors must have identical lengths")
    return sum(
        _actual_observation_cost(row, int(budget))
        for row, budget in zip(rows, budgets)
    )


def _candidate_record(
    rows: Sequence[Mapping[str, Any]], donor_video_id: str, recipient_video_id: str
) -> dict[str, Any]:
    budgets = _requested_budgets_for_pair(rows, donor_video_id, recipient_video_id)
    donor_indices = [
        index for index, row in enumerate(rows) if str(row["video_id"]) == donor_video_id
    ]
    recipient_indices = [
        index
        for index, row in enumerate(rows)
        if str(row["video_id"]) == recipient_video_id
    ]
    donor_changed = sum(
        _actual_observation_cost(rows[index], 256)
        != _actual_observation_cost(rows[index], BASELINE_BUDGET)
        for index in donor_indices
    )
    recipient_changed = sum(
        _actual_observation_cost(rows[index], 512)
        != _actual_observation_cost(rows[index], BASELINE_BUDGET)
        for index in recipient_indices
    )
    actual_cost = _actual_cost(rows, budgets)
    intervention_valid = donor_changed > 0 and recipient_changed > 0
    within_cost = actual_cost <= BASELINE_ACTUAL_COST
    return {
        "candidate_id": f"{donor_video_id}=>{recipient_video_id}",
        "donor_video_id": donor_video_id,
        "recipient_video_id": recipient_video_id,
        "donor_requested_budget": 256,
        "recipient_requested_budget": 512,
        "all_other_requested_budget": BASELINE_BUDGET,
        "donor_window_count": len(donor_indices),
        "recipient_window_count": len(recipient_indices),
        "donor_actual_nonbaseline_window_count": donor_changed,
        "recipient_actual_nonbaseline_window_count": recipient_changed,
        "actual_observation_cost": actual_cost,
        "actual_observation_cost_delta_vs_fixed": actual_cost - BASELINE_ACTUAL_COST,
        "actual_intervention_valid": intervention_valid,
        "within_fixed_cost": within_cost,
        "legal": intervention_valid and within_cost,
    }


def _build_candidate_manifest(
    rows: Sequence[Mapping[str, Any]], holdout_videos: Sequence[str]
) -> dict[str, Any]:
    baseline_budgets = [BASELINE_BUDGET] * len(rows)
    baseline_cost = _actual_cost(rows, baseline_budgets)
    if baseline_cost != BASELINE_ACTUAL_COST:
        raise RuntimeError(
            f"fixed K384 actual cost mismatch: expected {BASELINE_ACTUAL_COST}, got {baseline_cost}"
        )
    candidates = [
        _candidate_record(rows, donor, recipient)
        for donor in sorted(holdout_videos)
        for recipient in sorted(holdout_videos)
        if donor != recipient
    ]
    candidate_ids = [str(value["candidate_id"]) for value in candidates]
    if len(candidates) != EXPECTED_ORDERED_PAIRS:
        raise RuntimeError("ordered donor-recipient enumeration is incomplete")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("ordered donor-recipient enumeration contains duplicates")
    legal = [value for value in candidates if bool(value["legal"])]
    if not legal:
        raise RuntimeError("no legal whole-video donor-recipient candidate exists")
    return {
        "schema": "duca_whole_video_consistent_budget_candidate_manifest_v1",
        "method": "DUCA whole-video consistent-budget falsifier",
        "development_holdout_oracle": True,
        "official_validation_consumed": False,
        "official_test_consumed": False,
        "labels_or_ground_truth_read": False,
        "metric_evaluated": False,
        "candidate_generation_fields": [
            "sample_id",
            "video_id",
            "budget_accounting.actual_cost",
        ],
        "holdout_video_count": len(holdout_videos),
        "holdout_window_count": len(rows),
        "holdout_videos": list(sorted(holdout_videos)),
        "ordered_pair_count": len(candidates),
        "actual_intervention_candidate_count": sum(
            bool(value["actual_intervention_valid"]) for value in candidates
        ),
        "legal_candidate_count": len(legal),
        "fixed_actual_observation_cost": baseline_cost,
        "cost_limit": BASELINE_ACTUAL_COST,
        "candidates": candidates,
    }


def _identity_mismatches(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> list[str]:
    keys = (
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "checkpoint_state_key",
        "annotation_sha256",
        "class_map_sha256",
        "videomae_pretrain_sha256",
    )
    return [key for key in keys if left.get(key) != right.get(key)]


def _verify_input_identity(
    bundle: Mapping[str, Any], current_source: Mapping[str, Any]
) -> dict[str, Any]:
    paths = bundle["input_paths"]
    probe = bundle["probe_result"]
    cap_release = bundle["cap_release_result"]
    neighborhood = bundle["neighborhood_result"]

    expected_json = {
        "result": EXPECTED_PROBE_SHA256,
        "cap_release_result": EXPECTED_CAP_RELEASE_SHA256,
        "cap_release_neighborhood_result": EXPECTED_NEIGHBORHOOD_SHA256,
    }
    verified_json = {}
    for key, expected_sha in expected_json.items():
        actual_sha = _sha256(paths[key])
        if actual_sha != expected_sha:
            raise RuntimeError(f"{paths[key].name} SHA256 mismatch")
        verified_json[key] = {"path": str(paths[key]), "sha256": actual_sha}

    if probe.get("status") != "ORACLE_HEADROOM_GRAY_ZONE_RETURN_TO_PRO":
        raise RuntimeError("probe_result.json has the wrong terminal status")
    if cap_release.get("status") != "CAP_RELEASE_POINT_GATE_FAILED_STOP_CURRENT_MECHANISM":
        raise RuntimeError("oracle_cap_release_result.json has the wrong terminal status")
    if neighborhood.get("status") != "JOINT_NEIGHBORHOOD_GATE_FAILED_STOP_DIFFERENCE_REPAIR":
        raise RuntimeError("neighborhood result has the wrong terminal status")
    expected_source_heads = (
        (probe.get("source", {}), CAP_RELEASE_BASE_REVISION, "probe"),
        (cap_release.get("source", {}), CAP_RELEASE_TERMINAL_REVISION, "cap-release"),
        (neighborhood.get("source", {}), EVIDENCE_REVISION, "neighborhood"),
    )
    for source, expected_head, label in expected_source_heads:
        if source.get("git", {}).get("head") != expected_head:
            raise RuntimeError(f"{label} source has the wrong Git revision")
        mismatched = _identity_mismatches(current_source, source)
        if mismatched:
            raise RuntimeError(f"{label} source identity mismatch: {', '.join(mismatched)}")

    if cap_release.get("original_result", {}).get("sha256") != EXPECTED_PROBE_SHA256:
        raise RuntimeError("cap-release result does not bind the sealed probe result")
    terminal_cap = neighborhood.get("provenance", {}).get(
        "terminal_cap_release_result", {}
    )
    if terminal_cap.get("sha256") != EXPECTED_CAP_RELEASE_SHA256:
        raise RuntimeError("neighborhood result does not bind the terminal cap-release result")

    artifact_receipts = {
        "selection": "selection_receipt",
        "k256": "k256_receipt",
        "k512": "k512_receipt",
    }
    verified_artifacts = {}
    for artifact_key, receipt_key in artifact_receipts.items():
        artifact_sha = _sha256(paths[artifact_key])
        receipt = _read_json(paths[receipt_key])
        if artifact_sha != EXPECTED_STAGE_SHA256[artifact_key]:
            raise RuntimeError(f"sealed {artifact_key} artifact SHA256 mismatch")
        if receipt.get("artifact_sha256") != artifact_sha:
            raise RuntimeError(f"sealed {artifact_key} receipt does not bind its artifact")
        if receipt.get("source", {}).get("git", {}).get("head") != SEALED_PRODUCER_REVISION:
            raise RuntimeError(f"sealed {artifact_key} receipt has the wrong producer revision")
        if probe.get("stage_artifacts", {}).get(artifact_key, {}).get("sha256") != artifact_sha:
            raise RuntimeError(f"probe result does not bind sealed {artifact_key}")
        terminal_artifact = neighborhood.get("provenance", {}).get(
            "stage_artifacts", {}
        ).get(artifact_key, {})
        if terminal_artifact.get("sha256") != artifact_sha:
            raise RuntimeError(f"neighborhood result does not bind sealed {artifact_key}")
        verified_artifacts[artifact_key] = {
            "path": str(paths[artifact_key]),
            "sha256": artifact_sha,
            "producer_revision": SEALED_PRODUCER_REVISION,
        }

    sealed_pre_run = _read_json(paths["pre_run"])
    if (
        sealed_pre_run.get("status") != "PRE_RUN_PASS"
        or sealed_pre_run.get("source", {}).get("git", {}).get("head")
        != CAP_RELEASE_BASE_REVISION
    ):
        raise RuntimeError("sealed marginal PRE_RUN identity mismatch")
    if _identity_mismatches(current_source, sealed_pre_run.get("source", {})):
        raise RuntimeError("sealed marginal PRE_RUN source identity mismatch")
    return {
        "terminal_json": verified_json,
        "stage_artifacts": verified_artifacts,
        "sealed_marginal_pre_run": str(paths["pre_run"]),
    }


def _load_labeled_context(
    args: argparse.Namespace, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    from tools.bata.create_duca_frontend_split import validate_split_manifest

    (
        repo_root,
        config_path,
        cfg,
        checkpoint,
        checkpoint_sha,
        annotation,
        class_map,
        train_data,
        pretrain,
    ) = _load_config_and_paths(args)
    split_validation = validate_split_manifest(
        bundle["manifest_path"], annotation_path=annotation
    )
    manifest = bundle["split_manifest"]
    if int(manifest.get("train_video_count", -1)) != 160 or int(
        manifest.get("holdout_video_count", -1)
    ) != EXPECTED_HOLDOUT_VIDEOS:
        raise RuntimeError("frozen split count identity mismatch")
    if sorted(str(value) for value in manifest["holdout_videos"]) != list(
        bundle["holdout_videos"]
    ):
        raise RuntimeError("validated split holdout assignment changed")
    current_source = _stage_source(
        repo_root=repo_root,
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        annotation=annotation,
        class_map=class_map,
        train_data=train_data,
        pretrain=pretrain,
    )
    provenance = _verify_input_identity(bundle, current_source)
    provenance["split"] = split_validation
    return {
        "cfg": cfg,
        "annotation": annotation,
        "current_source": current_source,
        "provenance": provenance,
        "holdout_block_list": Path(split_validation["holdout_block_list"]),
    }


def _copy_holdout_block_list(source: Path, output_dir: Path) -> Path:
    target = output_dir / "frontend_holdout_block_list.txt"
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise FileExistsError(f"refusing to overwrite a different block list: {target}")
    else:
        shutil.copyfile(source, target)
    return target


def _budgets_from_sealed_allocation(
    rows: Sequence[Mapping[str, Any]], allocation: Mapping[str, Mapping[str, Any]]
) -> list[int]:
    budgets_by_sample: dict[str, int] = {}
    for video_id in sorted({str(row["video_id"]) for row in rows}):
        video_rows = sorted(
            (row for row in rows if str(row["video_id"]) == video_id),
            key=lambda row: str(row["sample_id"]),
        )
        if video_id not in allocation:
            raise RuntimeError(f"sealed allocation is missing {video_id}")
        values = [int(value) for value in allocation[video_id].get("budgets", [])]
        if len(values) != len(video_rows):
            raise RuntimeError(f"sealed allocation window count mismatch for {video_id}")
        for row, budget in zip(video_rows, values):
            if budget not in BUDGETS:
                raise RuntimeError(f"sealed allocation contains unsupported K{budget}")
            budgets_by_sample[str(row["sample_id"])] = budget
    return [budgets_by_sample[str(row["sample_id"])] for row in rows]


def _reproduce_anchors(
    *,
    bundle: Mapping[str, Any],
    context: Mapping[str, Any],
    holdout_block_list: Path,
    evaluator_threads: int,
) -> dict[str, Any]:
    rows = bundle["rows"]
    cap_release = bundle["cap_release_result"]
    expected = bundle["neighborhood_result"]["sealed_reference_metrics"]
    budget_vectors = {
        "fixed_h65_384": [BASELINE_BUDGET] * len(rows),
        "capped_oracle_384": _budgets_from_sealed_allocation(
            rows, cap_release["capped_allocation"]
        ),
        "cap_release_oracle_384": _budgets_from_sealed_allocation(
            rows, cap_release["cap_release_allocation"]
        ),
    }
    observed = {}
    errors = {}
    for label, budgets in budget_vectors.items():
        metrics, _predictions = _official_holdout_metrics(
            context["cfg"],
            raw_results=_raw_results_for_budgets(rows, budgets),
            annotation=context["annotation"],
            holdout_block_list=str(holdout_block_list),
            evaluator_threads=evaluator_threads,
        )
        observed[label] = metrics
        errors[label] = _metric_reproduction_error_pp(metrics, expected[label])
    maximum_error = max(
        value for metric_errors in errors.values() for value in metric_errors.values()
    )
    if maximum_error > 1.0e-6:
        raise RuntimeError("fixed/capped/released anchors did not reproduce within 1e-6 pp")
    return {
        "metrics": observed,
        "reproduction_error_pp": errors,
        "maximum_reproduction_error_pp": maximum_error,
    }


def run_pre_run_stage(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _output_paths(output_dir)
    git_identity = _verify_current_git(args.expected_head)
    if output_paths["pre_run"].is_file():
        existing = _read_json(output_paths["pre_run"])
        if existing.get("git", {}).get("head") != git_identity["head"]:
            raise RuntimeError("existing PRE_RUN receipt belongs to a different Git revision")
        return existing

    bundle = _prepare_unlabeled_bundle(Path(args.input_dir).expanduser().resolve())
    candidate_manifest = _build_candidate_manifest(
        bundle["rows"], bundle["holdout_videos"]
    )
    _write_json(output_paths["candidate_manifest"], candidate_manifest)

    # Scientific ordering contract: only after the complete candidate set exists
    # may annotation content and the mAP evaluator be accessed.
    bundle = _load_terminal_evidence(bundle)
    context = _load_labeled_context(args, bundle)
    block_list = _copy_holdout_block_list(context["holdout_block_list"], output_dir)
    anchors = _reproduce_anchors(
        bundle=bundle,
        context=context,
        holdout_block_list=block_list,
        evaluator_threads=args.evaluator_threads,
    )
    receipt = {
        "schema": "duca_whole_video_consistent_budget_pre_run_v1",
        "status": "PRE_RUN_PASS",
        "git": git_identity,
        "source": context["current_source"],
        "holdout_video_count": len(bundle["holdout_videos"]),
        "holdout_window_count": len(bundle["rows"]),
        "fixed_actual_observation_cost": candidate_manifest[
            "fixed_actual_observation_cost"
        ],
        "ordered_pair_count": candidate_manifest["ordered_pair_count"],
        "legal_candidate_count": candidate_manifest["legal_candidate_count"],
        "candidate_manifest": {
            "path": str(output_paths["candidate_manifest"]),
            "sha256": _sha256(output_paths["candidate_manifest"]),
            "canonical_sha256": _canonical_sha256(candidate_manifest),
            "generated_before_label_or_metric_access": True,
        },
        "anchor_reproduction": anchors,
        "provenance": context["provenance"],
        "training_performed": False,
        "detector_forward_executed": False,
        "scout_forward_executed": False,
        "gradient_computation_performed": False,
        "bootstrap_performed": False,
        "official_validation_consumed": False,
        "official_test_consumed": False,
    }
    _write_json(output_paths["pre_run"], receipt)
    return receipt


def _gate_margin(delta_pp: Mapping[str, float]) -> float:
    return min(
        float(delta_pp["average_mAP"]) - AVERAGE_MAP_GATE_PP,
        float(delta_pp["mAP@0.7"]) - MAP_07_GATE_PP,
    )


def _candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(record["joint_gate_margin_pp"]),
        int(record["actual_observation_cost"]),
        str(record["donor_video_id"]),
        str(record["recipient_video_id"]),
    )


def _best_by_metric(
    records: Sequence[Mapping[str, Any]], metric: str
) -> Mapping[str, Any]:
    return sorted(
        records,
        key=lambda record: (
            -float(record["metrics"][metric]),
            int(record["actual_observation_cost"]),
            str(record["donor_video_id"]),
            str(record["recipient_video_id"]),
        ),
    )[0]


def _verify_pre_run(
    *, output_paths: Mapping[str, Path], expected_head: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_json(output_paths["pre_run"])
    manifest = _read_json(output_paths["candidate_manifest"])
    if receipt.get("status") != "PRE_RUN_PASS":
        raise RuntimeError("formal evaluation requires PRE_RUN_PASS")
    if receipt.get("git", {}).get("head") != expected_head:
        raise RuntimeError("PRE_RUN was produced by a different Git revision")
    manifest_record = receipt.get("candidate_manifest", {})
    if _sha256(output_paths["candidate_manifest"]) != manifest_record.get("sha256"):
        raise RuntimeError("candidate manifest SHA256 changed after PRE_RUN")
    if _canonical_sha256(manifest) != manifest_record.get("canonical_sha256"):
        raise RuntimeError("candidate manifest content changed after PRE_RUN")
    return receipt, manifest


def run_evaluate_stage(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _output_paths(output_dir)
    git_identity = _verify_current_git(args.expected_head)
    if output_paths["result"].is_file():
        existing = _read_json(output_paths["result"])
        if existing.get("git", {}).get("head") != git_identity["head"]:
            raise RuntimeError("existing result belongs to a different Git revision")
        return existing

    bundle = _prepare_unlabeled_bundle(Path(args.input_dir).expanduser().resolve())
    regenerated = _build_candidate_manifest(
        bundle["rows"], bundle["holdout_videos"]
    )
    pre_run, candidate_manifest = _verify_pre_run(
        output_paths=output_paths, expected_head=args.expected_head
    )
    if _canonical_sha256(regenerated) != _canonical_sha256(candidate_manifest):
        raise RuntimeError("regenerated whole-video candidate set differs from PRE_RUN")

    bundle = _load_terminal_evidence(bundle)
    context = _load_labeled_context(args, bundle)
    block_list = _copy_holdout_block_list(context["holdout_block_list"], output_dir)
    fixed_metrics, _fixed_predictions = _official_holdout_metrics(
        context["cfg"],
        raw_results=_raw_results_for_budgets(
            bundle["rows"], [BASELINE_BUDGET] * len(bundle["rows"])
        ),
        annotation=context["annotation"],
        holdout_block_list=str(block_list),
        evaluator_threads=args.evaluator_threads,
    )
    fixed_error = _metric_reproduction_error_pp(
        fixed_metrics, pre_run["anchor_reproduction"]["metrics"]["fixed_h65_384"]
    )
    if max(fixed_error.values()) > 1.0e-6:
        raise RuntimeError("formal fixed K384 anchor differs from PRE_RUN")

    legal_candidates = [
        value for value in candidate_manifest["candidates"] if bool(value["legal"])
    ]
    records = []
    for index, candidate in enumerate(legal_candidates, start=1):
        budgets = _requested_budgets_for_pair(
            bundle["rows"],
            str(candidate["donor_video_id"]),
            str(candidate["recipient_video_id"]),
        )
        observed_cost = _actual_cost(bundle["rows"], budgets)
        if observed_cost != int(candidate["actual_observation_cost"]):
            raise RuntimeError(f"{candidate['candidate_id']}: actual cost changed")
        metrics, _predictions = _official_holdout_metrics(
            context["cfg"],
            raw_results=_raw_results_for_budgets(bundle["rows"], budgets),
            annotation=context["annotation"],
            holdout_block_list=str(block_list),
            evaluator_threads=args.evaluator_threads,
        )
        delta_pp = _metric_delta_pp(metrics, fixed_metrics)
        gate_margin = _gate_margin(delta_pp)
        passed = bool(
            delta_pp["average_mAP"] >= AVERAGE_MAP_GATE_PP
            and delta_pp["mAP@0.7"] >= MAP_07_GATE_PP
            and observed_cost <= BASELINE_ACTUAL_COST
        )
        records.append(
            {
                **candidate,
                "metrics": {key: float(metrics[key]) for key in METRIC_KEYS},
                "delta_vs_fixed_pp": delta_pp,
                "joint_gate_margin_pp": gate_margin,
                "gate_pass": passed,
            }
        )
        if index % 50 == 0 or index == len(legal_candidates):
            print(
                f"whole_video_candidates evaluated={index}/{len(legal_candidates)}",
                flush=True,
            )

    if not records:
        raise RuntimeError("formal evaluation has no legal candidates")
    passing = [record for record in records if bool(record["gate_pass"])]
    best_joint = sorted(records, key=_candidate_rank)[0]
    best_average = _best_by_metric(records, "average_mAP")
    best_high_tiou = _best_by_metric(records, "mAP@0.7")
    result = {
        "schema": "duca_whole_video_consistent_budget_falsifier_result_v1",
        "method": "DUCA whole-video consistent-budget falsifier",
        "status": (
            "WHOLE_VIDEO_HEADROOM_PASS_RETURN_TO_PRO"
            if passing
            else "WHOLE_VIDEO_NO_PASS_PROJECT_LEVEL_STOP"
        ),
        "development_holdout_oracle": True,
        "paper_claim_allowed": False,
        "deployable_policy_claim_allowed": False,
        "official_validation_consumed": False,
        "official_test_consumed": False,
        "training_performed": False,
        "detector_forward_executed": False,
        "scout_forward_executed": False,
        "gradient_computation_performed": False,
        "bootstrap_performed": False,
        "cuda_visible_devices_cleared": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "git": git_identity,
        "source": context["current_source"],
        "holdout_video_count": len(bundle["holdout_videos"]),
        "holdout_window_count": len(bundle["rows"]),
        "fixed_actual_observation_cost": BASELINE_ACTUAL_COST,
        "fixed_h65_384": {key: float(fixed_metrics[key]) for key in METRIC_KEYS},
        "fixed_reproduction_error_pp": fixed_error,
        "ordered_pair_count": candidate_manifest["ordered_pair_count"],
        "legal_candidate_count": len(records),
        "passing_candidate_count": len(passing),
        "passing_candidate_ids": [str(record["candidate_id"]) for record in passing],
        "frozen_gate": {
            "delta_average_mAP_pp_at_least": AVERAGE_MAP_GATE_PP,
            "delta_mAP_at_0.7_pp_at_least": MAP_07_GATE_PP,
            "actual_observation_cost_at_most": BASELINE_ACTUAL_COST,
        },
        "best_candidate_by_joint_gate": best_joint,
        "best_candidate_by_average_mAP": best_average,
        "best_candidate_by_mAP_at_0.7": best_high_tiou,
        "candidates": records,
        "evaluator_call_count": 1 + len(records),
        "pre_run": {
            "path": str(output_paths["pre_run"]),
            "sha256": _sha256(output_paths["pre_run"]),
        },
        "candidate_manifest": {
            "path": str(output_paths["candidate_manifest"]),
            "sha256": _sha256(output_paths["candidate_manifest"]),
        },
        "provenance": context["provenance"],
    }
    _write_json(output_paths["result"], result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively evaluate the frozen whole-video donor-recipient budget "
            "action space using sealed DUCA predictions only."
        )
    )
    parser.add_argument("--stage", required=True, choices=("pre-run", "evaluate"))
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT / "configs/adatad/thumos/duca_marginal_frozen_h65_probe.py"
        ),
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--annotation")
    parser.add_argument("--class-map")
    parser.add_argument("--train-data")
    parser.add_argument("--pretrain")
    parser.add_argument("--evaluator-threads", type=int, default=8)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = (
            run_pre_run_stage(args)
            if args.stage == "pre-run"
            else run_evaluate_stage(args)
        )
        result_path = _output_paths(output_dir)[
            "pre_run" if args.stage == "pre-run" else "result"
        ]
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "stage": args.stage,
                    "result_path": str(result_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        failure = {
            "status": "PRE_RUN_FAIL" if args.stage == "pre-run" else "EVALUATION_FAILED",
            "stage": args.stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(output_dir / f"failure_{args.stage}.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
