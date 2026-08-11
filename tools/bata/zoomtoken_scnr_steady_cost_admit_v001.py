#!/usr/bin/env python3
"""Admit a Job-9 candidate only after independent scheduler confirmation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.zoomtoken_scnr_steady_cost_contract_v001 import (
    LEAF_ORDERS,
    STUDY_ID,
    add_self_hash,
    analyze_complete_leaves_with_draws,
    exclusive_write_json,
    read_json_object,
    read_jsonl_objects,
    require_self_hash,
    validate_afterany_dependency,
    validate_bootstrap_draws,
    validate_pre_run,
    validate_scheduler_job,
)
from tools.bata.zoomtoken_scnr_steady_cost_finalize_v001 import (
    _validate_execution_classes,
    load_complete_leaf,
    query_sacct,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-run", type=Path, required=True)
    parser.add_argument("--finalizer-job-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.finalizer_job_id.isdigit():
        raise ValueError("steady-cost finalizer Job ID must be numeric")
    pre_run = validate_pre_run(
        read_json_object(args.pre_run, label="steady-cost full PRE_RUN"), phase="full"
    )
    aggregate_path = (
        Path(pre_run["results_root"])
        / STUDY_ID
        / pre_run["repair_sha"]
        / "aggregate"
        / "aggregate_candidate.json"
    )
    aggregate = read_json_object(aggregate_path, label="steady-cost aggregate candidate")
    require_self_hash(aggregate, field="aggregate_sha256", label="aggregate candidate")
    leaf_jobs = aggregate.get("leaf_job_ids")
    if (
        aggregate.get("schema_version")
        != "zoomtoken_scnr_steady_cost_aggregate_candidate_v001"
        or aggregate.get("status") != "AGGREGATE_CANDIDATE_ONLY"
        or aggregate.get("authoritative") is not False
        or aggregate.get("pre_run_sha256") != pre_run["pre_run_sha256"]
        or aggregate.get("repair_sha") != pre_run["repair_sha"]
        or aggregate.get("finalizer_job_id") != args.finalizer_job_id
        or not isinstance(leaf_jobs, dict)
        or list(leaf_jobs) != list(LEAF_ORDERS)
        or len(set(map(str, leaf_jobs.values()))) != 8
        or args.finalizer_job_id in set(map(str, leaf_jobs.values()))
    ):
        raise ValueError("steady-cost aggregate candidate is cross-run or malformed")
    all_jobs = [str(job_id) for job_id in leaf_jobs.values()] + [args.finalizer_job_id]
    scheduler = query_sacct(all_jobs)
    validate_afterany_dependency(
        scheduler[args.finalizer_job_id],
        expected_job_id=args.finalizer_job_id,
        expected_parent_job_ids=[str(job_id) for job_id in leaf_jobs.values()],
    )
    for leaf_id, job_id in leaf_jobs.items():
        validate_scheduler_job(
            scheduler[str(job_id)], expected_job_id=str(job_id), label=leaf_id
        )
    validate_scheduler_job(
        scheduler[args.finalizer_job_id],
        expected_job_id=args.finalizer_job_id,
        label="Job 9",
    )
    leaves = {}
    receipt_payloads = {}
    for leaf_id, job_id in leaf_jobs.items():
        receipt, rows = load_complete_leaf(
            pre_run, leaf_id=leaf_id, job_id=str(job_id)
        )
        expected = aggregate.get("leaf_receipts", {}).get(leaf_id)
        expected_path = (
            Path(pre_run["results_root"])
            / STUDY_ID
            / pre_run["repair_sha"]
            / "leaves"
            / leaf_id
            / "receipt.json"
        )
        if (
            not isinstance(expected, dict)
            or expected.get("path") != str(expected_path)
            or expected.get("receipt_sha256") != receipt["receipt_sha256"]
        ):
            raise ValueError("steady-cost aggregate leaf receipt ledger is invalid")
        leaves[leaf_id] = rows
        receipt_payloads[leaf_id] = receipt
    execution_class = _validate_execution_classes(pre_run, receipt_payloads)
    analysis, bootstrap_draws = analyze_complete_leaves_with_draws(leaves)
    draw_receipt = aggregate.get("bootstrap_draws")
    if not isinstance(draw_receipt, dict):
        raise ValueError("steady-cost aggregate lacks bootstrap draws")
    draw_path = Path(str(draw_receipt.get("path", ""))).resolve()
    try:
        draw_path.relative_to(aggregate_path.parent.resolve())
    except ValueError as error:
        raise ValueError("steady-cost bootstrap draw path escaped aggregate root") from error
    persisted_draws = read_jsonl_objects(
        draw_path, label="steady-cost persisted bootstrap draws"
    )
    validate_bootstrap_draws(persisted_draws, analysis=analysis)
    if (
        int(draw_receipt.get("row_count", -1)) != len(persisted_draws)
        or persisted_draws != bootstrap_draws
    ):
        raise ValueError("steady-cost bootstrap draws are not reproducible")
    if (
        aggregate.get("execution_class") != execution_class
        or aggregate.get("analysis") != analysis
        or aggregate.get("candidate_decision") != analysis["candidate_decision"]
    ):
        raise ValueError("steady-cost aggregate candidate is not reproducible from raw leaves")
    if aggregate.get("candidate_decision") not in {
        "PASS_COST_NONINFERIOR",
        "FAIL_COST_NONINFERIOR",
    } or not isinstance(aggregate.get("analysis"), dict):
        raise ValueError("steady-cost invalid candidate cannot be admitted authoritatively")
    final_path = aggregate_path.with_name("final_decision.json")
    decision = add_self_hash(
        {
            "schema_version": "zoomtoken_scnr_steady_cost_final_decision_v001",
            "study_id": STUDY_ID,
            "decision": aggregate["candidate_decision"],
            "authoritative": True,
            "aggregate_candidate": {
                "path": str(aggregate_path.resolve()),
                "aggregate_sha256": aggregate["aggregate_sha256"],
            },
            "pre_run_sha256": pre_run["pre_run_sha256"],
            "repair_sha": pre_run["repair_sha"],
            "scheduler_confirmation": scheduler,
            "analysis": aggregate["analysis"],
            "held_out_test_opened": False,
            "metric_evaluation_executed": False,
            "paper_claim_allowed": False,
            "automatic_rerun_allowed": False,
        },
        field="decision_sha256",
    )
    exclusive_write_json(final_path, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
