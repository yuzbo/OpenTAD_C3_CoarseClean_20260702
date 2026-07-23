#!/usr/bin/env python3
"""Result-blind Slurm DAG dispatcher for GeoRoute-AdaTAD development stages.

The dispatcher is intentionally the only component allowed to submit a later
stage.  P2/P3 are therefore prepared up front but cannot run before their
predeclared development decision exists.  It never opens the official test.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.finalize_georoute_p0_gate import finalize  # noqa: E402
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    DEVELOPMENT_SEEDS,
    MATCHED_K,
    P1_VARIANTS,
    P3_ABLATION_VARIANTS,
    P3_BUDGET_VARIANTS,
    canonical_sha256,
    select_p1_roi_candidate,
    select_p2_roi_candidate,
    sha256_file,
    stage_cell_relative_path,
)


GEOROUTE_DAG_SCHEMA = "georoute_adatad_development_dag_v1"
GEOROUTE_STAGE_RESULT_SCHEMA = "georoute_adatad_stage_result_v1"

# The N16R4 outer allocation is site-policy scaffolding only.  The matching
# launcher immediately enters an exact one-GPU/5-CPU/96G step for model work.
GPU_OUTER_SLURM_ARGS = ("--gpus", "2", "--cpus-per-task", "8")

# The site rejects control-plane jobs without a GPU declaration. Dispatchers
# do not create a model or execute CUDA; this is the smallest valid batch
# allocation needed to seal a receipt and submit a gated successor.  Do not
# pin --mem here: N16R4 rejected that otherwise harmless explicit request.
CONTROL_SLURM_ARGS = ("--gpus", "1", "--cpus-per-task", "1")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_stage_result(
    run_root: Path,
    stage: str,
    variant: str,
    seed: int,
    token_budget: int | None = None,
) -> dict[str, Any]:
    path = run_root / stage_cell_relative_path(
        stage=stage,
        variant=variant,
        seed=seed,
        token_budget=token_budget,
    ) / "stage_result.json"
    record = _read_json(path)
    digest = record.pop("stage_result_sha256", None)
    if digest != canonical_sha256(record):
        raise ValueError(f"stage result self-hash mismatch: {path}")
    record["stage_result_sha256"] = digest
    if record.get("schema_version") != GEOROUTE_STAGE_RESULT_SCHEMA:
        raise ValueError(f"unexpected stage result schema: {path}")
    if record.get("status") != "PASS_DEVELOPMENT_ONLY":
        raise ValueError(f"stage result is not PASS_DEVELOPMENT_ONLY: {path}")
    if record.get("official_test_opened") is not False:
        raise ValueError(f"stage result opened official test: {path}")
    return record


def _clean_export_value(value: str, *, name: str) -> str:
    if not value or value != value.strip() or "," in value or "\n" in value:
        raise ValueError(f"{name} cannot be safely exported through sbatch")
    return value


def _require_submit_capacity(*, additional_jobs: int) -> None:
    """Fail before a stage can leave an unusable partial Slurm matrix."""

    if not os.environ.get("SLURM_JOB_ID"):
        return
    if additional_jobs <= 0:
        raise ValueError("additional_jobs must be positive")
    user = os.environ.get("SLURM_JOB_USER") or os.environ.get("USER")
    account = os.environ.get("SLURM_JOB_ACCOUNT")
    if not user or not account:
        raise RuntimeError("GeoRoute cannot determine its Slurm user/account for submit-cap validation")
    active = subprocess.run(
        ["squeue", "-h", "-u", user],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if active.returncode != 0:
        raise RuntimeError(f"GeoRoute cannot query active Slurm jobs: {active.stderr.strip()}")
    limit_result = subprocess.run(
        [
            "sacctmgr",
            "-n",
            "-P",
            "show",
            "assoc",
            "where",
            f"user={user}",
            f"account={account}",
            "format=MaxSubmitJobs",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if limit_result.returncode != 0:
        raise RuntimeError(f"GeoRoute cannot query Slurm submit cap: {limit_result.stderr.strip()}")
    limits = [
        int(line.split("|", 1)[0])
        for line in limit_result.stdout.splitlines()
        if line.split("|", 1)[0].strip().isdigit()
    ]
    if not limits:
        raise RuntimeError("GeoRoute cannot determine a finite Slurm MaxSubmitJobs limit")
    active_count = len([line for line in active.stdout.splitlines() if line.strip()])
    submit_limit = min(limits)
    if active_count + additional_jobs > submit_limit:
        raise RuntimeError(
            "GeoRoute refuses a partial stage matrix: "
            f"active={active_count}, required_additional={additional_jobs}, "
            f"MaxSubmitJobs={submit_limit}"
        )


def _cancel_submitted_jobs(job_ids: Sequence[str]) -> None:
    if not job_ids:
        return
    subprocess.run(
        ["scancel", *job_ids],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _base_exports(args: argparse.Namespace, *, action: str) -> dict[str, str]:
    values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_RUN_ROOT": str(args.run_root.resolve()),
        "GEOROUTE_SOURCE_CONFIG": str(args.source_config.resolve()),
        "GEOROUTE_MANIFEST": str(args.manifest.resolve()),
        "GEOROUTE_DEVELOPMENT_ANNOTATION": str(args.development_annotation.resolve()),
        "GEOROUTE_CLASS_MAP": str(args.class_map.resolve()),
        "GEOROUTE_DEVELOPMENT_VIDEO_ROOT": str(args.development_video_root.resolve()),
        "GEOROUTE_PRETRAINED": str(args.pretrained.resolve()),
        "GEOROUTE_EXPECTED_COMMIT": str(args.expected_commit).lower(),
        "GEOROUTE_DAG_ACTION": action,
    }
    return {name: _clean_export_value(value, name=name) for name, value in values.items()}


def _sbatch(
    *,
    args: argparse.Namespace,
    name: str,
    script: Path,
    exports: Mapping[str, str],
    dependency_ids: Sequence[str] = (),
    gpu: bool,
    test_only: bool = False,
) -> str:
    slurm_dir = args.run_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        name,
        "--output",
        str(slurm_dir / f"{name}.%j.out"),
        "--error",
        str(slurm_dir / f"{name}.%j.err"),
    ]
    if test_only:
        command.append("--test-only")
    if dependency_ids:
        command.extend(["--dependency", "afterok:" + ":".join(map(str, dependency_ids))])
    if gpu:
        command.extend(GPU_OUTER_SLURM_ARGS)
    else:
        command.extend(CONTROL_SLURM_ARGS)
    export_items = ["ALL", *(f"{name}={value}" for name, value in sorted(exports.items()))]
    command.extend(["--export", ",".join(export_items), str(script)])
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"sbatch failed for {name}: {completed.stderr.strip() or completed.stdout.strip()}")
    if test_only:
        return "TEST_ONLY_PASS"
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned an invalid job id for {name}: {completed.stdout!r}")
    return job_id


def _write_submission(
    *, run_root: Path, label: str, jobs: Mapping[str, str], parent_receipt: str | None
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": GEOROUTE_DAG_SCHEMA,
        "label": label,
        "jobs": dict(jobs),
        "parent_receipt_sha256": parent_receipt,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    payload["submission_sha256"] = canonical_sha256(payload)
    path = run_root / "control" / f"{label}_submission.json"
    _atomic_write_json(path, payload)
    return path


def _validate_sealed_p0_parent(p0_run_root: Path) -> tuple[dict[str, Any], Path]:
    """Recompute and validate the immutable P0 suite before P1 can start."""

    p0_run_root = p0_run_root.resolve()
    if os.environ.get("SLURM_JOB_ID") and "/data/run01/sczc063/yuzibo/" not in p0_run_root.as_posix() + "/":
        raise ValueError("GeoRoute P0 parent must remain inside the remote write boundary")
    receipt_path = p0_run_root / "control" / "p0_finalization.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    stored = _read_json(receipt_path)
    recomputed = finalize(
        dense=p0_run_root / "p0" / "dense_native_parity.json",
        hybrid=p0_run_root / "p0" / "hybrid_straight_through.json",
        score_function=p0_run_root / "p0" / "roi_score_function.json",
    )
    if stored != recomputed:
        raise ValueError("sealed GeoRoute P0 receipt differs from its recomputed report suite")
    if stored.get("status") != "PASS_MECHANICAL_ONLY":
        raise ValueError("GeoRoute P0 parent is not a passed mechanical-only suite")
    if stored.get("verified_properties", {}).get("official_test_opened") is not False:
        raise ValueError("GeoRoute P0 parent opened the official test")
    return stored, receipt_path


def _validate_bootstrap_inputs(args: argparse.Namespace) -> None:
    input_paths = {
        "source config": args.source_config,
        "manifest": args.manifest,
        "development annotation": args.development_annotation,
        "class map": args.class_map,
        "pretrained checkpoint": args.pretrained,
    }
    for label, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"GeoRoute P1 bootstrap {label} is missing: {path}")
    if not args.development_video_root.is_dir():
        raise FileNotFoundError(
            f"GeoRoute P1 bootstrap development video root is missing: {args.development_video_root}"
        )
    if "test" in args.development_video_root.name.lower():
        raise ValueError("GeoRoute P1 bootstrap cannot use an official-test video root")


def _p1_bootstrap(args: argparse.Namespace) -> int:
    """Submit P1 from a sealed P0 suite without replaying P0.

    The resulting P1 receipt predeclares the P2/P3 successors but leaves their
    GPU jobs absent until the result-blind selectors authorize a survivor.
    """

    if args.p0_run_root is None:
        raise ValueError("p1-bootstrap requires --p0-run-root")
    _validate_bootstrap_inputs(args)
    p0_receipt, p0_receipt_path = _validate_sealed_p0_parent(args.p0_run_root)
    cells = [(variant, 3407, None) for variant in P1_VARIANTS]
    _require_submit_capacity(additional_jobs=len(cells) + 1)
    if args.run_root.exists():
        raise FileExistsError("GeoRoute P1 bootstrap namespace already exists")
    args.run_root.mkdir(parents=True, exist_ok=False)
    (args.run_root / "control").mkdir()
    (args.run_root / "slurm").mkdir()

    jobs = _submit_stage_matrix(
        args=args,
        stage="p1",
        cells=cells,
        parent_receipt=str(p0_receipt["suite_sha256"]),
        next_action="p1-select",
    )
    intent = {
        "schema_version": GEOROUTE_DAG_SCHEMA,
        "status": "P1_SUBMITTED_WITH_RESULT_GATED_P2_P3",
        "p0_parent": {
            "run_root": str(args.p0_run_root.resolve()),
            "finalization_path": str(p0_receipt_path),
            "finalization_file_sha256": sha256_file(p0_receipt_path),
            "suite_sha256": str(p0_receipt["suite_sha256"]),
        },
        "p1_cells": [
            {"stage": "p1", "variant": variant, "seed": seed, "token_budget": budget}
            for variant, seed, budget in cells
        ],
        "p1_jobs": jobs,
        "frozen_successor_policy": {
            "p2": "submit only when p1-select records ADVANCE_STRUCTURED_ROI_TO_P2",
            "p3": "submit only when p2-select records ADVANCE_STRUCTURED_ROI_TO_P3",
            "official_test_opened": False,
            "amod_included": False,
        },
        "paper_claim_allowed": False,
        "official_test_opened": False,
    }
    intent["bootstrap_sha256"] = canonical_sha256(intent)
    _atomic_write_json(args.run_root / "control" / "p1_bootstrap.json", intent)
    return 0


def _unique_cells(cells: Sequence[tuple[str, int, int | None]]) -> list[tuple[str, int, int | None]]:
    """Preserve frozen order while preventing a K=64 budget/ablation duplicate."""

    unique: list[tuple[str, int, int | None]] = []
    seen: set[tuple[str, int, int | None]] = set()
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    return unique


def _submit_stage_matrix(
    *,
    args: argparse.Namespace,
    stage: str,
    cells: Sequence[tuple[str, int, int | None]],
    parent_receipt: str,
    next_action: str,
) -> dict[str, str]:
    stage_script = ROOT / "scripts" / "run_georoute_stage_slurm.sh"
    dispatch_script = ROOT / "scripts" / "run_georoute_dispatch_slurm.sh"
    if not stage_script.is_file() or not dispatch_script.is_file():
        raise FileNotFoundError("GeoRoute Slurm stage/dispatch script is missing")
    _require_submit_capacity(additional_jobs=len(cells) + 1)
    prepared_cells: list[tuple[str, dict[str, str]]] = []
    for variant, seed, budget in cells:
        label = f"georoute_{stage}_{variant}_s{seed}" + (f"_k{budget}" if budget else "")
        exports = _base_exports(args, action="stage")
        exports.update(
            GEOROUTE_STAGE=stage,
            GEOROUTE_VARIANT=variant,
            GEOROUTE_SEED=str(seed),
            GEOROUTE_TOKEN_BUDGET="" if budget is None else str(budget),
        )
        prepared_cells.append((label, exports))
    dispatch_exports = _base_exports(args, action=next_action)
    dispatch_exports["GEOROUTE_PARENT_RECEIPT"] = parent_receipt

    # Reject a full matrix before its first real job is submitted. This avoids
    # a scheduler submit-cap from creating an unusable partial experiment.
    for label, exports in prepared_cells:
        _sbatch(
            args=args,
            name=label,
            script=stage_script,
            exports=exports,
            gpu=True,
            test_only=True,
        )
    _sbatch(
        args=args,
        name=f"georoute_{next_action.replace('-', '_')}",
        script=dispatch_script,
        exports=dispatch_exports,
        gpu=False,
        test_only=True,
    )

    jobs: dict[str, str] = {}
    try:
        for label, exports in prepared_cells:
            jobs[label] = _sbatch(args=args, name=label, script=stage_script, exports=exports, gpu=True)
        dispatcher = _sbatch(
            args=args,
            name=f"georoute_{next_action.replace('-', '_')}",
            script=dispatch_script,
            exports=dispatch_exports,
            dependency_ids=list(jobs.values()),
            gpu=False,
        )
    except Exception:
        _cancel_submitted_jobs(list(jobs.values()))
        raise
    jobs[f"{next_action}_dispatcher"] = dispatcher
    _write_submission(run_root=args.run_root, label=stage, jobs=jobs, parent_receipt=parent_receipt)
    return jobs


def _p0_finalize(args: argparse.Namespace) -> int:
    reports = args.run_root / "p0"
    receipt = finalize(
        dense=reports / "dense_native_parity.json",
        hybrid=reports / "hybrid_straight_through.json",
        score_function=reports / "roi_score_function.json",
    )
    receipt_path = args.run_root / "control" / "p0_finalization.json"
    _atomic_write_json(receipt_path, receipt)
    if receipt.get("status") != "PASS_MECHANICAL_ONLY":
        raise ValueError("GeoRoute P0 finalizer did not pass")
    cells = [(variant, 3407, None) for variant in P1_VARIANTS]
    _submit_stage_matrix(
        args=args,
        stage="p1",
        cells=cells,
        parent_receipt=str(receipt["suite_sha256"]),
        next_action="p1-select",
    )
    return 0


def _p1_select(args: argparse.Namespace) -> int:
    records = {variant: _load_stage_result(args.run_root, "p1", variant, 3407) for variant in P1_VARIANTS}
    decision = select_p1_roi_candidate(records)
    decision_path = args.run_root / "control" / "p1_selection.json"
    _atomic_write_json(decision_path, decision)
    if decision["status"] != "ADVANCE_STRUCTURED_ROI_TO_P2":
        return 0
    candidate = str(decision["best_structured_variant"])
    cells = [
        (variant, seed, None)
        for variant in ("fixed_lattice", "fixed_lattice_geometry", "random", "free", candidate)
        for seed in DEVELOPMENT_SEEDS
    ]
    _submit_stage_matrix(
        args=args,
        stage="p2",
        cells=cells,
        parent_receipt=str(decision["selection_sha256"]),
        next_action="p2-select",
    )
    return 0


def _p2_select(args: argparse.Namespace) -> int:
    p1 = _read_json(args.run_root / "control" / "p1_selection.json")
    if p1.get("status") != "ADVANCE_STRUCTURED_ROI_TO_P2":
        raise ValueError("P2 dispatcher was reached without P1 authorization")
    candidate = str(p1["best_structured_variant"])
    records = {
        variant: [_load_stage_result(args.run_root, "p2", variant, seed) for seed in DEVELOPMENT_SEEDS]
        for variant in ("fixed_lattice", "fixed_lattice_geometry", "random", "free", candidate)
    }
    decision = select_p2_roi_candidate(records, candidate_variant=candidate)
    decision_path = args.run_root / "control" / "p2_selection.json"
    _atomic_write_json(decision_path, decision)
    if decision["status"] != "ADVANCE_STRUCTURED_ROI_TO_P3":
        return 0
    budget_cells = [
        (variant, seed, budget)
        for budget in P3_BUDGET_VARIANTS
        for variant in ("fixed_lattice", "free", candidate)
        for seed in DEVELOPMENT_SEEDS
    ]
    mechanism_variants = ("fixed_lattice_geometry", "roi", "hybrid", *P3_ABLATION_VARIANTS)
    mechanism_cells = [
        (variant, seed, MATCHED_K)
        for variant in mechanism_variants
        for seed in DEVELOPMENT_SEEDS
    ]
    cells = _unique_cells([*budget_cells, *mechanism_cells])
    _submit_stage_matrix(
        args=args,
        stage="p3",
        cells=cells,
        parent_receipt=str(decision["selection_sha256"]),
        next_action="p3-finalize",
    )
    return 0


def _p3_finalize(args: argparse.Namespace) -> int:
    p2 = _read_json(args.run_root / "control" / "p2_selection.json")
    if p2.get("status") != "ADVANCE_STRUCTURED_ROI_TO_P3":
        raise ValueError("P3 finalizer was reached without P2 authorization")
    candidate = str(p2["candidate_variant"])
    cells = [
        (variant, seed, budget)
        for budget in P3_BUDGET_VARIANTS
        for variant in ("fixed_lattice", "free", candidate)
        for seed in DEVELOPMENT_SEEDS
    ]
    cells.extend(
        (variant, seed, MATCHED_K)
        for variant in ("fixed_lattice_geometry", "roi", "hybrid", *P3_ABLATION_VARIANTS)
        for seed in DEVELOPMENT_SEEDS
    )
    cells = _unique_cells(cells)
    records = []
    for variant, seed, budget in cells:
        record = _load_stage_result(args.run_root, "p3", variant, seed, token_budget=budget)
        if int(record["token_budget"]) != int(budget):
            raise ValueError("P3 result budget differs from its frozen cell")
        records.append(record)
    receipt = {
        "schema_version": GEOROUTE_DAG_SCHEMA,
        "status": "P3_DEVELOPMENT_COMPLETE_PENDING_GENERALIZATION",
        "record_count": len(records),
        "record_hashes": sorted(record["stage_result_sha256"] for record in records),
        "candidate_variant": candidate,
        "official_test_opened": False,
        "paper_claim_allowed": False,
        "next_required_evidence": [
            "paper-grade total-cost measurement",
            "second detector or dataset generalization",
            "frozen one-time official-test protocol",
        ],
    }
    receipt["completion_sha256"] = canonical_sha256(receipt)
    _atomic_write_json(args.run_root / "control" / "p3_development_completion.json", receipt)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("p0-finalize", "p1-bootstrap", "p1-select", "p2-select", "p3-finalize"),
        required=True,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--p0-run-root", type=Path)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.run_root = args.run_root.resolve()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GeoRoute DAG dispatcher must run inside Slurm")
    actions = {
        "p0-finalize": _p0_finalize,
        "p1-bootstrap": _p1_bootstrap,
        "p1-select": _p1_select,
        "p2-select": _p2_select,
        "p3-finalize": _p3_finalize,
    }
    return actions[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
