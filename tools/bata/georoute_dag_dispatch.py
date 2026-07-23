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
    stage_cell_relative_path,
)


GEOROUTE_DAG_SCHEMA = "georoute_adatad_development_dag_v1"
GEOROUTE_STAGE_RESULT_SCHEMA = "georoute_adatad_stage_result_v1"

# The N16R4 outer allocation is site-policy scaffolding only.  The matching
# launcher immediately enters an exact one-GPU/5-CPU/96G step for model work.
GPU_OUTER_SLURM_ARGS = ("--gpus", "2", "--cpus-per-task", "8")


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
    if dependency_ids:
        command.extend(["--dependency", "afterok:" + ":".join(map(str, dependency_ids))])
    if gpu:
        command.extend(GPU_OUTER_SLURM_ARGS)
    else:
        command.extend(["--cpus-per-task", "1", "--mem", "4G"])
    export_items = ["ALL", *(f"{name}={value}" for name, value in sorted(exports.items()))]
    command.extend(["--export", ",".join(export_items), str(script)])
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"sbatch failed for {name}: {completed.stderr.strip() or completed.stdout.strip()}")
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
    jobs: dict[str, str] = {}
    for variant, seed, budget in cells:
        label = f"georoute_{stage}_{variant}_s{seed}" + (f"_k{budget}" if budget else "")
        exports = _base_exports(args, action="stage")
        exports.update(
            GEOROUTE_STAGE=stage,
            GEOROUTE_VARIANT=variant,
            GEOROUTE_SEED=str(seed),
            GEOROUTE_TOKEN_BUDGET="" if budget is None else str(budget),
        )
        jobs[label] = _sbatch(args=args, name=label, script=stage_script, exports=exports, gpu=True)
    dispatch_exports = _base_exports(args, action=next_action)
    dispatch_exports["GEOROUTE_PARENT_RECEIPT"] = parent_receipt
    dispatcher = _sbatch(
        args=args,
        name=f"georoute_{next_action.replace('_', '_')}",
        script=dispatch_script,
        exports=dispatch_exports,
        dependency_ids=list(jobs.values()),
        gpu=False,
    )
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
        next_action="p1_select",
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
        next_action="p2_select",
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
        next_action="p3_finalize",
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
    parser.add_argument("--action", choices=("p0-finalize", "p1-select", "p2-select", "p3-finalize"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
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
        "p1-select": _p1_select,
        "p2-select": _p2_select,
        "p3-finalize": _p3_finalize,
    }
    return actions[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
