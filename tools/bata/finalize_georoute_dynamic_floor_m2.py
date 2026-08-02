#!/usr/bin/env python3
"""All-terminal, artifact-driven finalizer for dynamic ROI-floor M2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_dynamic_floor_m2_contract import (  # noqa: E402
    DYNAMIC_FLOOR_M2_ARM_ORDER,
    DYNAMIC_FLOOR_M2_DEPLOYMENT_SCHEMA,
    DYNAMIC_FLOOR_M2_SEED,
    dynamic_floor_m2_cell_relative_path,
    finalize_dynamic_floor_m2,
    require_clean_dynamic_floor_m2_checkout,
)
from tools.bata.georoute_experiment_contract import canonical_sha256  # noqa: E402
from tools.bata.georoute_stage_runner import _atomic_write_json  # noqa: E402


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _read_optional_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _validate_deployment(
    deployment: Mapping[str, Any], *, expected_commit: str
) -> dict[str, Any]:
    deployment = dict(deployment)
    unsigned = dict(deployment)
    observed_hash = unsigned.pop("deployment_sha256", None)
    jobs = deployment.get("jobs")
    dependencies = deployment.get("dependencies")
    if (
        deployment.get("schema_version") != DYNAMIC_FLOOR_M2_DEPLOYMENT_SCHEMA
        or deployment.get("status") != "DEPLOYED_DYNAMIC_FLOOR_M2_DAG"
        or deployment.get("runtime_commit") != expected_commit
        or observed_hash != canonical_sha256(unsigned)
        or not isinstance(jobs, Mapping)
        or set(jobs) != {"g1", "g2", "paired_cost", "finalizer"}
        or any(not str(job_id).isdigit() for job_id in jobs.values())
        or not isinstance(dependencies, Mapping)
        or dependencies.get("paired_cost")
        != {"type": "afterok", "predecessors": [str(jobs["g1"]), str(jobs["g2"])]}
        or dependencies.get("finalizer")
        != {
            "type": "afterany",
            "predecessors": [
                str(jobs["g1"]),
                str(jobs["g2"]),
                str(jobs["paired_cost"]),
            ],
        }
        or deployment.get("official_test_opened") is not False
        or deployment.get("paper_claim_allowed") is not False
    ):
        raise ValueError("dynamic floor M2 deployment receipt is invalid")
    if str(os.environ.get("SLURM_JOB_ID", "")) != str(jobs["finalizer"]):
        raise RuntimeError("dynamic floor M2 finalizer job differs from deployment")
    deployment["deployment_sha256"] = observed_hash
    return deployment


def _slurm_state(job_id: str) -> dict[str, Any]:
    query = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in query.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == str(job_id):
            rows.append(
                {"job_id": fields[0], "state": fields[1], "exit_code": fields[2]}
            )
    return {
        "query_returncode": query.returncode,
        "rows": rows,
        "stderr": query.stderr.strip()[:500],
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    require_clean_dynamic_floor_m2_checkout(
        expected_commit=expected_commit, root=ROOT
    )
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("dynamic floor M2 finalizer left the write boundary")
    deployment_path = args.deployment.resolve()
    deployment = _read_optional_object(deployment_path)
    if deployment is None:
        raise FileNotFoundError(deployment_path)
    deployment = _validate_deployment(
        deployment, expected_commit=expected_commit
    )

    stage_results: dict[str, dict[str, Any]] = {}
    stage_artifacts = {}
    for arm in DYNAMIC_FLOOR_M2_ARM_ORDER:
        cell = run_root / dynamic_floor_m2_cell_relative_path(
            arm=arm, seed=DYNAMIC_FLOOR_M2_SEED
        )
        result_path = cell / "stage_result.json"
        failure_path = cell / "stage_failure.json"
        result = _read_optional_object(result_path)
        if result is not None:
            stage_results[arm] = result
        stage_artifacts[arm] = {
            "stage_result_path": str(result_path),
            "stage_result_present": result is not None,
            "stage_failure_path": str(failure_path),
            "stage_failure_present": failure_path.is_file(),
        }
    cost_path = run_root / "cost" / "paired_cost_profile.json"
    cost_profile = _read_optional_object(cost_path)
    result = finalize_dynamic_floor_m2(
        stage_results,
        cost_profile,
        expected_commit=expected_commit,
    )
    result.pop("finalization_sha256", None)
    jobs = deployment["jobs"]
    result.update(
        deployment_path=str(deployment_path),
        deployment_sha256=deployment["deployment_sha256"],
        stage_artifacts=stage_artifacts,
        paired_cost_path=str(cost_path),
        paired_cost_present=cost_profile is not None,
        slurm_predecessor_states={
            name: _slurm_state(str(jobs[name]))
            for name in ("g1", "g2", "paired_cost")
        },
        finalizer_job_id=str(jobs["finalizer"]),
    )
    result["finalization_sha256"] = canonical_sha256(result)
    output = run_root / "control" / "finalization.json"
    if output.exists():
        raise FileExistsError(output)
    _atomic_write_json(output, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    result = finalize(_parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
