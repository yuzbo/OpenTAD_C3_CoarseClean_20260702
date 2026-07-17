from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.bata.duca_cellcf_submission_contract import (
    JOB_ORDER,
    expected_job_name,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/submit_duca_cellcf_suite.sh"
ROLES = {
    "uniform": "none",
    "transition_beta0": "none",
    "cellcf": "none",
    "aggregate": "afterok_three_arms",
    "cost": "afterok_aggregate",
    "completion": "afterok_aggregate_and_cost",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="formal submit precheck is a Linux shell contract",
)
@pytest.mark.parametrize("profile", ("exposure132", "official60"))
def test_prepared_suite_reopens_in_submit_precheck(
    tmp_path: Path,
    profile: str,
) -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    base = tmp_path / profile / "base"
    python_link = base / "conda_envs" / "opentad" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to(Path(sys.executable).resolve())
    run_root = base / "runs" / f"cellcf-{profile}"
    jobs_dir = run_root / "jobs"
    jobs_dir.mkdir(parents=True)

    manifest_path = run_root / "suite_manifest.json"
    manifest = {
        "schema": "duca_cellcf_suite_manifest_v1",
        "ok": True,
        "git_commit": commit,
        "seed": 0,
        "training_profile": profile,
        "real_loader_gate": {"path": str(run_root / "gate.json")},
        "ddp_pilot": {"path": str(run_root / "pilot.json")},
    }
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)

    jobs = []
    for key in JOB_ORDER:
        name = expected_job_name(profile, key, 0, commit)
        job_file = jobs_dir / f"{key}.sbatch"
        job_file.write_text(
            "\n".join(
                (
                    "#!/usr/bin/env bash",
                    f"#SBATCH --job-name={name}",
                    "#SBATCH --clusters=testcluster",
                    f"# DUCA_CELLCF_DEPENDENCY_ROLE={ROLES[key]}",
                    f"# manifest={manifest_sha256}",
                    f"# commit={commit}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        jobs.append(
            {
                "key": key,
                "job_name": name,
                "dependency_role": ROLES[key],
                "job_file": str(job_file.resolve()),
                "job_file_sha256": _sha256(job_file),
            }
        )

    canonical_env = run_root / "canonical_env.tsv"
    canonical_env.write_text(
        f"DUCA_CELLCF_TRAINING_PROFILE={profile}\n",
        encoding="utf-8",
    )
    jobs_tsv = run_root / "jobs.tsv"
    jobs_tsv.write_text("sealed prepared job ledger\n", encoding="utf-8")
    prepared_path = run_root / "prepared_submission.json"
    prepared = {
        "schema": "duca_cellcf_prepared_submission_v1",
        "git_commit": commit,
        "seed": 0,
        "training_profile": profile,
        "target_cluster": "testcluster",
        "checkpoint_interval": 5,
        "suite_manifest": str(manifest_path.resolve()),
        "suite_manifest_sha256": manifest_sha256,
        "canonical_env_file": str(canonical_env.resolve()),
        "canonical_env_sha256": _sha256(canonical_env),
        "jobs_tsv": str(jobs_tsv.resolve()),
        "jobs_tsv_sha256": _sha256(jobs_tsv),
        "job_order": list(JOB_ORDER),
        "jobs": jobs,
    }
    _write_json(prepared_path, prepared)
    prepared_path.with_suffix(".json.sha256").write_text(
        _sha256(prepared_path) + "\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "BASE": str(base.resolve()),
            "RUN_ROOT": str(run_root.resolve()),
            "PRECHECK_ONLY": "1",
        }
    )
    environment.pop("DUCA_CELLCF_TRAINING_PROFILE", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["bash", str(SUBMIT)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"PRECHECK PASS profile={profile}" in result.stdout
    assert not (run_root / "submission_receipts").exists()
