#!/usr/bin/env python3
"""Submit the complete result-blind GeoRoute P0->P3 development DAG.

Only the three CUDA P0 leaves are submitted directly.  Their afterok consumer
seals the mechanical gate and submits P1; P1/P2 consumers in turn submit the
next predeclared matrix only on a result-blind promotion decision.  This is the
only sound way to deploy all stages at once without training a P2/P3 candidate
whose identity is not yet known.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file  # noqa: E402


GEOROUTE_DEPLOYMENT_SCHEMA = "georoute_adatad_development_deployment_v1"

# N16R4 grants only 55 GB per outer GPU allocation.  The launchers below
# create an exact one-GPU/96G Slurm step for model work, so the outer request
# needs two GPUs and eight CPUs to satisfy that site policy without binding a
# physical device or charging a second model forward.
GPU_OUTER_SLURM_ARGS = ("--gpus", "2", "--cpus-per-task", "8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_clean_value(value: str, *, name: str) -> str:
    if not value or value != value.strip() or "," in value or "\n" in value:
        raise ValueError(f"{name} is not safe to export through sbatch")
    return value


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def _sbatch(
    *,
    name: str,
    script: Path,
    logs: Path,
    exports: Mapping[str, str],
    dependency: list[str] | None = None,
    gpu: bool,
) -> str:
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        name,
        "--output",
        str(logs / f"{name}.%j.out"),
        "--error",
        str(logs / f"{name}.%j.err"),
    ]
    if dependency:
        command.extend(["--dependency", "afterok:" + ":".join(dependency)])
    if gpu:
        command.extend(GPU_OUTER_SLURM_ARGS)
    else:
        command.extend(["--cpus-per-task", "1", "--mem", "4G"])
    command.extend(
        [
            "--export",
            ",".join(["ALL", *(f"{key}={value}" for key, value in sorted(exports.items()))]),
            str(script),
        ]
    )
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"sbatch failed for {name}: {completed.stderr.strip() or completed.stdout.strip()}")
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned invalid job id for {name}: {completed.stdout!r}")
    return job_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    run_root = args.run_root.resolve()
    if not run_root.as_posix().startswith("/data/run01/sczc063/yuzibo/"):
        raise ValueError("GeoRoute run root must stay inside the remote write boundary")
    if run_root.exists():
        raise FileExistsError("GeoRoute deployment namespace already exists")
    expected_commit = str(args.expected_commit).lower()
    actual_commit = _git_output("rev-parse", "HEAD").lower()
    if actual_commit != expected_commit:
        raise RuntimeError("source snapshot commit differs from --expected-commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("GeoRoute deployment requires a clean remote source snapshot")
    input_paths = {
        "GEOROUTE_SOURCE_CONFIG": args.source_config.resolve(),
        "GEOROUTE_MANIFEST": args.manifest.resolve(),
        "GEOROUTE_DEVELOPMENT_ANNOTATION": args.development_annotation.resolve(),
        "GEOROUTE_CLASS_MAP": args.class_map.resolve(),
        "GEOROUTE_DEVELOPMENT_VIDEO_ROOT": args.development_video_root.resolve(),
        "GEOROUTE_PRETRAINED": args.pretrained.resolve(),
    }
    for name, path in input_paths.items():
        if name == "GEOROUTE_DEVELOPMENT_VIDEO_ROOT":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)

    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "p0").mkdir()
    (run_root / "slurm").mkdir()
    control = run_root / "control"
    control.mkdir()
    base_values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        **{name: str(path) for name, path in input_paths.items()},
    }
    base_exports = {
        name: _require_clean_value(value, name=name) for name, value in base_values.items()
    }
    p0_script = ROOT / "scripts" / "run_georoute_p0_slurm.sh"
    dispatch_script = ROOT / "scripts" / "run_georoute_dispatch_slurm.sh"
    if not p0_script.is_file() or not dispatch_script.is_file():
        raise FileNotFoundError("GeoRoute P0 or dispatcher script is missing")

    p0_specs = {
        # 160x160 contains exactly 10x10 native patches.  Dense parity must
        # select all 100 rather than merely exercising the dense route mode.
        "dense_native_parity": {"mode": "dense", "estimator": "none", "tokens": "100", "context": "0"},
        "hybrid_straight_through": {"mode": "hybrid", "estimator": "straight_through", "tokens": "32", "context": "4"},
        "roi_score_function": {"mode": "roi", "estimator": "score_function", "tokens": "32", "context": "0"},
    }
    jobs: dict[str, str] = {}
    for label, spec in p0_specs.items():
        exports = dict(base_exports)
        exports.update(
            GEOROUTE_P0_OUTPUT=str(run_root / "p0" / f"{label}.json"),
            GEOROUTE_P0_ROUTE_MODE=spec["mode"],
            GEOROUTE_P0_POLICY_ESTIMATOR=spec["estimator"],
            GEOROUTE_P0_TOKENS_PER_TUBELET=spec["tokens"],
            GEOROUTE_P0_CONTEXT_TOKENS=spec["context"],
        )
        jobs[label] = _sbatch(
            name=f"georoute_p0_{label}",
            script=p0_script,
            logs=run_root / "slurm",
            exports=exports,
            gpu=True,
        )
    dispatch_exports = dict(base_exports)
    dispatch_exports["GEOROUTE_DAG_ACTION"] = "p0-finalize"
    jobs["p0_finalize_dispatcher"] = _sbatch(
        name="georoute_p0_finalize",
        script=dispatch_script,
        logs=run_root / "slurm",
        exports=dispatch_exports,
        dependency=list(jobs.values()),
        gpu=False,
    )
    deployment = {
        "schema_version": GEOROUTE_DEPLOYMENT_SCHEMA,
        "status": "SUBMITTED_P0_WITH_RESULT_BLIND_P1_P2_P3_DISPATCH",
        "runtime_commit": expected_commit,
        "source_root": str(ROOT),
        "run_root": str(run_root),
        "input_hashes": {
            name: sha256_file(path) if path.is_file() else None for name, path in input_paths.items()
        },
        "p0_jobs": jobs,
        "frozen_policy": {
            "p1_after_p0": True,
            "p2_only_if_p1_structured_roi_wins": True,
            "p3_only_if_p2_three_seed_structured_roi_wins": True,
            "official_test_opened": False,
            "amod_included": False,
            "amod_reason": "requires independent numerical and total-cost P0 gate",
        },
    }
    deployment["deployment_sha256"] = canonical_sha256(deployment)
    _atomic_write_json(control / "deployment.json", deployment)
    print(json.dumps(deployment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
