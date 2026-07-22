#!/usr/bin/env python3
"""Run one development-only GeoRoute-AdaTAD training/evaluation cell.

This is intentionally a narrow execution adapter around the existing OpenTAD
``tools/train.py`` and ``tools/test.py`` entry points.  It never opens an
official-test split, never chooses a checkpoint by a metric, and never emits a
paper-grade cost record.  The only checkpoint it evaluates is the final EMA
checkpoint of the frozen stage budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    GEOROUTE_EXPERIMENT_SCHEMA,
    DEVELOPMENT_SEEDS,
    bind_development_config,
    canonical_sha256,
    paper_variant_name,
    sha256_file,
    stage_cell_relative_path,
    stage_epochs,
    variant_spec,
)


GEOROUTE_STAGE_RESULT_SCHEMA = "georoute_adatad_stage_result_v1"
_AVERAGE_MAP = re.compile(r"Average-mAP:\s*([0-9]+(?:\.[0-9]+)?)\s*\(%\)")
_TIOU_MAP = re.compile(
    r"mAP at tIoU\s+([0-9]+(?:\.[0-9]+)?)\s+is\s+([0-9]+(?:\.[0-9]+)?)%"
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _run_logged(command: list[str], *, log_path: Path, env: Mapping[str, str]) -> None:
    """Stream one command to an immutable per-cell text log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {' '.join(command)}")


def parse_official_style_map(log_text: str) -> dict[str, float]:
    """Parse OpenTAD's evaluator log without accepting partial metrics."""

    average_matches = _AVERAGE_MAP.findall(log_text)
    if not average_matches:
        raise ValueError("test log has no Average-mAP line")
    values: dict[str, float] = {"average_mAP": float(average_matches[-1])}
    for threshold, metric in _TIOU_MAP.findall(log_text):
        normalized = f"{float(threshold):.1f}"
        values[f"mAP@{normalized}"] = float(metric)
    required = {f"mAP@{threshold}" for threshold in ("0.3", "0.4", "0.5", "0.6", "0.7")}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"test log lacks required tIoU metrics: {missing}")
    return values


def _development_profile(profile_path: Path) -> dict[str, Any]:
    profile = _read_json(profile_path)
    scope = profile.get("scope")
    if not isinstance(scope, dict) or scope.get("development_only") is not True:
        raise ValueError("GeoRoute profile is not development-only")
    if scope.get("paper_grade_end_to_end_claim_allowed") is not False:
        raise ValueError("GeoRoute development profiler was incorrectly promoted to paper-grade cost")
    for key in ("window_wall_p50_ms", "window_wall_p95_ms", "peak_allocated_mb"):
        value = profile.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"GeoRoute development profile lacks finite {key}")
    return {
        "development_window_wall_p50_ms": float(profile["window_wall_p50_ms"]),
        "development_window_wall_p95_ms": float(profile["window_wall_p95_ms"]),
        "development_peak_allocated_mb": float(profile["peak_allocated_mb"]),
        "paper_grade_end_to_end_claim_allowed": False,
        "profile_file_sha256": sha256_file(profile_path),
        "raw_scope": scope,
    }


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    commit = completed.stdout.strip().lower()
    if len(commit) != 40:
        raise RuntimeError("GeoRoute runner could not resolve a full runtime commit")
    return commit


def build_stage_result(
    *,
    stage: str,
    variant: str,
    seed: int,
    token_budget: int | None,
    binding: Mapping[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    prediction_path: Path,
    profile_path: Path,
    test_log_path: Path,
    runtime_commit: str,
) -> dict[str, Any]:
    """Build one result-blind, development-only cell receipt."""

    metrics = parse_official_style_map(test_log_path.read_text(encoding="utf-8", errors="replace"))
    profile = _development_profile(profile_path)
    raw_profile = _read_json(profile_path)
    audit = raw_profile.get("last_georoute_audit")
    if not isinstance(audit, dict):
        raise ValueError("GeoRoute development profile lacks the last routing audit")
    if int(audit.get("heavy_backbone_forward_count", -1)) != 1:
        raise ValueError("development cell did not preserve exactly one heavy backbone forward")
    if audit.get("uses_grid_sample") is not False or audit.get("uses_resized_local_crop") is not False:
        raise ValueError("development cell violated native-token routing constraints")
    selected_k = int(audit.get("target_k", -1))
    spec = variant_spec(variant, token_budget=token_budget)
    expected_k = selected_k if spec["tokens_per_tubelet"] is None else int(spec["tokens_per_tubelet"])
    if selected_k != expected_k:
        raise ValueError("routing audit target K does not match the bound variant")
    result = {
        "schema_version": GEOROUTE_STAGE_RESULT_SCHEMA,
        "status": "PASS_DEVELOPMENT_ONLY",
        "experiment_schema_version": GEOROUTE_EXPERIMENT_SCHEMA,
        "stage": stage,
        "variant": variant,
        "paper_variant": paper_variant_name(variant),
        "seed": int(seed),
        "token_budget": int(expected_k),
        "metrics": metrics,
        "profile": profile,
        "routing_audit": audit,
        "binding_sha256": str(binding["binding_sha256"]),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_sha256": sha256_file(prediction_path),
        "test_log_sha256": sha256_file(test_log_path),
        "runtime_commit": runtime_commit,
        "official_test_opened": False,
        "manual_roi_used": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "raw_prediction_cache_used": False,
        "paper_grade_result_record_emitted": False,
        "paper_claim_allowed": False,
    }
    result["stage_result_sha256"] = canonical_sha256(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("p1", "p2", "p3"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--token-budget", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.seed not in DEVELOPMENT_SEEDS:
        raise ValueError("stage runner seed is outside the frozen development seed set")
    if not args.expected_commit or _current_commit() != args.expected_commit.lower():
        raise RuntimeError("GeoRoute source snapshot does not match the bound commit")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GeoRoute development cell must run inside a Slurm job")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(","):
        raise RuntimeError("GeoRoute development cell must see exactly one Slurm GPU")

    run_root = args.run_root.resolve()
    if "/data/run01/sczc063/yuzibo/" not in run_root.as_posix() + "/":
        raise ValueError("GeoRoute run root must remain inside the remote write boundary")
    cell_path = stage_cell_relative_path(
        stage=args.stage,
        variant=args.variant,
        seed=args.seed,
        token_budget=args.token_budget,
    )
    budget_label = cell_path.parts[-2]
    work_root = run_root / cell_path
    bound_config = (
        run_root
        / "control"
        / "bound_configs"
        / f"{args.stage}_{args.variant}_{budget_label}_seed{args.seed}.py"
    )
    if work_root.exists() or bound_config.exists():
        raise FileExistsError("GeoRoute cell namespace already exists; refusing overwrite or resume")
    work_root.mkdir(parents=True, exist_ok=False)
    bound_config.parent.mkdir(parents=True, exist_ok=True)

    cfg = bind_development_config(
        source_config_path=args.source_config,
        variant=args.variant,
        stage=args.stage,
        seed=args.seed,
        work_dir=work_root,
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        token_budget=args.token_budget,
    )
    cfg.post_processing.save_dict = True
    cfg.georoute_development_profile = dict(enabled=True)
    cfg.dump(str(bound_config))

    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    train_log = work_root / "train.out"
    test_log = work_root / "test.out"
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
    _run_logged(
        [*torchrun, "tools/train.py", str(bound_config), "--seed", str(args.seed), "--id", "0"],
        log_path=train_log,
        env=inherited,
    )
    effective_work_dir = work_root / "gpu1_id0"
    checkpoint_path = effective_work_dir / "checkpoint" / f"epoch_{stage_epochs(args.stage) - 1}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GeoRoute final EMA checkpoint is missing: {checkpoint_path}")
    _run_logged(
        [
            *torchrun,
            "tools/test.py",
            str(bound_config),
            "--checkpoint",
            str(checkpoint_path),
            "--seed",
            str(args.seed),
            "--id",
            "0",
        ],
        log_path=test_log,
        env=inherited,
    )
    prediction_path = effective_work_dir / "result_detection.json"
    profile_path = effective_work_dir / "georoute_development_profile.json"
    for artifact in (prediction_path, profile_path):
        if not artifact.is_file():
            raise FileNotFoundError(f"GeoRoute evaluation artifact is missing: {artifact}")
    result = build_stage_result(
        stage=args.stage,
        variant=args.variant,
        seed=args.seed,
        token_budget=args.token_budget,
        binding=cfg.georoute_runtime_binding,
        config_path=bound_config,
        checkpoint_path=checkpoint_path,
        prediction_path=prediction_path,
        profile_path=profile_path,
        test_log_path=test_log,
        runtime_commit=args.expected_commit.lower(),
    )
    _atomic_write_json(work_root / "stage_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
