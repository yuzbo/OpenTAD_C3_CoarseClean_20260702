#!/usr/bin/env python3
"""Train and replay one dynamic SCNR G1/G2 floor M2 arm."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_dynamic_floor_m2_contract import (  # noqa: E402
    DYNAMIC_FLOOR_M2_EPOCHS,
    DYNAMIC_FLOOR_M2_SEED,
    DYNAMIC_FLOOR_M2_STAGE_RESULT_SCHEMA,
    DYNAMIC_FLOOR_M2_STUDY_ID,
    bind_dynamic_floor_m2_config,
    dynamic_floor_m2_arm_spec,
    dynamic_floor_m2_cell_relative_path,
    summarize_dynamic_floor_m2_telemetry,
    validate_dynamic_floor_m2_checkpoint_sidecar,
    validate_dynamic_floor_m2_config,
    validate_dynamic_floor_m2_stage_result,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _atomic_write_json,
    _read_json,
    _run_logged,
    _validate_rendezvous_receipt,
    build_torchrun_prefix,
    parse_official_style_map,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402
from tools.bata.run_georoute_dynamic_stage1_p0 import (  # noqa: E402
    validate_dynamic_stage1_p0_report,
)


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().lower()


def _require_exact_clean_source(expected_commit: str) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    origin = subprocess.run(
        [
            "git",
            "rev-parse",
            "refs/remotes/origin/codex/spatial-zoom-s1-audit-fix-20260715",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().lower()
    if _current_commit() != expected_commit or origin != expected_commit or status:
        raise RuntimeError("dynamic floor M2 requires one exact clean origin commit")


def _run_torchrun(
    *,
    phase: str,
    stage: str,
    arm: str,
    slurm_job_id: str,
    arguments: Sequence[str],
    log_path: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    prefix, receipt = build_torchrun_prefix(
        phase=phase,
        slurm_job_id=slurm_job_id,
        stage=stage,
        variant=arm,
        seed=DYNAMIC_FLOOR_M2_SEED,
        nproc_per_node=1,
    )
    _run_logged([*prefix, *arguments], log_path=log_path, env=env)
    return receipt


def _dump_configs(
    *,
    args: argparse.Namespace,
    cell_root: Path,
) -> tuple[Any, dict[str, Path]]:
    config_root = args.run_root.resolve() / "control" / "bound_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    paths = {
        name: config_root
        / f"{DYNAMIC_FLOOR_M2_STUDY_ID}_{args.arm}_seed{DYNAMIC_FLOOR_M2_SEED}_{name}.py"
        for name in ("train", "accuracy")
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("dynamic floor M2 bound config namespace exists")
    cfg = bind_dynamic_floor_m2_config(
        source_config_path=args.source_config,
        arm=args.arm,
        seed=DYNAMIC_FLOOR_M2_SEED,
        work_dir=cell_root / "training" / "gpu1_id0",
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        runtime_commit=str(args.expected_commit).lower(),
    )
    validate_dynamic_floor_m2_config(cfg, arm=args.arm)
    train_cfg = copy.deepcopy(cfg)
    train_cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = False
    train_cfg.georoute_diagnostic_telemetry = dict(enabled=False)
    train_cfg.georoute_development_profile = dict(enabled=False)
    accuracy_cfg = copy.deepcopy(cfg)
    accuracy_cfg.work_dir = str((cell_root / "accuracy" / "gpu1_id0").resolve())
    accuracy_cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = True
    accuracy_cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    accuracy_cfg.georoute_development_profile = dict(enabled=False)
    for name, bound in (("train", train_cfg), ("accuracy", accuracy_cfg)):
        validate_dynamic_floor_m2_config(bound, arm=args.arm, phase=name)
        bound.dump(str(paths[name]))
    return cfg, paths


def _p0_receipt(
    *,
    args: argparse.Namespace,
    config_path: Path,
    cell_root: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    report_path = cell_root / "p0_report.json"
    _run_logged(
        [
            sys.executable,
            "-m",
            "tools.bata.run_georoute_dynamic_stage1_p0",
            "--config",
            str(config_path),
            "--pretrained",
            str(args.pretrained.resolve()),
            "--output",
            str(report_path),
            "--expected-commit",
            str(args.expected_commit).lower(),
            "--device",
            "cuda:0",
            "--seed",
            str(DYNAMIC_FLOOR_M2_SEED),
            "--with-diagnostic-telemetry",
        ],
        log_path=cell_root / "p0.out",
        env=env,
    )
    report = _read_json(report_path)
    validate_dynamic_stage1_p0_report(report)
    telemetry = report.get("diagnostic_telemetry_p0")
    if not isinstance(telemetry, Mapping):
        raise RuntimeError("dynamic floor M2 P0 did not execute telemetry eval")
    return {
        "path": str(report_path.resolve()),
        "file_sha256": sha256_file(report_path),
        "report_sha256": str(report["report_sha256"]),
        "status": str(report["status"]),
        "telemetry_status": str(telemetry.get("status")),
    }


def _build_result(
    *,
    args: argparse.Namespace,
    cfg: Any,
    config_paths: Mapping[str, Path],
    checkpoint: Path,
    prediction: Path,
    telemetry_path: Path,
    accuracy_log: Path,
    p0_receipt: Mapping[str, Any],
    storage_receipt: Mapping[str, Any],
    rendezvous: Mapping[str, Any],
) -> dict[str, Any]:
    spec = dynamic_floor_m2_arm_spec(args.arm)
    metrics = parse_official_style_map(
        accuracy_log.read_text(encoding="utf-8", errors="replace")
    )
    metrics["high_iou_composite"] = 0.5 * (
        float(metrics["mAP@0.6"]) + float(metrics["mAP@0.7"])
    )
    telemetry_summary = summarize_dynamic_floor_m2_telemetry(telemetry_path)
    binding = dict(cfg.georoute_dynamic_floor_m2_binding)
    checkpoint_sidecar = validate_dynamic_floor_m2_checkpoint_sidecar(
        checkpoint,
        binding=binding,
        cfg=cfg,
    )
    checkpoint_metadata = checkpoint_sidecar["experiment_metadata"]
    checkpoint_sidecar_path = Path(str(checkpoint) + ".metadata.json").resolve()
    result: dict[str, Any] = {
        "schema_version": DYNAMIC_FLOOR_M2_STAGE_RESULT_SCHEMA,
        "status": "PASS_DYNAMIC_FLOOR_M2_TRAINING_AND_ACCURACY",
        "study_id": DYNAMIC_FLOOR_M2_STUDY_ID,
        "arm": args.arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "epochs": DYNAMIC_FLOOR_M2_EPOCHS,
        "metrics": metrics,
        "telemetry_summary": telemetry_summary,
        "population_sha256": telemetry_summary["population_sha256"],
        "binding": binding,
        "binding_sha256": binding["binding_sha256"],
        "config_receipts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in config_paths.items()
        },
        "checkpoint_receipt": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "size_bytes": int(checkpoint.stat().st_size),
            "epoch": DYNAMIC_FLOOR_M2_EPOCHS - 1,
            "state_key": "state_dict_ema",
            "policy": "final_epoch_ema_only_atomic",
            "sidecar_path": str(checkpoint_sidecar_path),
            "sidecar_sha256": sha256_file(checkpoint_sidecar_path),
            "metadata_sha256": checkpoint_metadata["metadata_sha256"],
            "successful_updates": int(checkpoint_metadata["successful_updates"]),
        },
        "artifact_receipts": {
            "prediction": {
                "path": str(prediction.resolve()),
                "sha256": sha256_file(prediction),
            },
            "telemetry": {
                "path": str(telemetry_path.resolve()),
                "sha256": sha256_file(telemetry_path),
            },
            "accuracy_log": {
                "path": str(accuracy_log.resolve()),
                "sha256": sha256_file(accuracy_log),
            },
        },
        "p0_receipt": dict(p0_receipt),
        "storage_receipt": dict(storage_receipt),
        "runtime_commit": str(args.expected_commit).lower(),
        "rendezvous": _validate_rendezvous_receipt(
            rendezvous,
            stage="dynamic_floor_m2",
            variant=args.arm,
            seed=DYNAMIC_FLOOR_M2_SEED,
            nproc_per_node=1,
        ),
        "cost_attached": False,
        "single_seed_descriptive_only": True,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_used": False,
        "raw_prediction_cache_used": False,
        "paper_claim_allowed": False,
    }
    result["stage_result_sha256"] = canonical_sha256(result)
    validate_dynamic_floor_m2_stage_result(
        result,
        expected_arm=args.arm,
        expected_commit=str(args.expected_commit).lower(),
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--precheck-only", action="store_true")
    return parser.parse_args()


def _precheck(args: argparse.Namespace) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    _require_exact_clean_source(expected_commit)
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("dynamic floor M2 precheck run root leaves write boundary")
    cfg = bind_dynamic_floor_m2_config(
        source_config_path=args.source_config,
        arm=args.arm,
        seed=DYNAMIC_FLOOR_M2_SEED,
        work_dir=run_root / "precheck" / args.arm / "training" / "gpu1_id0",
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        runtime_commit=expected_commit,
    )
    train_cfg = copy.deepcopy(cfg)
    accuracy_cfg = copy.deepcopy(cfg)
    accuracy_cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = True
    accuracy_cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    train_binding = validate_dynamic_floor_m2_config(
        train_cfg, arm=args.arm, phase="train"
    )
    accuracy_binding = validate_dynamic_floor_m2_config(
        accuracy_cfg, arm=args.arm, phase="accuracy"
    )
    if dict(train_binding) != dict(accuracy_binding):
        raise RuntimeError("dynamic floor M2 precheck split the arm binding")
    receipt = {
        "schema_version": "scnr_dynamic_floor_m2_stage_precheck_v1",
        "status": "PASS_DYNAMIC_FLOOR_M2_STAGE_PRECHECK",
        "arm": args.arm,
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "runtime_commit": expected_commit,
        "binding_sha256": train_binding["binding_sha256"],
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    receipt["precheck_sha256"] = canonical_sha256(receipt)
    return receipt


def _execute(args: argparse.Namespace, *, cell_root: Path) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    _require_exact_clean_source(expected_commit)
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if not slurm_job_id.isdigit() or len(visible) != 1:
        raise RuntimeError("dynamic floor M2 stage requires one Slurm-visible GPU")
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("dynamic floor M2 run root leaves the write boundary")
    storage_receipt = storage_capacity_receipt(run_root, cell_count=1)
    cell_root.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(cell_root / "storage_preflight.json", storage_receipt)
    cfg, config_paths = _dump_configs(args=args, cell_root=cell_root)
    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    p0_receipt = _p0_receipt(
        args=args,
        config_path=config_paths["train"],
        cell_root=cell_root,
        env=inherited,
    )

    train_receipt = _run_torchrun(
        phase="train",
        stage="dynamic_floor_m2",
        arm=args.arm,
        slurm_job_id=slurm_job_id,
        arguments=(
            "tools/train.py",
            str(config_paths["train"]),
            "--seed",
            str(DYNAMIC_FLOOR_M2_SEED),
            "--id",
            "0",
        ),
        log_path=cell_root / "train.out",
        env=inherited,
    )
    checkpoint = (
        cell_root
        / "training"
        / "gpu1_id0"
        / "checkpoint"
        / f"epoch_{DYNAMIC_FLOOR_M2_EPOCHS - 1}.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"dynamic floor M2 checkpoint is missing: {checkpoint}")
    payloads = sorted(checkpoint.parent.glob("*.pth"))
    temporaries = sorted(checkpoint.parent.glob("*.tmp*"))
    if payloads != [checkpoint] or temporaries:
        raise RuntimeError("dynamic floor M2 requires exactly one final checkpoint")
    validate_dynamic_floor_m2_checkpoint_sidecar(
        checkpoint,
        binding=cfg.georoute_dynamic_floor_m2_binding,
        cfg=cfg,
    )

    accuracy_log = cell_root / "accuracy.out"
    accuracy_receipt = _run_torchrun(
        phase="test",
        stage="dynamic_floor_m2",
        arm=args.arm,
        slurm_job_id=slurm_job_id,
        arguments=(
            "tools/test.py",
            str(config_paths["accuracy"]),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(DYNAMIC_FLOOR_M2_SEED),
            "--id",
            "0",
        ),
        log_path=accuracy_log,
        env=inherited,
    )
    accuracy_root = cell_root / "accuracy" / "gpu1_id0"
    prediction = accuracy_root / "result_detection.json"
    telemetry = accuracy_root / "georoute_diagnostic_telemetry.json"
    for path in (prediction, telemetry):
        if not path.is_file():
            raise FileNotFoundError(path)
    return _build_result(
        args=args,
        cfg=cfg,
        config_paths=config_paths,
        checkpoint=checkpoint,
        prediction=prediction,
        telemetry_path=telemetry,
        accuracy_log=accuracy_log,
        p0_receipt=p0_receipt,
        storage_receipt=storage_receipt,
        rendezvous={"train": train_receipt, "test": accuracy_receipt},
    )


def main() -> int:
    args = _parse_args()
    dynamic_floor_m2_arm_spec(args.arm)
    if args.precheck_only:
        print(json.dumps(_precheck(args), sort_keys=True))
        return 0
    run_root = args.run_root.resolve()
    cell_root = run_root / dynamic_floor_m2_cell_relative_path(
        arm=args.arm,
        seed=DYNAMIC_FLOOR_M2_SEED,
    )
    if cell_root.exists():
        raise FileExistsError("dynamic floor M2 cell exists; refusing overwrite/resume")
    try:
        result = _execute(args, cell_root=cell_root)
    except Exception as error:
        if cell_root.is_dir():
            trace = traceback.format_exc()
            failure: dict[str, Any] = {
                "schema_version": DYNAMIC_FLOOR_M2_STAGE_RESULT_SCHEMA,
                "status": "FAIL_DYNAMIC_FLOOR_M2_ARM",
                "study_id": DYNAMIC_FLOOR_M2_STUDY_ID,
                "arm": args.arm,
                "seed": DYNAMIC_FLOOR_M2_SEED,
                "expected_runtime_commit": str(args.expected_commit).lower(),
                "observed_runtime_commit": _current_commit(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2000],
                "traceback_sha256": hashlib.sha256(
                    trace.encode("utf-8", errors="replace")
                ).hexdigest(),
                "official_test_opened": False,
                "paper_claim_allowed": False,
            }
            failure["failure_sha256"] = canonical_sha256(failure)
            _atomic_write_json(cell_root / "stage_failure.json", failure)
        raise
    _atomic_write_json(cell_root / "stage_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
