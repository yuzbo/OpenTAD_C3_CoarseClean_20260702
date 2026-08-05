#!/usr/bin/env python3
"""Run one fresh G1 residual-centering matched-training variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_dynamic_floor_m2_contract import (  # noqa: E402
    DYNAMIC_FLOOR_M2_EPOCHS,
    DYNAMIC_FLOOR_M2_SEED,
    summarize_dynamic_floor_m2_telemetry,
    validate_dynamic_floor_m2_checkpoint_sidecar,
)
from tools.bata.georoute_dynamic_floor_m2_stage_runner import (  # noqa: E402
    BOUNDARY,
    _inside,
    _p0_receipt,
    _require_exact_clean_source,
    _run_torchrun,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_residual_centering_training_contract import (  # noqa: E402
    RESIDUAL_CENTERING_ACCURACY_REPLAYS,
    RESIDUAL_CENTERING_BASE_ARM,
    RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES,
    RESIDUAL_CENTERING_TRAINING_PRECHECK_SCHEMA,
    RESIDUAL_CENTERING_TRAINING_STAGE_SCHEMA,
    RESIDUAL_CENTERING_TRAINING_STUDY_ID,
    bind_residual_centering_training_config,
    configure_residual_centering_accuracy,
    residual_centering_reachability_gate,
    residual_centering_route_payload_sha256,
    residual_centering_training_cell_relative_path,
    residual_centering_training_variant_spec,
    summarize_residual_centering_training_branch,
    validate_residual_centering_training_config,
    validate_residual_centering_training_stage_result,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _atomic_write_json,
    parse_official_style_map,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402
from tools.bata.run_georoute_role_instrumentation_pair import (  # noqa: E402
    compare_prediction_artifacts,
)


def _current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().lower()


def _file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _metrics(path: Path) -> dict[str, float]:
    values = {
        key: float(value)
        for key, value in parse_official_style_map(
            path.read_text(encoding="utf-8", errors="replace")
        ).items()
    }
    values["high_iou_composite"] = 0.5 * (
        values["mAP@0.6"] + values["mAP@0.7"]
    )
    return values


def _dump_configs(
    *, args: argparse.Namespace, cell_root: Path
) -> tuple[Any, dict[str, Path]]:
    config_root = args.run_root.resolve() / "control" / "bound_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    paths = {
        name: config_root
        / (
            f"{RESIDUAL_CENTERING_TRAINING_STUDY_ID}_{args.variant}_"
            f"seed{DYNAMIC_FLOOR_M2_SEED}_{name}.py"
        )
        for name in ("train", *RESIDUAL_CENTERING_ACCURACY_REPLAYS)
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(
            "residual-centering matched-training bound config namespace exists"
        )
    train_cfg = bind_residual_centering_training_config(
        source_config_path=args.source_config,
        variant=args.variant,
        seed=DYNAMIC_FLOOR_M2_SEED,
        work_dir=cell_root / "training" / "gpu1_id0",
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        runtime_commit=str(args.expected_commit).lower(),
    )
    training_config_sha = canonical_sha256(train_cfg.to_dict())
    configs = {"train": train_cfg}
    for replay in RESIDUAL_CENTERING_ACCURACY_REPLAYS:
        configs[replay] = configure_residual_centering_accuracy(
            train_cfg,
            variant=args.variant,
            replay=replay,
            work_dir=cell_root / "accuracy" / replay / "gpu1_id0",
            training_config_sha256=training_config_sha,
        )
    for name, cfg in configs.items():
        validate_residual_centering_training_config(
            cfg,
            variant=args.variant,
            phase="train" if name == "train" else "accuracy",
        )
        cfg.dump(str(paths[name]))
    return train_cfg, paths


def _precheck(args: argparse.Namespace) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    _require_exact_clean_source(expected_commit)
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("residual-centering precheck run root leaves write boundary")
    cfg = bind_residual_centering_training_config(
        source_config_path=args.source_config,
        variant=args.variant,
        seed=DYNAMIC_FLOOR_M2_SEED,
        work_dir=run_root
        / "precheck"
        / args.variant
        / "training"
        / "gpu1_id0",
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        runtime_commit=expected_commit,
    )
    binding = validate_residual_centering_training_config(
        cfg, variant=args.variant, phase="train"
    )
    for replay in RESIDUAL_CENTERING_ACCURACY_REPLAYS:
        accuracy = configure_residual_centering_accuracy(
            cfg,
            variant=args.variant,
            replay=replay,
            work_dir=run_root / "precheck" / args.variant / replay / "gpu1_id0",
            training_config_sha256=canonical_sha256(cfg.to_dict()),
        )
        observed = validate_residual_centering_training_config(
            accuracy, variant=args.variant, phase="accuracy"
        )
        if dict(observed) != dict(binding):
            raise RuntimeError("residual-centering precheck split the study binding")
    receipt: dict[str, Any] = {
        "schema_version": RESIDUAL_CENTERING_TRAINING_PRECHECK_SCHEMA,
        "status": "PASS_RESIDUAL_CENTERING_MATCHED_TRAINING_PRECHECK",
        "study_id": RESIDUAL_CENTERING_TRAINING_STUDY_ID,
        "variant": args.variant,
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "runtime_commit": expected_commit,
        "binding_sha256": binding["binding_sha256"],
        "shared_protocol_sha256": binding["shared_protocol_sha256"],
        "expected_successful_updates": (
            RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES
        ),
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    receipt["precheck_sha256"] = canonical_sha256(receipt)
    return receipt


def _run_accuracy(
    *,
    args: argparse.Namespace,
    replay: str,
    config_path: Path,
    checkpoint: Path,
    cell_root: Path,
    slurm_job_id: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    log_path = cell_root / "accuracy" / replay / "test.out"
    rendezvous = _run_torchrun(
        phase="test",
        stage=f"residual_centering_{replay}",
        arm=args.variant,
        slurm_job_id=slurm_job_id,
        arguments=(
            "tools/test.py",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(DYNAMIC_FLOOR_M2_SEED),
            "--id",
            "0",
        ),
        log_path=log_path,
        env=env,
    )
    output_root = cell_root / "accuracy" / replay / "gpu1_id0"
    prediction = output_root / "result_detection.json"
    telemetry = output_root / "georoute_diagnostic_telemetry.json"
    profile = output_root / "georoute_development_profile.json"
    for path in (prediction, telemetry, log_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if profile.exists():
        raise RuntimeError(
            "residual-centering accuracy unexpectedly emitted a timed profile"
        )
    spec = residual_centering_training_variant_spec(args.variant)
    telemetry_summary = summarize_dynamic_floor_m2_telemetry(telemetry)
    branch_summary = summarize_residual_centering_training_branch(
        telemetry,
        expected_mode=spec["branch_calibration_mode"],
    )
    return {
        "metrics": _metrics(log_path),
        "telemetry_summary": telemetry_summary,
        "branch_summary": branch_summary,
        "population_sha256": telemetry_summary["population_sha256"],
        "route_payload_sha256": residual_centering_route_payload_sha256(
            telemetry
        ),
        "artifacts": {
            "prediction": _file_receipt(prediction),
            "telemetry": _file_receipt(telemetry),
            "accuracy_log": _file_receipt(log_path),
        },
        "rendezvous": rendezvous,
    }


def _execute(args: argparse.Namespace, *, cell_root: Path) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    _require_exact_clean_source(expected_commit)
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = [
        item
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item
    ]
    if not slurm_job_id.isdigit() or len(visible) != 1:
        raise RuntimeError(
            "residual-centering stage requires one Slurm-visible GPU"
        )
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("residual-centering run root leaves write boundary")
    storage_receipt = storage_capacity_receipt(run_root, cell_count=1)
    cell_root.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(cell_root / "storage_preflight.json", storage_receipt)
    train_cfg, config_paths = _dump_configs(args=args, cell_root=cell_root)
    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"

    p0_args = argparse.Namespace(**vars(args), arm=RESIDUAL_CENTERING_BASE_ARM)
    p0 = _p0_receipt(
        args=p0_args,
        config_path=config_paths["train"],
        cell_root=cell_root,
        env=inherited,
    )
    train_rendezvous = _run_torchrun(
        phase="train",
        stage="residual_centering_matched_train",
        arm=args.variant,
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
        raise FileNotFoundError(f"residual-centering checkpoint is missing: {checkpoint}")
    payloads = sorted(checkpoint.parent.glob("*.pth"))
    temporaries = sorted(checkpoint.parent.glob("*.tmp*"))
    if payloads != [checkpoint] or temporaries:
        raise RuntimeError(
            "residual-centering matched training requires exactly one final checkpoint"
        )
    sidecar = validate_dynamic_floor_m2_checkpoint_sidecar(
        checkpoint,
        binding=train_cfg.georoute_dynamic_floor_m2_binding,
        cfg=train_cfg,
    )
    metadata = sidecar["experiment_metadata"]
    if (
        int(metadata.get("train_batches_per_epoch", -1)) != 160
        or int(metadata.get("successful_updates", -1))
        != RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES
    ):
        raise RuntimeError(
            "residual-centering checkpoint did not complete the frozen 9600 updates"
        )

    replays = {
        replay: _run_accuracy(
            args=args,
            replay=replay,
            config_path=config_paths[replay],
            checkpoint=checkpoint,
            cell_root=cell_root,
            slurm_job_id=slurm_job_id,
            env=inherited,
        )
        for replay in RESIDUAL_CENTERING_ACCURACY_REPLAYS
    }
    a = replays["accuracy_a"]
    b = replays["accuracy_b"]
    prediction_comparison = compare_prediction_artifacts(
        Path(a["artifacts"]["prediction"]["path"]),
        Path(b["artifacts"]["prediction"]["path"]),
    )
    duplicate = {
        "prediction": prediction_comparison,
        "route_payload_sha256_parity": (
            a["route_payload_sha256"] == b["route_payload_sha256"]
        ),
        "metrics_parity": a["metrics"] == b["metrics"],
        "branch_summary_parity": a["branch_summary"] == b["branch_summary"],
        "population_sha256_parity": (
            a["population_sha256"] == b["population_sha256"]
        ),
    }
    if not all(
        (
            prediction_comparison["raw_sha256_parity"],
            prediction_comparison["json_semantic_parity"],
            duplicate["route_payload_sha256_parity"],
            duplicate["metrics_parity"],
            duplicate["branch_summary_parity"],
            duplicate["population_sha256_parity"],
        )
    ):
        raise RuntimeError(
            "residual-centering strict duplicate accuracy replay is not exact"
        )
    reachability = (
        residual_centering_reachability_gate(a["telemetry_summary"])
        if args.variant == "residual_window_center"
        else None
    )
    if reachability is not None and not reachability["passed"]:
        raise RuntimeError(
            "residual-centering treatment did not preserve selected context/ROI reachability"
        )
    sidecar_path = Path(str(checkpoint) + ".metadata.json").resolve()
    binding = dict(train_cfg.georoute_residual_centering_training_binding)
    spec = residual_centering_training_variant_spec(args.variant)
    result: dict[str, Any] = {
        "schema_version": RESIDUAL_CENTERING_TRAINING_STAGE_SCHEMA,
        "status": (
            "PASS_RESIDUAL_CENTERING_MATCHED_TRAINING_AND_DUPLICATE_ACCURACY"
        ),
        "study_id": RESIDUAL_CENTERING_TRAINING_STUDY_ID,
        "variant": args.variant,
        "variant_spec": spec,
        "variant_spec_sha256": canonical_sha256(spec),
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "epochs": DYNAMIC_FLOOR_M2_EPOCHS,
        "expected_successful_updates": (
            RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES
        ),
        "runtime_commit": expected_commit,
        "slurm_job_id": slurm_job_id,
        "binding": binding,
        "binding_sha256": binding["binding_sha256"],
        "config_receipts": {
            name: _file_receipt(path) for name, path in config_paths.items()
        },
        "checkpoint_receipt": {
            **_file_receipt(checkpoint),
            "size_bytes": int(checkpoint.stat().st_size),
            "epoch": DYNAMIC_FLOOR_M2_EPOCHS - 1,
            "state_key": "state_dict_ema",
            "policy": "final_epoch_ema_only_atomic",
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "metadata_sha256": metadata["metadata_sha256"],
            "successful_updates": int(metadata["successful_updates"]),
        },
        "accuracy_replays": replays,
        "duplicate_integrity": duplicate,
        "reachability_gate": reachability,
        "p0_receipt": p0,
        "storage_receipt": storage_receipt,
        "rendezvous": {
            "train": train_rendezvous,
            "accuracy_a": replays["accuracy_a"]["rendezvous"],
            "accuracy_b": replays["accuracy_b"]["rendezvous"],
        },
        "fresh_training": True,
        "old_g1_checkpoint_reused": False,
        "cost_attached": False,
        "additional_seeds_opened": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    result["stage_result_sha256"] = canonical_sha256(result)
    validate_residual_centering_training_stage_result(
        result,
        expected_variant=args.variant,
        expected_commit=expected_commit,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
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


def main() -> int:
    args = _parse_args()
    residual_centering_training_variant_spec(args.variant)
    if args.precheck_only:
        print(json.dumps(_precheck(args), sort_keys=True))
        return 0
    run_root = args.run_root.resolve()
    cell_root = run_root / residual_centering_training_cell_relative_path(
        variant=args.variant
    )
    if cell_root.exists():
        raise FileExistsError(
            "residual-centering cell exists; refusing overwrite or resume"
        )
    try:
        result = _execute(args, cell_root=cell_root)
    except Exception as error:
        if cell_root.is_dir():
            trace = traceback.format_exc()
            failure: dict[str, Any] = {
                "schema_version": RESIDUAL_CENTERING_TRAINING_STAGE_SCHEMA,
                "status": "FAIL_RESIDUAL_CENTERING_MATCHED_TRAINING_VARIANT",
                "study_id": RESIDUAL_CENTERING_TRAINING_STUDY_ID,
                "variant": args.variant,
                "seed": DYNAMIC_FLOOR_M2_SEED,
                "expected_runtime_commit": str(args.expected_commit).lower(),
                "observed_runtime_commit": _current_commit(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2000],
                "traceback_sha256": hashlib.sha256(
                    trace.encode("utf-8", errors="replace")
                ).hexdigest(),
                "paired_cost_authorized": False,
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
