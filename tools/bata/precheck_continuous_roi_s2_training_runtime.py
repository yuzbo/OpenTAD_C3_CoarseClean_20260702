from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_contract import (  # noqa: E402
    canonical_sha256,
)
from tools.bata.continuous_roi_s2_training import (  # noqa: E402
    S2_FAMILIES,
    S2_SOURCE_CONFIGS,
    S2_TRAINING_RUNTIME_PRECHECK_SCHEMA,
    S2_TRAINING_SEEDS,
    S2_UPDATES_PER_EPOCH,
    audit_development_video_census,
    audit_development_video_inventory,
    bind_training_config,
    build_experiment_identity,
    current_git_commit,
    require_clean_git_checkout,
    validate_external_training_inputs,
    validate_full_model_gate,
)


def _publish_once(path: Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("runtime sample value has no shape")
    return [int(item) for item in shape]


def _dtype(value: Any) -> str:
    return str(getattr(value, "dtype", ""))


def _audit_sample(family: str, sample: dict[str, Any], collated: dict[str, Any]):
    family = family.upper()
    expected_size = {"D160": 160, "G96": 96}.get(family)
    geometry = sample.get("metas", {}).get("continuous_roi_geometry")
    if not isinstance(geometry, dict):
        raise RuntimeError(f"{family} sample has no continuous ROI geometry audit")
    if (
        geometry.get("uses_gt") is not False
        or geometry.get("uses_teacher") is not False
        or geometry.get("uses_oracle") is not False
        or geometry.get("uses_test_evidence") is not False
        or geometry.get("source_resized_before_crop") is not False
    ):
        raise RuntimeError(f"{family} sample violates the no-leak geometry contract")

    sample_inputs = sample.get("inputs")
    collated_inputs = collated.get("inputs")
    if expected_size is not None:
        expected_sample_shape = [1, 3, 768, expected_size, expected_size]
        expected_collated_shape = [1, *expected_sample_shape]
        if (
            _shape(sample_inputs) != expected_sample_shape
            or _shape(collated_inputs) != expected_collated_shape
            or "uint8" not in _dtype(sample_inputs)
            or "uint8" not in _dtype(collated_inputs)
            or geometry.get("policy") != "full_frame_letterbox"
            or geometry.get("crop_applied") is not False
        ):
            raise RuntimeError(f"{family} runtime sample shape or policy changed")
        input_shapes = {"dense": expected_sample_shape}
        collated_shapes = {"dense": expected_collated_shape}
    else:
        if not isinstance(sample_inputs, dict) or not isinstance(
            collated_inputs, dict
        ):
            raise RuntimeError("U128 runtime did not preserve structured inputs")
        expected_sample_shapes = {
            "global": [1, 3, 768, 96, 96],
            "source": [1, 3, 768, 180, 320],
        }
        expected_collated_shapes = {
            "global": [1, 1, 3, 768, 96, 96],
            "source": [1, 1, 3, 768, 180, 320],
            "sample_key": [1],
            "window_start": [1],
        }
        for key, expected in expected_sample_shapes.items():
            if (
                _shape(sample_inputs.get(key)) != expected
                or "uint8" not in _dtype(sample_inputs.get(key))
            ):
                raise RuntimeError(f"U128 sample {key} contract changed")
        for key, expected in expected_collated_shapes.items():
            if _shape(collated_inputs.get(key)) != expected:
                raise RuntimeError(f"U128 collated {key} contract changed")
        if (
            geometry.get("policy") != "none_pre_policy_source"
            or geometry.get("decision_inputs") != []
            or geometry.get("source_float_video_materialized") is not False
        ):
            raise RuntimeError("U128 runtime unexpectedly contains a crop policy")
        input_shapes = {
            key: _shape(sample_inputs[key]) for key in sorted(expected_sample_shapes)
        }
        collated_shapes = {
            key: _shape(collated_inputs[key])
            for key in sorted(expected_collated_shapes)
        }

    if _shape(collated.get("masks")) != [1, 768]:
        raise RuntimeError(f"{family} collated temporal mask contract changed")
    return {
        "video_name": str(sample["metas"]["video_name"]),
        "input_shapes": input_shapes,
        "collated_input_shapes": collated_shapes,
        "collated_mask_shape": [1, 768],
        "geometry_policy": geometry["policy"],
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_test_evidence": False,
    }


def _audit_real_training_batch(family: str, batch: dict[str, Any]):
    family = family.upper()
    inputs = batch.get("inputs")
    expected_size = {"D160": 160, "G96": 96}.get(family)
    if expected_size is not None:
        expected = [2, 1, 3, 768, expected_size, expected_size]
        if _shape(inputs) != expected or "uint8" not in _dtype(inputs):
            raise RuntimeError(f"{family} real batch shape or dtype changed")
        input_shapes = {"dense": expected}
    else:
        if not isinstance(inputs, dict):
            raise RuntimeError("U128 real batch lost its structured inputs")
        expected_shapes = {
            "global": [2, 1, 3, 768, 96, 96],
            "source": [2, 1, 3, 768, 180, 320],
            "sample_key": [2],
            "window_start": [2],
        }
        for key, expected in expected_shapes.items():
            if _shape(inputs.get(key)) != expected:
                raise RuntimeError(f"U128 real batch {key} shape changed")
        if any(
            token in str(key).lower()
            for key in inputs
            for token in ("teacher", "oracle", "test", "gt")
        ):
            raise RuntimeError("U128 decision inputs contain a privileged field")
        input_shapes = expected_shapes
    if _shape(batch.get("masks")) != [2, 768]:
        raise RuntimeError(f"{family} real batch temporal mask changed")
    return {
        "batch_size": 2,
        "input_shapes": input_shapes,
        "mask_shape": [2, 768],
        "uses_gt_for_geometry": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_test_evidence": False,
    }


def _gate_window_identity(dataset) -> list[dict[str, int | str]]:
    records = []
    for row in dataset.data_list:
        centers = row[3]
        records.append(
            {
                "video_id": str(row[0]),
                "first_frame": int(centers[0]),
                "last_frame": int(centers[-1]),
                "frame_count": int(len(centers)),
            }
        )
    return records


def run_precheck(
    *,
    expected_commit: str,
    manifest_path: Path,
    development_annotation_path: Path,
    class_map_path: Path,
    development_video_root: Path,
    pretrained_checkpoint_path: Path,
    full_model_gate_path: Path,
) -> dict[str, Any]:
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.datasets.builder import collate
    from tools.bata.spatial_zoom_s1_training import (
        require_slurm_memory_limit_mb,
        require_slurm_single_gpu_allocation,
    )

    expected_commit = str(expected_commit).lower()
    if current_git_commit(ROOT) != expected_commit:
        raise RuntimeError("training runtime precheck commit mismatch")
    require_clean_git_checkout(
        expected_commit=expected_commit, repository_root=ROOT
    )
    slurm_gpu_identity = require_slurm_single_gpu_allocation()
    effective_memory_limit_mb = require_slurm_memory_limit_mb(
        minimum_mb=90000
    )
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 5:
        raise RuntimeError("runtime precheck requires the exact five-CPU step")
    gate = validate_full_model_gate(
        full_model_gate_path, expected_commit=expected_commit
    )
    external = validate_external_training_inputs(
        manifest_path=manifest_path,
        development_annotation_path=development_annotation_path,
        class_map_path=class_map_path,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
    )
    identity = build_experiment_identity(
        code_commit=expected_commit,
        gate_path=full_model_gate_path,
        gate=gate,
        external_inputs=external,
    )
    manifest = external["manifest"]
    development_ids = list(manifest["splits"]["fit"]) + list(
        manifest["splits"]["gate"]
    )
    video_inventory = audit_development_video_inventory(
        development_video_root.resolve(), development_ids
    )
    video_census = audit_development_video_census(
        development_video_root.resolve(), development_ids
    )
    if video_inventory["video_count"] != 200:
        raise RuntimeError("Continuous-RoI S2 development inventory is not 200 videos")

    bindings = []
    family_runtime = {}
    gate_window_counts = set()
    gate_window_hashes = set()
    for family in S2_FAMILIES:
        family_configs = {}
        for seed in S2_TRAINING_SEEDS:
            work_dir = (
                Path(identity["canonical_experiment_root"])
                / family.lower()
                / f"seed{seed}"
            )
            cfg = bind_training_config(
                source_config_path=ROOT / S2_SOURCE_CONFIGS[family],
                family=family,
                seed=seed,
                work_dir=work_dir,
                manifest_path=manifest_path,
                development_annotation_path=development_annotation_path,
                class_map_path=class_map_path,
                development_video_root=development_video_root,
                pretrained_checkpoint_path=pretrained_checkpoint_path,
                full_model_gate_path=full_model_gate_path,
                repository_root=ROOT,
                code_commit=expected_commit,
                require_runtime_precheck=False,
                require_runtime_authorization=False,
            )
            family_configs[seed] = cfg
            with tempfile.TemporaryDirectory(
                prefix=f"continuous-roi-s2-{family.lower()}-{seed}-"
            ) as temporary_directory:
                roundtrip_path = (
                    Path(temporary_directory) / "bound_config.py"
                )
                cfg.dump(str(roundtrip_path))
                reloaded = Config.fromfile(str(roundtrip_path))
                if canonical_sha256(reloaded.to_dict()) != canonical_sha256(
                    cfg.to_dict()
                ):
                    raise RuntimeError(
                        f"{family}/{seed} config dump/reload changed the binding"
                    )
            binding = cfg.continuous_roi_s2_runtime_binding
            bindings.append(
                {
                    "family": family,
                    "seed": seed,
                    "work_dir": binding["work_dir"],
                    "bound_config_sha256": canonical_sha256(cfg.to_dict()),
                    "successful_updates": binding["successful_updates"],
                    "updates_per_epoch": binding["updates_per_epoch"],
                    "checkpoint_selection": binding["checkpoint_selection"],
                    "config_dump_reload_valid": True,
                }
            )

        cfg = family_configs[S2_TRAINING_SEEDS[0]]
        train_dataset = build_dataset(cfg.dataset.train)
        val_dataset = build_dataset(cfg.dataset.val)
        development_gate_dataset = build_dataset(cfg.dataset.test)
        train_loader = build_dataloader(
            train_dataset,
            rank=0,
            world_size=1,
            shuffle=True,
            drop_last=True,
            **cfg.solver.train,
        )
        train_ids = {str(row[0]) for row in train_dataset.data_list}
        val_ids = {str(row[0]) for row in val_dataset.data_list}
        test_ids = {str(row[0]) for row in development_gate_dataset.data_list}
        if (
            train_ids != set(manifest["splits"]["fit"])
            or val_ids != set(manifest["splits"]["gate"])
            or test_ids != set(manifest["splits"]["gate"])
        ):
            raise RuntimeError(f"{family} runtime dataset split differs from protocol")
        if len(train_dataset) != 160 or len(train_loader) != S2_UPDATES_PER_EPOCH:
            raise RuntimeError(f"{family} runtime does not yield 80 updates per epoch")
        if len(val_dataset) != len(development_gate_dataset):
            raise RuntimeError(f"{family} development Gate loader populations differ")
        if len(val_dataset) != 129:
            raise RuntimeError(
                f"{family} development Gate must contain exactly 129 windows"
            )
        window_identity = _gate_window_identity(val_dataset)
        test_window_identity = _gate_window_identity(development_gate_dataset)
        if window_identity != test_window_identity:
            raise RuntimeError(
                f"{family} val/test development Gate windows differ"
            )
        gate_window_counts.add(len(val_dataset))
        gate_window_hashes.add(canonical_sha256(window_identity))
        sample = train_dataset[0]
        sample_audit = _audit_sample(family, sample, collate([sample]))
        real_training_batch = _audit_real_training_batch(
            family, next(iter(train_loader))
        )
        family_runtime[family] = {
            "fit_video_count": len(train_ids),
            "fit_sample_count": len(train_dataset),
            "development_gate_video_count": len(val_ids),
            "development_gate_window_count": len(val_dataset),
            "train_batches_per_epoch": len(train_loader),
            "gate_window_identity_sha256": canonical_sha256(window_identity),
            "sample_audit": sample_audit,
            "real_training_batch_audit": real_training_batch,
        }

    if (
        len(bindings) != 9
        or len(gate_window_counts) != 1
        or len(gate_window_hashes) != 1
    ):
        raise RuntimeError("Continuous-RoI S2 nine-cell runtime matrix is inconsistent")
    report = {
        "schema_version": S2_TRAINING_RUNTIME_PRECHECK_SCHEMA,
        "status": "PASS",
        "code_commit": expected_commit,
        "full_model_gate_path": str(full_model_gate_path.resolve()),
        "full_model_gate_file_sha256": identity[
            "full_model_gate_file_sha256"
        ],
        "full_model_gate_sha256": gate["gate_sha256"],
        "protocol_sha256": external["protocol"]["declared_protocol_sha256"],
        "experiment_namespace": identity["experiment_namespace"],
        "canonical_experiment_root": identity["canonical_experiment_root"],
        "development_video_inventory": video_inventory,
        "development_video_census": video_census,
        "development_gate_window_count": 129,
        "development_gate_window_identity_sha256": next(
            iter(gate_window_hashes)
        ),
        "all_nine_bindings_valid": True,
        "all_nine_config_dump_reload_valid": True,
        "bindings": bindings,
        "family_runtime": family_runtime,
        "train_batches_per_epoch": S2_UPDATES_PER_EPOCH,
        "official_test_annotation_records_loaded": 0,
        "official_test_video_files_opened": 0,
        "official_test_open_allowed": False,
        "learned_roi_policy_present": False,
        "paper_claim_allowed": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "slurm_step_gpu_identity": slurm_gpu_identity,
        "slurm_cpus_per_task": 5,
        "effective_memory_limit_mb": effective_memory_limit_mb,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    report["precheck_sha256"] = canonical_sha256(report)
    return report


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Precheck all nine Continuous-RoI S2 training bindings and real "
            "development data pipelines."
        )
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--full-model-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(
                "refusing to overwrite a Continuous-RoI S2 runtime precheck"
            )
        report = run_precheck(
            expected_commit=args.expected_commit,
            manifest_path=args.manifest.resolve(),
            development_annotation_path=args.development_annotation.resolve(),
            class_map_path=args.class_map.resolve(),
            development_video_root=args.development_video_root.resolve(),
            pretrained_checkpoint_path=args.pretrained.resolve(),
            full_model_gate_path=args.full_model_gate.resolve(),
        )
        _publish_once(args.output, report)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "precheck_sha256": report["precheck_sha256"],
                "experiment_namespace": report["experiment_namespace"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
