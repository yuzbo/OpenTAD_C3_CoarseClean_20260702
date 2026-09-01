#!/usr/bin/env python3
"""Fail-closed CPU preflight for frozen PhysTime decode cross replay."""

import argparse
import json
import os
from pathlib import Path

from tools.bata.run_phystime_decode_cross_gate import (
    P0_DATASET_MANIFEST_SHA256,
    P0_VIDEOMAE_SHA256,
    SOURCE_COMMIT,
    SOURCE_TREE,
    build_dataset_manifest,
    git_identity,
    read_json,
    require,
    sha256_file,
    validate_configs,
    validate_config_semantics_against_p0,
    validate_p0,
    validate_source_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute every immutable data/checkpoint/provenance binding "
            "before the first decode-cross sbatch submission."
        )
    )
    parser.add_argument("--selected-config", required=True)
    parser.add_argument("--physical-config", required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--physical-checkpoint", required=True)
    parser.add_argument("--videomae-checkpoint", required=True)
    parser.add_argument("--selected-source-dir", required=True)
    parser.add_argument("--physical-source-dir", required=True)
    parser.add_argument("--p0-run-root", required=True)
    parser.add_argument("--expected-runtime-commit", required=True)
    parser.add_argument("--expected-runtime-tree", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def checkpoint_file_report(path):
    path = Path(path).resolve()
    require(path.is_file(), f"checkpoint is missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def source_completion_report(path):
    path = Path(path).resolve()
    completion_path = path / "FULL_COMPLETE.json"
    manifest_path = path / "run_manifest.json"
    completion = read_json(completion_path, "source completion")
    manifest = read_json(manifest_path, "source manifest")
    return {
        "path": str(path),
        "completion": {
            "path": str(completion_path),
            "sha256": sha256_file(completion_path),
            "validation_pass": completion.get("validation_pass"),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "commit": manifest.get("commit"),
            "git_tree": manifest.get("git_tree"),
            "variant": manifest.get("variant"),
        },
    }


def main():
    args = parse_args()
    runtime_commit, runtime_tree, clean = git_identity()
    require(
        runtime_commit == args.expected_runtime_commit
        and runtime_tree == args.expected_runtime_tree
        and clean,
        "preflight runtime snapshot mismatch or dirty tree",
    )
    require(
        SOURCE_COMMIT == "0dc5851a8feb12b97d16bdb5ea8fc60e9273d132"
        and SOURCE_TREE == "bddc9b9386604d00d213275a47ce7997b35d3f4c",
        "reviewed full60 source constants changed",
    )

    selected_cfg, _, config_reports = validate_configs(
        args.selected_config,
        args.physical_config,
    )
    dataset_manifest, dataset_manifest_sha256 = build_dataset_manifest(
        selected_cfg,
        selected_cfg.evaluation.ground_truth_filename,
    )
    require(
        dataset_manifest_sha256 == P0_DATASET_MANIFEST_SHA256,
        "current THUMOS content differs from the reviewed P0 dataset",
    )

    videomae_path = Path(args.videomae_checkpoint).resolve()
    require(videomae_path.is_file(), "VideoMAE checkpoint is missing")
    videomae_sha256 = sha256_file(videomae_path)
    require(
        videomae_sha256 == P0_VIDEOMAE_SHA256,
        "current VideoMAE checkpoint differs from the reviewed artifact",
    )

    checkpoint_reports = {
        "selected_axis": checkpoint_file_report(args.selected_checkpoint),
        "physical_metric": checkpoint_file_report(args.physical_checkpoint),
    }
    source_reports = {
        "selected_axis": validate_source_dir(
            args.selected_source_dir,
            "selected_axis",
            checkpoint_reports["selected_axis"]["sha256"],
            dataset_manifest_sha256,
            videomae_sha256,
        ),
        "physical_metric": validate_source_dir(
            args.physical_source_dir,
            "physical_metric",
            checkpoint_reports["physical_metric"]["sha256"],
            dataset_manifest_sha256,
            videomae_sha256,
        ),
    }
    require(
        source_reports["selected_axis"]["source_gate"]
        == source_reports["physical_metric"]["source_gate"],
        "source arms do not bind the same gate path and SHA",
    )
    p0_report = validate_p0(args.p0_run_root, checkpoint_reports)
    validate_config_semantics_against_p0(config_reports, p0_report)

    payload = {
        "schema_version": "phystime_decode_cross_preflight_v1",
        "validation_pass": True,
        "runtime": {
            "commit": runtime_commit,
            "git_tree": runtime_tree,
            "clean": clean,
        },
        "source": {
            "commit": SOURCE_COMMIT,
            "git_tree": SOURCE_TREE,
            "arms": source_reports,
            "completion_artifacts": {
                "selected_axis": source_completion_report(
                    args.selected_source_dir
                ),
                "physical_metric": source_completion_report(
                    args.physical_source_dir
                ),
            },
        },
        "configs": {
            name: {
                **report,
                "path": str(
                    Path(
                        args.selected_config
                        if name == "selected_axis"
                        else args.physical_config
                    ).resolve()
                ),
                "sha256": sha256_file(
                    args.selected_config
                    if name == "selected_axis"
                    else args.physical_config
                ),
            }
            for name, report in config_reports.items()
        },
        "dataset": {
            "manifest": dataset_manifest,
            "manifest_sha256": dataset_manifest_sha256,
        },
        "videomae_checkpoint": {
            "path": str(videomae_path),
            "sha256": videomae_sha256,
            "size_bytes": int(videomae_path.stat().st_size),
        },
        "checkpoints": checkpoint_reports,
        "p0": p0_report,
        "evidence_inputs_validated": True,
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "validation_pass": True,
                "output": str(Path(args.output).resolve()),
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "videomae_checkpoint_sha256": videomae_sha256,
                "selected_checkpoint_sha256": checkpoint_reports[
                    "selected_axis"
                ]["sha256"],
                "physical_checkpoint_sha256": checkpoint_reports[
                    "physical_metric"
                ]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
