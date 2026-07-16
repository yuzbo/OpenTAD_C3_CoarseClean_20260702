#!/usr/bin/env python3
"""Freeze the official THUMOS full-video/sliding-window Gate-4 population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config

from opentad.evaluations.builder import remove_duplicate_annotations
from opentad.models.chronotransport.filesystem import (
    load_bound_json,
    open_bound_directory,
    publish_bytes_exclusive,
    read_bound_bytes,
    secure_lexical_path,
)
from opentad.models.chronotransport.gate4_population import (
    GATE4_CONFIG_RELATIVE,
    build_gate4_population_artifact,
    gate4_population_exact_bytes,
)
from opentad.models.chronotransport.protocol import canonical_sha256, validate_r2_manifest


_CONFIG_SOURCES = (
    "configs/adatad/thumos/c3_chronotransport_r2_stage_c.py",
    "configs/adatad/thumos/c3_chronotransport_r2_stage_b.py",
    "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_b.py",
    "configs/adatad/thumos/c3_chronotransport_adatad_videomae_s_768x1_160_stage_a.py",
    "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py",
    "configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",
    "configs/_base_/models/actionformer.py",
)
_CONFIG_OVERRIDE_ENV = {
    "CHRONOTRANSPORT_MODE",
    "CHRONOTRANSPORT_COST_JSON",
    "CHRONOTRANSPORT_SCHEDULE_COST_JSON",
    "CHRONOTRANSPORT_RISK_READY",
    "CHRONOTRANSPORT_ALLOW_UNMEASURED_DEBUG",
    "CHRONOTRANSPORT_MAX_CACHE_AGE",
    "CHRONOTRANSPORT_RISK_QUANTILE",
    "CHRONOTRANSPORT_RISK_EPSILON",
    "CHRONOTRANSPORT_PROFILE_SYNC_CUDA",
    "CHRONOTRANSPORT_COST_HARDWARE",
    "CHRONOTRANSPORT_COST_PRECISION",
    "CHRONOTRANSPORT_COST_STATISTIC",
}


def _resolve_repo_input(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return secure_lexical_path(path, label=label)


def _pipeline_scale_factor(pipeline: Any) -> int:
    load_frames = [
        item
        for item in pipeline
        if isinstance(item, Mapping) and str(item.get("type")) == "LoadFrames"
    ]
    if len(load_frames) != 1 or str(load_frames[0].get("method")) != "sliding_window":
        raise ValueError("Gate4 pipeline requires one sliding-window LoadFrames")
    return int(load_frames[0].get("scale_factor", 1))


def _fit_duration_quartiles(
    annotation: Mapping[str, Any], manifest: Mapping[str, Any], class_map: set[str]
) -> list[float]:
    windows = {str(row["window_id"]): row for row in manifest["windows"]}
    fit_videos = {
        str(windows[window_id]["video_id"])
        for window_id in manifest["splits"]["fit"]
    }
    durations = []
    database = annotation["database"]
    for video in sorted(fit_videos):
        info = database.get(video)
        if not isinstance(info, Mapping):
            raise ValueError("Gate4 fit video is absent from official annotation")
        for row in remove_duplicate_annotations(info.get("annotations", [])):
            label = str(row.get("label"))
            segment = row.get("segment")
            if label not in class_map or not isinstance(segment, list) or len(segment) != 2:
                continue
            duration = float(segment[1]) - float(segment[0])
            if duration > 0.0 and np.isfinite(duration):
                durations.append(duration)
    if len(durations) < 4:
        raise ValueError("Gate4 fit population has too few action durations")
    return [float(value) for value in np.percentile(durations, [25.0, 50.0, 75.0])]


def build_gate4_population_from_context(
    *,
    repository_root: Path,
    data_root: Path,
    fit_manifest_path: Path,
) -> dict[str, Any]:
    if secure_lexical_path(repository_root, label="Gate4 repository root") != secure_lexical_path(
        ROOT, label="fixed Gate4 repository root"
    ):
        raise ValueError("Gate4 population builder is bound to its repository root")
    overrides = sorted(name for name in _CONFIG_OVERRIDE_ENV if name in os.environ)
    if overrides:
        raise RuntimeError(f"Gate4 population forbids config overrides: {overrides}")
    config_path = ROOT / GATE4_CONFIG_RELATIVE
    cfg = Config.fromfile(str(config_path))
    test = cfg.dataset.test
    val = cfg.dataset.val
    if (
        str(test.type) != "ThumosSlidingDataset"
        or str(test.subset_name) != "validation"
        or bool(test.test_mode) is not True
        or test.block_list is not None
        or int(test.feature_stride) != 4
        or int(test.sample_stride) != 1
        or int(getattr(test, "offset_frames", 0)) != 0
        or int(test.window_size) != 768
        or float(test.window_overlap_ratio) != 0.5
        or str(cfg.evaluation.subset) != "validation"
        or bool(cfg.inference.load_from_raw_predictions)
        or bool(cfg.inference.save_raw_prediction)
    ):
        raise ValueError("Gate4 resolved config differs from official frozen contract")
    scale_factor = _pipeline_scale_factor(test.pipeline)
    if scale_factor != 1 or _pipeline_scale_factor(val.pipeline) != 1:
        raise ValueError("Gate4 official pipelines require scale_factor=1")

    config_sources = {
        relative: read_bound_bytes(
            ROOT / relative, label=f"Gate4 config source {relative}"
        )[2]
        for relative in _CONFIG_SOURCES
    }
    annotation_path = _resolve_repo_input(test.ann_file, label="Gate4 annotation")
    class_map_path = _resolve_repo_input(test.class_map, label="Gate4 class map")
    _, annotation, _, annotation_sha256 = load_bound_json(
        annotation_path, label="Gate4 annotation"
    )
    if not isinstance(annotation, Mapping) or not isinstance(
        annotation.get("database"), Mapping
    ):
        raise ValueError("Gate4 annotation database is invalid")
    _, class_map_bytes, class_map_sha256 = read_bound_bytes(
        class_map_path, label="Gate4 class map"
    )
    class_names = {
        line.strip()
        for line in class_map_bytes.decode("utf-8").splitlines()
        if line.strip()
    }
    if not class_names:
        raise ValueError("Gate4 class map is empty")

    exact_data_root = secure_lexical_path(data_root, label="Gate4 media root")
    video_rows = []
    ground_truth = []
    with open_bound_directory(exact_data_root, label="Gate4 media root") as media_root:
        for video_id, info in annotation["database"].items():
            if not isinstance(info, Mapping) or str(info.get("subset")) != "validation":
                continue
            media_path = f"{video_id}.mp4"
            with media_root.open_regular(
                media_path, label=f"Gate4 media {video_id}"
            ) as media:
                media_bytes, media_sha256 = media.size_and_sha256()
            video_rows.append(
                {
                    "official_video_id": str(video_id),
                    "media_path": media_path,
                    "media_bytes": media_bytes,
                    "media_sha256": media_sha256,
                    "frame": int(info["frame"]),
                    "duration": float(info["duration"]),
                }
            )
            for row in remove_duplicate_annotations(info.get("annotations", [])):
                label = str(row.get("label"))
                segment = row.get("segment")
                if label not in class_names:
                    continue
                if not isinstance(segment, list) or len(segment) != 2:
                    raise ValueError("Gate4 official annotation segment is invalid")
                ground_truth.append(
                    {
                        "official_video_id": str(video_id),
                        "label": label,
                        "segment": [float(segment[0]), float(segment[1])],
                    }
                )
    if not video_rows:
        raise ValueError("Gate4 official validation population is empty")

    _, manifest, _, _ = load_bound_json(
        fit_manifest_path, label="Gate4 fit manifest"
    )
    manifest = validate_r2_manifest(manifest)
    if manifest["data_identity"]["annotation_sha256"] != annotation_sha256:
        raise ValueError("Gate4 and fit manifest annotation identities differ")
    quartiles = _fit_duration_quartiles(annotation, manifest, class_names)
    contract = {
        "dataset_type": str(test.type),
        "subset": str(test.subset_name),
        "test_mode": bool(test.test_mode),
        "feature_stride": int(test.feature_stride),
        "sample_stride": int(test.sample_stride),
        "offset_frames": int(getattr(test, "offset_frames", 0)),
        "window_size": int(test.window_size),
        "window_overlap_ratio": float(test.window_overlap_ratio),
        "scale_factor": scale_factor,
        "test_pipeline_sha256": canonical_sha256(test.pipeline),
        "regret_pipeline_sha256": canonical_sha256(val.pipeline),
        "inference_sha256": canonical_sha256(cfg.inference),
        "post_processing_sha256": canonical_sha256(cfg.post_processing),
        "evaluation_sha256": canonical_sha256(cfg.evaluation),
    }
    return build_gate4_population_artifact(
        config_sources_sha256=config_sources,
        annotation={"path": str(annotation_path), "sha256": annotation_sha256},
        class_map={"path": str(class_map_path), "sha256": class_map_sha256},
        data_root=str(exact_data_root),
        dataset_contract=contract,
        videos=video_rows,
        ground_truth=ground_truth,
        fit_manifest_sha256=manifest["manifest_sha256"],
        fit_duration_quartile_thresholds=quartiles,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact = build_gate4_population_from_context(
        repository_root=args.repository_root,
        data_root=args.data_root,
        fit_manifest_path=args.fit_manifest,
    )
    payload = gate4_population_exact_bytes(artifact)
    publish_bytes_exclusive(args.output, payload, label="Gate4 population artifact")
    publish_bytes_exclusive(
        args.output.with_suffix(args.output.suffix + ".sha256"),
        (hashlib.sha256(payload).hexdigest() + "\n").encode("ascii"),
        label="Gate4 population SHA-256 sidecar",
    )
    print(json.dumps({"path": str(args.output), **artifact}, sort_keys=True))


if __name__ == "__main__":
    main()
