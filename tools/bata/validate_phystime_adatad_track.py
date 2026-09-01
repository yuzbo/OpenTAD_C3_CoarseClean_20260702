from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.datasets.transforms.end_to_end import LoadFrames


DEFAULT_CONFIGS = {
    "selected": ROOT / "configs/adatad/thumos/selected_axis_adatad_sparse_k384.py",
    "physical": ROOT / "configs/adatad/thumos/physical_grid_adatad_sparse_k384.py",
    "phystime": ROOT / "configs/adatad/thumos/phystime_adatad_sparse_k384.py",
}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _require_equal(name, values):
    normalized = [_canonical(value) for value in values]
    if len(set(normalized)) != 1:
        raise RuntimeError(f"matched PhysTime-AdaTAD contract differs for {name}: {normalized}")


def _pipeline(cfg, split):
    return list(cfg.dataset[split].pipeline)


def _load_frames(cfg, split):
    matches = [step for step in _pipeline(cfg, split) if step["type"] == "LoadFrames"]
    _require(len(matches) == 1, f"{split} must contain exactly one LoadFrames transform")
    return matches[0]


def _sampling_contract(cfg):
    contract = {}
    for split in ("train", "val", "test"):
        step = dict(_load_frames(cfg, split))
        step.pop("remap_gt_to_selected_axis", None)
        contract[split] = _plain(step)
    return contract


def _pipeline_without_allowed_geometry(cfg, split):
    normalized = []
    for step in _pipeline(cfg, split):
        if step["type"] == "BuildPhysTimeRawFrameGeometry":
            continue
        step = dict(step)
        if step["type"] == "LoadFrames":
            step.pop("remap_gt_to_selected_axis", None)
        normalized.append(_plain(step))
    return normalized


def _frame_checksum(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _runtime_sampling_checksum():
    sample = {
        "video_name": "phystime_contract_sample",
        "total_frames": 4000,
        "avg_fps": 30.0,
        "snippet_stride": 4,
        "window_size": 768,
        "feature_start_idx": 20,
        "feature_end_idx": 787,
    }
    checksums = []
    for remap in (True, False, False):
        loader = LoadFrames(
            method="random_fixed_subsample",
            method_base="sliding_window",
            keep_ratio=0.5,
            target_len=384,
            source_len=768,
            remap_gt_to_selected_axis=remap,
        )
        checksums.append(_frame_checksum(loader(dict(sample))["frame_inds"]))
    _require(len(set(checksums)) == 1, "runtime selected-frame checksums differ across heads")
    return checksums[0]


def validate_track(config_paths=None, output=None):
    paths = dict(DEFAULT_CONFIGS if config_paths is None else config_paths)
    _require(set(paths) == {"selected", "physical", "phystime"}, "three named head configs are required")
    cfgs = {name: Config.fromfile(str(path)) for name, path in paths.items()}
    values = list(cfgs.values())

    _require_equal("window_size", [cfg.window_size for cfg in values])
    _require_equal("dense_window_size", [cfg.dense_window_size for cfg in values])
    _require(int(values[0].window_size) == 384, "target sparse length must be K=384")
    _require(int(values[0].dense_window_size) == 768, "logical dense window must be 768")

    for split in ("train", "val", "test"):
        _require_equal(
            f"dataset.{split}.identity",
            [
                {
                    key: cfg.dataset[split].get(key)
                    for key in ("type", "ann_file", "subset_name", "class_map", "data_path", "sample_stride")
                }
                for cfg in values
            ],
        )
        _require_equal(
            f"dataset.{split}.pipeline_except_geometry",
            [_pipeline_without_allowed_geometry(cfg, split) for cfg in values],
        )
        types = [[step["type"] for step in _pipeline(cfg, split)] for cfg in values]
        for name, split_types in zip(cfgs, types):
            _require("LoadFeats" not in split_types, f"{name}/{split} must not use feature archives")
            _require("mmaction.DecordDecode" in split_types, f"{name}/{split} must decode selected RGB frames")

    sampling_contracts = [_sampling_contract(cfg) for cfg in values]
    _require_equal("sampling_contract", sampling_contracts)
    _require_equal("model.backbone", [cfg.model.backbone for cfg in values])
    for field in ("optimizer", "scheduler", "solver", "workflow", "evaluation", "post_processing"):
        _require_equal(field, [cfg[field] for cfg in values])

    _require(cfgs["selected"].model.type == "ActionFormer", "selected-axis must use ActionFormer")
    _require(cfgs["selected"].model.rpn_head.type == "ActionFormerHead", "selected-axis head mismatch")
    _require(cfgs["physical"].model.type == "ActionFormer", "physical-grid must use ActionFormer")
    _require(
        cfgs["physical"].model.rpn_head.physical_grid_actionformer.enabled is True,
        "physical-grid ActionFormer must be enabled",
    )
    _require(cfgs["phystime"].model.type == "PhysTimeTAD", "PhysTime detector type mismatch")
    _require(
        cfgs["phystime"].model.projection.type == "PhysTimeMeasureProjection",
        "PhysTime projection type mismatch",
    )
    _require(cfgs["phystime"].model.rpn_head.type == "PhysTimeHead", "PhysTime head type mismatch")
    _require(cfgs["phystime"].model.discretization_loss_weight == 0.0, "primary comparison forbids consistency")

    resolved_hashes = {name: _sha256(cfg.to_dict()) for name, cfg in cfgs.items()}
    payload = {
        "schema_version": "phystime_adatad_matched_contract_v1",
        "contract_pass": True,
        "raw_video_only": True,
        "target_len": 384,
        "dense_window_size": 768,
        "sampling_contract_sha256": _sha256(sampling_contracts[0]),
        "runtime_selected_indices_sha256": _runtime_sampling_checksum(),
        "resolved_config_sha256": resolved_hashes,
        "configs": {name: str(Path(path).resolve()) for name, path in paths.items()},
        "allowed_differences": [
            "selected-axis GT remap",
            "physical-grid ActionFormer assignment",
            "PhysTime raw geometry, projection, and head",
            "work_dir",
        ],
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the matched PhysTime-AdaTAD K384 track")
    parser.add_argument("--selected-config", type=Path, default=DEFAULT_CONFIGS["selected"])
    parser.add_argument("--physical-config", type=Path, default=DEFAULT_CONFIGS["physical"])
    parser.add_argument("--phystime-config", type=Path, default=DEFAULT_CONFIGS["phystime"])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = validate_track(
        config_paths={
            "selected": args.selected_config,
            "physical": args.physical_config,
            "phystime": args.phystime_config,
        },
        output=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
