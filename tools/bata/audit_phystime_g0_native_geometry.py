from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.datasets.transforms.phystime_raw import (
    BuildPhysTimeNativeTubeletGeometry,
    BuildPhysTimeRawFrameGeometry,
)
from opentad.models import build_detector


DEFAULT_CONFIGS = {
    "selected_axis": ROOT / "configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py",
    "physical_metric": ROOT / "configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py",
}
SCHEMA_VERSION = "phystime_g0_native_geometry_static_precheck_v2"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parameter_schema(model):
    schema = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]
    return {
        "sha256": _sha256_json(schema),
        "tensor_count": len(schema),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "schema": schema,
    }


def _step(cfg, split, step_type):
    return next(step for step in cfg.dataset[split].pipeline if step["type"] == step_type)


def _query_lengths(native_count, level_count):
    lengths = [int(native_count)]
    for _ in range(1, int(level_count)):
        lengths.append((lengths[-1] + 1) // 2)
    return lengths


def _normalized_model_config(cfg):
    return copy.deepcopy(cfg.model.to_dict())


def _synthetic_geometry(coordinate_mode, raw_count, dense_count, tubelet_size, native_cfg):
    rng = np.random.RandomState(20260713)
    dense_positions = np.sort(rng.choice(dense_count, size=raw_count, replace=False)).astype(np.float32)
    stride = 4
    frame_origin = 120
    frame_indices = (frame_origin + dense_positions * stride).astype(np.int64)
    sample = {
        "frame_inds": frame_indices,
        "selected_raw_frame_indices": frame_indices.copy(),
        "selected_dense_indices": dense_positions,
        "masks": torch.ones(raw_count, dtype=torch.bool),
        "snippet_stride": stride,
        "fps": 30.0,
        "avg_fps": 30.0,
        "total_frames": 6000,
        "duration": 200.0,
        "irregular_dense_valid_len": dense_count,
        "irregular_sampling_strategy": "random_fixed_subsample",
        "irregular_sampling_scope": "within_accepted_window",
        "irregular_window_crop_uses_gt": False,
        "irregular_subsample_uses_gt": False,
        "irregular_native_axis": True,
        "remap_gt_to_selected_axis": False,
        "gt_remapped_to_selected_axis": False,
    }
    sample = BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=False)(sample)
    sample = BuildPhysTimeNativeTubeletGeometry(
        tubelet_size=tubelet_size,
        chunk_size=int(native_cfg["chunk_size"]),
        transformer_depth=int(native_cfg["transformer_depth"]),
        adapter_indices=list(native_cfg["adapter_indices"]),
        adapter_kernel_size=int(native_cfg["adapter_kernel_size"]),
        adapter_dilation=int(native_cfg["adapter_dilation"]),
        coordinate_mode=coordinate_mode,
    )(sample)
    token_positions = np.asarray(sample["phystime_native_token_dense_positions"], dtype=np.float32)
    gaps = np.asarray(sample["phystime_patch_embed_atom_gap_sec"], dtype=np.float32)
    inflation = np.asarray(sample["phystime_patch_embed_envelope_inflation_sec"], dtype=np.float32)
    return {
        "raw_index_sha256": hashlib.sha256(frame_indices.tobytes()).hexdigest(),
        "native_position_sha256": hashlib.sha256(token_positions.tobytes()).hexdigest(),
        "raw_valid_count": int(sample["phystime_raw_valid_count"]),
        "native_valid_count": int(sample["phystime_native_valid_count"]),
        "disconnected_token_count": int((gaps > 0).sum()),
        "atom_gap_sec_mean": float(gaps.mean()),
        "atom_gap_sec_max": float(gaps.max()),
        "envelope_inflation_sec_sum": float(inflation.sum()),
        "patch_lineage_provenance": sample["phystime_patch_embed_lineage_provenance"],
        "final_feature_lineage": sample["phystime_native_final_feature_lineage"],
        "final_feature_support_is_exact": sample["phystime_native_final_feature_support_is_exact"],
        "final_feature_raw_slot_upper_bound": sample[
            "phystime_native_final_feature_raw_slot_upper_bound"
        ],
        "structural_lineage_range_sha256": _sha256_json(
            sample["phystime_native_final_feature_raw_slot_ranges_exclusive"]
        ),
        "window_crop_uses_gt": sample["phystime_window_crop_uses_gt"],
        "subsample_uses_gt": sample["phystime_subsample_uses_gt"],
    }


def run_audit(config_paths=DEFAULT_CONFIGS, *, build_models=True, output=None):
    configs = {name: Config.fromfile(path, lazy_import=False) for name, path in config_paths.items()}
    _require(set(configs) == {"selected_axis", "physical_metric"}, "G0 requires exactly two named arms")
    selected = configs["selected_axis"]
    physical = configs["physical_metric"]
    raw_counts = {int(cfg.raw_observation_count) for cfg in configs.values()}
    native_counts = {int(cfg.native_token_count) for cfg in configs.values()}
    _require(len(raw_counts) == len(native_counts) == 1, "G0 arms disagree on K or J")
    raw_count = raw_counts.pop()
    native_count = native_counts.pop()
    tubelet_sizes = {
        int(cfg.model.native_temporal_geometry.tubelet_size) for cfg in configs.values()
    }
    _require(len(tubelet_sizes) == 1, "G0 arms disagree on tubelet size")
    tubelet_size = tubelet_sizes.pop()
    _require(raw_count == native_count * tubelet_size, "G0 K/J/tubelet count is inconsistent")

    coordinate_modes = {}
    for name, cfg in configs.items():
        for split in ("train", "val", "test"):
            post_types = [step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline]
            _require("Interpolate" not in post_types, f"{name} illegally interpolates native features")
            loader = _step(cfg, split, "LoadFrames")
            _require(loader["method"] == "random_fixed_subsample", f"{name}/{split} sampling mismatch")
            _require(loader["remap_gt_to_selected_axis"] is False, f"{name}/{split} remaps GT before G0")
            native = _step(cfg, split, "BuildPhysTimeNativeTubeletGeometry")
            coordinate_modes.setdefault(name, set()).add(str(native["coordinate_mode"]))
        _require(len(coordinate_modes[name]) == 1, f"{name} changes coordinate mode across splits")

    _require(
        coordinate_modes["selected_axis"] == {"uniform_rank_seconds"},
        "selected control seconds mode mismatch",
    )
    _require(
        coordinate_modes["physical_metric"] == {"physical_time_seconds"},
        "physical seconds mode mismatch",
    )
    _require(
        _normalized_model_config(selected) == _normalized_model_config(physical),
        "G0 model configs must be exactly identical",
    )

    level_count = len(selected.model.rpn_head.prior_generator.strides)
    query_lengths = _query_lengths(native_count, level_count)
    synthetic = {
        name: _synthetic_geometry(
            next(iter(coordinate_modes[name])),
            raw_count,
            int(cfg.dense_window_size),
            tubelet_size,
            _step(cfg, "train", "BuildPhysTimeNativeTubeletGeometry"),
        )
        for name, cfg in configs.items()
    }
    _require(
        synthetic["selected_axis"]["raw_index_sha256"]
        == synthetic["physical_metric"]["raw_index_sha256"],
        "G0 arms selected different raw observations",
    )
    _require(
        synthetic["selected_axis"]["native_position_sha256"]
        == synthetic["physical_metric"]["native_position_sha256"],
        "G0 arms built different native tubelet positions",
    )

    schemas = {}
    if build_models:
        model_cfgs = {}
        for name, cfg in configs.items():
            model_cfg = copy.deepcopy(cfg.model)
            model_cfg.backbone.custom.pretrain = None
            model_cfgs[name] = model_cfg
        models = {name: build_detector(model_cfgs[name]) for name in configs}
        schemas = {name: parameter_schema(model) for name, model in models.items()}
        _require(
            schemas["selected_axis"]["schema"] == schemas["physical_metric"]["schema"],
            "G0 arms have different parameter schemas",
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": False,
        "static_precheck_pass": True,
        "status": "model_built_static_precheck_not_real_gate"
        if build_models
        else "static_precheck_only_not_real_gate",
        "K_raw_observations": raw_count,
        "J_native_tubelet_tokens": native_count,
        "tubelet_size": tubelet_size,
        "Q0_base_candidates": int(query_lengths[0]),
        "Q_level_lengths": query_lengths,
        "Q_total_candidates": int(sum(query_lengths)),
        "feature_interpolation": False,
        "representation_lift": "none_native_j_grid",
        "coordinate_modes": {name: next(iter(values)) for name, values in coordinate_modes.items()},
        "synthetic_geometry": synthetic,
        "synthetic_selected_index_checksum_match": True,
        "synthetic_native_position_checksum_match": True,
        "parameter_schema_match": bool(build_models),
        "executed_query_shapes_verified": False,
        "real_pipeline_verified": False,
        "lineage_evidence_level": "structural_graph_upper_bound_not_jacobian",
        "parameter_schemas": schemas,
        "config_sha256": {name: _sha256_json(cfg.to_dict()) for name, cfg in configs.items()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip(),
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Audit PhysTime G0 native K/J/Q provenance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_audit(build_models=not args.static_only, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
