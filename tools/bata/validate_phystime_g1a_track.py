from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = {
    "selected_axis": ROOT / "configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py",
    "physical_metric": ROOT / "configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py",
}
SCHEMA_VERSION = "phystime_g1a_track_contract_v2"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(value):
    return copy.deepcopy(value) if isinstance(value, Config) else Config.fromfile(value, lazy_import=False)


def _step(cfg, split, step_type):
    matches = [step for step in cfg.dataset[split].pipeline if step["type"] == step_type]
    _require(len(matches) == 1, f"{split} must contain exactly one {step_type}")
    return matches[0]


def _query_lengths(native_count, level_count):
    lengths = [int(native_count)]
    for _ in range(1, int(level_count)):
        lengths.append((lengths[-1] + 1) // 2)
    return lengths


def _normalized_dataset(cfg):
    dataset = copy.deepcopy(cfg.dataset.to_dict())
    for split in ("train", "val", "test"):
        native = next(
            step
            for step in dataset[split]["pipeline"]
            if step["type"] == "BuildPhysTimeNativeTubeletGeometry"
        )
        native["coordinate_mode"] = "<coordinate-only-control>"
    return dataset


def validate_track(config_paths=DEFAULT_CONFIGS, *, output=None):
    _require(set(config_paths) == {"selected_axis", "physical_metric"}, "G1a requires exactly two arms")
    configs = {name: _load(value) for name, value in config_paths.items()}

    raw_counts = {int(cfg.raw_observation_count) for cfg in configs.values()}
    native_counts = {int(cfg.native_token_count) for cfg in configs.values()}
    _require(raw_counts == {384}, "G1a requires K=384 raw observations")
    _require(native_counts == {192}, "G1a requires J=192 native tubelet tokens")

    coordinate_modes = {}
    expected_modes = {
        "selected_axis": "uniform_rank_seconds",
        "physical_metric": "physical_time_seconds",
    }
    for name, cfg in configs.items():
        _require(int(cfg.dense_window_size) == 768, f"{name} requires a 768-position logical window")
        _require(int(cfg.scale_factor) == 1, f"{name} G1a raw-slot audit requires scale_factor=1")
        _require(int(cfg.model.projection.max_seq_len) == 192, f"{name} projection must operate on J=192")
        _require(
            int(cfg.model.backbone.backbone.total_frames) == 384,
            f"{name} backbone must consume exactly K=384 raw slots",
        )
        native_cfg = cfg.model.native_temporal_geometry
        _require(int(native_cfg.expected_raw_count) == 384, f"{name} native K contract mismatch")
        _require(int(native_cfg.expected_token_count) == 192, f"{name} native J contract mismatch")
        _require(int(native_cfg.tubelet_size) == 2, f"{name} tubelet size must be two")
        backbone_cfg = cfg.model.backbone.backbone
        adapter_indices = [int(value) for value in backbone_cfg.adapter_index]
        _require(int(backbone_cfg.depth) == 12, f"{name} transformer depth must remain 12")
        _require(adapter_indices == list(range(12)), f"{name} must retain all 12 official temporal adapters")
        _require(int(native_cfg.expected_transformer_depth) == 12, f"{name} lineage depth mismatch")
        _require(
            [int(value) for value in native_cfg.expected_adapter_indices] == adapter_indices,
            f"{name} lineage adapter indices mismatch",
        )
        _require(int(native_cfg.expected_adapter_kernel_size) == 3, f"{name} adapter kernel mismatch")
        _require(int(native_cfg.expected_adapter_dilation) == 1, f"{name} adapter dilation mismatch")
        post_types = [step["type"] for step in cfg.model.backbone.custom.post_processing_pipeline]
        _require("Interpolate" not in post_types, f"{name} must not interpolate J192 features")
        _require(
            bool(cfg.model.backbone.custom.get("strict_temporal_padding_mask", False)) is True,
            f"{name} must isolate tail padding inside VideoMAE attention and temporal adapters",
        )

        modes = set()
        for split in ("train", "val", "test"):
            loader = _step(cfg, split, "LoadFrames")
            _require(loader["method"] == "random_fixed_subsample", f"{name}/{split} sampling changed")
            _require(int(loader["target_len"]) == 384, f"{name}/{split} requires K=384")
            _require(int(loader["source_len"]) == 768, f"{name}/{split} requires dense length 768")
            _require(loader["remap_gt_to_selected_axis"] is False, f"{name}/{split} remaps GT")
            expected_base = "random_trunc" if split == "train" else "sliding_window"
            _require(loader["method_base"] == expected_base, f"{name}/{split} window protocol changed")

            raw = _step(cfg, split, "BuildPhysTimeRawFrameGeometry")
            _require(
                bool(raw["convert_gt_to_seconds"]) is (split != "test"),
                f"{name}/{split} canonical-seconds GT contract changed",
            )
            for tolerance_key, expected_tolerance in (
                ("fps_relative_tolerance", 0.0125),
                ("duration_relative_tolerance", 0.0125),
                ("frame_count_relative_tolerance", 0.0001),
            ):
                _require(
                    float(raw.get(tolerance_key, -1.0)) == expected_tolerance,
                    f"{name}/{split} {tolerance_key} changed",
                )
            native = _step(cfg, split, "BuildPhysTimeNativeTubeletGeometry")
            _require(int(native["tubelet_size"]) == 2, f"{name}/{split} tubelet size changed")
            _require(int(native["chunk_size"]) == 16, f"{name}/{split} chunk context changed")
            _require(int(native["transformer_depth"]) == 12, f"{name}/{split} lineage depth changed")
            _require(
                [int(value) for value in native["adapter_indices"]] == adapter_indices,
                f"{name}/{split} lineage adapter indices changed",
            )
            _require(int(native["adapter_kernel_size"]) == 3, f"{name}/{split} adapter kernel changed")
            _require(int(native["adapter_dilation"]) == 1, f"{name}/{split} adapter dilation changed")
            modes.add(str(native["coordinate_mode"]))
        _require(modes == {expected_modes[name]}, f"{name} coordinate mode mismatch")
        coordinate_modes[name] = next(iter(modes))

    selected = configs["selected_axis"]
    physical = configs["physical_metric"]
    _require(selected.model.to_dict() == physical.model.to_dict(), "G1a model configs are not exactly matched")
    _require(
        _normalized_dataset(selected) == _normalized_dataset(physical),
        "G1a datasets differ beyond the coordinate mode",
    )
    for key in ("optimizer", "solver", "scheduler", "inference", "post_processing", "evaluation"):
        _require(
            copy.deepcopy(selected[key]) == copy.deepcopy(physical[key]),
            f"G1a {key} configs are not matched",
        )
    solver = physical.solver
    _require(bool(solver.get("amp", False)) is True, "G1a requires AMP")
    _require(float(solver.get("amp_init_scale", -1.0)) == 1024.0, "G1a AMP init scale must be 1024")
    _require(bool(solver.get("fp16_compress", True)) is False, "G1a single-GPU run forbids FP16 DDP compression")
    _require(bool(solver.get("fail_on_non_finite_grad", False)) is True, "G1a must fail on non-finite gradients")
    _require(int(solver.get("max_consecutive_amp_skips", -1)) == 4, "G1a consecutive AMP skip budget changed")
    _require(int(solver.get("max_total_amp_skips_per_epoch", -1)) == 8, "G1a epoch AMP skip budget changed")

    grid = physical.model.rpn_head.physical_grid_actionformer
    _require(grid.enabled is True and grid.required is True and grid.strict is True, "G1a seconds grid is not strict")
    _require(grid.positions_key == "phystime_g1a_axis_positions_sec", "G1a positions key mismatch")
    _require(grid.axis_start_key == "phystime_g1a_axis_start_sec", "G1a domain start key mismatch")
    _require(grid.axis_end_key == "phystime_g1a_axis_end_sec", "G1a domain end key mismatch")

    load_frames_source = (ROOT / "opentad/datasets/transforms/end_to_end.py").read_text(encoding="utf-8")
    _require(
        'results["irregular_subsample_uses_gt"] = False' in load_frames_source,
        "random-fixed within-window subsampling lacks an explicit no-GT audit flag",
    )
    level_lengths = _query_lengths(192, len(physical.model.rpn_head.prior_generator.strides))
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract_pass": True,
        "status": "static_contract_only_not_effectiveness_evidence",
        "K_raw_observations": 384,
        "J_native_tubelet_tokens": 192,
        "Q0_base_candidate_tensor_slots": int(level_lengths[0]),
        "Q0_base_candidates": int(level_lengths[0]),
        "Q_level_tensor_lengths": level_lengths,
        "Q_level_lengths": level_lengths,
        "Q_total_candidate_tensor_slots": int(sum(level_lengths)),
        "Q_total_candidates": int(sum(level_lengths)),
        "effective_candidate_count_policy": "semantic_anchor_prefix_reported_per_sample",
        "feature_interpolation": False,
        "strict_temporal_padding_isolation": True,
        "model_config_exact_match": True,
        "dataset_config_match_except_coordinate_mode": True,
        "coordinate_modes": coordinate_modes,
        "canonical_coordinate_unit": "seconds",
        "timebase_relative_tolerances": {
            "fps": 0.0125,
            "duration": 0.0125,
            "frame_count": 0.0001,
        },
        "train_window_crop_uses_gt": True,
        "within_window_subsample_uses_gt": False,
        "train_window_crop_scope": "standard_adatad_random_trunc_before_irregular_subsampling",
        "config_sha256": {name: _sha256_json(cfg.to_dict()) for name, cfg in configs.items()},
        "config_paths": {name: str(Path(DEFAULT_CONFIGS[name]).resolve()) for name in configs},
        "model_config_sha256": _sha256_json(physical.model.to_dict()),
        "normalized_dataset_sha256": _sha256_json(_normalized_dataset(physical)),
        "structural_lineage": {
            "transformer_depth": 12,
            "adapter_indices": list(range(12)),
            "adapter_kernel_size": 3,
            "adapter_dilation": 1,
            "attention_chunk_raw_frames": 16,
            "attention_chunk_native_tokens": 8,
            "evidence_level": "exact_patch_inputs_plus_structural_receptive_field_upper_bound",
        },
        "amp_contract": {
            "enabled": True,
            "init_scale": 1024.0,
            "fp16_compress": False,
            "fail_on_non_finite_grad": True,
            "max_consecutive_skips": 4,
            "max_total_skips_per_epoch": 8,
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip(),
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the matched PhysTime G1a track")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = validate_track(output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
