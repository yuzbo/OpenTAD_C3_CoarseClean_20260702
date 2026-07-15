from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import (
    S1_CHECKPOINT_RULE,
    S1_RESOLUTIONS,
    S1_TRAINING_SEEDS,
    canonical_sha256,
)

CONFIG_PATHS = {
    160: "configs/adatad/thumos/s1_dense160_videomae_s_768x1_adapter.py",
    224: "configs/adatad/thumos/s1_dense224_videomae_s_768x1_adapter.py",
    256: "configs/adatad/thumos/s1_dense256_videomae_s_768x1_adapter.py",
}
OFFICIAL_DENSE160_CONFIG = (
    ROOT / "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
)
EXPECTED_TRAIN_SHORT_SIDE = {160: 182, 224: 255, 256: 291}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Config):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _load(value: str | Path | Config) -> Config:
    if isinstance(value, Config):
        return value
    return Config.fromfile(str(value))


def _validate_pipeline(cfg: Config, resolution: int) -> None:
    train = cfg.dataset.train.pipeline
    val = cfg.dataset.val.pipeline
    test = cfg.dataset.test.pipeline
    _require(len(train) == 13, "S1 train pipeline length changed")
    _require(len(val) == 9 and len(test) == 9, "S1 val/test pipeline length changed")
    _require(train[2].type == "LoadFrames", "S1 train temporal loader changed")
    _require(int(train[2].trunc_len) == 768, "S1 train temporal window changed")
    _require(
        train[4].type == "mmaction.Resize", "S1 train short-side transform changed"
    )
    _require(
        tuple(train[4].scale) == (-1, EXPECTED_TRAIN_SHORT_SIDE[resolution]),
        "unexpected S1 train short side",
    )
    _require(
        train[5].type == "mmaction.RandomResizedCrop", "S1 augmentation family changed"
    )
    _require(train[6].type == "mmaction.Resize", "S1 final train resize changed")
    _require(
        tuple(train[6].scale) == (resolution, resolution),
        "unexpected S1 train resolution",
    )
    _require(train[6].keep_ratio is False, "S1 final train resize must be square")
    for split_name, pipeline in (("val", val), ("test", test)):
        _require(
            pipeline[2].type == "LoadFrames", f"S1 {split_name} temporal loader changed"
        )
        _require(
            pipeline[4].type == "mmaction.Resize", f"S1 {split_name} resize changed"
        )
        _require(
            tuple(pipeline[4].scale) == (-1, resolution),
            f"unexpected S1 {split_name} short side",
        )
        _require(
            pipeline[5].type == "mmaction.CenterCrop",
            f"S1 {split_name} crop family changed",
        )
        _require(
            int(pipeline[5].crop_size) == resolution,
            f"unexpected S1 {split_name} crop size",
        )


def _normalize_spatial_fields(cfg: Config) -> dict[str, Any]:
    normalized = copy.deepcopy(_plain(cfg))
    normalized.pop("work_dir", None)
    normalized.pop("spatial_zoom_s1_contract", None)
    for split_name in ("train", "val", "test"):
        pipeline = normalized["dataset"][split_name]["pipeline"]
        for transform in pipeline:
            if transform.get("type") == "mmaction.Resize":
                transform["scale"] = "<S1_SPATIAL_RESOLUTION>"
            if transform.get("type") == "mmaction.CenterCrop":
                transform["crop_size"] = "<S1_SPATIAL_RESOLUTION>"
    return normalized


def _normalize_official_interpolation_equivalence(cfg: Config) -> dict[str, Any]:
    normalized = _normalize_spatial_fields(cfg)
    pipeline = normalized["model"]["backbone"]["custom"][
        "post_processing_pipeline"
    ]
    transforms = [row for row in pipeline if row.get("type") == "Interpolate"]
    _require(len(transforms) == 1, "S1 requires one temporal interpolation transform")
    transform = transforms[0]
    _require(transform.get("mode", "linear") == "linear", "S1 interpolation mode changed")
    _require(int(transform["size"]) == 768, "S1 interpolation target changed")
    transform["mode"] = "linear"
    transform["deterministic"] = "<DETERMINISTIC_EQUIVALENT>"
    transform["expected_input_size"] = "<TEMPORAL_INPUT_POINTS>"
    return normalized


def _first_diff(left: Any, right: Any, path: str = "cfg") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys {sorted(set(left) ^ set(right))}"
        for key in sorted(left):
            diff = _first_diff(left[key], right[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            diff = _first_diff(left_item, right_item, f"{path}[{index}]")
            if diff:
                return diff
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def validate_config_matrix(
    configs: Mapping[int, str | Path | Config] | None = None,
) -> dict[str, Any]:
    configs = configs or {
        resolution: ROOT / path for resolution, path in CONFIG_PATHS.items()
    }
    _require(
        tuple(sorted(configs)) == S1_RESOLUTIONS,
        "S1 matrix must contain exactly 160/224/256",
    )
    loaded = {int(resolution): _load(value) for resolution, value in configs.items()}
    rows: dict[str, Any] = {}
    normalized = {}
    normalized_contracts = {}
    for resolution in S1_RESOLUTIONS:
        cfg = loaded[resolution]
        contract = cfg.spatial_zoom_s1_contract
        _require(
            contract.schema_version == "spatial_zoom_s1_config_v2",
            "unexpected S1 config schema",
        )
        _require(
            int(contract.runtime_resolution) == resolution,
            "S1 runtime resolution contract mismatch",
        )
        _require(
            int(contract.train_short_side) == EXPECTED_TRAIN_SHORT_SIDE[resolution],
            "S1 train short-side contract mismatch",
        )
        _require(int(contract.temporal_window) == 768, "S1 temporal contract changed")
        _require(
            int(contract.detector_time_grid) == 768, "S1 detector time grid changed"
        )
        _require(int(contract.tubelet_points) == 384, "S1 tubelet grid changed")
        _require(
            contract.temporal_interpolation
            == "linear_align_corners_false_2x_deterministic_v1",
            "S1 deterministic temporal interpolation contract changed",
        )
        _require(
            int(contract.temporal_interpolation_input_points) == 384,
            "S1 temporal interpolation input grid changed",
        )
        _require(
            list(contract.training_seeds) == list(S1_TRAINING_SEEDS),
            "S1 training seeds changed",
        )
        _require(
            contract.checkpoint_selection_rule == S1_CHECKPOINT_RULE,
            "S1 checkpoint rule changed",
        )
        _require(
            contract.fit_gate_manifest_required is True, "S1 manifest must be required"
        )
        _require(
            contract.official_test_sealed_until_protocol_freeze is True,
            "S1 official test must remain sealed",
        )
        for key in (
            "roi_policy_enabled",
            "teacher_oracle_enabled",
            "new_detector_enabled",
            "paper_claim_allowed",
        ):
            _require(getattr(contract, key) is False, f"S1 forbids {key}")
        _require(
            int(cfg.window_size) == 768 and int(cfg.chunk_num) == 48,
            "S1 temporal chunking changed",
        )
        _require(
            cfg.model.type == "ActionFormer",
            "S1 must retain the official-derived ActionFormer detector",
        )
        _require(
            cfg.model.get("frame_selector", None) is None,
            "S1 must not add a temporal selector",
        )
        _require(
            int(cfg.model.backbone.backbone.num_frames) == 16,
            "S1 VideoMAE clip length changed",
        )
        _require(
            int(cfg.model.backbone.backbone.get("tubelet_size", 2)) == 2,
            "S1 VideoMAE tubelet changed",
        )
        _require(
            int(cfg.model.projection.max_seq_len) == 768, "S1 projection grid changed"
        )
        temporal_interpolators = [
            row
            for row in cfg.model.backbone.custom.post_processing_pipeline
            if row.type == "Interpolate"
        ]
        _require(
            len(temporal_interpolators) == 1,
            "S1 requires exactly one temporal interpolation transform",
        )
        temporal_interpolator = temporal_interpolators[0]
        _require(
            temporal_interpolator.get("deterministic", False) is True,
            "S1 temporal interpolation must use the deterministic implementation",
        )
        _require(
            int(temporal_interpolator.expected_input_size) == 384
            and int(temporal_interpolator.size) == 768
            and temporal_interpolator.mode == "linear",
            "S1 deterministic interpolation shape or mode changed",
        )
        _require(
            cfg.inference.load_from_raw_predictions is False,
            "S1 raw prediction loading is forbidden",
        )
        _validate_pipeline(cfg, resolution)
        normalized_contract = _plain(contract)
        normalized_contract["runtime_resolution"] = "<S1_SPATIAL_RESOLUTION>"
        normalized_contract["train_short_side"] = "<S1_SPATIAL_RESOLUTION>"
        normalized_contracts[resolution] = normalized_contract
        normalized[resolution] = _normalize_spatial_fields(cfg)
        rows[str(resolution)] = {
            "runtime_resolution": resolution,
            "train_short_side": EXPECTED_TRAIN_SHORT_SIDE[resolution],
            "window_size": int(cfg.window_size),
            "tubelet_points": int(contract.tubelet_points),
            "resolved_config_sha256": canonical_sha256(_plain(cfg)),
        }
    baseline = normalized[160]
    official_dense160 = _normalize_official_interpolation_equivalence(
        _load(OFFICIAL_DENSE160_CONFIG)
    )
    s1_dense160_for_official = _normalize_official_interpolation_equivalence(
        loaded[160]
    )
    official_diff = _first_diff(official_dense160, s1_dense160_for_official)
    _require(
        official_diff is None,
        "S1 dense160 differs from the official-derived local baseline: "
        f"{official_diff}",
    )
    for resolution in (224, 256):
        contract_diff = _first_diff(
            normalized_contracts[160],
            normalized_contracts[resolution],
            "cfg.spatial_zoom_s1_contract",
        )
        _require(
            contract_diff is None,
            f"S1 contract differs outside spatial fields: {contract_diff}",
        )
        diff = _first_diff(baseline, normalized[resolution])
        _require(
            diff is None,
            f"S1 configs differ outside only permitted spatial fields: {diff}",
        )
    return {
        "status": "PASS",
        "schema_version": "spatial_zoom_s1_config_matrix_v1",
        "resolutions": list(S1_RESOLUTIONS),
        "only_spatial_resolution_differs": True,
        "official_dense160_matched": True,
        "temporal_protocol_matched": True,
        "model_optimizer_evaluator_matched": True,
        "deterministic_temporal_interpolation": True,
        "protocol_fingerprint": canonical_sha256(baseline),
        "configs": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the matched S1 dense-resolution matrix"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": "FileExistsError",
                    "error": "refusing to overwrite an S1 config-matrix certificate",
                },
                indent=2,
            )
        )
        return 1
    try:
        summary = validate_config_matrix()
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            )
        )
        return 1
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
