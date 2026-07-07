from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmengine.config import Config


CONFIG_DEFAULT = "configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py"
READY = "C3_DETECTOR_AWARE_LEDGER_FULL_TRAIN_GATE_PASS"
SOURCE = "learned_detector_aware_policy_checkpoint"
VARIANT_SPECS = {
    "detector_aware_fixed_384": dict(target_len=384, require_selected_count=384, strategy="detector_aware_fixed_384"),
    "detector_aware_fixed_768": dict(target_len=768, require_selected_count=768, strategy="detector_aware_fixed_768"),
    "detector_aware_dynamic": dict(target_len=768, require_selected_count=None, strategy="detector_aware_dynamic"),
}
FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "uses_checkpoint",
    "prediction_uses_gt",
    "training_only",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _find_loadframes(pipeline: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [step for step in pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
    _require(len(matches) == 1, f"expected exactly one LoadFrames step, got {len(matches)}")
    return matches[0]


def _find_collect(pipeline: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [step for step in pipeline if isinstance(step, dict) and step.get("type") == "Collect"]
    _require(len(matches) == 1, f"expected exactly one Collect step, got {len(matches)}")
    return matches[0]


def _variant_spec(cfg: Config) -> dict[str, Any]:
    variant = str(cfg.detector_aware_ledger_variant)
    _require(variant in VARIANT_SPECS, f"unknown detector_aware_ledger_variant={variant}")
    return dict(VARIANT_SPECS[variant])


def _validate_loader(name: str, dataset_cfg: Any, expected_ledger_path: str, cfg: Config) -> None:
    spec = _variant_spec(cfg)
    loader = _find_loadframes(dataset_cfg.pipeline)
    _require(loader.get("method") == "bata_value_transport_ledger_subsample", f"{name}: wrong LoadFrames method")
    _require(loader.get("method_base") == "sliding_window", f"{name}: method_base must be sliding_window")
    _require(int(loader.get("target_len")) == int(spec["target_len"]), f"{name}: wrong target_len")
    _require(loader.get("bata_value_transport_require_selected_count") == spec["require_selected_count"], f"{name}: wrong require_selected_count")
    _require(
        not _as_bool(loader.get("bata_value_transport_allow_short_valid_ratio_count")),
        f"{name}: short-tail ratio gate must be off for exact detector-aware budget",
    )
    _require(_as_bool(loader.get("bata_value_transport_require_deployable")), f"{name}: deployable ledger not required")
    _require(not _as_bool(loader.get("bata_value_transport_allow_missing_fallback")), f"{name}: missing fallback must be off")
    _require(_as_bool(loader.get("remap_gt_to_selected_axis")), f"{name}: selected-axis GT remap must be on")
    _require(loader.get("bata_value_transport_ledger_path") == expected_ledger_path, f"{name}: wrong ledger path")
    _require(loader.get("bata_value_transport_source") == SOURCE, f"{name}: wrong detector-aware source")


def _validate_dataset(cfg: Config) -> None:
    _require(cfg.dataset.train.type == "ThumosSlidingDataset", "train must use ThumosSlidingDataset")
    _require(cfg.dataset.val.type == "ThumosSlidingDataset", "val must use ThumosSlidingDataset")
    _require(cfg.dataset.test.type == "ThumosSlidingDataset", "test must use ThumosSlidingDataset")
    _require(cfg.dataset.train.subset_name == "training", "train subset must be training")
    _require(cfg.dataset.val.subset_name == "validation", "val subset must be validation")
    _require(cfg.dataset.test.subset_name == "validation", "test subset must be validation")
    _require(int(cfg.dataset.train.window_size) == 768, "train dense window must be 768")
    _require(int(cfg.dataset.val.window_size) == 768, "val dense window must be 768")
    _require(int(cfg.dataset.test.window_size) == 768, "test dense window must be 768")
    _validate_loader("train", cfg.dataset.train, cfg.train_ledger_path, cfg)
    _validate_loader("val", cfg.dataset.val, cfg.val_ledger_path, cfg)
    _validate_loader("test", cfg.dataset.test, cfg.test_ledger_path, cfg)
    train_collect = _find_collect(cfg.dataset.train.pipeline)
    val_collect = _find_collect(cfg.dataset.val.pipeline)
    test_collect = _find_collect(cfg.dataset.test.pipeline)
    _require("gt_segments" in train_collect.get("keys", []), "train must collect gt_segments")
    _require("gt_labels" in train_collect.get("keys", []), "train must collect gt_labels")
    _require("gt_segments" in val_collect.get("keys", []), "val must collect gt_segments")
    _require("gt_labels" in val_collect.get("keys", []), "val must collect gt_labels")
    _require("gt_segments" not in test_collect.get("keys", []), "test must not collect gt_segments")
    _require("gt_labels" not in test_collect.get("keys", []), "test must not collect gt_labels")
    for split_name, collect in (("train", train_collect), ("val", val_collect), ("test", test_collect)):
        meta_keys = set(collect.get("meta_keys", []))
        for key in ("selected_valid_len", "irregular_selected_positions", "bata_selected_dense_indices"):
            _require(key in meta_keys, f"{split_name}: missing meta key {key}")


def _validate_model_and_train(cfg: Config) -> None:
    spec = _variant_spec(cfg)
    target_len = int(spec["target_len"])
    _require(int(cfg.window_size) == target_len, "selected window_size mismatch")
    _require(int(cfg.dense_window_size) == 768, "dense_window_size must be 768")
    _require(int(cfg.model.backbone.backbone.total_frames) == target_len, "backbone total_frames mismatch")
    _require(int(cfg.model.projection.max_seq_len) == target_len, "projection max_seq_len mismatch")
    _require("frame_selector" not in repr(cfg.model), "AdaTAD detector config must not include online frame_selector")
    _require("teacher_utility" not in repr(cfg.model).lower(), "model must not consume teacher utility")
    _require(not _as_bool(cfg.inference.load_from_raw_predictions), "raw prediction loading must be off")
    _require(not _as_bool(cfg.inference.save_raw_prediction), "raw prediction saving must be off")
    _require(cfg.evaluation.type == "mAP", "evaluation must be mAP")
    _require(cfg.evaluation.subset == "validation", "evaluation subset must be validation")
    _require(int(cfg.workflow.end_epoch) == 60, "formal full train must run 60 epochs")
    _require(cfg.workflow.get("max_train_iters", None) is None, "full train must not cap train iterations")
    _require(int(cfg.workflow.val_eval_interval) == 10, "validation interval must be 10")
    _require("val_eval_epochs" not in cfg.workflow, "validation must use interval scheduling, not explicit epochs")
    _require(int(cfg.workflow.get("val_eval_interval_anchor_epoch", 0)) == 10, "validation anchor must be epoch 10")
    _require(int(cfg.workflow.val_start_epoch) == 9, "validation must start from zero-based epoch 9")
    _require(_as_bool(cfg.solver.get("ema", False)), "EMA should stay on to match the reviewed AdaTAD protocol")


def _validate_gate(cfg: Config, *, allow_launch_unlocked: bool = False) -> None:
    gate = cfg.c3_detector_aware_full_train_gate
    _require(gate.route == "C3_STAGE2_DIVERGENT_ROUTE", "wrong route")
    _require(gate.route_variant == "DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3", "wrong route variant")
    _require(gate.stage == "Stage-2 detector-aware offline selector", "wrong stage")
    _require(_as_bool(gate.full_train_candidate), "not marked as full train candidate")
    _require(_as_bool(gate.requires_launch_gate), "launch gate must be required")
    if allow_launch_unlocked:
        _require(_as_bool(gate.launch_gate_passed), "execution config must pass launch gate")
        _require(_as_bool(gate.get("reviewed_execution_config", False)), "execution config must be reviewed")
    else:
        _require(not _as_bool(gate.launch_gate_passed), "config must be locked by default")
    _require(tuple(gate.allowed_entrypoints) == ("tools/train.py",), "only tools/train.py may be allowed")
    _require("tools/test.py" in tuple(gate.forbidden_entrypoints), "tools/test.py must be forbidden")
    _require(cfg.experiment_scope.end_to_end is False, "route must declare end_to_end=false")
    _require(cfg.experiment_scope.passes_teacher_or_value_to_forward_test is False, "teacher/value must not enter forward_test")
    _require(cfg.experiment_scope.uses_uniform_scaffold is False, "uniform scaffold must be disabled")
    _require(cfg.experiment_scope.uses_uniform_fill is False, "uniform fill must be disabled")
    _require(cfg.experiment_scope.map_claim_allowed is False, "mAP claim must remain locked in config")
    _require(cfg.baseline_comparison.full_detector_map_required_for_claim is True, "full mAP required for claim")


def _validate_ledger_file(path: str | Path, *, cfg: Config, require_exists: bool) -> None:
    path = Path(path)
    _require(str(path) and "REPLACE_WITH" not in str(path), f"ledger path is unresolved: {path}")
    if not require_exists:
        return
    _require(path.is_file(), f"ledger file missing: {path}")
    spec = _variant_spec(cfg)
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            _require(row.get("deploy_selection_ledger") is True, f"{path}:{line_no}: deploy flag is not true")
            _require(row.get("diagnostic_only") is not True, f"{path}:{line_no}: diagnostic ledger row")
            for key in FORBIDDEN_TRUE_FLAGS:
                _require(row.get(key) is not True, f"{path}:{line_no}: forbidden flag {key}=True")
            diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
            _require(int(diagnostics.get("uniform_visible_fill_count", 0) or 0) == 0, f"{path}:{line_no}: uniform fill used")
            _require(str(diagnostics.get("source_strategy")) == str(spec["strategy"]), f"{path}:{line_no}: wrong source strategy")
            _require(row.get("policy_source", diagnostics.get("policy_source")) == SOURCE, f"{path}:{line_no}: wrong policy source")
            _require(diagnostics.get("stage_label") == "Stage-2 detector-aware offline selector", f"{path}:{line_no}: missing stage label")
            _require(diagnostics.get("end_to_end") is False, f"{path}:{line_no}: end_to_end must be false")
            valid_len = int(row.get("valid_len"))
            positions = [int(item) for item in row.get("selected_positions", [])]
            _require(positions == sorted(set(positions)), f"{path}:{line_no}: positions must be sorted unique")
            _require(all(0 <= item < valid_len for item in positions), f"{path}:{line_no}: position outside valid_len")
            required = spec["require_selected_count"]
            if required is not None and valid_len >= 768:
                _require(len(positions) == int(required), f"{path}:{line_no}: selected count mismatch")
    _require(rows > 0, f"ledger file has no rows: {path}")


def validate_config(config_path: str = CONFIG_DEFAULT, *, require_ledger_files: bool = False, allow_launch_unlocked: bool = False) -> Config:
    cfg = Config.fromfile(str(config_path))
    spec = _variant_spec(cfg)
    _require(cfg.experiment_scope.selection_strategy == spec["strategy"], "experiment scope selection strategy mismatch")
    _require(cfg.c3_value_transport_source == SOURCE, "value transport source mismatch")
    _validate_gate(cfg, allow_launch_unlocked=allow_launch_unlocked)
    _validate_dataset(cfg)
    _validate_model_and_train(cfg)
    for ledger_path in (cfg.train_ledger_path, cfg.val_ledger_path, cfg.test_ledger_path):
        _validate_ledger_file(ledger_path, cfg=cfg, require_exists=require_ledger_files)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--require-ledger-files", action="store_true")
    parser.add_argument("--allow-launch-unlocked", action="store_true")
    args = parser.parse_args(argv)
    validate_config(
        args.config,
        require_ledger_files=bool(args.require_ledger_files),
        allow_launch_unlocked=bool(args.allow_launch_unlocked),
    )
    print(READY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
