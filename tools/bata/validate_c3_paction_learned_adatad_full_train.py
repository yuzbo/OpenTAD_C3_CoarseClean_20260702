from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from mmengine.config import Config


CONFIG_DEFAULT = "configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py"
READY = "C3_PACTION_LEARNED_LEDGER_FULL_TRAIN_GATE_PASS"
FIXED_STRATEGY = "learned_paction_gap_loss_value"
DYNAMIC_STRATEGY = "learned_paction_gap_loss_dynamic_budget"
GAS_VT_SOURCE = "learned_paction_gas_vt_policy_checkpoint"
PACTION_CHECKPOINT_SOURCE = "learned_paction_gap_loss_policy_checkpoint"
LATTICE_ROUTE_VARIANT = "C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT"
LATTICE_RADIUS_ROUTE_VARIANT = "C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT_ADAPTIVE_RADIUS"
VARIANT_SPECS = {
    "learned_fixed_384": dict(target_len=384, require_selected_count=384, strategy=FIXED_STRATEGY, source=PACTION_CHECKPOINT_SOURCE, route_variant="C3_PACTION_LEARNED_STRICT_LEDGER"),
    "learned_fixed_768": dict(target_len=768, require_selected_count=768, strategy=FIXED_STRATEGY, source=PACTION_CHECKPOINT_SOURCE, route_variant="C3_PACTION_LEARNED_STRICT_LEDGER"),
    "learned_dynamic": dict(target_len=768, require_selected_count=None, strategy=DYNAMIC_STRATEGY, source=PACTION_CHECKPOINT_SOURCE, route_variant="C3_PACTION_LEARNED_STRICT_LEDGER"),
    "paction_lattice_radius_score_only_move25": dict(target_len=384, require_selected_count=384, strategy="paction_lattice_radius_score_only_move25", source=PACTION_CHECKPOINT_SOURCE, route_variant=LATTICE_RADIUS_ROUTE_VARIANT),
    "paction_lattice_replace_score_only_move25": dict(target_len=384, require_selected_count=384, strategy="paction_lattice_replace_score_only_move25", source=PACTION_CHECKPOINT_SOURCE, route_variant=LATTICE_ROUTE_VARIANT),
    "paction_lattice_replace_score_only_move50": dict(target_len=384, require_selected_count=384, strategy="paction_lattice_replace_score_only_move50", source=PACTION_CHECKPOINT_SOURCE, route_variant=LATTICE_ROUTE_VARIANT),
    "paction_lattice_replace_score_only_move75": dict(target_len=384, require_selected_count=384, strategy="paction_lattice_replace_score_only_move75", source=PACTION_CHECKPOINT_SOURCE, route_variant=LATTICE_ROUTE_VARIANT),
    "paction_lattice_replace_score_only_no_protect": dict(target_len=384, require_selected_count=384, strategy="paction_lattice_replace_score_only_no_protect", source=PACTION_CHECKPOINT_SOURCE, route_variant=LATTICE_ROUTE_VARIANT),
    "gas_vt_fixed_384": dict(target_len=384, require_selected_count=384, strategy="gas_vt_fixed_384", source=GAS_VT_SOURCE, route_variant="C3_GAS_VT_STRICT_LEDGER"),
    "gas_vt_fixed_768": dict(target_len=768, require_selected_count=768, strategy="gas_vt_fixed_768", source=GAS_VT_SOURCE, route_variant="C3_GAS_VT_STRICT_LEDGER"),
    "gas_vt_dynamic": dict(target_len=768, require_selected_count=None, strategy="gas_vt_dynamic", source=GAS_VT_SOURCE, route_variant="C3_GAS_VT_STRICT_LEDGER"),
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


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
    variant = str(cfg.gas_vt_ledger_variant if "gas_vt_ledger_variant" in cfg else cfg.paction_ledger_variant)
    _require(variant in VARIANT_SPECS, f"unknown paction_ledger_variant={variant}")
    return dict(VARIANT_SPECS[variant])


def _gate_cfg(cfg: Config) -> Any:
    if "c3_gas_vt_ledger_full_train_gate" in cfg:
        return cfg.c3_gas_vt_ledger_full_train_gate
    return cfg.c3_paction_learned_ledger_full_train_gate


def _validate_loader(name: str, dataset_cfg: Any, expected_ledger_path: str, cfg: Config) -> None:
    spec = _variant_spec(cfg)
    loader = _find_loadframes(dataset_cfg.pipeline)
    _require(loader.get("method") == "bata_value_transport_ledger_subsample", f"{name}: wrong LoadFrames method")
    _require(loader.get("method_base") == "sliding_window", f"{name}: method_base must be sliding_window")
    _require(int(loader.get("target_len")) == int(spec["target_len"]), f"{name}: wrong target_len")
    _require(int(loader.get("scale_factor", 1)) == 1, f"{name}: scale_factor must be 1")
    _require(
        loader.get("bata_value_transport_require_selected_count") == spec["require_selected_count"],
        f"{name}: wrong require_selected_count",
    )
    _require(_as_bool(loader.get("bata_value_transport_allow_short_valid_ratio_count")), f"{name}: short-tail ratio gate off")
    _require(_as_bool(loader.get("bata_value_transport_require_deployable")), f"{name}: deployable ledger not required")
    _require(not _as_bool(loader.get("bata_value_transport_allow_missing_fallback")), f"{name}: missing fallback must be off")
    _require(_as_bool(loader.get("remap_gt_to_selected_axis")), f"{name}: selected-axis GT remap must be on")
    _require(loader.get("bata_value_transport_ledger_path") == expected_ledger_path, f"{name}: wrong ledger path")
    _require(loader.get("bata_value_transport_source") == cfg.c3_value_transport_source, f"{name}: wrong ledger source")
    _require(
        loader.get("bata_value_transport_source") == spec["source"],
        f"{name}: loader source must identify the expected policy checkpoint",
    )


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
    expected_eval_interval = int(os.environ.get("C3_PACTION_ADATAD_EXPECT_VAL_EVAL_INTERVAL", os.environ.get("C3_PACTION_ADATAD_VAL_EVAL_INTERVAL", "10")))
    expected_eval_anchor = int(
        os.environ.get(
            "C3_PACTION_ADATAD_EXPECT_VAL_EVAL_INTERVAL_ANCHOR_EPOCH",
            os.environ.get("C3_PACTION_ADATAD_VAL_EVAL_INTERVAL_ANCHOR_EPOCH", str(expected_eval_interval)),
        )
    )
    expected_val_start = int(
        os.environ.get(
            "C3_PACTION_ADATAD_EXPECT_VAL_START_EPOCH",
            os.environ.get("C3_PACTION_ADATAD_VAL_START_EPOCH", str(max(0, expected_eval_interval - 1))),
        )
    )
    _require(int(cfg.window_size) == target_len, "selected window_size mismatch")
    _require(int(cfg.dense_window_size) == 768, "dense_window_size must be 768")
    _require(int(cfg.model.backbone.backbone.total_frames) == target_len, "backbone total_frames mismatch")
    _require(int(cfg.model.projection.max_seq_len) == target_len, "projection max_seq_len mismatch")
    _require("frame_selector" not in repr(cfg.model), "AdaTAD detector config must not include online frame_selector")
    _require("pc_ot_mras_reader" not in repr(cfg.model), "AdaTAD detector config must not include reader")
    _require(not _as_bool(cfg.inference.load_from_raw_predictions), "raw prediction loading must be off")
    _require(not _as_bool(cfg.inference.save_raw_prediction), "raw prediction saving must be off")
    _require(cfg.evaluation.type == "mAP", "evaluation must be mAP")
    _require(cfg.evaluation.subset == "validation", "evaluation subset must be validation")
    _require(
        cfg.evaluation.ground_truth_filename == cfg.annotation_path,
        "evaluation ground_truth_filename must follow THUMOS14_ANNOTATION_PATH",
    )
    _require(int(cfg.workflow.end_epoch) == 60, "formal full train must run 60 epochs")
    _require(cfg.workflow.get("max_train_iters", None) is None, "full train must not cap train iterations")
    _require(
        int(cfg.workflow.val_eval_interval) == expected_eval_interval,
        f"validation interval must be {expected_eval_interval}",
    )
    _require("val_eval_epochs" not in cfg.workflow, "validation must use interval scheduling, not explicit epochs")
    _require(
        int(cfg.workflow.get("val_eval_interval_anchor_epoch", 0)) == expected_eval_anchor,
        f"validation anchor must be epoch {expected_eval_anchor}",
    )
    _require(
        int(cfg.workflow.val_start_epoch) == expected_val_start,
        f"validation must start from zero-based epoch {expected_val_start}",
    )
    _require(_as_bool(cfg.solver.get("ema", False)), "EMA should stay on to match the reviewed AdaTAD protocol")


def _validate_gate(cfg: Config, *, allow_launch_unlocked: bool = False) -> None:
    spec = _variant_spec(cfg)
    gate = _gate_cfg(cfg)
    _require(gate.route == "C3_MAINLINE_OPTIMIZATION", "wrong route")
    _require(gate.route_variant == spec["route_variant"], "wrong route variant")
    _require(_as_bool(gate.full_train_candidate), "not marked as full train candidate")
    _require(_as_bool(gate.requires_launch_gate), "launch gate must be required")
    if allow_launch_unlocked:
        _require(_as_bool(gate.launch_gate_passed), "execution config must pass launch gate")
        _require(_as_bool(gate.get("reviewed_execution_config", False)), "execution config must be reviewed")
    else:
        _require(not _as_bool(gate.launch_gate_passed), "config must be locked by default")
    _require(tuple(gate.allowed_entrypoints) == ("tools/train.py",), "only tools/train.py may be allowed")
    _require("tools/test.py" in tuple(gate.forbidden_entrypoints), "tools/test.py must be forbidden by this launcher")
    text = repr(dict(experiment_scope=cfg.experiment_scope, model=cfg.model, dataset=cfg.dataset, inference=cfg.inference)).lower()
    for forbidden in ("bh_sdc", "event_surprise", "boundary_microscope", "frame_token_hybrid", "uniform_scaffold", "uniform_fill"):
        _require(forbidden not in text or forbidden in ("uniform_scaffold", "uniform_fill"), f"forbidden route token present: {forbidden}")
    for forbidden in ("load_from_raw_predictions': true", "save_raw_prediction': true"):
        _require(forbidden not in text, f"forbidden raw prediction flag present: {forbidden}")
    _require(cfg.experiment_scope.uses_uniform_scaffold is False, "uniform scaffold must be disabled")
    _require(cfg.experiment_scope.uses_uniform_fill is False, "uniform fill must be disabled")


def _expected_short_count(required: int, *, valid_len: int, dense_len: int) -> int:
    if valid_len >= dense_len:
        return int(required)
    return max(1, min(int(required), int(valid_len), int(math.ceil(valid_len * float(required) / float(dense_len)))))


def _validate_paction_provenance(path: Path, line_no: int, diagnostics: dict[str, Any]) -> None:
    provenance = diagnostics.get("p_action_provenance")
    _require(isinstance(provenance, dict), f"{path}:{line_no}: missing p_action provenance")
    _require(_nonempty_text(provenance.get("p_action_source")), f"{path}:{line_no}: missing p_action provenance source")
    model_markers = (
        provenance.get("probe_model"),
        provenance.get("matrix_model_id"),
        provenance.get("official_action_seg_backend"),
    )
    _require(
        any(_nonempty_text(item) for item in model_markers),
        f"{path}:{line_no}: missing p_action provenance model marker",
    )
    _require(provenance.get("no_gt_generation") is True, f"{path}:{line_no}: p_action provenance must not use GT generation")
    for key in (
        "uses_teacher",
        "uses_oracle",
        "uses_cache",
        "uses_prediction_cache",
        "uses_raw_prediction",
        "prediction_uses_gt",
    ):
        _require(provenance.get(key) is False, f"{path}:{line_no}: p_action provenance must set {key}=false")


def _validate_ledger_file(path: str | Path, *, cfg: Config, require_exists: bool) -> None:
    path = Path(path)
    _require(str(path) and "REPLACE_WITH" not in str(path), f"ledger path is unresolved: {path}")
    if not require_exists:
        return
    _require(path.is_file(), f"ledger file missing: {path}")
    spec = _variant_spec(cfg)
    seen: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            sample_id = row.get("sample_id")
            _require(sample_id and "|" in sample_id, f"{path}:{line_no}: invalid sample_id")
            _require(sample_id not in seen, f"{path}:{line_no}: duplicate sample_id {sample_id}")
            seen.add(sample_id)
            _require(row.get("deploy_selection_ledger") is True, f"{path}:{line_no}: deploy flag is not true")
            _require(row.get("diagnostic_only") is not True, f"{path}:{line_no}: diagnostic ledger row")
            for key in FORBIDDEN_TRUE_FLAGS:
                _require(row.get(key) is not True, f"{path}:{line_no}: forbidden flag {key}=True")
            diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
            _require(int(diagnostics.get("uniform_visible_fill_count", 0) or 0) == 0, f"{path}:{line_no}: uniform fill used")
            _require(str(diagnostics.get("source_strategy")) == str(spec["strategy"]), f"{path}:{line_no}: wrong source strategy")
            _require(
                row.get("policy_source", diagnostics.get("policy_source")) == spec["source"],
                f"{path}:{line_no}: missing learned checkpoint policy_source",
            )
            checkpoint_sha256 = row.get("policy_checkpoint_sha256", diagnostics.get("policy_checkpoint_sha256"))
            _require(isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64, f"{path}:{line_no}: missing checkpoint sha256")
            if isinstance(cfg.c3_value_transport_config_hash, str) and len(cfg.c3_value_transport_config_hash) == 64:
                _require(
                    checkpoint_sha256 == cfg.c3_value_transport_config_hash,
                    f"{path}:{line_no}: checkpoint sha256 does not match config hash",
                )
            _validate_paction_provenance(path, line_no, diagnostics)
            valid_len = int(row.get("valid_len"))
            dense_len = int(row.get("dense_len"))
            target_len = int(row.get("target_len"))
            positions = [int(item) for item in row.get("selected_positions", [])]
            _require(dense_len == 768, f"{path}:{line_no}: dense_len must be 768")
            _require(target_len == int(spec["target_len"]), f"{path}:{line_no}: target_len mismatch")
            _require(positions == sorted(set(positions)), f"{path}:{line_no}: positions must be sorted unique")
            _require(len(positions) == int(row.get("selected_count")), f"{path}:{line_no}: selected_count mismatch")
            _require(0 < len(positions) <= target_len, f"{path}:{line_no}: invalid selected_count")
            _require(all(0 <= item < valid_len for item in positions), f"{path}:{line_no}: position outside valid_len")
            required = spec["require_selected_count"]
            if required is not None:
                expected = _expected_short_count(int(required), valid_len=valid_len, dense_len=dense_len)
                _require(len(positions) == expected, f"{path}:{line_no}: expected {expected} selected positions")
    _require(rows > 0, f"ledger file has no rows: {path}")


def validate_config(config_path: str = CONFIG_DEFAULT, *, require_ledger_files: bool = False, allow_launch_unlocked: bool = False) -> Config:
    cfg = Config.fromfile(str(config_path))
    spec = _variant_spec(cfg)
    _require(cfg.experiment_scope.selection_strategy == spec["strategy"], "experiment scope selection strategy mismatch")
    _require(cfg.c3_value_transport_source == spec["source"], "value transport source mismatch")
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
