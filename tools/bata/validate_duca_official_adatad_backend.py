from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DEFAULT = "configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py"
OFFICIAL_BASE_CONFIG = "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _as_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    return value


def _load(path: str | Path) -> Config:
    return Config.fromfile(str(path))


def _config_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def validate_config(config_path: str = CONFIG_DEFAULT) -> dict[str, Any]:
    config_path = str(config_path)
    cfg = _load(config_path)
    official = _load(str(ROOT / OFFICIAL_BASE_CONFIG))
    contract = cfg.duca_online_main_contract
    selector = cfg.model.frame_selector
    head = cfg.model.rpn_head
    model_text = repr(cfg.model).lower()
    full_text = _config_text(config_path)

    _require(contract.official_adatad_backend is True, "contract must declare official_adatad_backend=True")
    _require(contract.main_method_candidate is True, "contract must declare main_method_candidate=True")
    _require(contract.diagnostic_only is False, "main official backend config must not be diagnostic_only")
    _require(contract.no_ledger_decision is True, "main config must be online/no-ledger")
    _require(contract.pre_backbone_plugin is True, "DUCA must be declared as pre-backbone plugin")
    _require(contract.changes_detector_head is False, "main config must not change detector head")
    _require(contract.changes_loss_assignment is False, "main config must not change detector assignment/loss")
    _require(contract.physical_grid_actionformer_required is False, "main config must not require physical-grid head")
    _require(cfg.model.type == "ActionFormer", "official AdaTAD backend must keep ActionFormer detector")
    _require(selector.type == "DucaOnlineFrameSelector", "main config must use DucaOnlineFrameSelector")
    _require(int(selector.budget) <= 384, "selector budget must be <=384")
    _require(int(selector.dense_window_size) == 768, "candidate dense window must stay 768")
    _require(selector.coordinate_space == "original_time", "selected positions must be original-time")
    _require(selector.detector_output_coordinate_space == "selected_axis_index", "detector outputs must be selected-axis before wrapper remap")
    _require(selector.remap_gt_to_selected_axis is True, "official head path must remap GT to selected-axis")
    _require(selector.no_ledger_decision is True, "selector must be online/no-ledger")
    _require(selector.forbid_ledger is True, "selector must forbid ledger payload")
    _require(head.type == "ActionFormerHead", "official backend must use ActionFormerHead")
    _require(_as_plain(head) == _as_plain(official.model.rpn_head), "ActionFormerHead config must match official base")
    _require("physical_grid_actionformer" not in head, "main config must not enable physical-grid ActionFormer")
    _require("bata_value_transport" not in full_text, "main config must not use value-transport ledger sampling")
    _require("ledger_path" not in model_text, "model must not include ledger_path")
    _require("load_from_raw_predictions': true" not in repr(cfg.inference).lower(), "raw prediction loading must be disabled")
    _require("save_raw_prediction': true" not in repr(cfg.inference).lower(), "raw prediction saving must be disabled")
    _require(int(cfg.window_size) == 384, "detector-consumed DUCA sequence length must be 384")
    _require(int(cfg.dense_window_size) == 768, "candidate dense observation length must be 768")
    _require(int(cfg.chunk_num) == 24, "VideoMAE chunk count must match 384/16")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "VideoMAE backend must consume selected 384 frames")
    _require(int(cfg.model.projection.max_seq_len) == 384, "projection max_seq_len must match selected budget")
    _require(cfg.dataset.train.pipeline[2].method == "random_trunc", "train LoadFrames must stay online random_trunc, not ledger")
    _require(cfg.dataset.val.pipeline[2].method == "sliding_window", "val LoadFrames must stay online sliding_window, not ledger")
    _require(cfg.dataset.test.pipeline[2].method == "sliding_window", "test LoadFrames must stay online sliding_window, not ledger")

    source_cfg = selector.actionness_source_cfg
    for key in ("uses_labels", "uses_teacher", "uses_gt", "uses_prediction_cache"):
        _require(source_cfg.get(key) is False, f"actionness source must set {key}=False")
    for key in ("no_train_gt", "no_teacher", "no_oracle", "no_raw_prediction_cache", "no_gt_generation"):
        _require(source_cfg.get(key) is True, f"actionness source must set {key}=True")

    return {
        "ok": True,
        "config_path": config_path,
        "official_base_config": OFFICIAL_BASE_CONFIG,
        "official_adatad_backend": True,
        "model_type": str(cfg.model.type),
        "selector_type": str(selector.type),
        "rpn_head_type": str(head.type),
        "rpn_head_matches_official_base": True,
        "physical_grid_actionformer_enabled": False,
        "uses_ledger_for_decision": False,
        "budget_lte_384": True,
        "budget": int(selector.budget),
        "dense_window_size": int(selector.dense_window_size),
        "detector_consumed_length": int(cfg.window_size),
        "changes_detector_head": False,
        "changes_loss_assignment": False,
        "pre_backbone_plugin": True,
        "selected_positions_unit": str(selector.selected_positions_unit),
        "detector_output_coordinate_space": str(selector.detector_output_coordinate_space),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = validate_config(args.config)
    except Exception as exc:
        summary = {
            "ok": False,
            "config_path": str(args.config),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return 1
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
