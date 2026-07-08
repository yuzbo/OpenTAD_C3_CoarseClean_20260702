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

CONFIG_DEFAULT = "configs/adatad/thumos/duca_must_dynamic_official_adatad_backend_full_train.py"
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


def validate_config(
    config_path: str = CONFIG_DEFAULT,
    *,
    max_budget: int = 384,
    require_online_c3_actionness: bool = True,
) -> dict[str, Any]:
    config_path = str(config_path)
    max_budget = int(max_budget)
    cfg = _load(config_path)
    official = _load(str(ROOT / OFFICIAL_BASE_CONFIG))
    contract = cfg.duca_must_dynamic_contract
    selector = cfg.model.frame_selector
    head = cfg.model.rpn_head
    model_text = repr(cfg.model).lower()
    full_text = _config_text(config_path)

    _require("duca_online_budget" not in full_text, "dynamic main config must not use external budget override env")
    _require(contract.official_adatad_backend is True, "contract must declare official_adatad_backend=True")
    _require(contract.main_method_candidate is True, "contract must declare main_method_candidate=True")
    _require(contract.diagnostic_only is False, "dynamic main config must not be diagnostic_only")
    _require(contract.dynamic_budget is True, "contract must declare dynamic_budget=True")
    _require(contract.budget_policy == "prefix_marginal_utility_stop", "unexpected dynamic budget policy")
    _require(contract.external_budget_override_allowed is False, "dynamic main config must forbid external budget override")
    _require(contract.forced_budget_curve is False, "dynamic main config must not be a forced-budget curve")
    _require(contract.no_ledger_decision is True, "main config must be online/no-ledger")
    _require(contract.pre_backbone_plugin is True, "DUCA-MUST must be declared as pre-backbone plugin")
    _require(contract.changes_detector_head is False, "main config must not change detector head")
    _require(contract.changes_loss_assignment is False, "main config must not change detector assignment/loss")
    _require(contract.actual_variable_length_detector is False, "current backend must declare padded cap detector input")
    _require(contract.runtime_flops_claim_allowed is False, "padded cap backend must not claim runtime FLOPs savings")
    _require(cfg.model.type == "ActionFormer", "official AdaTAD backend must keep ActionFormer detector")
    _require(selector.type == "DucaOnlineFrameSelector", "main config must use DucaOnlineFrameSelector")
    _require(selector.budget is None, "dynamic main selector must not set a fixed budget")
    _require(selector.budget_mode == "dynamic_must", "selector must use dynamic_must")
    _require(selector.allow_external_budget_override is False, "selector must forbid external budget override")
    _require(int(selector.budget_max) <= max_budget, f"selector budget_max must be <={max_budget}")
    _require(int(selector.budget_min) > 0, "selector budget_min must be positive")
    _require(int(selector.budget_min) <= int(selector.budget_max), "selector budget_min must be <= budget_max")
    _require(int(selector.target_budget) <= int(selector.budget_max), "target_budget must be <= budget_max")
    _require(int(selector.budget_multiple) > 0, "budget_multiple must be positive")
    _require(
        (int(selector.budget_max) - int(selector.budget_min)) % int(selector.budget_multiple) == 0,
        "budget_multiple must divide budget_max - budget_min",
    )
    _require(int(selector.dense_window_size) == 768, "candidate dense window must stay 768")
    _require(selector.coordinate_space == "original_time", "selected positions must be original-time")
    _require(selector.detector_output_coordinate_space == "selected_axis_index", "detector outputs must be selected-axis")
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
    _require(int(cfg.window_size) == int(selector.budget_max), "current padded detector length must match budget_max")
    _require(int(cfg.dense_window_size) == 768, "candidate dense observation length must be 768")
    _require(int(cfg.window_size) % 16 == 0, "detector-consumed cap length must be divisible by 16")
    _require(int(cfg.chunk_num) == int(cfg.window_size) // 16, "VideoMAE chunk count must match cap/16")
    _require(
        int(cfg.model.backbone.backbone.total_frames) == int(cfg.window_size),
        "VideoMAE backend must consume the selected cap length",
    )
    _require(int(cfg.model.projection.max_seq_len) == int(cfg.window_size), "projection max_seq_len must match cap")
    _require(cfg.dataset.train.pipeline[2].method == "random_trunc", "train LoadFrames must stay online random_trunc")
    _require(cfg.dataset.val.pipeline[2].method == "sliding_window", "val LoadFrames must stay online sliding_window")
    _require(cfg.dataset.test.pipeline[2].method == "sliding_window", "test LoadFrames must stay online sliding_window")

    source_cfg = selector.actionness_source_cfg
    if require_online_c3_actionness:
        _require(
            contract.actionness_source == "online_trainable_c3_coarse_probe",
            "main source must be online C3 coarse probe",
        )
        _require(source_cfg.type == "C3CoarseProbeActionnessSource", "main source must be online C3 coarse probe module")
        _require(
            source_cfg.probe_model in {"mobilenetv3", "temporal-tcn", "official-action-seg", "matrix-zoo"},
            "unsupported coarse probe model",
        )
        _require(source_cfg.get("trainable") is True, "coarse probe must be trainable for the main end-to-end config")
        _require(source_cfg.get("frozen") is False, "coarse probe must not be frozen for the main end-to-end config")
        _require(contract.coarse_probe_joint_trainable is True, "contract must declare joint-trainable coarse probe")
        _require(contract.runtime_profile_available is True, "contract must expose compute/latency profiling")
        for key in ("uses_labels", "uses_teacher", "uses_gt", "uses_prediction_cache"):
            _require(source_cfg.get(key) is False, f"actionness source must set {key}=False")

    summary = {
        "ok": True,
        "config_path": config_path,
        "official_base_config": OFFICIAL_BASE_CONFIG,
        "official_adatad_backend": True,
        "model_type": str(cfg.model.type),
        "selector_type": str(selector.type),
        "rpn_head_type": str(head.type),
        "rpn_head_matches_official_base": True,
        "dynamic_budget": True,
        "budget_policy": str(contract.budget_policy),
        "budget_min": int(selector.budget_min),
        "budget_max": int(selector.budget_max),
        "budget_target": int(selector.target_budget),
        "budget_multiple": int(selector.budget_multiple),
        "budget_max_lte_384": int(selector.budget_max) <= 384,
        "budget_lte_max": int(selector.budget_max) <= max_budget,
        "external_budget_override_allowed": False,
        "uses_env_budget_override": False,
        "forced_budget_curve": False,
        "runtime_flops_claim_allowed": False,
        "actual_variable_length_detector": False,
        "uses_ledger_for_decision": False,
        "detector_consumed_cap_length": int(cfg.window_size),
        "dense_window_size": int(cfg.dense_window_size),
        "changes_detector_head": False,
        "changes_loss_assignment": False,
        "pre_backbone_plugin": True,
        "selected_positions_unit": str(selector.selected_positions_unit),
        "detector_output_coordinate_space": str(selector.detector_output_coordinate_space),
    }
    if require_online_c3_actionness:
        summary.update(
            {
                "actionness_source": str(contract.actionness_source),
                "coarse_probe_model": str(source_cfg.probe_model),
                "coarse_probe_tcn_variant": str(source_cfg.get("tcn_variant", "")),
                "coarse_probe_joint_trainable": bool(contract.coarse_probe_joint_trainable),
                "runtime_profile_available": bool(contract.runtime_profile_available),
            }
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--max-budget", type=int, default=384)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = validate_config(args.config, max_budget=int(args.max_budget))
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
