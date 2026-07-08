from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.validate_duca_official_adatad_backend import validate_config as validate_official_config


CONFIG_DEFAULT = "configs/adatad/thumos/duca_online_x3d_official_adatad_backend_full_train.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load(path: str | Path) -> Config:
    return Config.fromfile(str(path))


def _validate_x3d_external_path(cfg: Config, *, require_jsonl_exists: bool) -> dict[str, Any]:
    selector = cfg.model.frame_selector
    contract = cfg.duca_online_main_contract
    _require(contract.external_actionness_source == "train_free_x3d_jsonl", "contract must declare train_free_x3d_jsonl")
    _require(contract.requires_external_actionness is True, "contract must require external actionness")
    _require(contract.x3d_downstream_detector_full_train is True, "contract must declare X3D downstream detector full train")
    _require(contract.uses_offline_deploy_selection_ledger is False, "X3D path must not use selection ledger")
    _require(selector.external_actionness_meta_key == "duca_external_p_action", "selector must read external p_action")
    _require(
        selector.external_actionness_logits_meta_key == "duca_external_actionness_logits",
        "selector must read external actionness logits",
    )
    _require(
        selector.external_actionness_provenance_meta_key == "duca_external_actionness_provenance",
        "selector must read external provenance",
    )
    _require(
        selector.external_actionness_source_meta_key == "duca_external_actionness_source",
        "selector must read external source name",
    )
    _require(selector.require_external_actionness is True, "selector must fail closed without external actionness")
    for split in ("train", "val", "test"):
        pipeline = cfg.dataset[split].pipeline
        _require(pipeline[2].type == "LoadFrames", f"{split} pipeline must load dense online frames first")
        _require(pipeline[3].type == "DucaExternalActionnessFromJsonl", f"{split} pipeline must inject X3D JSONL actionness")
        _require(
            pipeline[3].actionness_jsonl == cfg.duca_x3d_actionness_jsonl,
            f"{split} pipeline JSONL path must match config variable",
        )
        meta_keys = set(pipeline[-1].meta_keys)
        for key in (
            "duca_external_p_action",
            "duca_external_actionness_logits",
            "duca_external_actionness_provenance",
            "duca_external_actionness_source",
        ):
            _require(key in meta_keys, f"{split} Collect meta_keys must include {key}")
    if require_jsonl_exists:
        _require(Path(cfg.duca_x3d_actionness_jsonl).expanduser().exists(), "DUCA_X3D_ACTIONNESS_JSONL file must exist")
    return {
        "external_actionness_source": "train_free_x3d_jsonl",
        "requires_external_actionness": True,
        "x3d_actionness_jsonl": str(cfg.duca_x3d_actionness_jsonl),
        "x3d_downstream_detector_full_train": True,
        "uses_offline_deploy_selection_ledger": False,
    }


def validate_config(config_path: str = CONFIG_DEFAULT, *, max_budget: int = 384, require_jsonl_exists: bool = False) -> dict[str, Any]:
    summary = validate_official_config(config_path, max_budget=max_budget)
    cfg = _load(config_path)
    summary.update(_validate_x3d_external_path(cfg, require_jsonl_exists=bool(require_jsonl_exists)))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--max-budget", type=int, default=int(os.environ.get("DUCA_VALIDATOR_MAX_BUDGET", "384")))
    parser.add_argument(
        "--require-jsonl-exists",
        action="store_true",
        default=os.environ.get("DUCA_X3D_REQUIRE_JSONL_EXISTS", "0") == "1",
    )
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = validate_config(
            args.config,
            max_budget=int(args.max_budget),
            require_jsonl_exists=bool(args.require_jsonl_exists),
        )
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
