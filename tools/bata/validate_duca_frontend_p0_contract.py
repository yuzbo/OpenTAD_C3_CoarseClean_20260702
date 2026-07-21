from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from mmengine.config import Config

from opentad.duca_loss_contract import DUCA_LOSS_WEIGHT_DEFAULTS


SCHEMA_VERSION = "duca_frontend_p0_contract_v1"
ACTIVE_LOSSES = {"actionness", "transition", "transition_boundary"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"DUCA frontend P0 contract failed: {message}")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
    ).strip()


def validate_config(config_path: str | Path) -> dict[str, Any]:
    resolved = Path(config_path).resolve()
    _require(resolved.is_file(), f"config is missing: {resolved}")
    cfg = Config.fromfile(str(resolved))
    selector = cfg.model.frame_selector
    weights: Mapping[str, float] = selector.loss_weights

    _require(bool(cfg.model.selector_train_only), "selector_train_only must be true")
    _require(
        bool(cfg.model.selector_train_only_skip_detector),
        "the detector must be skipped during P0",
    )
    _require(selector.detector_gradient_mode == "none", "detector gradient must be disabled")
    _require(bool(selector.strict_loss_contract), "strict_loss_contract must be true")
    _require(
        set(weights) == set(DUCA_LOSS_WEIGHT_DEFAULTS),
        "the explicit loss inventory is incomplete",
    )
    _require(
        all(float(weights[key]) > 0.0 for key in ACTIVE_LOSSES),
        "all three declared P0 objectives must be active",
    )
    _require(
        all(
            float(value) == 0.0
            for key, value in weights.items()
            if key not in ACTIVE_LOSSES
        ),
        "an undeclared P0 objective has nonzero weight",
    )
    _require(
        selector.actionness_loss_mode == "class_balanced_mean",
        "actionness must use positive/negative class means",
    )
    _require(
        float(selector.auxiliary_hidden_gradient_scale) == 0.0,
        "transition supervision must not rewrite coarse hidden features",
    )
    _require(
        selector.actionness_source_cfg.spatial_norm == "groupnorm",
        "the spatial stem must use padding-invariant GroupNorm",
    )
    _require(bool(cfg.optimizer.paramwise), "frontend optimizer must use explicit groups")
    _require("backbone" not in cfg.optimizer, "frontend optimizer leaked a detector backbone group")
    _require(
        float(cfg.solver.clip_grad_norm) <= 0.0,
        "global gradient clipping would couple coarse and selector objectives",
    )
    _require(cfg.dataset.val is None and cfg.dataset.test is None, "P0 must not consume evaluation splits")
    _require(cfg.workflow.val_eval_interval == -1, "P0 validation mAP must be disabled")

    root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "status": "frontend_p0_contract_passed",
        "task": "offline_temporal_action_detection",
        "git_commit": _git_commit(root),
        "config_path": str(resolved),
        "config_sha256": _sha256(resolved),
        "detector_executed": False,
        "detector_trained": False,
        "test_subset_consumed": False,
        "loss_weights": {key: float(weights[key]) for key in sorted(weights)},
        "active_losses": sorted(ACTIVE_LOSSES),
        "actionness_loss_mode": str(selector.actionness_loss_mode),
        "auxiliary_hidden_gradient_scale": float(
            selector.auxiliary_hidden_gradient_scale
        ),
        "spatial_norm": str(selector.actionness_source_cfg.spatial_norm),
        "optimizer": {
            "type": str(cfg.optimizer.type),
            "paramwise": bool(cfg.optimizer.paramwise),
            "contains_backbone_group": "backbone" in cfg.optimizer,
            "global_gradient_clipping_enabled": float(cfg.solver.clip_grad_norm)
            > 0.0,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the strict DUCA frontend P0 contract.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P0 contract evidence: {output}")
    payload = validate_config(args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
