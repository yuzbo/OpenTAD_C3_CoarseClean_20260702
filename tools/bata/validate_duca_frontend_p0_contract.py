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
    transition_updates_coarse = bool(
        cfg.duca_transition_only_contract.get(
            "transition_supervision_updates_coarse_representation", False
        )
    )
    auxiliary_hidden_gradient_scale = float(
        selector.auxiliary_hidden_gradient_scale
    )
    policy_hidden_gradient_scale = float(selector.policy_hidden_gradient_scale)
    if transition_updates_coarse:
        _require(
            0.0 < auxiliary_hidden_gradient_scale <= 0.25,
            "adaptive transition supervision requires a protected scale in (0,0.25]",
        )
        _require(
            0.0 < policy_hidden_gradient_scale <= 0.05,
            "adaptive boundary coverage requires a protected policy scale in (0,0.05]",
        )
        _require(
            selector.actionness_source_cfg.policy_hidden_gradient_scope
            == "asformer_last_encoder_layer",
            "boundary coverage may update only the final ASFormer encoder layer",
        )
        _require(
            cfg.duca_transition_only_contract.get(
                "transition_distribution_updates"
            )
            == "asformer_last_encoder_layer_only",
            "transition distribution may update only the final ASFormer encoder layer",
        )
    else:
        _require(
            auxiliary_hidden_gradient_scale == 0.0,
            "detached transition supervision requires zero coarse-hidden gradient",
        )
        _require(
            policy_hidden_gradient_scale == 0.0,
            "detached boundary coverage requires zero policy-hidden gradient",
        )
    _require(
        selector.actionness_source_cfg.spatial_norm == "groupnorm",
        "the spatial stem must use padding-invariant GroupNorm",
    )
    official_asformer_sha256 = str(
        cfg.duca_transition_only_contract.get(
            "official_asformer_source_normalized_lf_sha256", ""
        )
    ).lower()
    _require(
        len(official_asformer_sha256) == 64
        and all(char in "0123456789abcdef" for char in official_asformer_sha256),
        "the official ASFormer normalized-LF SHA256 must be declared",
    )
    transition_objective = str(selector.get("transition_objective", "gaussian_mass"))
    _require(
        transition_objective in {"gaussian_mass", "boundary_burst"},
        "unsupported transition objective",
    )
    if transition_objective == "boundary_burst":
        local_bilateral_utility = bool(
            selector.get("boundary_burst_require_bilateral_offsets", False)
        )
        global_mandatory_groups = bool(
            selector.get(
                "boundary_burst_require_global_mandatory_groups",
                False,
            )
        )
        _require(
            int(selector.transition_target_radius) == 0,
            "boundary-burst supervision must use exact endpoint events",
        )
        _require(
            int(selector.transition_boundary_radius) > 0
            and float(selector.boundary_burst_quota) > 0.0,
            "boundary-burst radius/quota must be positive",
        )
        load_frames = next(
            item for item in cfg.dataset.train.pipeline if item.type == "LoadFrames"
        )
        collect = next(
            item for item in cfg.dataset.train.pipeline if item.type == "Collect"
        )
        _require(
            bool(load_frames.get("emit_boundary_validity", False)),
            "boundary-burst P0 must emit crop-boundary validity",
        )
        _require(
            "gt_boundary_validity" in collect["keys"],
            "boundary-burst P0 must collect crop-boundary validity",
        )
        _require(
            local_bilateral_utility,
            "boundary-burst P0 must keep the local center/left/right utility relaxation",
        )
        _require(
            bool(
                cfg.duca_transition_only_contract.get(
                    "local_bilateral_utility_relaxation",
                    False,
                )
            )
            is local_bilateral_utility,
            "local bilateral utility contract disagrees with the selector",
        )
        _require(
            bool(
                cfg.duca_transition_only_contract.get(
                    "global_mandatory_group_decoder",
                    False,
                )
            )
            is global_mandatory_groups,
            "global mandatory-group contract disagrees with the selector",
        )
        if global_mandatory_groups:
            _require(
                float(selector.boundary_burst_quota) >= 3.0,
                "global mandatory bilateral groups require quota >= 3",
            )
            _require(
                cfg.duca_transition_only_contract.get("hard_global_burst_support")
                == "mandatory_group_constrained_exact_k_max_hole",
                "global mandatory groups lack the constrained exact-K decoder contract",
            )
        else:
            _require(
                cfg.duca_transition_only_contract.get("hard_global_burst_support")
                == "none",
                "soft global decoder arm must not claim mandatory burst support",
            )
    _require(bool(cfg.optimizer.paramwise), "frontend optimizer must use explicit groups")
    _require("backbone" not in cfg.optimizer, "frontend optimizer leaked a detector backbone group")
    _require(
        float(cfg.solver.clip_grad_norm) <= 0.0,
        "global gradient clipping would couple coarse and selector objectives",
    )
    _require(cfg.dataset.val is None and cfg.dataset.test is None, "P0 must not consume evaluation splits")
    _require(cfg.workflow.val_eval_interval == -1, "P0 validation mAP must be disabled")

    component_lrs = {
        "coarse_trunk": float(selector.coarse_trunk_lr),
        "action_head": float(selector.action_head_lr),
        "transition_scorer": float(selector.transition_scorer_lr),
    }
    _require(
        all(value > 0.0 for value in component_lrs.values()),
        "all frontend component learning rates must be positive",
    )

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
        "component_lrs": component_lrs,
        "actionness_loss_mode": str(selector.actionness_loss_mode),
        "auxiliary_hidden_gradient_scale": auxiliary_hidden_gradient_scale,
        "policy_hidden_gradient_scale": policy_hidden_gradient_scale,
        "policy_hidden_gradient_scope": str(
            selector.actionness_source_cfg.get(
                "policy_hidden_gradient_scope", "none"
            )
        ),
        "transition_supervision_updates_coarse_representation": (
            transition_updates_coarse
        ),
        "transition_objective": transition_objective,
        "boundary_burst": (
            {
                "radius": int(selector.transition_boundary_radius),
                "quota": float(selector.boundary_burst_quota),
                "budget_fraction": float(selector.boundary_burst_budget_fraction),
                "local_bilateral_utility_relaxation": bool(
                    selector.get(
                        "boundary_burst_require_bilateral_offsets", False
                    )
                ),
                "global_mandatory_groups": bool(
                    selector.get(
                        "boundary_burst_require_global_mandatory_groups",
                        False,
                    )
                ),
                "global_decoder_contract": str(
                    cfg.duca_transition_only_contract.get(
                        "hard_global_burst_support", "none"
                    )
                ),
            }
            if transition_objective == "boundary_burst"
            else None
        ),
        "spatial_norm": str(selector.actionness_source_cfg.spatial_norm),
        "official_asformer_source_normalized_lf_sha256": (
            official_asformer_sha256
        ),
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
