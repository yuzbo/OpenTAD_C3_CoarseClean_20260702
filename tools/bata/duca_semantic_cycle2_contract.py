"""Executable, data-free contract for the DUCA semantic cycle-2 arms."""
from copy import deepcopy
from pathlib import Path
import json

from opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector import PCOTMRASPreBackboneFrameSelector

ARM_NAMES = (
    "dense_placeholder", "native_uniform_fixed_k", "actionness_only_fixed_k_control",
    "actionness_boundary_fixed_k", "actionness_boundary_dynamic_k_headline",
    "direct_selector_ablation",
)

RUNTIME_BINDING_SCHEMA = "duca.runtime.binding.v1"


def runtime_binding(cfg, model, optimizer, scheduler, *, seed):
    """Return identities from constructed runtime objects, not arm metadata."""
    model_obj = model.module if hasattr(model, "module") else model
    return {
        "schema": RUNTIME_BINDING_SCHEMA,
        "detector": f"{model_obj.__class__.__module__}.{model_obj.__class__.__qualname__}",
        "loss": repr(getattr(cfg.model, "loss", None)),
        "nms": repr(getattr(cfg, "nms", getattr(cfg, "post_processing", None))),
        "evaluator": repr(getattr(cfg, "evaluation", getattr(cfg, "evaluator", None))),
        "update": f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}|{scheduler.__class__.__module__}.{scheduler.__class__.__qualname__}",
        "seed": int(seed),
    }


def validate_shared_runtime_bindings(bindings):
    if not bindings:
        raise ValueError("runtime binding list is empty")
    keys = ("detector", "loss", "nms", "evaluator", "update", "seed")
    reference = {key: bindings[0][key] for key in keys}
    for index, binding in enumerate(bindings[1:], start=1):
        if any(binding[key] != reference[key] for key in keys):
            raise ValueError(f"arm {index} changed shared runtime binding")
    return reference

def validate_manifests(manifests):
    ids = {}
    for split in ("FIT", "CAL", "HOLD"):
        values = manifests.get(split)
        if not isinstance(values, (list, tuple)) or not values or any(not isinstance(x, str) for x in values):
            raise ValueError(f"{split} manifest must contain non-empty string sample ids")
        if len(set(values)) != len(values):
            raise ValueError(f"{split} manifest contains duplicate sample ids")
        ids[split] = set(values)
    for left, right in (("FIT", "CAL"), ("FIT", "HOLD"), ("CAL", "HOLD")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise ValueError(f"manifest overlap {left}/{right}: {sorted(overlap)}")
    return {key: tuple(sorted(value)) for key, value in ids.items()}

def build_arm(cfg, arm_name):
    if arm_name not in ARM_NAMES:
        raise ValueError(f"unknown DUCA arm: {arm_name}")
    arm = deepcopy(cfg.arms[arm_name])
    selector_cfg = dict(cfg.selector)
    selector_cfg.update(arm.get("selector", {}))
    selector_cfg.pop("type", None)
    if arm_name == "dense_placeholder":
        return {"arm": arm_name, "placeholder": True, "selector": None, "data_entry": arm["data_entry"]}
    selector = PCOTMRASPreBackboneFrameSelector(**selector_cfg)
    return {"arm": arm_name, "placeholder": False, "selector": selector, "policy": selector_cfg, "data_entry": arm["data_entry"]}

def validate_deploy_entry(entry):
    forbidden = {"gt", "teacher", "raw_prediction_cache", "validation_gt", "heldout_gt"}
    bad = forbidden & set(entry)
    if bad:
        raise ValueError(f"deployment data entry contains forbidden fields: {sorted(bad)}")
