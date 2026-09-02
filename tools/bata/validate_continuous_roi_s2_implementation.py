from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmengine.config import Config

from tools.bata.continuous_roi_s2_contract import (
    canonical_sha256,
    finalize_self_hash,
    load_protocol,
    validate_protocol,
)


IMPLEMENTATION_AUDIT_SCHEMA = "continuous_roi_s2_implementation_static_audit_v1"
# This validator is the pre-training one-step implementation gate. The separate
# FULL200 3x3 matrix is validated by continuous_roi_s2_v3_full200_compute.py;
# the two protocols intentionally have different test-split and authorization rules.
REFERENCE_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "e2e_thumos_videomae_s_768x1_160_adapter.py"
)
CONFIGS = {
    "D160": ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "continuous_roi_s2_d160_videomae_s_768x1_adapter.py",
    "G96": ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "continuous_roi_s2_g96_videomae_s_768x1_adapter.py",
    "U128": ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "continuous_roi_s2_u128_videomae_s_768x1_adapter.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pipeline_types(cfg) -> list[str]:
    return [str(step["type"]) for step in cfg]


def _strip_u128_wrapper_fields(model: dict) -> dict:
    model = copy.deepcopy(model)
    custom = model["backbone"]["custom"]
    for key in tuple(custom):
        if (
            key == "wrapper_type"
            or key.startswith("native_crop_")
            or key.startswith("continuous_roi_")
        ):
            custom.pop(key)
    return model


def _validate_gate(cfg, family: str) -> None:
    gate = cfg.continuous_roi_s2_gate
    _require(gate.route == "spatial-zoom-continuous-roi-s2", f"{family} route")
    _require(gate.precheck_only is True, f"{family} precheck gate")
    for key in (
        "allow_detector_training",
        "allow_tools_train",
        "allow_tools_test",
        "allow_detector_map",
        "official_test_open_allowed",
        "learned_crop_policy_allowed",
        "paper_claim_allowed",
    ):
        _require(gate[key] is False, f"{family} must fail closed on {key}")
    _require(int(gate.selector_parameters) == 0, f"{family} contains a selector")


def _validate_workflow(cfg, family: str) -> None:
    workflow = cfg.workflow
    _require(int(workflow.end_epoch) == 60, f"{family} epoch contract")
    _require(
        int(workflow.max_amp_retries_per_batch) == 8,
        f"{family} AMP retry contract",
    )
    _require(workflow.fail_on_skipped_update is True, f"{family} skip contract")
    _require(
        workflow.schedule_and_ema_on_success_only is True,
        f"{family} scheduler/EMA contract",
    )
    _require(
        bool(workflow.require_successful_update_hook) == (family == "U128"),
        f"{family} successful-update hook contract",
    )


def _validate_pipeline(cfg, family: str) -> dict:
    expected_prefix = [
        "PrepareVideoInfo",
        "mmaction.DecordInit",
        "LoadFrames",
        "mmaction.DecordDecode",
    ]
    records = {}
    for split_name in ("train", "val"):
        split = cfg.dataset[split_name]
        _require(split.subset_name == "training", f"{family} uses test subset")
        types = _pipeline_types(split.pipeline)
        _require(types[:4] == expected_prefix, f"{family} decode prefix changed")
        transform = split.pipeline[4]
        if family == "U128":
            _require(
                transform.type == "ContinuousRoiSourceViews"
                and int(transform.global_size) == 96
                and int(transform.required_source_height) == 180
                and int(transform.required_source_width) == 320,
                f"{family} source/global transform changed",
            )
        else:
            expected_size = 160 if family == "D160" else 96
            _require(
                transform.type == "FullFrameLetterboxView"
                and int(transform.output_size) == expected_size,
                f"{family} letterbox transform changed",
            )
        forbidden = [
            value
            for value in types[5:]
            if "Resize" in value or "Crop" in value
        ]
        _require(not forbidden, f"{family} has hidden post-source spatial transforms")
        records[split_name] = types
    _require(cfg.dataset.test is None, f"{family} materializes official test")
    return records


def validate_implementation() -> dict:
    protocol_audit = validate_protocol(load_protocol())
    reference = Config.fromfile(str(REFERENCE_CONFIG))
    configs = {name: Config.fromfile(str(path)) for name, path in CONFIGS.items()}
    pipeline_audits = {}
    config_hashes = {}
    reference_model = copy.deepcopy(reference.model.to_dict())
    for family, cfg in configs.items():
        _validate_gate(cfg, family)
        _validate_workflow(cfg, family)
        pipeline_audits[family] = _validate_pipeline(cfg, family)
        candidate_model = copy.deepcopy(cfg.model.to_dict())
        if family == "U128":
            _require(
                _strip_u128_wrapper_fields(candidate_model) == reference_model,
                "U128 changed the inherited AdaTAD-derived detector surface",
            )
        else:
            _require(
                candidate_model == reference_model,
                f"{family} changed the inherited detector surface",
            )
        _require(
            cfg.post_processing.to_dict() == reference.post_processing.to_dict(),
            f"{family} changed post-processing",
        )
        config_hashes[family] = canonical_sha256(cfg.to_dict())

    u128 = configs["U128"]
    custom = u128.model.backbone.custom
    _require(
        custom.wrapper_type == "continuous_roi_common_support_u128"
        and int(custom.continuous_roi_knots) == 12
        and int(custom.native_crop_chunk_num) == 48
        and int(custom.continuous_roi_frames_per_clip) == 16
        and int(custom.native_crop_intermediate_length) == 384
        and int(custom.native_crop_output_length) == 768,
        "U128 temporal or wrapper contract changed",
    )
    optimizer_custom = [dict(value) for value in u128.optimizer.backbone.custom]
    _require(
        [value["name"] for value in optimizer_custom]
        == ["adapter", "fusion", "global_aux_head", "local_aux_head"],
        "U128 optimizer does not cover the registered trainable surfaces",
    )
    _require(
        u128.optimizer.backbone.exclude == ["backbone"],
        "U128 pretrained core exclusion changed",
    )
    protocol = load_protocol()
    _require(
        protocol["models"]["new_u128_parameters"]
        == {
            "auxiliary_heads": 15400,
            "fusion": 594049,
            "policy_head": 0,
            "total": 609449,
        },
        "protocol U128 parameter contract changed",
    )
    return finalize_self_hash(
        {
            "schema_version": IMPLEMENTATION_AUDIT_SCHEMA,
            "status": "PASS",
            "protocol_sha256": protocol_audit["protocol_sha256"],
            "config_hashes": config_hashes,
            "pipeline_audits": pipeline_audits,
            "detector_model_surface_matches_reference": True,
            "post_processing_matches_reference": True,
            "u128_selector_parameters": 0,
            "u128_new_parameters": 609449,
            "official_test_materialized": False,
            "training_authorized": False,
            "full_model_cuda_gate_required": True,
        },
        "implementation_audit_sha256",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the fail-closed Continuous-RoI S2 implementation"
    )
    parser.parse_args(argv)
    print(json.dumps(validate_implementation(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
