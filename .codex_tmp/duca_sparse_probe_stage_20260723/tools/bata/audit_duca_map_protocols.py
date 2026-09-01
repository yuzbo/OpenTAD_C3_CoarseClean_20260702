"""Audit which DUCA experiment outputs are comparable official THUMOS mAP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from mmengine.config import Config

from tools.bata.duca_p0_evaluation import (
    EXPECTED_TIOU_THRESHOLDS,
    official_evaluator_identity,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/adatad/thumos"
FORMAL_CONFIGS = {
    "two_stage_exact_uniform": (
        "duca_two_stage_exact_uniform_fixed384_official60.py"
    ),
    "gaussian_matched_g0": (
        "duca_global_curriculum_g0_no_feedback_fixed384_official60.py"
    ),
    "boundary_burst_r2q3_g0": (
        "duca_boundary_burst_g0_no_feedback_fixed384_official60.py"
    ),
    "boundary_burst_r4q5_g0": (
        "duca_boundary_burst_r4q5_g0_no_feedback_fixed384_official60.py"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_formal_configs() -> list[dict[str, object]]:
    # Learned configs only require these values to resolve their immutable
    # initialization contract. The audit does not open the checkpoint.
    placeholders = {
        "DUCA_FRONTEND_CHECKPOINT": "audit-placeholder.pth",
        "DUCA_FRONTEND_CHECKPOINT_SHA256": "0" * 64,
        "DUCA_FRONTEND_CHECKPOINT_EPOCH": "19",
    }
    for key, value in placeholders.items():
        if not os.environ.get(key):
            os.environ[key] = value
    rows = []
    for variant, filename in FORMAL_CONFIGS.items():
        path = (CONFIG_ROOT / filename).resolve()
        cfg = Config.fromfile(str(path))
        evaluation = cfg.evaluation.to_dict()
        thresholds = [float(value) for value in evaluation["tiou_thresholds"]]
        row = {
            "variant": variant,
            "config": str(path),
            "config_sha256": _sha256(path),
            "formal_protocol": str(cfg.workflow.formal_protocol),
            "epochs": int(cfg.workflow.end_epoch),
            "terminal_checkpoint": "epoch_59.pth/state_dict_ema",
            "test_subset": str(cfg.dataset.test.subset_name),
            "evaluation_type": str(evaluation["type"]),
            "evaluation_subset": str(evaluation["subset"]),
            "tiou_thresholds": thresholds,
            "annotation": str(evaluation["ground_truth_filename"]),
            "class_map": str(cfg.dataset.test.class_map),
            "official_validation_comparable": (
                str(cfg.workflow.formal_protocol)
                == "duca_selected_axis_optimization_v1"
                and int(cfg.workflow.end_epoch) == 60
                and str(cfg.dataset.test.subset_name) == "validation"
                and str(evaluation["type"]) == "mAP"
                and str(evaluation["subset"]) == "validation"
                and thresholds == EXPECTED_TIOU_THRESHOLDS
            ),
        }
        rows.append(row)
    annotations = {row["annotation"] for row in rows}
    class_maps = {row["class_map"] for row in rows}
    if len(annotations) != 1 or len(class_maps) != 1:
        raise RuntimeError("formal DUCA arms do not share evaluation data")
    if not all(row["official_validation_comparable"] for row in rows):
        raise RuntimeError("one or more formal DUCA arms are not comparable")
    return rows


def build_audit() -> dict[str, object]:
    return {
        "schema": "duca_map_protocol_audit_v1",
        "task": "offline_temporal_action_detection",
        "official_evaluator": official_evaluator_identity(),
        "official_tiou_thresholds": EXPECTED_TIOU_THRESHOLDS,
        "formal_official60_arms": _load_formal_configs(),
        "experiment_classes": [
            {
                "stage": "R0_holdout_replay",
                "uses_opentad_map_arithmetic": True,
                "official_validation_comparable": False,
                "paper_table_role": "diagnostic_only",
                "reason": (
                    "training_internal_holdout with a detector checkpoint already "
                    "trained on the source training subset"
                ),
            },
            {
                "stage": "R2_frontend_pretraining_and_quality",
                "uses_opentad_map_arithmetic": False,
                "official_validation_comparable": False,
                "paper_table_role": "training_or_diagnostic_only",
                "reason": "frontend losses and selection-quality proxies are not mAP",
            },
            {
                "stage": "R3_official60_terminal_evaluation",
                "uses_opentad_map_arithmetic": True,
                "official_validation_comparable": True,
                "paper_table_role": "main_result",
                "reason": (
                    "full THUMOS validation, OpenTAD mAP, frozen tIoU thresholds, "
                    "and terminal epoch-59 EMA"
                ),
            },
            {
                "stage": "R1_R3_full_model_gates",
                "uses_opentad_map_arithmetic": False,
                "official_validation_comparable": False,
                "paper_table_role": "mechanism_gate_only",
                "reason": "one-step gradient and optimizer checks do not report mAP",
            },
            {
                "stage": "R4_hard_swap_alignment",
                "uses_opentad_map_arithmetic": False,
                "official_validation_comparable": False,
                "paper_table_role": "mechanism_diagnostic_only",
                "reason": "signed utility alignment is not the final TAD metric",
            },
            {
                "stage": "R5_terminal_matrix_cells",
                "uses_opentad_map_arithmetic": True,
                "official_validation_comparable": None,
                "protocol_eligible_for_official_validation": True,
                "paper_table_role": "eligible_only_after_terminal_artifact_validation",
                "reason": (
                    "the frozen protocol targets full validation and terminal epoch-59 "
                    "EMA, but each unfinished cell requires runtime evidence validation"
                ),
            },
            {
                "stage": "R5_cost_profiles_and_aggregate",
                "uses_opentad_map_arithmetic": False,
                "official_validation_comparable": False,
                "paper_table_role": "cost_or_collation_only",
                "reason": "cost profiles are not accuracy measurements",
            },
        ],
        "bootstrap_contract": {
            "required_for_official_point_map": False,
            "role": "optional_uncertainty_analysis_only",
            "may_block_official60_training": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_audit()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
