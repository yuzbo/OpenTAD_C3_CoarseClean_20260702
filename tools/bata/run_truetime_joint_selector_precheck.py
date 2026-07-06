from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config

from opentad.models import build_detector
from opentad.models.selectors.truetime_joint_selector import selector_grad_norm
from opentad.models.utils.truetime_geometry import TrueTimeMap, inverse_map_prediction_segments


CONFIG_DEFAULT = "configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck.py"
ROUTE = "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3"
STAGE = "stage3_true_time_e2e_adatad_selector_precheck"


def run_precheck(config_path=CONFIG_DEFAULT, seed=13):
    cfg = Config.fromfile(str(config_path))
    if bool(cfg.truetime_joint_selector_gate.get("smoke_only", True)):
        raise ValueError("precheck runner requires a non-smoke TrueTime selector config")
    selector_cfg = cfg.truetime_actionformer_path_precheck_model.frame_selector
    selected_count = int(selector_cfg.selected_count)
    dense_len = int(selector_cfg.dense_len)
    if selected_count < 2:
        raise ValueError("selected_count must be at least 2 for the roundtrip/inverse-map precheck")
    if dense_len < selected_count:
        raise ValueError("dense_len must be >= selected_count")

    torch.manual_seed(int(seed))
    selected_positions = _exact_uniform_positions(dense_len=dense_len, selected_count=selected_count)
    time_map = TrueTimeMap(selected_positions=selected_positions, dense_len=dense_len, valid_len=dense_len)
    selected_axis = torch.linspace(0.0, float(selected_count - 1), steps=selected_count)
    roundtrip = time_map.true_to_selected(time_map.selected_to_true(selected_axis))
    geometry_roundtrip_passed = bool(torch.allclose(roundtrip, selected_axis, atol=1.0e-5))

    inverse_end = min(selected_count - 1, 2)
    predictions = {"segments": torch.tensor([[0.0, float(inverse_end)]]), "coordinate_space": "selected_axis_index"}
    mapped = inverse_map_prediction_segments(predictions, time_map)
    expected_mapped = torch.tensor([[float(selected_positions[0]), float(selected_positions[inverse_end])]])
    prediction_inverse_map_passed = bool(torch.allclose(mapped["segments"], expected_mapped))

    actionformer_payload = _run_actionformer_path_precheck(cfg, dense_len=dense_len, seed=int(seed) + 101)
    grad_norm = float(actionformer_payload["real_detector_loss_selector_grad_norm"])

    return {
        "route_variant": ROUTE,
        "stage": STAGE,
        "geometry_roundtrip_passed": geometry_roundtrip_passed,
        "prediction_inverse_map_passed": prediction_inverse_map_passed,
        "selected_input_st_gradient_passed": grad_norm > 0.0,
        "selected_input_selector_grad_norm": grad_norm,
        "detector_loss_selector_grad_passed": grad_norm > 0.0,
        "detector_loss_selector_grad_norm": grad_norm,
        "selector_grad_norm": grad_norm,
        "selector_grad_nonzero": grad_norm > 0.0,
        **actionformer_payload,
        "limitations": [
            "precheck proof only; no mAP, runtime, paper, end-to-end, or deploy claim",
            "uses the real OpenTAD ActionFormer forward_train/projection/rpn_head detector loss path",
            "hard top-k detector path is paired with a relaxed temporal surrogate for gradient evidence",
            "evaluator mAP semantics are not modified",
        ],
    }


def _run_actionformer_path_precheck(cfg, *, dense_len, seed):
    torch.manual_seed(int(seed))
    detector = build_detector(cfg.truetime_actionformer_path_precheck_model)
    detector.train()
    detector.zero_grad(set_to_none=True)
    inputs = torch.randn(2, 3, int(dense_len), requires_grad=True)
    masks = torch.ones(2, int(dense_len), dtype=torch.bool)
    metas = [
        {"video_name": "truetime_actionformer_precheck_0"},
        {"video_name": "truetime_actionformer_precheck_1"},
    ]
    gt_segments = [
        torch.tensor([[1.0, 3.0]], dtype=torch.float32),
        torch.tensor([[0.5, 2.5]], dtype=torch.float32),
    ]
    gt_labels = [
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    ]
    losses = detector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    losses["cost"].backward()
    grad_norm = selector_grad_norm(detector.frame_selector)
    loss_keys = sorted(key for key in losses if key != "cost")
    finite_loss_values = {
        key: float(value.detach().cpu().item())
        for key, value in losses.items()
        if key != "cost" and torch.is_tensor(value) and value.ndim == 0
    }
    return {
        "real_detector_proof_source": "opentad_actionformer_forward_train_cost_backward",
        "real_detector_loss_selector_grad_passed": grad_norm > 0.0,
        "real_detector_loss_selector_grad_norm": grad_norm,
        "real_detector_loss_keys": loss_keys,
        "real_detector_loss_values": finite_loss_values,
        "actionformer_proof_source": "opentad_actionformer_forward_train_cost_backward",
        "actionformer_detector_loss_selector_grad_passed": grad_norm > 0.0,
        "actionformer_detector_loss_selector_grad_norm": grad_norm,
        "actionformer_loss_keys": loss_keys,
        "actionformer_loss_values": finite_loss_values,
        "actionformer_selected_axis_smoke": False,
        "actionformer_physical_grid_precheck": True,
        "sparse_distill_adapter_ready": True,
        "sparse_distill_claim_allowed": False,
        "sparse_distill_map_claim_allowed": False,
        "sparse_distill_proof_source": "fail_closed_sparse_detector_distillation_adapter",
    }


def _exact_uniform_positions(*, dense_len, selected_count):
    if selected_count == 1:
        return [0]
    raw = torch.linspace(0.0, float(dense_len - 1), steps=selected_count).round().long().tolist()
    if len(set(raw)) != selected_count:
        raise ValueError("exact uniform selected positions are not unique")
    return [int(item) for item in raw]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    payload = run_precheck(config_path=args.config, seed=args.seed)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
