from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config

from opentad.models import build_detector
from opentad.models.selectors.truetime_joint_selector import selector_grad_norm
from opentad.models.utils.truetime_geometry import TrueTimeMap, inverse_map_prediction_segments


def run_smoke(config_path, seed=13):
    cfg = Config.fromfile(str(config_path))
    smoke_selector_cfg = cfg.truetime_detector_path_smoke_model.frame_selector
    selected_count = int(smoke_selector_cfg.selected_count)
    dense_len = int(smoke_selector_cfg.dense_len)
    if selected_count < 2:
        raise ValueError("selected_count must be at least 2 for the roundtrip/inverse-map smoke")
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

    detector = build_detector(cfg.truetime_detector_path_smoke_model)
    detector.train()
    detector.zero_grad(set_to_none=True)
    inputs = torch.randn(2, int(cfg.truetime_detector_path_smoke_model.in_channels), dense_len, 2, 2, requires_grad=True)
    masks = torch.ones(2, dense_len, dtype=torch.bool)
    losses = detector.forward_train(inputs=inputs, masks=masks, metas=[{}, {}])
    losses["cost"].backward()
    out = detector.last_selector_outputs
    hard_forward_values_passed = bool(
        torch.allclose(detector.last_selected_inputs.detach(), detector.last_hard_selected_inputs.detach(), atol=1.0e-6)
    )
    grad_norm = selector_grad_norm(detector.frame_selector)
    selected_input_grad_norm = grad_norm
    loss_keys = sorted(key for key in losses if key != "cost")
    actionformer_payload = _run_actionformer_path_smoke(cfg, dense_len=dense_len, seed=int(seed) + 101)

    return {
        "route_variant": "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3",
        "stage": "stage3_4_experimental_smoke",
        "proof_source": "registered_detector_forward_train_cost_backward",
        "geometry_roundtrip_passed": geometry_roundtrip_passed,
        "prediction_inverse_map_passed": prediction_inverse_map_passed,
        "selected_input_hard_forward_values_passed": hard_forward_values_passed,
        "selected_input_st_gradient_passed": selected_input_grad_norm > 0.0,
        "selected_input_selector_grad_norm": selected_input_grad_norm,
        "detector_loss_selector_grad_passed": grad_norm > 0.0,
        "detector_loss_selector_grad_norm": grad_norm,
        "loss_keys": loss_keys,
        "selector_grad_norm": grad_norm,
        "selector_grad_nonzero": grad_norm > 0.0,
        **actionformer_payload,
        "selected_count_mean": float(out["selected_count_mean"].detach().cpu().item()),
        "selected_count_std": float(out["selected_count_std"].detach().cpu().item()),
        "entropy": float(out["entropy"].detach().cpu().item()),
        "selector_grad_path": out["selector_grad_path"],
        "limitations": [
            "smoke proof only; no mAP, runtime, paper, or deploy claim",
            "hard top-k detector path is paired with a relaxed temporal surrogate for gradient evidence",
            "evaluator mAP semantics are not modified",
        ],
    }


def _run_actionformer_path_smoke(cfg, *, dense_len, seed):
    if "truetime_actionformer_path_smoke_model" not in cfg:
        raise ValueError("config must define truetime_actionformer_path_smoke_model")
    torch.manual_seed(int(seed))
    detector = build_detector(cfg.truetime_actionformer_path_smoke_model)
    detector.train()
    detector.zero_grad(set_to_none=True)
    inputs = torch.randn(2, 3, int(dense_len), requires_grad=True)
    masks = torch.ones(2, int(dense_len), dtype=torch.bool)
    metas = [
        {"video_name": "truetime_actionformer_smoke_0"},
        {"video_name": "truetime_actionformer_smoke_1"},
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
    actionformer_loss_keys = sorted(key for key in losses if key != "cost")
    finite_loss_values = {
        key: float(value.detach().cpu().item())
        for key, value in losses.items()
        if key != "cost" and torch.is_tensor(value) and value.ndim == 0
    }
    return {
        "actionformer_proof_source": "opentad_actionformer_forward_train_cost_backward",
        "actionformer_detector_loss_selector_grad_passed": grad_norm > 0.0,
        "actionformer_detector_loss_selector_grad_norm": grad_norm,
        "actionformer_loss_keys": actionformer_loss_keys,
        "actionformer_loss_values": finite_loss_values,
        "actionformer_selected_axis_smoke": True,
        "actionformer_physical_grid_smoke": False,
        "sparse_distill_adapter_ready": True,
        "sparse_distill_claim_allowed": False,
        "sparse_distill_map_claim_allowed": False,
        "sparse_distill_proof_source": "fail_closed_sparse_detector_distillation_adapter",
        "actionformer_smoke_limitations": [
            "uses the real OpenTAD ActionFormer forward_train/projection/rpn_head path",
            "uses a tiny selected-axis synthetic batch; it is not a full AdaTAD mAP run",
            "physical-grid dense-axis assignment remains a separate full-train/precheck gate",
        ],
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
    parser.add_argument("--config", default="configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke.py")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    payload = run_smoke(config_path=args.config, seed=args.seed)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
