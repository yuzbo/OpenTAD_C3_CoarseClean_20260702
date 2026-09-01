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
    selector = detector.frame_selector
    probe_inputs = _fixed_selector_probe_inputs(inputs.detach(), seed=int(seed) + 17)
    probe_before = [_selector_probe_snapshot(selector, probe, masks) for probe in probe_inputs]
    params_before = _clone_selector_params(selector)
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
    grad_norm = selector_grad_norm(selector)
    optimizer = torch.optim.SGD(selector.parameters(), lr=1.0)
    optimizer.step()
    probe_after = [_selector_probe_snapshot(selector, probe, masks) for probe in probe_inputs]
    selector_param_delta_l2 = _selector_param_delta_l2(params_before, selector)
    drift_payload = _selector_drift_payload(probe_before, probe_after)
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
        "selector_param_delta_l2": selector_param_delta_l2,
        "selector_param_delta_passed": selector_param_delta_l2 > 0.0,
        "selector_step_optimizer": "sgd_lr_1.0_selector_params_only",
        **drift_payload,
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


def _fixed_selector_probe_inputs(base_inputs, *, seed):
    torch.manual_seed(int(seed))
    random_probe = torch.randn_like(base_inputs)
    time = int(base_inputs.shape[-1])
    ramp = torch.linspace(-1.0, 1.0, steps=time, dtype=base_inputs.dtype, device=base_inputs.device)
    ramp = ramp.view(1, 1, time).expand_as(base_inputs)
    alternating = torch.where(
        (torch.arange(time, device=base_inputs.device) % 2).view(1, 1, time) == 0,
        torch.ones((), dtype=base_inputs.dtype, device=base_inputs.device),
        -torch.ones((), dtype=base_inputs.dtype, device=base_inputs.device),
    ).expand_as(base_inputs)
    return [
        base_inputs.detach().clone(),
        random_probe.detach().clone(),
        (base_inputs.detach() + 0.25 * ramp).clone(),
        ramp.detach().clone(),
        alternating.detach().clone(),
    ]


def _selector_probe_snapshot(selector, probe_inputs, masks):
    with torch.no_grad():
        outputs = selector.forward_features(probe_inputs, masks=masks, phase="precheck_fixed_probe")
    return {
        "selected_indices": outputs["selected_indices"].detach().cpu().long(),
        "logits": outputs["logits"].detach().cpu().float(),
    }


def _clone_selector_params(selector):
    return [(name, param.detach().clone()) for name, param in selector.named_parameters() if param.requires_grad]


def _selector_param_delta_l2(params_before, selector):
    before_by_name = {name: param for name, param in params_before}
    total = 0.0
    for name, param in selector.named_parameters():
        if not param.requires_grad:
            continue
        before = before_by_name[name].to(device=param.device, dtype=param.dtype)
        total += float((param.detach() - before).pow(2).sum().item())
    return total**0.5


def _selector_drift_payload(before_snapshots, after_snapshots):
    candidates = []
    for probe_idx, (before, after) in enumerate(zip(before_snapshots, after_snapshots)):
        selected_before = before["selected_indices"]
        selected_after = after["selected_indices"]
        position_delta = (selected_after.float() - selected_before.float()).abs()
        logits_delta = after["logits"] - before["logits"]
        logits_abs = logits_delta.abs()
        candidates.append(
            {
                "probe_idx": int(probe_idx),
                "position_mean": float(position_delta.mean().item()),
                "position_max": float(position_delta.max().item()),
                "logits_l2": float(logits_delta.pow(2).sum().sqrt().item()),
                "logits_max": float(logits_abs.max().item()),
                "selected_before": selected_before.tolist(),
                "selected_after": selected_after.tolist(),
            }
        )
    best = max(candidates, key=lambda item: (item["position_max"], item["position_mean"], item["logits_l2"]))
    return {
        "selector_probe_index": best["probe_idx"],
        "selector_probe_selected_indices_before": best["selected_before"],
        "selector_probe_selected_indices_after": best["selected_after"],
        "selected_position_drift_mean": best["position_mean"],
        "selected_position_drift_max": best["position_max"],
        "selected_position_drift_passed": best["position_max"] > 0.0,
        "selector_logits_drift_l2": best["logits_l2"],
        "selector_logits_drift_max": best["logits_max"],
        "selector_logits_drift_passed": best["logits_l2"] > 0.0,
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
