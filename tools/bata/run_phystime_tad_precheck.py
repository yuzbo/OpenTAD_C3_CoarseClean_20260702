import argparse
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PhysTime-TAD Gate 0B model precheck")
    parser.add_argument(
        "--config",
        default="configs/adatad/thumos/phystime_tad_i3d_feature_gate0b.py",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def synthetic_batch(in_channels, device):
    observation_count = 12
    duration = 6.0
    centers = [0.25 + 0.5 * index for index in range(observation_count)]
    supports = [[0.5 * index, 0.5 * (index + 1)] for index in range(observation_count)]
    inputs = torch.randn(1, in_channels, observation_count, device=device)
    masks = torch.ones((1, observation_count), dtype=torch.bool, device=device)
    meta = {
        "video_name": "phystime_synthetic_gate0b",
        "duration": duration,
        "fps": 24.0,
        "snippet_stride": 12,
        "offset_frames": 0,
        "phystime_timestamps_sec": centers,
        "phystime_support_intervals_sec": supports,
        "phystime_duration_sec": duration,
        "phystime_domain_start_sec": 0.0,
        "phystime_domain_end_sec": duration,
        "phystime_support_provenance": "synthetic_explicit_support",
        "gt_time_unit": "seconds",
        "prediction_time_unit": "seconds",
        "irregular_native_axis": True,
        "remap_gt_to_selected_axis": False,
        "gt_remapped_to_selected_axis": False,
    }
    return inputs, masks, [meta]


def has_nonzero_finite_gradient(parameter):
    return bool(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum().item() > 0
    )


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)
    model = build_detector(cfg.model).to(device).train()
    inputs, masks, metas = synthetic_batch(int(cfg.model.projection.in_channels), device)
    gt_segments = [torch.tensor([[0.75, 5.25]], dtype=torch.float32, device=device)]
    gt_labels = [torch.tensor([3], dtype=torch.long, device=device)]

    paired_inputs = inputs + 0.05 * torch.randn_like(inputs)
    paired_metas = [dict(metas[0])]
    losses = model.forward_train(
        inputs,
        masks,
        metas,
        gt_segments,
        gt_labels,
        paired_inputs=paired_inputs,
        paired_masks=masks.clone(),
        paired_metas=paired_metas,
    )
    if not torch.isfinite(losses["cost"]):
        raise RuntimeError("PhysTime precheck produced a non-finite total loss")
    losses["cost"].backward()

    optim_groups = model.get_optim_groups(dict(lr=1.0e-4, weight_decay=0.05))
    grouped_ids = [id(parameter) for group in optim_groups for parameter in group["params"]]
    expected_ids = [
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    optimizer_coverage = len(grouped_ids) == len(set(grouped_ids)) and set(grouped_ids) == set(expected_ids)

    model.eval()
    with torch.no_grad():
        proposals, scores = model.forward_test(inputs, masks, metas)
    finite_predictions = bool(torch.isfinite(proposals[0]).all() and torch.isfinite(scores[0]).all())

    report = {
        "precheck_pass": bool(optimizer_coverage and finite_predictions),
        "model_type": model.__class__.__name__,
        "device": str(device),
        "prediction_time_unit": metas[0]["prediction_time_unit"],
        "optimizer_coverage": optimizer_coverage,
        "projection_gradient_nonzero": has_nonzero_finite_gradient(
            model.projection.level_attentions[0].value_proj.weight
        ),
        "classification_gradient_nonzero": has_nonzero_finite_gradient(model.rpn_head.cls_head.weight),
        "regression_gradient_nonzero": has_nonzero_finite_gradient(model.rpn_head.reg_head.weight),
        "endpoint_gradient_nonzero": has_nonzero_finite_gradient(model.rpn_head.endpoint_head.weight),
        "proposal_count": int(proposals[0].shape[0]),
        "finite_predictions": finite_predictions,
        "uses_selector": hasattr(model, "frame_selector"),
        "uses_ledger": False,
        "paired_discretization_loss_active": "discretization_loss" in losses,
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
    }
    report["precheck_pass"] = bool(
        report["precheck_pass"]
        and report["projection_gradient_nonzero"]
        and report["classification_gradient_nonzero"]
        and report["regression_gradient_nonzero"]
        and report["endpoint_gradient_nonzero"]
        and report["paired_discretization_loss_active"]
        and not report["uses_selector"]
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["precheck_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
