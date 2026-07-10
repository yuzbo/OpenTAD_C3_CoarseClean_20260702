import argparse
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Run PhysTime-TAD on one real THUMOS feature sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def move_batch(batch, device):
    for key in ("inputs", "masks", "paired_inputs", "paired_masks"):
        if key in batch:
            batch[key] = batch[key].to(device)
    for key in ("gt_segments", "gt_labels"):
        batch[key] = [value.to(device) for value in batch[key]]
    return batch


def nonzero_gradient(parameter):
    return bool(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum().item() > 0
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.dataset.train)
    sample = dataset[0]
    batch = move_batch(collate([sample]), device)
    model = build_detector(cfg.model).to(device).train()

    losses = model(
        inputs=batch["inputs"],
        masks=batch["masks"],
        metas=batch["metas"],
        gt_segments=batch["gt_segments"],
        gt_labels=batch["gt_labels"],
        paired_inputs=batch.get("paired_inputs"),
        paired_masks=batch.get("paired_masks"),
        paired_metas=batch.get("paired_metas"),
        return_loss=True,
    )
    if not torch.isfinite(losses["cost"]):
        raise RuntimeError("real-data gate produced non-finite cost")
    losses["cost"].backward()

    groups = model.get_optim_groups(dict(lr=1.0e-4, weight_decay=0.05))
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    optimizer_coverage = len(grouped) == len(set(grouped)) and set(grouped) == set(expected)

    meta = batch["metas"][0]
    supports = torch.as_tensor(meta["phystime_support_intervals_sec"], dtype=torch.float32)
    gt = batch["gt_segments"][0]
    metadata_valid = bool(
        meta.get("gt_time_unit") == "seconds"
        and meta.get("prediction_time_unit") == "seconds"
        and meta.get("phystime_support_provenance") == "original_feature_ownership_cells"
        and supports.ndim == 2
        and supports.shape[1] == 2
        and torch.all(supports[:, 1] > supports[:, 0])
        and torch.all(gt[:, 1] >= gt[:, 0])
    )

    model.eval()
    with torch.no_grad():
        proposals, scores = model.forward_test(batch["inputs"], batch["masks"], batch["metas"])
    finite_predictions = bool(torch.isfinite(proposals[0]).all() and torch.isfinite(scores[0]).all())
    report = {
        "gate_pass": bool(metadata_valid and optimizer_coverage and finite_predictions),
        "sample_video": meta["video_name"],
        "observation_count": int(batch["masks"].sum().item()),
        "metadata_valid": metadata_valid,
        "optimizer_coverage": optimizer_coverage,
        "finite_predictions": finite_predictions,
        "paired_view_active": "paired_inputs" in batch,
        "discretization_loss_active": "discretization_loss" in losses,
        "projection_gradient_nonzero": nonzero_gradient(
            model.projection.level_attentions[0].value_proj.weight
        ),
        "classification_gradient_nonzero": nonzero_gradient(model.rpn_head.cls_head.weight),
        "regression_gradient_nonzero": nonzero_gradient(model.rpn_head.reg_head.weight),
        "endpoint_gradient_nonzero": nonzero_gradient(model.rpn_head.endpoint_head.weight),
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
    }
    report["gate_pass"] = bool(
        report["gate_pass"]
        and report["paired_view_active"]
        and report["discretization_loss_active"]
        and report["projection_gradient_nonzero"]
        and report["classification_gradient_nonzero"]
        and report["regression_gradient_nonzero"]
        and report["endpoint_gradient_nonzero"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
