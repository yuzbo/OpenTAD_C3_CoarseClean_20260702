from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from opentad.models.duca.acquisition import C3CoarseProbeActionnessSource


def _gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().item())
    return total


def run(output: Path, *, temporal_len: int = 32) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("sparse probe gate requires exactly one Slurm-visible GPU")
    device = torch.device("cuda:0")
    rows = []
    for stride in (1, 2, 3, 4):
        torch.manual_seed(3407)
        source = C3CoarseProbeActionnessSource(
            probe_model="official-action-seg",
            official_action_seg_backend="official_asformer",
            spatial_size=64,
            tcn_hidden_dim=96,
            official_num_layers=2,
            spatial_norm="groupnorm",
            hidden_output_kind="official_asformer_encoder_hidden",
            policy_hidden_gradient_scope="asformer_last_encoder_layer",
            temporal_probe_stride=stride,
            temporal_interpolation_mode="hidden_linear",
            frozen=False,
            trainable=True,
            return_hidden_features=True,
            require_hidden_features=True,
            train_split_supervised=True,
            thumos_trained=True,
            uses_labels=True,
            training_dataset="THUMOS14_train_split",
            training_supervision_scope="binary_actionness",
        ).to(device)
        source.train()
        inputs = torch.rand(1, 1, 3, temporal_len, 64, 64, device=device)
        valid = torch.ones(1, temporal_len, dtype=torch.bool, device=device)
        output_row = source(inputs, valid_mask=valid)
        hidden = output_row["coarse_hidden_features"]
        policy_hidden = output_row["policy_hidden_features"]
        if output_row["logits"].shape != (1, temporal_len):
            raise RuntimeError("sparse probe gate logits are not reconstructed to dense time")
        if hidden.shape[:2] != (1, temporal_len) or policy_hidden.shape != hidden.shape:
            raise RuntimeError("sparse probe gate hidden features are not dense and aligned")
        loss = (
            output_row["logits"].mean()
            + hidden.square().mean()
            + 0.1 * policy_hidden.square().mean()
        )
        loss.backward()
        spatial_grad = _gradient_norm(source.probe_module.spatial_stem.parameters())
        temporal_grad = _gradient_norm(source.probe_module.official_temporal.parameters())
        if not (spatial_grad > 0.0 and temporal_grad > 0.0):
            raise RuntimeError("sparse hidden interpolation did not preserve coarse-probe gradients")
        profile = output_row["compute_profile"]
        rows.append(
            {
                "stride_dense_candidates": stride,
                "interval_source_frames": 4 * stride,
                "sparse_anchor_count": int(profile["sparse_anchor_count"]),
                "dense_output_length": int(profile["dense_output_length"]),
                "estimated_macs": int(profile["estimated_macs"]),
                "spatial_gradient_norm": spatial_grad,
                "temporal_gradient_norm": temporal_grad,
                "finite": bool(
                    torch.isfinite(output_row["logits"]).all().item()
                    and torch.isfinite(hidden).all().item()
                ),
            }
        )
        del source, inputs, output_row, hidden, policy_hidden, loss
        torch.cuda.empty_cache()
    macs = [row["estimated_macs"] for row in rows]
    if not all(left > right for left, right in zip(macs, macs[1:])):
        raise RuntimeError("sparse probe estimated cost must decrease as stride grows")
    result = {
        "schema": "duca_sparse_probe_cuda_gate_v1",
        "ok": True,
        "task": "offline_temporal_action_detection",
        "device": str(device),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-len", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve(), temporal_len=args.temporal_len), indent=2))


if __name__ == "__main__":
    main()
