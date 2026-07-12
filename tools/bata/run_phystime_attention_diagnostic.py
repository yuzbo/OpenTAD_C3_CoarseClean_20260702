from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.models import build_detector
from opentad.models.utils.phystime_geometry import (
    build_physical_query_pyramid,
    geometry_from_metas,
    support_overlap_mass,
)
from tools.bata.analyze_phystime_performance_drop import summarize_attention_rows


SCHEMA_VERSION = "phystime_attention_checkpoint_diagnostic_v1"


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit trained PhysTime attention concentration.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def strip_distributed_prefix(state_dict):
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


def choose_indices(length, count):
    count = max(1, min(int(count), int(length)))
    return sorted(set(np.linspace(0, length - 1, count, dtype=np.int64).tolist()))


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("PhysTime attention diagnostics require CUDA")
    device = torch.device("cuda:0")

    cfg = Config.fromfile(str(Path(args.config).resolve()))
    cfg.model.backbone.custom.pretrain = str(Path(args.pretrain).resolve())
    dataset = build_dataset(cfg.dataset[args.split])
    sample_indices = choose_indices(len(dataset), args.max_samples)

    model = build_detector(cfg.model).to(device).eval()
    checkpoint = torch.load(str(Path(args.checkpoint).resolve()), map_location=device)
    state_dict = checkpoint.get("state_dict_ema", checkpoint.get("state_dict"))
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has neither state_dict_ema nor state_dict")
    model.load_state_dict(strip_distributed_prefix(state_dict), strict=True)

    level_count = len(model.projection.level_attentions)
    accumulators = [
        {
            "weights": [],
            "mass": [],
            "total_logits": [],
            "content_logits": [],
            "relative_logits": [],
            "query_norms": [],
            "key_norms": [],
            "output_temporal_std": [],
            "adjacent_cosine": [],
        }
        for _ in range(level_count)
    ]
    observation_norms = []
    observation_temporal_std = []
    score_values = []
    proposal_counts = []
    sample_videos = []

    for sample_index in sample_indices:
        batch = collate([dataset[sample_index]])
        inputs = batch["inputs"].to(device, non_blocking=False)
        masks = batch["masks"].to(device, non_blocking=False)
        metas = batch["metas"]
        sample_videos.append(str(metas[0]["video_name"]))

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(cfg.solver.get("amp", False))):
            raw_observations = model._extract_observations(inputs, masks)
        observations = raw_observations.float().transpose(1, 2)
        valid_observations = observations[0, masks[0]]
        observation_norms.extend(valid_observations.norm(dim=-1).cpu().tolist())
        observation_temporal_std.append(
            float(valid_observations.std(dim=0, unbiased=False).mean().cpu().item())
        )

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            geometry = geometry_from_metas(metas, masks, dtype=torch.float32, device=device)
            query_pyramid = build_physical_query_pyramid(
                geometry["duration_sec"],
                geometry["domain_start_sec"],
                geometry["domain_end_sec"],
                base_spacing_sec=model.projection.base_spacing_sec,
                num_levels=model.projection.num_levels,
            )
            features = []
            level_masks = []
            level_geometry = []
            for level_index, (attention, query_geometry) in enumerate(
                zip(model.projection.level_attentions, query_pyramid)
            ):
                mass = support_overlap_mass(
                    geometry["ownership_intervals_sec"],
                    query_geometry["intervals_sec"],
                    geometry["valid_mask"],
                )
                queries = attention.query_embedding(
                    query_geometry["centers_sec"],
                    query_geometry["widths_sec"],
                    geometry["duration_sec"],
                )
                keys = attention.key_proj(observations)
                content_logits = torch.einsum("bqd,bkd->bqk", queries, keys) / math.sqrt(
                    attention.attention_channels
                )
                safe_width = query_geometry["widths_sec"][:, :, None].clamp_min(attention.eps)
                signed_offset = (
                    geometry["timestamps_sec"][:, None, :] - query_geometry["centers_sec"][:, :, None]
                ) / safe_width
                support_width = (
                    geometry["ownership_intervals_sec"][..., 1]
                    - geometry["ownership_intervals_sec"][..., 0]
                )
                relative_features = torch.stack(
                    (
                        signed_offset,
                        signed_offset.abs(),
                        support_width[:, None, :] / safe_width,
                        torch.log1p(signed_offset.abs()),
                    ),
                    dim=-1,
                )
                relative_logits = attention.relative_time_mlp(relative_features).squeeze(-1)
                total_logits = content_logits + relative_logits
                output, output_mask, diagnostics = attention(
                    observations, geometry, query_geometry
                )

                acc = accumulators[level_index]
                acc["weights"].append(diagnostics["attention_weights"][0].cpu().numpy())
                acc["mass"].append(mass[0].cpu().numpy())
                acc["total_logits"].append(total_logits[0].cpu().numpy())
                acc["content_logits"].append(content_logits[0].cpu().numpy())
                acc["relative_logits"].append(relative_logits[0].cpu().numpy())
                acc["query_norms"].extend(queries[0, output_mask[0]].norm(dim=-1).cpu().tolist())
                acc["key_norms"].extend(keys[0, geometry["valid_mask"][0]].norm(dim=-1).cpu().tolist())
                valid_output = output[0, output_mask[0]]
                acc["output_temporal_std"].append(
                    float(valid_output.std(dim=0, unbiased=False).mean().cpu().item())
                )
                if valid_output.shape[0] > 1:
                    adjacent = F.cosine_similarity(valid_output[:-1], valid_output[1:], dim=-1)
                    acc["adjacent_cosine"].extend(adjacent.cpu().tolist())

                features.append(output.transpose(1, 2))
                level_masks.append(output_mask)
                info = dict(query_geometry)
                info["domain_valid_mask"] = query_geometry["valid_mask"]
                info["valid_mask"] = output_mask
                info["coverage_sec"] = diagnostics["coverage_sec"]
                level_geometry.append(info)

            proposals, scores = model.rpn_head.forward_test(
                tuple(features), tuple(level_masks), tuple(level_geometry)
            )
            proposal_counts.append(int(proposals[0].shape[0]))
            score_values.extend(scores[0].max(dim=-1).values.cpu().tolist())

    level_reports = []
    for level_index, acc in enumerate(accumulators):
        weights = np.concatenate(acc["weights"], axis=0)
        mass = np.concatenate(acc["mass"], axis=0)
        total_logits = np.concatenate(acc["total_logits"], axis=0)
        content_logits = np.concatenate(acc["content_logits"], axis=0)
        relative_logits = np.concatenate(acc["relative_logits"], axis=0)
        total_report = summarize_attention_rows(
            weights=weights, mass=mass, logits=total_logits
        )
        content_report = summarize_attention_rows(
            weights=weights, mass=mass, logits=content_logits
        )
        relative_report = summarize_attention_rows(
            weights=weights, mass=mass, logits=relative_logits
        )
        level_reports.append(
            {
                "level": level_index,
                "spacing_sec": float(model.projection.base_spacing_sec * (2**level_index)),
                "attention": total_report,
                "content_logit_span": content_report["covered_logit_span"],
                "relative_logit_span": relative_report["covered_logit_span"],
                "query_norm": distribution(acc["query_norms"]),
                "key_norm": distribution(acc["key_norms"]),
                "output_temporal_std": distribution(acc["output_temporal_std"]),
                "adjacent_cosine": distribution(acc["adjacent_cosine"]),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "state_dict_source": "state_dict_ema" if "state_dict_ema" in checkpoint else "state_dict",
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "sample_indices": sample_indices,
        "sample_videos": sample_videos,
        "observation_norm": distribution(observation_norms),
        "observation_temporal_std": distribution(observation_temporal_std),
        "proposal_count": distribution(proposal_counts),
        "max_class_score": distribution(score_values),
        "levels": level_reports,
    }


def main(argv=None):
    args = parse_args(argv)
    report = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
