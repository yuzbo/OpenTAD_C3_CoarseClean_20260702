from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from opentad.models.duca import temporal_max_gap_hole_loss
from opentad.models.duca.structured_selection import (
    continuous_density_transport,
    exact_uniform_positions,
    global_structured_topk,
)


def _max_hole(positions: list[int], temporal_len: int) -> int:
    sentinels = [-1, *positions, int(temporal_len)]
    return max(
        right - left - 1
        for left, right in zip(sentinels[:-1], sentinels[1:])
    )


def _metrics(
    positions: list[int], boundaries: list[int], temporal_len: int
) -> dict[str, Any]:
    if not positions:
        raise ValueError("selection positions cannot be empty")
    nearest = [
        min(
            (abs(position - boundary) for boundary in boundaries),
            default=temporal_len,
        )
        for position in positions
    ]
    return {
        "selected_count": len(positions),
        "max_unselected_hole": _max_hole(positions, temporal_len),
        "mean_selected_to_boundary_distance": mean(nearest),
        "selected_within_r1": sum(distance <= 1 for distance in nearest),
        "selected_within_r2": sum(distance <= 2 for distance in nearest),
        "selected_within_r4": sum(distance <= 4 for distance in nearest),
        "selected_within_r8": sum(distance <= 8 for distance in nearest),
        "per_boundary_r4_cluster_size": [
            sum(abs(position - boundary) <= 4 for position in positions)
            for boundary in boundaries
        ],
    }


def _row_logits(row: dict[str, Any], key: str, fallback: str) -> torch.Tensor:
    payload = row.get(key, row.get(fallback))
    if payload is None:
        raise ValueError(f"missing {key!r} and fallback {fallback!r}")
    return torch.as_tensor(payload, dtype=torch.float32).reshape(1, -1)


def _decode_density(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    budget: int,
    max_unselected_hole: int | None,
    temperature: float,
    coverage_floor: float,
    smoothing_kernel: int,
    component_logits: torch.Tensor | None = None,
    mixture_logits: torch.Tensor | None = None,
):
    return continuous_density_transport(
        logits,
        valid,
        k=int(budget),
        max_unselected_hole=max_unselected_hole,
        component_logits=component_logits,
        component_mixture_logits=mixture_logits,
        temperature=float(temperature),
        coverage_floor=float(coverage_floor),
        smoothing_kernel=int(smoothing_kernel),
        training=False,
    )


def compare_sample(
    row: dict[str, Any],
    *,
    budget: int,
    hard_max_unselected_hole: int,
    soft_max_unselected_hole: int,
    temperature: float,
    coverage_floor: float,
    smoothing_kernel: int,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id", "unknown"))
    no_max_logits = _row_logits(row, "density_logits_no_max", "density_logits")
    soft_logits = _row_logits(row, "density_logits_soft_max", "density_logits")
    hard_logits = _row_logits(row, "density_logits_hard_max", "density_logits")
    lengths = {int(no_max_logits.shape[1]), int(soft_logits.shape[1]), int(hard_logits.shape[1])}
    if len(lengths) != 1:
        raise ValueError(f"{sample_id}: density logit lengths do not match")
    temporal_len = lengths.pop()
    valid_len = int(row.get("valid_len", temporal_len))
    if valid_len <= 0 or valid_len > temporal_len:
        raise ValueError(f"{sample_id}: invalid valid_len")
    valid = torch.arange(temporal_len)[None, :] < valid_len
    effective_k = min(int(budget), valid_len)
    boundaries = sorted({int(value) for value in row.get("boundary_positions", [])})

    no_max = _decode_density(
        no_max_logits,
        valid,
        budget=budget,
        max_unselected_hole=None,
        temperature=temperature,
        coverage_floor=coverage_floor,
        smoothing_kernel=smoothing_kernel,
    )
    soft_max = _decode_density(
        soft_logits,
        valid,
        budget=budget,
        max_unselected_hole=None,
        temperature=temperature,
        coverage_floor=coverage_floor,
        smoothing_kernel=smoothing_kernel,
    )
    hard_max = _decode_density(
        hard_logits,
        valid,
        budget=budget,
        max_unselected_hole=int(hard_max_unselected_hole),
        temperature=temperature,
        coverage_floor=coverage_floor,
        smoothing_kernel=smoothing_kernel,
    )
    methods = {
        "exact_uniform": exact_uniform_positions(valid_len, effective_k).tolist(),
        "structured_hard_score": global_structured_topk(
            no_max_logits[:, :valid_len],
            k=effective_k,
            max_unselected_hole=int(hard_max_unselected_hole),
            training=False,
        ).selected_positions[0].tolist(),
        "density_no_max": no_max.selected_positions[0, :effective_k].tolist(),
        "density_soft_max": soft_max.selected_positions[0, :effective_k].tolist(),
        "density_hard_max": hard_max.selected_positions[0, :effective_k].tolist(),
    }
    density_payloads = {
        "density_no_max": no_max,
        "density_soft_max": soft_max,
        "density_hard_max": hard_max,
    }

    if "mixture_component_logits" in row and "mixture_logits" in row:
        component_logits = torch.as_tensor(
            row["mixture_component_logits"], dtype=torch.float32
        ).reshape(1, -1, temporal_len)
        mixture_logits = torch.as_tensor(
            row["mixture_logits"], dtype=torch.float32
        ).reshape(1, -1)
        mixture_base = _row_logits(row, "density_logits_mixture", "density_logits")
        mixture = _decode_density(
            mixture_base,
            valid,
            budget=budget,
            max_unselected_hole=None,
            temperature=temperature,
            coverage_floor=coverage_floor,
            smoothing_kernel=smoothing_kernel,
            component_logits=component_logits,
            mixture_logits=mixture_logits,
        )
        methods["density_boundary_uncertainty_context"] = mixture.selected_positions[
            0, :effective_k
        ].tolist()
        density_payloads["density_boundary_uncertainty_context"] = mixture

    method_records = {}
    for name, positions in methods.items():
        record = {
            "selected_positions": positions,
            "metrics": _metrics(positions, boundaries, valid_len),
        }
        if name in density_payloads:
            decoded = density_payloads[name]
            record["density_probabilities"] = decoded.density[0, :valid_len].tolist()
            record["continuous_positions"] = decoded.continuous_positions[
                0, :effective_k
            ].tolist()
        method_records[name] = record

    soft_penalty = temporal_max_gap_hole_loss(
        soft_max.soft_occupancy,
        valid,
        max_unselected_hole=int(soft_max_unselected_hole),
        min_window_mass=1.0,
    )
    return {
        "sample_id": sample_id,
        "valid_len": valid_len,
        "boundary_positions": boundaries,
        "soft_max_gap_target": int(soft_max_unselected_hole),
        "soft_max_gap_penalty": float(soft_penalty.detach().cpu().item()),
        "methods": method_records,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(records[0]["methods"]) if records else []
    metric_names = [
        "max_unselected_hole",
        "mean_selected_to_boundary_distance",
        "selected_within_r1",
        "selected_within_r2",
        "selected_within_r4",
        "selected_within_r8",
    ]
    return {
        "sample_count": len(records),
        "methods": {
            method: {
                metric: mean(
                    float(record["methods"][method]["metrics"][metric])
                    for record in records
                )
                for metric in metric_names
            }
            for method in methods
        },
        "mean_soft_max_gap_penalty": mean(
            float(record["soft_max_gap_penalty"]) for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--hard-max-unselected-hole", type=int, default=14)
    parser.add_argument("--soft-max-unselected-hole", type=int, default=14)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--coverage-floor", type=float, default=0.05)
    parser.add_argument("--smoothing-kernel", type=int, default=5)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("input JSONL is empty")
    records = [
        compare_sample(
            row,
            budget=args.budget,
            hard_max_unselected_hole=args.hard_max_unselected_hole,
            soft_max_unselected_hole=args.soft_max_unselected_hole,
            temperature=args.temperature,
            coverage_floor=args.coverage_floor,
            smoothing_kernel=args.smoothing_kernel,
        )
        for row in rows
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "density_selection_distribution.samples.jsonl"
    detail_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = _summary(records)
    summary.update(
        {
            "budget": args.budget,
            "hard_max_unselected_hole": args.hard_max_unselected_hole,
            "soft_max_unselected_hole": args.soft_max_unselected_hole,
            "detail_path": str(detail_path),
        }
    )
    summary_path = args.output_dir / "density_selection_distribution.summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
