"""Read-only audit of a normalized raw-transition residual on DUCA scores."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from opentad.models.duca.structured_selection import (
    exact_uniform_positions,
    global_structured_topk,
)
from tools.bata.audit_duca_homotopy_trajectory import (
    _evaluate_gt,
    _records_source,
    _sha256_file,
    _sha256_json,
    _write_json_fail_closed,
    selection_geometry,
)


SCHEMA_VERSION = "duca_delta_residual_sweep_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def parse_gamma_grid(value: str | Sequence[float]) -> list[float]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    gammas = [float(item) for item in raw]
    _require(bool(gammas), "gamma grid must not be empty")
    _require(
        all(math.isfinite(gamma) and gamma >= 0.0 for gamma in gammas),
        "gamma values must be finite and non-negative",
    )
    _require(gammas == sorted(set(gammas)), "gamma grid must be unique and increasing")
    _require(gammas[0] == 0.0, "gamma grid must start at zero")
    return gammas


def standardized_residual_scores(
    learned_scores: torch.Tensor,
    delta_scores: torch.Tensor,
    gamma: float,
    *,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return z(learned) + gamma*z(delta), preserving the gamma-zero hard path."""

    _require(learned_scores.ndim == delta_scores.ndim == 1, "score rows must be one-dimensional")
    _require(learned_scores.shape == delta_scores.shape, "learned and delta rows must align")
    _require(learned_scores.is_floating_point() and delta_scores.is_floating_point(), "scores must be floating point")
    _require(bool(torch.isfinite(learned_scores).all().item()), "learned scores must be finite")
    _require(bool(torch.isfinite(delta_scores).all().item()), "delta scores must be finite")
    gamma = float(gamma)
    _require(math.isfinite(gamma) and gamma >= 0.0, "gamma must be finite and non-negative")

    learned_centered = learned_scores - learned_scores.mean()
    delta_centered = delta_scores - delta_scores.mean()
    learned_scale = learned_centered.square().mean().sqrt()
    delta_scale = delta_centered.square().mean().sqrt()
    learned_z = learned_centered / learned_scale.clamp_min(float(epsilon))
    delta_z = delta_centered / delta_scale.clamp_min(float(epsilon))
    if float(learned_scale.item()) < float(epsilon):
        learned_z = learned_centered
    if float(delta_scale.item()) < float(epsilon):
        delta_z = delta_centered
    return learned_z + gamma * delta_z, {
        "learned_rms": float(learned_scale.item()),
        "delta_rms": float(delta_scale.item()),
    }


def _strict_score_row(record: Mapping[str, Any], key: str, valid_len: int) -> torch.Tensor:
    values = record.get(key)
    _require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) >= valid_len,
        f"{key} is missing or shorter than valid_len",
    )
    row = torch.tensor([float(value) for value in values[:valid_len]], dtype=torch.float32)
    _require(bool(torch.isfinite(row).all().item()), f"{key} contains non-finite values")
    return row


def _aggregate_gamma(samples: Sequence[Mapping[str, Any]], gamma: float) -> dict[str, Any]:
    rows = [sample["gamma_results"][str(gamma)] for sample in samples]
    recalls: dict[str, float | None] = {}
    for radius in ("r0", "r1", "r2", "r4"):
        values = [row["gt_evaluation"]["boundary_recall"][radius] for row in rows]
        finite = [float(value) for value in values if value is not None]
        recalls[radius] = _mean(finite) if finite else None
    distances = [row["gt_evaluation"]["mean_nearest_selected_distance"] for row in rows]
    finite_distances = [float(value) for value in distances if value is not None]
    return {
        "gamma": float(gamma),
        "mean_boundary_recall": recalls,
        "mean_nearest_selected_distance": _mean(finite_distances) if finite_distances else None,
        "mean_uniform_overlap_rate": _mean([row["uniform_overlap_rate"] for row in rows]),
        "mean_gamma0_overlap_rate": _mean([row["gamma0_overlap_rate"] for row in rows]),
        "mean_adjacent_selection_rate": _mean([row["geometry"]["adjacent_selection_rate"] for row in rows]),
        "mean_max_hole": _mean([row["geometry"]["max_hole"] for row in rows]),
        "max_observed_hole": int(max(row["geometry"]["max_hole"] for row in rows)),
    }


def run_residual_sweep(
    *,
    records_jsonl: str | Path,
    output_json: str | Path,
    gamma_grid: str | Sequence[float],
    budget: int,
    max_unselected_hole: int,
    device: str = "cpu",
) -> dict[str, Any]:
    records_path = Path(records_jsonl).expanduser().resolve(strict=True)
    output_path = Path(output_json).expanduser().resolve()
    rows: list[Mapping[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, Mapping), f"record line {line_index} must be a mapping")
            _require(
                str(value.get("schema_version")) == "duca_selection_quality_record_v2",
                f"record line {line_index} has an unsupported schema",
            )
            rows.append(value)
    _require(bool(rows), "records JSONL is empty")
    source = _records_source(rows)
    gammas = parse_gamma_grid(gamma_grid)
    budget = int(budget)
    max_unselected_hole = int(max_unselected_hole)
    _require(budget > 0, "budget must be positive")
    _require(max_unselected_hole >= 0, "max_unselected_hole must be non-negative")

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA device requested but unavailable")
        torch.cuda.set_device(torch_device)

    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            sample_id = str(row.get("sample_id", ""))
            _require(bool(sample_id), f"record {index} has no sample_id")
            valid_len = int(row.get("valid_len", 0))
            _require(valid_len > 0, f"{sample_id}: valid_len must be positive")
            effective_k = min(budget, valid_len)
            _require(
                valid_len - effective_k <= (effective_k + 1) * max_unselected_hole,
                f"{sample_id}: infeasible exact-K/max-hole contract",
            )
            learned = _strict_score_row(row, "transition_policy_scores", valid_len).to(torch_device)
            delta = _strict_score_row(row, "abs_delta_p_action", valid_len).to(torch_device)
            uniform = set(int(value) for value in exact_uniform_positions(valid_len, effective_k).tolist())
            decoded_by_gamma: dict[str, tuple[int, ...]] = {}
            gamma_results: dict[str, Any] = {}
            scales: dict[str, float] | None = None
            for gamma in gammas:
                scores, current_scales = standardized_residual_scores(learned, delta, gamma)
                scales = current_scales
                decoded = global_structured_topk(
                    scores[None, :],
                    k=effective_k,
                    max_unselected_hole=max_unselected_hole,
                    training=False,
                )
                positions = tuple(int(value) for value in decoded.selected_positions[0].tolist())
                decoded_by_gamma[str(gamma)] = positions
                selected = set(positions)
                geometry = selection_geometry(positions, valid_len)
                _require(geometry["max_hole"] <= max_unselected_hole, f"{sample_id}: max-hole violation")
                gamma_results[str(gamma)] = {
                    "selected_positions": list(positions),
                    "uniform_overlap_rate": float(len(selected & uniform) / effective_k),
                    "geometry": geometry,
                    "gt_evaluation": _evaluate_gt(positions, row.get("gt_segments", []), valid_len),
                }
            gamma0 = set(decoded_by_gamma[str(gammas[0])])
            for gamma in gammas:
                selected = set(decoded_by_gamma[str(gamma)])
                gamma_results[str(gamma)]["gamma0_overlap_rate"] = float(len(selected & gamma0) / effective_k)
            samples.append(
                {
                    "sample_id": sample_id,
                    "valid_len": valid_len,
                    "effective_k": effective_k,
                    "score_scales": scales,
                    "gamma_results": gamma_results,
                    "gt_role": "evaluation_only_not_selection_input",
                }
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "gamma_grid": gammas,
        "selection_contract": {
            "decoder": "global_structured_topk",
            "requested_budget": budget,
            "max_unselected_hole": max_unselected_hole,
            "score_formula": "z(transition_policy_scores)+gamma*z(abs_delta_p_action)",
            "inference_endpoint_scores": True,
            "gt_used_for_selection": False,
        },
        "aggregate": {
            "sample_count": len(samples),
            "by_gamma": [_aggregate_gamma(samples, gamma) for gamma in gammas],
        },
        "samples": samples,
        "provenance": {
            "records_jsonl": str(records_path),
            "records_sha256": _sha256_file(records_path),
            "record_source": source,
            "gamma_grid_sha256": _sha256_json(gammas),
            "audit_script_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "read_only_contract": {
            "selector_constructed": False,
            "adatad_detector_constructed": False,
            "optimizer_constructed": False,
            "gradients_enabled": False,
            "gt_used_for_selection": False,
            "model_selection_allowed": False,
        },
    }
    _write_json_fail_closed(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a normalized raw-delta residual without training a model.")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gamma-grid", default="0,0.1,0.25,0.5,0.75,1,2")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--max-unselected-hole", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    report = run_residual_sweep(
        records_jsonl=args.records_jsonl,
        output_json=args.output_json,
        gamma_grid=args.gamma_grid,
        budget=args.budget,
        max_unselected_hole=args.max_unselected_hole,
        device=args.device,
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
