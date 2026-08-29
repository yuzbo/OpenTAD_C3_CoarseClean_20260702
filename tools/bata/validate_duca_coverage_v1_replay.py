"""Unlabeled real-video intervention gate for DUCA Coverage-v1."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.datasets import build_dataloader, build_dataset
from opentad.models.builder import build_selector
from tools.bata.duca_frontend_initialization import selector_state_dict


def _checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state = payload.get("state_dict_ema")
    if not isinstance(state, Mapping):
        raise ValueError("Stage-1 checkpoint lacks state_dict_ema")
    return state


def _load_selector(config_path: str, checkpoint_state: Mapping[str, Any], device: torch.device):
    cfg = Config.fromfile(config_path)
    selector = build_selector(cfg.model.frame_selector)
    incompatible = selector.load_state_dict(
        selector_state_dict(checkpoint_state), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "selector checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return cfg, selector.to(device).eval()


def _positions(row: torch.Tensor) -> torch.Tensor:
    return row[row >= 0].long()


def _anchor_coverage(positions: torch.Tensor, valid_len: int, anchor_count: int = 96) -> float:
    if positions.numel() == 0:
        return 0.0
    denom = max(int(valid_len) - 1, 1)
    times = positions.float() / float(denom)
    anchors = torch.linspace(0.0, 1.0, anchor_count, device=positions.device)
    sigma = 1.0 / float(anchor_count - 1)
    kernel = torch.exp(-torch.abs(times[:, None] - anchors[None, :]) / sigma)
    return float(kernel.amax(dim=0).mean().cpu())


def _max_unselected_gap(positions: torch.Tensor, valid_len: int) -> int:
    if positions.numel() == 0:
        return int(valid_len)
    gaps = [int(positions[0])]
    if positions.numel() > 1:
        gaps.extend(int(value) for value in (positions[1:] - positions[:-1] - 1).cpu())
    gaps.append(int(valid_len - 1 - int(positions[-1])))
    return max(gaps)


def _normalized_priority(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    active = scores[valid].float()
    low = active.amin()
    high = active.amax()
    quality = torch.zeros_like(scores, dtype=torch.float32)
    quality[valid] = (active - low) / (high - low + 1.0e-6)
    return quality


def validate_replay(
    *,
    control_config: str,
    candidate_config: str,
    checkpoint: str,
    output_json: str,
    split: str = "train",
    device: str = "cuda:0",
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_samples: int = 0,
    seed: int = 3407,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-1 checkpoint must be a mapping")
    checkpoint_state = _checkpoint_state(payload)
    control_cfg, control = _load_selector(control_config, checkpoint_state, torch_device)
    _, candidate = _load_selector(candidate_config, checkpoint_state, torch_device)

    dataset = build_dataset(control_cfg.dataset[split], default_args=dict(logger=None))
    loader_cfg = dict(control_cfg.solver[split])
    if batch_size is not None:
        loader_cfg["batch_size"] = int(batch_size)
    if num_workers is not None:
        loader_cfg["num_workers"] = int(num_workers)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )

    coverage_control: list[float] = []
    coverage_candidate: list[float] = []
    gap_control: list[int] = []
    gap_candidate: list[int] = []
    set_change: list[float] = []
    priority_ratio: list[float] = []
    exact_rows = 0
    sample_count = 0
    priority_identity = True
    finite = True

    with torch.no_grad():
        for data in loader:
            inputs = data["inputs"].to(torch_device, non_blocking=True)
            masks = data["masks"].to(torch_device, non_blocking=True).bool()
            with torch.cuda.amp.autocast(
                dtype=torch.float16, enabled=torch_device.type == "cuda"
            ):
                control_out = control.forward_test(inputs=inputs, masks=masks, metas=None)
                candidate_out = candidate.forward_test(inputs=inputs, masks=masks, metas=None)
            control_scores = control_out["selector_outputs"]["center_scores"].float()
            candidate_scores = candidate_out["selector_outputs"]["center_scores"].float()
            priority_identity = priority_identity and bool(
                torch.allclose(control_scores, candidate_scores, rtol=0.0, atol=0.0)
            )
            control_rows = control_out["selector_outputs"]["grid"].selected_positions
            candidate_rows = candidate_out["selector_outputs"]["grid"].selected_positions
            for row_idx in range(inputs.shape[0]):
                valid = masks[row_idx]
                valid_len = int(valid.long().sum().cpu())
                expected = min(384, valid_len)
                old = _positions(control_rows[row_idx])
                new = _positions(candidate_rows[row_idx])
                old_set = set(int(value) for value in old.cpu())
                new_set = set(int(value) for value in new.cpu())
                row_exact = (
                    old.numel() == expected
                    and new.numel() == expected
                    and len(old_set) == expected
                    and len(new_set) == expected
                    and all(0 <= value < valid_len for value in old_set | new_set)
                )
                exact_rows += int(row_exact)
                quality = _normalized_priority(control_scores[row_idx], valid)
                old_mass = float(quality[old].sum().cpu())
                new_mass = float(quality[new].sum().cpu())
                priority_ratio.append(
                    1.0 if old_mass <= 1.0e-12 and new_mass <= 1.0e-12
                    else new_mass / max(old_mass, 1.0e-12)
                )
                set_change.append(1.0 - len(old_set & new_set) / float(max(expected, 1)))
                coverage_control.append(_anchor_coverage(old, valid_len))
                coverage_candidate.append(_anchor_coverage(new, valid_len))
                gap_control.append(_max_unselected_gap(old, valid_len))
                gap_candidate.append(_max_unselected_gap(new, valid_len))
                finite = finite and all(
                    math.isfinite(value)
                    for value in (
                        priority_ratio[-1],
                        set_change[-1],
                        coverage_control[-1],
                        coverage_candidate[-1],
                    )
                )
                sample_count += 1
                if max_samples > 0 and sample_count >= max_samples:
                    break
            if max_samples > 0 and sample_count >= max_samples:
                break

    if sample_count <= 0:
        raise RuntimeError("unlabeled replay produced no samples")
    control_coverage_median = float(np.median(coverage_control))
    candidate_coverage_median = float(np.median(coverage_candidate))
    coverage_gain = (
        candidate_coverage_median / max(control_coverage_median, 1.0e-12) - 1.0
    )
    control_gap_p95 = float(np.percentile(gap_control, 95))
    candidate_gap_p95 = float(np.percentile(gap_candidate, 95))
    gap_reduction = 1.0 - candidate_gap_p95 / max(control_gap_p95, 1.0)
    metrics = {
        "sample_count": sample_count,
        "exact_valid_unique_fraction": exact_rows / float(sample_count),
        "priority_tensor_identity": priority_identity,
        "finite": finite,
        "median_set_change_fraction": float(np.median(set_change)),
        "control_anchor_coverage_median": control_coverage_median,
        "candidate_anchor_coverage_median": candidate_coverage_median,
        "relative_anchor_coverage_gain": coverage_gain,
        "control_max_gap_p95": control_gap_p95,
        "candidate_max_gap_p95": candidate_gap_p95,
        "relative_max_gap_p95_reduction": gap_reduction,
        "retained_normalized_h65_priority_median_ratio": float(
            np.median(priority_ratio)
        ),
    }
    thresholds = {
        "exact_valid_unique_fraction": 1.0,
        "median_set_change_fraction_min": 0.80,
        "relative_anchor_coverage_gain_min": 0.10,
        "relative_max_gap_p95_reduction_min": 0.20,
        "retained_normalized_h65_priority_median_ratio_min": 0.90,
    }
    passed = (
        metrics["exact_valid_unique_fraction"] == 1.0
        and priority_identity
        and finite
        and metrics["median_set_change_fraction"] >= 0.80
        and metrics["relative_anchor_coverage_gain"] >= 0.10
        and metrics["relative_max_gap_p95_reduction"] >= 0.20
        and metrics["retained_normalized_h65_priority_median_ratio"] >= 0.90
    )
    result = {
        "schema": "duca_coverage_v1_unlabeled_replay_gate",
        "passed": bool(passed),
        "selection_uses_gt": False,
        "selection_uses_teacher": False,
        "selection_uses_prediction_cache": False,
        "split": split,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_state_key": "state_dict_ema",
        "control_config": str(Path(control_config).resolve()),
        "candidate_config": str(Path(candidate_config).resolve()),
        "thresholds": thresholds,
        "metrics": metrics,
    }
    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)
    result = validate_replay(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
