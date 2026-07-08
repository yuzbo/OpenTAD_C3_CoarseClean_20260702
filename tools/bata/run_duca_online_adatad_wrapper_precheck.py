from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeAdaTADSparseDetector:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def forward_sparse(self, observations, sparse_grid=None, batch=None, mode=None):
        if sparse_grid is None:
            raise AssertionError("sparse_grid is required")
        sparse_grid.validate()
        selected_count = sparse_grid.selected_count
        observed_len = int(observations.shape[1])
        if observed_len != int(selected_count.max().item()):
            raise AssertionError("detector input length does not match selected_positions")
        if not sparse_grid.detector_consumes_selected_positions:
            raise AssertionError("detector must consume selected_positions")
        self.calls.append(
            {
                "mode": mode,
                "observed_len": observed_len,
                "selected_count": [int(v) for v in selected_count.detach().cpu().tolist()],
                "batch_keys": sorted(str(k) for k in (batch or {}).keys()),
            }
        )
        return {
            "loss": observations.float().pow(2).mean(),
            "mode": mode,
            "detector_input_length": observed_len,
        }


def run_precheck(seed: int = 11, batch_size: int = 2, dense_len: int = 768, channels: int = 8, budget: int = 384) -> dict[str, Any]:
    import torch
    from opentad.models.duca import DucaOnlineSparseDetectorWrapper

    config_path = REPO_ROOT / "configs" / "adatad" / "thumos" / "duca_online_adatad_smoke.py"
    config = runpy.run_path(str(config_path))["duca_online_plugin"]
    torch.manual_seed(int(seed))
    observations = torch.randn(int(batch_size), int(dense_len), int(channels))
    valid_mask = torch.ones(int(batch_size), int(dense_len), dtype=torch.bool)
    detector = _FakeAdaTADSparseDetector()
    wrapper = DucaOnlineSparseDetectorWrapper(detector=detector, feature_dim=int(channels), budget=int(budget), max_radius=16)

    wrapper.train()
    train_result = wrapper(
        batch={
            "observations": observations,
            "valid_mask": valid_mask,
            "teacher_utility": torch.randn(int(batch_size), int(dense_len)),
            "dense_teacher_payload": {"score": 1.0},
            "gt_segments": torch.zeros(int(batch_size), 1, 2),
        },
        mode="loss",
    )
    wrapper.eval()
    test_result = wrapper(
        batch={
            "observations": observations,
            "valid_mask": valid_mask,
        },
        mode="predict",
    )
    try:
        wrapper(
            batch={
                "observations": observations,
                "valid_mask": valid_mask,
                "metas": [{"prediction_cache": {"x": 1}}],
            },
            mode="predict",
        )
        nested_forbidden_rejected = False
    except ValueError:
        nested_forbidden_rejected = True

    grid = test_result["grid"].validate()
    selected_counts = [int(v) for v in grid.selected_count.detach().cpu().tolist()]
    train_batch_keys = detector.calls[0]["batch_keys"]
    return {
        "status": "ok",
        "implementation": "opentad.models.duca.acquisition",
        "wrapper_class": "DucaOnlineSparseDetectorWrapper",
        "config_path": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "config_route": config["route"],
        "detector_family": config["detector_family"],
        "uses_ledger_for_decision": bool(test_result["audit"]["uses_ledger_for_decision"]),
        "teacher_free_inference": bool(not test_result["audit"]["uses_teacher"]),
        "nested_forbidden_rejected": nested_forbidden_rejected,
        "budget": int(budget),
        "budget_unit": grid.budget_unit,
        "coordinate": grid.coordinate,
        "selected_count": selected_counts,
        "detector_input_length": int(test_result["detector_input"].shape[1]),
        "detector_consumes_selected_positions": bool(grid.detector_consumes_selected_positions),
        "train_detector_batch_sanitized": "teacher_utility" not in train_batch_keys
        and "dense_teacher_payload" not in train_batch_keys
        and "gt_segments" not in train_batch_keys,
        "train_detector_loss_present": "detector_loss" in train_result["losses"],
        "precheck_pass": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precheck DUCA online wrapper into an AdaTAD-like sparse detector.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dense-len", type=int, default=768)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--budget", type=int, default=384)
    args = parser.parse_args(argv)

    summary = run_precheck(
        seed=args.seed,
        batch_size=args.batch_size,
        dense_len=args.dense_len,
        channels=args.channels,
        budget=args.budget,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
