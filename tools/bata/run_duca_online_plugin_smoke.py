from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _DummySparseDetector:
    def __init__(self) -> None:
        self.last_input_length = None

    def forward_sparse(self, observations, sparse_grid=None, batch=None, **kwargs):
        self.last_input_length = int(observations.shape[1])
        return {"loss": observations.float().mean(), "detector_input_length": self.last_input_length}


def run_smoke(seed: int = 7, batch_size: int = 2, dense_len: int = 768, channels: int = 8, budget: int = 384) -> dict[str, Any]:
    try:
        import torch
        from opentad.models.duca import DucaAcquisitionAdapter, duca_forward_test, duca_forward_train
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"torch/opentad.models.duca unavailable: {type(exc).__name__}: {exc}",
            "implementation": "unavailable",
        }

    torch.manual_seed(int(seed))
    dense_observations = torch.randn(int(batch_size), int(dense_len), int(channels))
    valid_mask = torch.ones(int(batch_size), int(dense_len), dtype=torch.bool)
    teacher_utility = torch.randn(int(batch_size), int(dense_len))
    adapter = DucaAcquisitionAdapter(feature_dim=int(channels), budget=int(budget), max_radius=16)
    detector = _DummySparseDetector()

    train_result = duca_forward_train(
        detector=detector,
        adapter=adapter,
        batch={
            "observations": dense_observations,
            "valid_mask": valid_mask,
            "teacher_utility": teacher_utility,
        },
    )
    test_result = duca_forward_test(
        detector=detector,
        adapter=adapter,
        batch={
            "observations": dense_observations,
            "valid_mask": valid_mask,
        },
    )
    grid = test_result["grid"]
    selected_counts = [int(item) for item in grid.selected_count.detach().cpu().tolist()]
    budget_violation_rate = sum(1 for count in selected_counts if count > int(budget)) / max(1, len(selected_counts))

    return {
        "status": "ok",
        "implementation": "opentad.models.duca.acquisition",
        "route": "DUCA_online_temporal_acquisition_plugin_smoke",
        "selected_count": selected_counts[0] if len(set(selected_counts)) == 1 else selected_counts,
        "budget": int(budget),
        "detector_input_length": int(test_result["detector_input"].shape[1]),
        "budget_violation_rate": float(budget_violation_rate),
        "uses_ledger_for_decision": bool(test_result["audit"]["uses_ledger_for_decision"]),
        "teacher_utility_used_train_only": "teacher_utility_loss" in train_result["losses"],
        "train_forward": "duca_forward_train",
        "test_forward": "duca_forward_test",
        "selected_positions_contract": "original-time detector-consumed positions",
        "budget_unit": grid.budget_unit,
        "coordinate": grid.coordinate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test for the real DUCA online acquisition plugin.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dense-len", type=int, default=768)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--budget", type=int, default=384)
    args = parser.parse_args(argv)

    summary = run_smoke(
        seed=args.seed,
        batch_size=args.batch_size,
        dense_len=args.dense_len,
        channels=args.channels,
        budget=args.budget,
    )
    text = json.dumps(summary, sort_keys=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
