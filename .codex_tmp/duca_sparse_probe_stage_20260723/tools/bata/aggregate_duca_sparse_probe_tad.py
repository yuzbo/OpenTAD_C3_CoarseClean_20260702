from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate(run_root: Path, output: Path) -> dict:
    rows = []
    for stride in (1, 2, 3, 4):
        completion = run_root / f"d{stride}" / "completion.json"
        if not completion.is_file():
            raise FileNotFoundError(completion)
        payload = json.loads(completion.read_text(encoding="utf-8"))
        expected_variant = f"sparse_probe_hidden_linear_d{stride}"
        if not payload.get("ok") or payload.get("variant") != expected_variant:
            raise ValueError(f"invalid sparse probe completion for d{stride}")
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"missing terminal metrics for d{stride}")
        rows.append(
            {
                "stride_dense_candidates": stride,
                "interval_source_frames": 4 * stride,
                "variant": expected_variant,
                "official_validation_comparable": bool(
                    payload.get("official_validation_comparable")
                ),
                "metrics": metrics,
                "completion": str(completion.resolve()),
            }
        )
    result = {
        "schema": "duca_sparse_probe_hidden_linear_tad_v1",
        "ok": True,
        "task": "offline_temporal_action_detection",
        "reconstruction": "multidimensional_temporal_hidden_linear_to_768",
        "selector_receives_anchor_metadata": False,
        "hard_budget": 384,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
