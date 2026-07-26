from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.monitor_duca_jct_experiment_suite import _extract_metrics  # noqa: E402


SCHEMA_VERSION = "duca_matched_baselines_v1"
BASELINE_NAME = "uniform384"
METRIC_KEYS = (
    "average_mAP_percent",
    "mAP@0.30_percent",
    "mAP@0.40_percent",
    "mAP@0.50_percent",
    "mAP@0.60_percent",
    "mAP@0.70_percent",
)


def _path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = _path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_text(paths: Sequence[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _candidate_logs(run_root: Path) -> list[Path]:
    patterns = (
        "logs/train.out",
        "logs/train.log",
        "slurm_logs/*.out",
        "slurm_logs/*.err",
        "**/train.out",
        "**/train.log",
    )
    logs: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(run_root.glob(pattern)):
            if path.is_file() and path not in seen:
                logs.append(path)
                seen.add(path)
    return logs


def _result_artifacts(run_root: Path) -> list[str]:
    artifacts: list[str] = []
    seen: set[Path] = set()
    for pattern in ("**/result_detection.json", "**/result_detection*.json"):
        for path in sorted(run_root.glob(pattern)):
            if path.is_file() and path not in seen:
                artifacts.append(str(path))
                seen.add(path)
    return artifacts


def collect_uniform384_baseline_summary(run_root: str | Path) -> dict[str, Any]:
    root = _path(run_root)
    if not root.exists():
        raise FileNotFoundError(f"uniform384 run root does not exist: {root}")
    logs = _candidate_logs(root)
    metrics_all = _extract_metrics(_read_text(logs))
    metrics = {key: float(metrics_all[key]) for key in METRIC_KEYS if key in metrics_all}
    artifacts = _result_artifacts(root)
    complete = bool(artifacts) and all(key in metrics for key in ("average_mAP_percent", "mAP@0.60_percent", "mAP@0.70_percent"))
    return {
        "schema_version": SCHEMA_VERSION,
        "primary_baseline": BASELINE_NAME,
        "baselines": {BASELINE_NAME: metrics},
        "baseline_runs": {
            BASELINE_NAME: {
                "run_root": str(root),
                "status": "completed" if complete else "incomplete",
                "log_paths": [str(path) for path in logs],
                "result_artifacts": artifacts,
                "complete": bool(complete),
            }
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect DUCA matched uniform384 baseline summary.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    payload = collect_uniform384_baseline_summary(args.run_root)
    _write_json(args.output_json, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
