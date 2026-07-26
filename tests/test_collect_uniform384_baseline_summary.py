from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "tools" / "bata" / "collect_uniform384_baseline_summary.py"


def test_collect_uniform384_baseline_summary_writes_evidence_baseline(tmp_path: Path) -> None:
    run_root = tmp_path / "uniform384"
    log_dir = run_root / "logs"
    work_dir = run_root / "work_dir"
    log_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    (log_dir / "train.out").write_text(
        "\n".join(
            [
                "Train INFO: Average-mAP: 65.25 (%)",
                "Train INFO: mAP at tIoU 0.30 is 80.00%",
                "Train INFO: mAP at tIoU 0.40 is 75.00%",
                "Train INFO: mAP at tIoU 0.50 is 66.00%",
                "Train INFO: mAP at tIoU 0.60 is 52.00%",
                "Train INFO: mAP at tIoU 0.70 is 36.00%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (work_dir / "result_detection.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "matched_baselines.json"

    subprocess.run(
        [sys.executable, str(COLLECTOR), "--run-root", str(run_root), "--output-json", str(out)],
        cwd=str(ROOT),
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "duca_matched_baselines_v1"
    assert payload["primary_baseline"] == "uniform384"
    assert payload["baselines"]["uniform384"]["average_mAP_percent"] == 65.25
    assert payload["baselines"]["uniform384"]["mAP@0.60_percent"] == 52.0
    assert payload["baselines"]["uniform384"]["mAP@0.70_percent"] == 36.0
    assert payload["baseline_runs"]["uniform384"]["status"] == "completed"
    assert payload["baseline_runs"]["uniform384"]["complete"] is True
