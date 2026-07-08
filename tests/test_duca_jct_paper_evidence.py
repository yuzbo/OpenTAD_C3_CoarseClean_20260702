from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "tools" / "bata" / "collect_duca_jct_paper_evidence.py"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _monitor_summary(
    tmp_path: Path,
    *,
    include_high_iou: bool = True,
    include_trainfree_results: bool = True,
    include_joint_grad_proof: bool = True,
) -> Path:
    metrics = {
        "average_mAP_percent": 66.4,
        "mAP@0.30_percent": 73.0,
        "mAP@0.40_percent": 69.0,
        "mAP@0.50_percent": 62.0,
        "mAP@0.60_percent": 51.2,
    }
    if include_high_iou:
        metrics["mAP@0.70_percent"] = 36.1
    return _write_json(
        tmp_path / "duca_jct_suite_monitor.summary.json",
        {
            "schema_version": "duca_jct_suite_monitor_v1",
            "commit": "327d70d",
            "branch": "codex/gas-vt-stage23-detector-aware-20260706",
            "run_root": str(tmp_path),
            "hard_failures": [],
            "missing_results": [],
            "missing_prerequisites": [],
            "formal_x3d_actionness": {"ready": True, "train_free_baseline": True, "not_main_method": True},
            "joint_grad_proof": {
                "ready": include_joint_grad_proof,
                "proof_passed": include_joint_grad_proof,
                "fixed_coarse_probe_grad_sum": 1.0 if include_joint_grad_proof else 0.0,
                "fixed_selector_encoder_grad_sum": 1.0 if include_joint_grad_proof else 0.0,
                "duca_must_coarse_probe_grad_sum": 1.0 if include_joint_grad_proof else 0.0,
                "duca_must_selector_encoder_grad_sum": 1.0 if include_joint_grad_proof else 0.0,
                "duca_must_budget_controller_grad_sum": 1.0 if include_joint_grad_proof else 0.0,
            },
            "jobs": {
                "duca384": {
                    "status": "completed",
                    "metrics": {
                        "average_mAP_percent": 65.9,
                        "mAP@0.60_percent": 50.4,
                        "mAP@0.70_percent": 35.3,
                    },
                    "result_artifacts": [str(tmp_path / "duca384" / "result_detection.json")],
                },
                "duca_must": {
                    "status": "completed",
                    "metrics": metrics,
                    "result_artifacts": [str(tmp_path / "duca_must" / "result_detection.json")],
                },
                "x3d_duca384": {
                    "status": "completed" if include_trainfree_results else "pending",
                    "metrics": {
                        "average_mAP_percent": 64.8,
                        "mAP@0.60_percent": 49.0,
                        "mAP@0.70_percent": 34.0,
                    }
                    if include_trainfree_results
                    else {},
                    "result_artifacts": [str(tmp_path / "x3d384" / "result_detection.json")]
                    if include_trainfree_results
                    else [],
                },
                "x3d_must": {
                    "status": "completed" if include_trainfree_results else "pending",
                    "metrics": {
                        "average_mAP_percent": 65.1,
                        "mAP@0.60_percent": 49.4,
                        "mAP@0.70_percent": 34.2,
                    }
                    if include_trainfree_results
                    else {},
                    "result_artifacts": [str(tmp_path / "x3d_must" / "result_detection.json")]
                    if include_trainfree_results
                    else [],
                },
            },
        },
    )


def _baseline_summary(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "matched_baselines.json",
        {
            "schema_version": "duca_matched_baselines_v1",
            "primary_baseline": "uniform384",
            "baselines": {
                "uniform384": {
                    "average_mAP_percent": 65.0,
                    "mAP@0.60_percent": 50.0,
                    "mAP@0.70_percent": 35.0,
                }
            },
        },
    )


def test_duca_jct_paper_evidence_allows_claim_only_with_baseline_and_high_iou(tmp_path: Path) -> None:
    from tools.bata.collect_duca_jct_paper_evidence import collect_evidence

    summary = collect_evidence(
        monitor_summary=_monitor_summary(tmp_path),
        baseline_summary=_baseline_summary(tmp_path),
        min_avg_delta=0.7,
    )

    assert summary["schema_version"] == "duca_jct_paper_evidence_v1"
    assert summary["main_duca_results_complete"] is True
    assert summary["trainfree_baseline_results_complete"] is True
    assert summary["joint_grad_proof_ready"] is True
    assert summary["paper_claim_allowed"] is True
    assert summary["claim_gate"]["primary_baseline"] == "uniform384"
    assert summary["claim_gate"]["best_main_method"] == "duca_must"
    assert summary["claim_gate"]["best_main_average_mAP_delta"] == 1.4
    assert not summary["claim_gate"]["blockers"]
    assert [row["method"] for row in summary["table_rows"]] == [
        "duca384",
        "duca_must",
        "x3d_duca384",
        "x3d_must",
    ]


def test_duca_jct_paper_evidence_blocks_claim_without_baseline_or_high_iou(tmp_path: Path) -> None:
    from tools.bata.collect_duca_jct_paper_evidence import collect_evidence

    no_baseline = collect_evidence(monitor_summary=_monitor_summary(tmp_path / "no_baseline"))
    assert no_baseline["paper_claim_allowed"] is False
    assert "missing_matched_reference_baseline" in no_baseline["claim_gate"]["blockers"]

    missing_high_iou = collect_evidence(
        monitor_summary=_monitor_summary(tmp_path / "missing_high_iou", include_high_iou=False),
        baseline_summary=_baseline_summary(tmp_path / "missing_high_iou"),
    )
    assert missing_high_iou["paper_claim_allowed"] is False
    assert "missing_high_iou_metric:mAP@0.70_percent" in missing_high_iou["claim_gate"]["blockers"]

    missing_trainfree = collect_evidence(
        monitor_summary=_monitor_summary(tmp_path / "missing_trainfree", include_trainfree_results=False),
        baseline_summary=_baseline_summary(tmp_path / "missing_trainfree"),
    )
    assert missing_trainfree["trainfree_baseline_results_complete"] is False
    assert missing_trainfree["paper_claim_allowed"] is False
    assert "trainfree_method_not_completed:x3d_duca384" in missing_trainfree["claim_gate"]["blockers"]
    assert "missing_trainfree_high_iou_metric:mAP@0.70_percent" in missing_trainfree["claim_gate"]["blockers"]

    missing_grad_proof = collect_evidence(
        monitor_summary=_monitor_summary(tmp_path / "missing_grad_proof", include_joint_grad_proof=False),
        baseline_summary=_baseline_summary(tmp_path / "missing_grad_proof"),
    )
    assert missing_grad_proof["paper_claim_allowed"] is False
    assert "missing_or_failed_joint_grad_proof" in missing_grad_proof["claim_gate"]["blockers"]


def test_duca_jct_paper_evidence_cli_writes_json_and_tsv(tmp_path: Path) -> None:
    output_json = tmp_path / "paper_evidence.json"
    output_tsv = tmp_path / "paper_evidence.tsv"

    subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            "--monitor-summary",
            str(_monitor_summary(tmp_path)),
            "--baseline-summary",
            str(_baseline_summary(tmp_path)),
            "--output-json",
            str(output_json),
            "--output-tsv",
            str(output_tsv),
        ],
        cwd=str(ROOT),
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["paper_claim_allowed"] is True
    rows = list(csv.DictReader(output_tsv.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert rows[1]["method"] == "duca_must"
    assert rows[1]["average_mAP_percent"] == "66.4"
    assert rows[1]["delta_vs_primary_average_mAP"] == "1.4"

    wrapper_text = (ROOT / "scripts" / "collect_duca_jct_paper_evidence.sh").read_text(encoding="utf-8")
    assert "/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python" in wrapper_text
    assert "collect_duca_jct_paper_evidence.py" in wrapper_text
