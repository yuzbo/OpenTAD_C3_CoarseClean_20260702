from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MONITOR = ROOT / "tools" / "bata" / "monitor_duca_jct_experiment_suite.py"
WRAPPER = ROOT / "scripts" / "monitor_duca_jct_experiment_suite.sh"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _deployment(tmp_path: Path, *, with_x3d: bool = True, with_grad_proof: bool = True) -> Path:
    run_root = tmp_path / "duca_jct_suite"
    materialization = run_root / "trainfree_frozen_actionness" / "best_x3d_actionness.materialization.json"
    actionness = run_root / "trainfree_frozen_actionness" / "best_x3d_actionness.jsonl"
    if with_x3d:
        _write_text(actionness, '{"video_name":"v_test","p_action":[0.1,0.9]}\n')
        _write_json(
            materialization,
            {
                "schema_version": "trainfree_x3d_actionness_materialization_v1",
                "decision": "TRAINFREE_X3D_ACTIONNESS_MATERIALIZED",
                "downstream_detector_ready": True,
                "train_free_baseline": True,
                "not_main_method": True,
                "output_jsonl": str(actionness),
                "output_row_count": 1,
            },
        )
    if with_grad_proof:
        _write_json(
            run_root / "duca_jct_one_step_grad_proof.json",
            {
                "schema_version": "duca_jct_one_step_grad_proof_v1",
                "proof_passed": True,
                "fixed384": {
                    "coarse_probe_grad_sum": 1.0,
                    "selector_encoder_grad_sum": 1.0,
                    "budget_controller_grad_sum": None,
                    "loss_schedule_step_update": {"updated": True, "source": "optimizer_step"},
                    "dynamic_budget_dual_update": None,
                },
                "duca_must": {
                    "coarse_probe_grad_sum": 1.0,
                    "selector_encoder_grad_sum": 1.0,
                    "budget_controller_grad_sum": 1.0,
                    "loss_schedule_step_update": {"updated": True, "source": "optimizer_step"},
                    "dynamic_budget_dual_update": {"updated": True},
                },
            },
        )
    return _write_json(
        run_root / "deployment_summary.json",
        {
            "schema_version": "duca_jct_experiment_suite_deployment_v1",
            "commit": "6b2a3cf",
            "branch": "codex/gas-vt-stage23-detector-aware-20260706",
            "run_root": str(run_root),
            "formal_x3d_actionness_jsonl": str(actionness),
            "formal_x3d_materialization_summary": str(materialization),
            "duca_jct_tests_job": "101",
            "duca384_job": "102",
            "duca_must_job": "103",
            "x3d_grid_job": "104",
            "x3d_duca384_job": "105",
            "x3d_must_job": "106",
            "x3d_downstream_dependency": "afterok:104",
        },
    )


def test_duca_jct_suite_monitor_classifies_jobs_and_x3d_readiness(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, with_x3d=True)
    run_root = deployment.parent
    _write_text(run_root / "slurm_logs" / "duca_jct_tests_101.out", "28 passed in 4.61s\n")
    _write_text(
        run_root / "slurm_logs" / "duca_jct_must_103.err",
        "Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n",
    )
    _write_text(
        run_root / "slurm_logs" / "duca_x3d_grid_104.out",
        '{"decision":"TRAINFREE_X3D_ACTIONNESS_MATERIALIZED"}\n',
    )
    _write_json(run_root / "duca384_jct" / "work_dir" / "result_detection.json", {"version": "fake"})
    squeue = _write_text(
        tmp_path / "squeue.tsv",
        "JOBID|NAME|STATE|REASON\n102|duca_jct_384|RUNNING|None\n105|duca_x3d_384|PENDING|Dependency\n",
    )

    from tools.bata.monitor_duca_jct_experiment_suite import monitor_suite

    summary = monitor_suite(deployment_summary=deployment, squeue_text=squeue)

    assert summary["schema_version"] == "duca_jct_suite_monitor_v1"
    assert summary["joint_grad_proof"]["ready"] is True
    assert summary["joint_grad_proof"]["proof_passed"] is True
    assert summary["joint_grad_proof"]["duca_must_budget_controller_grad_sum"] == 1.0
    assert summary["joint_grad_proof"]["fixed_loss_schedule_step_update"]["source"] == "optimizer_step"
    assert summary["joint_grad_proof"]["duca_must_loss_schedule_step_update"]["source"] == "optimizer_step"
    assert summary["formal_x3d_actionness"]["ready"] is True
    assert summary["formal_x3d_actionness"]["downstream_detector_ready"] is True
    assert summary["jobs"]["duca_jct_tests"]["status"] == "completed"
    assert summary["jobs"]["duca384"]["status"] == "running"
    assert summary["jobs"]["duca_must"]["status"] == "failed"
    assert summary["jobs"]["x3d_duca384"]["status"] == "pending"
    assert "duca_must" in summary["hard_failures"]
    assert "duca384" in summary["running_jobs"]
    assert "duca384" not in summary["missing_results"]


def test_duca_jct_suite_monitor_queries_live_squeue_by_default(monkeypatch, tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, with_x3d=True)
    run_root = deployment.parent
    _write_text(run_root / "slurm_logs" / "duca_jct_384_102.out", "Average-mAP: 12.34 (%)\n")

    from tools.bata import monitor_duca_jct_experiment_suite as monitor

    class _Completed:
        returncode = 0
        stdout = "102|duca_jct_384|RUNNING|None\n"

    def fake_run(cmd, **kwargs):
        assert cmd[:2] == ["squeue", "-h"]
        assert "-j" in cmd
        return _Completed()

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)

    summary = monitor.monitor_suite(deployment_summary=deployment, squeue_text=None)

    assert summary["jobs"]["duca384"]["status"] == "running"
    assert "duca384" in summary["running_jobs"]
    assert "duca384" not in summary["missing_results"]


def test_duca_jct_suite_monitor_blocks_x3d_downstream_when_formal_jsonl_missing(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, with_x3d=False)

    from tools.bata.monitor_duca_jct_experiment_suite import monitor_suite

    summary = monitor_suite(deployment_summary=deployment, squeue_text=None)

    assert summary["formal_x3d_actionness"]["ready"] is False
    assert "formal_x3d_materialization_summary" in summary["missing_prerequisites"]
    assert "formal_x3d_actionness_jsonl" in summary["missing_prerequisites"]
    assert summary["jobs"]["x3d_duca384"]["status"] == "blocked_missing_x3d_actionness"
    assert summary["jobs"]["x3d_must"]["status"] == "blocked_missing_x3d_actionness"


def test_duca_jct_suite_monitor_requires_one_step_joint_grad_proof(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, with_x3d=True, with_grad_proof=False)

    from tools.bata.monitor_duca_jct_experiment_suite import monitor_suite

    summary = monitor_suite(deployment_summary=deployment, squeue_text=None)

    assert summary["joint_grad_proof"]["ready"] is False
    assert "duca_jct_one_step_grad_proof" in summary["missing_prerequisites"]


def test_duca_jct_suite_monitor_cli_writes_json_and_wrapper_uses_squeue(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, with_x3d=True)
    output = tmp_path / "monitor_summary.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(MONITOR),
            "--deployment-summary",
            str(deployment),
            "--output-json",
            str(output),
        ],
        cwd=str(ROOT),
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "duca_jct_suite_monitor_v1"
    assert json.loads(completed.stdout)["output_json"] == str(output)

    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    assert "squeue" in wrapper_text
    assert "/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python" in wrapper_text
    assert "monitor_duca_jct_experiment_suite.py" in wrapper_text
