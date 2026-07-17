from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.bata.capture_duca_cellcf_slurm_cost import capture_slurm_cost
from tools.bata.duca_cellcf_training import canonical_sha256
from tools.bata.summarize_duca_cellcf_training_cost import summarize_training_cost


COMMIT = "a" * 40
VARIANTS = ("uniform", "transition_beta0", "cellcf")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_capture_slurm_cost_uses_top_level_completed_allocation(
    tmp_path: Path,
) -> None:
    raw = (
        "123|cellcf-uniform|cluster-a|COMPLETED|0:0|7200|"
        "billing=4,cpu=4,gres/gpu=1,mem=32G,node=1|12G|720000|"
        "2026-07-17T01:00:00|2026-07-17T03:00:00\n"
    )

    payload = capture_slurm_cost(
        job_id=123,
        expected_job_name="cellcf-uniform",
        expected_cluster="cluster-a",
        raw_output_path=tmp_path / "uniform.sacct.psv",
        run_command=lambda _command: raw,
    )

    assert payload["allocation_gpu_hours"] == 2.0
    assert payload["allocation_peak_cpu_rss_bytes"] == 12 * 1024**3
    assert payload["allocation_consumed_energy_joules"] == 720000.0
    assert "terminal_evaluation" in payload["measurement_scope"]
    assert payload["gpu_peak_memory_bytes"] is None
    assert "JobName%128" in payload["sacct_command"][-1]
    assert "AllocTRES%512" in payload["sacct_command"][-1]
    assert payload["sacct_raw_replayable"] is True
    assert Path(payload["sacct_raw_artifact_path"]).read_text(
        encoding="utf-8"
    ) == raw
    assert payload["sacct_raw_artifact_sha256"] == hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def test_capture_slurm_cost_rejects_failed_or_multi_gpu_job() -> None:
    failed = (
        "123|cellcf-uniform|cluster-a|FAILED|1:0|10|"
        "cpu=4,gres/gpu=1|1G||start|end\n"
    )
    with pytest.raises(ValueError, match="not successfully complete"):
        capture_slurm_cost(
            job_id=123,
            expected_job_name="cellcf-uniform",
            expected_cluster="cluster-a",
            run_command=lambda _command: failed,
        )


def _cost_fixture(tmp_path: Path):
    post_runs = {}
    costs = {}
    for index, variant in enumerate(VARIANTS):
        job_id = 100 + index
        audit_path = tmp_path / variant / "training_audit.json"
        _write(audit_path, {"slurm_job_id": str(job_id)})
        post_path = tmp_path / variant / "post_run.json"
        _write(
            post_path,
            {
                "schema": "duca_cellcf_post_run_evidence_v1",
                "ok": True,
                "variant": variant,
                "git_commit": COMMIT,
                "training_profile": "official60",
                "successful_optimizer_updates": 6000,
                "training_audit_path": str(audit_path.resolve()),
                "training_audit_sha256": hashlib.sha256(
                    audit_path.read_bytes()
                ).hexdigest(),
            },
        )
        cost_path = tmp_path / variant / "slurm_cost.json"
        elapsed = 3600 + index * 600
        raw = (
            f"{job_id}|cellcf-{variant}|cluster-a|COMPLETED|0:0|{elapsed}|"
            "billing=4,cpu=4,gres/gpu=1,mem=32G,node=1|12G||"
            f"2026-07-17T0{index + 1}:00:00|"
            f"2026-07-17T0{index + 2}:00:00\n"
        )
        cost = capture_slurm_cost(
            job_id=job_id,
            expected_job_name=f"cellcf-{variant}",
            expected_cluster="cluster-a",
            raw_output_path=tmp_path / variant / "sacct.psv",
            run_command=lambda _command, raw=raw: raw,
        )
        _write(cost_path, cost)
        post_runs[variant] = post_path
        costs[variant] = cost_path
    suite_binding = {
        "path": str((tmp_path / "aggregate.json").resolve()),
        "sha256": "b" * 64,
        "git_commit": COMMIT,
        "training_profile": "official60",
        "seed": 0,
        "variant_order": list(VARIANTS),
        "shared_protocol_sha256": "c" * 64,
        "ordered_exposure_sha256": "d" * 64,
        "real_loader_gate": {"path": "gate", "sha256": "e" * 64},
        "ddp_pilot": {"path": "pilot", "sha256": "f" * 64},
        "post_runs": {
            variant: {
                "path": str(post_runs[variant].resolve()),
                "sha256": hashlib.sha256(
                    post_runs[variant].read_bytes()
                ).hexdigest(),
                "payload": json.loads(
                    post_runs[variant].read_text(encoding="utf-8")
                ),
                "training_audit": {
                    "path": str(
                        (tmp_path / variant / "training_audit.json").resolve()
                    ),
                    "sha256": hashlib.sha256(
                        (
                            tmp_path / variant / "training_audit.json"
                        ).read_bytes()
                    ).hexdigest(),
                    "slurm_job_id": 100 + index,
                },
            }
            for index, variant in enumerate(VARIANTS)
        },
    }
    return post_runs, costs, suite_binding


def test_training_cost_summary_binds_jobs_to_audits_and_reports_availability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    post_runs, costs, suite_binding = _cost_fixture(tmp_path)
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    payload = summarize_training_cost(
        expected_commit=COMMIT,
        suite_aggregate_path="aggregate.json",
        suite_aggregate_sha256="b" * 64,
        post_run_paths=post_runs,
        slurm_cost_paths=costs,
        suite_loader=lambda *_args, **_kwargs: suite_binding,
    )

    assert payload["training_profile"] == "official60"
    assert payload["training_protocol"]["end_epoch"] == 60
    assert payload["total_three_arm_allocation_gpu_hours"] == pytest.approx(3.5)
    assert payload["training_only_gpu_hours"]["available"] is False
    assert payload["availability"]["all_energy_measured"] is False
    assert payload["break_even"]["available"] is False


def test_training_cost_summary_rejects_job_identity_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    post_runs, costs, suite_binding = _cost_fixture(tmp_path)
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    path = costs["cellcf"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["job_id"] = 999
    payload.pop("record_sha256")
    payload["record_sha256"] = canonical_sha256(payload)
    _write(path, payload)

    with pytest.raises(ValueError, match="another job id"):
        summarize_training_cost(
            expected_commit=COMMIT,
            suite_aggregate_path="aggregate.json",
            suite_aggregate_sha256="b" * 64,
            post_run_paths=post_runs,
            slurm_cost_paths=costs,
            suite_loader=lambda *_args, **_kwargs: suite_binding,
        )


def test_training_cost_summary_rejects_rehashed_derived_cost_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    post_runs, costs, suite_binding = _cost_fixture(tmp_path)
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    path = costs["cellcf"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["allocation_elapsed_seconds"] += 1
    payload.pop("record_sha256")
    payload["record_sha256"] = canonical_sha256(payload)
    _write(path, payload)

    with pytest.raises(ValueError, match="replayed raw sacct data"):
        summarize_training_cost(
            expected_commit=COMMIT,
            suite_aggregate_path="aggregate.json",
            suite_aggregate_sha256="b" * 64,
            post_run_paths=post_runs,
            slurm_cost_paths=costs,
            suite_loader=lambda *_args, **_kwargs: suite_binding,
        )


def test_training_cost_summary_rejects_raw_sacct_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    post_runs, costs, suite_binding = _cost_fixture(tmp_path)
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    cost = json.loads(costs["uniform"].read_text(encoding="utf-8"))
    raw_path = Path(cost["sacct_raw_artifact_path"])
    raw_path.write_text(
        raw_path.read_text(encoding="utf-8").replace("|3600|", "|3601|"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw sacct artifact hash mismatch"):
        summarize_training_cost(
            expected_commit=COMMIT,
            suite_aggregate_path="aggregate.json",
            suite_aggregate_sha256="b" * 64,
            post_run_paths=post_runs,
            slurm_cost_paths=costs,
            suite_loader=lambda *_args, **_kwargs: suite_binding,
        )


def test_training_cost_shell_uses_submitted_ledger_and_three_bound_arms() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/summarize_duca_cellcf_training_cost.sh"
    ).read_text(encoding="utf-8")

    assert "jobs.submitted.tsv" in source
    assert "tools.bata.capture_duca_cellcf_slurm_cost" in source
    assert "tools.bata.summarize_duca_cellcf_training_cost" in source
    assert 'for key in ("uniform", "transition_beta0", "cellcf")' in source
    assert "must contain exactly one" in source
    assert "DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256" in source
    assert "--raw-output" in source
