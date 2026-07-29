from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tools.bata import georoute_rendezvous_gate as rdzv_gate
from tools.bata import georoute_stage_runner as stage_runner
from tools.bata import georoute_dag_dispatch as dag
from tools.bata.finalize_georoute_p0_gate import finalize
from tools.bata.georoute_dag_dispatch import GEOROUTE_STAGE_RESULT_SCHEMA
from tools.bata.georoute_experiment_contract import (
    DEVELOPMENT_SEEDS,
    PAPER_VARIANT_NAMES,
    P1_VARIANTS,
    canonical_sha256,
    paper_variant_name,
    select_p1_roi_candidate,
    select_p2_roi_candidate,
    sha256_file,
    stage_cell_relative_path,
    variant_spec,
)
from tools.bata.georoute_rendezvous_gate import (
    GEOROUTE_RENDEZVOUS_GATE_SCHEMA,
    READINESS_TIMEOUT_SECONDS,
    validate_rendezvous_gate_receipt,
)
from tools.bata.georoute_stage_runner import (
    _validate_rendezvous_receipt,
    build_torchrun_prefix,
    parse_official_style_map,
)
from tools.bata.run_georoute_p0_gate import (
    GEOROUTE_P0_GATE_SCHEMA,
    build_p0_gate_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage_runner_write_boundary_is_path_structural(tmp_path):
    boundary = (tmp_path / "yuzibo").resolve()
    assert stage_runner._inside((boundary / "pilot").resolve(), boundary)
    assert not stage_runner._inside(boundary, boundary)
    assert not stage_runner._inside(
        (tmp_path / "yuzibo_evil" / "pilot").resolve(),
        boundary,
    )


def test_run_logged_starts_session_and_cleans_failed_process_group(
    tmp_path,
    monkeypatch,
):
    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 1207001
        stdout = iter(("one line\n",))

        def wait(self, timeout=None):
            observed.setdefault("wait_timeouts", []).append(timeout)
            return 7

        def poll(self):
            return 7

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["start_new_session"] = kwargs.get("start_new_session")
        return FakeProcess()

    cleaned: list[FakeProcess] = []
    monkeypatch.setattr(stage_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        stage_runner,
        "_stop_logged_process_group",
        lambda process: cleaned.append(process),
    )
    with pytest.raises(RuntimeError, match="command failed with exit code 7"):
        stage_runner._run_logged(
            ["synthetic", "failure"],
            log_path=tmp_path / "train.out",
            env={},
        )
    assert observed["start_new_session"] is True
    assert cleaned
    assert "one line" in (tmp_path / "train.out").read_text(encoding="utf-8")


def _record(*, stage: str, variant: str, seed: int, high_iou: float, cost: float) -> dict:
    return {
        "schema_version": GEOROUTE_STAGE_RESULT_SCHEMA,
        "status": "PASS_DEVELOPMENT_ONLY",
        "stage": stage,
        "variant": variant,
        "seed": seed,
        "metrics": {"mAP@0.6": high_iou + 1.0, "mAP@0.7": high_iou - 1.0},
        "profile": {
            "development_window_wall_p50_ms": cost,
            "paper_grade_end_to_end_claim_allowed": False,
        },
        "official_test_opened": False,
    }


def _p0_report(*, estimator: str, claim: str, target_k: int, scout_gradient: bool) -> dict:
    route_mode = "dense" if estimator == "none" else "hybrid" if estimator == "straight_through" else "roi"
    components = {"rpn_head", "projection", "sparse_adapter", "videomae_adapter"}
    if scout_gradient:
        components.update(("scout_geometry", "scout_residual"))
    report = {
        "schema_version": GEOROUTE_P0_GATE_SCHEMA,
        "status": "PASS",
        "official_test_opened": False,
        "heavy_backbone_forward_count": 1,
        "shared_backbone_instances": 1,
        "uses_grid_sample": False,
        "uses_resized_local_crop": False,
        "exact_k": {"target_k": target_k, "observed_min": target_k, "observed_max": target_k, "duplicates": 0},
        "estimator": {"name": estimator, "claim": claim},
        "memory": {"peak_allocated_bytes": 4, "peak_reserved_bytes": 8},
        "losses": {"cost": 1.0},
        "gradient": {
            "all_required_gradients_finite": True,
            "nonzero_components": sorted(components),
            "required_components": sorted(components),
            "missing_required_components": [],
        },
        "detector": {
            "training_forward": True,
            "backward_completed": True,
            "output_length": 768,
            "detector_loss_keys": ["cls_loss", "reg_loss"],
        },
        "route_mode": route_mode,
        "source_grid": {"patch_capacity": 100},
        "native_route": {
            "selected_native_tubelet_shape": [1, 384, target_k, 3, 2, 16, 16],
            "output_shape": [1, 384, 768],
            "selected_unique_count_min": target_k,
            "selected_unique_count_max": target_k,
            "native_packed_invocation_counter_before": 7,
            "native_packed_invocation_counter_after": 8,
        },
        "dense_native_reference": (
            {
                "passed": True,
                "reference_heavy_backbone_forward_count": 1,
                "real_route_heavy_backbone_forward_count": 1,
                "reference_autograd_mode": "enabled_matches_real_packed_forward",
            }
            if route_mode == "dense"
            else None
        ),
        "score_function_detector_binding": (
            {"detector_loss_keys": ["cls_loss", "reg_loss"]}
            if estimator == "score_function"
            else None
        ),
        "component_trace": {
            "packed_attention_forward_count": 12,
            "packed_mlp_forward_count": 12,
            "packed_adapter_forward_count": 12,
            "dense_adapter_forward_count": 0,
            "adapter_execution": "coordinate_lineage_packed",
        },
        "checkpoint_receipt": {
            "checkpoint_count": 0,
            "policy": "p0_no_checkpoint",
        },
        "storage_receipt": {
            "status": "PASS_STORAGE_PREFLIGHT",
            "atomic_publish_peak_included": True,
        },
        "runtime_commit": "a" * 40,
        "checkpoint_storage_measurement": {
            "checkpoint_policy": "final_only",
            "checkpoint_upper_bound_bytes": 4096,
            "peak_checkpoint_copies_per_cell": 1,
            "auxiliary_upper_bound_bytes_per_cell": 2048,
            "stage_fixed_overhead_bytes": 1024,
            "safety_fraction": 0.25,
            "safety_bytes": 1024,
            "measurement_method": "unit_test",
        },
        "p0_scope": {"synthetic_inputs_only": True, "full_training": False, "official_evaluation": False},
    }
    return build_p0_gate_report(report)


def _rendezvous_receipt(*, slurm_job_id: str) -> dict:
    probes = {}
    for index, label in enumerate(("short", "long")):
        _, rendezvous = build_torchrun_prefix(
            phase="train",
            slurm_job_id=slurm_job_id,
            stage="p0",
            variant=f"rendezvous_probe_{label}",
            seed=3407,
            rendezvous_slot=1 if label == "short" else 0,
        )
        probes[label] = {
            "rendezvous": rendezvous,
            "runtime_identity": {
                "event": "GEOROUTE_RDZV_READY",
                "label": label,
                "rank": 0,
                "world_size": 1,
                "torchelastic_run_id": rendezvous["rendezvous_id"],
                "master_addr": "g0001",
                "master_port": str(45001 + index),
                "node_name": "g0001",
                "slurm_job_id": slurm_job_id,
            },
            "exit_code": 0,
            "ready_marker_seen": True,
            "done_marker_seen": True,
            "peer_exit_marker_seen": True,
            "output_sha256": ("a" if label == "short" else "b") * 64,
            "requested_post_release_seconds": 0.1,
        }
    core = {
        "schema_version": GEOROUTE_RENDEZVOUS_GATE_SCHEMA,
        "status": "PASS_CONCURRENT_RENDEZVOUS_ISOLATION",
        "runtime_commit": "a" * 40,
        "slurm_job_id": slurm_job_id,
        "node_name": "g0001",
        "same_node_concurrent": True,
        "long_probe_alive_after_short_exit": True,
        "release_to_short_exit_seconds": 0.6,
        "release_to_long_exit_seconds": 2.1,
        "readiness_timeout_seconds": READINESS_TIMEOUT_SECONDS,
        "elapsed_seconds": 3.0,
        "probes": probes,
        "official_test_opened": False,
        "model_forward_executed": False,
        "paper_claim_allowed": False,
    }
    return {**core, "gate_sha256": canonical_sha256(core)}


def _write_p0_bundle(
    report_path: Path,
    report: dict,
    *,
    slurm_job_id: str,
) -> None:
    rendezvous = _rendezvous_receipt(slurm_job_id=slurm_job_id)
    rendezvous_path = report_path.with_name(f"{report_path.stem}.rendezvous.json")
    rendezvous_path.write_text(
        json.dumps(rendezvous),
        encoding="utf-8",
    )
    bound_report = dict(report)
    bound_report["slurm_job_id"] = slurm_job_id
    bound_report["rendezvous_isolation"] = {
        "path": str(rendezvous_path.resolve()),
        "file_sha256": sha256_file(rendezvous_path),
        "gate_sha256": rendezvous["gate_sha256"],
        "slurm_job_id": slurm_job_id,
        "status": rendezvous["status"],
    }
    report_path.write_text(
        json.dumps(build_p0_gate_report(bound_report)),
        encoding="utf-8",
    )


def test_p0_suite_requires_dense_native_parity_and_both_scout_gradient_paths():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        payloads = {
            "dense.json": _p0_report(estimator="none", claim="no_policy_gradient", target_k=100, scout_gradient=False),
            "hybrid.json": _p0_report(estimator="straight_through", claim="biased_straight_through", target_k=32, scout_gradient=True),
            "score.json": _p0_report(estimator="score_function", claim="score_function_candidate", target_k=32, scout_gradient=True),
        }
        for index, (name, payload) in enumerate(payloads.items()):
            report_path = root / name
            _write_p0_bundle(
                report_path,
                payload,
                slurm_job_id=str(1000 + index),
            )
        summary = finalize(dense=root / "dense.json", hybrid=root / "hybrid.json", score_function=root / "score.json")
        dense_rendezvous = root / "dense.rendezvous.json"
        dense_original = dense_rendezvous.read_text(encoding="utf-8")
        dense_rendezvous.write_text(
            (root / "hybrid.rendezvous.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="same-leaf rendezvous receipt"):
            finalize(
                dense=root / "dense.json",
                hybrid=root / "hybrid.json",
                score_function=root / "score.json",
            )
        dense_rendezvous.write_text(dense_original, encoding="utf-8")
    assert summary["status"] == "PASS_MECHANICAL_ONLY"
    assert summary["suite_sha256"]
    assert summary["verified_properties"]["same_node_concurrent_rendezvous_isolation_passed"] is True
    assert summary["rendezvous_isolation"]["distinct_slurm_job_count"] == 3


def test_p1_bootstrap_reuses_a_sealed_p0_parent_and_only_submits_p1(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    p0_root = tmp_path / "p0_parent"
    p0_dir = p0_root / "p0"
    p0_dir.mkdir(parents=True)
    payloads = {
        "dense_native_parity.json": _p0_report(
            estimator="none", claim="no_policy_gradient", target_k=100, scout_gradient=False
        ),
        "hybrid_straight_through.json": _p0_report(
            estimator="straight_through", claim="biased_straight_through", target_k=32, scout_gradient=True
        ),
        "roi_score_function.json": _p0_report(
            estimator="score_function", claim="score_function_candidate", target_k=32, scout_gradient=True
        ),
    }
    for index, (name, payload) in enumerate(payloads.items()):
        report_path = p0_dir / name
        _write_p0_bundle(
            report_path,
            payload,
            slurm_job_id=str(2000 + index),
        )
    receipt = finalize(
        dense=p0_dir / "dense_native_parity.json",
        hybrid=p0_dir / "hybrid_straight_through.json",
        score_function=p0_dir / "roi_score_function.json",
    )
    (p0_root / "control").mkdir()
    (p0_root / "control" / "p0_finalization.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )

    source_config = tmp_path / "source.py"
    manifest = tmp_path / "manifest.json"
    annotation = tmp_path / "development.json"
    class_map = tmp_path / "class_map.txt"
    pretrained = tmp_path / "pretrained.pth"
    for path in (source_config, manifest, annotation, class_map, pretrained):
        path.write_text("placeholder", encoding="utf-8")
    video_root = tmp_path / "validation_videos"
    video_root.mkdir()
    run_root = tmp_path / "p1_p2_p3"
    args = argparse.Namespace(
        run_root=run_root,
        p0_run_root=p0_root,
        source_config=source_config,
        manifest=manifest,
        development_annotation=annotation,
        class_map=class_map,
        development_video_root=video_root,
        pretrained=pretrained,
        expected_commit="a" * 40,
    )
    captured: dict[str, object] = {}

    def fake_submit_stage_matrix(**kwargs):
        captured.update(kwargs)
        return {"georoute_p1_dense_native_s3407": "12345"}

    monkeypatch.setattr(dag, "_submit_stage_matrix", fake_submit_stage_matrix)
    assert dag._p1_bootstrap(args) == 0

    intent = json.loads((run_root / "control" / "p1_bootstrap.json").read_text(encoding="utf-8"))
    assert captured["stage"] == "p1"
    assert captured["next_action"] == "p1-select"
    assert captured["parent_receipt"] == receipt["suite_sha256"]
    assert captured["cells"] == [(variant, 3407, None) for variant in P1_VARIANTS]
    assert intent["p0_parent"]["suite_sha256"] == receipt["suite_sha256"]
    assert intent["frozen_successor_policy"]["p2"].startswith("submit only when p1-select")
    assert not (run_root / "p2").exists()
    assert not (run_root / "p3").exists()


def test_stage_matrix_uses_scheduler_test_only_for_every_leaf_before_submission(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    for name in ("config.py", "manifest.json", "annotation.json", "class_map.txt", "pretrained.pth"):
        (tmp_path / name).write_text("placeholder", encoding="utf-8")
    video_root = tmp_path / "validation_videos"
    video_root.mkdir()
    args = argparse.Namespace(
        run_root=tmp_path / "run",
        source_config=tmp_path / "config.py",
        manifest=tmp_path / "manifest.json",
        development_annotation=tmp_path / "annotation.json",
        class_map=tmp_path / "class_map.txt",
        development_video_root=video_root,
        pretrained=tmp_path / "pretrained.pth",
        expected_commit="a" * 40,
    )
    calls: list[tuple[str, bool]] = []

    def fake_sbatch(*, name, test_only=False, **_kwargs):
        calls.append((name, test_only))
        return "TEST_ONLY_PASS" if test_only else str(1000 + len(calls))

    monkeypatch.setattr(dag, "_sbatch", fake_sbatch)
    monkeypatch.setattr(
        dag,
        "storage_capacity_receipt",
        lambda *_args, **_kwargs: {"status": "PASS_STORAGE_PREFLIGHT"},
    )
    jobs = dag._submit_stage_matrix(
        args=args,
        stage="p1",
        cells=[("dense_native", 3407, None), ("roi", 3407, None)],
        parent_receipt="parent",
        next_action="p1-select",
    )

    assert calls == [
        ("georoute_p1_dense_native_s3407", True),
        ("georoute_p1_roi_s3407", True),
        ("georoute_p1_select", True),
        ("georoute_p1_dense_native_s3407", False),
        ("georoute_p1_roi_s3407", False),
        ("georoute_p1_select", False),
    ]
    assert set(jobs) == {
        "georoute_p1_dense_native_s3407",
        "georoute_p1_roi_s3407",
        "p1-select_dispatcher",
    }


def test_submit_capacity_rejects_a_matrix_before_any_leaf_is_created(monkeypatch):
    class Result:
        def __init__(self, *, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **_kwargs):
        if command[0] == "squeue":
            return Result(stdout="\n".join("active" for _ in range(9)) + "\n")
        assert command[0] == "sacctmgr"
        return Result(stdout="16|\n")

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_USER", "sczc063")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "sczc063")
    monkeypatch.setattr(dag.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="active=9, required_additional=8, MaxSubmitJobs=16"):
        dag._require_submit_capacity(additional_jobs=8)


def test_stage_matrix_cancels_submitted_leaves_when_selector_submission_fails(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    for name in ("config.py", "manifest.json", "annotation.json", "class_map.txt", "pretrained.pth"):
        (tmp_path / name).write_text("placeholder", encoding="utf-8")
    video_root = tmp_path / "validation_videos"
    video_root.mkdir()
    args = argparse.Namespace(
        run_root=tmp_path / "run",
        source_config=tmp_path / "config.py",
        manifest=tmp_path / "manifest.json",
        development_annotation=tmp_path / "annotation.json",
        class_map=tmp_path / "class_map.txt",
        development_video_root=video_root,
        pretrained=tmp_path / "pretrained.pth",
        expected_commit="a" * 40,
    )
    submitted: list[str] = []
    cancelled: list[str] = []

    def fake_sbatch(*, name, test_only=False, **_kwargs):
        if test_only:
            return "TEST_ONLY_PASS"
        if name == "georoute_p1_select":
            raise RuntimeError("selector rejected")
        job_id = str(2000 + len(submitted))
        submitted.append(job_id)
        return job_id

    monkeypatch.setattr(dag, "_sbatch", fake_sbatch)
    monkeypatch.setattr(dag, "_cancel_submitted_jobs", lambda job_ids: cancelled.extend(job_ids))
    monkeypatch.setattr(
        dag,
        "storage_capacity_receipt",
        lambda *_args, **_kwargs: {"status": "PASS_STORAGE_PREFLIGHT"},
    )

    with pytest.raises(RuntimeError, match="selector rejected"):
        dag._submit_stage_matrix(
            args=args,
            stage="p1",
            cells=[("dense_native", 3407, None), ("roi", 3407, None)],
            parent_receipt="parent",
            next_action="p1-select",
        )
    assert cancelled == submitted == ["2000", "2001"]


def test_p1_and_p2_selection_are_predeclared_and_result_blind():
    p1 = {
        "dense_native": _record(stage="p1", variant="dense_native", seed=3407, high_iou=64.0, cost=50.0),
        "fixed_lattice": _record(stage="p1", variant="fixed_lattice", seed=3407, high_iou=60.0, cost=12.0),
        "fixed_lattice_geometry": _record(
            stage="p1", variant="fixed_lattice_geometry", seed=3407, high_iou=61.0, cost=14.0
        ),
        "random": _record(stage="p1", variant="random", seed=3407, high_iou=59.0, cost=14.0),
        "free": _record(stage="p1", variant="free", seed=3407, high_iou=62.0, cost=14.0),
        "roi": _record(stage="p1", variant="roi", seed=3407, high_iou=63.0, cost=14.0),
        "hybrid": _record(stage="p1", variant="hybrid", seed=3407, high_iou=64.0, cost=14.0),
    }
    p1_decision = select_p1_roi_candidate(p1)
    assert p1_decision["status"] == "ADVANCE_HYBRID_TO_P2"
    assert p1_decision["selected_variant"] == "hybrid"

    p2 = {}
    for variant, base_score, base_cost in (
        ("fixed_lattice", 60.0, 15.0),
        ("fixed_lattice_geometry", 61.0, 14.0),
        ("random", 59.0, 15.0),
        ("free", 62.0, 14.0),
        ("hybrid", 63.0, 14.0),
    ):
        p2[variant] = [
            _record(stage="p2", variant=variant, seed=seed, high_iou=base_score + 0.1 * index, cost=base_cost)
            for index, seed in enumerate(DEVELOPMENT_SEEDS)
        ]
    p2_decision = select_p2_roi_candidate(p2, candidate_variant="hybrid")
    assert p2_decision["status"] == "ADVANCE_GEOMETRY_ROUTE_A_TO_P3"
    assert p2_decision["official_test_opened"] is False


def test_p1_selector_can_advance_native_route_b_and_random_can_stop_learning():
    def records(scores):
        return {
            variant: _record(
                stage="p1",
                variant=variant,
                seed=3407,
                high_iou=score,
                cost={
                    "dense_native": 40.0,
                    "fixed_lattice": 12.0,
                    "fixed_lattice_geometry": 14.0,
                    "random": 12.0,
                    "free": 14.0,
                    "roi": 15.0,
                    "hybrid": 15.0,
                }[variant],
            )
            for variant, score in scores.items()
        }

    native = records(
        {
            "dense_native": 66.0,
            "fixed_lattice": 60.0,
            "fixed_lattice_geometry": 61.0,
            "random": 59.0,
            "free": 64.0,
            "roi": 63.0,
            "hybrid": 62.0,
        }
    )
    decision = select_p1_roi_candidate(native)
    assert decision["status"] == "ADVANCE_FREE_TO_P2"
    assert decision["selected_route"] == "B"
    assert decision["selected_variant"] == "free"

    random_wins = records(
        {
            "dense_native": 66.0,
            "fixed_lattice": 60.0,
            "fixed_lattice_geometry": 61.0,
            "random": 65.0,
            "free": 64.0,
            "roi": 63.0,
            "hybrid": 62.0,
        }
    )
    stopped = select_p1_roi_candidate(random_wins)
    assert stopped["status"] == "STOP_LEARNED_ROUTING"
    assert stopped["selected_variant"] is None

    geometry_cannot_rescue_failed_native = records(
        {
            "dense_native": 66.0,
            "fixed_lattice": 64.0,
            "fixed_lattice_geometry": 63.0,
            "random": 62.0,
            "free": 61.0,
            "roi": 65.0,
            "hybrid": 67.0,
        }
    )
    ambiguous = select_p1_roi_candidate(geometry_cannot_rescue_failed_native)
    assert ambiguous["native_accuracy_gate"] is False
    assert ambiguous["hybrid_accuracy_gate"] is False
    assert ambiguous["status"] == "STOP_AMBIGUOUS_NO_PREDECLARED_WINNER"
    assert ambiguous["selected_variant"] is None
    assert ambiguous["selected_route"] == "C"


def test_paper_names_are_frozen_and_log_parser_requires_all_iou_thresholds():
    assert paper_variant_name("hybrid") == "roi_residual"
    assert PAPER_VARIANT_NAMES["free"] == "free_token_select"
    assert variant_spec("fixed_lattice_geometry")["geometry_side_channel"] is True
    metrics = parse_official_style_map(
        "\n".join(
            [
                "Average-mAP: 66.50 (%)",
                "mAP at tIoU 0.30 is 80.00%",
                "mAP at tIoU 0.40 is 75.00%",
                "mAP at tIoU 0.50 is 69.00%",
                "mAP at tIoU 0.60 is 60.00%",
                "mAP at tIoU 0.70 is 48.00%",
            ]
        )
    )
    assert metrics["average_mAP"] == 66.5
    assert metrics["mAP@0.7"] == 48.0


def test_georoute_torchrun_uses_job_scoped_kernel_assigned_rendezvous():
    commands = {}
    receipts = {}
    for phase in ("train", "test"):
        command, receipt = build_torchrun_prefix(
            phase=phase,
            slurm_job_id="1199999",
            stage="p1",
            variant="free",
            seed=3407,
        )
        commands[phase] = command
        receipts[phase] = receipt
        joined = " ".join(command)
        assert "--standalone" not in joined
        assert "--master_port" not in joined
        assert "--rdzv_backend=c10d" in command
        assert f"--rdzv_endpoint={receipt['endpoint']}" in command
        assert receipt["endpoint"].startswith("127.")
        assert receipt["endpoint"].endswith(":0")
        assert (
            receipt["endpoint_policy"]
            == "job_scoped_loopback_and_kernel_assigned_port"
        )
        assert f"--rdzv_id=georoute-1199999-p1-free-s3407-{phase}" in command
    validated = _validate_rendezvous_receipt(
        receipts,
        stage="p1",
        variant="free",
        seed=3407,
    )
    assert validated["train"]["rendezvous_id"] != validated["test"]["rendezvous_id"]

    _, other_leaf = build_torchrun_prefix(
        phase="train",
        slurm_job_id="1200000",
        stage="p1",
        variant="free",
        seed=3407,
    )
    assert other_leaf["rendezvous_id"] != receipts["train"]["rendezvous_id"]
    assert other_leaf["endpoint_host"] != receipts["train"]["endpoint_host"]
    with pytest.raises(ValueError, match="unsafe GeoRoute rendezvous slurm_job_id"):
        build_torchrun_prefix(
            phase="train",
            slurm_job_id="1199999,1200000",
            stage="p1",
            variant="free",
            seed=3407,
        )


def test_georoute_rendezvous_gate_receipt_fails_closed_on_store_reuse():
    receipt = _rendezvous_receipt(slurm_job_id="1199999")
    validate_rendezvous_gate_receipt(receipt, expected_commit="a" * 40)
    with pytest.raises(ValueError, match="differs from the current leaf"):
        validate_rendezvous_gate_receipt(
            receipt,
            expected_commit="a" * 40,
            expected_node_name="g9999",
        )

    reused = copy.deepcopy(receipt)
    reused["probes"]["long"]["runtime_identity"]["master_port"] = reused[
        "probes"
    ]["short"]["runtime_identity"]["master_port"]
    core = dict(reused)
    core.pop("gate_sha256")
    reused["gate_sha256"] = canonical_sha256(core)
    with pytest.raises(ValueError, match="reused an actual TCPStore port"):
        validate_rendezvous_gate_receipt(reused, expected_commit="a" * 40)

    no_peer_exit = copy.deepcopy(receipt)
    no_peer_exit["probes"]["long"]["peer_exit_marker_seen"] = False
    core = dict(no_peer_exit)
    core.pop("gate_sha256")
    no_peer_exit["gate_sha256"] = canonical_sha256(core)
    with pytest.raises(ValueError, match="long rendezvous probe did not pass"):
        validate_rendezvous_gate_receipt(
            no_peer_exit,
            expected_commit="a" * 40,
        )


def test_georoute_rendezvous_failure_writes_hashed_diagnostics_and_stops_groups(
    tmp_path,
    monkeypatch,
):
    expected_commit = "a" * 40
    output = tmp_path / "run" / "p0" / "probe.rendezvous.json"
    monkeypatch.setenv("SLURM_JOB_ID", "1206999")
    monkeypatch.setattr(
        rdzv_gate,
        "_git_output",
        lambda *arguments: (
            expected_commit if arguments == ("rev-parse", "HEAD") else ""
        ),
    )

    def fake_prefix(**kwargs):
        return (
            [sys.executable, "-c", "import time; time.sleep(60)"],
            {
                "phase": kwargs["phase"],
                "backend": "c10d",
                "endpoint": "127.1.1.1:0",
                "endpoint_host": "127.1.1.1",
                "endpoint_policy": (
                    "job_scoped_loopback_and_kernel_assigned_port"
                ),
                "rendezvous_slot": kwargs["rendezvous_slot"],
                "rendezvous_id": f"synthetic-{kwargs['variant']}",
                "slurm_job_id": kwargs["slurm_job_id"],
                "stage": kwargs["stage"],
                "variant": kwargs["variant"],
                "seed": kwargs["seed"],
                "nnodes": 1,
                "nproc_per_node": 1,
            },
        )

    monkeypatch.setattr(rdzv_gate, "build_torchrun_prefix", fake_prefix)
    monkeypatch.setattr(
        rdzv_gate,
        "_wait_until_ready",
        lambda **_: (_ for _ in ()).throw(
            TimeoutError("synthetic rendezvous timeout")
        ),
    )
    original_popen = subprocess.Popen
    processes = []

    def recording_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(rdzv_gate.subprocess, "Popen", recording_popen)
    with pytest.raises(RuntimeError, match="rendezvous failure receipt"):
        rdzv_gate.run_gate(
            output=output,
            expected_commit=expected_commit,
            write_boundary=tmp_path,
        )
    failure_path = output.with_suffix(".failure.json")
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    digest = payload.pop("failure_sha256")
    assert payload["status"] == "FAIL_CONCURRENT_RENDEZVOUS_ISOLATION"
    assert payload["exception_type"] == "TimeoutError"
    assert payload["model_forward_executed"] is False
    assert payload["paper_claim_allowed"] is False
    assert set(payload["probes"]) == {"short", "long"}
    assert all(
        "output_sha256" in probe and "output_tail" in probe
        for probe in payload["probes"].values()
    )
    assert digest == canonical_sha256(payload)
    assert processes and all(process.poll() is not None for process in processes)


def test_georoute_rendezvous_prevalidation_failure_is_sealed(
    tmp_path,
    monkeypatch,
):
    expected_commit = "a" * 40
    output = tmp_path / "run" / "p0" / "preflight.rendezvous.json"
    monkeypatch.setattr(
        rdzv_gate,
        "_git_output",
        lambda *_: "b" * 40,
    )
    rdzv_gate._write_gate_failsafe_failure(
        output=output,
        expected_commit=expected_commit,
        error=RuntimeError("synthetic clean-tree mismatch"),
        write_boundary=tmp_path,
    )
    failure_path = output.with_suffix(".failure.json")
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    digest = payload.pop("failure_sha256")
    assert payload["failure_phase"] == "gate_prevalidation_or_namespace_setup"
    assert payload["expected_runtime_commit"] == expected_commit
    assert payload["probes"] == {}
    assert payload["model_forward_executed"] is False
    assert payload["paper_claim_allowed"] is False
    assert digest == canonical_sha256(payload)


def test_p0_launcher_runs_rendezvous_isolation_before_model_gate():
    launcher = (ROOT / "scripts" / "run_georoute_p0_slurm.sh").read_text(
        encoding="utf-8"
    )
    runner = (
        ROOT / "tools" / "bata" / "georoute_stage_runner.py"
    ).read_text(encoding="utf-8")
    assert "--standalone" not in runner
    assert "job_scoped_loopback_and_kernel_assigned_port" in runner
    assert "tools.bata.georoute_rendezvous_gate" in launcher
    assert launcher.index("tools.bata.georoute_rendezvous_gate") < launcher.index(
        "tools.bata.run_georoute_p0_gate"
    )


def test_p3_cell_namespaces_include_exact_k_to_prevent_budget_curve_overwrites():
    k32 = stage_cell_relative_path(stage="p3", variant="hybrid", seed=3407, token_budget=32)
    k64 = stage_cell_relative_path(stage="p3", variant="hybrid", seed=3407, token_budget=64)
    assert k32 != k64
    assert str(k32).replace("\\", "/") == "p3/hybrid/k32/seed3407"
    assert str(k64).replace("\\", "/") == "p3/hybrid/k64/seed3407"


def test_gpu_submission_uses_n16r4_outer_resources_and_exact_inner_step():
    deployer = (ROOT / "tools" / "bata" / "deploy_georoute_development_dag.py").read_text(
        encoding="utf-8"
    )
    dispatcher = (ROOT / "tools" / "bata" / "georoute_dag_dispatch.py").read_text(
        encoding="utf-8"
    )
    p0_launcher = (ROOT / "scripts" / "run_georoute_p0_slurm.sh").read_text(encoding="utf-8")
    stage_launcher = (ROOT / "scripts" / "run_georoute_stage_slurm.sh").read_text(encoding="utf-8")

    for source in (deployer, dispatcher):
        assert 'GPU_OUTER_SLURM_ARGS = ("--gpus", "2", "--cpus-per-task", "8")' in source
        assert 'CONTROL_SLURM_ARGS = ("--gpus", "1", "--cpus-per-task", "1")' in source
        assert '"--mem", "4G"' not in source
        assert '"--mem", "96G"' not in source
    for source in (p0_launcher, stage_launcher):
        assert "srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M" in source


def test_p0_finalizer_launcher_seals_p0_without_dispatching_development_stages():
    launcher = (ROOT / "scripts" / "run_georoute_p0_finalize_slurm.sh").read_text(encoding="utf-8")
    assert "python -m tools.bata.finalize_georoute_p0_gate" in launcher
    assert "python tools/bata/finalize_georoute_p0_gate.py" not in launcher
    assert "georoute_dag_dispatch.py" not in launcher
    assert "deploy_georoute_development_dag.py" not in launcher
    assert "run_georoute_stage_slurm.sh" not in launcher


def test_dispatch_and_stage_launchers_use_package_entrypoints_and_accept_p1_bootstrap():
    dispatcher = (ROOT / "tools" / "bata" / "georoute_dag_dispatch.py").read_text(encoding="utf-8")
    dispatch_launcher = (ROOT / "scripts" / "run_georoute_dispatch_slurm.sh").read_text(encoding="utf-8")
    stage_launcher = (ROOT / "scripts" / "run_georoute_stage_slurm.sh").read_text(encoding="utf-8")

    assert '"p1-bootstrap"' in dispatcher
    assert 'next_action="p1-select"' in dispatcher
    assert 'next_action="p2-select"' in dispatcher
    assert 'next_action="p3-finalize"' in dispatcher
    assert "python -m tools.bata.georoute_dag_dispatch" in dispatch_launcher
    assert "--p0-run-root" in dispatch_launcher
    assert "python -m tools.bata.georoute_stage_runner" in stage_launcher
