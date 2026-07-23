from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pytest

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
    stage_cell_relative_path,
    variant_spec,
)
from tools.bata.georoute_stage_runner import parse_official_style_map
from tools.bata.run_georoute_p0_gate import (
    GEOROUTE_P0_GATE_SCHEMA,
    build_p0_gate_report,
)


ROOT = Path(__file__).resolve().parents[1]


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
        "p0_scope": {"synthetic_inputs_only": True, "full_training": False, "official_evaluation": False},
    }
    return build_p0_gate_report(report)


def test_p0_suite_requires_dense_native_parity_and_both_scout_gradient_paths():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        payloads = {
            "dense.json": _p0_report(estimator="none", claim="no_policy_gradient", target_k=100, scout_gradient=False),
            "hybrid.json": _p0_report(estimator="straight_through", claim="biased_straight_through", target_k=32, scout_gradient=True),
            "score.json": _p0_report(estimator="score_function", claim="score_function_candidate", target_k=32, scout_gradient=True),
        }
        for name, payload in payloads.items():
            (root / name).write_text(json.dumps(payload), encoding="utf-8")
        summary = finalize(dense=root / "dense.json", hybrid=root / "hybrid.json", score_function=root / "score.json")
    assert summary["status"] == "PASS_MECHANICAL_ONLY"
    assert summary["suite_sha256"]


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
    for name, payload in payloads.items():
        (p0_dir / name).write_text(json.dumps(payload), encoding="utf-8")
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
        "random": _record(stage="p1", variant="random", seed=3407, high_iou=59.0, cost=12.0),
        "free": _record(stage="p1", variant="free", seed=3407, high_iou=62.0, cost=14.0),
        "roi": _record(stage="p1", variant="roi", seed=3407, high_iou=63.0, cost=14.0),
        "hybrid": _record(stage="p1", variant="hybrid", seed=3407, high_iou=64.0, cost=14.0),
    }
    p1_decision = select_p1_roi_candidate(p1)
    assert p1_decision["status"] == "ADVANCE_STRUCTURED_ROI_TO_P2"
    assert p1_decision["best_structured_variant"] == "hybrid"

    p2 = {}
    for variant, base_score, base_cost in (
        ("fixed_lattice", 60.0, 12.0),
        ("fixed_lattice_geometry", 61.0, 14.0),
        ("random", 59.0, 12.0),
        ("free", 62.0, 14.0),
        ("hybrid", 63.0, 14.0),
    ):
        p2[variant] = [
            _record(stage="p2", variant=variant, seed=seed, high_iou=base_score + 0.1 * index, cost=base_cost)
            for index, seed in enumerate(DEVELOPMENT_SEEDS)
        ]
    p2_decision = select_p2_roi_candidate(p2, candidate_variant="hybrid")
    assert p2_decision["status"] == "ADVANCE_STRUCTURED_ROI_TO_P3"
    assert p2_decision["official_test_opened"] is False


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
