from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.bata.duca_full_stack_cost import build_profile_summary
from tools.bata.duca_p0_evaluation import official_evaluator_identity
from tools.bata.plot_duca_r5_performance_cost import (
    COMPLETE_STATUS,
    UNAVAILABLE,
    build_performance_cost_report,
    export_performance_cost,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(payload: dict) -> dict:
    unsigned = dict(payload)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    payload["result_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _sample(scale: float, selected: float) -> dict:
    return {
        "input_pipeline_serial_ms": 1.0 * scale,
        "h2d_ms": 2.0 * scale,
        "model_forward_ms": 20.0 * scale,
        "postprocess_ms": 3.0 * scale,
        "frame_selector_total_ms": 4.0 * scale,
        "backbone_wrapper_total_ms": 10.0 * scale,
        "projection_ms": 2.0 * scale,
        "neck_ms": 1.0 * scale,
        "head_ms": 1.0 * scale,
        "coarse_probe_ms": 1.5 * scale,
        "heavy_backbone_ms": 8.0 * scale,
        "selected_count": selected,
        "peak_gpu_memory_mb": 100.0 * scale,
    }


def _profile(method: str, *, scale: float, selected: float, session: str, pair: str, order: int) -> dict:
    metadata = {
        "method": method,
        "protocol": "offline_full_window_runtime_selection",
        "hardware_fingerprint": "gpu",
        "host_fingerprint": "host",
        "software_fingerprint": "software",
        "config_commit": "a" * 40,
        "tracked_tree_clean": True,
        "dataset_fingerprint": "dataset",
        "inference_fingerprint": "inference",
        "detector_stack_fingerprint": "actionformer",
        "batch_size": 1,
        "loader_workers": 0,
        "warmup_samples": 1,
        "amp": True,
        "uses_ema": True,
        "random_init": False,
        "power_sampling_enabled": False,
        "power_interval_ms": 20,
        "power_gpu_id": None,
        "profile_session_id": session,
        "profile_pair_id": pair,
        "profile_repeat_index": 1,
        "profile_order_position": order,
    }
    profile = build_profile_summary([_sample(scale, selected)], metadata=metadata)
    if method != "dense-adatad":
        profile["r5_cost_binding"] = {"method": method}
        # Raw profile summaries produced by the actual profiler include this binding before rebuilding.
        profile["raw_samples"] = profile["raw_samples"]
    return profile


def _write_profile(path: Path, profile: dict) -> None:
    # The focused plot tool validates profile reconstruction. R5 binding is top-level profiler metadata in production.
    if "r5_cost_binding" in profile:
        binding = profile.pop("r5_cost_binding")
        metadata = {key: value for key, value in profile.items() if key not in {"schema_version", "sample_count", "stage_semantics", "stages", "selected_count", "resources", "energy", "claims", "raw_samples"}}
        metadata["r5_cost_binding"] = binding
        profile = build_profile_summary(profile["raw_samples"], metadata=metadata)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    rows = []
    candidate_refs = []
    dense_refs = []
    cost_paths = []
    for backend in ("actionformer", "temporalmaxer"):
        for arm in ("uniform", "learned"):
            for budget in (256, 384):
                for seed in (3407, 5801, 8123):
                    cell_id = f"{backend}_{arm}_k{budget}_s{seed}"
                    rows.append({
                        "id": cell_id,
                        "backend": backend,
                        "arm": arm,
                        "budget": budget,
                        "seed": seed,
                        "average_mAP": 50.0 + (1.0 if arm == "learned" else 0.0) + budget / 1000.0 + seed / 100000.0,
                        "iou_mAP": {f"mAP@{iou:.1f}": 40.0 + iou for iou in (0.3, 0.4, 0.5, 0.6, 0.7)},
                        "evaluator": official_evaluator_identity(),
                        "evaluation_config": {"subset": "validation", "blocked_videos": None},
                    })
                    if backend == "actionformer" and seed == 3407:
                        candidate_path = tmp_path / f"{cell_id}.summary.json"
                        dense_path = tmp_path / f"{cell_id}.dense.summary.json"
                        _write_profile(candidate_path, _profile(cell_id, scale=1.0, selected=float(budget), session="session", pair=cell_id, order=2))
                        _write_profile(dense_path, _profile("dense-adatad", scale=2.0, selected=768.0, session="session", pair=cell_id, order=1))
                        candidate_refs.append({"source_cell": cell_id, "backend": backend, "summary_path": str(candidate_path.resolve()), "summary_sha256": _sha(candidate_path)})
                        dense_refs.append({"backend": backend, "summary_path": str(dense_path.resolve()), "summary_sha256": _sha(dense_path)})
                        cost_paths.extend((candidate_path, dense_path))
    aggregate = _seal({
        "schema": "duca_r5_paper_matrix_results_v1",
        "ok": True,
        "status": COMPLETE_STATUS,
        "task": "offline_temporal_action_detection",
        "rows": rows,
        "costs": candidate_refs,
        "paired_dense_costs": dense_refs,
        "dense_baseline_receipt": {"backend": "actionformer"},
    })
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return aggregate_path, cost_paths


def test_exports_raw_rows_aggregates_and_figures_from_paired_fixture(tmp_path: Path) -> None:
    aggregate, profiles = _fixture(tmp_path)
    output = tmp_path / "out"
    report = export_performance_cost(aggregate_json=aggregate, raw_cost_summaries=profiles, output_dir=output)

    assert len(report["performance_rows"]) == 28
    assert len(report["performance_aggregates"]) == 8
    assert (output / "duca_r5_performance_cost_raw.csv").is_file()
    assert (output / "duca_r5_performance_cost_raw.tsv").is_file()
    assert (output / "duca_r5_performance_cost.json").is_file()
    assert (output / "duca_r5_performance_cost_figures.tex").is_file()
    temporalmaxer = next(row for row in report["performance_rows"] if row["backend"] == "temporalmaxer")
    assert temporalmaxer["total_latency_ms_p50"] == UNAVAILABLE
    actionformer = next(row for row in report["performance_rows"] if row["id"] == "actionformer_learned_k384_s3407")
    assert actionformer["FLOPs"] == UNAVAILABLE
    assert actionformer["speedup_vs_same_backend_dense"] == pytest.approx(2.0)
    if report["plot_status"] == "generated":
        assert (output / "duca_r5_map_vs_latency.png").is_file()
        assert (output / "duca_r5_map_vs_latency.pdf").is_file()


def test_rejects_nonofficial_map(tmp_path: Path) -> None:
    aggregate, profiles = _fixture(tmp_path)
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    payload["rows"][0]["evaluator"] = {"module": "fake"}
    payload.pop("result_sha256")
    aggregate.write_text(json.dumps(_seal(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not official mAP"):
        build_performance_cost_report(aggregate_json=aggregate, raw_cost_summaries=profiles)


def test_rejects_missing_or_cross_backend_raw_cost_set(tmp_path: Path) -> None:
    aggregate, profiles = _fixture(tmp_path)

    with pytest.raises(RuntimeError, match="exactly match"):
        build_performance_cost_report(aggregate_json=aggregate, raw_cost_summaries=profiles[:-1])

    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    payload["costs"][0]["backend"] = "temporalmaxer"
    payload.pop("result_sha256")
    aggregate.write_text(json.dumps(_seal(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cross-backend"):
        build_performance_cost_report(aggregate_json=aggregate, raw_cost_summaries=profiles)
