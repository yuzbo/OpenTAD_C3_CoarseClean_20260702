import json
from pathlib import Path

import pytest

from tools.bata.duca_full_stack_cost import (
    ModuleStageHooks,
    MethodStageHooks,
    StageRecorder,
    OFFLINE_FULL_WINDOW_PROTOCOL,
    build_profile_summary,
    build_cost_matrix,
    compare_profile_summaries,
    integrate_power_samples,
    validate_and_rebuild_profile_summary,
    write_profile_artifacts,
    write_cost_matrix_artifacts,
)


def _sample(scale: float = 1.0, *, selected_count: int = 384) -> dict:
    return {
        "input_pipeline_serial_ms": 20.0 * scale,
        "h2d_ms": 5.0 * scale,
        "model_forward_ms": 100.0 * scale,
        "postprocess_ms": 10.0 * scale,
        "frame_selector_total_ms": 12.0 * scale,
        "coarse_probe_ms": 4.0 * scale,
        "backbone_wrapper_total_ms": 70.0 * scale,
        "heavy_backbone_ms": 60.0 * scale,
        "projection_ms": 8.0 * scale,
        "neck_ms": 4.0 * scale,
        "head_ms": 5.0 * scale,
        "peak_gpu_memory_mb": 4096.0,
        "selected_count": selected_count,
        "gpu_energy_j": 30.0 * scale,
    }


def test_profile_and_matrix_writers_refuse_to_overwrite(
    tmp_path: Path,
) -> None:
    report = build_profile_summary(
        [_sample()], metadata=_metadata("candidate")
    )
    profile_prefix = tmp_path / "profile"
    profile_paths = write_profile_artifacts(report, profile_prefix)
    profile_json = profile_paths["json"].read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_profile_artifacts(report, profile_prefix)
    assert profile_paths["json"].read_bytes() == profile_json

    baseline = build_profile_summary(
        [_sample(1.2)], metadata=_metadata("baseline")
    )
    matrix = build_cost_matrix(baseline, [report])
    matrix_prefix = tmp_path / "matrix"
    matrix_paths = write_cost_matrix_artifacts(matrix, matrix_prefix)
    matrix_json = matrix_paths["json"].read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_cost_matrix_artifacts(matrix, matrix_prefix)
    assert matrix_paths["json"].read_bytes() == matrix_json


def _metadata(name: str) -> dict:
    return {
        "method": name,
        "protocol": OFFLINE_FULL_WINDOW_PROTOCOL,
        "hardware_fingerprint": "n16r4-a100-gpu1",
        "host_fingerprint": "same-node-cpu-os",
        "software_fingerprint": "same-python-torch-decode-stack",
        "config_commit": "deadbeef",
        "tracked_tree_clean": True,
        "dataset_fingerprint": "same-test-dataset-and-pipeline",
        "inference_fingerprint": "same-inference-and-postprocess",
        "detector_stack_fingerprint": "same-heavy-detector-schema",
        "batch_size": 1,
        "loader_workers": 0,
        "warmup_samples": 5,
        "amp": True,
        "uses_ema": True,
        "random_init": False,
        "power_sampling_enabled": True,
        "power_interval_ms": 20,
        "power_gpu_id": "1",
    }


def test_profile_summary_reports_full_stack_cost_and_nonoverlapping_residuals() -> None:
    report = build_profile_summary(
        [_sample(1.0), _sample(2.0), _sample(3.0)],
        metadata=_metadata("duca-fixed384"),
    )

    assert report["protocol"] == OFFLINE_FULL_WINDOW_PROTOCOL
    assert report["sample_count"] == 3
    assert report["stages"]["end_to_end_serial_ms"]["p50"] == pytest.approx(270.0)
    assert report["stages"]["end_to_end_serial_ms"]["p95"] == pytest.approx(391.5)
    assert report["stages"]["selector_policy_ms"]["p50"] == pytest.approx(16.0)
    assert report["stages"]["backbone_wrapper_overhead_ms"]["p50"] == pytest.approx(20.0)
    assert report["stages"]["model_unattributed_ms"]["p50"] == pytest.approx(2.0)
    assert report["selected_count"]["p50"] == pytest.approx(384.0)
    assert report["energy"]["gpu_energy_j"]["p50"] == pytest.approx(60.0)
    assert report["claims"]["full_stack_latency_measured"] is True
    assert report["claims"]["decoder_and_preprocess_separated"] is False


def test_profile_summary_must_reconstruct_exactly_from_raw_samples() -> None:
    report = build_profile_summary(
        [_sample(1.0), _sample(2.0)],
        metadata=_metadata("duca-fixed384"),
    )

    fingerprints = validate_and_rebuild_profile_summary(report)

    assert len(fingerprints["ordered_sha256"]) == 64
    assert len(fingerprints["multiset_sha256"]) == 64
    tampered = json.loads(json.dumps(report))
    tampered["stages"]["end_to_end_serial_ms"]["p50"] += 1.0
    with pytest.raises(ValueError, match="does not reconstruct exactly"):
        validate_and_rebuild_profile_summary(tampered)


def test_profile_summary_rejects_missing_or_inconsistent_stage_costs() -> None:
    missing = _sample()
    missing.pop("h2d_ms")
    with pytest.raises(ValueError, match="h2d_ms"):
        build_profile_summary([missing], metadata=_metadata("broken"))

    overlapping = _sample()
    overlapping["frame_selector_total_ms"] = 60.0
    overlapping["backbone_wrapper_total_ms"] = 70.0
    with pytest.raises(ValueError, match="model_forward_ms"):
        build_profile_summary([overlapping], metadata=_metadata("broken"))

    unsupported = _sample()
    unsupported["repeat_nonce"] = "metadata-disguised-as-a-sample"
    with pytest.raises(ValueError, match="unsupported fields"):
        build_profile_summary([unsupported], metadata=_metadata("broken"))


def test_compare_profiles_enforces_protocol_and_hardware_and_reports_cost_gates() -> None:
    baseline = build_profile_summary([_sample(2.0, selected_count=768)], metadata=_metadata("dense768"))
    candidate = build_profile_summary([_sample(1.0)], metadata=_metadata("duca-fixed384"))

    comparison = compare_profile_summaries(baseline, candidate)

    assert comparison["comparable"] is True
    assert comparison["end_to_end_serial"]["latency_saving_fraction"] == pytest.approx(0.5)
    assert comparison["end_to_end_serial"]["speedup"] == pytest.approx(2.0)
    assert comparison["end_to_end_serial"]["p95_latency_saving_fraction"] == pytest.approx(0.5)
    assert comparison["heavy_backbone"]["latency_saving_ms"] == pytest.approx(60.0)
    assert comparison["frontend_overhead"]["candidate_ms"] == pytest.approx(12.0)
    assert comparison["gates"]["end_to_end_saving_at_least_15pct"] is True
    assert comparison["gates"]["frontend_consumes_at_most_40pct_of_backbone_saving"] is True

    incompatible = json.loads(json.dumps(candidate))
    incompatible["hardware_fingerprint"] = "different-gpu"
    with pytest.raises(ValueError, match="hardware_fingerprint"):
        compare_profile_summaries(baseline, incompatible)

    incompatible_commit = json.loads(json.dumps(candidate))
    incompatible_commit["config_commit"] = "different-commit"
    with pytest.raises(ValueError, match="config_commit"):
        compare_profile_summaries(baseline, incompatible_commit)

    for key, value in (
        ("batch_size", 8),
        ("amp", False),
        ("dataset_fingerprint", "different-data"),
        ("inference_fingerprint", "different-postprocess"),
        ("detector_stack_fingerprint", "different-detector"),
        ("power_interval_ms", 100),
        ("power_gpu_id", "0"),
    ):
        incompatible_protocol = json.loads(json.dumps(candidate))
        incompatible_protocol[key] = value
        with pytest.raises(ValueError, match=key):
            compare_profile_summaries(baseline, incompatible_protocol)

    random_init = json.loads(json.dumps(candidate))
    random_init["random_init"] = True
    with pytest.raises(ValueError, match="random_init"):
        compare_profile_summaries(baseline, random_init)

    dirty = json.loads(json.dumps(candidate))
    dirty["tracked_tree_clean"] = False
    with pytest.raises(ValueError, match="tracked_tree_clean"):
        compare_profile_summaries(baseline, dirty)


def test_compare_profiles_allows_different_sampling_pipelines_for_same_source_dataset() -> None:
    baseline_meta = _metadata("bare-uniform384")
    candidate_meta = _metadata("cellcf384")
    baseline_meta["dataset_fingerprint"] = "canonical-predecode-384"
    candidate_meta["dataset_fingerprint"] = "decode-768-then-cellcf"
    baseline_meta["source_dataset_fingerprint"] = "thumos14-test-source"
    candidate_meta["source_dataset_fingerprint"] = "thumos14-test-source"
    baseline = build_profile_summary([_sample(1.0)], metadata=baseline_meta)
    candidate = build_profile_summary([_sample(1.1)], metadata=candidate_meta)

    assert compare_profile_summaries(baseline, candidate)["comparable"] is True

    candidate["source_dataset_fingerprint"] = "different-source"
    with pytest.raises(ValueError, match="source_dataset_fingerprint"):
        compare_profile_summaries(baseline, candidate)


def test_cost_matrix_writes_raw_comparisons_and_gate_table(tmp_path) -> None:
    baseline = build_profile_summary([_sample(2.0, selected_count=768)], metadata=_metadata("dense768"))
    fixed384 = build_profile_summary([_sample(1.0)], metadata=_metadata("duca-fixed384"))
    uniform384 = build_profile_summary([_sample(1.2)], metadata=_metadata("uniform384"))

    matrix = build_cost_matrix(baseline, [fixed384, uniform384])
    assert matrix["baseline_method"] == "dense768"
    assert [row["candidate_method"] for row in matrix["comparisons"]] == [
        "duca-fixed384",
        "uniform384",
    ]
    assert matrix["comparisons"][0]["gates"]["all_cost_gates_pass"] is True

    paths = write_cost_matrix_artifacts(matrix, tmp_path / "cost_matrix")
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["baseline_method"] == "dense768"
    tsv = paths["tsv"].read_text(encoding="utf-8")
    assert "candidate_method\tselected_count_p50\te2e_p50_ms" in tsv
    assert "duca-fixed384" in tsv


def test_power_samples_are_integrated_in_joules() -> None:
    energy = integrate_power_samples(
        [(0.0, 100.0), (0.5, 100.0), (1.0, 100.0)],
        start_time_s=0.0,
        end_time_s=1.0,
    )

    assert energy["sample_count"] == 3
    assert energy["interior_sample_count"] == 1
    assert energy["window_bracketed"] is True
    assert energy["average_power_w"] == pytest.approx(100.0)
    assert energy["energy_j"] == pytest.approx(100.0)

    with pytest.raises(ValueError, match="bracket"):
        integrate_power_samples([(0.0, 100.0)], start_time_s=10.0, end_time_s=11.0)
    with pytest.raises(ValueError, match="bracket"):
        integrate_power_samples([(10.5, 100.0), (11.5, 100.0)], start_time_s=10.0, end_time_s=11.0)


def test_profile_artifacts_are_auditable_json_and_tsv(tmp_path) -> None:
    report = build_profile_summary([_sample()], metadata=_metadata("duca-fixed384"))

    paths = write_profile_artifacts(report, tmp_path / "duca_fixed384")

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["method"] == "duca-fixed384"
    tsv = paths["tsv"].read_text(encoding="utf-8")
    assert "stage\tcount\tmean_ms\tp50_ms\tp95_ms" in tsv
    assert "end_to_end_serial_ms" in tsv


def test_stage_recorder_accumulates_repeated_nested_measurements() -> None:
    ticks = iter([1.0, 1.1, 2.0, 2.2, 3.0, 3.3])
    sync_calls = []
    recorder = StageRecorder(clock=lambda: next(ticks), synchronize=lambda: sync_calls.append(True))
    recorder.begin_sample()

    with recorder.measure("coarse_probe_ms"):
        pass
    with recorder.measure("coarse_probe_ms"):
        pass
    with recorder.measure("head_ms"):
        pass

    sample = recorder.end_sample()
    assert sample["coarse_probe_ms"] == pytest.approx(300.0)
    assert sample["head_ms"] == pytest.approx(300.0)
    assert len(sync_calls) == 6


def test_stage_recorder_rejects_unbalanced_samples() -> None:
    recorder = StageRecorder(clock=lambda: 0.0, synchronize=lambda: None)
    with pytest.raises(RuntimeError, match="begin_sample"):
        with recorder.measure("head_ms"):
            pass

    recorder.begin_sample()
    with pytest.raises(RuntimeError, match="already active"):
        recorder.begin_sample()


class _HookHandle:
    def __init__(self, callbacks: list, callback) -> None:
        self.callbacks = callbacks
        self.callback = callback

    def remove(self) -> None:
        self.callbacks.remove(self.callback)


class _FakeModule:
    def __init__(self) -> None:
        self.pre_hooks = []
        self.post_hooks = []

    def register_forward_pre_hook(self, callback):
        self.pre_hooks.append(callback)
        return _HookHandle(self.pre_hooks, callback)

    def register_forward_hook(self, callback):
        self.post_hooks.append(callback)
        return _HookHandle(self.post_hooks, callback)

    def __call__(self) -> None:
        for callback in list(self.pre_hooks):
            callback(self, ())
        for callback in list(self.post_hooks):
            callback(self, (), None)


def test_module_stage_hooks_measure_real_module_call_boundaries() -> None:
    ticks = iter([1.0, 1.125])
    recorder = StageRecorder(clock=lambda: next(ticks), synchronize=lambda: None)
    module = _FakeModule()
    hooks = ModuleStageHooks(recorder)
    hooks.register("head_ms", module)
    recorder.begin_sample()

    module()

    assert recorder.end_sample()["head_ms"] == pytest.approx(125.0)
    hooks.close()
    assert module.pre_hooks == []
    assert module.post_hooks == []


def test_method_stage_hooks_measure_forward_test_calls_that_bypass_module_hooks() -> None:
    class Target:
        def forward_test(self, value):
            return value + 1

    ticks = iter([4.0, 4.25])
    recorder = StageRecorder(clock=lambda: next(ticks), synchronize=lambda: None)
    target = Target()
    hooks = MethodStageHooks(recorder)
    hooks.register("frame_selector_total_ms", target, "forward_test")
    recorder.begin_sample()

    assert target.forward_test(2) == 3

    assert recorder.end_sample()["frame_selector_total_ms"] == pytest.approx(250.0)
    hooks.close()
    assert target.forward_test(2) == 3
