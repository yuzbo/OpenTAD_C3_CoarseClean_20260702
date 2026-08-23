import copy

from tools.bata.duca_full_stack_cost import build_profile_summary
from tools.bata.finalize_duca_h65_singleclock_cost import finalize_cost


def _report(method, repeat, order, scale):
    sample = {
        "input_pipeline_serial_ms": 1.0 * scale,
        "h2d_ms": 1.0 * scale,
        "model_forward_ms": 10.0 * scale,
        "postprocess_ms": 1.0 * scale,
        "frame_selector_total_ms": 1.0 * scale,
        "coarse_probe_ms": 0.5 * scale,
        "backbone_wrapper_total_ms": 8.0 * scale,
        "heavy_backbone_ms": 7.0 * scale,
        "projection_ms": 0.2 * scale,
        "neck_ms": 0.2 * scale,
        "head_ms": 0.2 * scale,
        "selected_count": 384,
        "peak_gpu_memory_mb": 1000.0 * scale,
    }
    metadata = {
        "method": method,
        "protocol": "offline_full_window_runtime_selection",
        "hardware_fingerprint": "gpu",
        "host_fingerprint": "host",
        "software_fingerprint": "software",
        "config_commit": "a" * 40,
        "trained_commit": "a" * 40,
        "evidence_git_commit": "b" * 40,
        "inference_code_tree_binding": None,
        "profile_config_git_binding": None,
        "profile_session_id": "session",
        "profile_pair_id": "pair",
        "profile_repeat_index": repeat,
        "profile_order_position": order,
        "profile_config_sha256": method,
        "profile_resolved_config_sha256": method,
        "config_fingerprint": method,
        "gate_zero_normalized_config_fingerprint": "normalized",
        "single_clock_gate_zero": method == "zero",
        "dataset_fingerprint": "dataset",
        "source_dataset_fingerprint": "source",
        "inference_fingerprint": "inference",
        "detector_stack_fingerprint": "stack",
        "tracked_tree_clean": True,
        "config_path": method,
        "device": "cuda:0",
        "batch_size": 1,
        "loader_workers": 0,
        "warmup_samples": 50,
        "complete_official_workload": True,
        "full_workload_batch_count": 2,
        "amp": True,
        "uses_ema": True,
        "random_init": False,
        "power_sampling_enabled": False,
        "power_interval_ms": 20,
        "power_gpu_id": None,
        "checkpoint_path": "epoch_59.pth",
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "checkpoint_sha256": "c" * 64,
    }
    return build_profile_summary([copy.deepcopy(sample), copy.deepcopy(sample)], metadata=metadata)


def test_cost_finalizer_uses_three_complete_workloads():
    on = [_report("on", repeat, 1 if repeat != 2 else 2, 1.005) for repeat in range(1, 4)]
    zero = [_report("zero", repeat, 2 if repeat != 2 else 1, 1.0) for repeat in range(1, 4)]
    result = finalize_cost(on, zero)
    assert result["complete_workload_repeats"] == 3
    assert result["median_latency_ratio_on_over_gate_zero"] < 1.01
    assert result["p90_latency_ratio_on_over_gate_zero"] < 1.02
