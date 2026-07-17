from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import pytest

from tools.bata.profile_duca_full_stack_cost import (
    _cuda_nvml_uuid,
    _detector_stack_fingerprint,
    build_arg_parser,
    component_elapsed_ms,
    discover_profile_modules,
    parse_nvidia_smi_power_lines,
    strip_ddp_prefix,
)
from tools.bata.compare_duca_full_stack_cost import main as compare_cost_main
from tools.bata.duca_full_stack_cost import OFFLINE_FULL_WINDOW_PROTOCOL, build_profile_summary


def test_strip_ddp_prefix_supports_training_and_ema_checkpoints() -> None:
    state = {
        "module.frame_selector.weight": 1,
        "module.backbone.weight": 2,
    }

    assert strip_ddp_prefix(state) == {
        "frame_selector.weight": 1,
        "backbone.weight": 2,
    }
    assert strip_ddp_prefix({"backbone.weight": 2}) == {"backbone.weight": 2}


def test_power_parser_ignores_headers_and_invalid_samples() -> None:
    values = parse_nvidia_smi_power_lines(
        [
            "power.draw [W]",
            "123.50 W",
            "N/A",
            "125.0",
            "",
        ]
    )

    assert values == pytest.approx([123.5, 125.0])


def test_component_cost_uses_cuda_timeline_without_double_counting_cpu_enqueue() -> None:
    assert component_elapsed_ms(cuda_elapsed_ms=20.0, cpu_enqueue_ms=35.0) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="CUDA"):
        component_elapsed_ms(cuda_elapsed_ms=-1.0, cpu_enqueue_ms=2.0)


def test_power_device_uuid_uses_torch_nvml_mapping_not_cuda_logical_index() -> None:
    fake_cuda = SimpleNamespace(
        _get_device_index=lambda _device, optional=True: 0,
        _parse_visible_devices=lambda: [1],
        _raw_device_uuid_nvml=lambda: ["GPU-physical-zero", "GPU-physical-one"],
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda)

    assert _cuda_nvml_uuid(fake_torch, "cuda:0") == "GPU-physical-one"


def test_detector_stack_fingerprint_tracks_classes_and_parameter_schema() -> None:
    class FakeParameter:
        shape = (3, 4)
        dtype = "float32"
        requires_grad = True

    class FakeModule:
        def named_modules(self):
            return [("", self)]

        def named_parameters(self):
            return [("weight", FakeParameter())]

    model = SimpleNamespace(
        backbone=FakeModule(),
        projection=None,
        neck=None,
        rpn_head=FakeModule(),
        token_compressor=None,
    )

    first = _detector_stack_fingerprint(model)
    second = _detector_stack_fingerprint(model)
    assert first == second
    model.rpn_head = None
    assert _detector_stack_fingerprint(model) != first


def test_profile_module_discovery_is_hierarchical_and_zero_fills_absent_modules() -> None:
    probe = object()
    selector = SimpleNamespace(raw_actionness_source=probe)
    heavy = object()
    backbone = SimpleNamespace(model=SimpleNamespace(backbone=heavy))
    projection = object()
    head = object()
    model = SimpleNamespace(
        frame_selector=selector,
        backbone=backbone,
        projection=projection,
        neck=None,
        rpn_head=head,
    )

    modules, zero_stages = discover_profile_modules(model)

    assert modules == {
        "frame_selector_total_ms": selector,
        "coarse_probe_ms": probe,
        "backbone_wrapper_total_ms": backbone,
        "heavy_backbone_ms": heavy,
        "projection_ms": projection,
        "head_ms": head,
    }
    assert zero_stages == {"neck_ms"}


def test_cli_requires_a_checkpoint_unless_random_init_is_explicit() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(["config.py", "--output-prefix", "out/profile"])
    with pytest.raises(ValueError, match="checkpoint"):
        args.validate()

    args = parser.parse_args(
        ["config.py", "--output-prefix", "out/profile", "--allow-random-init", "--samples", "3"]
    )
    args.validate()
    assert args.samples == 3
    assert args.loader_workers == 0
    assert args.power_gpu_id is None


def test_dense_paper_profile_requires_hash_bound_checkpoint_evidence() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "dense.py",
            "--checkpoint",
            "dense.pth",
            "--use-ema",
            "--method-name",
            "dense-adatad",
            "--output-prefix",
            "out/dense",
        ]
    )
    with pytest.raises(ValueError, match="checkpoint evidence"):
        args.validate()

    args = parser.parse_args(
        [
            "dense.py",
            "--checkpoint",
            "dense.pth",
            "--use-ema",
            "--method-name",
            "dense-adatad",
            "--checkpoint-evidence",
            "binding.json",
            "--checkpoint-evidence-sha256",
            "a" * 64,
            "--profile-session-id",
            "slurm-1",
            "--profile-pair-id",
            "repeat-1",
            "--profile-repeat-index",
            "1",
            "--profile-order-position",
            "1",
            "--output-prefix",
            "out/dense",
        ]
    )
    args.validate()
    args.profile_session_id = ""
    with pytest.raises(ValueError, match="profile-session-id"):
        args.validate()


def test_cli_separates_profiler_and_trained_commits() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "dense.py",
            "--checkpoint",
            "dense.pth",
            "--use-ema",
            "--method-name",
            "dense-adatad",
            "--checkpoint-evidence",
            "binding.json",
            "--checkpoint-evidence-sha256",
            "a" * 64,
            "--trained-commit",
            "b" * 40,
            "--profile-session-id",
            "slurm-1",
            "--profile-pair-id",
            "repeat-1",
            "--profile-repeat-index",
            "1",
            "--profile-order-position",
            "1",
            "--output-prefix",
            "out/dense",
        ]
    )
    args.validate()
    assert args.trained_commit == "b" * 40

    args.trained_commit = "short"
    with pytest.raises(ValueError, match="trained-commit"):
        args.validate()


def test_paper_profile_clean_tree_check_includes_untracked_files() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tools/bata/profile_duca_full_stack_cost.py"
    ).read_text(encoding="utf-8")

    assert '"status", "--porcelain", "--untracked-files=normal"' in source


def test_gpu1_launcher_is_fail_closed_and_uses_the_full_stack_profiler() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "run_duca_full_stack_cost_profile_gpu1.sh"
    source = launcher.read_text(encoding="utf-8")

    assert "PRECHECK_ONLY" in source
    assert "PROFILE_CHECKPOINT" in source
    assert "ALLOW_RANDOM_INIT" in source
    assert "profile_duca_full_stack_cost.py" in source
    assert "DUCA_PROFILE_RUNTIME=0" in source
    assert "--loader-workers 0" in source
    assert "--sample-power" in source
    assert "PROFILE_POWER_GPU_ID" in source
    assert "--power-gpu-id" in source
    assert "SLURM_JOB_ID" in source


def test_compare_cli_builds_multi_candidate_matrix(tmp_path) -> None:
    metadata = {
        "protocol": OFFLINE_FULL_WINDOW_PROTOCOL,
        "hardware_fingerprint": "same-gpu",
        "host_fingerprint": "same-host",
        "software_fingerprint": "same-software",
        "config_commit": "abc123",
        "tracked_tree_clean": True,
        "dataset_fingerprint": "same-dataset",
        "inference_fingerprint": "same-inference",
        "detector_stack_fingerprint": "same-detector",
        "batch_size": 1,
        "loader_workers": 0,
        "warmup_samples": 5,
        "amp": True,
        "uses_ema": True,
        "random_init": False,
        "power_sampling_enabled": False,
        "power_interval_ms": 20,
        "power_gpu_id": None,
    }
    sample = {
        "input_pipeline_serial_ms": 10.0,
        "h2d_ms": 2.0,
        "model_forward_ms": 50.0,
        "postprocess_ms": 3.0,
        "frame_selector_total_ms": 5.0,
        "coarse_probe_ms": 2.0,
        "backbone_wrapper_total_ms": 35.0,
        "heavy_backbone_ms": 30.0,
        "projection_ms": 4.0,
        "neck_ms": 2.0,
        "head_ms": 3.0,
        "selected_count": 384,
    }
    baseline_sample = {
        key: value * 2 if key.endswith("_ms") else value
        for key, value in sample.items()
    }
    baseline = build_profile_summary([baseline_sample], metadata={**metadata, "method": "dense768"})
    candidate = build_profile_summary([sample], metadata={**metadata, "method": "duca384"})
    baseline_path = tmp_path / "dense.json"
    candidate_path = tmp_path / "duca.json"
    baseline_path.write_text(__import__("json").dumps(baseline), encoding="utf-8")
    candidate_path.write_text(__import__("json").dumps(candidate), encoding="utf-8")

    assert compare_cost_main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output-prefix",
            str(tmp_path / "matrix"),
        ]
    ) == 0
    assert (tmp_path / "matrix.json").exists()
    assert (tmp_path / "matrix.tsv").exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "tools/bata/profile_duca_full_stack_cost.py",
        "tools/bata/compare_duca_full_stack_cost.py",
    ],
)
def test_cost_cli_file_entrypoints_resolve_the_repo_package(relative_path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / relative_path), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
