import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "bata"
    / "validate_actionformer_native_grid_sparse_head_cuda.py"
)
SPEC = importlib.util.spec_from_file_location("native_grid_cuda_gate", MODULE_PATH)
cuda_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cuda_gate)


def test_gate_pins_official_checkpoint_and_primary_config():
    assert cuda_gate.EXPECTED_BASE_COMMIT == (
        "61ea7eb9308a568b0cf45e3804830836e30061de"
    )
    assert cuda_gate.EXPECTED_BASE_TREE == (
        "7b06c5261ba244788c942a0d73e304581bc35154"
    )
    assert cuda_gate.EXPECTED_CHECKPOINT_EPOCH == 34
    assert cuda_gate.EXPECTED_QUERY_BUDGET == 384
    assert cuda_gate.MINIMUM_ISOLATED_HEAD_SPEEDUP == 1.05
    assert cuda_gate.EXPECTED_CONFIG_RELATIVE == (
        "configs/thumos_i3d_sparsehead_k384_uniform.yaml"
    )
    assert len(cuda_gate.OFFICIAL_CONFIG_SHA256) == 64
    assert len(cuda_gate.OFFICIAL_CONFIG_LOADER_SHA256) == 64
    assert len(cuda_gate.OFFICIAL_EFFECTIVE_CONFIG_SHA256) == 64


def test_time_size_parser_is_fail_closed():
    assert cuda_gate.parse_time_sizes("2304,1152,576") == [2304, 1152, 576]
    with pytest.raises(argparse.ArgumentTypeError):
        cuda_gate.parse_time_sizes("2304,0")


def test_checkpoint_head_prefix_extraction_is_strict():
    state = {
        "module.cls_head.weight": "cls",
        "module.reg_head.weight": "reg",
        "module.backbone.weight": "backbone",
    }
    extracted, prefix = cuda_gate.extract_submodule_state(state, "cls_head")
    assert prefix == "module.cls_head."
    assert extracted == {"weight": "cls"}
    with pytest.raises(ValueError, match="checkpoint has no state"):
        cuda_gate.extract_submodule_state(state, "missing")


def test_gate_never_upgrades_isolated_head_timing_to_end_to_end_claim():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'execution["wall_clock_claim_allowed"] = False' in source
    assert "isolated_head_path_wall_clock_claim_allowed" in source
    assert "candidate changes protected official effective-config fields" in source
