import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.bata.continuous_roi_s2_v3_full200_compute_infer import build_checkpoint_seal
from tools.bata.zoomtoken_stopped_matrix_eval import (
    validate_selected_cells, validate_training_state,
)


def full_training():
    receipt = {
        "complete": True, "epochs": 60, "successful_updates": 6000,
        "training_identity_count": 200, "checkpoint_state": "epoch_59_state_dict_ema_update_6000",
        "sample_order_trace_sha256": "a" * 64,
        "update_audit": {"consumed_batches": 6000, "ema_updates": 6000, "scheduler_advances": 6000},
    }
    checkpoint = {
        "epoch": 59, "successful_updates": 6000, "identity_hashes": {"identity": "frozen"},
        "sample_order_trace_sha256": "a" * 64, "state_dict_ema": {"weight": [1.0]},
        "optimizer": {"state": {0: {"step": 6000.0}, 1: {"step": 6000.0}}},
    }
    return checkpoint, receipt, {"identity": "frozen"}


def test_full_data_final_ema_and_actual_steps_are_accepted():
    assert validate_training_state(*full_training())["optimizer_step_min"] == 6000


@pytest.mark.parametrize("field,value", [("epochs", 59), ("training_identity_count", 199),
                                        ("successful_updates", 5999), ("complete", False)])
def test_partial_training_is_rejected(field, value):
    checkpoint, receipt, identity = full_training()
    receipt[field] = value
    with pytest.raises(ValueError, match="partial training"):
        validate_training_state(checkpoint, receipt, identity)


def test_a_false_successful_update_counter_cannot_admit_skipped_adamw_steps():
    checkpoint, receipt, identity = full_training()
    checkpoint["optimizer"]["state"][0]["step"] = 5997.0
    with pytest.raises(ValueError, match="actual AdamW"):
        validate_training_state(checkpoint, receipt, identity)


@pytest.mark.parametrize("field", ["state_dict_ema", "identity_hashes", "sample_order_trace_sha256"])
def test_checkpoint_must_corroborate_receipt(field):
    checkpoint, receipt, identity = full_training()
    checkpoint.pop(field)
    with pytest.raises(ValueError, match="corroborate"):
        validate_training_state(checkpoint, receipt, identity)


def test_diagnostic_seed_selection_is_fixed_and_matched():
    rows = [{"arm": arm, "seed": seed} for arm in ("D160", "G96", "D2S-U128-B128")
            for seed in (4407, 4408)]
    validate_selected_cells(rows, "d2s")
    for invalid in (rows[:-1], rows + [rows[-1]], copy.deepcopy(rows)):
        if len(invalid) == 6:
            invalid[-1]["seed"] = 4409
        with pytest.raises(ValueError, match="frozen matched cell"):
            validate_selected_cells(invalid, "d2s")


def test_diagnostic_cannot_bypass_original_nine_cell_seal(tmp_path):
    matrix = tmp_path / "partial.json"
    matrix.write_text(json.dumps({"cells": [{}] * 6}))
    with pytest.raises(ValueError, match="exactly 9"):
        build_checkpoint_seal(matrix_path=matrix, population_manifest_sha256="f" * 64,
                              expected_commit="a" * 40, output_path=tmp_path / "seal.json")


def test_inference_failure_cannot_open_diagnostic_gt(tmp_path, monkeypatch):
    from tools.bata import zoomtoken_stopped_matrix_eval as diagnostic
    from tools.bata import continuous_roi_s2_v3_full200_compute_eval as evaluator

    rows = [{"arm": arm, "seed": 4407} for arm in ("D160", "G96", "PATAD-U128-B128")]
    args = SimpleNamespace(output_root=tmp_path, training_root=tmp_path / "training",
                           expected_commit="evaluation-commit")
    plan = {"execution_commit": args.expected_commit, "training_commit": diagnostic.TRAINING_COMMIT,
            "training_root": args.training_root.as_posix(), "population_manifest_sha256": "population",
            "matrix_kind": "patad", "diagnostic_only": True, "rows": rows}
    (tmp_path / "control").mkdir()
    (tmp_path / "control/diagnostic_plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(diagnostic, "_cell_from_seal", lambda plan, **key: key)
    calls = []

    def interrupted_inference(args, manifest, row):
        calls.append(row["arm"])
        if len(calls) == 2:
            raise RuntimeError("inference failed")

    monkeypatch.setattr(diagnostic, "run_cell", interrupted_inference)
    monkeypatch.setattr(evaluator, "_read_complete_ground_truth",
                        lambda **kw: pytest.fail("GT must stay closed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        diagnostic.evaluate(args, {"manifest_sha256": "population"}, diagnostic.get_matrix_spec("patad"))
    assert not (tmp_path / "control/diagnostic_gt_open.json").exists()


def test_launcher_is_evaluation_only_and_does_not_override_slurm_devices():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/run_zoomtoken_stopped_matrix_eval_n16r4.sh").read_text()
    assert launcher.startswith("#!/bin/bash\n")
    assert "PRECHECK_ONLY" in launcher
    assert "torchrun" not in launcher
    assert "CUDA_VISIBLE_DEVICES" not in launcher
