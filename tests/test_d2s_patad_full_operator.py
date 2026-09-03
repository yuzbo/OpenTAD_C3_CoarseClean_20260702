import json
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.continuous_roi_s2_v3_full200_compute_profile import (
    ARMS,
    CANDIDATE_ARM,
    FullOperatorLedger,
    LOW_COST_CONTROL_ARM,
    REFERENCE_ARM,
    compare_c_exec_receipts,
    validate_c_exec_comparison,
)
from tools.bata.trace_d2s_patad_full_operator import (
    _sort_comparison_upper_bound,
)
from tools.bata.continuous_roi_s2_v3_full200_compute import sha256_file
from tools.bata.continuous_roi_s2_v3_full200_compute_train import (
    bind_pretrained_checkpoint,
)
from tools.bata.zoomtoken_full200_matrix_spec import (
    binding_from_config,
    get_matrix_spec,
)


ROOT = Path(__file__).resolve().parents[1]


IDENTITY = {
    "candidate_commit": "a" * 40,
    "protocol_sha256": "b" * 64,
    "evaluation_manifest_sha256": "c" * 64,
    "checkpoint_policy": "epoch_59_state_dict_ema_update_6000_shape_invariant",
    "dtype": "float16",
    "batch_size": 1,
    "ordered_window_count": 792,
}
BOUNDARY = {
    "start": "first_arm_dependent_decoded_rgb_transform",
    "end": "pre_nms_raw_detections",
    "nms_called": False,
    "evaluator_called": False,
}


def test_active_matrix_full_operator_comparison_is_complete_and_exact():
    counts = {
        REFERENCE_ARM: 1000,
        LOW_COST_CONTROL_ARM: 700,
        CANDIDATE_ARM: 900,
    }
    receipts = {}
    for arm in ARMS:
        ledger = FullOperatorLedger(arm=arm)
        ledger.add_automatic(
            event_id=f"{arm}/mm",
            operator="aten.mm",
            integer_operations=counts[arm],
        )
        receipts[arm] = ledger.receipt(
            execution_identity=IDENTITY,
            boundary_trace=BOUNDARY,
        )
    comparison = compare_c_exec_receipts(receipts)
    assert comparison["primary_exact_10u_le_9d"]
    assert comparison["g96_not_more_than_candidate"]
    assert validate_c_exec_comparison(comparison) == comparison


def test_full_operator_comparison_rejects_tampering_and_missing_arm():
    receipts = {}
    for arm in ARMS:
        ledger = FullOperatorLedger(arm=arm)
        ledger.add_automatic(
            event_id=f"{arm}/mm",
            operator="aten.mm",
            integer_operations=1000,
        )
        receipts[arm] = ledger.receipt(
            execution_identity=IDENTITY,
            boundary_trace=BOUNDARY,
        )
    with pytest.raises(ValueError, match="complete selected 3-arm matrix"):
        compare_c_exec_receipts({arm: receipts[arm] for arm in ARMS[:-1]})

    comparison = compare_c_exec_receipts(receipts)
    tampered = json.loads(json.dumps(comparison))
    tampered["counts"][REFERENCE_ARM] += 1
    with pytest.raises(ValueError, match="self-hash mismatch"):
        validate_c_exec_comparison(tampered)


def test_runtime_trace_sort_bound_is_integer_and_shape_aware():
    assert _sort_comparison_upper_bound([2, 16], 3) == 384


def test_active_matrix_every_cell_has_an_explicit_training_binding():
    spec = get_matrix_spec()
    assert spec.key in {"d2s", "patad"}
    if spec.key == "d2s":
        from tools.bata.d2s_tad_full200_compute import config_path
    else:
        from tools.bata.patad_full200_compute import config_path

    for arm in spec.arms:
        for seed in (4407, 4408, 4409):
            cfg = Config.fromfile(config_path(ROOT, arm, seed))
            binding = binding_from_config(cfg, spec)
            assert binding.protocol == spec.protocol_id
            assert binding.arm == arm
            assert int(binding.seed) == seed


def test_formal_training_binds_the_sealed_absolute_pretrained_path(tmp_path):
    checkpoint = tmp_path / "pretrained.pth"
    checkpoint.write_bytes(b"sealed-pretrained")
    cfg = Config(
        dict(model=dict(backbone=dict(custom=dict(pretrain="pretrained/relative.pth"))))
    )
    identity = {"pretrained_sha256": sha256_file(checkpoint)}

    resolved = bind_pretrained_checkpoint(cfg, checkpoint, identity)
    assert cfg.model.backbone.custom.pretrain == resolved.as_posix()
    with pytest.raises(ValueError, match="sealed identity"):
        bind_pretrained_checkpoint(
            cfg,
            checkpoint,
            {"pretrained_sha256": "0" * 64},
        )


def test_all_formal_launchers_pass_the_absolute_pretrained_argument():
    for name in (
        "run_zoomtoken_continuous_roi_s2_v3_full200_compute_n16r4.sh",
        "run_zoomtoken_d2s_tad_full200_compute_n16r4.sh",
        "run_zoomtoken_patad_full200_compute_n16r4.sh",
    ):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert '--pretrained "${PRETRAINED}"' in text
