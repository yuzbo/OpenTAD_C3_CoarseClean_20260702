import copy
import inspect
import json
from types import SimpleNamespace

import pytest

import tools.bata.run_phystime_g1a_real_gate as gate_module

from tools.bata.run_phystime_g1a_real_gate import (
    SCHEMA_VERSION,
    _aggregate_gradient_step_reports,
    _directory_inventory,
    _state_dict_sha256,
    audit_dataset_timebases,
    validate_gate_report,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
GIT_SHA = "d" * 40


def test_state_dict_hash_supports_scalar_integer_buffers():
    import torch

    module = torch.nn.Module()
    module.register_buffer("step", torch.tensor(1, dtype=torch.long))
    before = _state_dict_sha256(module)
    module.step.add_(1)
    after = _state_dict_sha256(module)

    assert len(before) == 64
    assert before != after


def test_decoded_temporal_length_uses_time_axis_for_six_dimensional_batches():
    import torch

    batch = torch.empty(2, 1, 3, 384, 2, 2)
    sample = torch.empty(1, 3, 384, 2, 2)

    assert gate_module._decoded_temporal_length(batch) == 384
    assert gate_module._decoded_temporal_length(sample) == 384
    assert int(batch.shape[2]) == 3


def test_g1a_selected_index_checksum_accepts_variable_valid_lengths():
    digest, values = gate_module._selected_index_checksum_g1a(
        {"selected_raw_frame_indices": list(range(269))}
    )

    assert len(digest) == 64
    assert values.size == 269

    with pytest.raises(RuntimeError):
        gate_module._selected_index_checksum_g1a(
            {"selected_raw_frame_indices": [0, 2, 2]}
        )


def _step_reports():
    reports = []
    for step in range(3):
        gradients = {}
        for name in (
            "adapter_gradient",
            "projection_gradient",
            "classification_gradient",
            "regression_gradient",
        ):
            nonzero = not (name == "regression_gradient" and step == 0)
            gradients[name] = {
                "parameter_count": 2,
                "finite_gradient_count": 2,
                "nonzero_gradient_count": 2 if nonzero else 0,
                "gradient_l1": 1.0 if nonzero else 0.0,
                "nonzero": nonzero,
                "all_finite": True,
            }
        reports.append({
            "step": step,
            "losses": {
                "cost": 1.0 + step,
                "cls_loss": 0.5,
                "reg_loss": 0.25,
            },
            "assignment_debug": {
                "assignment_num_positive": 3,
                "assignment_positive_per_sample": [2, 1],
                "assignment_valid_point_count": 700,
                "assignment_valid_point_per_sample": [378, 322],
                "assignment_gt_count": 2,
                "assignment_positive_fraction": 3.0 / 700.0,
                "assignment_regression_raw_count": 6,
                "assignment_regression_raw_positive_count": 2 if step else 0,
                "assignment_regression_active_location_count": 1 if step else 0,
            },
            "gradients": gradients,
            "learning_rates_before": [0.0 if step == 0 else 1.0e-6],
            "learning_rates_after": [1.0e-6 * (step + 1)],
            "clip_grad_norm": 1.0,
            "scheduler_last_epoch_after": step + 1,
            "optimizer_state_parameter_count_after": 2,
            "optimizer_state_min_step_after": step + 1,
            "optimizer_state_max_step_after": step + 1,
            "optimizer_state_parameter_names_sha256_after": SHA_A,
            "ema_updated": True,
            "amp_scale_before": 1024.0,
            "amp_scale_after": 1024.0,
        })
    return reports


def test_gradient_gate_aggregates_nonzero_evidence_across_three_steps():
    aggregated = _aggregate_gradient_step_reports(
        [report["gradients"] for report in _step_reports()]
    )

    assert aggregated["regression_gradient"]["per_step_nonzero"] == [
        False,
        True,
        True,
    ]
    assert aggregated["regression_gradient"]["nonzero"] is True
    assert aggregated["regression_gradient"]["nonzero_step_count"] == 2


def test_trainable_parameter_hash_ignores_mutable_buffers_but_detects_parameter_updates():
    import torch

    module = torch.nn.Linear(2, 1, bias=False)
    module.register_buffer("loss_normalizer", torch.tensor(1.0))
    parameter_before = gate_module._trainable_parameter_sha256(module)
    full_before = _state_dict_sha256(module)

    module.loss_normalizer.add_(1.0)

    assert _state_dict_sha256(module) != full_before
    assert gate_module._trainable_parameter_sha256(module) == parameter_before

    with torch.no_grad():
        module.weight.add_(1.0)
    assert gate_module._trainable_parameter_sha256(module) != parameter_before


def test_optimizer_parameter_hash_uses_a_fixed_parameter_set_when_requires_grad_changes():
    import torch

    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-3)
    before = gate_module._optimizer_parameter_sha256(module, optimizer)

    module.weight.requires_grad_(False)
    assert gate_module._optimizer_parameter_sha256(module, optimizer) == before

    with torch.no_grad():
        module.weight.add_(1.0)
    assert gate_module._optimizer_parameter_sha256(module, optimizer) != before


def test_build_dataloader_really_drops_an_odd_final_batch():
    import torch

    from opentad.datasets import build_dataloader

    class _OddDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, index):
            return {"inputs": torch.tensor([index], dtype=torch.float32)}

    loader = build_dataloader(
        _OddDataset(),
        batch_size=2,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    assert loader.drop_last is True
    assert [int(batch["inputs"].shape[0]) for batch in loader] == [2, 2]


def test_build_dataloader_seed_reproduces_the_shuffled_sample_order():
    import torch

    from opentad.datasets import build_dataloader

    class _IndexedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 11

        def __getitem__(self, index):
            return {"inputs": torch.tensor(index, dtype=torch.int64)}

    def sample_order(seed):
        loader = build_dataloader(
            _IndexedDataset(),
            batch_size=2,
            rank=0,
            world_size=1,
            shuffle=True,
            drop_last=True,
            seed=seed,
            num_workers=0,
        )
        loader.sampler.set_epoch(0)
        return [int(value) for batch in loader for value in batch["inputs"]]

    assert sample_order(17) == sample_order(17)
    assert sample_order(17) != sample_order(18)


def test_gate_device_copy_does_not_mutate_or_retain_the_cpu_batch_mapping():
    import torch

    source = {
        "inputs": torch.ones(1, 1),
        "masks": torch.ones(1, 1, dtype=torch.bool),
        "gt_segments": [torch.tensor([[1.0, 2.0]])],
        "gt_labels": [torch.tensor([1])],
        "metas": [{"video_name": "sample"}],
    }

    moved = gate_module._copy_batch_to_device(source, torch.device("cpu"))
    moved["metas"][0]["video_name"] = "changed"
    moved["gt_segments"].append(torch.tensor([[3.0, 4.0]]))

    assert moved is not source
    assert source["metas"][0]["video_name"] == "sample"
    assert len(source["gt_segments"]) == 1


def test_regression_gradient_family_includes_actionformer_scales():
    assert (
        gate_module._gradient_family_for_parameter("rpn_head.scale.0.scale")
        == "regression_gradient"
    )


def test_gate_run_variant_uses_formal_scheduler_ema_and_optimizer_step_order():
    source = inspect.getsource(gate_module._run_variant)

    for required_call in (
        "build_scheduler(",
        "ModelEma(",
        "_call_after_optimizer_step(",
        "scheduler.step()",
        "model_ema.update(model)",
    ):
        assert required_call in source

    assert source.index("scaler.step(optimizer)") < source.index("scheduler.step()")
    assert source.index("scheduler.step()") < source.index("model_ema.update(model)")


def _timebase_audit():
    return {
        "audit_pass": True,
        "audit_scope": "dataset_consumed_videos_only",
        "video_count": 2,
        "missing_consumed_video_count": 0,
        "frame_count_mismatch_count": 0,
        "records_sha256": SHA_A,
        "audited_video_names_sha256": SHA_C,
        "unreferenced_records_sha256": SHA_B,
        "split_counts": {"train": 1, "test": 1},
    }


def _variant():
    step_reports = _step_reports()
    gradients = _aggregate_gradient_step_reports(
        [report["gradients"] for report in step_reports]
    )
    return {
        "decoded_frame_count": 384,
        "raw_valid_count": 320,
        "backbone_feature_length": 192,
        "inference_backbone_feature_length": 192,
        "finite_loss": True,
        "finite_predictions": True,
        "optimizer_coverage": True,
        "optimizer_steps_requested": 3,
        "optimizer_steps_completed": 3,
        "parameter_state_changed": True,
        "initial_optimizer_parameter_sha256": SHA_A,
        "final_optimizer_parameter_sha256": SHA_B,
        "trainable_parameter_delta_l1": 1.0,
        "trainable_parameter_delta_max": 0.5,
        "changed_trainable_parameter_count": 2,
        "changed_trainable_parameter_names_sha256": SHA_C,
        "optimizer_state_parameter_count": 2,
        "optimizer_state_min_step": 3,
        "optimizer_state_max_step": 3,
        "optimizer_expected_parameter_count": 2,
        "optimizer_parameter_names_sha256": SHA_A,
        "optimizer_state_parameter_names_sha256": SHA_A,
        "parameter_schema": {"schema": [{"name": "adapter.weight"}]},
        "optimizer_schema": [{"group": 0, "names": ["adapter.weight"]}],
        "optimizer_base_lrs": [1.0e-4],
        "scheduler_initial_lrs": [0.0],
        "production_train_dataloader": True,
        "production_train_batch_size": 2,
        "production_train_drop_last": True,
        "production_train_shuffle": True,
        "production_train_raw_valid_counts": [384, 320, 255, 384, 192, 384],
        "production_train_min_raw_valid_count": 192,
        "production_train_max_raw_valid_count": 384,
        "production_scheduler": True,
        "scheduler_class": "LinearWarmupCosineAnnealingLR",
        "scheduler_positive_lr_observed": True,
        "model_ema_enabled": True,
        "model_ema_updates": 3,
        "amp_contract_verified": True,
        "train_window_crop_uses_gt": True,
        "train_subsample_uses_gt": False,
        "tail_window_crop_uses_gt": False,
        "tail_subsample_uses_gt": False,
        **gradients,
        "native_geometry_audit": {
            "feature_interpolation": False,
            "query_tensor_count": 378,
            "lineage_evidence_level": "exact_patch_inputs_plus_structural_receptive_field_upper_bound",
        },
        "tail_native_geometry_audit": {
            "raw_valid_counts": [127],
            "invalid_native_features_zeroed": True,
            "padding_repeat_counts": [257],
            "valid_tokens_may_depend_on_padding_repeats": [True],
            "valid_tokens_depend_on_padding_after_isolation": False,
            "candidate_mask_policy": "semantic_anchor_prefix",
            "backbone_temporal_padding_isolation": {
                "strict_isolation_verified": True,
                "attention_key_value_masked": True,
                "adapter_convolution_masked": True,
                "output_invalid_features_zeroed": True,
            },
        },
        "head_geometry_debug": {
            "physical_grid_actionformer_enabled": True,
            "physical_grid_actionformer_valid_points": 378,
            "physical_grid_actionformer_axis_start_key": "phystime_g1a_axis_start_sec",
            "physical_grid_actionformer_axis_end_key": "phystime_g1a_axis_end_sec",
        },
        "tail_head_geometry_debug": {
            "physical_grid_actionformer_enabled": True,
            "physical_grid_actionformer_valid_points": 199,
        },
        "production_single_video_eval_executed": True,
        "production_single_video_detection_count": 1,
        "production_single_video_metrics": {"average_mAP": 0.0},
        "full_post_processing_executed": True,
        "prediction_time_unit": "seconds",
        "optimizer_step_reports": step_reports,
        "initial_state_sha256": SHA_A,
        "final_state_sha256": SHA_B,
        "canonical_config_sha256": SHA_C,
    }


def _valid_report():
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "K_raw_observations": 384,
        "J_native_tubelet_tokens": 192,
        "Q0_base_candidates": 192,
        "Q_total_candidates": 378,
        "selected_index_checksum_match": True,
        "decoded_input_checksum_match": True,
        "target_checksum_match": True,
        "parameter_schema_match": True,
        "initial_state_match": True,
        "optimizer_schema_match": True,
        "tree_clean": True,
        "git_tree": GIT_SHA,
        "real_g0_pass": True,
        "optimizer_steps": 3,
        "amp_contract_verified": True,
        "timebase_audit": _timebase_audit(),
        "dataset_manifest_sha256": SHA_A,
        "checkpoint_sha256": SHA_B,
        "contract_sha256": SHA_C,
        "static_g0_sha256": SHA_A,
        "git_commit": GIT_SHA,
        "selected_index_sha256": [SHA_A, SHA_B, SHA_C],
        "selected_index_lengths": [[384, 320], [255, 384], [192, 384]],
        "decoded_input_sha256": [SHA_A, SHA_B, SHA_C],
        "target_sha256": [SHA_A, SHA_B, SHA_C],
        "tail_selected_index_sha256": SHA_A,
        "tail_selected_index_length": 127,
        "tail_decoded_input_sha256": SHA_B,
        "variants": {"selected_axis": _variant(), "physical_metric": _variant()},
    }


def test_g1a_gate_schema_is_versioned_for_production_engine_evidence():
    assert SCHEMA_VERSION == "phystime_g1a_real_gate_v3"


def test_g1a_real_gate_contract_requires_native_counts_and_both_matched_arms():
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "K_raw_observations": 384,
        "J_native_tubelet_tokens": 192,
        "Q0_base_candidates": 192,
        "Q_total_candidates": 378,
        "selected_index_checksum_match": True,
        "decoded_input_checksum_match": True,
        "target_checksum_match": True,
        "parameter_schema_match": True,
        "initial_state_match": True,
        "optimizer_schema_match": True,
        "tree_clean": True,
        "git_tree": GIT_SHA,
        "real_g0_pass": True,
        "optimizer_steps": 3,
        "amp_contract_verified": True,
        "timebase_audit": _timebase_audit(),
        "dataset_manifest_sha256": SHA_A,
        "checkpoint_sha256": SHA_B,
        "contract_sha256": SHA_C,
        "static_g0_sha256": SHA_A,
        "git_commit": GIT_SHA,
        "selected_index_sha256": [SHA_A, SHA_B, SHA_C],
        "selected_index_lengths": [[384, 320], [255, 384], [192, 384]],
        "decoded_input_sha256": [SHA_A, SHA_B, SHA_C],
        "target_sha256": [SHA_A, SHA_B, SHA_C],
        "tail_selected_index_sha256": SHA_A,
        "tail_selected_index_length": 127,
        "tail_decoded_input_sha256": SHA_B,
        "variants": {"selected_axis": _variant(), "physical_metric": _variant()},
    }

    assert validate_gate_report(report) is True


def test_g1a_real_gate_contract_fails_closed_on_provenance_or_seconds_mismatch():
    base = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "K_raw_observations": 384,
        "J_native_tubelet_tokens": 192,
        "Q0_base_candidates": 192,
        "Q_total_candidates": 378,
        "selected_index_checksum_match": True,
        "decoded_input_checksum_match": True,
        "target_checksum_match": True,
        "parameter_schema_match": True,
        "initial_state_match": True,
        "optimizer_schema_match": True,
        "tree_clean": True,
        "git_tree": GIT_SHA,
        "real_g0_pass": True,
        "optimizer_steps": 3,
        "amp_contract_verified": True,
        "timebase_audit": _timebase_audit(),
        "dataset_manifest_sha256": SHA_A,
        "checkpoint_sha256": SHA_B,
        "contract_sha256": SHA_C,
        "static_g0_sha256": SHA_A,
        "git_commit": GIT_SHA,
        "selected_index_sha256": [SHA_A, SHA_B, SHA_C],
        "selected_index_lengths": [[384, 320], [255, 384], [192, 384]],
        "decoded_input_sha256": [SHA_A, SHA_B, SHA_C],
        "target_sha256": [SHA_A, SHA_B, SHA_C],
        "tail_selected_index_sha256": SHA_A,
        "tail_selected_index_length": 127,
        "tail_decoded_input_sha256": SHA_B,
        "variants": {"selected_axis": _variant(), "physical_metric": _variant()},
    }

    for mutator in (
        lambda report: report.update(initial_state_match=False),
        lambda report: report.update(dataset_manifest_sha256=""),
        lambda report: report.update(target_checksum_match=False),
        lambda report: report.update(selected_index_lengths=[[384, 320]]),
        lambda report: report.update(selected_index_lengths=[[384, 0], [255, 384], [192, 384]]),
        lambda report: report.update(tail_selected_index_length=0),
        lambda report: report.update(real_g0_pass=False),
        lambda report: report["timebase_audit"].update(audit_pass=False),
        lambda report: report["timebase_audit"].update(
            missing_consumed_video_count=1
        ),
        lambda report: report["variants"]["selected_axis"].update(parameter_state_changed=False),
        lambda report: report["variants"]["selected_axis"].update(
            final_optimizer_parameter_sha256=report["variants"]["selected_axis"][
                "initial_optimizer_parameter_sha256"
            ]
        ),
        lambda report: report["variants"]["selected_axis"].update(
            trainable_parameter_delta_l1=0.0
        ),
        lambda report: report["variants"]["selected_axis"].update(
            optimizer_state_max_step=0
        ),
        lambda report: report["variants"]["selected_axis"].update(
            optimizer_state_parameter_count=1
        ),
        lambda report: report["variants"]["selected_axis"].update(
            optimizer_state_min_step=2
        ),
        lambda report: report["variants"]["selected_axis"]["optimizer_step_reports"][1].update(
            optimizer_state_parameter_names_sha256_after=SHA_B
        ),
        lambda report: report["variants"]["selected_axis"].update(
            production_train_batch_size=1
        ),
        lambda report: report["variants"]["selected_axis"].update(raw_valid_count=0),
        lambda report: report["variants"]["selected_axis"].update(
            production_train_raw_valid_counts=[384, 0, 255, 384, 192, 384]
        ),
        lambda report: report["variants"]["selected_axis"].update(
            production_train_min_raw_valid_count=1
        ),
        lambda report: report["variants"]["selected_axis"].update(
            production_scheduler=False
        ),
        lambda report: report["variants"]["selected_axis"].update(
            scheduler_positive_lr_observed=False
        ),
        lambda report: report["variants"]["selected_axis"].update(optimizer_step_reports=[]),
        lambda report: report["variants"]["selected_axis"]["optimizer_step_reports"][1].update(
            amp_scale_after=float("nan")
        ),
        lambda report: report["variants"]["selected_axis"].update(
            final_state_sha256=report["variants"]["selected_axis"]["initial_state_sha256"]
        ),
        lambda report: report.update(checkpoint_sha256="not-a-digest"),
        lambda report: report["variants"]["selected_axis"].update(prediction_time_unit="dense_index"),
        lambda report: report["variants"]["physical_metric"]["tail_native_geometry_audit"].update(
            raw_valid_counts=[384]
        ),
    ):
        report = copy.deepcopy(base)
        mutator(report)
        with pytest.raises(RuntimeError):
            validate_gate_report(report)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda variant: [
            step["gradients"]["regression_gradient"].update(
                nonzero=False, nonzero_gradient_count=0, gradient_l1=0.0
            )
            for step in variant["optimizer_step_reports"]
        ],
        lambda variant: variant["optimizer_step_reports"][1]["gradients"][
            "adapter_gradient"
        ].update(nonzero=False, nonzero_gradient_count=0, gradient_l1=0.0),
        lambda variant: variant["optimizer_step_reports"][0]["gradients"][
            "regression_gradient"
        ].update(nonzero=False, nonzero_gradient_count=1, gradient_l1=1.0),
        lambda variant: variant["regression_gradient"].update(
            gradient_l1_across_steps=999.0
        ),
    ],
)
def test_gate_validator_recomputes_gradient_contract_from_step_evidence(mutator):
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "K_raw_observations": 384,
        "J_native_tubelet_tokens": 192,
        "Q0_base_candidates": 192,
        "Q_total_candidates": 378,
        "selected_index_checksum_match": True,
        "decoded_input_checksum_match": True,
        "target_checksum_match": True,
        "parameter_schema_match": True,
        "initial_state_match": True,
        "optimizer_schema_match": True,
        "tree_clean": True,
        "git_tree": GIT_SHA,
        "real_g0_pass": True,
        "optimizer_steps": 3,
        "amp_contract_verified": True,
        "timebase_audit": _timebase_audit(),
        "dataset_manifest_sha256": SHA_A,
        "checkpoint_sha256": SHA_B,
        "contract_sha256": SHA_C,
        "static_g0_sha256": SHA_A,
        "git_commit": GIT_SHA,
        "selected_index_sha256": [SHA_A, SHA_B, SHA_C],
        "selected_index_lengths": [[384, 320], [255, 384], [192, 384]],
        "decoded_input_sha256": [SHA_A, SHA_B, SHA_C],
        "target_sha256": [SHA_A, SHA_B, SHA_C],
        "tail_selected_index_sha256": SHA_A,
        "tail_selected_index_length": 127,
        "tail_decoded_input_sha256": SHA_B,
        "variants": {"selected_axis": _variant(), "physical_metric": _variant()},
    }
    mutator(report["variants"]["selected_axis"])

    with pytest.raises(RuntimeError):
        validate_gate_report(report)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report["variants"]["physical_metric"]["parameter_schema"].update(
            schema=["different-parameter-schema"]
        ),
        lambda report: report["variants"]["physical_metric"].update(
            initial_state_sha256=SHA_C
        ),
        lambda report: report["variants"]["physical_metric"].update(
            optimizer_schema=[{"different": True}]
        ),
    ],
)
def test_gate_validator_recomputes_cross_arm_schema_matches(mutator):
    report = _valid_report()
    mutator(report)
    report["parameter_schema_match"] = True
    report["initial_state_match"] = True
    report["optimizer_schema_match"] = True

    with pytest.raises(RuntimeError):
        validate_gate_report(report)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda debug: debug.update(assignment_num_positive=4),
        lambda debug: debug.update(assignment_positive_per_sample=[3]),
        lambda debug: debug.update(assignment_positive_fraction=0.5),
        lambda debug: debug.update(assignment_regression_raw_count=5),
        lambda debug: debug.update(assignment_regression_raw_positive_count=7),
        lambda debug: debug.update(assignment_valid_point_count=3, assignment_positive_fraction=1.0),
        lambda debug: debug.update(assignment_valid_point_per_sample=[378]),
        lambda debug: debug.update(assignment_valid_point_per_sample=[378, 0]),
        lambda debug: debug.update(assignment_valid_point_per_sample=[1, 1]),
        lambda debug: debug.update(assignment_gt_count=0),
        lambda debug: debug.update(
            assignment_regression_raw_positive_count=6,
            assignment_regression_active_location_count=1,
        ),
    ],
)
def test_gate_validator_rejects_internally_inconsistent_assignment_debug(mutator):
    variant = _variant()
    mutator(variant["optimizer_step_reports"][0]["assignment_debug"])
    report = _valid_report()
    report["variants"]["selected_axis"] = variant

    with pytest.raises(RuntimeError):
        validate_gate_report(report)


def test_actionformer_relu_dead_zone_can_have_positive_diou_loss_and_zero_raw_gradient():
    import torch

    from opentad.models.losses.iou_loss import DIOULoss

    raw_regression = torch.nn.Parameter(torch.tensor([[-1.0, -2.0]]))
    distance = torch.relu(raw_regression)
    center = torch.tensor([5.0])
    predicted = torch.stack(
        (center - distance[:, 0], center + distance[:, 1]), dim=-1
    )
    target = torch.tensor([[4.0, 6.0]])
    loss = DIOULoss()(predicted, target, reduction="sum")

    assert float(loss.item()) > 0.0
    loss.backward()
    assert float(raw_regression.grad.abs().sum().item()) == 0.0


def test_dataset_inventory_hashes_file_content_not_only_size(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"same-size-a")
    before = _directory_inventory(tmp_path)

    video.write_bytes(b"same-size-b")
    after = _directory_inventory(tmp_path)

    assert before["inventory_sha256"] != after["inventory_sha256"]
    assert before["files"][0]["sha256"] != after["files"][0]["sha256"]
    assert before["hash_scope"] == "full_file_content_sha256_merkle_v1"


def test_full_dataset_timebase_audit_uses_the_same_fail_closed_contract(tmp_path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "video_validation_1.mp4").write_bytes(b"train")
    (test_dir / "video_test_1.mp4").write_bytes(b"test")
    (test_dir / "video_test_unused.mp4").write_bytes(b"unused")
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "video_validation_1": {"frame": 400, "duration": 20.0},
                    "video_test_1": {"frame": 200, "duration": 10.0},
                }
            }
        ),
        encoding="utf-8",
    )
    raw_step = {
        "type": "BuildPhysTimeRawFrameGeometry",
        "fps_relative_tolerance": 0.0125,
        "duration_relative_tolerance": 0.0125,
        "frame_count_relative_tolerance": 0.0001,
    }
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            train=SimpleNamespace(data_path=str(train_dir), pipeline=[raw_step]),
            test=SimpleNamespace(data_path=str(test_dir), pipeline=[raw_step]),
        )
    )

    consumed_video_names = {
        "train": {"video_validation_1"},
        "test": {"video_test_1"},
    }
    report = audit_dataset_timebases(
        cfg,
        annotation,
        decoder_probe=lambda path: (20.0, 400 if "validation" in path.name else 200),
        dataset_video_names=consumed_video_names,
    )

    assert report["audit_pass"] is True
    assert report["audit_scope"] == "dataset_consumed_videos_only"
    assert report["video_count"] == 2
    assert report["directory_file_counts"] == {"train": 1, "test": 2}
    assert report["unreferenced_file_counts"] == {"train": 0, "test": 1}
    assert report["unreferenced_video_names"] == {
        "train": [],
        "test": ["video_test_unused"],
    }
    assert report["missing_consumed_video_count"] == 0
    assert report["frame_count_mismatch_count"] == 0
    assert report["records_sha256"]

    with pytest.raises(ValueError, match="FPS"):
        audit_dataset_timebases(
            cfg,
            annotation,
            decoder_probe=lambda path: (25.0, 400 if "validation" in path.name else 200),
            dataset_video_names=consumed_video_names,
        )
