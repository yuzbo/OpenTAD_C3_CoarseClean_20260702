from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.bata.georoute_dynamic_floor_m2_contract import (
    DYNAMIC_FLOOR_M2_ARM_ORDER,
    DYNAMIC_FLOOR_M2_ARMS,
    DYNAMIC_FLOOR_M2_CHECKPOINT_SIDECAR_SCHEMA,
    DYNAMIC_FLOOR_M2_COST_SCHEMA,
    DYNAMIC_FLOOR_M2_COST_ORDER,
    DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_PROBE_SCHEMA,
    DYNAMIC_FLOOR_M2_ROLE_NEUTRALITY_PAIR_SCHEMA,
    DYNAMIC_FLOOR_M2_ROLE_STRICT_TRIPLET_SCHEMA,
    DYNAMIC_FLOOR_M2_SEED,
    DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
    bind_dynamic_floor_m2_config,
    build_dynamic_floor_m2_cost_config,
    build_dynamic_floor_m2_checkpoint_metadata,
    finalize_dynamic_floor_m2,
    resolve_dynamic_floor_m2_accuracy_execution_commit,
    summarize_dynamic_floor_m2_telemetry,
    validate_dynamic_floor_m2_checkpoint_sidecar,
    validate_dynamic_floor_m2_config,
    validate_dynamic_floor_m2_cost_profile,
    validate_frozen_dynamic_floor_m2_contract,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file
from tools.bata.profile_georoute_dynamic_floor_m2 import (
    _build_cost_cuda_events,
    _invalid_cost_cuda_stages,
    _read_cost_cuda_timings,
    _validate_cost_audit,
)
from tools.bata.finalize_georoute_dynamic_floor_m2 import _validate_deployment
from tools.bata.run_georoute_residual_centering_probe import (
    _build_probe_test_arguments,
    _route_payload_sha256,
    classify_residual_centering_role_gate,
    summarize_residual_centering_branch_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def _route_telemetry() -> dict:
    geometry = [[0.5, 0.5, 0.25, 0.25] for _ in range(384)]
    k_t = [64 for _ in range(384)]
    roles = [[16, 32, 16] for _ in range(384)]
    return {
        "schema_version": "georoute_dynamic_diagnostic_window_telemetry_v1",
        "measurement_scope": "accuracy_replay_only_excluded_from_timed_cost",
        "batch_size": 1,
        "tubelet_count": 384,
        "item_count": 220,
        "source_grid_hw": [11, 20],
        "window_token_budget": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
        "selected_physical_index_sha256": "a" * 64,
        "k_t": {"values": k_t},
        "roles": {"per_tubelet_counts": roles},
        "geometry": {
            "values": geometry,
            "width_floor_saturation_rate": 0.25,
            "height_floor_saturation_rate": 0.50,
            "width_ceiling_saturation_rate": 0.0,
            "height_ceiling_saturation_rate": 0.0,
        },
        "ragged_execution": {
            "attention_pairs": 4096,
            "requested_physical_tokens": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
            "unique_physical_tokens": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
            "padded_heavy_tokens": 0,
            "executed_patch_tokens": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
        },
        "gt_for_route_used": False,
        "teacher_used": False,
        "oracle_used": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }


def _telemetry_payload() -> dict:
    return {
        "schema_version": "georoute_formal_development_telemetry_v1",
        "development_only": True,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "oracle_used": False,
        "raw_prediction_cache_used": False,
        "world_size": 1,
        "local_batch_size": 1,
        "dataset_count": 1,
        "record_count": 1,
        "unique_dataset_count": 1,
        "sampler_padding_count": 0,
        "population_sha256": "b" * 64,
        "records": [
            {
                "dataset_index": 0,
                "video_id": "gate-a",
                "route": _route_telemetry(),
            }
        ],
    }


def _bound_config(tmp_path: Path, *, arm: str = "native_1cell_main"):
    source = (
        ROOT / "configs" / "adatad" / "thumos" / "georoute_dynamic_scnr_stage1_base.py"
    )
    annotation = tmp_path / "development.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "fit-a": {"subset": "training", "annotations": []},
                    "gate-a": {"subset": "training", "annotations": []},
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"fit": ["fit-a"], "gate": ["gate-a"]}}),
        encoding="utf-8",
    )
    class_map = tmp_path / "class_map.txt"
    class_map.write_text("action\n", encoding="utf-8")
    pretrained = tmp_path / "pretrained.pth"
    pretrained.write_bytes(b"contract-only")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    return bind_dynamic_floor_m2_config(
        source_config_path=source,
        arm=arm,
        seed=DYNAMIC_FLOOR_M2_SEED,
        work_dir=tmp_path / "training" / "gpu1_id0",
        manifest_path=manifest,
        development_annotation_path=annotation,
        class_map_path=class_map,
        development_video_root=video_root,
        pretrained_checkpoint_path=pretrained,
        runtime_commit="c" * 40,
    )


def test_dynamic_floor_m2_freezes_only_floor_and_counterbalances_cost():
    validate_frozen_dynamic_floor_m2_contract()
    assert DYNAMIC_FLOOR_M2_ARM_ORDER == tuple(DYNAMIC_FLOOR_M2_ARMS)
    assert [
        DYNAMIC_FLOOR_M2_ARMS[arm]["roi_extent_floor_cells"]
        for arm in DYNAMIC_FLOOR_M2_ARM_ORDER
    ] == [1, 2]
    assert DYNAMIC_FLOOR_M2_COST_ORDER == (
        DYNAMIC_FLOOR_M2_ARM_ORDER[0],
        DYNAMIC_FLOOR_M2_ARM_ORDER[1],
        DYNAMIC_FLOOR_M2_ARM_ORDER[1],
        DYNAMIC_FLOOR_M2_ARM_ORDER[0],
    )


def test_dynamic_floor_m2_role_replay_separates_model_and_execution_commits(
    tmp_path: Path,
):
    cfg = _bound_config(tmp_path)
    cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = True
    cfg.model.backbone.custom.georoute_role_calibration_telemetry_enabled = True
    cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    cfg.georoute_development_profile = dict(enabled=False)
    binding = validate_dynamic_floor_m2_config(
        cfg,
        arm="native_1cell_main",
        phase="accuracy",
    )
    cfg.georoute_phase_m_binding = dict(
        schema_version="georoute_phase_m_diagnostic_replay_v1",
        variant="native_1cell_main",
        seed=DYNAMIC_FLOOR_M2_SEED,
        source_experiment_commit="c" * 40,
        runtime_commit="d" * 40,
        source_bound_config_sha256="1" * 64,
        source_checkpoint_sha256="2" * 64,
        source_prediction_sha256="3" * 64,
        source_population_sha256="4" * 64,
        source_dataset_count=136,
        role_calibration_telemetry_enabled=True,
        instrumentation_only=True,
        fixed_role_quota_used=False,
        changes_route_or_execution=False,
        official_test_opened=False,
    )

    assert (
        resolve_dynamic_floor_m2_accuracy_execution_commit(
            cfg,
            binding=binding,
        )
        == "d" * 40
    )
    cfg.georoute_phase_m_binding.changes_route_or_execution = True
    with pytest.raises(ValueError, match="execution binding"):
        resolve_dynamic_floor_m2_accuracy_execution_commit(
            cfg,
            binding=binding,
        )


@pytest.mark.parametrize(
    ("pair_mode", "role_calibration_enabled"),
    [("role_off", False), ("role_on", True)],
)
def test_dynamic_floor_m2_role_neutrality_pair_separates_execution_commit(
    tmp_path: Path,
    pair_mode: str,
    role_calibration_enabled: bool,
):
    cfg = _bound_config(tmp_path)
    cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = True
    cfg.model.backbone.custom.georoute_role_calibration_telemetry_enabled = (
        role_calibration_enabled
    )
    cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    cfg.georoute_development_profile = dict(enabled=False)
    binding = validate_dynamic_floor_m2_config(
        cfg,
        arm="native_1cell_main",
        phase="accuracy",
    )
    cfg.georoute_phase_m_binding = dict(
        schema_version=DYNAMIC_FLOOR_M2_ROLE_NEUTRALITY_PAIR_SCHEMA,
        pair_mode=pair_mode,
        variant="native_1cell_main",
        seed=DYNAMIC_FLOOR_M2_SEED,
        source_experiment_commit="c" * 40,
        runtime_commit="d" * 40,
        source_bound_config_sha256="1" * 64,
        source_checkpoint_sha256="2" * 64,
        source_prediction_sha256="3" * 64,
        source_population_sha256="4" * 64,
        source_dataset_count=136,
        role_calibration_telemetry_enabled=role_calibration_enabled,
        instrumentation_only=True,
        same_slurm_job=True,
        same_visible_gpu=True,
        serial_execution=True,
        fixed_role_quota_used=False,
        changes_route_or_execution=False,
        official_test_opened=False,
        gt_for_route_used=False,
        teacher_for_route_used=False,
        oracle_used=False,
        raw_prediction_cache_used=False,
    )

    assert (
        resolve_dynamic_floor_m2_accuracy_execution_commit(
            cfg,
            binding=binding,
        )
        == "d" * 40
    )
    cfg.georoute_phase_m_binding.same_visible_gpu = False
    with pytest.raises(ValueError, match="neutrality pair execution binding"):
        resolve_dynamic_floor_m2_accuracy_execution_commit(
            cfg,
            binding=binding,
        )


@pytest.mark.parametrize(
    ("pair_mode", "role_calibration_enabled"),
    [("role_off", False), ("role_on", True)],
)
def test_dynamic_floor_m2_strict_role_triplet_declares_execution_override(
    tmp_path: Path,
    pair_mode: str,
    role_calibration_enabled: bool,
):
    cfg = _bound_config(tmp_path)
    cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = True
    cfg.model.backbone.custom.georoute_role_calibration_telemetry_enabled = (
        role_calibration_enabled
    )
    cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    cfg.georoute_development_profile = dict(enabled=False)
    binding = validate_dynamic_floor_m2_config(
        cfg,
        arm="native_1cell_main",
        phase="accuracy",
    )
    cfg.georoute_phase_m_binding = dict(
        schema_version=DYNAMIC_FLOOR_M2_ROLE_STRICT_TRIPLET_SCHEMA,
        pair_mode=pair_mode,
        variant="native_1cell_main",
        seed=DYNAMIC_FLOOR_M2_SEED,
        source_experiment_commit="c" * 40,
        runtime_commit="d" * 40,
        source_bound_config_sha256="1" * 64,
        source_checkpoint_sha256="2" * 64,
        source_prediction_sha256="3" * 64,
        source_population_sha256="4" * 64,
        source_dataset_count=136,
        role_calibration_telemetry_enabled=role_calibration_enabled,
        instrumentation_only=False,
        same_slurm_job=True,
        same_visible_gpu=True,
        serial_execution=True,
        fixed_role_quota_used=False,
        changes_route_or_execution=True,
        strict_deterministic_algorithms=True,
        sdp_backend="math",
        tf32_enabled=False,
        deterministic_override_changes_heavy_execution=True,
        role_calibration_instrumentation_only=True,
        role_calibration_changes_route_or_execution=False,
        source_execution_reproduced=False,
        strict_determinism_diagnostic_only=True,
        official_test_opened=False,
        gt_for_route_used=False,
        teacher_for_route_used=False,
        oracle_used=False,
        raw_prediction_cache_used=False,
    )

    assert (
        resolve_dynamic_floor_m2_accuracy_execution_commit(
            cfg,
            binding=binding,
        )
        == "d" * 40
    )
    cfg.georoute_phase_m_binding.sdp_backend = "memory_efficient"
    with pytest.raises(ValueError, match="neutrality pair execution binding"):
        resolve_dynamic_floor_m2_accuracy_execution_commit(
            cfg,
            binding=binding,
        )


@pytest.mark.parametrize("probe_mode", ["centered_a", "centered_b"])
def test_dynamic_floor_m2_residual_centering_probe_is_no_metric_route_intervention(
    tmp_path: Path,
    probe_mode: str,
):
    cfg = _bound_config(tmp_path)
    custom = cfg.model.backbone.custom
    custom.georoute_branch_calibration_mode = "residual_window_center"
    custom.georoute_diagnostic_telemetry_enabled = True
    custom.georoute_role_calibration_telemetry_enabled = True
    cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    cfg.georoute_development_profile = dict(enabled=False)
    binding = validate_dynamic_floor_m2_config(
        cfg,
        arm="native_1cell_main",
        phase="accuracy",
    )
    cfg.georoute_phase_m_binding = dict(
        schema_version=DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_PROBE_SCHEMA,
        probe_mode=probe_mode,
        variant="native_1cell_main",
        seed=DYNAMIC_FLOOR_M2_SEED,
        source_experiment_commit="c" * 40,
        runtime_commit="d" * 40,
        source_bound_config_sha256="1" * 64,
        source_checkpoint_sha256="2" * 64,
        source_prediction_sha256="3" * 64,
        source_population_sha256="4" * 64,
        source_dataset_count=136,
        branch_calibration_mode="residual_window_center",
        branch_calibration_scope="complete_window_all_valid_candidates",
        role_calibration_telemetry_enabled=True,
        mechanism_probe_only=True,
        training_performed=False,
        same_slurm_job=True,
        same_visible_gpu=True,
        serial_execution=True,
        strict_deterministic_algorithms=True,
        sdp_backend="math",
        tf32_enabled=False,
        fixed_role_quota_used=False,
        q_ctx_used=False,
        changes_route_or_execution=True,
        metric_evaluation_enabled=False,
        official_test_opened=False,
        gt_for_route_used=False,
        teacher_for_route_used=False,
        oracle_used=False,
        raw_prediction_cache_used=False,
    )

    assert (
        resolve_dynamic_floor_m2_accuracy_execution_commit(cfg, binding=binding)
        == "d" * 40
    )
    cfg.georoute_phase_m_binding.metric_evaluation_enabled = True
    with pytest.raises(ValueError, match="residual-centering probe execution binding"):
        resolve_dynamic_floor_m2_accuracy_execution_commit(cfg, binding=binding)


def test_residual_centering_probe_test_arguments_force_no_metric_evaluation():
    arguments = _build_probe_test_arguments(
        command_prefix=["python"],
        bound_config=Path("bound.py"),
        checkpoint=Path("checkpoint.pth"),
        seed=DYNAMIC_FLOOR_M2_SEED,
    )
    assert arguments == [
        "python",
        "tools/test.py",
        "bound.py",
        "--checkpoint",
        "checkpoint.pth",
        "--seed",
        str(DYNAMIC_FLOOR_M2_SEED),
        "--id",
        "0",
        "--not_eval",
    ]


def _residual_centering_branch_telemetry(*, mean_after: float = 0.0) -> dict:
    return {
        "records": [
            {
                "dataset_index": 0,
                "video_id": "gate-a",
                "route": {
                    "tubelet_count": 2,
                    "item_count": 3,
                    "selected_physical_index_sha256": "a" * 64,
                    "branch_calibration": {
                        "schema_version": "scnr_dynamic_branch_calibration_window_v1",
                        "mode": "residual_window_center",
                        "target": "delta_residual",
                        "scope": "complete_window_all_valid_candidates",
                        "changes_q_base": False,
                        "changes_delta_roi": False,
                        "changes_context_zero_modifier": False,
                        "changes_budget_or_role_quota": False,
                        "mean_detached": False,
                        "valid_candidate_count": 6,
                        "residual_valid_mean_before": 2.5,
                        "residual_valid_mean_after": mean_after,
                    },
                },
            }
        ]
    }


def test_residual_centering_probe_validates_transform_and_route_hash():
    telemetry = _residual_centering_branch_telemetry(mean_after=1e-7)
    summary = summarize_residual_centering_branch_payload(telemetry)
    assert summary["transform_receipts_valid"] is True
    assert summary["metric_consumed"] is False
    assert summary["valid_candidate_count_min"] == 6
    first_hash = _route_payload_sha256(telemetry)
    changed = copy.deepcopy(telemetry)
    changed["records"][0]["route"]["selected_physical_index_sha256"] = "b" * 64
    assert _route_payload_sha256(changed) != first_hash

    invalid = _residual_centering_branch_telemetry(mean_after=1e-3)
    with pytest.raises(ValueError, match="branch receipt is invalid"):
        summarize_residual_centering_branch_payload(invalid)


def test_residual_centering_probe_gate_passes_or_holds_without_training():
    passing = classify_residual_centering_role_gate(
        {
            "roles": {
                "valid": {
                    "counts": {"context": 2, "roi": 3, "residual": 7}
                },
                "selected": {
                    "counts": {"context": 0, "roi": 1, "residual": 5}
                },
            }
        }
    )
    assert passing["passed"] is True
    assert passing["performance_claim_allowed"] is False

    held = classify_residual_centering_role_gate(
        {
            "roles": {
                "valid": {
                    "counts": {"context": 0, "roi": 0, "residual": 12}
                },
                "selected": {
                    "counts": {"context": 0, "roi": 0, "residual": 6}
                },
            }
        }
    )
    assert held["passed"] is False
    assert held["status"].endswith("NO_TRAINING")
    assert held["conditions"] == {
        "valid_context_reachable": False,
        "valid_roi_reachable": False,
        "selected_non_residual_reachable": False,
        "residual_not_all_valid_candidates": False,
    }


def _timed_cost_audit(*, attention_pairs: int | None = None) -> dict:
    clip_counts = [DYNAMIC_FLOOR_M2_WINDOW_BUDGET // 48 for _ in range(48)]
    expected_attention_pairs = sum(value**2 for value in clip_counts)
    return {
        "route_mode": "dynamic_scnr",
        "policy_estimator": "straight_through",
        "window_token_budget": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
        "window_budget_is_global": True,
        "fixed_context_quota": False,
        "fixed_per_tubelet_k": False,
        "k_t_allows_zero": True,
        "zero_carrier_mode": "masked_zero",
        "requested_physical_tokens_per_window": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
        "unique_physical_tokens_per_window": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
        "executed_patch_tokens_per_window": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
        "padded_heavy_tokens_per_window": 0,
        "heavy_backbone_forward_count": 1,
        "geometry_extent_floor_mode": "native_cells",
        "geometry_extent_floor_cells": 1,
        "diagnostic_telemetry_enabled": False,
        "uses_gt_for_route": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_test_evidence": False,
        "k_per_tubelet": [[64 for _ in range(384)]],
        "physical_indices_sha256": "a" * 64,
        "k_t_min": 64,
        "k_t_max": 64,
        "k_t_zero_count": 0,
        "role_counts": {
            "context": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
            "roi": 0,
            "residual": 0,
        },
        "packed": {
            "schema_version": "videomae_native_ragged_v1",
            "execution_mode": "true_clip_ragged_no_padding",
            "batch_size": 1,
            "clip_token_counts": [clip_counts],
            "attention_pairs_per_window": [
                expected_attention_pairs if attention_pairs is None else attention_pairs
            ],
        },
    }


def test_dynamic_floor_m2_cost_audit_reads_native_ragged_attention_ledger():
    validated = _validate_cost_audit(_timed_cost_audit(), floor_cells=1)
    assert validated["attention_pairs"] == 48 * 512**2


def test_dynamic_floor_m2_cost_audit_rejects_inconsistent_attention_ledger():
    with pytest.raises(RuntimeError, match="ragged ledger is invalid"):
        _validate_cost_audit(_timed_cost_audit(attention_pairs=1), floor_cells=1)


class _FakeCudaEvent:
    def record(self):
        return None

    def elapsed_time(self, _end):
        return 1.0


class _FakeCuda:
    @staticmethod
    def Event(*, enable_timing):
        assert enable_timing is True
        return _FakeCudaEvent()


class _FakeHookHandle:
    def remove(self):
        return None


class _FakeModule:
    def __init__(self):
        self._before = []
        self._after = []

    def register_forward_pre_hook(self, hook):
        self._before.append(hook)
        return _FakeHookHandle()

    def register_forward_hook(self, hook):
        self._after.append(hook)
        return _FakeHookHandle()

    def __call__(self):
        for hook in self._before:
            hook(self, ())
        result = object()
        for hook in self._after:
            hook(self, (), result)
        return result


class _FakeMethodTarget:
    def forward_dynamic(self):
        return "scout"

    def forward_native_ragged(self):
        return "heavy"

    def forward_ragged(self):
        return "sparse"

    def forward_test(self):
        return "forward"

    def post_processing(self):
        return "post"


def test_dynamic_floor_m2_cost_instruments_direct_sparse_ragged_method():
    torch_module = SimpleNamespace(cuda=_FakeCuda())
    wrapper = _FakeModule()
    wrapper.scout = _FakeMethodTarget()
    wrapper.sparse_adapter = _FakeMethodTarget()
    heavy = _FakeMethodTarget()
    heavy.patch_embed = _FakeModule()
    model = _FakeMethodTarget()
    model.projection = _FakeModule()
    model.neck = _FakeModule()
    model.rpn_head = _FakeMethodTarget()

    module_events, method_events = _build_cost_cuda_events(
        torch_module,
        model=model,
        wrapper=wrapper,
        heavy=heavy,
    )
    try:
        wrapper()
        heavy.patch_embed()
        model.projection()
        model.neck()
        wrapper.scout.forward_dynamic()
        heavy.forward_native_ragged()
        wrapper.sparse_adapter.forward_ragged()
        model.rpn_head.forward_test()
        model.forward_test()
        model.post_processing()
        timings = _read_cost_cuda_timings(module_events, method_events)
        assert timings["sparse_adapter_ms"] == pytest.approx(1.0)
        assert _invalid_cost_cuda_stages(timings) == ()
        timings["sparse_adapter_ms"] = 0.0
        assert _invalid_cost_cuda_stages(timings) == ("sparse_adapter_ms",)
    finally:
        module_events.close()
        for event in reversed(tuple(method_events.values())):
            event.close()


def test_dynamic_floor_m2_binder_preserves_dynamic_recipe_and_fit_gate(
    tmp_path: Path,
):
    source = (
        ROOT / "configs" / "adatad" / "thumos" / "georoute_dynamic_scnr_stage1_base.py"
    )
    annotation = tmp_path / "development.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "fit-a": {"subset": "training", "annotations": []},
                    "gate-a": {"subset": "training", "annotations": []},
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"fit": ["fit-a"], "gate": ["gate-a"]}}),
        encoding="utf-8",
    )
    class_map = tmp_path / "class_map.txt"
    class_map.write_text("action\n", encoding="utf-8")
    pretrained = tmp_path / "pretrained.pth"
    pretrained.write_bytes(b"contract-only")
    video_root = tmp_path / "videos"
    video_root.mkdir()

    configs = {}
    for arm in DYNAMIC_FLOOR_M2_ARM_ORDER:
        configs[arm] = bind_dynamic_floor_m2_config(
            source_config_path=source,
            arm=arm,
            seed=DYNAMIC_FLOOR_M2_SEED,
            work_dir=tmp_path / arm,
            manifest_path=manifest,
            development_annotation_path=annotation,
            class_map_path=class_map,
            development_video_root=video_root,
            pretrained_checkpoint_path=pretrained,
            runtime_commit="c" * 40,
        )
        binding = validate_dynamic_floor_m2_config(configs[arm], arm=arm)
        assert binding["training_video_ids"] == ["fit-a"]
        assert binding["evaluation_video_ids"] == ["gate-a"]
        assert configs[arm].dataset.train.block_list == ["gate-a"]
        assert configs[arm].dataset.test.block_list == ["fit-a"]
        assert configs[arm].solver.fp16_compress is False
        assert configs[arm].solver.static_graph is False
        assert configs[arm].workflow.capture_amp_rng_state is True

    left = configs[DYNAMIC_FLOOR_M2_ARM_ORDER[0]].to_dict()
    right = configs[DYNAMIC_FLOOR_M2_ARM_ORDER[1]].to_dict()
    assert left["model"]["backbone"]["custom"]["georoute_roi_extent_floor_cells"] == 1
    assert right["model"]["backbone"]["custom"]["georoute_roi_extent_floor_cells"] == 2


def test_dynamic_floor_m2_telemetry_summarizes_full_tubelet_state(tmp_path: Path):
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(_telemetry_payload()), encoding="utf-8")
    summary = summarize_dynamic_floor_m2_telemetry(path)
    assert summary["dataset_count"] == 1
    assert summary["geometry"]["width"]["p50"] == pytest.approx(0.25)
    assert summary["geometry"]["height_floor_saturation_rate"] == pytest.approx(0.50)
    assert summary["k_t"]["distribution"]["min"] == 64
    assert summary["k_t"]["zero_count"] == 0
    assert summary["roles"]["counts"] == {
        "context": 6144,
        "roi": 12288,
        "residual": 6144,
    }
    assert summary["official_test_opened"] is False


def test_dynamic_floor_m2_telemetry_rejects_padding(tmp_path: Path):
    payload = copy.deepcopy(_telemetry_payload())
    payload["records"][0]["route"]["ragged_execution"]["padded_heavy_tokens"] = 1
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="true-ragged exact B"):
        summarize_dynamic_floor_m2_telemetry(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (("geometry", "values", 0, 2), -0.1, "K_t/role partition"),
        (("geometry", "width_floor_saturation_rate"), 1.1, "must be in"),
        (("ragged_execution", "attention_pairs"), -1, "non-negative"),
    ],
)
def test_dynamic_floor_m2_telemetry_rejects_impossible_ranges(
    tmp_path: Path, field, value, message
):
    payload = copy.deepcopy(_telemetry_payload())
    target = payload["records"][0]["route"]
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        summarize_dynamic_floor_m2_telemetry(path)


def test_dynamic_floor_m2_checkpoint_sidecar_proves_exact_training(tmp_path: Path):
    cfg = _bound_config(tmp_path)
    metadata = build_dynamic_floor_m2_checkpoint_metadata(
        cfg,
        seed=DYNAMIC_FLOOR_M2_SEED,
        epoch=59,
        successful_updates=120,
        train_batches_per_epoch=2,
        amp_skipped_attempts=3,
        max_amp_retries_observed=1,
        optimizer_attempts=123,
        consumed_batches=120,
        replay_attempts=3,
        scheduler_advances=120,
        ema_updates=120,
        world_size=1,
    )
    checkpoint = tmp_path / "epoch_59.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint-payload")
    sidecar = {
        "schema_version": DYNAMIC_FLOOR_M2_CHECKPOINT_SIDECAR_SCHEMA,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "experiment_metadata": metadata,
    }
    sidecar["sidecar_sha256"] = canonical_sha256(sidecar)
    sidecar_path = Path(str(checkpoint) + ".metadata.json")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    validated = validate_dynamic_floor_m2_checkpoint_sidecar(
        checkpoint,
        binding=cfg.georoute_dynamic_floor_m2_binding,
        cfg=cfg,
    )
    assert validated["experiment_metadata"]["successful_updates"] == 120
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sidecar is invalid"):
        validate_dynamic_floor_m2_checkpoint_sidecar(checkpoint)


def _cost_profile_fixture(tmp_path: Path) -> dict:
    run_root = tmp_path / "run"
    cost_root = run_root / "cost"
    cost_root.mkdir(parents=True)
    population = "f" * 64
    accuracy_population = "b" * 64
    fake_stages = {}
    stage_result_receipts = {}
    expected_cost_config_hashes = {}
    for arm in DYNAMIC_FLOOR_M2_ARM_ORDER:
        arm_inputs = run_root / "fixture_inputs" / arm
        arm_inputs.mkdir(parents=True)
        train_cfg = _bound_config(arm_inputs, arm=arm)
        accuracy_cfg = copy.deepcopy(train_cfg)
        accuracy_cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = True
        accuracy_cfg.georoute_diagnostic_telemetry = dict(enabled=True)
        train_path = arm_inputs / "train.py"
        accuracy_path = arm_inputs / "accuracy.py"
        train_cfg.dump(str(train_path))
        accuracy_cfg.dump(str(accuracy_path))
        stage = {
            "arm": arm,
            "runtime_commit": "c" * 40,
            "population_sha256": accuracy_population,
            "checkpoint_receipt": {
                "sha256": ("c" if arm == DYNAMIC_FLOOR_M2_ARM_ORDER[0] else "d") * 64
            },
            "config_receipts": {
                "train": {
                    "path": str(train_path.resolve()),
                    "sha256": sha256_file(train_path),
                },
                "accuracy": {
                    "path": str(accuracy_path.resolve()),
                    "sha256": sha256_file(accuracy_path),
                },
            },
        }
        cost_cfg = build_dynamic_floor_m2_cost_config(stage, arm=arm)
        assert cost_cfg.post_processing.sliding_window is True
        expected_cost_config_hashes[arm] = canonical_sha256(cost_cfg.to_dict())
        stage["stage_result_sha256"] = canonical_sha256(stage)
        stage_path = (
            run_root
            / "development"
            / DYNAMIC_FLOOR_M2_ARMS[arm]["slug"]
            / "seed3407"
            / "stage_result.json"
        )
        stage_path.parent.mkdir(parents=True)
        stage_path.write_text(json.dumps(stage), encoding="utf-8")
        fake_stages[arm] = stage
        stage_result_receipts[arm] = {
            "path": str(stage_path.resolve()),
            "sha256": sha256_file(stage_path),
            "stage_result_sha256": stage["stage_result_sha256"],
        }
    latency_keys = (
        "input_pipeline_serial_ms",
        "h2d_ms",
        "model_forward_ms",
        "postprocess_ms",
        "decode_to_window_output_wall_ms",
        "final_video_nms_ms",
        "end_to_end_serial_ms",
    )
    raw_rows = []
    pass_receipts = []
    for pass_index, arm in enumerate(DYNAMIC_FLOOR_M2_COST_ORDER):
        base = float(pass_index + 1)
        energy_start = 1.0 + 2.0 * pass_index
        energy_end = energy_start + 0.1
        nms_start = energy_start + 0.2
        nms_end = nms_start + 0.1
        row = {
            "schema_version": "scnr_dynamic_floor_m2_cost_sample_v1",
            "pass_index": pass_index,
            "arm": arm,
            "sample_ordinal": 0,
            "population_sha256": population,
            **{key: base for key in latency_keys},
            "decode_to_window_output_wall_ms": 100.0,
            "final_video_nms_ms": 100.0,
            "end_to_end_serial_ms": 200.0,
            "backbone_wrapper_ms": 9.0,
            "scout_ms": 1.0,
            "patch_embed_ms": 1.0,
            "heavy_backbone_ms": 5.0,
            "sparse_adapter_ms": 1.0,
            "projection_ms": 1.0,
            "neck_ms": 1.0,
            "head_ms": 1.0,
            "peak_gpu_allocated_mb": 100.0 + base,
            "peak_gpu_reserved_mb": 120.0 + base,
            "gpu_energy_j": 20.0,
            "energy_window_monotonic_s": [energy_start, energy_end],
            "nms_energy_window_monotonic_s": [nms_start, nms_end],
            "route_audit": {
                "exact_window_budget": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
                "padded_heavy_tokens": 0,
            },
            "window_id": "gate-a:0:exposure0",
            "physical_window_id": "gate-a:0",
            "video_id": "gate-a",
            "loader_ordinal": 0,
        }
        row["sample_sha256"] = canonical_sha256(row)
        raw_rows.append(row)
        receipt = {
            "pass_index": pass_index,
            "arm": arm,
            "sample_count": 1,
            "population_sha256": population,
            "accuracy_population_sha256": accuracy_population,
            "sample_manifest_sha256": canonical_sha256([row["window_id"]]),
            "checkpoint_sha256": fake_stages[arm]["checkpoint_receipt"]["sha256"],
            "bound_accuracy_config_sha256": fake_stages[arm]["config_receipts"][
                "accuracy"
            ]["sha256"],
            "cost_config_sha256": expected_cost_config_hashes[arm],
            "diagnostic_telemetry_inside_timed_forward": False,
        }
        receipt["pass_sha256"] = canonical_sha256(receipt)
        pass_receipts.append(receipt)

    raw_path = cost_root / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    power_path = cost_root / "power.jsonl"
    power_path.write_text(
        "".join(
            json.dumps(
                {
                    "sequence": index,
                    "monotonic_s": float(index),
                    "timestamp_ms": float(index) * 1000.0,
                    "power_w": 100.0,
                }
            )
            + "\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    attempt_trace = cost_root / "attempt.power.jsonl"
    attempt_trace.write_text('{"monotonic_ns":1,"power_w":100.0}\n', encoding="utf-8")
    attempt_report = {
        "schema_version": "spatial_zoom_s1_power_buffered_sidecar_attempt_v1",
        "status": "PASS",
        "interval_ms": 20,
        "trace_io_inside_sampling_loop": False,
        "trace_file_sha256": sha256_file(attempt_trace),
    }
    attempt_report["attempt_sha256"] = canonical_sha256(attempt_report)
    attempt_report_path = cost_root / "attempt.json"
    attempt_report_path.write_text(json.dumps(attempt_report), encoding="utf-8")

    def distribution(values):
        values = sorted(values)
        p50 = 0.5 * (values[0] + values[1])
        p95 = values[0] * 0.05 + values[1] * 0.95
        return {
            "mean": p50,
            "p50": p50,
            "p95": p95,
            "min": values[0],
            "max": values[1],
        }

    arm_summaries = {}
    for arm in DYNAMIC_FLOOR_M2_ARM_ORDER:
        rows = [row for row in raw_rows if row["arm"] == arm]
        arm_summaries[arm] = {
            "pass_count": 2,
            "sample_count": 2,
            "population_sha256": population,
            "latency_ms": {
                key: distribution([row[key] for row in rows]) for key in latency_keys
            },
            "resources": {
                "peak_gpu_allocated_mb": max(
                    row["peak_gpu_allocated_mb"] for row in rows
                ),
                "peak_gpu_reserved_mb": max(
                    row["peak_gpu_reserved_mb"] for row in rows
                ),
                "gross_gpu_energy_j": sum(row["gpu_energy_j"] for row in rows),
            },
        }
    artifacts = {
        "raw_samples": {
            "path": str(raw_path.resolve()),
            "sha256": sha256_file(raw_path),
        },
        "power_trace": {
            "path": str(power_path.resolve()),
            "sha256": sha256_file(power_path),
        },
        "sidecar_attempt_report": {
            "path": str(attempt_report_path.resolve()),
            "sha256": sha256_file(attempt_report_path),
        },
        "sidecar_attempt_trace": {
            "path": str(attempt_trace.resolve()),
            "sha256": sha256_file(attempt_trace),
        },
    }
    profile = {
        "schema_version": DYNAMIC_FLOOR_M2_COST_SCHEMA,
        "status": "PASS_DYNAMIC_FLOOR_M2_FULL_STACK_COST",
        "study_id": "scnr_dynamic_floor_m2_v1",
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "runtime_commit": "c" * 40,
        "run_root": str(run_root.resolve()),
        "profile_order": list(DYNAMIC_FLOOR_M2_COST_ORDER),
        "warmup_samples_per_pass": 50,
        "batch_size": 1,
        "loader_workers": 0,
        "world_size": 1,
        "power_interval_ms": 20,
        "population_sha256": population,
        "accuracy_population_sha256": accuracy_population,
        "raw_sample_count": 4,
        "scope": {
            "decode": True,
            "preprocess": True,
            "host_to_device": True,
            "scout": True,
            "route": True,
            "patch_embed": True,
            "backbone": True,
            "adapter": True,
            "detector": True,
            "nms": True,
            "diagnostic_telemetry_inside_timed_forward": False,
            "same_gpu_counterbalanced": True,
            "development_only": True,
        },
        "arm_summaries": arm_summaries,
        "pass_receipts": pass_receipts,
        "stage_result_receipts": stage_result_receipts,
        "hardware_fingerprint": "a" * 64,
        "software_fingerprint": "b" * 64,
        "artifact_receipts": artifacts,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    profile["profile_sha256"] = canonical_sha256(profile)
    return profile


def test_dynamic_floor_m2_cost_profile_recomputes_raw_artifacts(
    tmp_path: Path, monkeypatch
):
    profile = _cost_profile_fixture(tmp_path)
    monkeypatch.setattr(
        "tools.bata.georoute_dynamic_floor_m2_contract.validate_dynamic_floor_m2_stage_result",
        lambda result, **_kwargs: result,
    )
    validated = validate_dynamic_floor_m2_cost_profile(
        profile, expected_commit="c" * 40
    )
    assert validated["arm_summaries"]["native_1cell_main"]["pass_count"] == 2
    raw_path = Path(profile["artifact_receipts"]["raw_samples"]["path"])
    raw_path.write_text(raw_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_dynamic_floor_m2_cost_profile(profile)


def test_dynamic_floor_m2_cost_rejects_self_hashed_forged_energy(
    tmp_path: Path, monkeypatch
):
    profile = _cost_profile_fixture(tmp_path)
    monkeypatch.setattr(
        "tools.bata.georoute_dynamic_floor_m2_contract.validate_dynamic_floor_m2_stage_result",
        lambda result, **_kwargs: result,
    )
    raw_path = Path(profile["artifact_receipts"]["raw_samples"]["path"])
    rows = [
        json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["gpu_energy_j"] = 999.0
    rows[0].pop("sample_sha256")
    rows[0]["sample_sha256"] = canonical_sha256(rows[0])
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    profile["artifact_receipts"]["raw_samples"]["sha256"] = sha256_file(raw_path)
    arm = rows[0]["arm"]
    profile["arm_summaries"][arm]["resources"]["gross_gpu_energy_j"] = sum(
        row["gpu_energy_j"] for row in rows if row["arm"] == arm
    )
    profile.pop("profile_sha256")
    profile["profile_sha256"] = canonical_sha256(profile)
    with pytest.raises(ValueError, match="energy is not reproducible"):
        validate_dynamic_floor_m2_cost_profile(profile)


def test_dynamic_floor_m2_incomplete_finalizer_never_selects_a_floor():
    result = finalize_dynamic_floor_m2({}, None, expected_commit="c" * 40)
    assert result["status"] == "FAIL_INCOMPLETE_NO_FLOOR_INFERENCE"
    assert result["single_seed_floor_selection_allowed"] is False
    assert result["descriptive_contrasts"] == {}


def test_dynamic_floor_m2_complete_finalizer_remains_descriptive(monkeypatch):
    metrics = {
        "average_mAP": 50.0,
        "mAP@0.3": 60.0,
        "mAP@0.4": 55.0,
        "mAP@0.5": 50.0,
        "mAP@0.6": 45.0,
        "mAP@0.7": 40.0,
        "high_iou_composite": 42.5,
    }
    stages = {}
    for index, arm in enumerate(DYNAMIC_FLOOR_M2_ARM_ORDER):
        stages[arm] = {
            "arm": arm,
            "metrics": {key: value - index for key, value in metrics.items()},
            "telemetry_summary": {
                "geometry": {
                    "width_floor_saturation_rate": 0.1 + 0.1 * index,
                    "height_floor_saturation_rate": 0.2 + 0.1 * index,
                    "area": {"p50": 0.3 + 0.1 * index},
                }
            },
        }
    cost = {
        "arm_summaries": {
            arm: {
                "latency_ms": {
                    "end_to_end_serial_ms": {
                        "p50": 10.0 + index,
                        "p95": 12.0 + index,
                    }
                }
            }
            for index, arm in enumerate(DYNAMIC_FLOOR_M2_ARM_ORDER)
        }
    }
    monkeypatch.setattr(
        "tools.bata.georoute_dynamic_floor_m2_contract.validate_dynamic_floor_m2_stage_result",
        lambda result, **_kwargs: result,
    )
    monkeypatch.setattr(
        "tools.bata.georoute_dynamic_floor_m2_contract.validate_dynamic_floor_m2_cost_profile",
        lambda profile, **_kwargs: profile,
    )
    result = finalize_dynamic_floor_m2(stages, cost, expected_commit="c" * 40)
    assert result["status"] == "PASS_COMPLETE_DESCRIPTIVE_FLOOR_SENSITIVITY"
    assert (
        result["decision"]
        == "COMPLETE_DESCRIPTIVE_ONLY_M3_REQUIRED_FOR_FLOOR_SELECTION"
    )
    assert result["single_seed_floor_selection_allowed"] is False
    assert result["m3_confirmation_required"] is True


def test_dynamic_floor_m2_recovery_finalizer_binds_runtime_and_execution(
    monkeypatch,
):
    runtime_commit = "a" * 40
    execution_commit = "b" * 40
    cost_execution_commit = "c" * 40
    deployment = {
        "schema_version": "scnr_dynamic_floor_m2_deployment_v1",
        "status": "DEPLOYED_DYNAMIC_FLOOR_M2_DAG",
        "runtime_commit": runtime_commit,
        "jobs": {
            "g1": "1",
            "g2": "2",
            "paired_cost": "3",
            "finalizer": "4",
        },
        "dependencies": {
            "paired_cost": {"type": "afterok", "predecessors": ["1", "2"]},
            "finalizer": {
                "type": "afterany",
                "predecessors": ["1", "2", "3"],
            },
        },
        "official_test_opened": False,
        "paper_claim_allowed": False,
        "recovery": {
            "model_runtime_commit": runtime_commit,
            "execution_commit": execution_commit,
            "cost_execution_commit": cost_execution_commit,
            "model_or_config_code_changed": False,
            "retrained_or_resumed_arms": False,
        },
    }
    deployment["deployment_sha256"] = canonical_sha256(deployment)
    monkeypatch.setenv("SLURM_JOB_ID", "4")
    validated = _validate_deployment(
        deployment,
        expected_commit=runtime_commit,
        expected_execution_commit=execution_commit,
        expected_cost_execution_commit=cost_execution_commit,
    )
    assert validated["deployment_sha256"] == deployment["deployment_sha256"]
    with pytest.raises(ValueError, match="deployment receipt is invalid"):
        _validate_deployment(
            deployment,
            expected_commit=runtime_commit,
            expected_execution_commit="d" * 40,
            expected_cost_execution_commit=cost_execution_commit,
        )
    with pytest.raises(ValueError, match="deployment receipt is invalid"):
        _validate_deployment(
            deployment,
            expected_commit=runtime_commit,
            expected_execution_commit=execution_commit,
            expected_cost_execution_commit="d" * 40,
        )

    original = copy.deepcopy(deployment)
    original.pop("recovery")
    original.pop("deployment_sha256")
    original["deployment_sha256"] = canonical_sha256(original)
    validated_original = _validate_deployment(
        original,
        expected_commit=runtime_commit,
    )
    assert validated_original["deployment_sha256"] == original["deployment_sha256"]


def test_dynamic_floor_m2_execution_sources_freeze_dag_and_cost_scope():
    profiler = (
        ROOT / "tools" / "bata" / "profile_georoute_dynamic_floor_m2.py"
    ).read_text(encoding="utf-8")
    deployer = (
        ROOT / "tools" / "bata" / "deploy_georoute_dynamic_floor_m2.py"
    ).read_text(encoding="utf-8")
    deploy_common = (
        ROOT / "tools" / "bata" / "georoute_hybrid_causal_deploy_common.py"
    ).read_text(encoding="utf-8")
    stage = (
        ROOT / "tools" / "bata" / "georoute_dynamic_floor_m2_stage_runner.py"
    ).read_text(encoding="utf-8")
    scripts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "scripts" / "run_georoute_dynamic_floor_m2_stage_slurm.sh",
            ROOT / "scripts" / "run_georoute_dynamic_floor_m2_cost_slurm.sh",
            ROOT / "scripts" / "run_georoute_dynamic_floor_m2_finalizer_slurm.sh",
        )
    )
    assert "for pass_index, arm in enumerate(DYNAMIC_FLOOR_M2_COST_ORDER)" in profiler
    profiler_names = {
        node.id for node in ast.walk(ast.parse(profiler)) if isinstance(node, ast.Name)
    }
    assert "_cost_config" not in profiler_names
    assert "build_dynamic_floor_m2_cost_config" in profiler_names
    assert '"diagnostic_telemetry_inside_timed_forward": False' in profiler
    assert '"execution_commit": expected_execution_commit' in profiler
    assert "--expected-execution-commit" in scripts
    assert 'dependency_type="afterok"' in deployer
    assert 'dependency_type="afterany"' in deployer
    assert "kill_invalid_dependency=True" in deployer
    assert deployer.count('resource="control"') == 2
    assert 'resource="cpu_control"' not in deployer
    assert '"cpu_control"' not in deploy_common
    assert '"finalizer_gpu_allocation_is_scheduling_overhead": True' in deployer
    assert '"--with-diagnostic-telemetry"' in stage
    assert "CUDA_VISIBLE_DEVICES=" not in scripts
    assert 'taskset -c "${DETECTOR_CPUS}"' in scripts
