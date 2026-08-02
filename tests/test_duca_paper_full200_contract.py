from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata import duca_paper_training
from tools.bata import validate_duca_paper_code_gate as code_gate_validator
from tools.bata import validate_duca_paper_short_window_gate as short_window_gate
from tools.bata.build_duca_paper_matrix_manifest import build_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "dense": "configs/adatad/thumos/duca_paper_dense_actionformer_full200.py",
    "uniform_fixed_k384": (
        "configs/adatad/thumos/duca_paper_uniform_fixed_k384_full200.py"
    ),
    "uniform_mixed_train_k384_eval": (
        "configs/adatad/thumos/duca_paper_uniform_mixed_train_k384_eval_full200.py"
    ),
    "duca_fixed_k384": (
        "configs/adatad/thumos/duca_paper_duca_fixed_k384_full200.py"
    ),
}


@pytest.mark.parametrize("arm", duca_paper_training.ARMS)
def test_stage_a_configs_are_exact_full200_actionformer_contracts(arm):
    cfg = Config.fromfile(REPO_ROOT / CONFIGS[arm])
    contract = duca_paper_training.validate_static_config(cfg)
    assert contract["variant"] == arm
    assert contract["train_video_count"] == 200
    assert contract["evaluation_video_count"] == 211
    assert contract["world_size"] == 2
    assert contract["global_batch_size"] == 2
    assert contract["expected_successful_optimizer_updates"] == 6000
    assert cfg.dataset.val is None
    assert cfg.dataset.train.block_list is None
    assert cfg.dataset.test.block_list is None


@pytest.mark.parametrize("seed", duca_paper_training.SEEDS)
def test_paper_evaluation_accepts_every_registered_seed(seed):
    cfg = Config.fromfile(REPO_ROOT / CONFIGS["duca_fixed_k384"])
    request = duca_paper_training.validate_evaluation_request(
        cfg,
        arm="duca_fixed_k384",
        seed=seed,
        expected_checkpoint_epoch=59,
        checkpoint_state_key="state_dict_ema",
        metrics_json="terminal.json",
    )
    assert request["seed"] == seed
    assert request["evaluation_video_count"] == 211


def test_paper_evaluation_rejects_old_protected_seed():
    cfg = Config.fromfile(REPO_ROOT / CONFIGS["duca_fixed_k384"])
    with pytest.raises(RuntimeError, match="registered arm/seed"):
        duca_paper_training.validate_evaluation_request(
            cfg,
            arm="duca_fixed_k384",
            seed=3407,
            expected_checkpoint_epoch=59,
            checkpoint_state_key="state_dict_ema",
            metrics_json="terminal.json",
        )


class DucaStatelessThumosPaddingDataset:
    def __init__(self, annotation_path: Path):
        self.data_list = [(f"train_{index:03d}", {}) for index in range(200)]
        self.subset_name = "training"
        self.stateless_seed = 3407
        self.ann_file = str(annotation_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return index


def test_two_rank_loader_covers_every_training_video_once_per_epoch(tmp_path):
    if os.name == "nt":
        pytest.skip("the authoritative loader-contract test runs in the Linux gate")
    try:
        import torch
    except OSError as exc:
        pytest.skip(f"local PyTorch runtime is unavailable: {exc}")
    annotation = tmp_path / "annotation.json"
    annotation.write_text("{}\n", encoding="utf-8")
    dataset = DucaStatelessThumosPaddingDataset(annotation)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=42,
    )
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1)
    cfg = Config(
        dict(
            dataset=dict(
                train=dict(
                    type="DucaStatelessThumosPaddingDataset",
                    stateless_seed=3407,
                    subset_name="training",
                )
            ),
            solver=dict(train=dict(batch_size=2)),
        )
    )
    contract = duca_paper_training.derive_train_loader_contract(
        cfg=cfg,
        train_dataset=dataset,
        train_loader=loader,
        world_size=2,
    )
    assert contract["batches_per_rank_per_epoch"] == 100
    assert contract["global_video_exposures"] == 12000
    assert contract["per_video_exposure_count"] == 60


def _write_official_annotation(path: Path, *, include_training: bool = False):
    database = {
        f"validation_{index:03d}": {"subset": "validation", "annotations": []}
        for index in range(211)
    }
    if include_training:
        database.update(
            {
                f"training_{index:03d}": {"subset": "training", "annotations": []}
                for index in range(200)
            }
        )
    path.write_text(
        json.dumps({"database": database}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return database


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_code_gate_fixture(path: Path, *, commit: str) -> None:
    pytest_log = path.with_name("pytest.out")
    pytest_log.write_text("15 passed\n", encoding="utf-8")
    payload = {
        "schema_version": code_gate_validator.SCHEMA,
        "status": "passed",
        "git_commit": commit,
        "slurm_job_id": "12344",
        "pytest_log_path": str(pytest_log.resolve()),
        "pytest_log_sha256": _sha256(pytest_log),
        "official_train_video_count": 200,
        "official_evaluation_video_count": 211,
        "stage_a_logical_cell_count": 12,
        "short_window_gate_pending": True,
        "stage_a_manifest_created": False,
        "stage_a_released": False,
        "stage_b_enabled": False,
        "paper_metric_claim_allowed": False,
    }
    payload["content_sha256"] = code_gate_validator._canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_short_window_gate_fixture(
    path: Path,
    *,
    commit: str,
    pretrain: Path,
    annotation: Path,
    class_map: Path,
    train_data: Path,
    code_gate: Path,
) -> None:
    config = REPO_ROOT / short_window_gate.CONFIG_DEFAULT
    executions = []
    for requested in short_window_gate.REQUESTED_BUDGETS:
        effective = min(requested, 224)
        selected = list(range(effective))
        executions.append(
            {
                "requested_k": requested,
                "effective_k": effective,
                "unique_k": effective,
                "backbone_input_k": effective,
                "dense_valid_length": 231,
                "selected_dense_indices": selected,
                "selected_dense_indices_sha256": (
                    short_window_gate._canonical_sha256(selected)
                ),
                "no_padding": True,
                "no_repetition": True,
                "no_invalid_index": True,
                "heavy_backbone_forward_completed": True,
                "backbone_input_contract": {
                    "schema_version": "duca_dynamic_backbone_input_v1",
                    "measurement_source": (
                        "actual_backbone_wrapper_and_videomae_input_tensors"
                    ),
                    "wrapper_temporal_k": effective,
                    "inner_reconstructed_k": effective,
                    "padding_or_repetition_observed": False,
                },
            }
        )
    payload = {
        "schema_version": short_window_gate.SCHEMA,
        "status": "passed",
        "fail_closed": True,
        "git_commit": commit,
        "prerequisite_clean_linux_code_gate": {
            "path": str(code_gate.resolve()),
            "sha256": _sha256(code_gate),
            "git_commit": commit,
            "slurm_job_id": "12344",
            "status": "passed",
            "claim_scope": "engineering_clean_linux_pytorch_code_only",
        },
        "config": {
            "path": str(config.resolve()),
            "sha256": _sha256(config),
            "arm": "uniform_mixed_train_k384_eval",
        },
        "assets": {
            "pretrain": {"path": str(pretrain), "sha256": _sha256(pretrain)},
            "annotation": {
                "path": str(annotation),
                "sha256": _sha256(annotation),
            },
            "class_map": {"path": str(class_map), "sha256": _sha256(class_map)},
            "train_data_path": str(train_data),
        },
        "audited_file_sha256": {
            relative: _sha256(REPO_ROOT / relative)
            for relative in short_window_gate.AUDITED_PATHS
        },
        "dataset": {
            "dataset_class": "DucaStatelessThumosPaddingDataset",
            "dataset_size": 200,
            "natural_short_count": 1,
            "subquantum_count": 0,
            "selected_sample": {
                "video_exists": True,
                "annotation_valid_length": 231,
            },
        },
        "input_provenance": "real_thumos14_full200_train_video_decode",
        "synthetic_inputs_used": False,
        "validation_or_test_data_used": False,
        "mixed_training_mode": True,
        "requested_budget_order": list(short_window_gate.REQUESTED_BUDGETS),
        "executions": executions,
        "selector_to_unique_gather_to_heavy_backbone_completed": True,
        "slurm_cuda_binding": {
            "slurm_job_id": "12345",
            "logical_device": "cuda:0",
            "logical_cuda_device_count": 1,
            "physical_gpu_index_assumed": False,
        },
        "final_clean_binding": {
            "git_commit_unchanged": True,
            "git_tree_unchanged": True,
            "git_tree_clean_after_gate": True,
        },
        "paper_metric_claim_allowed": False,
        "paper_method_performance_evidence": False,
        "claim_scope": "engineering_short_window_execution_only",
        "stage_a_rerun_required": True,
        "stage_b_enabled": False,
        "official_final_consumed": False,
    }
    payload["content_sha256"] = short_window_gate._canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_exact211_execution_rejects_missing_video(tmp_path):
    annotation = tmp_path / "annotation.json"
    database = _write_official_annotation(annotation)
    video_ids = sorted(database)
    prediction = tmp_path / "prediction.json"
    prediction.write_text(
        json.dumps({"results": {video_id: [] for video_id in video_ids}}) + "\n",
        encoding="utf-8",
    )
    summary = {
        "video_count": 211,
        "post_processing_execution": {
            "window_counts": {video_id: 1 for video_id in video_ids},
            "world_size": 1,
            "dataset_is_sliding_window": True,
            "full_detector_window_merge_nms_evaluation_completed": True,
            "evaluator_evaluate_succeeded": True,
            "evaluation_config": {"subset": "validation", "blocked_videos": None},
        },
    }
    receipt = duca_paper_training.validate_official_evaluation_execution(
        evaluation_summary=summary,
        annotation_path=annotation,
        prediction_path=prediction,
    )
    assert receipt["evaluation_video_count"] == 211
    del summary["post_processing_execution"]["window_counts"][video_ids[-1]]
    with pytest.raises(RuntimeError, match="exact 211-video"):
        duca_paper_training.validate_official_evaluation_execution(
            evaluation_summary=summary,
            annotation_path=annotation,
            prediction_path=prediction,
        )


def test_matrix_manifest_freezes_twelve_full_dataset_cells(tmp_path):
    pretrain = tmp_path / "pretrain.pth"
    pretrain.write_bytes(b"fixture-videomae")
    annotation = tmp_path / "annotation.json"
    _write_official_annotation(annotation, include_training=True)
    class_map = tmp_path / "category_idx.txt"
    class_map.write_text(
        "\n".join(f"class_{index} {index}" for index in range(20)) + "\n",
        encoding="utf-8",
    )
    manifest = build_manifest(
        repo_root=REPO_ROOT,
        expected_commit="a" * 40,
        pretrain_path=pretrain,
        annotation_path=annotation,
        class_map_path=class_map,
        require_clean_checkout=False,
    )
    assert manifest["status"] == "frozen"
    assert len(manifest["cells"]) == 12
    assert manifest["training_consumes_validation"] is False
    assert manifest["single_seed_claim_allowed"] is False
    assert manifest["assets"]["training_video_count"] == 200
    assert manifest["assets"]["validation_video_count"] == 211
    budget = manifest["budget_semantics"]
    mixed = budget["mixed_k"]
    assert budget["version"] == duca_paper_training.BUDGET_SEMANTICS
    assert budget["execution_quantum"] == 16
    assert budget["padding_or_repetition_allowed"] is False
    assert mixed["candidate_budgets"] == [192, 256, 384, 512]
    assert mixed["schedule_counts"] == [8, 12, 16, 24]
    assert mixed["nominal_requested_mean_k"] == 384.0
    assert mixed["schedule_sha256"] == (
        duca_paper_training.mixed_k_requested_schedule_sha256()
    )


def test_matrix_manifest_binds_validated_real_short_window_gate(tmp_path):
    commit = "a" * 40
    pretrain = tmp_path / "pretrain.pth"
    pretrain.write_bytes(b"fixture-videomae")
    annotation = tmp_path / "annotation.json"
    _write_official_annotation(annotation, include_training=True)
    class_map = tmp_path / "category_idx.txt"
    class_map.write_text(
        "\n".join(f"class_{index} {index}" for index in range(20)) + "\n",
        encoding="utf-8",
    )
    train_data = tmp_path / "train-videos"
    train_data.mkdir()
    code_gate = tmp_path / "code-gate.json"
    _write_code_gate_fixture(code_gate, commit=commit)
    gate = tmp_path / "short-window-gate.json"
    _write_short_window_gate_fixture(
        gate,
        commit=commit,
        pretrain=pretrain,
        annotation=annotation,
        class_map=class_map,
        train_data=train_data,
        code_gate=code_gate,
    )
    binding = short_window_gate.validate_gate_artifact(
        gate,
        expected_commit=commit,
        expected_sha256=_sha256(gate),
    )
    assert binding["status"] == "passed"
    manifest = build_manifest(
        repo_root=REPO_ROOT,
        expected_commit=commit,
        pretrain_path=pretrain,
        annotation_path=annotation,
        class_map_path=class_map,
        short_window_gate_path=gate,
        require_clean_checkout=False,
    )
    frozen = manifest["prerequisite_gates"][
        "real_natural_short_window_heavy_backbone"
    ]
    assert frozen["path"] == str(gate.resolve())
    assert frozen["sha256"] == _sha256(gate)
    assert frozen["performance_evidence"] is False
    frozen_code = manifest["prerequisite_gates"]["clean_linux_pytorch_code"]
    assert frozen_code["path"] == str(code_gate.resolve())
    assert frozen_code["sha256"] == _sha256(code_gate)


def test_mixed_k_epoch_budget_audit_separates_requested_and_realized_k():
    ordered = [f"train_{index:03d}" for index in range(200)]
    schedule = duca_paper_training.mixed_k_requested_schedule()
    rows = []
    for sample_index, video_id in enumerate(ordered):
        requested = schedule[sample_index % len(schedule)]
        effective = min(requested, 224)
        selected = list(range(effective))
        rows.append(
            {
                "schema_version": "duca_paper_committed_budget_row_v1",
                "rank": sample_index % 2,
                "video_id": video_id,
                "window_start_frame": 0,
                "duca_stateless_epoch": 0,
                "duca_stateless_sample_index": sample_index,
                "dense_valid_len": 231,
                "execution_quantum": 16,
                "requested_k": requested,
                "effective_k": effective,
                "unique_k": effective,
                "backbone_input_k": effective,
                "backbone_input_measurement_source": (
                    "actual_backbone_wrapper_and_videomae_input_tensors"
                ),
                "backbone_input_contract_sha256": "c" * 64,
                "padded_k": effective,
                "selected_dense_indices": selected,
                "budget_semantics": duca_paper_training.BUDGET_SEMANTICS,
            }
        )
    epoch = duca_paper_training.build_epoch_budget_audit(
        arm="uniform_mixed_train_k384_eval",
        epoch=0,
        rows=rows,
        ordered_video_ids=ordered,
    )
    assert epoch["row_count"] == 200
    assert epoch["effective_histogram"] == {"192": 24, "224": 176}
    assert epoch["realized_backbone_mean_k"] < 384.0
    summary = duca_paper_training.summarize_budget_epoch_records(
        arm="uniform_mixed_train_k384_eval",
        epoch_records=[{"budget_audit": epoch}],
    )
    assert summary["row_count"] == 200
    assert summary["feasibility_shrink_count"] > 0
    assert summary["observed_requested_mean_k"] != summary[
        "realized_backbone_mean_k"
    ]


def test_exact211_budget_execution_requires_realized_k_vector(tmp_path):
    ledger_root = tmp_path / "ledger"
    ledger_root.mkdir()
    protocol_sha = "a" * 64
    video_ids = [f"validation_{index:03d}" for index in range(211)]
    ledger = ledger_root / "inference_ledger.rank0000.jsonl"
    rows = []
    for video_id in video_ids:
        rows.append(
            {
                "schema_version": "duca_rime_inference_ledger_v1",
                "video_id": video_id,
                "window_start_frame": 0,
                "arm": "exact_uniform",
                "requested_k": 384,
                "effective_k": 224,
                "unique_k": 224,
                "backbone_input_k": 224,
                "backbone_input_measurement_source": (
                    "actual_backbone_wrapper_and_videomae_input_tensors"
                ),
                "backbone_input_contract_sha256": "b" * 64,
                "padded_k": 224,
                "dense_valid_len": 231,
                "selected_dense_indices": list(range(224)),
                "budget_protocol_sha256": protocol_sha,
            }
        )
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = duca_paper_training.validate_evaluation_budget_execution(
        arm="uniform_fixed_k384",
        evaluation_summary={
            "post_processing_execution": {
                "window_counts": {video_id: 1 for video_id in video_ids}
            }
        },
        ledger_root=ledger_root,
        protocol_sha256=protocol_sha,
    )
    assert receipt["window_count"] == 211
    assert receipt["requested_budget_is_dynamic"] is False
    assert len(receipt["window_budget_vector_sha256"]) == 64


def test_formal_manifest_rejects_immutable_failed_stage_a_source(tmp_path):
    with pytest.raises(RuntimeError, match="immutable failed source"):
        build_manifest(
            repo_root=tmp_path,
            expected_commit="2df0103ec1c26ff7cff7ed15f399e78e640df211",
            pretrain_path=tmp_path / "missing-pretrain.pth",
            annotation_path=tmp_path / "missing-annotation.json",
            class_map_path=tmp_path / "missing-class-map.txt",
            short_window_gate_path=tmp_path / "missing-gate.json",
            require_clean_checkout=True,
        )


def test_stage_a_launchers_remain_paper_facing_and_fail_closed():
    code_gate = (REPO_ROOT / "scripts/run_duca_paper_code_gate.sh").read_text(
        encoding="utf-8"
    )
    short_gate = (
        REPO_ROOT / "scripts/run_duca_paper_short_window_gate.sh"
    ).read_text(encoding="utf-8")
    cell = (REPO_ROOT / "scripts/run_duca_paper_stage_a_cell.sh").read_text(
        encoding="utf-8"
    )
    submit = (REPO_ROOT / "scripts/submit_duca_paper_stage_a.sh").read_text(
        encoding="utf-8"
    )
    seal = (REPO_ROOT / "scripts/run_duca_paper_stage_a_seal.sh").read_text(
        encoding="utf-8"
    )
    seed = (REPO_ROOT / "scripts/run_duca_paper_stage_a_seed.sh").read_text(
        encoding="utf-8"
    )
    grouped = (
        REPO_ROOT / "scripts/submit_duca_paper_stage_a_grouped.sh"
    ).read_text(encoding="utf-8")
    assert "--nproc_per_node=2 tools/train.py" in cell
    assert "--nproc_per_node=1 tools/test.py" in cell
    assert "--expected-checkpoint-epoch 59" in cell
    assert "training_consumed_validation" in cell
    assert "DUCA_PAPER_SHORT_WINDOW_GATE_SHA256" in cell
    assert "short_window_gate_sha256" in cell
    assert "validate_duca_paper_code_gate" in cell
    assert "--gres=gpu:2" in submit
    assert "[[ \"${#job_ids[@]}\" == 12 ]]" in submit
    assert "complete_three_seed_matrix" in seal
    assert "metrics_withheld_from_engineering_receipt" in seal
    assert "short_window_gate_sha256" in seal
    assert "validate_duca_paper_code_gate" in seal
    assert "sequential_scheduler_grouping_only" in seed
    assert "DUCA_PAPER_GROUP" in seed
    assert "mixed_k_failure_blocks_duca_arm" in seed
    assert "[[ \"${#job_ids[@]}\" == 6 ]]" in grouped
    assert "active_jobs + 7 <= max_jobs" in grouped
    assert "scheduler_job_count\": 7" in grouped
    assert "logical_cell_count\": 12" in grouped
    assert grouped.count("--hold") == 2
    assert "validate_duca_paper_short_window_gate" in grouped
    assert "validate_duca_paper_code_gate" in grouped
    assert "--short-window-gate" in grouped
    assert "immutable failed Stage-A source cannot be redeployed" in grouped
    assert "submission_manifest.json.receipt.sha256" in grouped
    assert '"short_window_gate_pending": True' in code_gate
    assert '"stage_a_manifest_created": False' in code_gate
    assert "--output \"${DUCA_PAPER_MATRIX_MANIFEST}\"" not in code_gate
    assert "DUCA_PAPER_CODE_GATE_RECEIPT_SHA256" in short_gate
    assert "validate_duca_paper_code_gate" in short_gate
    assert grouped.rfind("trap - ERR INT TERM") > grouped.rfind(
        "os.replace(temporary, target)"
    )
