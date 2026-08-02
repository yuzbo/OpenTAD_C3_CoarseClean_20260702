from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata import duca_paper_training
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


def test_stage_a_launchers_remain_paper_facing_and_fail_closed():
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
    assert "--gres=gpu:2" in submit
    assert "[[ \"${#job_ids[@]}\" == 12 ]]" in submit
    assert "complete_three_seed_matrix" in seal
    assert "metrics_withheld_from_engineering_receipt" in seal
    assert "sequential_scheduler_grouping_only" in seed
    assert "logical_cell_count\": 4" in seed
    assert "[[ \"${#job_ids[@]}\" == 3 ]]" in grouped
    assert "active_jobs + 4 <= max_jobs" in grouped
    assert "scheduler_job_count\": 4" in grouped
    assert "logical_cell_count\": 12" in grouped
    assert grouped.count("--hold") == 2
    assert grouped.rfind("trap - ERR INT TERM") > grouped.rfind(
        "os.replace(temporary, target)"
    )
