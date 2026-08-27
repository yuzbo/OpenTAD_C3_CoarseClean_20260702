from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

from opentad.models.chronotransport.formal_stage_b import (
    calibrate_stage_b_records,
    build_split_manifest,
    compact_stage_b_record,
    save_calibrated_stage_b_checkpoint,
    select_schedule_for_step,
    summarize_stage_b_evaluation,
)
from tools.bata.train_chronotransport_stage_b import run_training
from tools.bata.chronotransport_opentad_factory import filter_dataset_by_video_ids


ROOT = Path(__file__).resolve().parents[1]


class _Runtime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transport = nn.Linear(1, 1, bias=False)
        self.risk_predictor = nn.Linear(1, 1, bias=False)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chronotransport = _Runtime()
        self.heavy = nn.Linear(1, 1, bias=False)


def _batch_factory(epoch: int):
    for index in range(2):
        yield {
            "inputs": torch.tensor([[float(epoch + index + 1)]]),
            "sample_id": f"video_{epoch}_{index}",
        }


def _loss_step(model: nn.Module, batch: dict, step: int) -> dict:
    prediction = model.chronotransport.transport(batch["inputs"])
    risk = model.chronotransport.risk_predictor(batch["inputs"])
    task = prediction.square().mean()
    risk_loss = (risk - 0.25).square().mean()
    return {
        "loss": task + risk_loss,
        "task": task,
        "transport": task.detach() * 0.0,
        "risk": risk_loss,
        "regret": float(step) / 10.0,
        "schedule": select_schedule_for_step(
            step,
            ("periodic2_transport", "periodic4_transport"),
        ),
    }


def _optimizer(parameters):
    return torch.optim.SGD(list(parameters), lr=0.1)


def test_formal_split_manifest_is_deterministic_disjoint_and_hashed() -> None:
    video_ids = [f"video_{index:03d}" for index in range(20)]
    first = build_split_manifest(video_ids, seed=3407, ratios=(0.7, 0.15, 0.15))
    second = build_split_manifest(reversed(video_ids), seed=3407, ratios=(0.7, 0.15, 0.15))
    assert first == second
    assert [len(first["splits"][name]) for name in ("fit", "calibration", "evaluation")] == [14, 3, 3]
    flattened = sum((first["splits"][name] for name in ("fit", "calibration", "evaluation")), [])
    assert sorted(flattened) == sorted(video_ids)
    assert len(set(flattened)) == len(video_ids)
    assert set(first["split_hashes"]) == {"fit", "calibration", "evaluation"}


def test_training_schedule_cycles_from_step_one() -> None:
    candidates = ("p2", "p4", "p8")
    assert [select_schedule_for_step(step, candidates) for step in range(1, 7)] == [
        "p2",
        "p4",
        "p8",
        "p2",
        "p4",
        "p8",
    ]


def test_dataset_filter_keeps_only_manifest_video_ids_and_rejects_unknown() -> None:
    class Dataset:
        data_list = [
            ["video_a", {}, {}],
            ["video_b", {}, {}],
            ["video_c", {}, {}],
        ]

    dataset = Dataset()
    filter_dataset_by_video_ids(dataset, ["video_a", "video_c"])
    assert [row[0] for row in dataset.data_list] == ["video_a", "video_c"]

    dataset = Dataset()
    try:
        filter_dataset_by_video_ids(dataset, ["video_missing"])
    except ValueError as exc:
        assert "not present" in str(exc)
    else:
        raise AssertionError("unknown split ids must be rejected")


def test_training_loop_writes_metrics_periodic_checkpoint_and_real_ema(tmp_path: Path) -> None:
    output = tmp_path / "stage_b.pth"
    metrics = tmp_path / "train.jsonl"
    result = run_training(
        model=_Model(),
        batch_source=_batch_factory,
        loss_step=_loss_step,
        optimizer_factory=_optimizer,
        output=output,
        steps=3,
        metrics_path=metrics,
        checkpoint_interval=2,
        ema_decay=0.5,
        split_hashes={"fit": "fit-hash"},
        seed=3407,
    )
    checkpoint = torch.load(output, map_location="cpu")
    assert result["steps"] == 3
    assert checkpoint["meta"]["ema_semantics"] == "trainable_parameter_ema"
    key = "module.chronotransport.transport.weight"
    assert not torch.equal(checkpoint["state_dict"][key], checkpoint["state_dict_ema"][key])
    assert (tmp_path / "stage_b.step2.pth").is_file()
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert all(row["seed"] == 3407 and row["schedule"] for row in rows)


def test_training_loop_resumes_optimizer_ema_and_global_step(tmp_path: Path) -> None:
    first = tmp_path / "first.pth"
    run_training(
        model=_Model(),
        batch_source=_batch_factory,
        loss_step=_loss_step,
        optimizer_factory=_optimizer,
        output=first,
        steps=3,
        ema_decay=0.5,
        split_hashes={"fit": "fit-hash"},
        seed=3407,
    )
    resumed = tmp_path / "resumed.pth"
    result = run_training(
        model=_Model(),
        batch_source=_batch_factory,
        loss_step=_loss_step,
        optimizer_factory=_optimizer,
        output=resumed,
        steps=4,
        resume=first,
        ema_decay=0.5,
        split_hashes={"fit": "fit-hash"},
        seed=3407,
    )
    checkpoint = torch.load(resumed, map_location="cpu")
    assert result["steps"] == 4
    assert checkpoint["meta"]["resumed_from"] == str(first)
    assert checkpoint["meta"]["start_step"] == 3


def test_formal_evaluation_reports_calibration_correlation_and_paired_gate() -> None:
    records = []
    for index in range(6):
        transport_regret = 0.1 + index * 0.1
        hold_regret = transport_regret + 1.0
        for schedule, regret, feature_mse in (
            ("periodic2_transport", transport_regret, 0.2 + index * 0.01),
            ("periodic2_hold", hold_regret, 1.2 + index * 0.01),
        ):
            records.append(
                {
                    "sample_id": f"video_{index}",
                    "split": "evaluation",
                    "schedule": schedule,
                    "predicted_risk": regret - 0.05,
                    "upper_risk": regret + 0.01,
                    "regret": regret,
                    "feature_mse": feature_mse,
                }
            )
    summary = summarize_stage_b_evaluation(
        records,
        coverage_target=0.9,
        min_spearman=0.2,
        bootstrap_samples=200,
        bootstrap_seed=3407,
    )
    assert summary["status"] == "PASS"
    assert summary["coverage"] == 1.0
    assert summary["risk_regret_spearman"] > 0.9
    assert summary["transport_vs_hold"]["regret_improvement_ci95"][0] > 0.0
    assert summary["transport_vs_hold"]["feature_improvement_ci95"][0] > 0.0


def test_calibration_sets_finite_sample_offset_and_reports_empirical_coverage() -> None:
    records = [
        {"predicted_risk": 0.1, "regret": 0.2},
        {"predicted_risk": 0.2, "regret": 0.5},
        {"predicted_risk": 0.4, "regret": 0.45},
    ]
    result = calibrate_stage_b_records(records, coverage=0.75)
    assert result["offset"] == 0.3
    assert result["coverage"] >= 0.75
    assert result["records"] == 3


def test_formal_record_is_compact_and_contains_no_predictions_or_features() -> None:
    record = compact_stage_b_record(
        sample_id="video_1",
        split="evaluation",
        schedule="periodic2_transport",
        predicted_risk=0.2,
        upper_risk=0.3,
        regret=0.25,
        feature_mse=0.01,
        dense_loss=1.0,
        counterfactual_loss=1.25,
        cost={"recompute_rows": 72, "transport_rows": 72, "hold_rows": 0},
    )
    assert set(record) == {
        "sample_id",
        "split",
        "schedule",
        "signals",
        "pooled_targets",
        "cost",
        "regret",
    }
    assert "predictions" not in json.dumps(record)
    assert "full_features" not in json.dumps(record)


def test_calibrated_checkpoint_updates_both_states_but_keeps_claims_locked(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trained.pth"
    raw = {
        "module.backbone.chronotransport.risk_predictor.calibration_offset": torch.tensor(0.0),
        "module.backbone.chronotransport.transport.weight": torch.tensor([1.0]),
    }
    ema = {name: value.clone() for name, value in raw.items()}
    torch.save(
        {
            "epoch": 5,
            "state_dict": raw,
            "state_dict_ema": ema,
            "optimizer": {"state": {}},
            "meta": {"calibration_ready": False},
        },
        source,
    )
    output = tmp_path / "calibrated.pth"
    save_calibrated_stage_b_checkpoint(
        source,
        output,
        calibration_offset=0.25,
        split_hashes={"fit": "f", "calibration": "c", "evaluation": "e"},
        p3_gate_status="PASS",
    )
    checkpoint = torch.load(output, map_location="cpu")
    for state_key in ("state_dict", "state_dict_ema"):
        offsets = [
            value
            for name, value in checkpoint[state_key].items()
            if name.endswith("risk_predictor.calibration_offset")
        ]
        assert len(offsets) == 1 and float(offsets[0]) == 0.25
    assert checkpoint["meta"]["calibration_ready"] is True
    assert checkpoint["meta"]["measured_cost_ready"] is False
    assert checkpoint["meta"]["p3_gate_status"] == "PASS"
    assert checkpoint["meta"]["deploy_claim_allowed"] is False
    assert checkpoint["meta"]["metric_claim_allowed"] is False
    assert checkpoint["meta"]["latency_claim_allowed"] is False
    assert checkpoint["meta"]["paper_claim_allowed"] is False


def test_formal_gpu1_launcher_is_guarded_and_keeps_claims_locked() -> None:
    launcher = (ROOT / "scripts/run_chronotransport_stage_b_formal_gpu1.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]]' in launcher
    assert "SLURM_JOB_ID" in launcher
    assert "CHRONOTRANSPORT_STAGE_B_SEED" in launcher
    assert "CHRONOTRANSPORT_SPLIT_MANIFEST" in launcher
    assert "run_chronotransport_stage_b_formal.py" in launcher
    assert "CHRONOTRANSPORT_RISK_READY=1" not in launcher


def test_formal_stage_b_runner_is_directly_executable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/bata/run_chronotransport_stage_b_formal.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--seed" in result.stdout
    assert "--epochs" in result.stdout
