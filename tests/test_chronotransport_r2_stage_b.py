from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import opentad.models.chronotransport.formal_stage_b as formal_stage_b_module

from opentad.models.chronotransport.environment import (
    REQUIRED_ENVIRONMENT_SCHEMA,
    build_test_only_observed_environment,
    observed_environment_to_provenance,
)
from opentad.models.chronotransport.formal_stage_b import (
    _atomic_torch_save,
    _atomic_write_ledger,
    _state_dict_sha256,
    build_r2_stage_b_phase_completion_marker,
    build_fit_schedule_constant_artifact,
    FormalStageBInvalid,
    logical_risk_predictor_state_sha256,
    StageBReplayOutput,
    StageBUpdateState,
    run_r2_stage_b_training,
    validate_r2_stage_b_phase_completion_marker,
)
from opentad.models.chronotransport.losses import compose_r2_stage_b_loss
from opentad.models.chronotransport.protocol import build_stage_b_exposure_artifact
from opentad.models.chronotransport.protocol import canonical_json_bytes, canonical_sha256
from opentad.models.chronotransport.replay import (
    paired_detector_losses,
    validate_candidate_order_invariance,
)
from opentad.models.chronotransport.scheduler import R2_NON_DENSE_NAMES
import tools.bata.train_chronotransport_r2_stage_b as stage_b_cli
from tools.bata.train_chronotransport_r2_stage_b import build_parser
from tools.bata.chronotransport_r2_stage_b_factory import (
    _validate_registered_stage_b_inputs,
    sealed_stage_b_replay,
)


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def test_repository_stage_b_factory_reads_nested_registered_artifacts() -> None:
    manifest = {"manifest_sha256": SHA_A}
    exposure = {"artifact_sha256": SHA_B}
    registration = {
        "window_manifest": {"artifact": dict(manifest)},
        "exposures": {"stage_b": dict(exposure)},
    }

    _validate_registered_stage_b_inputs(
        registration=registration,
        manifest=manifest,
        exposure_artifact=exposure,
    )

    stale_flat_registration = {
        "window_manifest": {"manifest_sha256": SHA_A},
        "exposures": {"stage_b_sha256": SHA_B},
    }
    with pytest.raises(ValueError, match="nested registered Stage-B artifacts"):
        _validate_registered_stage_b_inputs(
            registration=stale_flat_registration,
            manifest=manifest,
            exposure_artifact=exposure,
        )


class ChronoTransportRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forced_schedule = None
        self.latest_output = None


class _StatefulTransform:
    def __init__(self) -> None:
        self.calls = 0

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = int(state["calls"])


class _RNGDetector(nn.Module):
    def __init__(self, transform: _StatefulTransform) -> None:
        super().__init__()
        self.chronotransport = ChronoTransportRuntime()
        self.weight = nn.Parameter(torch.ones(()))
        self.transform = transform
        self.branch_draws: list[tuple[str, float, float, float, int]] = []

    def forward(self, inputs):
        self.transform.calls += 1
        draws = (
            str(self.chronotransport.forced_schedule),
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            self.transform.calls,
        )
        self.branch_draws.append(draws)
        feature = inputs * self.weight
        self.chronotransport.latest_output = feature
        return {"loss_cls": feature.square().mean()}


class _FailingRNGDetector(_RNGDetector):
    def forward(self, inputs):
        result = super().forward(inputs)
        if self.chronotransport.forced_schedule != "dense":
            raise RuntimeError("candidate failure")
        return result


def test_paired_replay_restores_all_rng_and_transform_state_and_binds_batch_hash() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    transform = _StatefulTransform()
    detector = _RNGDetector(transform)
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = paired_detector_losses(
        detector,
        {"inputs": inputs},
        counterfactual_schedule="periodic2_transport",
        stateful_objects={"data_transform": transform},
        augmentation_sha256=SHA_A,
    )

    dense, candidate = detector.branch_draws
    assert dense[1:] == candidate[1:]
    assert transform.calls == 0
    assert result.materialized_window_sha256 == result.counterfactual_window_sha256
    assert result.augmentation_sha256 == SHA_A
    assert torch.equal(inputs, torch.arange(6, dtype=torch.float32).reshape(2, 3))


def test_paired_replay_restores_rng_and_transform_when_a_branch_raises() -> None:
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    transform = _StatefulTransform()
    detector = _FailingRNGDetector(transform)
    with pytest.raises(RuntimeError, match="candidate failure"):
        paired_detector_losses(
            detector,
            {"inputs": torch.ones(1, 1)},
            counterfactual_schedule="periodic2_transport",
            stateful_objects={"data_transform": transform},
            augmentation_sha256=SHA_A,
        )
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected
    assert transform.calls == 0


def test_candidate_order_regression_fails_closed_when_regret_changes() -> None:
    canonical = {name: float(index) for index, name in enumerate(R2_NON_DENSE_NAMES)}
    permuted = dict(reversed(tuple(canonical.items())))
    validate_candidate_order_invariance(canonical, permuted)
    permuted["periodic4_transport"] += 1e-4
    with pytest.raises(RuntimeError, match="candidate-order-dependent"):
        validate_candidate_order_invariance(canonical, permuted)


def test_r2_stage_b_loss_is_exact_detector_plus_mse_plus_pinball() -> None:
    counterfactual = torch.tensor(2.0, requires_grad=True)
    transported = torch.tensor([1.0, 3.0], requires_grad=True)
    dense = torch.tensor([0.0, 1.0], requires_grad=True)
    prediction = torch.tensor([0.2], requires_grad=True)
    regret = torch.tensor([0.7], requires_grad=True)
    losses = compose_r2_stage_b_loss(
        counterfactual_task_loss=counterfactual,
        counterfactual_features=transported,
        dense_features=dense,
        predicted_quantile=prediction,
        regret_target=regret,
    )
    assert float(losses.detector.detach()) == pytest.approx(2.0)
    assert float(losses.transport.detach()) == pytest.approx(2.5)
    assert float(losses.risk.detach()) == pytest.approx(0.45)
    assert float(losses.total.detach()) == pytest.approx(2.0 + 0.1 * 2.5 + 0.1 * 0.45)
    losses.total.backward()
    assert dense.grad is None
    assert regret.grad is None


def test_success_cursor_uses_successful_update_and_never_advances_on_skip_or_retry() -> None:
    windows = [f"window-{index:03d}" for index in range(140)]
    artifact = build_stage_b_exposure_artifact(windows)
    state = StageBUpdateState(
        seed=3407,
        exposure_artifact=artifact,
        candidate_names=R2_NON_DENSE_NAMES,
    )
    first = state.current()
    state.record_retry("infrastructure")
    assert state.current() == first
    state.record_skip("nonfinite")
    assert state.current() == first
    state.record_success(ledger_row_sha256=SHA_A)
    assert state.current()["successful_update"] == 1
    assert state.current()["candidate"] != first["candidate"]
    assert state.counters == {
        "attempted_optimizer_updates": 2,
        "successful_optimizer_updates": 1,
        "nonfinite_or_skipped_updates": 1,
        "amp_skips": 0,
        "ema_updates": 1,
        "lr_scheduler_updates": 1,
        "infrastructure_resume_events": 1,
    }


@pytest.mark.parametrize("seed,cursor", [(3407.0, 0), (True, 0), (3407, 0.0), (3407, False)])
def test_success_cursor_rejects_numeric_impersonators(seed, cursor) -> None:
    windows = [f"window-{index:03d}" for index in range(140)]
    artifact = build_stage_b_exposure_artifact(windows)
    with pytest.raises(ValueError, match="integer|seed"):
        StageBUpdateState(
            seed=seed,
            exposure_artifact=artifact,
            candidate_names=R2_NON_DENSE_NAMES,
            successful_update_cursor=cursor,
        )


class _RiskScheduler(nn.Module):
    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor


class _ToyRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transport = nn.Linear(1, 1, bias=False)
        self.risk_predictor = nn.Linear(1, 1, bias=False)
        self.scheduler = _RiskScheduler(self.risk_predictor)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chronotransport = _ToyRuntime()
        self.frozen_detector = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.chronotransport.transport.weight.fill_(0.5)
            self.chronotransport.risk_predictor.weight.fill_(0.25)
            self.frozen_detector.weight.fill_(2.0)


class _BufferedToyModel(_ToyModel):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("frozen_running_state", torch.tensor([3.0]))


class _SealedRisk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, signals, actions):
        batch, candidates = int(actions.shape[0]), int(actions.shape[1])
        pooled = signals.mean(dim=(1, 2, 3)).reshape(batch, 1)
        return pooled.expand(batch, candidates) * self.scale


class _SealedRuntime(nn.Module):
    def __init__(self, action_by_schedule) -> None:
        super().__init__()
        self.forced_schedule = None
        self.action_by_schedule = action_by_schedule
        self.risk_predictor = _SealedRisk()
        self.latest_output = None
        self.latest_schedule = None
        self.latest_summary = None
        self.latest_signals = None


_SealedRuntime.__name__ = "ChronoTransportRuntime"


class _SealedDetector(nn.Module):
    def __init__(self, action_by_schedule) -> None:
        super().__init__()
        self.chronotransport = _SealedRuntime(action_by_schedule)
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, inputs, return_loss=True):
        assert return_loss is True
        schedule = str(self.chronotransport.forced_schedule)
        actions = self.chronotransport.action_by_schedule[schedule].unsqueeze(0)
        feature = inputs * self.weight + actions.float().mean() / 10.0
        signals = inputs.reshape(1, 1, 1, 1).expand(1, 48, 3, 6)
        self.chronotransport.latest_output = feature
        self.chronotransport.latest_schedule = type(
            "ExecutedSchedule", (), {"actions": actions}
        )()
        action_sha256 = canonical_sha256(actions[0].tolist())
        self.chronotransport.latest_summary = {
            "executed_schedule_name": schedule,
            "selected_schedule_names": [schedule],
            "evidence_valid": True,
            "requested_action_sha256": action_sha256,
            "executed_action_sha256": action_sha256,
        }
        self.chronotransport.latest_signals = signals
        return {"loss_cls": feature.square().mean()}


def _sealed_registration_and_actions():
    actions = {"dense": torch.zeros(48, 3, dtype=torch.long)}
    candidates = []
    for index, name in enumerate(R2_NON_DENSE_NAMES):
        action = torch.full((48, 3), index % 3, dtype=torch.long)
        action[0, 0] = (index // 3) % 3
        actions[name] = action
        candidates.append(
            {"name": name, "action_sha256": canonical_sha256(action.tolist())}
        )
    return {"candidate_library": {"candidates": candidates}}, actions


def test_sealed_repository_replay_binds_actual_runtime_action_and_rejects_tamper() -> None:
    registration, actions = _sealed_registration_and_actions()
    model = _SealedDetector(actions)
    schedule = R2_NON_DENSE_NAMES[7]
    output = sealed_stage_b_replay(
        model,
        {"inputs": torch.ones(1, 1)},
        schedule,
        registration=registration,
        track_grad=True,
    )
    expected = canonical_sha256(actions[schedule].tolist())
    assert output.requested_action_sha256 == output.executed_action_sha256 == expected
    assert output.counterfactual_task_loss.requires_grad is True
    registration["candidate_library"]["candidates"][7]["action_sha256"] = SHA_A
    with pytest.raises(RuntimeError, match="executed action tensor"):
        sealed_stage_b_replay(
            model,
            {"inputs": torch.ones(1, 1)},
            schedule,
            registration=registration,
            track_grad=False,
        )


def _fit_windows():
    return [f"fit/window-{index:03d}" for index in range(140)]


def _candidate_action_hashes(value: str = SHA_C):
    return {name: value for name in R2_NON_DENSE_NAMES}


def _required_environment():
    required_environment = {
        "schema": REQUIRED_ENVIRONMENT_SCHEMA,
        "gpu_model": "NVIDIA A100-SXM4-80GB",
        "driver": "535.54",
        "cuda": "11.8",
        "pytorch": "2.0.1",
        "cudnn": "8902",
        "precision": "amp_fp16",
        "batch_size": 1,
    }
    required_environment["environment_sha256"] = canonical_sha256(
        required_environment
    )
    return required_environment


def _registered_provenance():
    required_environment = _required_environment()
    observed = build_test_only_observed_environment(required_environment)
    return {
        "registration_sha256": SHA_A,
        "registration_commit": "4" * 40,
        "spec_commit": "1" * 40,
        "spec_sha256": SHA_B,
        "implementation_commit": "2" * 40,
        "source_files_sha256": SHA_C,
        "upstream_commits_sha256": "d" * 64,
        "split_hashes_sha256": "e" * 64,
        "action_library_sha256": "f" * 64,
        "environment_sha256": required_environment["environment_sha256"],
        **observed_environment_to_provenance(
            observed, required_environment=required_environment
        ),
        "cost_plan_sha256": "1" * 64,
        "gate1_unlock_payload_sha256": "2" * 64,
        "gate1_unlock_file_sha256": "3" * 64,
        "gate1_status": "PASS",
    }


def _batches():
    return [
        {
            "video_id": f"video-{index:03d}",
            "window_id": window_id,
            "inputs": torch.tensor([[1.0 + index / 140.0]]),
            "augmentation_sha256": hashlib.sha256(
                f"augmentation-{index}".encode("utf-8")
            ).hexdigest(),
        }
        for index, window_id in enumerate(_fit_windows())
    ]


def _replay_step(model: _ToyModel, batch: dict, schedule: str) -> StageBReplayOutput:
    del schedule
    transported = model.chronotransport.transport(batch["inputs"])
    dense = model.frozen_detector(batch["inputs"]).detach()
    predicted = model.chronotransport.risk_predictor(batch["inputs"]).reshape(-1)
    counterfactual_task = (transported - 0.75).square().mean()
    regret = (transported.detach() - dense).abs().reshape(-1)
    materialized = hashlib.sha256(batch["inputs"].numpy().tobytes()).hexdigest()
    return StageBReplayOutput(
        counterfactual_task_loss=counterfactual_task,
        counterfactual_features=transported,
        dense_features=dense,
        predicted_quantile=predicted,
        regret_target=regret,
        materialized_window_sha256=materialized,
        counterfactual_window_sha256=materialized,
        augmentation_sha256=batch["augmentation_sha256"],
        requested_action_sha256=SHA_C,
        executed_action_sha256=SHA_C,
        amp_skipped=False,
    )


def _run(
    tmp_path: Path,
    *,
    seed: int,
    model: nn.Module | None = None,
    output_name: str = "stage_b.pth",
    resume_from: Path | None = None,
    stop_after_successful: int | None = None,
    replay_step=_replay_step,
    checkpoint_frequency: int = 0,
):
    active_model = _ToyModel() if model is None else model
    dense_checkpoint = tmp_path / "dense.pth"
    if not dense_checkpoint.exists():
        state = {
            name: value.detach().clone() for name, value in active_model.state_dict().items()
        }
        torch.save(
            {
                "state_dict": state,
                "state_dict_ema": {name: value.clone() for name, value in state.items()},
                "meta": {"upstream": "registered-toy"},
            },
            dense_checkpoint,
        )
    dense_sha = hashlib.sha256(dense_checkpoint.read_bytes()).hexdigest()
    return run_r2_stage_b_training(
        model=active_model,
        batches=_batches(),
        replay_step=replay_step,
        seed=seed,
        exposure_artifact=build_stage_b_exposure_artifact(_fit_windows()),
        dense_checkpoint_path=dense_checkpoint,
        dense_checkpoint_sha256=dense_sha,
        dense_checkpoint_use_ema=False,
        manifest_sha256=SHA_A,
        library_sha256=SHA_B,
        config_sha256=SHA_C,
        output_checkpoint=tmp_path / output_name,
        ledger_path=tmp_path / output_name.replace(".pth", ".jsonl"),
        registered_provenance=_registered_provenance(),
        resume_from=resume_from,
        stop_after_successful=stop_after_successful,
        checkpoint_frequency=checkpoint_frequency,
    )


def test_formal_stage_b_rejects_arbitrary_bytes_as_dense_checkpoint(tmp_path: Path) -> None:
    bad = tmp_path / "arbitrary.pth"
    bad.write_bytes(b"not a torch checkpoint")
    with pytest.raises((ValueError, RuntimeError, OSError, EOFError), match="checkpoint|pickle|invalid"):
        run_r2_stage_b_training(
            model=_ToyModel(),
            batches=_batches(),
            replay_step=_replay_step,
            seed=3407,
            exposure_artifact=build_stage_b_exposure_artifact(_fit_windows()),
            dense_checkpoint_path=bad,
            dense_checkpoint_sha256=hashlib.sha256(bad.read_bytes()).hexdigest(),
            dense_checkpoint_use_ema=False,
            manifest_sha256=SHA_A,
            library_sha256=SHA_B,
            config_sha256=SHA_C,
            output_checkpoint=tmp_path / "out.pth",
            ledger_path=tmp_path / "out.jsonl",
            registered_provenance=_registered_provenance(),
        )


def test_dense_checkpoint_rejects_conflicting_predictor_alias_tensors(tmp_path: Path) -> None:
    model = _ToyModel()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    state["chronotransport.scheduler.predictor.weight"] = torch.full_like(
        state["chronotransport.scheduler.predictor.weight"], 99.0
    )
    checkpoint = tmp_path / "alias-conflict.pth"
    torch.save({"state_dict": state}, checkpoint)
    with pytest.raises(ValueError, match="alias"):
        run_r2_stage_b_training(
            model=model,
            batches=_batches(),
            replay_step=_replay_step,
            seed=3407,
            exposure_artifact=build_stage_b_exposure_artifact(_fit_windows()),
            dense_checkpoint_path=checkpoint,
            dense_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            dense_checkpoint_use_ema=False,
            manifest_sha256=SHA_A,
            library_sha256=SHA_B,
            config_sha256=SHA_C,
            output_checkpoint=tmp_path / "out.pth",
            ledger_path=tmp_path / "out.jsonl",
            registered_provenance=_registered_provenance(),
        )


@pytest.mark.parametrize("seed", [3407, 3408, 3409])
def test_formal_stage_b_completes_exact_140_successes_and_frozen_exposures(
    tmp_path: Path, seed: int
) -> None:
    result = _run(tmp_path, seed=seed, output_name=f"seed-{seed}.pth")
    assert result["status"] == "TRAINING_COMPLETE_BASELINE_PENDING"
    assert result["counters"] == {
        "attempted_optimizer_updates": 140,
        "successful_optimizer_updates": 140,
        "nonfinite_or_skipped_updates": 0,
        "amp_skips": 0,
        "ema_updates": 140,
        "lr_scheduler_updates": 140,
        "infrastructure_resume_events": 0,
    }
    rows = [
        json.loads(line)
        for line in Path(result["ledger_path"]).read_text(encoding="utf-8").splitlines()
    ]
    expected = build_stage_b_exposure_artifact(_fit_windows())["matrices"][str(seed)]
    assert len(rows) == 140
    assert [row["successful_update"] for row in rows] == list(range(140))
    assert [row["candidate_index"] for row in rows] == [row["candidate"] for row in expected]
    assert [row["window_id"] for row in rows] == _fit_windows()
    assert all(row["dense_checkpoint_sha256"] == result["dense_checkpoint_sha256"] for row in rows)
    assert all(row["materialized_window_sha256"] == row["counterfactual_window_sha256"] for row in rows)


def test_formal_stage_b_resume_reproduces_exact_cursor_ledger_digest_and_weights(
    tmp_path: Path,
) -> None:
    uninterrupted = _run(tmp_path, seed=3407, output_name="uninterrupted.pth")
    partial = _run(
        tmp_path,
        seed=3407,
        output_name="partial.pth",
        stop_after_successful=19,
    )
    assert partial["status"] == "INCOMPLETE_INFRASTRUCTURE_CHECKPOINT"
    resumed = _run(
        tmp_path,
        seed=3407,
        output_name="resumed.pth",
        resume_from=Path(partial["checkpoint_path"]),
    )
    assert resumed["status"] == "TRAINING_COMPLETE_BASELINE_PENDING"
    assert resumed["ledger_sha256"] == uninterrupted["ledger_sha256"]
    left = torch.load(uninterrupted["checkpoint_path"], map_location="cpu")
    right = torch.load(resumed["checkpoint_path"], map_location="cpu")
    assert left["meta"]["successful_update_cursor"] == right["meta"]["successful_update_cursor"] == 140
    assert left["meta"]["risk_predictor_path"] == "chronotransport.risk_predictor"
    assert left["meta"]["scheduler_predictor_alias"] is True
    for name in left["state_dict"]:
        assert torch.equal(left["state_dict"][name], right["state_dict"][name]), name


def test_formal_stage_b_writes_atomic_safe_prefixes_and_verifies_external_ledger_on_resume(
    tmp_path: Path,
) -> None:
    partial = _run(
        tmp_path,
        seed=3407,
        output_name="safe.pth",
        stop_after_successful=3,
        checkpoint_frequency=1,
    )
    for cursor in (1, 2, 3):
        assert (tmp_path / f"safe.step{cursor}.pth").is_file()
        assert (tmp_path / f"safe.step{cursor}.jsonl").is_file()
    ledger_path = Path(partial["ledger_path"])
    ledger_path.write_text(ledger_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="external ledger"):
        _run(
            tmp_path,
            seed=3407,
            output_name="safe-resumed.pth",
            resume_from=Path(partial["checkpoint_path"]),
        )


@pytest.mark.parametrize(
    ("checkpoint_frequency", "stop_after_successful", "ledger_name"),
    [
        (0, None, "interrupted-final.jsonl"),
        (1, 1, "interrupted-prefix.step1.jsonl"),
    ],
)
def test_formal_stage_b_recovers_exact_dangling_ledger_after_checkpoint_interrupt(
    tmp_path: Path,
    monkeypatch,
    checkpoint_frequency: int,
    stop_after_successful: int | None,
    ledger_name: str,
) -> None:
    output_name = (
        "interrupted-final.pth"
        if checkpoint_frequency == 0
        else "interrupted-prefix.pth"
    )
    real_save = formal_stage_b_module._atomic_torch_save

    def interrupt_checkpoint(payload, output):
        del payload, output
        raise OSError("simulated interruption after ledger publication")

    monkeypatch.setattr(
        formal_stage_b_module, "_atomic_torch_save", interrupt_checkpoint
    )
    with pytest.raises(OSError, match="after ledger publication"):
        _run(
            tmp_path,
            seed=3407,
            output_name=output_name,
            stop_after_successful=stop_after_successful,
            checkpoint_frequency=checkpoint_frequency,
        )

    dangling_ledger = tmp_path / ledger_name
    original_bytes = dangling_ledger.read_bytes()
    original_identity = dangling_ledger.stat().st_ino
    monkeypatch.setattr(formal_stage_b_module, "_atomic_torch_save", real_save)

    recovered = _run(
        tmp_path,
        seed=3407,
        output_name=output_name,
        stop_after_successful=stop_after_successful,
        checkpoint_frequency=checkpoint_frequency,
    )

    expected_status = (
        "TRAINING_COMPLETE_BASELINE_PENDING"
        if stop_after_successful is None
        else "INCOMPLETE_INFRASTRUCTURE_CHECKPOINT"
    )
    assert recovered["status"] == expected_status
    assert dangling_ledger.read_bytes() == original_bytes
    assert dangling_ledger.stat().st_ino == original_identity


@pytest.mark.parametrize("mutation", ["parameter", "buffer"])
def test_formal_stage_b_validates_registered_dense_frozen_state_before_prefix_publish(
    tmp_path: Path, mutation: str
) -> None:
    model = _BufferedToyModel()

    def mutate_frozen_state(model, batch, schedule):
        output = _replay_step(model, batch, schedule)
        with torch.no_grad():
            if mutation == "parameter":
                model.frozen_detector.weight.add_(1.0)
            else:
                model.frozen_running_state.add_(1.0)
        return output

    with pytest.raises(FormalStageBInvalid, match="frozen|registered dense"):
        _run(
            tmp_path,
            seed=3407,
            model=model,
            output_name="frozen-fail.pth",
            replay_step=mutate_frozen_state,
            stop_after_successful=1,
            checkpoint_frequency=1,
        )
    assert not (tmp_path / "frozen-fail.step1.pth").exists()
    assert not (tmp_path / "frozen-fail.step1.jsonl").exists()


@pytest.mark.parametrize(
    "frozen_name",
    ["frozen_detector.weight", "frozen_running_state"],
)
def test_formal_stage_b_resume_rejects_frozen_state_that_differs_from_registered_dense(
    tmp_path: Path, frozen_name: str
) -> None:
    partial = _run(
        tmp_path,
        seed=3407,
        model=_BufferedToyModel(),
        output_name="registered-dense-prefix.pth",
        stop_after_successful=3,
    )
    checkpoint = torch.load(partial["checkpoint_path"], map_location="cpu")
    checkpoint["state_dict"][frozen_name].add_(1.0)
    checkpoint["state_dict_ema"][frozen_name].add_(1.0)
    checkpoint["meta"]["state_dict_sha256"] = _state_dict_sha256(
        checkpoint["state_dict"]
    )
    checkpoint["meta"]["state_dict_ema_sha256"] = _state_dict_sha256(
        checkpoint["state_dict_ema"]
    )
    tampered = tmp_path / f"resume-frozen-{frozen_name.replace('.', '-')}.pth"
    torch.save(checkpoint, tampered)
    tampered.with_suffix(".jsonl").write_bytes(
        Path(partial["ledger_path"]).read_bytes()
    )

    with pytest.raises(ValueError, match="frozen|registered dense"):
        _run(
            tmp_path,
            seed=3407,
            model=_BufferedToyModel(),
            output_name="resume-frozen-output.pth",
            resume_from=tampered,
            stop_after_successful=4,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "float_cursor",
        "extra_meta",
        "row_digest",
        "optimizer_lr",
        "scheduler_step",
        "optimizer_stale_step",
        "state_alias_conflict",
        "state_nan",
        "ema_nan",
        "ema_state_dict_divergence",
        "rng_structure",
    ],
)
def test_formal_stage_b_resume_rejects_noncanonical_metadata_and_ledger(
    tmp_path: Path, tamper: str
) -> None:
    partial = _run(
        tmp_path,
        seed=3407,
        output_name="partial-strict.pth",
        stop_after_successful=3,
    )
    checkpoint = torch.load(partial["checkpoint_path"], map_location="cpu")
    if tamper == "float_cursor":
        checkpoint["meta"]["successful_update_cursor"] = 3.0
    elif tamper == "extra_meta":
        checkpoint["meta"]["unregistered"] = True
    elif tamper == "row_digest":
        checkpoint["ledger_rows"][0]["row_sha256"] = SHA_C
        payload = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in checkpoint["ledger_rows"]
        )
        checkpoint["meta"]["ledger_sha256"] = hashlib.sha256(payload).hexdigest()
    elif tamper == "optimizer_lr":
        checkpoint["optimizer"]["param_groups"][0]["lr"] = 9e-4
    elif tamper == "scheduler_step":
        checkpoint["lr_scheduler"]["last_epoch"] = 99
    elif tamper == "optimizer_stale_step":
        for parameter_state in checkpoint["optimizer"]["state"].values():
            parameter_state["step"] = torch.as_tensor(2.0)
    elif tamper == "state_alias_conflict":
        checkpoint["state_dict"][
            "chronotransport.scheduler.predictor.weight"
        ].add_(1.0)
        checkpoint["meta"]["state_dict_sha256"] = _state_dict_sha256(
            checkpoint["state_dict"]
        )
    elif tamper == "state_nan":
        checkpoint["state_dict"]["chronotransport.transport.weight"].fill_(
            float("nan")
        )
        checkpoint["meta"]["state_dict_sha256"] = _state_dict_sha256(
            checkpoint["state_dict"]
        )
    elif tamper == "ema_nan":
        first = next(iter(checkpoint["ema"]))
        checkpoint["ema"][first].fill_(float("nan"))
    elif tamper == "ema_state_dict_divergence":
        first = next(iter(checkpoint["ema"]))
        checkpoint["ema"][first].add_(1.0)
    else:
        checkpoint["rng_state"].pop("numpy")
    tampered = tmp_path / f"tampered-{tamper}.pth"
    torch.save(checkpoint, tampered)
    tampered.with_suffix(".jsonl").write_bytes(
        Path(partial["ledger_path"]).read_bytes()
    )
    with pytest.raises(
        ValueError,
        match="metadata|cursor|ledger row|optimizer|scheduler|state|alias|EMA|RNG",
    ):
        _run(
            tmp_path,
            seed=3407,
            output_name=f"resume-{tamper}.pth",
            resume_from=tampered,
        )


@pytest.mark.parametrize("failure", ["nonfinite", "amp_skip"])
def test_formal_stage_b_fails_closed_without_advancing_on_nonfinite_or_amp_skip(
    tmp_path: Path, failure: str
) -> None:
    def bad_step(model, batch, schedule):
        output = _replay_step(model, batch, schedule)
        if failure == "nonfinite":
            output.counterfactual_task_loss = torch.tensor(float("nan"), requires_grad=True)
        else:
            output.amp_skipped = True
        return output

    with pytest.raises(FormalStageBInvalid, match="INVALID_IMPLEMENTATION") as exc:
        _run(tmp_path, seed=3407, replay_step=bad_step)
    assert exc.value.counters["successful_optimizer_updates"] == 0
    assert exc.value.successful_update_cursor == 0


def test_formal_stage_b_rejects_non_fp32_forward_outputs(tmp_path: Path) -> None:
    def half_step(model, batch, schedule):
        output = _replay_step(model, batch, schedule)
        output.counterfactual_features = output.counterfactual_features.half()
        output.dense_features = output.dense_features.half()
        return output

    with pytest.raises(FormalStageBInvalid, match="FP32"):
        _run(tmp_path, seed=3407, replay_step=half_step)


def test_fit_only_constant_baseline_uses_all_140_targets_per_schedule_and_rank127() -> None:
    rows = []
    for window in range(140):
        for candidate, schedule in enumerate(R2_NON_DENSE_NAMES):
            rows.append(
                {
                    "seed": 3407,
                    "window_id": f"fit/window-{window:03d}",
                    "candidate_index": candidate,
                    "schedule": schedule,
                    "regret": float(window + candidate / 100.0),
                    "materialized_window_sha256": SHA_A,
                    "augmentation_sha256": SHA_B,
                    "requested_action_sha256": SHA_C,
                    "executed_action_sha256": SHA_C,
                }
            )
    artifact = build_fit_schedule_constant_artifact(
        rows,
        seed=3407,
        fit_window_ids=_fit_windows(),
        candidate_action_sha256_by_name=_candidate_action_hashes(),
        provenance={
            "registration_sha256": SHA_A,
            "manifest_sha256": SHA_B,
            "library_sha256": SHA_C,
            "trained_checkpoint_sha256": "d" * 64,
            "predictor_state_sha256": "e" * 64,
        },
    )
    assert len(artifact["rows"]) == 140 * 16
    assert artifact["schedule_constants"]["periodic2_transport"] == 126.0
    assert artifact["schedule_constants"]["periodic2_hold"] == pytest.approx(126.01)
    assert len(artifact["artifact_sha256"]) == 64


def _fit_baseline_for_checkpoint(result):
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
    predictor_state_sha256 = logical_risk_predictor_state_sha256(
        _ToyModel(), checkpoint["state_dict_ema"]
    )
    rows = []
    for window in range(140):
        for candidate, schedule in enumerate(R2_NON_DENSE_NAMES):
            rows.append(
                {
                    "seed": 3407,
                    "window_id": f"fit/window-{window:03d}",
                    "candidate_index": candidate,
                    "schedule": schedule,
                    "regret": float(window + candidate / 100.0),
                    "materialized_window_sha256": SHA_A,
                    "augmentation_sha256": SHA_B,
                    "requested_action_sha256": SHA_C,
                    "executed_action_sha256": SHA_C,
                }
            )
    return build_fit_schedule_constant_artifact(
        rows,
        seed=3407,
        fit_window_ids=_fit_windows(),
        candidate_action_sha256_by_name=_candidate_action_hashes(),
        provenance={
            "registration_sha256": SHA_A,
            "manifest_sha256": SHA_A,
            "library_sha256": SHA_B,
            "trained_checkpoint_sha256": result["checkpoint_sha256"],
            "predictor_state_sha256": predictor_state_sha256,
        },
    )


def test_phase_complete_requires_atomic_marker_and_revalidates_all_bound_artifacts(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, seed=3407, output_name="phase.pth")
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
    assert checkpoint["meta"]["status"] == "TRAINING_COMPLETE_BASELINE_PENDING"
    baseline = _fit_baseline_for_checkpoint(result)
    baseline_path = tmp_path / "fit-baseline.json"
    baseline_path.write_bytes(canonical_json_bytes(baseline) + b"\n")
    marker_path = tmp_path / "phase-complete.json"
    dense_checkpoint_path = tmp_path / "dense.pth"
    common = {
        "registration_sha256": SHA_A,
        "registration_commit": "4" * 40,
        "seed": 3407,
        "model": _ToyModel(),
        "batches": _batches(),
        "exposure_artifact": build_stage_b_exposure_artifact(_fit_windows()),
        "dense_checkpoint_path": dense_checkpoint_path,
        "dense_checkpoint_sha256": hashlib.sha256(
            dense_checkpoint_path.read_bytes()
        ).hexdigest(),
        "dense_checkpoint_use_ema": False,
        "registered_provenance": _registered_provenance(),
        "checkpoint_path": Path(result["checkpoint_path"]),
        "ledger_path": Path(result["ledger_path"]),
        "fit_baseline_path": baseline_path,
        "candidate_action_sha256_by_name": _candidate_action_hashes(),
        "manifest_sha256": SHA_A,
        "library_sha256": SHA_B,
        "config_sha256": SHA_C,
    }
    with pytest.raises(FileNotFoundError):
        validate_r2_stage_b_phase_completion_marker(marker_path, **common)
    marker = build_r2_stage_b_phase_completion_marker(**common)
    marker_path.write_bytes(canonical_json_bytes(marker) + b"\n")
    validated = validate_r2_stage_b_phase_completion_marker(marker_path, **common)
    assert validated["status"] == "PHASE_COMPLETE"
    assert validated["fit_baseline"]["row_count"] == 140 * 16

    artifacts = [
        Path(result["checkpoint_path"]),
        Path(result["ledger_path"]),
        baseline_path,
    ]
    for artifact_path in artifacts:
        original = artifact_path.read_bytes()
        artifact_path.write_bytes(original + b"tamper")
        with pytest.raises((ValueError, RuntimeError)):
            validate_r2_stage_b_phase_completion_marker(marker_path, **common)
        artifact_path.write_bytes(original)

    tampered_marker = dict(marker)
    tampered_marker["registration_commit"] = "5" * 40
    marker_path.write_bytes(canonical_json_bytes(tampered_marker) + b"\n")
    with pytest.raises(ValueError, match="marker|artifact|registration"):
        validate_r2_stage_b_phase_completion_marker(marker_path, **common)


def test_phase_completion_rejects_parent_symlink_alias_for_bound_artifacts(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, seed=3407, output_name="phase-alias.pth")
    baseline = _fit_baseline_for_checkpoint(result)
    baseline_path = tmp_path / "fit-baseline-alias.json"
    baseline_path.write_bytes(canonical_json_bytes(baseline) + b"\n")
    alias = tmp_path.parent / f"{tmp_path.name}-artifact-alias"
    try:
        alias.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")

    common = {
        "registration_sha256": SHA_A,
        "registration_commit": "4" * 40,
        "seed": 3407,
        "model": _ToyModel(),
        "batches": _batches(),
        "exposure_artifact": build_stage_b_exposure_artifact(_fit_windows()),
        "dense_checkpoint_path": tmp_path / "dense.pth",
        "dense_checkpoint_sha256": hashlib.sha256(
            (tmp_path / "dense.pth").read_bytes()
        ).hexdigest(),
        "dense_checkpoint_use_ema": False,
        "registered_provenance": _registered_provenance(),
        "checkpoint_path": Path(result["checkpoint_path"]),
        "ledger_path": Path(result["ledger_path"]),
        "fit_baseline_path": baseline_path,
        "candidate_action_sha256_by_name": _candidate_action_hashes(),
        "manifest_sha256": SHA_A,
        "library_sha256": SHA_B,
        "config_sha256": SHA_C,
    }
    artifact_aliases = dict(common)
    artifact_aliases.update(
        {
            "checkpoint_path": alias / Path(result["checkpoint_path"]).name,
            "ledger_path": alias / Path(result["ledger_path"]).name,
            "fit_baseline_path": alias / baseline_path.name,
        }
    )
    with pytest.raises(ValueError, match="symlink"):
        build_r2_stage_b_phase_completion_marker(**artifact_aliases)

    dense_alias = dict(common)
    dense_alias["dense_checkpoint_path"] = alias / "dense.pth"
    with pytest.raises(ValueError, match="symlink"):
        build_r2_stage_b_phase_completion_marker(**dense_alias)


@pytest.mark.parametrize("tamper", ["fit_windows", "registered_actions"])
def test_phase_complete_binds_baseline_to_current_fit_windows_and_registered_actions(
    tmp_path: Path, tamper: str
) -> None:
    result = _run(tmp_path, seed=3407, output_name=f"phase-bind-{tamper}.pth")
    baseline = _fit_baseline_for_checkpoint(result)
    source_rows = []
    for row in baseline["rows"]:
        unsigned = dict(row)
        unsigned.pop("row_sha256")
        source_rows.append(unsigned)
    windows = _fit_windows()
    action_hashes = _candidate_action_hashes()
    if tamper == "fit_windows":
        windows = [f"other/window-{index:03d}" for index in range(140)]
        for window_index in range(140):
            for candidate_index in range(16):
                source_rows[window_index * 16 + candidate_index]["window_id"] = windows[
                    window_index
                ]
    else:
        action_hashes = _candidate_action_hashes("d" * 64)
        for row in source_rows:
            row["requested_action_sha256"] = "d" * 64
            row["executed_action_sha256"] = "d" * 64
    forged = build_fit_schedule_constant_artifact(
        source_rows,
        seed=3407,
        fit_window_ids=windows,
        candidate_action_sha256_by_name=action_hashes,
        provenance=baseline["provenance"],
    )
    baseline_path = tmp_path / f"forged-{tamper}.json"
    baseline_path.write_bytes(canonical_json_bytes(forged) + b"\n")
    dense_checkpoint_path = tmp_path / "dense.pth"
    common = {
        "registration_sha256": SHA_A,
        "registration_commit": "4" * 40,
        "seed": 3407,
        "model": _ToyModel(),
        "batches": _batches(),
        "exposure_artifact": build_stage_b_exposure_artifact(_fit_windows()),
        "dense_checkpoint_path": dense_checkpoint_path,
        "dense_checkpoint_sha256": hashlib.sha256(
            dense_checkpoint_path.read_bytes()
        ).hexdigest(),
        "dense_checkpoint_use_ema": False,
        "registered_provenance": _registered_provenance(),
        "checkpoint_path": Path(result["checkpoint_path"]),
        "ledger_path": Path(result["ledger_path"]),
        "fit_baseline_path": baseline_path,
        "candidate_action_sha256_by_name": _candidate_action_hashes(),
        "manifest_sha256": SHA_A,
        "library_sha256": SHA_B,
        "config_sha256": SHA_C,
    }
    with pytest.raises(ValueError, match="fit window|registered action"):
        build_r2_stage_b_phase_completion_marker(**common)


@pytest.mark.parametrize(
    "tamper",
    ["checkpoint_ledger_order", "checkpoint_ledger_missing", "optimizer_step"],
)
def test_phase_complete_reuses_full_resume_validators_and_binds_internal_external_ledger(
    tmp_path: Path, tamper: str
) -> None:
    result = _run(tmp_path, seed=3407, output_name=f"phase-{tamper}.pth")
    checkpoint_path = Path(result["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if tamper == "checkpoint_ledger_order":
        checkpoint["ledger_rows"][0], checkpoint["ledger_rows"][1] = (
            checkpoint["ledger_rows"][1],
            checkpoint["ledger_rows"][0],
        )
    elif tamper == "checkpoint_ledger_missing":
        checkpoint["ledger_rows"].pop()
    else:
        for parameter_state in checkpoint["optimizer"]["state"].values():
            parameter_state["step"] = torch.as_tensor(139.0)
    torch.save(checkpoint, checkpoint_path)

    baseline = _fit_baseline_for_checkpoint(result)
    baseline["provenance"]["trained_checkpoint_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    unsigned_baseline = dict(baseline)
    unsigned_baseline.pop("artifact_sha256")
    baseline["artifact_sha256"] = canonical_sha256(unsigned_baseline)
    baseline_path = tmp_path / f"fit-baseline-{tamper}.json"
    baseline_path.write_bytes(canonical_json_bytes(baseline) + b"\n")
    dense_checkpoint_path = tmp_path / "dense.pth"
    common = {
        "registration_sha256": SHA_A,
        "registration_commit": "4" * 40,
        "seed": 3407,
        "model": _ToyModel(),
        "batches": _batches(),
        "exposure_artifact": build_stage_b_exposure_artifact(_fit_windows()),
        "dense_checkpoint_path": dense_checkpoint_path,
        "dense_checkpoint_sha256": hashlib.sha256(
            dense_checkpoint_path.read_bytes()
        ).hexdigest(),
        "dense_checkpoint_use_ema": False,
        "registered_provenance": _registered_provenance(),
        "checkpoint_path": checkpoint_path,
        "ledger_path": Path(result["ledger_path"]),
        "fit_baseline_path": baseline_path,
        "candidate_action_sha256_by_name": _candidate_action_hashes(),
        "manifest_sha256": SHA_A,
        "library_sha256": SHA_B,
        "config_sha256": SHA_C,
    }
    with pytest.raises(ValueError, match="ledger|optimizer|cursor|training"):
        build_r2_stage_b_phase_completion_marker(**common)


def test_stage_b_cli_passes_current_R_and_formal_repository_context(monkeypatch) -> None:
    captured = {}
    registration_path = ROOT / "artifacts/chronotransport_pre_gate1_registration.json"
    monkeypatch.setattr(stage_b_cli, "_load_json", lambda path: {"payload": str(path)})
    monkeypatch.setattr(stage_b_cli, "_git_head", lambda root: "6" * 40)

    def validate(payload, **kwargs):
        captured.update(kwargs)
        return {"registration_sha256": SHA_A}

    monkeypatch.setattr(stage_b_cli, "validate_pre_gate1_registration", validate)
    stage_b_cli._load_formal_registration(registration_path)
    assert captured == {
        "repository_root": ROOT,
        "context_mode": "formal",
        "registration_commit": "6" * 40,
        "registration_relpath": "artifacts/chronotransport_pre_gate1_registration.json",
    }

    def reject_source_context(payload, **kwargs):
        raise ValueError("actual implementation tree differs from registration")

    monkeypatch.setattr(
        stage_b_cli, "validate_pre_gate1_registration", reject_source_context
    )
    with pytest.raises(ValueError, match="implementation tree"):
        stage_b_cli._load_formal_registration(registration_path)


def test_stage_b_cli_hashes_nested_manifest_split_identity(tmp_path: Path) -> None:
    unlock_path = tmp_path / "gate1-unlock.json"
    unlock_path.write_text("{}\n", encoding="utf-8")
    split_hashes = {"fit": SHA_A, "calibration": SHA_B, "evaluation": SHA_C}
    registration = {
        "registration_sha256": SHA_A,
        "spec": {"commit": "1" * 40, "sha256": SHA_B},
        "implementation_commit": "2" * 40,
        "source_files": {"source.py": SHA_A},
        "upstream_commits": {"OpenTAD": "3" * 40},
        "window_manifest": {"artifact": {"split_hashes": split_hashes}},
        "candidate_library": {"library_sha256": SHA_C},
        "environment": _required_environment(),
        "profiler": {"profile": "fixed"},
    }
    provenance = stage_b_cli._registered_provenance(
        registration,
        {"status": "PASS"},
        unlock_path,
        "4" * 40,
        build_test_only_observed_environment(_required_environment()),
    )
    assert provenance["split_hashes_sha256"] == canonical_sha256(split_hashes)


def test_stage_b_completion_artifact_publish_is_atomic_no_clobber(
    tmp_path: Path,
) -> None:
    output = tmp_path / "phase-complete.json"
    stage_b_cli._atomic_write(output, b"first\n")
    with pytest.raises(FileExistsError):
        stage_b_cli._atomic_write(output, b"second\n")
    assert output.read_bytes() == b"first\n"


def test_stage_b_completion_artifact_publish_reuses_only_exact_bytes_after_interrupt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "phase-complete.json"
    payload = b'exact-phase-artifact\n'
    stage_b_cli._atomic_write(output, payload)
    original_identity = output.stat().st_ino

    stage_b_cli._atomic_write(output, payload)

    assert output.read_bytes() == payload
    assert output.stat().st_ino == original_identity


def test_stage_b_checkpoint_and_ledger_publish_are_atomic_no_clobber(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stage-b.pth"
    ledger = tmp_path / "stage-b.jsonl"
    _atomic_torch_save({"value": torch.tensor(1)}, checkpoint)
    _atomic_write_ledger([{"value": 1}], ledger)
    first_checkpoint = checkpoint.read_bytes()
    first_ledger = ledger.read_bytes()

    with pytest.raises(FileExistsError):
        _atomic_torch_save({"value": torch.tensor(2)}, checkpoint)
    with pytest.raises(FileExistsError):
        _atomic_write_ledger([{"value": 2}], ledger)
    assert checkpoint.read_bytes() == first_checkpoint
    assert ledger.read_bytes() == first_ledger


def test_stage_b_ledger_publish_reuses_only_exact_bytes_after_interrupt(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "stage-b.jsonl"
    rows = [{"value": 1}, {"value": 2}]
    first_digest = _atomic_write_ledger(rows, ledger)
    first_bytes = ledger.read_bytes()
    original_identity = ledger.stat().st_ino

    second_digest = _atomic_write_ledger(rows, ledger)

    assert second_digest == first_digest
    assert ledger.read_bytes() == first_bytes
    assert ledger.stat().st_ino == original_identity


def test_stage_b_cli_finalizes_and_reuses_an_exact_existing_training_pair(
    tmp_path: Path, monkeypatch
) -> None:
    training = _run(
        tmp_path,
        seed=3407,
        output_name="existing-training.pth",
    )
    dense_checkpoint = tmp_path / "dense.pth"
    exposure = build_stage_b_exposure_artifact(_fit_windows())
    baseline_rows = [
        {key: value for key, value in row.items() if key != "row_sha256"}
        for row in _fit_baseline_for_checkpoint(training)["rows"]
    ]

    class _MaterializedBatches(list):
        @property
        def window_ids(self):
            return [row["window_id"] for row in self]

    components = SimpleNamespace(
        model=_ToyModel(),
        batches=_MaterializedBatches(_batches()),
        exposure_artifact=exposure,
        dense_checkpoint_use_ema=False,
        manifest={"manifest_sha256": SHA_A},
        config_sha256=SHA_C,
        fit_baseline_rows=lambda: baseline_rows,
        replay_step=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed training pair must not be replayed")
        ),
        candidate_order_probe=lambda: (_ for _ in ()).throw(
            AssertionError("completed training pair must not rerun preflight")
        ),
    )
    registration = {
        "registration_sha256": SHA_A,
        "candidate_library": {
            "library_sha256": SHA_B,
            "candidates": [
                {"name": name, "action_sha256": SHA_C}
                for name in R2_NON_DENSE_NAMES
            ],
        },
        "dense_checkpoint": {
            "sha256": hashlib.sha256(dense_checkpoint.read_bytes()).hexdigest(),
            "bytes": dense_checkpoint.stat().st_size,
        },
        "environment": _required_environment(),
    }
    outputs = {
        "output": Path(training["checkpoint_path"]),
        "ledger": Path(training["ledger_path"]),
        "fit_baseline": tmp_path / "fit-baseline.json",
        "phase_marker": tmp_path / "PHASE_COMPLETE.json",
    }
    args = SimpleNamespace(
        gate1_unlock=tmp_path / "gate1-unlock.json",
        exposure_artifact=tmp_path / "exposure.json",
        checkpoint=dense_checkpoint,
        manifest=tmp_path / "manifest.json",
        media_registry=tmp_path / "registry.json",
        config_identity=tmp_path / "config.json",
        seed=3407,
        resume=None,
        checkpoint_frequency=1,
    )

    monkeypatch.setattr(
        stage_b_cli,
        "_formal_guard",
        lambda *args, **kwargs: build_test_only_observed_environment(
            _required_environment()
        ),
    )
    monkeypatch.setattr(
        stage_b_cli,
        "audit_formal_python_runtime",
        lambda **kwargs: {"status": "TEST_ONLY"},
    )
    monkeypatch.setattr(
        stage_b_cli,
        "_load_json",
        lambda path: exposure if Path(path) == args.exposure_artifact else {"status": "PASS"},
    )
    monkeypatch.setattr(stage_b_cli, "_validate_unlock", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        stage_b_cli, "build_repository_stage_b_components", lambda **kwargs: components
    )
    monkeypatch.setattr(
        stage_b_cli,
        "_registered_provenance",
        lambda *args, **kwargs: _registered_provenance(),
    )
    monkeypatch.setattr(
        stage_b_cli,
        "run_r2_stage_b_training",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("completed training pair must not be retrained")
        ),
    )

    alias = tmp_path.parent / f"{tmp_path.name}-final-pair-alias"
    try:
        alias.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        alias = None
    if alias is not None:
        aliased_outputs = dict(outputs)
        aliased_outputs["output"] = alias / outputs["output"].name
        aliased_outputs["ledger"] = alias / outputs["ledger"].name
        with pytest.raises(ValueError, match="symlink"):
            stage_b_cli._run_locked(
                args,
                registration=registration,
                registration_commit="4" * 40,
                registration_relpath="artifacts/registration.json",
                outputs=aliased_outputs,
            )

    nonregular = tmp_path / "nonregular-checkpoint"
    nonregular.mkdir()
    nonregular_outputs = dict(outputs)
    nonregular_outputs["output"] = nonregular
    with pytest.raises(ValueError, match="regular"):
        stage_b_cli._run_locked(
            args,
            registration=registration,
            registration_commit="4" * 40,
            registration_relpath="artifacts/registration.json",
            outputs=nonregular_outputs,
        )

    first = stage_b_cli._run_locked(
        args,
        registration=registration,
        registration_commit="4" * 40,
        registration_relpath="artifacts/registration.json",
        outputs=outputs,
    )
    baseline_identity = outputs["fit_baseline"].stat().st_ino
    marker_identity = outputs["phase_marker"].stat().st_ino
    second = stage_b_cli._run_locked(
        args,
        registration=registration,
        registration_commit="4" * 40,
        registration_relpath="artifacts/registration.json",
        outputs=outputs,
    )

    assert first["status"] == second["status"] == "PHASE_COMPLETE"
    assert first["reused_existing_training_pair"] is True
    assert second["reused_existing_training_pair"] is True
    assert outputs["fit_baseline"].stat().st_ino == baseline_identity
    assert outputs["phase_marker"].stat().st_ino == marker_identity

    outputs["fit_baseline"].unlink()
    with pytest.raises(RuntimeError, match="phase marker exists without"):
        stage_b_cli._run_locked(
            args,
            registration=registration,
            registration_commit="4" * 40,
            registration_relpath="artifacts/registration.json",
            outputs=outputs,
        )
    assert not outputs["fit_baseline"].exists()


def test_stage_b_formal_writer_lock_is_exclusive(tmp_path: Path) -> None:
    lock = tmp_path / ".stage-b.lock"
    with stage_b_cli._exclusive_run_lock(lock):
        with pytest.raises(FileExistsError):
            with stage_b_cli._exclusive_run_lock(lock):
                raise AssertionError("unreachable")
    assert not lock.exists()


def test_stage_b_formal_writer_lock_rejects_symlink_parent_and_preserves_replacement(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-lock-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-lock-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")
    with pytest.raises(ValueError, match="symlink"):
        with stage_b_cli._exclusive_run_lock(linked_parent / "run.lock"):
            pass

    lock = real_parent / "run.lock"
    with stage_b_cli._exclusive_run_lock(lock):
        lock.unlink()
        lock.write_text("replacement\n", encoding="ascii")
    assert lock.read_text(encoding="ascii") == "replacement\n"


def test_stage_b_cli_validates_gate1_unlock_in_exact_R_context(monkeypatch) -> None:
    captured = {}

    def validate(artifact, **kwargs):
        captured["artifact"] = artifact
        captured.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(stage_b_cli, "validate_gate1_unlock_artifact", validate)
    registration = {"registration_sha256": SHA_A}
    unlock = {"schema": "formal-unlock"}
    result = stage_b_cli._validate_unlock(
        unlock,
        registration=registration,
        repository_root=ROOT,
        registration_commit="6" * 40,
        registration_relpath="artifacts/chronotransport_pre_gate1_registration.json",
    )
    assert result == {"status": "PASS"}
    assert captured == {
        "artifact": unlock,
        "registration": registration,
        "repository_root": str(ROOT),
        "registration_commit": "6" * 40,
        "registration_relpath": "artifacts/chronotransport_pre_gate1_registration.json",
    }


def test_stage_b_cli_confines_distinct_outputs_to_canonical_r_seed_root(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "formal-output"
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(stage_b_cli, "FORMAL_OUTPUT_BASE", str(base))
    registration = {"output_root": {"base": str(base)}}
    registration_commit = "6" * 40
    seed_root = base / registration_commit / "3407"
    inputs = [repository / "registration.json", tmp_path / "dense.pth"]
    for path in inputs:
        path.write_bytes(b"input")
    args = SimpleNamespace(
        seed=3407,
        output=seed_root / "stage-b.pth",
        ledger=seed_root / "stage-b.jsonl",
        fit_baseline=seed_root / "fit-baseline.json",
        phase_marker=seed_root / "PHASE_COMPLETE.json",
        registration=inputs[0],
        gate1_unlock=tmp_path / "unlock.json",
        manifest=tmp_path / "manifest.json",
        media_registry=tmp_path / "media.json",
        config_identity=tmp_path / "config.json",
        exposure_artifact=tmp_path / "exposure.json",
        checkpoint=inputs[1],
        resume=None,
    )
    resolved = stage_b_cli._resolve_formal_stage_b_paths(
        args,
        registration=registration,
        registration_commit=registration_commit,
        repository_root=repository,
    )
    assert set(resolved) == {"output", "ledger", "fit_baseline", "phase_marker"}
    assert all(path.parent == seed_root.resolve() for path in resolved.values())

    args.phase_marker = repository / "registration.json"
    with pytest.raises(ValueError, match="canonical.*seed|repository"):
        stage_b_cli._resolve_formal_stage_b_paths(
            args,
            registration=registration,
            registration_commit=registration_commit,
            repository_root=repository,
        )

    args.phase_marker = args.output
    with pytest.raises(ValueError, match="distinct"):
        stage_b_cli._resolve_formal_stage_b_paths(
            args,
            registration=registration,
            registration_commit=registration_commit,
            repository_root=repository,
        )


def test_stage_b_cli_rejects_symlink_escape_from_canonical_seed_root(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "formal-output"
    registration_commit = "6" * 40
    seed_root = base / registration_commit / "3407"
    seed_root.mkdir(parents=True)
    escape = tmp_path / "escape"
    escape.mkdir()
    link = seed_root / "linked"
    try:
        link.symlink_to(escape, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")
    monkeypatch.setattr(stage_b_cli, "FORMAL_OUTPUT_BASE", str(base))
    args = SimpleNamespace(
        seed=3407,
        output=seed_root / "stage-b.pth",
        ledger=seed_root / "stage-b.jsonl",
        fit_baseline=seed_root / "fit-baseline.json",
        phase_marker=link / "marker.json",
        registration=tmp_path / "registration.json",
        gate1_unlock=tmp_path / "unlock.json",
        manifest=tmp_path / "manifest.json",
        media_registry=tmp_path / "media.json",
        config_identity=tmp_path / "config.json",
        exposure_artifact=tmp_path / "exposure.json",
        checkpoint=tmp_path / "dense.pth",
        resume=None,
    )
    with pytest.raises(ValueError, match="canonical.*seed|symlink"):
        stage_b_cli._resolve_formal_stage_b_paths(
            args,
            registration={"output_root": {"base": str(base)}},
            registration_commit=registration_commit,
            repository_root=tmp_path / "repository",
        )


def test_stage_b_cli_rejects_symlink_alias_to_canonical_seed_root(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "formal-output"
    registration_commit = "6" * 40
    seed_root = base / registration_commit / "3407"
    seed_root.mkdir(parents=True)
    alias = tmp_path / "seed-root-alias"
    try:
        alias.symlink_to(seed_root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")
    monkeypatch.setattr(stage_b_cli, "FORMAL_OUTPUT_BASE", str(base))
    args = SimpleNamespace(
        seed=3407,
        output=seed_root / "stage-b.pth",
        ledger=seed_root / "stage-b.jsonl",
        fit_baseline=seed_root / "fit-baseline.json",
        phase_marker=alias / "PHASE_COMPLETE.json",
        registration=tmp_path / "registration.json",
        gate1_unlock=tmp_path / "unlock.json",
        manifest=tmp_path / "manifest.json",
        media_registry=tmp_path / "media.json",
        config_identity=tmp_path / "config.json",
        exposure_artifact=tmp_path / "exposure.json",
        checkpoint=tmp_path / "dense.pth",
        resume=None,
    )
    with pytest.raises(ValueError, match="symlink"):
        stage_b_cli._resolve_formal_stage_b_paths(
            args,
            registration={"output_root": {"base": str(base)}},
            registration_commit=registration_commit,
            repository_root=tmp_path / "repository",
        )


def test_r2_stage_b_cli_is_executable_and_legacy_runner_stays_locked() -> None:
    option_strings = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert "--factory" not in option_strings
    assert "--registration" in option_strings
    assert "--manifest" in option_strings
    assert "--phase-marker" in option_strings
    resume = next(action for action in build_parser()._actions if "--resume" in action.option_strings)
    assert "formal periodic" in resume.help
    current = subprocess.run(
        [sys.executable, str(ROOT / "tools/bata/train_chronotransport_r2_stage_b.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert current.returncode == 0, current.stderr
    legacy = subprocess.run(
        [sys.executable, str(ROOT / "tools/bata/run_chronotransport_stage_b_formal.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert legacy.returncode != 0
    assert "superseded by CT-P3R-3S-r2" in legacy.stderr
