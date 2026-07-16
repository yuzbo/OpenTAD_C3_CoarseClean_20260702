from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest

import opentad.models.chronotransport.environment as environment_module
from opentad.models.chronotransport.environment import (
    InvalidEnvironmentError,
    REQUIRED_ENVIRONMENT_SCHEMA,
    build_test_only_observed_environment,
    observe_formal_slurm_environment,
    observed_environment_from_provenance,
    observed_environment_to_provenance,
    validate_observed_environment,
)
from opentad.models.chronotransport.protocol import canonical_sha256


def _required_environment(**overrides):
    value = {
        "schema": REQUIRED_ENVIRONMENT_SCHEMA,
        "gpu_model": "NVIDIA A100-SXM4-80GB",
        "driver": "535.54",
        "cuda": "11.8",
        "pytorch": "2.0.1",
        "cudnn": "8902",
        "precision": "amp_fp16",
        "batch_size": 1,
        **overrides,
    }
    value["environment_sha256"] = canonical_sha256(value)
    return value


class _FakeCuda:
    def __init__(self, *, count: int = 1, model: str = "NVIDIA A100-SXM4-80GB"):
        self._count = count
        self._model = model
        self.selected = None

    def is_available(self):
        return True

    def device_count(self):
        return self._count

    def set_device(self, index):
        self.selected = index

    def current_device(self):
        return self.selected

    def synchronize(self):
        return None

    def get_device_name(self, index):
        assert index == 0
        return self._model


class _FakeTorch:
    __version__ = "2.0.1"

    def __init__(self, *, count: int = 1):
        self.cuda = _FakeCuda(count=count)
        self.version = SimpleNamespace(cuda="11.8")
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 8902)
        )

    @staticmethod
    def device(value):
        return value

    @staticmethod
    def empty(shape, *, device):
        assert shape == (1,)
        assert device == "cuda:0"
        return object()


def _bind_fake_live_environment(monkeypatch, *, count: int = 1, gpu_uuid: str = "GPU-A100-3"):
    fake_torch = _FakeTorch(count=count)
    monkeypatch.setattr(environment_module, "_torch_module", lambda: fake_torch)

    def nvidia_smi(arguments):
        query = arguments[0]
        if query.startswith("--query-compute-apps"):
            return f"{os.getpid()}, {gpu_uuid}\n"
        if query.startswith("--query-gpu"):
            return f"{gpu_uuid}, NVIDIA A100-SXM4-80GB, 535.54\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(environment_module, "_run_nvidia_smi", nvidia_smi)
    return fake_torch


def _slurm_env(monkeypatch):
    values = {
        "CUDA_VISIBLE_DEVICES": "3",
        "SLURM_JOB_ID": "81234",
        "SLURM_STEP_ID": "7",
        "SLURM_JOB_GPUS": "3",
        "SLURM_STEP_GPUS": "3",
        "SLURM_GPUS_ON_NODE": "1",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def test_physical_gpu3_allocation_is_accepted_without_rewriting_visibility(monkeypatch):
    raw = _slurm_env(monkeypatch)
    fake_torch = _bind_fake_live_environment(monkeypatch)
    observed = observe_formal_slurm_environment(_required_environment())

    assert observed["cuda_visible_devices_raw"] == raw["CUDA_VISIBLE_DEVICES"] == "3"
    assert observed["slurm_job_gpus_raw"] == "3"
    assert observed["gpu_uuid"] == "GPU-A100-3"
    assert observed["logical_cuda_index"] == 0
    assert observed["torch_device_count"] == 1
    assert fake_torch.cuda.selected == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_formal_observer_rejects_missing_slurm_step_and_multiple_visible_devices(monkeypatch):
    _slurm_env(monkeypatch)
    monkeypatch.delenv("SLURM_STEP_ID")
    _bind_fake_live_environment(monkeypatch)
    with pytest.raises(InvalidEnvironmentError, match="SLURM_JOB_ID.*SLURM_STEP_ID"):
        observe_formal_slurm_environment(_required_environment())

    monkeypatch.setenv("SLURM_STEP_ID", "7")
    _bind_fake_live_environment(monkeypatch, count=2)
    with pytest.raises(InvalidEnvironmentError, match="exactly one"):
        observe_formal_slurm_environment(_required_environment())


def test_formal_observer_rejects_model_software_and_mig_identity_mismatch(monkeypatch):
    _slurm_env(monkeypatch)
    _bind_fake_live_environment(monkeypatch)
    with pytest.raises(InvalidEnvironmentError, match="pytorch"):
        observe_formal_slurm_environment(_required_environment(pytorch="2.1.0"))

    _bind_fake_live_environment(monkeypatch, gpu_uuid="MIG-deadbeef")
    with pytest.raises(InvalidEnvironmentError, match="full-GPU UUID"):
        observe_formal_slurm_environment(_required_environment())


def test_observed_allocation_provenance_round_trip_is_exact_and_tamper_evident():
    required = _required_environment()
    observed = build_test_only_observed_environment(required)
    provenance = observed_environment_to_provenance(
        observed, required_environment=required
    )
    assert observed_environment_from_provenance(
        provenance, required_environment=required
    ) == observed

    damaged = dict(observed, slurm_step_id="different")
    with pytest.raises(InvalidEnvironmentError, match="allocation identity"):
        validate_observed_environment(damaged, required_environment=required)


def test_formal_environment_observer_has_no_caller_identity_parameters():
    assert tuple(inspect.signature(observe_formal_slurm_environment).parameters) == (
        "required_environment",
    )
