import inspect
import os
from pathlib import Path

import pytest

from tools.bata import continuous_roi_s2_training as training
from tools.bata import finalize_continuous_roi_s2_training_matrix as matrix
from tools.bata.continuous_roi_s2_contract import canonical_sha256


def _self_hashed(payload: dict, key: str) -> dict:
    result = dict(payload)
    result[key] = canonical_sha256(result)
    return result


def test_training_completion_validation_rebuilds_live_evidence(monkeypatch):
    receipt = _self_hashed(
        {
            "schema_version": "continuous_roi_s2_training_completion_v1",
            "checkpoint_sha256": "a" * 64,
        },
        "completion_sha256",
    )
    monkeypatch.setattr(
        training,
        "_build_training_completion_and_audit",
        lambda **kwargs: (dict(receipt), {"checkpoint_epoch": 59}),
    )
    assert (
        training.validate_training_completion(
            receipt,
            cfg=object(),
            seed=3407,
            checkpoint_path=Path("epoch_59.pth"),
        )
        == receipt
    )

    changed = dict(receipt)
    changed["checkpoint_sha256"] = "b" * 64
    changed["completion_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "completion_sha256"}
    )
    monkeypatch.setattr(
        training,
        "_build_training_completion_and_audit",
        lambda **kwargs: (changed, {"checkpoint_epoch": 59}),
    )
    with pytest.raises(ValueError, match="no longer matches its artifacts"):
        training.validate_training_completion(
            receipt,
            cfg=object(),
            seed=3407,
            checkpoint_path=Path("epoch_59.pth"),
        )


def test_pure_bound_config_loader_rejects_executable_syntax(tmp_path):
    safe = tmp_path / "safe.py"
    safe.write_text(
        "family = 'U128'\nsettings = dict(seed=3407, values=[1, -2, None])\n",
        encoding="utf-8",
    )
    assert training.load_pure_data_config(safe).family == "U128"

    unsafe = tmp_path / "unsafe.py"
    unsafe.write_text(
        "payload = open('/sealed/test').read()\nfamily = 'U128'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pure data assignment"):
        training.load_pure_data_config(unsafe)

    inherited = tmp_path / "inherited.py"
    inherited.write_text("_base_ = 'evil.py'\nfamily = 'U128'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reserved or duplicate key"):
        training.load_pure_data_config(inherited)

    custom_imports = tmp_path / "custom_imports.py"
    custom_imports.write_text(
        "custom_imports = dict(imports=['evil'])\nfamily = 'U128'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reserved or duplicate key"):
        training.load_pure_data_config(custom_imports)


def test_final_checkpoint_audit_rejects_incomplete_or_stale_state():
    if os.name == "nt":
        pytest.skip("checkpoint tensor audit runs in the bound Linux runtime")
    try:
        import torch
    except (ImportError, OSError):
        pytest.skip("local Torch runtime is unavailable")

    metadata = {
        "epoch": 59,
        "updates_per_epoch": 80,
        "successful_updates": 4800,
        "optimizer_attempts": 4803,
        "amp_skipped_attempts": 3,
        "max_amp_retries_per_batch": 8,
        "max_amp_retries_observed": 2,
    }
    checkpoint = {
        "epoch": 59,
        "state_dict": {"weight": torch.tensor([1.0])},
        "state_dict_ema": {"weight": torch.tensor([0.9])},
        "optimizer": {"state": {0: {"step": torch.tensor(4800)}}, "param_groups": [{}]},
        "scheduler": {
            "last_epoch": 4800,
            "_step_count": 4801,
            "_last_lr": [1e-4],
            "max_epoch": 8000,
            "warmup_epoch": 400,
        },
        "experiment_metadata": metadata,
    }
    audit = training.audit_final_checkpoint_state(
        checkpoint,
        metadata=metadata,
        expected_scheduler_max_epoch=8000,
        expected_scheduler_warmup_epoch=400,
    )
    assert audit["checkpoint_epoch"] == 59
    assert audit["ema_changed_value_count"] == 1
    assert audit["nonfinite_value_count"] == 0

    stale = dict(checkpoint)
    stale["state_dict_ema"] = {"weight": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="non-finite or stale EMA"):
        training.audit_final_checkpoint_state(
            stale,
            metadata=metadata,
            expected_scheduler_max_epoch=8000,
            expected_scheduler_warmup_epoch=400,
        )

    incomplete = dict(checkpoint)
    incomplete["scheduler"] = {}
    with pytest.raises(ValueError, match="complete training state"):
        training.audit_final_checkpoint_state(
            incomplete,
            metadata=metadata,
            expected_scheduler_max_epoch=8000,
            expected_scheduler_warmup_epoch=400,
        )

    stale_optimizer = dict(checkpoint)
    stale_optimizer["optimizer"] = {
        "state": {0: {"step": torch.tensor(4799)}},
        "param_groups": [{}],
    }
    with pytest.raises(ValueError, match="do not close at 4,800 updates"):
        training.audit_final_checkpoint_state(
            stale_optimizer,
            metadata=metadata,
            expected_scheduler_max_epoch=8000,
            expected_scheduler_warmup_epoch=400,
        )

    wrong_schedule = dict(checkpoint)
    wrong_schedule["scheduler"] = dict(checkpoint["scheduler"], max_epoch=4800)
    with pytest.raises(ValueError, match="complete training state"):
        training.audit_final_checkpoint_state(
            wrong_schedule,
            metadata=metadata,
            expected_scheduler_max_epoch=8000,
            expected_scheduler_warmup_epoch=400,
        )

    strict_model = torch.nn.Linear(1, 1)
    strict_optimizer = torch.optim.AdamW(strict_model.parameters(), lr=1e-4)
    strict_optimizer.zero_grad(set_to_none=True)
    strict_model(torch.ones(1, 1)).sum().backward()
    strict_optimizer.step()
    strict_checkpoint = {
        "state_dict": strict_model.state_dict(),
        "state_dict_ema": strict_model.state_dict(),
        "optimizer": strict_optimizer.state_dict(),
    }
    validation_model = torch.nn.Linear(1, 1)
    validation_optimizer = torch.optim.AdamW(validation_model.parameters(), lr=1e-4)
    strict_audit = training.audit_checkpoint_against_model(
        strict_checkpoint,
        model=validation_model,
        optimizer=validation_optimizer,
    )
    assert strict_audit["raw_model_strict_load_valid"] is True

    class BufferHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("loss_normalizer", torch.tensor(100))

    class BufferDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.rpn_head = BufferHead()
            self.register_buffer("runtime_only", torch.tensor(0), persistent=False)

        def forward(self, value):
            return value * self.weight

    class BufferModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = BufferDetector()

        def forward(self, value):
            return self.module(value)

    trained_buffer_model = BufferModel()
    trained_buffer_optimizer = torch.optim.AdamW(
        trained_buffer_model.parameters(), lr=1e-4
    )
    trained_buffer_model(torch.ones(1)).sum().backward()
    trained_buffer_optimizer.step()
    raw_buffer_state = trained_buffer_model.state_dict()
    raw_buffer_state["module.rpn_head.loss_normalizer"] = torch.tensor(42.5)
    buffer_checkpoint = {
        "state_dict": raw_buffer_state,
        "state_dict_ema": trained_buffer_model.state_dict(),
        "optimizer": trained_buffer_optimizer.state_dict(),
    }
    validation_buffer_model = BufferModel()
    buffer_audit = training.audit_checkpoint_against_model(
        buffer_checkpoint,
        model=validation_buffer_model,
        optimizer=torch.optim.AdamW(validation_buffer_model.parameters(), lr=1e-4),
    )
    assert buffer_audit["raw_buffer_dtype_cast_keys"] == list(
        training.S2_ALLOWED_RAW_BUFFER_DTYPE_CAST_KEYS
    )
    assert set(buffer_audit["raw_buffer_dtype_cast_keys"]) == {
        "module.rpn_head.loss_normalizer"
    }

    class UnknownBufferHead(BufferHead):
        def __init__(self):
            super().__init__()
            self.register_buffer("unexpected", torch.tensor(1))

    class UnknownBufferDetector(BufferDetector):
        def __init__(self):
            super().__init__()
            self.rpn_head = UnknownBufferHead()

    class UnknownBufferModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = UnknownBufferDetector()

        def forward(self, value):
            return self.module(value)

    unknown_buffer_model = UnknownBufferModel()
    unknown_buffer_optimizer = torch.optim.AdamW(
        unknown_buffer_model.parameters(), lr=1e-4
    )
    unknown_buffer_state = {
        key: value.clone() for key, value in unknown_buffer_model.state_dict().items()
    }
    unknown_buffer_state["module.rpn_head.unexpected"] = torch.tensor(1.5)
    unknown_buffer_checkpoint = {
        "state_dict": unknown_buffer_state,
        "state_dict_ema": unknown_buffer_model.state_dict(),
        "optimizer": unknown_buffer_optimizer.state_dict(),
    }
    validation_unknown_buffer_model = UnknownBufferModel()
    with pytest.raises(ValueError, match="raw buffer dtype casts are not allowlisted"):
        training.audit_checkpoint_against_model(
            unknown_buffer_checkpoint,
            model=validation_unknown_buffer_model,
            optimizer=torch.optim.AdamW(
                validation_unknown_buffer_model.parameters(), lr=1e-4
            ),
        )
    raw_parameter_state = {
        key: value.clone() for key, value in strict_checkpoint["state_dict"].items()
    }
    raw_parameter_state["weight"] = raw_parameter_state["weight"].half()
    parameter_dtype_checkpoint = dict(strict_checkpoint, state_dict=raw_parameter_state)
    parameter_dtype_model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="raw parameter dtype differs"):
        training.audit_checkpoint_against_model(
            parameter_dtype_checkpoint,
            model=parameter_dtype_model,
            optimizer=torch.optim.AdamW(parameter_dtype_model.parameters(), lr=1e-4),
        )

    class AliasModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.nn.Parameter(torch.ones(1))
            self.primary = shared
            self.alias = shared

    trained_alias_model = AliasModel()
    alias_optimizer = torch.optim.AdamW(trained_alias_model.parameters(), lr=1e-4)
    alias_raw_state = {
        key: value.clone() for key, value in trained_alias_model.state_dict().items()
    }
    alias_raw_state["alias"] = alias_raw_state["alias"].half()
    alias_checkpoint = {
        "state_dict": alias_raw_state,
        "state_dict_ema": trained_alias_model.state_dict(),
        "optimizer": alias_optimizer.state_dict(),
    }
    validation_alias_model = AliasModel()
    with pytest.raises(ValueError, match="raw parameter dtype differs for alias"):
        training.audit_checkpoint_against_model(
            alias_checkpoint,
            model=validation_alias_model,
            optimizer=torch.optim.AdamW(validation_alias_model.parameters(), lr=1e-4),
        )
    orphan_optimizer = strict_optimizer.state_dict()
    orphan_optimizer["state"][999] = {"step": torch.tensor(4800)}
    orphan_checkpoint = dict(strict_checkpoint, optimizer=orphan_optimizer)
    orphan_model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="orphan state"):
        training.audit_checkpoint_against_model(
            orphan_checkpoint,
            model=orphan_model,
            optimizer=torch.optim.AdamW(orphan_model.parameters(), lr=1e-4),
        )
    duplicate_optimizer = strict_optimizer.state_dict()
    duplicate_optimizer["param_groups"][0]["params"].append(
        duplicate_optimizer["param_groups"][0]["params"][0]
    )
    duplicate_checkpoint = dict(strict_checkpoint, optimizer=duplicate_optimizer)
    duplicate_model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="repeats a parameter ID"):
        training.audit_checkpoint_against_model(
            duplicate_checkpoint,
            model=duplicate_model,
            optimizer=torch.optim.AdamW(duplicate_model.parameters(), lr=1e-4),
        )
    malformed = dict(strict_checkpoint)
    malformed["state_dict"] = {"wrong.weight": torch.ones(1)}
    with pytest.raises(ValueError, match="checkpoint keys differ"):
        malformed_model = torch.nn.Linear(1, 1)
        training.audit_checkpoint_against_model(
            malformed,
            model=malformed_model,
            optimizer=torch.optim.AdamW(malformed_model.parameters(), lr=1e-4),
        )

    invalid_amp = dict(checkpoint)
    invalid_amp_metadata = dict(metadata, max_amp_retries_observed=0)
    invalid_amp["experiment_metadata"] = invalid_amp_metadata
    with pytest.raises(ValueError, match="complete training state"):
        training.audit_final_checkpoint_state(
            invalid_amp,
            metadata=invalid_amp_metadata,
            expected_scheduler_max_epoch=8000,
            expected_scheduler_warmup_epoch=400,
        )


def test_matrix_accounting_binds_job_name_and_comment():
    job_ids = ["1177668", "1177669"]
    expected_jobs = {
        job_id: {
            "job_name": f"job-{job_id}",
            "job_token": f"token-{job_id}",
        }
        for job_id in job_ids
    }
    accounting = {
        job_id: {
            "job_name": expected_jobs[job_id]["job_name"],
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed": "03:44:46",
            "comment": f"crs2:{expected_jobs[job_id]['job_token']}",
        }
        for job_id in job_ids
    }
    checked = matrix.validate_accounting_rows(
        job_ids, accounting, expected_jobs=expected_jobs
    )
    assert checked == accounting

    wrong_job = {key: dict(value) for key, value in accounting.items()}
    wrong_job["1177669"]["comment"] = "crs2:wrong-token"
    with pytest.raises(ValueError, match="identity differs"):
        matrix.validate_accounting_rows(job_ids, wrong_job, expected_jobs=expected_jobs)

    failed = {key: dict(value) for key, value in accounting.items()}
    failed["1177669"]["exit_code"] = "1:0"
    with pytest.raises(ValueError, match="did not complete successfully"):
        matrix.validate_accounting_rows(job_ids, failed, expected_jobs=expected_jobs)


def test_matrix_validator_commit_cannot_be_injected():
    parameters = inspect.signature(matrix.build_training_matrix_completion).parameters
    assert "validation_code_commit" not in parameters
    assert "accounting" not in parameters
    validation_parameters = inspect.signature(
        matrix.validate_training_matrix_completion
    ).parameters
    assert "accounting" not in validation_parameters
    source = Path("tools/bata/finalize_continuous_roi_s2_training_matrix.py").read_text(
        encoding="utf-8"
    )
    assert "require_clean_git_checkout" in source
    assert 'git", "show' in source
    assert "VALIDATOR_SOURCE_PATHS" in source


def test_matrix_finalizer_is_training_only_and_fail_closed():
    source = Path("tools/bata/finalize_continuous_roi_s2_training_matrix.py").read_text(
        encoding="utf-8"
    )
    assert '"reference_sweep_completed": False' in source
    assert '"crop_sufficiency_established": False' in source
    assert '"official_test_open_allowed": False' in source
    assert '"official_test_runtime_access_audited": False' in source
    assert '"official_test_opened": None' in source
    assert '"paper_claim_allowed": False' in source
    assert "validate_training_completion_with_audit" in source
    assert "build_checkpoint_validation_runtime" in source
    assert '"all_checkpoints_strict_loaded_into_real_models": True' in source
    assert '"all_raw_buffer_dtype_casts_match_frozen_allowlist": True' in source
    assert "S2_ALLOWED_RAW_BUFFER_DTYPE_CAST_KEYS" in source
    training_source = Path("tools/bata/continuous_roi_s2_training.py").read_text(
        encoding="utf-8"
    )
    assert "generic_dtype_mismatches != strict_buffer_casts" in training_source
    assert '"bound_config_sha256"' in source
    assert "bound config differs from the submitted cell intent" in source
    assert "JobIDRaw,JobName,State,ExitCode,Elapsed,Comment" in source
    assert "expected-deployment-sha256" in source
    assert "weights_only=True" in Path(
        "tools/bata/continuous_roi_s2_training.py"
    ).read_text(encoding="utf-8")
