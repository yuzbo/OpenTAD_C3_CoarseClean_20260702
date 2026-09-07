from copy import deepcopy
import io
import os
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata import duca_evidence_training as training


ROOT = Path(__file__).resolve().parents[1]


def resume_case(tmp_path):
    old_dir = tmp_path / "original" / "gpu1_id0"
    config = {"work_dir": str(old_dir), "seed": 8261, "scheduler": {"max_epoch": 60}}
    counters = training.new_update_audit()
    counters.update(
        attempted_batches=4000,
        optimizer_attempts=4000,
        successful_optimizer_updates=4000,
        scheduler_updates=4000,
        ema_updates=4000,
    )
    contract = training.formal_training_contract(Config.fromfile(
        ROOT / "configs/adatad/thumos/duca_evidence_recovery_no_coverage.py"
    ))
    old_bindings = dict(
        git_commit="a" * 40,
        slurm_job_id="1274924",
        arm_id="A1",
        seed=8261,
        source_config_path=str(tmp_path / "old_source" / "arm.py"),
        source_config_sha256="b" * 64,
        resolved_config_sha256="c" * 64,
        runtime_config_sha256=training.canonical_sha256(config),
        h65_ledgers={"train": {"path": "/original/ledger", "sha256": "d" * 64}},
    )
    audit = dict(
        old_bindings,
        schema_version=training.DUCA_P0_TRAINING_AUDIT_SCHEMA,
        checkpoint_criterion=contract["checkpoint_criterion"],
        last_completed_epoch=39,
        update_audit=counters,
        epoch_records=[{"epoch": epoch} for epoch in range(40)],
    )
    audit["audit_sha256"] = training.canonical_sha256(audit)
    checkpoint = dict(epoch=39, experiment_metadata=training.build_checkpoint_metadata(audit))
    config["work_dir"] = str(tmp_path / "resumed" / "gpu1_id0")
    bindings = dict(old_bindings)
    bindings.update(
        git_commit="e" * 40,
        slurm_job_id="1277000",
        source_config_path=str(tmp_path / "new_source" / "arm.py"),
        runtime_config_sha256=training.canonical_sha256(config),
    )
    return checkpoint, dict(
        contract=contract,
        bindings=bindings,
        checkpoint_path=old_dir / "checkpoint" / "epoch_39.pth",
        runtime_config=config,
        source_commit="a" * 40,
    )


def test_relocated_resume_keeps_original_counters_and_provenance(tmp_path):
    checkpoint, kwargs = resume_case(tmp_path)
    original = deepcopy(checkpoint)
    counters, records, bindings = training.restore_relocated_training_state(checkpoint, **kwargs)
    assert checkpoint == original
    assert counters["successful_optimizer_updates"] == 4000
    assert len(records) == 40
    assert bindings["git_commit"] == "e" * 40
    assert bindings["resume_precheck"] is False
    assert bindings["runtime_config_sha256"] == kwargs["bindings"]["runtime_config_sha256"]
    assert bindings["resume_lineage"][0]["git_commit"] == "a" * 40
    assert bindings["resume_lineage"][0]["successful_optimizer_updates"] == 4000


@pytest.mark.parametrize("key", ["seed", "source_config_sha256", "resolved_config_sha256", "h65_ledgers"])
def test_relocated_resume_rejects_changed_scientific_identity(tmp_path, key):
    checkpoint, kwargs = resume_case(tmp_path)
    kwargs["bindings"][key] = "changed"
    with pytest.raises(RuntimeError, match="resume binding mismatch"):
        training.restore_relocated_training_state(checkpoint, **kwargs)


def test_relocated_resume_rejects_changed_runtime_recipe(tmp_path):
    checkpoint, kwargs = resume_case(tmp_path)
    kwargs["runtime_config"]["scheduler"]["max_epoch"] = 100
    with pytest.raises(RuntimeError, match="runtime_config_sha256"):
        training.restore_relocated_training_state(checkpoint, **kwargs)


def test_relocated_resume_requires_explicit_original_commit(tmp_path):
    checkpoint, kwargs = resume_case(tmp_path)
    kwargs["source_commit"] = "f" * 40
    with pytest.raises(RuntimeError, match="source commit mismatch"):
        training.restore_relocated_training_state(checkpoint, **kwargs)


def test_relocated_resume_preserves_old_output_directory(tmp_path):
    checkpoint, kwargs = resume_case(tmp_path)
    kwargs["runtime_config"]["work_dir"] = str(kwargs["checkpoint_path"].parent.parent)
    with pytest.raises(RuntimeError, match="preserve the old work_dir"):
        training.restore_relocated_training_state(checkpoint, **kwargs)


def test_relocated_resume_requires_matching_checkpoint_epoch(tmp_path):
    checkpoint, kwargs = resume_case(tmp_path)
    checkpoint["epoch"] = 29
    with pytest.raises(RuntimeError, match="checkpoint epoch mismatch"):
        training.restore_relocated_training_state(checkpoint, **kwargs)


def test_resume_precheck_cannot_become_a_formal_parent(tmp_path):
    checkpoint, kwargs = resume_case(tmp_path)
    _, _, bindings = training.restore_relocated_training_state(checkpoint, **kwargs, precheck=True)
    assert bindings["resume_precheck"] is True
    audit = dict(checkpoint["experiment_metadata"]["training_audit"])
    audit.pop("audit_sha256")
    audit["resume_precheck"] = True
    audit["audit_sha256"] = training.canonical_sha256(audit)
    checkpoint["experiment_metadata"] = training.build_checkpoint_metadata(audit)
    with pytest.raises(RuntimeError, match="cannot resume a PRECHECK"):
        training.restore_relocated_training_state(checkpoint, **kwargs)


def test_existing_strict_resume_does_not_silently_accept_relocation(tmp_path):
    checkpoint, kwargs = resume_case(tmp_path)
    with pytest.raises(RuntimeError, match="git_commit"):
        training.restore_training_state(checkpoint, contract=kwargs["contract"], bindings=kwargs["bindings"])


def test_epoch_boundary_resume_reproduces_two_worker_loader_prefix():
    if os.name == "nt":
        pytest.skip("Windows torch/c10.dll unavailable; run the loader witness on N16R4")
    import random
    import numpy as np
    import torch
    from opentad.datasets import build_dataloader

    class RandomWindowDataset:
        def __len__(self):
            return 30

        def __getitem__(self, index):
            return {"inputs": torch.tensor([index, random.random(), np.random.rand(), torch.rand(()).item()])}

    def make_loader():
        return build_dataloader(RandomWindowDataset(), batch_size=2, rank=0,
                                world_size=1, shuffle=True, drop_last=True, num_workers=2)

    def prefix(loader, epoch):
        loader.sampler.set_epoch(epoch)
        batches = []
        for index, batch in enumerate(loader):
            batches.append(batch["inputs"])
            torch.rand(5)  # Stand in for main-process stochastic model work.
            if index == 2:
                break
        return batches

    random.seed(8261)
    np.random.seed(8261)
    torch.manual_seed(8261)
    loader = make_loader()
    prefix(loader, 39)
    serialized = io.BytesIO()
    torch.save(training.capture_global_rng_state(), serialized)
    expected = prefix(loader, 40)
    expected_rng = torch.get_rng_state()

    resumed_loader = make_loader()
    torch.rand(100)
    serialized.seek(0)
    training.restore_global_rng_state(torch.load(serialized, map_location="cpu", weights_only=False))
    observed = prefix(resumed_loader, 40)
    for left, right in zip(expected, observed):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
