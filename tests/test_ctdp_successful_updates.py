import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from mmengine.config import Config

from tools.bata.ctdp_training import (
    validate_ctdp_training, validate_ctdp_progress, validate_ctdp_resume,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arm", ["g0", "g1", "g2", "g3"])
def test_geometry_arms_enable_non_s1_successful_updates(arm):
    cfg = Config.fromfile(str(ROOT / f"configs/adatad/thumos/duca_ctdp_geometry_{arm}.py"))
    assert cfg.ctdp_training
    assert "spatial_zoom_s1_contract" not in cfg
    contract = validate_ctdp_training(cfg, 100, 1)
    assert contract["formal"] and contract["expected_successful_optimizer_updates"] == 6000
    assert cfg.scheduler.max_epoch == 100  # Preserve the original LR recipe.
    with pytest.raises(ValueError, match="100 batches"):
        validate_ctdp_training(cfg, 99, 1)
    with pytest.raises(ValueError, match="single-GPU"):
        validate_ctdp_training(cfg, 100, 2)


def test_precheck_cannot_be_mistaken_for_formal_training():
    cfg = Config.fromfile(str(ROOT / "configs/adatad/thumos/duca_ctdp_geometry_g0.py"))
    cfg.workflow.update(formal_successful_update_contract=False, end_epoch=2, max_train_iters=3)
    contract = validate_ctdp_training(cfg, 100, 1)
    assert contract["expected_successful_optimizer_updates"] == 6
    assert not contract["formal"]
    cfg.workflow.formal_successful_update_contract = True
    with pytest.raises(ValueError, match="100 batches"):
        validate_ctdp_training(cfg, 100, 1)


@pytest.mark.parametrize("missing_key", ["successful_optimizer_updates", "scheduler_updates", "ema_updates"])
def test_terminal_accounting_rejects_old_deficits(missing_key):
    contract = dict(epochs=60, batches_per_epoch=100, expected_successful_optimizer_updates=6000)
    audit = dict(successful_optimizer_updates=6000, scheduler_updates=6000, ema_updates=6000)
    validate_ctdp_progress(contract, 59, 6000, audit, 6000)
    audit[missing_key] = 5997
    with pytest.raises(RuntimeError, match="update mismatch"):
        validate_ctdp_progress(contract, 59, 6000, audit, 6000)


def test_resume_requires_bound_code_and_accounting():
    contract = dict(epochs=60, batches_per_epoch=100, expected_successful_optimizer_updates=6000)
    with pytest.raises(ValueError, match="same code"):
        validate_ctdp_resume({}, contract, "new", 3407)
    metadata = dict(route="CT-DP", git_commit="new", seed=3407, contract=contract,
                    update_audit=dict(successful_optimizer_updates=200))
    assert validate_ctdp_resume(metadata, contract, "new", 3407) == metadata["update_audit"]
    with pytest.raises(ValueError, match="same code"):
        validate_ctdp_resume(metadata, contract, "old", 3407)


def test_train_entrypoint_passes_ctdp_replay_and_saves_recovery_state():
    text = (ROOT / "tools/train.py").read_text(encoding="utf-8")
    assert "strict_updates = s1_binding is not None or ctdp_run" in text
    assert "fail_on_skipped_update=strict_updates" in text
    assert "max_amp_retries_per_batch=amp_retry_limit" in text
    assert 'experiment_sidecar_schema="ctdp_training_v1"' in text
    assert 'scaler.load_state_dict(state["scaler"])' in text


@pytest.mark.skipif(os.name == "nt", reason="CUDA witness runs in the N16R4 Torch environment")
def test_cuda_amp_replay_counts_real_optimizer_scheduler_and_ema(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA allocation required")
    from opentad.cores.train_engine import train_one_epoch
    from opentad.utils.checkpoint import save_checkpoint

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5, device="cuda"))
            self.register_buffer("visits", torch.zeros((), device="cuda"))
            self.noises = []

        def forward(self, x, return_loss=True):
            self.visits.add_(1)
            noise = torch.rand_like(x)
            self.noises.append(noise.detach().clone())
            return {"cost": ((x + noise) * self.weight).square().mean()}

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = Toy()

        def forward(self, **kwargs):
            return self.module(**kwargs)

    torch.manual_seed(3407)
    model = Wrapped()
    calls = []
    def overflow_once(grad):
        calls.append(1)
        return torch.full_like(grad, float("inf")) if len(calls) == 1 else grad
    hook = model.module.weight.register_hook(overflow_once)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.cuda.amp.GradScaler(init_scale=128)
    ema_calls = []
    ema = SimpleNamespace(update=lambda _: ema_calls.append(1), module=model)
    audit = {}
    updates = train_one_epoch(
        [{"x": torch.ones(4, device="cuda")} for _ in range(2)],
        model, optimizer, scheduler, 0, logging.getLogger(__name__),
        model_ema=ema, scaler=scaler, clip_grad_l2norm=1.0,
        fail_on_skipped_update=True, max_amp_retries_per_batch=8, update_audit=audit,
    )
    hook.remove()
    assert updates == scheduler.last_epoch == len(ema_calls) == 2
    assert int(optimizer.state[model.module.weight]["step"].item()) == 2
    assert audit["amp_skipped_attempts"] == 1
    assert audit["ema_updates"] == audit["scheduler_updates"] == 2
    assert model.module.visits.item() == 2
    assert torch.equal(model.module.noises[0], model.module.noises[1])
    save_checkpoint(model, ema, optimizer, scheduler, 0, work_dir=str(tmp_path),
                    experiment_metadata=dict(route="CT-DP", update_audit=audit),
                    experiment_sidecar_schema="ctdp_training_v1",
                    training_state={"scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state()})
    saved = torch.load(tmp_path / "checkpoint/epoch_0.pth", map_location="cpu", weights_only=False)
    assert saved["experiment_metadata"]["update_audit"] == audit
    assert saved["training_state"]["scaler"] == scaler.state_dict()
    assert saved["scheduler"]["last_epoch"] == 2
