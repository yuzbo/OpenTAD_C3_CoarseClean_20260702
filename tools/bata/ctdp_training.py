"""Successful-update accounting for the existing CT-DP training recipe."""


def validate_ctdp_training(cfg, loader_batches, world_size):
    workflow = cfg.workflow
    formal = bool(workflow.get("formal_successful_update_contract", True))
    epochs = int(workflow.end_epoch)
    limit = workflow.get("max_train_iters")
    batches = int(loader_batches) if limit is None else min(int(loader_batches), int(limit))
    if world_size != 1 or not cfg.solver.amp or not cfg.solver.ema:
        raise ValueError("CT-DP requires single-GPU AMP training with EMA")
    if formal and (epochs != 60 or batches != 100 or limit is not None):
        raise ValueError("formal CT-DP requires 60 complete epochs of 100 batches")
    if not formal and (epochs not in (1, 2) or limit is None or not 0 < batches <= 3):
        raise ValueError("CT-DP precheck must be one/two epochs of at most three batches")
    if int(workflow.get("max_amp_retries_per_batch", 0)) <= 0:
        raise ValueError("CT-DP must enable bounded same-batch AMP replay")
    return {"formal": formal, "epochs": epochs, "batches_per_epoch": batches,
            "expected_successful_optimizer_updates": epochs * batches}


def validate_ctdp_progress(contract, epoch, successful_updates, update_audit, scheduler_step):
    expected = (int(epoch) + 1) * contract["batches_per_epoch"]
    counts = [successful_updates, update_audit["successful_optimizer_updates"],
              update_audit["scheduler_updates"], update_audit["ema_updates"], scheduler_step]
    if any(int(value) != expected for value in counts):
        raise RuntimeError(f"CT-DP update mismatch: expected {expected}, observed {counts}")
    if epoch == contract["epochs"] - 1 and expected != contract["expected_successful_optimizer_updates"]:
        raise RuntimeError("CT-DP terminal update budget mismatch")


def validate_ctdp_resume(metadata, contract, commit, seed):
    if (metadata.get("route") != "CT-DP" or metadata.get("git_commit") != commit
            or metadata.get("seed") != seed or metadata.get("contract") != contract):
        raise ValueError("CT-DP resume requires the same code, seed and update contract")
    return dict(metadata["update_audit"])
