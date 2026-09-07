"""Successful-update checks shared by BA-FDR training and checkpoint consumers."""


def validate_optimizer_steps(optimizer_state, expected_updates):
    states = optimizer_state.get("state", {})
    steps = {int(state["step"]) for state in states.values() if "step" in state}
    if not steps or steps != {int(expected_updates)}:
        raise RuntimeError(
            f"BAFDR actual optimizer steps {sorted(steps)} != {expected_updates}"
        )


def validate_update_progress(optimizer_state, scheduler_state, audit, expected_updates):
    validate_optimizer_steps(optimizer_state, expected_updates)
    for key in ("successful_optimizer_updates", "scheduler_updates", "ema_updates"):
        if audit.get(key) != expected_updates:
            raise RuntimeError(f"BAFDR {key} != {expected_updates}")
    if scheduler_state.get("last_epoch") != expected_updates:
        raise RuntimeError(f"BAFDR actual scheduler step != {expected_updates}")


def validate_terminal_checkpoint(checkpoint):
    if checkpoint.get("epoch") != 59 or not checkpoint.get("state_dict_ema"):
        raise RuntimeError("BAFDR requires terminal epoch 59 state_dict_ema")
    if checkpoint.get("total_successful_updates") != 6000:
        raise RuntimeError("BAFDR terminal checkpoint requires 6000 successful updates")
    validate_update_progress(
        checkpoint.get("optimizer", {}), checkpoint.get("scheduler", {}),
        checkpoint.get("update_audit", {}), 6000,
    )
