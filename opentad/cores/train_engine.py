import copy
import json
import os
import random

import numpy as np
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss


def _capture_model_buffers(model):
    return {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if buffer is not None
    }


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(snapshot):
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def _restore_model_buffers(model, snapshot):
    current = {
        name: buffer for name, buffer in model.named_buffers() if buffer is not None
    }
    if set(current) != set(snapshot):
        raise RuntimeError("model buffer registry changed during an AMP replay")
    with torch.no_grad():
        for name, saved in snapshot.items():
            current[name].copy_(saved)


def _capture_custom_replay_state(model):
    snapshots = {}
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return snapshots
    for name, module in named_modules():
        capture = getattr(module, "capture_amp_replay_state", None)
        if callable(capture):
            snapshots[name] = capture()
    return snapshots


def _restore_custom_replay_state(model, snapshots):
    if not snapshots:
        return
    modules = dict(model.named_modules())
    if not set(snapshots).issubset(modules):
        missing = sorted(set(snapshots) - set(modules))
        raise RuntimeError(f"model module registry changed during an AMP replay: {missing}")
    for name, snapshot in snapshots.items():
        restore = getattr(modules[name], "restore_amp_replay_state", None)
        if not callable(restore):
            raise RuntimeError(f"AMP replay state provider lost its restore hook: {name}")
        restore(snapshot)


def _inject_nonfinite_gradient(model):
    for parameter in model.parameters():
        if parameter.grad is None or parameter.grad.numel() == 0:
            continue
        parameter.grad.detach().reshape(-1)[0] = float("inf")
        return
    raise RuntimeError("forced AMP overflow could not find a populated gradient")


def _write_update_audit_snapshot(
    path,
    *,
    curr_epoch,
    batch_index,
    event,
    update_audit,
):
    if not path or update_audit is None:
        return
    target = os.path.abspath(os.path.expanduser(str(path)))
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = target + ".tmp"
    payload = {
        "schema_version": "opentad_training_update_audit_v1",
        "epoch": int(curr_epoch),
        "batch_index": int(batch_index),
        "event": str(event),
        "update_audit": dict(update_audit),
    }
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    scaler=None,
    max_train_iters=None,
    collect_training_probe=False,
    max_amp_retries_per_batch=0,
    max_nonfinite_loss_retries=0,
    fail_on_amp_replay_exhaustion=False,
    require_finite_loss=False,
    force_amp_overflow_attempts=0,
    update_audit=None,
    update_audit_json=None,
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    if max_train_iters is not None:
        max_train_iters = int(max_train_iters)
        if max_train_iters <= 0:
            raise ValueError("max_train_iters must be positive when provided")
        num_iters = min(num_iters, max_train_iters)
    max_amp_retries_per_batch = int(max_amp_retries_per_batch)
    if max_amp_retries_per_batch < 0:
        raise ValueError("max_amp_retries_per_batch must be non-negative")
    if max_amp_retries_per_batch > 0 and scaler is None:
        raise ValueError("AMP replay requires a GradScaler")
    max_nonfinite_loss_retries = int(max_nonfinite_loss_retries)
    if max_nonfinite_loss_retries < 0:
        raise ValueError("max_nonfinite_loss_retries must be non-negative")
    if max_nonfinite_loss_retries > 0 and not require_finite_loss:
        raise ValueError("non-finite loss replay requires require_finite_loss")
    if max_nonfinite_loss_retries > 0 and update_audit is None:
        raise ValueError("non-finite loss replay requires an update audit")
    if fail_on_amp_replay_exhaustion and max_amp_retries_per_batch <= 0:
        raise ValueError("fail_on_amp_replay_exhaustion requires a positive replay limit")
    force_amp_overflow_attempts = int(force_amp_overflow_attempts)
    if force_amp_overflow_attempts < 0:
        raise ValueError("force_amp_overflow_attempts must be non-negative")
    if force_amp_overflow_attempts > 0 and (scaler is None or update_audit is None):
        raise ValueError("forced AMP overflow requires a GradScaler and update audit")
    if update_audit is not None:
        for key in (
            "attempted_batches",
            "optimizer_attempts",
            "successful_optimizer_updates",
            "amp_skipped_attempts",
            "replayed_batches",
            "replay_exhaustions",
            "scheduler_updates",
            "ema_updates",
            "duca_schedule_updates",
            "forced_amp_overflow_attempts",
        ):
            update_audit.setdefault(key, 0)
        update_audit.setdefault("max_amp_retries_observed", 0)
        if max_nonfinite_loss_retries > 0:
            for key in (
                "nonfinite_loss_attempts",
                "nonfinite_loss_replays",
                "nonfinite_loss_replay_exhaustions",
            ):
                update_audit.setdefault(key, 0)
            update_audit.setdefault("max_nonfinite_loss_retries_observed", 0)
            update_audit.setdefault("replay_state_restorations", 0)
        if collect_training_probe:
            update_audit.setdefault("replay_state_restorations", 0)
            update_audit.setdefault("attempt_batch_indices", [])
            update_audit.setdefault("amp_scale_history", [])
    use_amp = False if scaler is None else True
    probe_state = _new_training_probe_state(model) if collect_training_probe else None

    model.train()
    for iter_idx, data_dict in enumerate(train_loader):
        # current learning rate
        curr_backbone_lr = None
        root_model = getattr(model, "module", model)
        if hasattr(root_model, "backbone"):  # if backbone exists
            if root_model.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        retry_count = 0
        nonfinite_loss_retry_count = 0
        rng_state = None
        model_buffer_state = None
        custom_replay_state = None
        if max_amp_retries_per_batch > 0 or max_nonfinite_loss_retries > 0:
            rng_state = _capture_rng_state()
            model_buffer_state = _capture_model_buffers(model)
            custom_replay_state = _capture_custom_replay_state(model)
        if update_audit is not None:
            update_audit["attempted_batches"] += 1

        while True:
            if retry_count > 0 or nonfinite_loss_retry_count > 0:
                _restore_rng_state(rng_state)
            optimizer.zero_grad()

            try:
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                    losses = model(**data_dict, return_loss=True)
            except FloatingPointError as exc:
                # Preserve the original exception while attaching the execution
                # coordinates needed to reproduce the first invalid tensor.
                amp_scale = float(scaler.get_scale()) if scaler is not None else None
                context = (
                    f"; epoch={curr_epoch}; batch_index={iter_idx}; "
                    f"amp_scale={amp_scale}; rng_initial_seed={torch.initial_seed()}"
                )
                _write_update_audit_snapshot(
                    update_audit_json,
                    curr_epoch=curr_epoch,
                    batch_index=iter_idx,
                    event="nonfinite_tensor",
                    update_audit={
                        **(update_audit or {}),
                        "error": f"{exc}{context}",
                        "amp_scale": amp_scale,
                        "rng_initial_seed": int(torch.initial_seed()),
                    },
                )
                raise FloatingPointError(f"{exc}{context}") from exc
            if require_finite_loss and not bool(torch.isfinite(losses["cost"]).all().item()):
                if max_nonfinite_loss_retries <= 0:
                    raise FloatingPointError(
                        "formal training produced a non-finite loss before AMP scaling"
                    )
                update_audit["nonfinite_loss_attempts"] += 1
                if nonfinite_loss_retry_count >= max_nonfinite_loss_retries:
                    update_audit["nonfinite_loss_replay_exhaustions"] += 1
                    if model_buffer_state is not None:
                        _restore_model_buffers(model, model_buffer_state)
                    if custom_replay_state is not None:
                        _restore_custom_replay_state(model, custom_replay_state)
                    _restore_rng_state(rng_state)
                    if "replay_state_restorations" in update_audit:
                        update_audit["replay_state_restorations"] += 1
                    _write_update_audit_snapshot(
                        update_audit_json,
                        curr_epoch=curr_epoch,
                        batch_index=iter_idx,
                        event="nonfinite_loss_replay_exhausted",
                        update_audit=update_audit,
                    )
                    raise FloatingPointError(
                        "formal training produced a non-finite loss after "
                        f"{max_nonfinite_loss_retries} bounded replays"
                    )
                if model_buffer_state is not None:
                    _restore_model_buffers(model, model_buffer_state)
                if custom_replay_state is not None:
                    _restore_custom_replay_state(model, custom_replay_state)
                if "replay_state_restorations" in update_audit:
                    update_audit["replay_state_restorations"] += 1
                nonfinite_loss_retry_count += 1
                update_audit["nonfinite_loss_replays"] += 1
                update_audit["max_nonfinite_loss_retries_observed"] = max(
                    update_audit["max_nonfinite_loss_retries_observed"],
                    nonfinite_loss_retry_count,
                )
                _write_update_audit_snapshot(
                    update_audit_json,
                    curr_epoch=curr_epoch,
                    batch_index=iter_idx,
                    event="nonfinite_loss_replay",
                    update_audit=update_audit,
                )
                logger.info(
                    "[Train]: non-finite pre-AMP loss in batch %d; replay %d/%d",
                    iter_idx,
                    nonfinite_loss_retry_count,
                    max_nonfinite_loss_retries,
                )
                continue

            if use_amp:
                scaler.scale(losses["cost"]).backward()
            else:
                losses["cost"].backward()

            if (
                update_audit is not None
                and update_audit["forced_amp_overflow_attempts"]
                < force_amp_overflow_attempts
            ):
                _inject_nonfinite_gradient(model)
                update_audit["forced_amp_overflow_attempts"] += 1

            if clip_grad_l2norm > 0.0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

            if probe_state is not None:
                _record_training_probe_backward(probe_state, model, losses["cost"])
            if update_audit is not None and "attempt_batch_indices" in update_audit:
                update_audit["optimizer_attempts"] += 1
                update_audit["attempt_batch_indices"].append(int(iter_idx))
            elif update_audit is not None:
                update_audit["optimizer_attempts"] += 1

            optimizer_step_ran = True
            if use_amp:
                scale_before_step = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after_step = float(scaler.get_scale())
                optimizer_step_ran = scale_after_step >= scale_before_step
                if update_audit is not None and "amp_scale_history" in update_audit:
                    update_audit["amp_scale_history"].append(
                        {
                            "batch_index": int(iter_idx),
                            "retry_index": int(retry_count),
                            "before": scale_before_step,
                            "after": scale_after_step,
                            "optimizer_step_ran": bool(optimizer_step_ran),
                        }
                    )
            else:
                optimizer.step()

            if probe_state is not None:
                _record_training_probe_step(probe_state, model, optimizer_step_ran)
            if optimizer_step_ran:
                break

            if update_audit is not None:
                update_audit["amp_skipped_attempts"] += 1
            if model_buffer_state is not None:
                _restore_model_buffers(model, model_buffer_state)
            if custom_replay_state is not None:
                _restore_custom_replay_state(model, custom_replay_state)
            if update_audit is not None and "replay_state_restorations" in update_audit:
                update_audit["replay_state_restorations"] += 1
            retry_count += 1
            if update_audit is not None:
                update_audit["max_amp_retries_observed"] = max(
                    update_audit["max_amp_retries_observed"], retry_count
                )
            if retry_count > max_amp_retries_per_batch:
                if update_audit is not None:
                    update_audit["replay_exhaustions"] += 1
                if fail_on_amp_replay_exhaustion:
                    raise FloatingPointError(
                        "formal AMP replay could not produce a successful optimizer update "
                        f"after {max_amp_retries_per_batch} retries"
                    )
                break
            logger.info(
                "[Train]: AMP skipped batch %d; replay %d/%d with scale %.1f",
                iter_idx,
                retry_count,
                max_amp_retries_per_batch,
                float(scaler.get_scale()),
            )

        if (retry_count > 0 or nonfinite_loss_retry_count > 0) and update_audit is not None:
            update_audit["replayed_batches"] += 1

        if optimizer_step_ran:
            schedule_summary = _call_after_optimizer_step(model)
            scheduler.step()
            if model_ema is not None:
                model_ema.update(model)
            if update_audit is not None:
                update_audit["successful_optimizer_updates"] += 1
                update_audit["scheduler_updates"] += 1
                update_audit["ema_updates"] += int(model_ema is not None)
                if isinstance(schedule_summary, dict) and schedule_summary.get("updated") is True:
                    update_audit["duca_schedule_updates"] += 1

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
            # print to terminal
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(curr_epoch, iter_idx, num_iters - 1)
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            blocks = [block1, block2, "  ".join(block3), block4, block5]
            selector_diagnostics = _format_frame_selector_diagnostics(model)
            if selector_diagnostics:
                blocks.append(selector_diagnostics)
            logger.info("  ".join(blocks))
        if max_train_iters is not None and (iter_idx + 1) >= max_train_iters:
            logger.info("[Train]: max_train_iters=%d reached; ending smoke epoch early", max_train_iters)
            break
    _write_update_audit_snapshot(
        update_audit_json,
        curr_epoch=curr_epoch,
        batch_index=min(num_iters, iter_idx + 1),
        event="epoch_complete",
        update_audit=update_audit,
    )
    if probe_state is not None:
        return _finalize_training_probe(probe_state)
    return None


def _parameter_probe_group(name):
    normalized = name.removeprefix("module.")
    if normalized.startswith("backbone."):
        return "backbone"
    if normalized.startswith((
        "frame_selector.raw_actionness_source.",
        "frame_selector.actionness_source.",
    )):
        return "coarse_probe"
    if normalized.startswith("frame_selector."):
        return "selector"
    if normalized.startswith("projection."):
        return "projection"
    if normalized.startswith("neck."):
        return "neck"
    if normalized.startswith("rpn_head."):
        return "detector_head"
    return "other"


def _new_training_probe_state(model):
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        "attempted_steps": 0,
        "successful_optimizer_steps": 0,
        "skipped_optimizer_steps": 0,
        "finite_loss_steps": 0,
        "finite_gradient_steps": 0,
        "trainable_parameter_names": trainable,
        "gradient_seen_names": set(),
        "selector_steps": [],
    }


def _record_training_probe_backward(state, model, cost):
    state["attempted_steps"] += 1
    if bool(torch.isfinite(cost.detach()).all().item()):
        state["finite_loss_steps"] += 1
    grad_names = []
    gradients_finite = True
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad_names.append(name)
        if not bool(torch.isfinite(parameter.grad.detach()).all().item()):
            gradients_finite = False
    state["gradient_seen_names"].update(grad_names)
    if grad_names and gradients_finite:
        state["finite_gradient_steps"] += 1


def _record_training_probe_step(state, model, optimizer_step_ran):
    if optimizer_step_ran:
        state["successful_optimizer_steps"] += 1
    else:
        state["skipped_optimizer_steps"] += 1
    snapshot = _selector_probe_snapshot(model)
    snapshot["optimizer_step_ran"] = bool(optimizer_step_ran)
    state["selector_steps"].append(snapshot)


def _selector_probe_snapshot(model):
    module = getattr(model, "module", model)
    selector = getattr(module, "frame_selector", None)
    summary = getattr(selector, "last_forward_summary", None)
    snapshot = {}
    if isinstance(summary, dict):
        for key in (
            "requested_budget", "effective_budget", "detector_gradient_weight",
            "policy_mix_alpha", "selection_path", "selector_variant",
        ):
            if key in summary:
                snapshot[key] = _probe_jsonable(summary[key])
        schedule = summary.get("loss_weight_schedule")
        if isinstance(schedule, dict):
            snapshot["loss_weight_schedule"] = {
                key: _probe_jsonable(schedule[key])
                for key in ("step", "phase", "progress", "detector_gradient_weight", "weights")
                if key in schedule
            }
        supervision = summary.get("supervision_loss_audit")
        if isinstance(supervision, dict):
            snapshot["supervision_loss_audit"] = _probe_jsonable(supervision)
        if "frontend_only_detector_skipped" in summary:
            snapshot["frontend_only_detector_skipped"] = bool(
                summary["frontend_only_detector_skipped"]
            )
    counterfactual = getattr(selector, "last_counterfactual_summary", None)
    if isinstance(counterfactual, dict):
        snapshot["counterfactual"] = {
            key: _probe_jsonable(counterfactual[key])
            for key in (
                "candidate_count",
                "finite",
                "teacher_kind",
                "distillation_loss_kind",
                "candidate_utility_values",
                "candidate_cell_indices",
                "utility_alignment_informative",
            )
            if key in counterfactual
        }
    return snapshot


def _probe_jsonable(value):
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, dict):
        return {str(key): _probe_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_probe_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _finalize_training_probe(state):
    trainable = state["trainable_parameter_names"]
    seen = state["gradient_seen_names"]
    group_counts = {}
    for name in sorted(trainable):
        group = _parameter_probe_group(name)
        counts = group_counts.setdefault(group, {"trainable": 0, "gradient_seen": 0})
        counts["trainable"] += 1
        if name in seen:
            counts["gradient_seen"] += 1
    return {
        "schema_version": "duca_training_probe_v1",
        "attempted_steps": int(state["attempted_steps"]),
        "successful_optimizer_steps": int(state["successful_optimizer_steps"]),
        "skipped_optimizer_steps": int(state["skipped_optimizer_steps"]),
        "finite_loss_steps": int(state["finite_loss_steps"]),
        "finite_gradient_steps": int(state["finite_gradient_steps"]),
        "parameter_group_coverage": group_counts,
        "gradient_never_seen": sorted(trainable - seen),
        "gradient_seen": sorted(seen),
        "selector_steps": state["selector_steps"],
        "max_cuda_memory_mb": float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
    }


def _call_after_optimizer_step(model):
    module = getattr(model, "module", None)
    targets = []
    if module is not None:
        targets.append(module)
    root = module if module is not None else model
    frame_selector = getattr(root, "frame_selector", None)
    if frame_selector is not None:
        targets.append(frame_selector)
    targets.append(model)
    seen = set()
    for target in targets:
        target_id = id(target)
        if target_id in seen:
            continue
        seen.add(target_id)
        hook = getattr(target, "after_optimizer_step", None)
        if callable(hook):
            return hook()
    return None


def _format_frame_selector_diagnostics(model):
    module = getattr(model, "module", model)
    selector = getattr(module, "frame_selector", None)
    summary = getattr(selector, "last_forward_summary", None)
    if not isinstance(summary, dict):
        return ""
    items = []
    schedule = summary.get("loss_weight_schedule")
    if isinstance(schedule, dict):
        if "step" in schedule:
            items.append("duca_schedule_step={}".format(int(schedule["step"])))
        phase = schedule.get("phase")
        if phase:
            items.append("duca_phase={}".format(str(phase)))
        if "progress" in schedule:
            items.append("duca_schedule_progress={:.4f}".format(float(schedule["progress"])))
        if "detector_gradient_weight" in schedule:
            items.append("duca_detector_grad_w={:.4f}".format(float(schedule["detector_gradient_weight"])))
        weights = schedule.get("weights")
        if isinstance(weights, dict):
            for key in (
                "actionness",
                "transition",
                "transition_boundary",
                "policy_alpha",
                "detector_utility",
                "hole",
                "lagrangian_budget",
            ):
                if key in weights:
                    items.append("duca_{}_w={:.4f}".format(key, float(weights[key])))
    supervision = summary.get("supervision_loss_audit")
    if isinstance(supervision, dict):
        for loss_name, label in (
            ("actionness_bce_loss", "duca_action_raw"),
            ("transition_distribution_loss", "duca_transition_raw"),
            ("transition_boundary_coverage_loss", "duca_boundary_raw"),
        ):
            entry = supervision.get(loss_name)
            if isinstance(entry, dict) and entry.get("unweighted") is not None:
                items.append("{}={:.4f}".format(label, float(entry["unweighted"])))
    if summary.get("frontend_only_detector_skipped") is True:
        items.append("duca_detector_path=skipped")
    homotopy = summary.get("policy_homotopy")
    if isinstance(homotopy, dict) and homotopy.get("enabled") is True:
        if "step" in homotopy:
            items.append("duca_policy_step={}".format(int(homotopy["step"])))
        if "alpha" in homotopy:
            items.append("duca_policy_alpha={:.4f}".format(float(homotopy["alpha"])))
        if homotopy.get("phase"):
            items.append("duca_policy_phase={}".format(str(homotopy["phase"])))
    for key, label in (
        ("budget", "duca_budget"),
        ("dynamic_budget", "duca_dynamic_budget"),
        ("budget_policy", "duca_budget_policy"),
    ):
        if key in summary:
            items.append("{}={}".format(label, summary[key]))
    for key, label in (("requested_budget", "duca_requested_budget"), ("effective_budget", "duca_effective_budget")):
        value = summary.get(key)
        if isinstance(value, (list, tuple)) and value:
            try:
                mean_value = sum(float(item) for item in value) / len(value)
            except (TypeError, ValueError):
                continue
            items.append("{}_mean={:.2f}".format(label, mean_value))
    return "  ".join(items)


def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: compute the loss"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                losses = model(**data_dict, return_loss=True)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
