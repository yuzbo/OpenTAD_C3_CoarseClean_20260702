import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss


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
    use_amp = False if scaler is None else True
    probe_state = _new_training_probe_state(model) if collect_training_probe else None

    model.train()
    for iter_idx, data_dict in enumerate(train_loader):
        optimizer.zero_grad()

        # current learning rate
        curr_backbone_lr = None
        if hasattr(model.module, "backbone"):  # if backbone exists
            if model.module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            losses = model(**data_dict, return_loss=True)

        # compute the gradients
        if use_amp:
            scaler.scale(losses["cost"]).backward()
        else:
            losses["cost"].backward()

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

        if probe_state is not None:
            _record_training_probe_backward(probe_state, model, losses["cost"])

        # update parameters
        optimizer_step_ran = True
        if use_amp:
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler silently skips optimizer.step() on non-finite grads and
            # lowers the scale. DUCA schedule/dual hooks must track real updates.
            optimizer_step_ran = scaler.get_scale() >= scale_before_step
        else:
            optimizer.step()

        if optimizer_step_ran:
            _call_after_optimizer_step(model)

            # update scheduler
            scheduler.step()

            # update ema
            if model_ema is not None:
                model_ema.update(model)

        if probe_state is not None:
            _record_training_probe_step(probe_state, model, optimizer_step_ran)

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
    if probe_state is not None:
        return _finalize_training_probe(probe_state)
    return None


def _parameter_probe_group(name):
    normalized = name.removeprefix("module.")
    if normalized.startswith("backbone."):
        return "backbone"
    if normalized.startswith("frame_selector.actionness_source"):
        return "coarse_probe"
    if normalized.startswith("frame_selector."):
        return "selector"
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
    if gradients_finite:
        state["finite_gradient_steps"] += 1


def _record_training_probe_step(state, model, optimizer_step_ran):
    if optimizer_step_ran:
        state["successful_optimizer_steps"] += 1
    else:
        state["skipped_optimizer_steps"] += 1
    state["selector_steps"].append(_selector_probe_snapshot(model))


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
    counterfactual = getattr(selector, "last_counterfactual_summary", None)
    if isinstance(counterfactual, dict):
        snapshot["counterfactual"] = {
            key: _probe_jsonable(counterfactual[key])
            for key in ("candidate_count", "finite", "teacher_kind")
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
    targets.append(model)
    seen = set()
    for target in targets:
        target_id = id(target)
        if target_id in seen:
            continue
        seen.add(target_id)
        hook = getattr(target, "after_optimizer_step", None)
        if callable(hook):
            hook()
            return


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
            for key in ("actionness", "detector_utility", "hole", "lagrangian_budget"):
                if key in weights:
                    items.append("duca_{}_w={:.4f}".format(key, float(weights[key])))
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
