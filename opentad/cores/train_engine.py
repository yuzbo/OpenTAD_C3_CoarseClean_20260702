import copy
import torch
import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss


def _capture_model_buffers(model):
    return {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if buffer is not None
    }


def _restore_model_buffers(model, snapshot):
    current = {
        name: buffer for name, buffer in model.named_buffers() if buffer is not None
    }
    if set(current) != set(snapshot):
        raise RuntimeError("model buffer registry changed during an AMP retry")
    for name, saved in snapshot.items():
        current[name].copy_(saved)


def _set_successful_update_index(model, index, *, required):
    unwrapped = getattr(model, "module", model)
    backbone = getattr(unwrapped, "backbone", None)
    callback = getattr(backbone, "set_successful_update_index", None)
    if callback is None:
        if required:
            raise RuntimeError(
                "training protocol requires backbone.set_successful_update_index"
            )
        return False
    callback(int(index))
    return True


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
    fail_on_skipped_update=False,
    max_amp_retries_per_batch=0,
    update_audit=None,
    successful_update_start=0,
    require_successful_update_hook=False,
    schedule_and_ema_on_success_only=False,
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
        raise ValueError("AMP retries require a GradScaler")
    successful_update_start = int(successful_update_start)
    if successful_update_start < 0:
        raise ValueError("successful_update_start must be non-negative")
    if update_audit is not None:
        update_audit.setdefault("optimizer_attempts", 0)
        update_audit.setdefault("amp_skipped_attempts", 0)
        update_audit.setdefault("max_amp_retries_observed", 0)
    use_amp = False if scaler is None else True

    model.train()
    successful_updates = 0
    for iter_idx, data_dict in enumerate(train_loader):
        _set_successful_update_index(
            model,
            successful_update_start + successful_updates,
            required=bool(require_successful_update_hook),
        )
        # current learning rate
        curr_backbone_lr = None
        if hasattr(model.module, "backbone"):  # if backbone exists
            if model.module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        retry_count = 0
        cpu_rng_state = None
        cuda_rng_states = None
        model_buffer_state = None
        if max_amp_retries_per_batch > 0:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all()
            model_buffer_state = _capture_model_buffers(model)

        while True:
            if retry_count > 0:
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state_all(cuda_rng_states)
            optimizer.zero_grad()

            # Replaying a skipped AMP attempt restores the stochastic state, so the
            # successful update still corresponds to this exact sampled batch.
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                losses = model(**data_dict, return_loss=True)
            if max_amp_retries_per_batch > 0 and not bool(
                torch.isfinite(losses["cost"]).all()
            ):
                raise FloatingPointError(
                    "S1 produced a non-finite loss before AMP scaling"
                )

            if use_amp:
                scaler.scale(losses["cost"]).backward()
            else:
                losses["cost"].backward()

            if clip_grad_l2norm > 0.0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

            if update_audit is not None:
                update_audit["optimizer_attempts"] += 1
            if use_amp:
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                update_succeeded = scale_after >= scale_before
            else:
                optimizer.step()
                update_succeeded = True

            if update_succeeded:
                break
            if update_audit is not None:
                update_audit["amp_skipped_attempts"] += 1
            if model_buffer_state is not None:
                _restore_model_buffers(model, model_buffer_state)
            retry_count += 1
            if update_audit is not None:
                update_audit["max_amp_retries_observed"] = max(
                    update_audit["max_amp_retries_observed"], retry_count
                )
            if retry_count > max_amp_retries_per_batch:
                if fail_on_skipped_update:
                    raise FloatingPointError(
                        "S1 AMP could not produce a successful optimizer update "
                        f"after {max_amp_retries_per_batch} retries"
                    )
                break
            logger.info(
                "[Train]: AMP skipped batch %d; retry %d/%d with scale %.1f",
                iter_idx,
                retry_count,
                max_amp_retries_per_batch,
                scale_after,
            )

        successful_updates += int(update_succeeded)

        # Registered protocols may advance optimization state only on a real
        # optimizer update. The default preserves legacy OpenTAD behavior.
        if update_succeeded or not schedule_and_ema_on_success_only:
            scheduler.step()

        # update ema
        if model_ema is not None and (
            update_succeeded or not schedule_and_ema_on_success_only
        ):
            model_ema.update(model)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or (
            (iter_idx + 1) == num_iters
        ):
            # print to terminal
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(
                curr_epoch, iter_idx, num_iters - 1
            )
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = [
                "{:s}={:.4f}".format(key, value.avg)
                for key, value in losses_tracker.items()
                if key != "cost"
            ]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            block5 = "mem={:.0f}MB".format(
                torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            )
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))
        if max_train_iters is not None and (iter_idx + 1) >= max_train_iters:
            logger.info(
                "[Train]: max_train_iters=%d reached; ending smoke epoch early",
                max_train_iters,
            )
            break
    return successful_updates


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
    block3 = [
        "{:s}={:.4f}".format(key, value.avg)
        for key, value in losses_tracker.items()
        if key != "cost"
    ]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
