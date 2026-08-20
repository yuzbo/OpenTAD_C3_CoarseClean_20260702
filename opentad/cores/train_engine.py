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
    successful_update_index=None,
    max_amp_retries_per_batch=None,
):
    """Training the model for one epoch"""

    logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
    losses_tracker = {}
    num_iters = len(train_loader)
    use_amp = False if scaler is None else True
    retry_skipped_updates = max_amp_retries_per_batch is not None
    if retry_skipped_updates:
        if (
            not isinstance(max_amp_retries_per_batch, int)
            or max_amp_retries_per_batch <= 0
        ):
            raise ValueError("maximum AMP retries per batch must be a positive integer")
    else:
        max_amp_retries_per_batch = 0
    if successful_update_index is not None:
        if not retry_skipped_updates:
            raise ValueError("successful update indexing requires AMP retry semantics")
        if not isinstance(successful_update_index, int) or successful_update_index < 0:
            raise ValueError("successful update index must be a non-negative integer")
        georoute_backbone = model.module.backbone
        if not hasattr(georoute_backbone, "set_successful_update_index") or not hasattr(
            georoute_backbone, "consume_training_auxiliary_losses"
        ):
            raise ValueError("successful update indexing requires a GeoRoute backbone")
    else:
        georoute_backbone = None

    model.train()
    for iter_idx, data_dict in enumerate(train_loader):
        # current learning rate
        curr_backbone_lr = None
        if hasattr(model.module, "backbone"):  # if backbone exists
            if model.module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        max_attempts = 1 + max_amp_retries_per_batch
        optimizer_update_succeeded = False
        for attempt_idx in range(max_attempts):
            optimizer.zero_grad()

            # forward pass
            if georoute_backbone is not None:
                georoute_backbone.set_successful_update_index(successful_update_index)
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                losses = model(**data_dict, return_loss=True)
            if georoute_backbone is not None:
                auxiliary_losses = georoute_backbone.consume_training_auxiliary_losses(
                    masks=data_dict["masks"],
                    gt_segments=data_dict["gt_segments"],
                    gt_labels=data_dict["gt_labels"],
                )
                colliding_loss_keys = set(losses).intersection(auxiliary_losses)
                if colliding_loss_keys:
                    raise ValueError(
                        "GeoRoute auxiliary loss keys collide with detector losses: "
                        f"{sorted(colliding_loss_keys)}"
                    )
                auxiliary_cost = sum(auxiliary_losses.values())
                losses.update(auxiliary_losses)
                losses["cost"] = losses["cost"] + auxiliary_cost

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

            # update parameters
            if use_amp:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_update_succeeded = scaler.get_scale() >= scale_before
            else:
                optimizer.step()
                optimizer_update_succeeded = True
            if optimizer_update_succeeded or not retry_skipped_updates:
                break
        if retry_skipped_updates and not optimizer_update_succeeded:
            raise RuntimeError(
                "AMP optimizer update failed after "
                f"{max_amp_retries_per_batch} retries for epoch {curr_epoch}, "
                f"batch {iter_idx}"
            )
        if georoute_backbone is not None and optimizer_update_succeeded:
            successful_update_index += 1

        # update scheduler
        scheduler.step()

        # update ema
        if model_ema is not None:
            model_ema.update(model)

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
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))
    return successful_update_index


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
