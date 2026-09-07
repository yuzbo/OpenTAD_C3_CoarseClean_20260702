"""Single allocated GPU runner; no scheduler, server or metric-based promotion."""
import copy
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from opentad.datasets.builder import build_dataset, collate
from opentad.models.builder import build_detector
from opentad.cores.scheduler import build_scheduler
from . import detector as registration  # register the concrete model and transforms
from .records import save_json, load_json, file_id, checkpoint_identity, require_same_checkpoint
from .protocol import resolved_model_protocol


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(config, batch_size, workers, seed, training=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(build_dataset(copy.deepcopy(config)), batch_size=batch_size, shuffle=training,
                      drop_last=training, num_workers=workers, collate_fn=collate, pin_memory=True,
                      worker_init_fn=seed_worker, generator=generator)


def subset_batch(batch, begin, end):
    return {key: value[begin:end] for key, value in batch.items()}


def require_gpu():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("this deployment uses Slurm: training/benchmark must run in an allocation")
    if os.environ.get("SLURM_JOB_PARTITION") == "a100x":
        allocated = os.environ.get("SLURM_JOB_GPUS", "")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not allocated.isdigit() or not visible.isdigit():
            raise RuntimeError("A100 deployment requires one Slurm-allocated GPU, with one visible device")
    elif os.environ.get("SLURM_JOB_GPUS") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("require Slurm physical GPU1 mapped to container GPU0 (approved 2026-09-07)")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one allocated CUDA device")


class ModelEMA:
    """Match upstream EMA arithmetic, including all weights and buffer dtypes."""
    def __init__(self, model):
        self.names = checkpoint_state_names(model)
        self.shadow = {n: p.detach().clone() for n, p in model.state_dict().items() if n in self.names}

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if name not in self.shadow:
                continue
            self.shadow[name].copy_(.999 * self.shadow[name] + .001 * value)

    def state_dict(self):
        return {n: value.detach().cpu() for n, value in self.shadow.items()}


def checkpoint_state_names(model):
    return set(model.state_dict())


def restore_mutable_state(model, checkpoint, key):
    """The model must already contain the provenance-checked recognition base."""
    if checkpoint.get("format") != "geosparse_full_state_v2":
        raise ValueError("checkpoint is not the registered full-state format")
    values = checkpoint[key]
    expected = checkpoint_state_names(model)
    if set(values) != expected:
        raise ValueError(f"checkpoint state differs: missing={sorted(expected - set(values))}, unexpected={sorted(set(values) - expected)}")
    complete = model.state_dict()
    complete.update(values)
    model.load_state_dict(complete, strict=True)


def optimizer_for(model, cfg):
    # Use the untouched ActionFormer grouping for its actual detector modules.
    detector_modules = torch.nn.Module()
    for name in ("projection", "neck", "rpn_head"):
        detector_modules.add_module(name, getattr(model, name))
    from opentad.models.detectors.actionformer import ActionFormer
    groups = ActionFormer.get_optim_groups(detector_modules, cfg)
    small = [(n, p) for n, p in model.geosparse.named_parameters()
             if p.requires_grad and not n.startswith("encoder.source.")]
    for decay in (True, False):
        groups.append(dict(params=[p for n, p in small if (p.ndim > 1 and not n.endswith("bias")) == decay],
                           lr=cfg.lr, weight_decay=cfg.weight_decay if decay else 0.))
    adapters = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("geosparse.encoder.source.")]
    adapter_ids = {id(p) for p in adapters}
    for group in groups:
        group["params"] = [p for p in group["params"] if id(p) not in adapter_ids]
    if adapters:
        groups.append(dict(params=adapters, lr=cfg.backbone.custom[0].lr,
                           weight_decay=cfg.backbone.custom[0].weight_decay))
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("optimizer must contain each trainable parameter exactly once")
    return optimizer


def save_checkpoint(path, model, ema, optimizer, scheduler, scaler, epoch, provenance, updates):
    names = checkpoint_state_names(model)
    state = dict(format="geosparse_full_state_v2", epoch=epoch, completed_epochs=epoch + 1,
                 state_dict={n: p.detach().cpu() for n, p in model.state_dict().items() if n in names},
                 state_dict_ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), grad_scaler=scaler.state_dict(),
                 minibatch=model.minibatch, successful_optimizer_updates=updates, provenance=provenance,
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                          cuda=torch.cuda.get_rng_state_all()))
    temporary = Path(path).with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train(job, cfg, output, provenance, runtime):
    from .training_validation import periodic_validation, update_best_checkpoint, repair_missing_validations

    require_gpu()
    if job["epochs"] != 60 or job["seed"] not in {0, 1, 2}:
        raise ValueError("registered training requires 60 epochs and seed 0/1/2")
    effective, micro = runtime["effective_batch"], runtime["microbatch"]
    if effective % micro:
        raise ValueError("microbatch must divide the fixed effective batch")
    seed_all(job["seed"])
    data = loader(cfg.dataset.train, effective, runtime["num_workers"], job["seed"], training=True)
    if not len(data):
        raise ValueError("training split is smaller than the effective batch")
    model = build_detector(cfg.model).cuda()
    optimizer = optimizer_for(model, cfg.optimizer)
    scheduler, _ = build_scheduler(copy.deepcopy(cfg.scheduler), optimizer, len(data))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.solver.amp)
    ema = ModelEMA(model)
    checkpoint_dir = Path(output) / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)
    last = checkpoint_dir / "last.pth"
    start, updates = 0, 0
    if last.exists():
        saved = torch.load(last, map_location="cpu")
        if saved["provenance"] != provenance:
            raise ValueError("resume source, configuration, split or initialization differs")
        restore_mutable_state(model, saved, "state_dict")
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["grad_scaler"])
        ema.shadow = {n: saved["state_dict_ema"][n].cuda() for n in ema.names}
        model.minibatch = saved["minibatch"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        start, updates = saved["epoch"] + 1, saved["successful_optimizer_updates"]
        model.epoch = saved["epoch"]
    # Catch up if an allocation ended after checkpointing but before validation.
    validation = periodic_validation(model, ema, cfg, runtime, job, provenance, output, start)
    update_best_checkpoint(validation, last, output)
    with open(Path(output) / "train.log", "a", buffering=1) as log, open(Path(output) / "cost_trace.jsonl", "a", buffering=1) as trace:
        for epoch in range(start, job["epochs"]):
            model.train()
            model.epoch = epoch
            data.generator.manual_seed(job["seed"] * 1000 + epoch)
            for step, batch in enumerate(data):
                if epoch == start and step == 0:
                    save_json(Path(output) / "resolved_protocol.json", resolved_model_protocol(model, batch["inputs"].shape))
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                aggregate = {}
                for begin in range(0, effective, micro):
                    with torch.cuda.amp.autocast(enabled=cfg.solver.amp):
                        losses = model(**subset_batch(batch, begin, begin + micro), return_loss=True)
                        loss = losses["cost"] * micro / effective
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite loss at epoch={epoch} batch={step}")
                    scaler.scale(loss).backward()
                    for name, value in losses.items():
                        aggregate[name] = aggregate.get(name, 0.) + float(value.detach()) * micro / effective
                    if step == 0 or model.latest_acquisition is not None:
                        encoder = model.geosparse.encoder
                        trace.write(json.dumps(dict(epoch=epoch, step=step, microbatch=begin // micro,
                            video_ids=[m["video_name"] for m in batch["metas"][begin:begin + micro]],
                            layers=[] if encoder is None else encoder.trace,
                            requested_budget=model.latest_route_plan.requested_budget.tolist(),
                            selected_native_members=model.latest_route_plan.realized_token_count.tolist(),
                            heavy_tokens_by_layer={} if encoder is None else {
                                str(layer): sum(row["qkv_tokens"] for row in encoder.trace if row["layer"] == layer)
                                for layer in sorted({row["layer"] for row in encoder.trace})},
                            acquisition=model.latest_acquisition)) + "\n")
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.solver.clip_grad_norm)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                successful = scaler.get_scale() >= previous_scale
                if successful:
                    model.after_optimizer_step()
                    scheduler.step()
                    ema.update(model)
                    updates += 1
                else:
                    model.pending_cost.clear()
                row = dict(epoch=epoch, step=step, losses=aggregate, successful_update=successful,
                           gradient_norm=float(gradient_norm) if math.isfinite(float(gradient_norm)) else None,
                           gradient_status="finite" if torch.isfinite(gradient_norm) else "AMP_overflow",
                           lr=optimizer.param_groups[0]["lr"],
                           seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 20 == 0:
                    print(json.dumps(row), flush=True)
            save_checkpoint(last, model, ema, optimizer, scheduler, scaler, epoch, provenance, updates)
            if epoch in job["checkpoints"] or (epoch + 1) % 5 == 0:
                import shutil
                shutil.copyfile(last, checkpoint_dir / f"epoch_{epoch}.pth")
            save_json(Path(output) / "progress.json", dict(job_id=job["job_id"], status="RUNNING",
                      completed_epochs=epoch + 1, successful_optimizer_updates=updates, checkpoint_path=str(last)))
            validation = periodic_validation(model, ema, cfg, runtime, job, provenance, output, epoch + 1)
            update_best_checkpoint(validation, last, output)
            if validation is not None:
                log.write(json.dumps(dict(event="full_validation", **validation), allow_nan=False) + "\n")
                save_json(Path(output) / "progress.json", dict(job_id=job["job_id"], status="RUNNING",
                          completed_epochs=epoch + 1, successful_optimizer_updates=updates, checkpoint_path=str(last),
                          last_validation={key: validation[key] for key in ("status", "completed_epochs", "seconds")},
                          validation_metrics=validation.get("metrics"),
                          best=load_json(Path(output) / "best.json") if (Path(output) / "best.json").exists() else None))
    final = checkpoint_dir / "epoch_59.pth"
    if not final.is_file():
        raise RuntimeError("final checkpoint was not produced")
    missing = repair_missing_validations(model, ema, cfg, runtime, job, provenance, output)
    best_path = Path(output) / "best.json"
    best = load_json(best_path) if best_path.exists() else {}
    return dict(completed_epochs=60, training_complete=True, selection_complete=not missing,
                checkpoint_path=best.get("checkpoint_path"),
                checkpoint_identity=best.get("checkpoint_identity"),
                last_checkpoint_path=str(final.resolve()), selected_checkpoint_epoch=best.get("checkpoint_epoch"),
                checkpoint_selection="best_full_validation_average_mAP", best_validation_score=best.get("score"),
                best_checkpoint_selection_complete=not missing, successful_optimizer_updates=updates,
                intermediate_validation_path=str(Path(output) / "intermediate_eval"),
                failed_validation_epochs=missing)


def load_trained_model(job, cfg, output, provenance):
    dependency = Path(job["dependency_outputs"][job["depends_on"][0]])
    receipt = load_json(dependency / "result.json")
    if receipt["status"] != "completed" or receipt["is_mock"] or receipt["completed_epochs"] != 60:
        raise ValueError("dependency is not its completed real training run")
    if not receipt.get("best_checkpoint_selection_complete") or not receipt.get("selection_complete"):
        raise ValueError("dependency requires complete scheduled best checkpoint selection")
    checkpoint = torch.load(receipt["checkpoint_path"], map_location="cpu")
    identity = checkpoint_identity(receipt["job_id"], checkpoint["epoch"], checkpoint["provenance"], file_id(receipt["checkpoint_path"]))
    require_same_checkpoint(identity, receipt.get("checkpoint_identity"))
    if checkpoint["epoch"] != receipt.get("selected_checkpoint_epoch", 59):
        raise ValueError("selected checkpoint epoch differs from its training receipt")
    for key in ("source_commit", "resolved_config_sha256", "split_sha256", "weights_sha256"):
        if checkpoint["provenance"][key] != provenance[key]:
            raise ValueError(f"dependency {key} differs from current run")
    model = build_detector(cfg.model).cuda().eval()
    restore_mutable_state(model, checkpoint, "state_dict_ema")
    model.epoch = checkpoint["epoch"]
    model.inference_seed = job["seed"]
    model.checkpoint_identity = identity
    return model


def merge_windows(results, nms):
    from opentad.models.utils.post_processing import batched_nms
    merged = {}
    for name, predictions in results.items():
        if not predictions:
            merged[name] = []
            continue
        labels = sorted(set(p["label"] for p in predictions))
        segments, scores, classes = batched_nms(torch.tensor([p["segment"] for p in predictions]),
            torch.tensor([p["score"] for p in predictions]),
            torch.tensor([labels.index(p["label"]) for p in predictions]), **nms)
        merged[name] = [dict(segment=[round(float(t), 2) for t in segment], label=labels[int(label)],
                            score=round(float(score), 4)) for segment, score, label in zip(segments, scores, classes)]
    return merged
