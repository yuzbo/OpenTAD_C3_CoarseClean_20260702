"""Run as a file: gradients of the next batch after an ordinary epoch checkpoint.

One frozen training state, one original next-epoch batch, no optimizer update.
This is not a reconstruction of an arbitrary historical minibatch.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys


def restore_probe_state(model, saved, completed_epochs, amp):
    """Select ordinary weights and resume counters/RNG, never EMA weights."""
    import numpy as np
    import torch
    from geosparse_ext.runtime import restore_mutable_state

    if not 1 <= completed_epochs < 60 or saved.get("epoch") != completed_epochs - 1:
        raise ValueError("probe requires its immutable completed epoch before epoch60")
    if saved.get("completed_epochs") != completed_epochs:
        raise ValueError("checkpoint completed-epoch fields disagree")
    scale = saved["grad_scaler"].get("scale") if amp else 1.
    if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("AMP probe requires the checkpoint GradScaler scale")
    restore_mutable_state(model, saved, "state_dict")
    model.epoch = completed_epochs
    model.minibatch = saved["minibatch"]
    model.train()
    random.setstate(saved["rng"]["python"])
    np.random.set_state(saved["rng"]["numpy"])
    torch.set_rng_state(saved["rng"]["torch"])
    if next(model.parameters()).is_cuda:
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
    return dict(weights="ordinary_state_dict", checkpoint_epoch=completed_epochs - 1,
                next_training_epoch=completed_epochs, next_completed_epoch=completed_epochs + 1,
                minibatch=model.minibatch, loss_scale=float(scale),
                loss_normalizer=float(model.rpn_head.loss_normalizer))


def ordinary_gradient_norm(model, batch, microbatch_size, amp, loss_scale):
    """Same source backward as runtime.py; unscaled norm without an update."""
    import torch
    from geosparse_ext.runtime import subset_batch
    from geosparse_research.training_gradients import _training_measurement_state

    size = len(batch["inputs"])
    with _training_measurement_state(model):
        for parameter in model.parameters():
            parameter.grad = None
        for begin in range(0, size, microbatch_size):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                losses = model.forward_train(**subset_batch(batch, begin, begin + microbatch_size))
            (losses["cost"] * (microbatch_size / size) * loss_scale).backward()
        squares = sum(float((p.grad.detach().double() / loss_scale).square().sum())
                      for p in model.parameters() if p.grad is not None)
    return math.sqrt(squares) if math.isfinite(squares) else None


def run(args):
    training = args.training_run.resolve()
    bindings = json.loads((training / "bindings.json").read_text())
    root = Path(bindings["repo_root"]).resolve()
    research = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(research)]
    import torch
    from mmengine.config import Config
    import geosparse_ext.detector as detector_module
    import geosparse_research.training_gradients as gradient_module
    from geosparse_ext.records import load_json, save_json, source_commit, content_id, file_id
    from geosparse_ext.runtime import require_gpu, seed_all, loader
    from geosparse_ext.cuda_identity import allocated_cuda_device
    from opentad.models.builder import build_detector

    provenance = load_json(training / "source_commits.json")
    job = load_json(training / "job.json")
    runtime = bindings["runtime"]
    if (source_commit(root) != provenance["source_commit"]
            or Path(detector_module.__file__).resolve().parents[1] != root
            or Path(gradient_module.__file__).resolve().parents[1] != research):
        raise ValueError("model or measurement imports escaped their recorded source")
    measurement_commit = source_commit(research)
    resolved = load_json(training / "resolved_config.json")
    cfg = Config.fromfile(str(training / "resolved_opentad.py"))
    if (content_id(resolved) != provenance["resolved_config_sha256"]
            or json.loads(json.dumps(cfg.to_dict())) != resolved["opentad"]
            or file_id(cfg.model.recognition_checkpoint) != provenance["weights_sha256"]):
        raise ValueError("configuration or recognition initialization differs from training")
    split = load_json(Path(bindings["protocol_root"]) / job["dataset"] / "split.json")
    if content_id(split) != provenance["split_sha256"] or len(split["training"]) != 200:
        raise ValueError("probe requires the unchanged full200 training pool")
    if job["kind"] != "train" or job["route"] not in {"A", "B", "C"}:
        raise ValueError("probe requires an existing A/B/C training job")
    effective, micro = runtime["effective_batch"], runtime["microbatch"]
    if effective % micro:
        raise ValueError("microbatch must divide the recorded effective batch")
    require_gpu()
    device_identity = allocated_cuda_device()
    seed_all(job["seed"])
    data = loader(cfg.dataset.train, effective, runtime["num_workers"], job["seed"], training=True)
    model = build_detector(cfg.model).cuda()
    checkpoint = training / "checkpoint" / f"epoch_{args.completed_epochs - 1}.pth"
    saved = torch.load(checkpoint, map_location="cpu")
    if saved["provenance"] != provenance:
        raise ValueError("ordinary checkpoint provenance differs from training")
    context = restore_probe_state(model, saved, args.completed_epochs, bool(cfg.solver.amp))
    del saved
    # Matches runtime.train's first resumed epoch. The loader owns its generator;
    # main-process RNG is restored before any num_workers=0 augmentation as well.
    data.generator.manual_seed(job["seed"] * 1000 + model.epoch)
    batch = next(iter(data))
    names = [meta["video_name"] for meta in batch["metas"]]
    if len(names) != effective or not set(names) <= set(split["training"]):
        raise ValueError("probe batch escaped the original training pool/batch size")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(measurement="ordinary_checkpoint_next_training_batch_gradients", is_mock=False,
        model_source_commit=provenance["source_commit"], measurement_source_commit=measurement_commit,
        source_train_id=job["job_id"], training_provenance=provenance,
        checkpoint_path=str(checkpoint), checkpoint_sha256=file_id(checkpoint), **context,
        seed=job["seed"], subset="training", population="IN_SAMPLE", batch_index=0,
        videos=names, input_shape=list(batch["inputs"].shape), valid_frames=batch["masks"].sum(-1).tolist(),
        gt_counts=[len(x) for x in batch["gt_segments"]], effective_batch=effective,
        microbatch=micro, num_workers=runtime["num_workers"], amp=bool(cfg.solver.amp),
        gpu_identity=device_identity,
        new_training=False, optimizer_step_performed=False,
        interpretation="first next-epoch batch at saved ordinary weights/RNG; not an EMA probe or a population-wide gradient conclusion")
    save_json(args.output / "measurement.json", identity)
    try:
        result = gradient_module.measure_training_gradients(model, batch, microbatch_size=micro,
            amp=bool(cfg.solver.amp), loss_scale=context["loss_scale"], clip_norm=float(cfg.solver.clip_grad_norm))
        plain = ordinary_gradient_norm(model, batch, micro, bool(cfg.solver.amp), context["loss_scale"])
        measured = result["gradients"]["total"]["norm"]
        if result["gradient_status"] == "FINITE":
            torch.testing.assert_close(measured, plain, rtol=1e-4, atol=1e-5)
        elif plain is not None:
            raise ValueError("decomposed nonfinite gradients disagree with ordinary backward")
        save_json(args.output / "gradients.json", result)
        save_json(args.output / "result.json", dict(identity, status="completed_diagnostic",
            gradient_status=result["gradient_status"], ordinary_backward_norm=plain,
            same_batch_ordinary_backward_checked=True, state_restored=True,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            observed_components=list(result["gradients"]), population_conclusion_supported=False))
    except Exception as error:
        save_json(args.output / "failure.json", dict(identity, status="failed", error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--completed-epochs", type=int, required=True, choices=range(1, 60))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
