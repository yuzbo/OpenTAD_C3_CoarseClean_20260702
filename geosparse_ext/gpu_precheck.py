"""Production shapes and real source equality, with no scientific result receipt."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import traceback
import torch
from .records import load_json, save_json, prepare_split, source_commit
from .protocol import resolve_opentad_config
from .runtime import require_gpu, loader, optimizer_for, seed_all, ModelEMA
from .data import video_batch
from .sparse import NativeEncoder
from .training_validation import validate_dataset
from opentad.models.builder import build_detector


# The production GPU witness found that default SDPA failed even source-vs-
# source repeatability. Use the same math kernel for both structural references;
# these scoped contexts restore training/benchmark kernels, including on failure.
@torch.backends.cudnn.flags(benchmark=False, deterministic=True)
@torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
def dense_limit(model, batch, route):
    source = model.geosparse.encoder.source.eval()
    video = video_batch(batch["inputs"], batch["masks"], batch["metas"], "cuda")
    b, c, t, h, w = video.frames_hi.shape
    frames = video.frames_hi.reshape(b, c, t // 16, 16, h, w).permute(0, 2, 1, 3, 4, 5)
    frames = frames.reshape(-1, c, 16, h, w).contiguous().detach().requires_grad_()
    parameters = [p for n, p in source.named_parameters() if ".adapter." in n and p.requires_grad]
    source.zero_grad(set_to_none=True)
    expected = source(frames)
    weight = torch.randn_like(expected) / expected.numel()
    # The actual source uses reentrant checkpointing, which supports backward()
    # but rejects autograd.grad(). Preserve that source execution convention.
    (expected * weight).sum().backward()
    expected_gradient = [value.grad.detach().clone() for value in [frames, *parameters]]
    reference = expected.detach()
    del expected
    checkpoint_flags = [block.with_cp for block in source.blocks]
    try:
        for block in source.blocks:
            block.with_cp = False
        direct = source(frames.detach().clone().requires_grad_())
        checkpoint_difference = float((direct.detach() - reference).abs().max())
        del direct
    finally:
        for block, flag in zip(source.blocks, checkpoint_flags):
            block.with_cp = flag
    source.zero_grad(set_to_none=True)
    second = frames.detach().clone().requires_grad_()
    hn, wn = h // source.patch_size, w // source.patch_size
    mask = torch.ones((len(frames), 8 * hn * wn) if route == "A" else (len(frames), 8, hn // 2, wn // 2), device="cuda", dtype=torch.bool)
    candidate = NativeEncoder(source, route).cuda().eval()
    actual = candidate(second, mask)
    (actual * weight).sum().backward()
    gradients = [value.grad.detach().clone() for value in [second, *parameters]]
    print(json.dumps(dict(source_checkpoint_vs_direct_max_abs_error=checkpoint_difference,
                          route=route, full_limit_max_abs_error=float((actual - reference).abs().max()))), flush=True)
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
    for observed, correct in zip(gradients, expected_gradient):
        torch.testing.assert_close(observed, correct, atol=2e-5, rtol=2e-4)
    source.zero_grad(set_to_none=True)
    return dict(status="PASS", actual_source="upstream VisionTransformerAdapter", frames=t,
                parent_count=len(frames), native_tokens_per_parent=8 * hn * wn,
                reference_attention_kernel="math_sdpa", deterministic_convolution=True,
                output_max_abs_error=float((actual - reference).abs().max()),
                source_checkpoint_vs_direct_max_abs_error=checkpoint_difference,
                gradient_max_abs_error=max(float((a - b).abs().max()) for a, b in zip(gradients, expected_gradient)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-ids", nargs="+", help="Exact registered seed0 configurations to check in this run")
    parser.add_argument("--certify", action="store_true", help="Publish exact passed capabilities after each configuration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    require_gpu()
    seed_all(0)
    commit = source_commit(Path(__file__).resolve().parents[1])
    bindings = load_json(args.bindings)
    if bindings["source_commit"] != commit:
        raise RuntimeError("GPU precheck must use the bound clean source snapshot")
    witness = torch.randn(16, 16, device="cuda")
    assert torch.isfinite(witness @ witness).all()
    print("CUDA_WITNESS", torch.cuda.get_device_name(), flush=True)
    jobs = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    split_dir = Path(bindings["protocol_root"]) / "thumos14"
    prepare_split(bindings["annotations"], split_dir)
    summary_path = args.output / "gpu_precheck.json"
    previous = load_json(summary_path) if summary_path.is_file() else None
    if previous and previous["source_commit"] != commit:
        raise RuntimeError("use a new precheck output directory for a changed source snapshot")
    summary = dict(source_commit=commit, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   slurm_job_gpus=os.environ["SLURM_JOB_GPUS"], cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                   cuda=torch.version.cuda, completed_epochs=0, is_mock=False, routes=previous["routes"] if previous else {})
    if args.train_ids:
        by_id = {j["job_id"]: j for j in jobs}
        selected_jobs = [by_id[value] for value in args.train_ids]
        if len(set(args.train_ids)) != len(args.train_ids) or any(j["kind"] != "train" or j["seed"] != 0 or j["dataset"] != "thumos14" for j in selected_jobs):
            raise ValueError("precheck IDs must be distinct registered THUMOS training configurations at seed0")
    else:
        selected_jobs = [next(j for j in jobs if j["kind"] == "train" and j["route"] == route and j["dataset"] == "thumos14"
                   and j["seed"] == 0 and j["family"] == ("F00" if route in {"DENSE", "COARSE"} else "F01")
                   and j["model"]["axis"] == "ST" and j["model"]["budget_mode"] == "fixed"
                   and j["model"]["budget"] == (1. if route == "DENSE" else 0. if route == "COARSE" else .5))
                         for route in ("A", "B", "C", "DENSE", "COARSE")]
    for job in selected_jobs:
        route, key = job["route"], job["job_id"] if args.train_ids else job["route"]
        model = optimizer = losses = predictions = gradients = batch = ema = None
        try:
            cfg = resolve_opentad_config(job, bindings, split_dir / "annotations.json")
            seed_all(0)
            data = loader(cfg.dataset.train, 1, 0, 0)
            batch = next(iter(data))
            model = build_detector(cfg.model).cuda().train()
            model.epoch = 6
            optimizer = optimizer_for(model, cfg.optimizer)
            ema = ModelEMA(model)
            torch.cuda.reset_peak_memory_stats()
            with torch.cuda.amp.autocast():
                losses = model(**batch, return_loss=True)
            assert torch.isfinite(losses["cost"])
            losses["cost"].backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.solver.clip_grad_norm)
            optimizer.step()
            model.after_optimizer_step()
            ema.update(model)
            assert torch.isfinite(model.rpn_head.loss_normalizer)
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                predictions = model.forward_test(batch["inputs"], batch["masks"], batch["metas"])
                post = copy.deepcopy(cfg.post_processing)
                post.sliding_window = False
                detections = model.post_processing(predictions, batch["metas"], post, data.dataset.class_map)
            row = dict(status="PASS", job_id=job["job_id"], input_shape=list(batch["inputs"].shape),
                       finite_gradient_tensors=len(gradients), detection_video_count=len(detections),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), optimizer_updates=1,
                       ema_update="PASS", source_limit="not_applicable")
            losses = predictions = gradients = None
            # A correctness preflight is not a validation score. Verify the
            # real test loader/decoder/model/NMS once; training validation then
            # covers all 211 videos / 792 windows at every registered epoch.
            test_data = loader(cfg.dataset.test, 1, 0, 0)
            assert len(test_data.dataset) == 792
            test_batch = next(iter(test_data))
            with torch.no_grad(), torch.cuda.amp.autocast():
                test_predictions = model.forward_test(test_batch["inputs"], test_batch["masks"], test_batch["metas"])
                model.post_processing(test_predictions, test_batch["metas"], post, test_data.dataset.class_map)
            row["evaluation_pipeline_check"] = dict(status="PASS", observed_windows=1,
                registered_full_test_windows=792, scope="correctness only; no partial-set scientific metric")
            test_batch = test_predictions = None
            if route in {"A", "C"}:
                row["source_limit"] = dense_limit(model, batch, route)
            summary["routes"][key] = row
        except Exception as exc:
            summary["routes"][key] = dict(status="FAIL", job_id=job["job_id"], reason=str(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            model = optimizer = losses = predictions = gradients = batch = ema = None
            gc.collect()
            torch.cuda.empty_cache()
            save_json(summary_path, summary)
            if args.certify:
                from .capabilities import certify
                certify(jobs, summary_path, bindings)
            print(json.dumps(summary["routes"][key]), flush=True)
    return 0 if all(v["status"] == "PASS" for v in summary["routes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
