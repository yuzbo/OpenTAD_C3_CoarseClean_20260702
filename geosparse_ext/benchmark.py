"""Measured window latency at three explicit boundaries; no FLOP-derived speed."""
import copy
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
from opentad.datasets.builder import build_dataset, collate
from .data import video_batch
from .records import save_json
from .runtime import require_gpu, load_trained_model, seed_all
from .cuda_identity import allocated_cuda_device


def timing_summary(samples_ms, batch_size):
    values = np.asarray(samples_ms, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("timing samples must be actual positive finite durations")
    return dict(samples=len(values), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)), mean_ms=float(values.mean()),
                throughput_windows_per_second=float(batch_size * 1000 / values.mean()))


def isolated_gpu():
    """Query the single visible allocated device on either cluster."""
    device = allocated_cuda_device()["cuda_uuid"]
    output = subprocess.check_output(["nvidia-smi", "-i", device, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    pids = {int(value.strip()) for value in output.splitlines() if value.strip()}
    if pids - {os.getpid()}:
        raise RuntimeError(f"allocated GPU is not isolated: other compute PIDs {sorted(pids - {os.getpid()})}")
    return subprocess.check_output(["nvidia-smi", "-i", device,
        "--query-gpu=uuid,name,driver_version,memory.total,clocks.sm,power.draw", "--format=csv,noheader"], text=True).strip()


def video_round_robin(data_list, count, seed):
    """Fixed video subset; interleave windows so a long video cannot dominate."""
    groups = {}
    for index, row in enumerate(data_list):
        groups.setdefault(row[0], []).append(index)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    names = names[:count]
    if len(names) != count:
        raise ValueError(f"benchmark needs {count} videos; found {len(names)}")
    indices = [groups[name][window] for window in range(max(len(groups[name]) for name in names))
               for name in names if window < len(groups[name])]
    return names, indices


def batch_at(dataset, indices, step, size):
    return collate([dataset[indices[(step * size + i) % len(indices)]] for i in range(size)])


def validate_benchmark_protocol(job):
    amendment = job.get("benchmark_amendment")
    if amendment not in {None, "batch32-to16-20260908"}:
        raise ValueError("unregistered benchmark amendment")
    expected = [16] if amendment else [1, 8, 32]
    if not job["exclusive"] or job["batches"] != expected or job["warmup"] != 50 or job["repeats"] != 200:
        raise ValueError(f"benchmark differs from registered batches={expected}, isolated 50/200 protocol")
    if job["modes"] != ["device_model", "decoded_tensor_to_output", "encoded_video_to_output"] or job["implementation_pairs"] != ["reference", "optimized"]:
        raise ValueError("unregistered benchmark boundary or implementation")
    return 16 if amendment else 1


@torch.no_grad()
def benchmark(job, cfg, output, provenance, runtime, split):
    require_gpu()
    probe_batch = validate_benchmark_protocol(job)
    seed_all(job["seed"])
    model = load_trained_model(job, cfg, output, provenance)
    dataset = build_dataset(copy.deepcopy(cfg.dataset.test))
    names, indices = video_round_robin(dataset.data_list, job["video_subset_size"], job["seed"])
    model.inference_video_index = {name: i for i, name in enumerate(sorted(split["validation"]))}
    post = copy.deepcopy(cfg.post_processing)
    post.sliding_window = False
    output = Path(output)
    summary = dict(job_id=job["job_id"], units="batch of 768-position windows", videos=names,
        source_train_id=job["source_train_id"], selected_checkpoint_epoch=model.epoch,
        checkpoint_identity=model.checkpoint_identity, gpu_identity=allocated_cuda_device(),
        dataset_window_indices=indices, batches=job["batches"],
        benchmark_amendment=job.get("benchmark_amendment"),
        slurm_job_gpus=os.environ["SLURM_JOB_GPUS"], cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        warmup=job["warmup"], repeats=job["repeats"], torch=torch.__version__, cuda=torch.version.cuda,
        precision="fp16_autocast" if cfg.solver.amp else "fp32", num_decode_workers=0,
        io_state="warm OS page cache; source files reopened and decoded on each encoded trial",
        boundaries=dict(device_model="normalized VideoBatch on GPU through detector proposals; no CPU NMS",
            decoded_tensor_to_output="collated CPU RGB tensors through transfer, normalization, model and window NMS",
            encoded_video_to_output="source video open/decode/PTS/transforms/collate through transfer, model and window NMS"),
        scope="window pipeline, not the wall time of detecting every window in a whole long video", cases=[])
    encoder = model.geosparse.encoder
    # Compare the same trained model and deterministic evidence plan before timing.
    if encoder is not None and encoder.route != "DENSE":
        probe = batch_at(dataset, indices, 0, probe_batch)
        with torch.cuda.amp.autocast(enabled=cfg.solver.amp):
            encoder.implementation = "reference"
            reference = model.forward_test(**probe)
            encoder.implementation = "bucket"
            optimized = model.forward_test(**probe)
        for a, b in zip(reference[0] + reference[1], optimized[0] + optimized[1]):
            torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-3)
        del probe, reference, optimized
        summary["same_plan_output_check"] = "PASS"
    else:
        summary["same_plan_output_check"] = "same dense/cheap execution for both labels"
    if job.get("benchmark_amendment"):
        save_json(output / "precheck.json", dict(status="PASS", batch=probe_batch,
            same_plan_output_check=summary["same_plan_output_check"],
            checkpoint_identity=model.checkpoint_identity, gpu_identity=summary["gpu_identity"],
            is_scientific_result=False))
    with open(output / "timings.jsonl", "w", buffering=1) as raw:
        for implementation in job["implementation_pairs"]:
            if encoder is not None:
                encoder.implementation = "reference" if implementation == "reference" else "bucket"
            for size in job["batches"]:
                for mode in job["modes"]:
                    case = dict(implementation=implementation, batch=size, mode=mode)
                    batch = device = predictions = detections = None
                    values = []
                    try:
                        case["gpu_start"] = isolated_gpu()
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        for step in range(job["warmup"] + job["repeats"]):
                            # No prefetched dataset iterator can hide decoding in the encoded boundary.
                            if mode != "encoded_video_to_output":
                                batch = batch_at(dataset, indices, step, size)
                                if mode == "device_model":
                                    device = video_batch(batch["inputs"], batch["masks"], batch["metas"], "cuda")
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            if mode == "encoded_video_to_output":
                                batch = batch_at(dataset, indices, step, size)
                            with torch.cuda.amp.autocast(enabled=cfg.solver.amp):
                                predictions = (model.forward_video_batch(device, batch["metas"]) if mode == "device_model"
                                               else model.forward_test(**batch))
                            if mode != "device_model":
                                detections = model.post_processing(predictions, batch["metas"], post, dataset.class_map)
                            torch.cuda.synchronize()
                            ms = (time.perf_counter() - started) * 1000
                            if step >= job["warmup"]:
                                values.append(ms)
                                raw.write(json.dumps(dict(**case, repeat=step - job["warmup"], ms=ms,
                                    video_ids=[m["video_name"] for m in batch["metas"]],
                                    window_ids=[f"{m['video_name']}:{int(m['geosparse']['source_frame_id'][0])}" for m in batch["metas"]],
                                    selected_tokens=model.latest_route_plan.realized_token_count.tolist()), allow_nan=False) + "\n")
                            if step == job["warmup"]:
                                save_json(output / f"trace-{implementation}-b{size}-{mode}.json",
                                          dict(layers=[] if encoder is None else encoder.trace))
                            batch = device = predictions = detections = None
                            if step % 25 == 0:
                                isolated_gpu()
                        case.update(status="MEASURED", **timing_summary(values, size),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu_end=isolated_gpu())
                    except torch.cuda.OutOfMemoryError as exc:
                        case.update(status="OOM", reason=str(exc), completed_repeats=len(values))
                    except Exception as exc:
                        case.update(status="FAILED", reason=str(exc), traceback=traceback.format_exc())
                    finally:
                        batch = device = predictions = detections = None
                        gc.collect()
                        torch.cuda.empty_cache()
                    summary["cases"].append(case)
                    save_json(output / "hardware.json", summary)
                    print(json.dumps(case), flush=True)
    failed = [c for c in summary["cases"] if c["status"] != "MEASURED"]
    if failed:
        raise RuntimeError(f"{len(failed)} benchmark cases failed or OOM; all per-case evidence retained in hardware.json")
    return dict(hardware_path=str(output / "hardware.json"), timing_samples_path=str(output / "timings.jsonl"),
                checkpoint_identity=model.checkpoint_identity, selected_checkpoint_epoch=model.epoch,
                gpu_identity=summary["gpu_identity"], dataset_window_indices=indices,
                measured_cases=sum(c["status"] == "MEASURED" for c in summary["cases"]),
                failed_cases=sum(c["status"] != "MEASURED" for c in summary["cases"]))
