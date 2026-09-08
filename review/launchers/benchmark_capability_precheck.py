"""Isolated batch-1 hardware-path witness for a completed frozen GeoSparse run.

This is a correctness preflight, not a reduced scientific benchmark. Formal
1/8/32, 50-warmup/200-repeat requests remain unchanged and may still report OOM.
The script lives outside the immutable model checkout.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


MODES = ("device_model", "decoded_tensor_to_output", "encoded_video_to_output")
IMPLEMENTATIONS = ("reference", "optimized")


def load_source_config(path, resolved):
    from mmengine.config import Config
    # JSON is the provenance record, but loses tuples required by e.g. Resize.
    # The training entry also saves this exact executable Python configuration.
    cfg = Config.fromfile(str(path))
    if json.loads(json.dumps(cfg.to_dict())) != resolved:
        raise ValueError("saved Python configuration differs from its resolved training record")
    return cfg


def validate_witness(receipt, commit, train_id):
    if (receipt.get("status") != "PASS" or receipt.get("is_mock") is not False
            or receipt.get("is_scientific_result") is not False
            or receipt.get("source_commit") != commit
            or receipt.get("source_train_id") != train_id):
        raise ValueError("not the real passed witness for this source and training run")
    if not receipt.get("selection_complete") or receipt.get("completed_epochs") != 60:
        raise ValueError("complete training and checkpoint selection are required")
    identity = receipt.get("checkpoint_identity") or {}
    if (identity.get("source_train_id") != train_id or identity.get("source_commit") != commit
            or identity.get("weights") != "ema" or not identity.get("checkpoint_sha256")):
        raise ValueError("witness does not identify the actual selected EMA checkpoint")
    if (not receipt.get("slurm_job_id") or not receipt.get("gpu_identity", {}).get("cuda_uuid")
            or receipt.get("same_plan_output_check") != "PASS"):
        raise ValueError("Slurm/CUDA identity and same-plan output witness are required")
    cases = receipt.get("cases", [])
    expected = {(i, m) for i in IMPLEMENTATIONS for m in MODES}
    if len(cases) != 6 or {(c.get("implementation"), c.get("mode")) for c in cases} != expected:
        raise ValueError("all six hardware paths must be checked exactly once")
    for case in cases:
        samples = case.get("sanity_durations_ms", [])
        if (case.get("status") != "PASS" or case.get("batch") != 1
                or case.get("warmup") != 1 or len(samples) != 2
                or not case.get("gpu_start") or not case.get("gpu_end")
                or not case.get("window_ids")
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in samples)):
            raise ValueError("a hardware path did not produce a complete isolated timing witness")


def run(args):
    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    sys.path.insert(0, bindings["repo_root"])
    import torch
    from opentad.datasets.builder import build_dataset
    from geosparse_ext.benchmark import isolated_gpu, video_round_robin, batch_at
    from geosparse_ext.cuda_identity import allocated_cuda_device
    from geosparse_ext.data import video_batch
    from geosparse_ext.records import load_json, save_json, source_commit, content_id
    from geosparse_ext.runtime import require_gpu, seed_all, load_trained_model

    require_gpu()
    commit = source_commit(Path(bindings["repo_root"]))
    if commit != bindings["source_commit"]:
        raise ValueError("bound model source differs")
    jobs = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = [j for j in jobs if j["kind"] == "benchmark" and j["source_train_id"] == args.train_id]
    if len(candidates) != 1:
        raise ValueError("expected one registered benchmark for this training run")
    job = candidates[0]
    if (job["seed"] != 0 or not job["exclusive"] or job["batches"] != [1, 8, 32]
            or job["modes"] != list(MODES) or job["implementation_pairs"] != list(IMPLEMENTATIONS)
            or job["warmup"] != 50 or job["repeats"] != 200):
        raise ValueError("unexpected registered benchmark protocol")
    train = Path(bindings["work_root"]) / "runs" / args.train_id
    parent_job = load_json(train / "job.json")
    if parent_job["job_id"] != args.train_id or parent_job["kind"] != "train" or parent_job["seed"] != job["seed"]:
        raise ValueError("incorrect training dependency")
    result = load_json(train / "result.json")
    if (result.get("status") != "completed" or result.get("is_mock") is not False
            or result.get("completed_epochs") != 60 or not result.get("selection_complete")
            or not result.get("best_checkpoint_selection_complete")):
        raise ValueError("training or full checkpoint selection is incomplete")
    provenance = load_json(train / "source_commits.json")
    resolved = load_json(train / "resolved_config.json")
    split = load_json(Path(bindings["protocol_root"]) / parent_job["dataset"] / "split.json")
    if (provenance["source_commit"] != commit or content_id(resolved) != provenance["resolved_config_sha256"]
            or content_id(split) != provenance["split_sha256"]):
        raise ValueError("saved configuration or split differs from training")
    cfg = load_source_config(train / "resolved_opentad.py", resolved["opentad"])
    job = dict(job, model=parent_job["model"], dependency_outputs={args.train_id: str(train)})
    seed_all(job["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status="RUNNING", is_mock=False, is_scientific_result=False,
        source_commit=commit, source_train_id=args.train_id, benchmark_job_id=job["job_id"],
        completed_epochs=60, selection_complete=True, slurm_job_id=os.environ["SLURM_JOB_ID"],
        gpu_identity=allocated_cuda_device(), cases=[],
        scope="batch-1 hardware-path correctness only; no performance claim or batch-8/32 memory certification")
    receipt_path = args.output / "benchmark_precheck.json"
    cap_path = Path(bindings["capability_dir"]) / args.train_id / "benchmark.json"
    try:
        isolated_gpu()
        model = load_trained_model(job, cfg, args.output, provenance)
        receipt["checkpoint_identity"] = model.checkpoint_identity
        dataset = build_dataset(copy.deepcopy(cfg.dataset.test))
        names, indices = video_round_robin(dataset.data_list, job["video_subset_size"], job["seed"])
        receipt["registered_video_subset"] = names
        model.inference_video_index = {name: i for i, name in enumerate(sorted(split["validation"]))}
        post = copy.deepcopy(cfg.post_processing)
        post.sliding_window = False
        encoder = model.geosparse.encoder
        with torch.no_grad():
            probe = batch_at(dataset, indices, 0, 1)
            with torch.cuda.amp.autocast(enabled=cfg.solver.amp):
                if encoder is not None:
                    encoder.implementation = "reference"
                reference = model.forward_test(**probe)
                if encoder is not None:
                    encoder.implementation = "bucket"
                optimized = model.forward_test(**probe)
            for left, right in zip(reference[0] + reference[1], optimized[0] + optimized[1]):
                torch.testing.assert_close(left, right, atol=2e-3, rtol=2e-3)
            del probe, reference, optimized
            receipt["same_plan_output_check"] = "PASS"
            for implementation in IMPLEMENTATIONS:
                if encoder is not None:
                    encoder.implementation = "reference" if implementation == "reference" else "bucket"
                for mode in MODES:
                    case = dict(implementation=implementation, mode=mode, batch=1, warmup=1,
                                status="RUNNING", sanity_durations_ms=[], gpu_start=isolated_gpu())
                    receipt["cases"].append(case)
                    torch.cuda.reset_peak_memory_stats()
                    for step in range(3):
                        batch = device = predictions = detections = None
                        if mode != "encoded_video_to_output":
                            batch = batch_at(dataset, indices, 0, 1)
                            if mode == "device_model":
                                device = video_batch(batch["inputs"], batch["masks"], batch["metas"], "cuda")
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        if mode == "encoded_video_to_output":
                            batch = batch_at(dataset, indices, 0, 1)
                        with torch.cuda.amp.autocast(enabled=cfg.solver.amp):
                            predictions = (model.forward_video_batch(device, batch["metas"]) if mode == "device_model"
                                           else model.forward_test(**batch))
                        if mode != "device_model":
                            detections = model.post_processing(predictions, batch["metas"], post, dataset.class_map)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - started) * 1000
                        if step:
                            case["sanity_durations_ms"].append(elapsed)
                        case["window_ids"] = [f"{m['video_name']}:{int(m['geosparse']['source_frame_id'][0])}" for m in batch["metas"]]
                        batch = device = predictions = detections = None
                    case.update(status="PASS", peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_end=isolated_gpu())
                    save_json(receipt_path, receipt)
                    print(json.dumps({k: case[k] for k in ("implementation", "mode", "status")}), flush=True)
        receipt.update(status="PASS", finished_at=time.time())
        validate_witness(receipt, commit, args.train_id)
        save_json(receipt_path, receipt)
        if args.certify:
            save_json(cap_path, dict(ready=True, commit=commit, test_receipt=str(receipt_path.resolve()),
                supported_variants=[args.train_id], scope=receipt["scope"], checkpoint_identity=model.checkpoint_identity))
        print(json.dumps(dict(status="PASS", receipt=str(receipt_path), certified=args.certify)), flush=True)
    except Exception as exc:
        receipt.update(status="FAILED", reason=str(exc), traceback=traceback.format_exc(), finished_at=time.time())
        save_json(receipt_path, receipt)
        if args.certify:
            save_json(cap_path, dict(ready=False, commit=commit, test_receipt=str(receipt_path.resolve()),
                                    supported_variants=[], scope=receipt["scope"]))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--train-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--certify", action="store_true")
    run(parser.parse_args())
