"""Real job entry and registration CLI. Unimplemented fields return exit 78."""
import argparse
import collections
import json
from pathlib import Path
import sys
import traceback
from .protocol import implementation_blockers, resolve_opentad_config, resolved_model_protocol
from .records import load_json, save_json, prepare_split, content_id, file_id, source_commit


def job_blockers(job, bindings):
    implementation = implementation_blockers(job["model"])
    if job["dataset"] != "thumos14":
        implementation.append(f"{job['dataset']} raw-video protocol integration pending")
    if job["kind"] not in {"train", "evaluate", "benchmark"}:
        implementation.append(f"{job['kind']} measurement suite integration pending")
    assets = []
    for requirement in job["external_requirements"]:
        group, _, name = requirement.partition(":")
        value = bindings.get({"dataset": "datasets", "checkpoint": "checkpoints"}.get(group, ""), {}).get(name) if name else bindings.get(group)
        if not value or not Path(value).exists():
            assets.append(requirement)
    if job["dataset"] == "thumos14":
        for key in ("annotations", "class_map"):
            if not Path(bindings.get(key, "/missing-asset")).is_file():
                assets.append(key)
        for split in ("training", "validation"):
            if not Path(bindings.get("video_roots", {}).get(split, "/missing-asset")).is_dir():
                assets.append(f"raw_video_root:{split}")
    return implementation, assets


def register(manifest, bindings, output):
    jobs = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {j["job_id"] for j in jobs}
    by_id = {j["job_id"]: j for j in jobs}
    if len(ids) != len(jobs):
        raise ValueError("duplicate job IDs")
    states = {}
    for job in jobs:
        if "metric_gate" in job or (job["kind"] == "train" and job["depends_on"]):
            raise ValueError("scientific result gates are forbidden")
        if set(job["depends_on"]) - ids:
            raise ValueError("unknown artifact dependency")
        resolved_job = job if job["kind"] == "train" else dict(job, model=by_id[job["source_train_id"]]["model"])
        implementation, assets = job_blockers(resolved_job, bindings)
        status = "BLOCKED_IMPLEMENTATION" if implementation else "BLOCKED_EXTERNAL_ASSET" if assets else "BLOCKED_ARTIFACT" if job["depends_on"] else "BLOCKED_CAPABILITY"
        states[job["job_id"]] = dict(status=status, implementation=implementation, external_assets=assets,
            depends_on=job["depends_on"], capabilities=job["capabilities"],
            reason="All registered tasks retained; correctness receipts and resource allocation are required before launch")
    result = dict(manifest_sha256=file_id(manifest), counts=dict(collections.Counter(j["kind"] for j in jobs)),
                  status_counts=dict(collections.Counter(v["status"] for v in states.values())), jobs=states)
    save_json(Path(output) / "registry.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "jobs"}, ensure_ascii=False))
    return result


def run(job, bindings, output, precheck=False):
    if job["kind"] != "train":
        dependency = job.get("dependency_outputs", {}).get(job["source_train_id"])
        if not dependency or not (Path(dependency) / "job.json").is_file():
            save_json(output / "blocked.json", dict(job_id=job["job_id"], status="BLOCKED_ARTIFACT",
                      depends_on=job["depends_on"], is_mock=False))
            return 78
        parent = load_json(Path(dependency) / "job.json")
        if parent["job_id"] != job["source_train_id"] or parent["kind"] != "train":
            raise ValueError("dependency is not the registered source training job")
        job = dict(job, model=parent["model"])
    implementation, assets = job_blockers(job, bindings)
    if implementation or assets:
        save_json(output / "blocked.json", dict(job_id=job["job_id"], status="BLOCKED_IMPLEMENTATION" if implementation else "BLOCKED_EXTERNAL_ASSET",
                  implementation=implementation, external_assets=assets, is_mock=False))
        print(json.dumps(dict(implementation=implementation, external_assets=assets), ensure_ascii=False))
        return 78
    runtime = bindings["runtime"]
    if set(runtime) != {"effective_batch", "microbatch", "evaluation_batch", "num_workers"}:
        raise ValueError("runtime binding has unknown or missing fields")
    split_dir = Path(bindings["protocol_root"]) / job["dataset"]
    split = prepare_split(bindings["annotations"], split_dir)
    cfg = resolve_opentad_config(job, bindings, split_dir / "annotations.json")
    cfg.evaluation.ground_truth_filename = bindings["annotations"]
    resolved = dict(opentad=cfg.to_dict(), model=job["model"], dataset=job["dataset"], seed=job["seed"],
                    epochs=60, runtime=runtime, protocol="geosparse-full-validation-best-20260907",
                    checkpoint_selection=dict(subset="validation", interval_epochs=5, weights="ema",
                                              metric="average_mAP@0.3:0.1:0.7", ties="earlier_checkpoint"))
    resolved = json.loads(json.dumps(resolved))
    if (output / "resolved_config.json").exists() and load_json(output / "resolved_config.json") != resolved:
        raise ValueError("run configuration changed; use an amended run directory")
    save_json(output / "resolved_config.json", resolved)
    cfg.dump(str(output / "resolved_opentad.py"))
    from .runtime import loader, train
    if precheck:
        from opentad.models.builder import build_detector
        model = build_detector(cfg.model)
        data = loader(cfg.dataset.train, 1, 0, job["seed"], training=False)
        sample = next(iter(data))
        save_json(output / "resolved_protocol.json", resolved_model_protocol(model, sample["inputs"].shape))
        receipt = dict(job_id=job["job_id"], status="ASSET_AND_CONSTRUCTION_CHECKED", ready=False,
                       input_shape=list(sample["inputs"].shape), valid_frames=int(sample["masks"].sum()),
                       video_id=sample["metas"][0]["video_name"], pts_status=sample["metas"][0]["geosparse"]["pts_status"],
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       full_gpu_forward="PENDING", completed_epochs=0, is_mock=False)
        save_json(output / "precheck.json", receipt)
        print(json.dumps(receipt), flush=True)
        return 0
    root = Path(__file__).resolve().parents[1]
    commit = source_commit(root)
    if bindings["source_commit"] != commit:
        raise ValueError("source snapshot differs from binding")
    for name in job["capabilities"]:
        cap = job.get("capability_receipts", {}).get(name)
        variant = job.get("source_train_id", job["job_id"])
        if (not cap or cap.get("commit") != commit or not cap.get("ready")
                or not cap.get("test_receipt") or not Path(cap["test_receipt"]).is_file()
                or variant not in cap.get("supported_variants", [])):
            raise NotImplementedError(f"correctness capability missing for {name} at {commit}")
    provenance = dict(source_commit=commit, resolved_config_sha256=content_id(resolved),
                      split_sha256=content_id(split), weights_sha256=file_id(bindings["checkpoints"][job["model"]["backbone"]]))
    save_json(output / "source_commits.json", provenance)
    if job["kind"] == "train":
        result = train(job, cfg, output, provenance, runtime)
    elif job["kind"] == "evaluate":
        from .evaluation import evaluate
        result = evaluate(job, cfg, output, provenance, runtime, split, load_json(split_dir / "risk_thresholds.json"))
    elif job["kind"] == "benchmark":
        from .benchmark import benchmark
        result = benchmark(job, cfg, output, provenance, runtime, split)
    else:
        raise NotImplementedError(job["kind"])
    save_json(output / "result.json", dict(job_id=job["job_id"], status="completed", is_mock=False, **provenance, **result))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--precheck", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bindings = load_json(args.bindings)
    if args.register:
        if not args.manifest or args.job or args.precheck:
            parser.error("registration requires --manifest and cannot execute a job")
        register(args.manifest, bindings, args.output)
        return 0
    if not args.job or args.manifest:
        parser.error("execution requires --job")
    job = load_json(args.job)
    try:
        return run(job, bindings, args.output, args.precheck)
    except NotImplementedError as exc:
        save_json(args.output / "blocked.json", dict(job_id=job["job_id"], status="BLOCKED_IMPLEMENTATION", reason=str(exc), is_mock=False))
        print(str(exc), file=sys.stderr)
        return 78
    except Exception as exc:
        save_json(args.output / "failure.json", dict(job_id=job["job_id"], status="FAILED", reason=str(exc), is_mock=False,
                  traceback=traceback.format_exc()))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
