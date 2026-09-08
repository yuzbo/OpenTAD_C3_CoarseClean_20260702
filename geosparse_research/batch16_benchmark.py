"""User-authorized batch32-to16 supplement, importing the original frozen model."""
import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import traceback


def amended_job(original):
    if original["kind"] != "benchmark" or original["batches"] != [1, 8, 32]:
        raise ValueError("expected the original 1/8/32 benchmark")
    job = copy.deepcopy(original)
    job.update(job_id=original["job_id"] + "-b16-20260908", batches=[16],
               benchmark_amendment="batch32-to16-20260908", amended_from_job=original["job_id"])
    return job


def load_measurement_module(path):
    # Relative model imports resolve inside the already-loaded frozen package.
    spec = importlib.util.spec_from_file_location("geosparse_ext._batch16_measurement", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    original = args.original_benchmark_run.resolve()
    source_job = json.loads((original / "job.json").read_text())
    job = amended_job(source_job)
    training = Path(job["dependency_outputs"][job["source_train_id"]])
    bindings = json.loads((training / "bindings.json").read_text())
    model_root = Path(bindings["repo_root"]).resolve()
    measurement_root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(model_root), str(measurement_root)]
    from mmengine.config import Config
    import geosparse_ext.detector as detector_module
    from geosparse_ext.records import load_json, save_json, source_commit, content_id

    provenance = load_json(original / "source_commits.json")
    if (source_commit(model_root) != provenance["source_commit"]
            or bindings["source_commit"] != provenance["source_commit"]
            or Path(detector_module.__file__).resolve().parents[1] != model_root):
        raise ValueError("model imports differ from the original frozen benchmark")
    cfg = Config.fromfile(str(original / "resolved_opentad.py"))
    resolved = load_json(original / "resolved_config.json")
    if (content_id(resolved) != provenance["resolved_config_sha256"]
            or json.loads(json.dumps(cfg.to_dict())) != resolved["opentad"]):
        raise ValueError("source benchmark configuration changed")
    split = load_json(Path(bindings["protocol_root"]) / job["dataset"] / "split.json")
    if content_id(split) != provenance["split_sha256"]:
        raise ValueError("source benchmark split changed")
    measurement_commit = source_commit(measurement_root)
    module = load_measurement_module(measurement_root / "geosparse_ext/benchmark.py")
    module.validate_benchmark_protocol(job)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(job_id=job["job_id"], is_mock=False, **provenance,
                    measurement_source_commit=measurement_commit,
                    benchmark_amendment=job["benchmark_amendment"],
                    amended_from_job=source_job["job_id"], original_benchmark_run=str(original),
                    new_training=False, reused_batches=[1, 8], measured_batches=[16])
    save_json(args.output / "job.json", job)
    save_json(args.output / "measurement.json", identity)
    try:
        result = module.benchmark(job, cfg, args.output, provenance, bindings["runtime"], split)
        save_json(args.output / "result.json", dict(identity, status="completed", **result,
            completed_at_utc=datetime.now(timezone.utc).isoformat()))
    except Exception as error:
        save_json(args.output / "failure.json", dict(identity, status="FAILED",
            reason=str(error), error_type=type(error).__name__, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-benchmark-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
