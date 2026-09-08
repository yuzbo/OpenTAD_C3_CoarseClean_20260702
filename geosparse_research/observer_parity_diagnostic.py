"""Run as a file against frozen M to diagnose Slurm1280030's failed precheck."""
import argparse
import json
from pathlib import Path
import sys


def run(args):
    training = args.training_run.resolve()
    bindings = json.loads((training / "bindings.json").read_text())
    model_root = Path(bindings["repo_root"]).resolve()
    research_root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(model_root), str(research_root)]
    import numpy as np
    import torch
    from mmengine.config import Config
    from geosparse_ext.prediction_export import training_source, evaluation_dataset
    from geosparse_ext.records import load_json, save_json, source_commit, content_id, require_same_checkpoint
    from geosparse_ext.runtime import require_gpu, seed_all, loader, load_trained_model
    import geosparse_ext.detector as detector_module
    from geosparse_research.fixed_budget_diagnostic import fixed_half_budget
    from geosparse_research.observer_parity import compare_observers

    bindings, training_receipt, provenance, training_job = training_source(training)
    if source_commit(model_root) != provenance["source_commit"] or Path(detector_module.__file__).resolve().parents[1] != model_root:
        raise ValueError("model imports escaped the recorded frozen snapshot")
    failed = load_json(args.failed_precheck / "measurement.json")
    require_same_checkpoint(training_receipt["checkpoint_identity"], failed["checkpoint_identity"])
    if load_json(args.failed_precheck / "failure.json")["error_type"] != "AssertionError":
        raise ValueError("expected the preserved observer equality failure")
    require_gpu()
    seed_all(training_job["seed"])
    cfg = Config.fromfile(str(training / "resolved_opentad.py"))
    resolved = load_json(training / "resolved_config.json")
    if content_id(resolved) != provenance["resolved_config_sha256"] or json.loads(json.dumps(cfg.to_dict())) != resolved["opentad"]:
        raise ValueError("configuration differs from the failed diagnostic")
    split = load_json(Path(bindings["protocol_root"]) / training_job["dataset"] / "split.json")
    if content_id(split) != provenance["split_sha256"]:
        raise ValueError("split differs from the failed diagnostic")
    dataset_cfg, _ = evaluation_dataset(cfg, "validation")
    data = loader(dataset_cfg, 1, bindings["runtime"]["num_workers"], training_job["seed"])
    args.output.mkdir(parents=True, exist_ok=False)
    job = dict(training_job, kind="evaluate", depends_on=[training_job["job_id"]],
               dependency_outputs={training_job["job_id"]: str(training)})
    model = load_trained_model(job, cfg, args.output, provenance)
    model.inference_video_index = {name: i for i, name in enumerate(sorted(split["validation"]))}
    batch = next(iter(data))
    original = torch.load(args.failed_precheck / "raw_predictions/batch_000000.pth", map_location="cpu")

    def metadata(value):
        if isinstance(value, dict):
            return {key: metadata(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [metadata(item) for item in value]
        if isinstance(value, (np.ndarray, np.generic, torch.Tensor)):
            return value.tolist()
        return value

    if metadata(original["metas"]) != metadata(batch["metas"]):
        raise ValueError("first test window metadata differs from the original failed pass")
    identity = dict(model_source_commit=provenance["source_commit"], measurement_source_commit=source_commit(research_root),
                    checkpoint_identity=model.checkpoint_identity, failed_precheck=str(args.failed_precheck),
                    metas=metadata(batch["metas"]), original_failure_slurm_id="1280030", scientific_result=False,
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    save_json(args.output / "measurement.json", identity)
    reports = []
    try:
        with fixed_half_budget(model):
            for amp in (bool(cfg.solver.amp), False) if cfg.solver.amp else (False,):
                report, outputs = compare_observers(model, batch, amp=amp)
                name = "amp" if amp else "fp32"
                torch.save(outputs, args.output / (name + ".raw.pth"))
                save_json(args.output / (name + ".json"), report)
                reports.append(dict(mode=name, status=report["status"]))
                print(json.dumps(reports[-1]), flush=True)
        save_json(args.output / "result.json", dict(identity, status="COMPLETED_DIFFERENTIAL_DIAGNOSTIC", reports=reports,
                  full_evaluation_authorized_by_this_result=False))
    except Exception as error:
        save_json(args.output / "failure.json", dict(identity, status="FAILED", completed_modes=reports,
                  error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--failed-precheck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
