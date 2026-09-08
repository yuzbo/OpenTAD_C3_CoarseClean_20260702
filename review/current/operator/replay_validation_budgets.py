"""Replay frozen EMA Scouts on all test windows; never run or modify Heavy/training.

Budget/selection are actual outputs. Heavy MACs are analytical counts from the
executed selection plan and frozen executor rules, not latency measurements.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True, help="training_id:completed_epochs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding = read(args.binding)
    frozen = Path(binding["repo_root"])
    actual = subprocess.check_output(["git", "-C", str(frozen), "rev-parse", "HEAD"], text=True).strip()
    if actual != binding["source_commit"]:
        raise ValueError("source mismatch")
    step_gpu = os.environ.get("SLURM_STEP_GPUS", "")
    if not os.environ.get("SLURM_JOB_ID") or not step_gpu.isdigit() or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("diagnostic requires the existing allocation's single mapped GPU")
    if binding["cluster"] == "N16R4" and step_gpu != "1":
        raise RuntimeError("N16 diagnostic requires physical GPU1")
    sys.path.insert(0, str(frozen))
    import numpy as np
    import torch
    from mmengine.config import Config
    from geosparse_ext.data import video_batch
    from geosparse_ext.geometry import native_layout, canonical_atoms
    from geosparse_ext.routing import Scout, atom_features, make_plan
    from geosparse_ext.runtime import seed_all, loader
    from geosparse_ext.prediction_export import evaluation_dataset
    from geosparse_ext.records import content_id, file_id
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expected one allocated CUDA device")
    torch.set_num_threads(1)
    seed_all(0)
    free, total = torch.cuda.mem_get_info()
    if free < 4 * 1024**3:
        raise RuntimeError("insufficient spare VRAM; preserve primary training")
    torch.cuda.set_per_process_memory_fraction(min(1., 2 * 1024**3 / total), 0)
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    dataset_config = None
    for item in args.cases:
        jid, completed = item.split(":")
        completed = int(completed)
        run = Path(binding["work_root"]) / "runs" / jid
        job, provenance = read(run / "job.json"), read(run / "source_commits.json")
        resolved = read(run / "resolved_config.json")
        cfg = Config.fromfile(str(run / "resolved_opentad.py"))
        if provenance["source_commit"] != actual or content_id(resolved) != provenance["resolved_config_sha256"]:
            raise ValueError("run provenance mismatch")
        if json.loads(json.dumps(cfg.to_dict())) != resolved["opentad"]:
            raise ValueError("typed configuration mismatch")
        checkpoint_path = run / "checkpoint" / f"epoch_{completed-1}.pth"
        saved = torch.load(checkpoint_path, map_location="cpu")
        if saved["epoch"] != completed-1 or saved["provenance"] != provenance:
            raise ValueError("immutable checkpoint mismatch")
        config = job["model"]
        if config["source_resolution"] != 160 or config["backbone"] != "videomae_b":
            raise ValueError("this measurement covers the current six 160px VideoMAE-B methods")
        scout = Scout(config["scout_width"], config["scout_resolution"], config["scout_temporal_stride"])
        prefix = "geosparse.scout."
        state = {n[len(prefix):]: value for n, value in saved["state_dict_ema"].items() if n.startswith(prefix)}
        scout.load_state_dict(state, strict=True)
        scout.cuda().eval()
        protocol = read(run / "resolved_protocol.json")
        metric_path = run / "intermediate_eval" / f"epoch_{completed:03d}" / "metrics.json"
        metrics = read(metric_path) if metric_path.exists() else None
        if metrics and metrics["status"] != "completed_validation":
            metrics = None
        data_cfg, _ = evaluation_dataset(cfg, "validation")
        if dataset_config is None:
            dataset_config = data_cfg
        elif data_cfg != dataset_config:
            raise ValueError("shared decoding requires identical validation pipeline")
        case = dict(job_id=jid, completed_epochs=completed, config=config, scout=scout,
            checkpoint_path=str(checkpoint_path), checkpoint_sha256=file_id(checkpoint_path),
            provenance=provenance, protocol=protocol, amp=bool(cfg.solver.amp),
            controller={key: float(saved["state_dict"][f"geosparse.{key}"]) for key in ("dual", "cost_ema")},
            validation_metrics=metrics["metrics"] if metrics else None)
        if metrics and (metrics["provenance"] != provenance or metrics["windows"] != 792):
            raise ValueError("validation identity/coverage mismatch")
        cases.append(case)
        del saved, state
        gc.collect()
    split = read(Path(binding["protocol_root"]) / "thumos14" / "split.json")
    names = sorted(split["validation"])
    indices = {name: i for i, name in enumerate(names)}
    data = loader(dataset_config, 1, 2, 0)
    if len(names) != 211 or len(data.dataset) != 792:
        raise ValueError("expected all 211 validation videos and 792 windows")
    static = [{k:v for k,v in c.items() if k != "scout"} for c in cases]
    header = dict(status="running", source_commit=actual, cases=static, videos=names,
        expected_windows=792, gpu=str(torch.cuda.get_device_properties(0)),
        slurm_job_id=os.environ["SLURM_JOB_ID"], slurm_step_gpu=step_gpu,
        cost_scope="analytical Heavy QKV/attention/MLP MAC from actual plan; not total model or latency",
        execution="frozen EMA Scout and original plan builder; no Heavy forward, optimizer, or training state access",
        memory_cap_bytes=2 * 1024**3, loader_workers=2, is_mock=False, weights="ema")
    (args.output / "header.json").write_text(json.dumps(header, indent=2))
    paths = {c["job_id"]+":"+str(c["completed_epochs"]): args.output / f"{c['job_id']}.epoch_{c['completed_epochs']:03d}.windows.jsonl" for c in cases}
    streams = {key: p.open("a", buffering=1) for key,p in paths.items()}
    completed_rows = {}
    for key,p in paths.items():
        existing = [json.loads(line) for line in p.read_text().splitlines()]
        if [r["window_index"] for r in existing] != list(range(len(existing))):
            raise ValueError("resume rows are not a contiguous prefix")
        completed_rows[key] = len(existing)
    start = min(completed_rows.values())
    with (args.output / "segments.jsonl").open("a") as segment:
        segment.write(json.dumps(dict(start_window=start, started_at=time.time(), loader_workers=2,
                                      allocation=os.environ["SLURM_JOB_ID"], prior_case_windows=completed_rows))+"\n")
    all_rows = list(data.dataset.data_list)
    data.dataset.data_list = all_rows[start:]
    started = time.time()
    atoms = None
    try:
        with torch.no_grad():
            for index, raw in enumerate(data, start=start):
                batch = video_batch(raw["inputs"], raw["masks"], raw["metas"], device)
                layout = native_layout(batch)
                if atoms is None:
                    atoms = canonical_atoms(384, 10, 10, temporal_atom=1, spatial_group=2, axis="ST", device=device)
                for case in cases:
                    key = case["job_id"]+":"+str(case["completed_epochs"])
                    if index < completed_rows[key]:
                        continue
                    config = case["config"]
                    if config["axis"] != "ST" or config["temporal_atom_tubelets"] != 1 or config["spatial_group"] != 2:
                        raise ValueError("current-case selection atoms differ")
                    with torch.cuda.amp.autocast(enabled=case["amp"]):
                        feats, budget_logits, baseline = case["scout"](batch.frames_hi, batch.valid_frames)
                        pooled = atom_features(feats, atoms, (10,10))
                        gain = case["scout"].gain(pooled).squeeze(-1)
                        actionness = case["scout"].actionness(pooled).squeeze(-1)
                        generators = [torch.Generator(device=device).manual_seed(
                            (indices[batch.video_id[0]] * 1000003 + int(batch.source_frame_id[0,0])) % (2**63-1))]
                        plan = make_plan(gain, atoms, layout.valid, layout.parent_tubelets, (10,10),
                            config, budget_logits, actionness, None, case["completed_epochs"]-1, False, generators)
                    menu = [.25,.5,.75,1.] if config["route"] == "C" else [0.,.25,.5,.75,1.]
                    if not config["allow_global_zero"]:
                        menu = [q for q in menu if q > 0]
                    logits = budget_logits[0,[round(q*4) for q in menu]].float()
                    selected = plan.selected_native[0].sum(-1).to(torch.float64)
                    valid = layout.valid[0].reshape(48,-1).sum(-1).to(torch.float64)
                    counts = valid/4 + selected*3/4 if config["route"] == "C" else selected
                    proto = case["protocol"]
                    width, depth = proto["source_width"], proto["source_layers"]
                    mac = float((12*counts*width**2+2*counts.square()*width).sum())*depth
                    valid_full_mac = float((12*valid*width**2+2*valid.square()*width).sum())*depth
                    full_mac = depth*48*(12*800*width**2+2*800**2*width)
                    row = dict(window_index=index,video_id=batch.video_id[0],window_id=batch.window_id[0],
                        valid_frames=int(batch.valid_frames.sum()),requested_budget=float(plan.requested_budget[0]),
                        menu=menu,budget_logits=logits.cpu().tolist(),budget_probabilities=logits.softmax(0).cpu().tolist(),
                        selected_native_members=int(selected.sum()),valid_native_members=int(valid.sum()),
                        heavy_tokens_by_parent=counts.cpu().tolist(),heavy_macs=mac,
                        padded_full_heavy_macs=full_mac,valid_full_heavy_macs=valid_full_mac,
                        heavy_ratio_padded_full=mac/full_mac,
                        heavy_ratio_valid_full=mac/valid_full_mac if valid_full_mac else None)
                    streams[key].write(json.dumps(row)+"\n")
                    completed_rows[key] = index+1
                    del feats, pooled, gain, actionness, plan, baseline, budget_logits
                del batch, raw
                if index % 20 == 0:
                    print(json.dumps(dict(event="budget_replay",windows=index+1,total=792,seconds=time.time()-started)),flush=True)
    finally:
        for f in streams.values():
            f.close()
    if any(n != 792 for n in completed_rows.values()):
        raise ValueError("incomplete budget replay")
    for key,p in paths.items():
        observed = {json.loads(line)["video_id"] for line in p.open()}
        if observed != set(names):
            raise ValueError("budget replay video coverage mismatch")
    header.update(status="completed_budget_replay",windows=792,seconds=time.time()-started,
                  output_files={k:str(v) for k,v in paths.items()})
    (args.output / "receipt.json").write_text(json.dumps(header,indent=2))
    print(json.dumps(dict(status=header["status"],windows=792,cases=len(cases))),flush=True)


if __name__ == "__main__":
    main()
