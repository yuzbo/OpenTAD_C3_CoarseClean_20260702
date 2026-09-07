"""Certify only configurations that actually passed the production precheck."""
import argparse
import json
from pathlib import Path
from .records import load_json, save_json


def supported_training_jobs(jobs, precheck):
    if precheck["is_mock"] or precheck["completed_epochs"] != 0:
        raise ValueError("expected a real correctness precheck, not a scientific run")
    by_id = {j["job_id"]: j for j in jobs if j["kind"] == "train"}
    passed = [by_id[row["job_id"]] for row in precheck["routes"].values()
              if row["status"] == "PASS" and row.get("optimizer_updates", 0) >= 1 and row.get("ema_update") == "PASS"]
    return [j for j in by_id.values() if any(j["model"] == checked["model"] and j["dataset"] == checked["dataset"] for checked in passed)]


def certify(jobs, precheck_path, bindings):
    precheck = load_json(precheck_path)
    if precheck["source_commit"] != bindings["source_commit"]:
        raise ValueError("precheck and bound source commits differ")
    supported = supported_training_jobs(jobs, precheck)
    caps = {}
    by_id = {j["job_id"]: j for j in jobs if j["kind"] == "train"}
    for job in supported:
        for name in job["capabilities"]:
            caps.setdefault(name, []).append(job["job_id"])
        checked = [row for row in precheck["routes"].values() if row.get("status") == "PASS"
                   and by_id[row["job_id"]]["model"] == job["model"] and by_id[row["job_id"]]["dataset"] == job["dataset"]]
        if any(row.get("evaluation_pipeline_check", {}).get("status") == "PASS" for row in checked):
            caps.setdefault("evaluation", []).append(job["job_id"])
    for path in Path(bindings["capability_dir"]).glob("*.json"):
        if load_json(path).get("commit") == precheck["source_commit"]:
            caps.setdefault(path.stem, [])  # A failed rerun must withdraw its earlier capability.
    for name, variants in caps.items():
        save_json(Path(bindings["capability_dir"]) / f"{name}.json",
                  dict(ready=bool(variants), commit=precheck["source_commit"], test_receipt=str(Path(precheck_path).resolve()),
                       supported_variants=sorted(variants), protocol_version="geosparse-official-full-data-20260908",
                       scope="production forward/backward, optimizer/EMA and numerical correctness; evaluation additionally requires real video inference/NMS/mAP; no accuracy criterion"))
    return {key: len(value) for key, value in caps.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--precheck", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    args = parser.parse_args()
    jobs = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    print(json.dumps(certify(jobs, args.precheck, load_json(args.bindings))))


if __name__ == "__main__":
    main()
