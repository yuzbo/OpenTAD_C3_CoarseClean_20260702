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
    counts = {}
    by_id = {j["job_id"]: j for j in jobs if j["kind"] == "train"}
    # Independent prechecks must not withdraw each other's configurations.
    # A failed rerun only withdraws the variants covered by that report.
    observed = [by_id[row["job_id"]] for row in precheck["routes"].values() if row.get("job_id") in by_id]
    covered = [j for j in by_id.values() if any(j["model"] == checked["model"] and j["dataset"] == checked["dataset"] for checked in observed)]
    supported_ids = {j["job_id"] for j in supported}
    for job in covered:
        caps = set(job["capabilities"]) if job["job_id"] in supported_ids else set()
        checked = [row for row in precheck["routes"].values() if row.get("status") == "PASS"
                   and by_id[row["job_id"]]["model"] == job["model"] and by_id[row["job_id"]]["dataset"] == job["dataset"]]
        if caps and any(row.get("evaluation_pipeline_check", {}).get("status") == "PASS" for row in checked):
            caps.add("evaluation")
        directory = Path(bindings["capability_dir"]) / job["job_id"]
        names = caps | {path.stem for path in directory.glob("*.json")}
        for name in names:
            ready = name in caps
            save_json(directory / f"{name}.json",
                      dict(ready=ready, commit=precheck["source_commit"], test_receipt=str(Path(precheck_path).resolve()),
                           supported_variants=[job["job_id"]] if ready else [], protocol_version="geosparse-official-full-data-20260908",
                           scope="production forward/backward, optimizer/EMA and numerical correctness; evaluation additionally requires real video inference/NMS; no partial-set scientific metric or accuracy criterion"))
            counts[name] = counts.get(name, 0) + int(ready)
    return counts


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
